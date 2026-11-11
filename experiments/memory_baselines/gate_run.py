"""Gate: literature memory baselines against the repo's own rungs, on one set of frozen assets.

    python -m experiments.memory_baselines.gate_run --config <resolved_config.json> \
        --snapshot <clean_future_control/src> --out results/memory_baselines/gate --seed 17

One call handles one predictor seed and one part and writes rows_seed<seed>_<part>.json; `--aggregate` pools the
seeds that are present into summary.json. Every policy is scored on the same 192 validation
episodes with the same frozen predictor, R=4 recent frames and B=2 memory items.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import gate_data as gd

LABELS = {
    "recent_only": "Recent frames only",
    "oracle_keyframe": "Oracle keyframe (knows the cue position)",
    "full_history": "Full history (no memory bound)",
    "deletion_cost_student": "Ours: deletion-cost scorer",
    "attention_proxy_student": "Feature-attention-proxy scorer (repo)",
    "pafu_kv": "PaFu-KV (one stored score per frame, as published)",
    "pafu_kv_repo": "PaFu-KV, repo's earlier adaptation (re-scores the three residents at each eviction)",
    "pafu_teacher_top2": "PaFu-KV teacher, top 2 (reads the future)",
    "pacc_full": "Compress to Remember, as published (reads whole history)",
    "pacc_recurrent": "Compress to Remember, online 2-slot variant",
}
for _visual, _name in (("ijepa", "I-JEPA features"), ("dinov2", "DINOv2, as published")):
    LABELS[f"fm_{_visual}_published"] = f"FrameMorrow ({_name}; ranks whole history)"
    LABELS[f"fm_{_visual}_online"] = f"FrameMorrow ({_name}; online 2-slot variant)"
    LABELS[f"fm_{_visual}_teacher_top2"] = f"FrameMorrow teacher ({_name}), top 2 (reads the future)"


def crosscheck(assets, model, episodes):
    """My metric rows must equal the project's own append_metric_rows on the same predictions."""
    import run_matrix as base
    import run_online_matrix as metrics
    from phase1.data import load_episode
    from phase1.evaluation import pair_diagnostic
    from phase1.inputs import collate, selected_evidence
    from .gate_policies import oracle_keyframe

    selected = oracle_keyframe(episodes)
    mine = gd.metric_rows(assets, episodes, gd.predict_selection(model, episodes, selected, assets.device),
                          selected, "check", 0)
    theirs = []
    for e, kept in zip(episodes, selected):
        rec = assets.store.records[e.id]
        sample, target, info = base.sample_with_indices(assets.store, rec, kept, load_episode, selected_evidence)
        with torch.inference_mode():
            pred = model(collate([sample], assets.device)).cpu().numpy()[0]
        metrics.append_metric_rows(theirs, seed=0, task="gate", split=rec["split"], policy="check", rec=rec,
                                   chosen=kept, target=target, prediction=pred, sample_info=info, c=assets.c,
                                   store=assets.store, decoders=assets.decoder, load_episode=load_episode,
                                   pair_diagnostic=pair_diagnostic)
    for a, b in zip(mine, theirs):
        if a["route_accuracy"] != b["state_accuracy"] or abs(a["normalized_mse"] - b["normalized_mse"]) > 1e-5:
            raise RuntimeError(f"metric mismatch on {a['episode']} h{a['horizon']}: {a} vs {b}")
    return len(mine)


def run_seed(args, seed):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rows_seed{seed}_{args.part}.json"
    if path.exists() and not args.force:
        print(f"seed {seed}: {path} exists, skipping")
        return
    device = torch.device(args.device)
    assets = gd.load_gate(args.config, device)
    train, val = assets.split("train"), assets.split("validation")
    model = assets.predictor(seed)
    rows, info, started = [], {}, time.time()

    def log(msg):
        print(f"[seed {seed} +{time.time() - started:6.0f}s] {msg}", flush=True)

    def add(policy, selected=None, prediction=None):
        if prediction is None:
            prediction = gd.predict_selection(model, val, selected, device)
        new = gd.metric_rows(assets, val, prediction, selected, policy, seed)
        rows.extend(new)
        log(f"{policy:32s} route {np.mean([r['route_accuracy'] for r in new]):.4f}  "
            f"mse {np.mean([r['normalized_mse'] for r in new]):.5f}")

    if args.part == "core":
        core_policies(args, seed, assets, model, train, val, device, add, info, rows)
    elif args.part == "pafu_repo":
        from . import pafu_repo
        student, info["pafu_kv_repo"] = pafu_repo.train(model, train, seed, device)
        add("pafu_kv_repo", pafu_repo.select(student, val))
    else:
        _, visual_name, variant = args.part.split("_")
        fm_policies(args, seed, assets, train, val, device, add, info, visual_name, variant)

    payload = {"seed": seed, "part": args.part, "device": str(device), "assets": args.assets_note,
               "validation_episodes": len(val),
               "train_episodes": len(train),
               "seconds": time.time() - started, "info": info, "rows": rows}
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(payload, allow_nan=False))
    tmp.replace(path)
    log(f"wrote {path}")


ORIGINAL_NAMES = {"oracle_keyframe": "manual_keyframe"}   # this package's name -> the 29 Oct run's name


def reference_check(path, seed, rows):
    """Compare this run with the team's original validation output, row by row (same seed, episode, horizon)."""
    original = {}
    for r in json.loads(Path(path).read_text())["rows"]:
        if r["seed"] == seed:
            original[(r["condition"], r["episode"], r["horizon"])] = (r["state_accuracy"], r["normalized_mse"])
    out = {}
    for policy in sorted({r["policy"] for r in rows}):
        theirs = ORIGINAL_NAMES.get(policy, policy)
        pairs = [(r, original[(theirs, r["episode"], r["horizon"])]) for r in rows
                 if r["policy"] == policy and (theirs, r["episode"], r["horizon"]) in original]
        if not pairs:
            continue
        out[policy] = {"rows": len(pairs),
                       "route_mismatches": int(sum(r["route_accuracy"] != o[0] for r, o in pairs)),
                       "max_abs_mse_diff": float(max(abs(r["normalized_mse"] - o[1]) for r, o in pairs)),
                       "original_route_accuracy": float(np.mean([o[0] for _, o in pairs])),
                       "this_run_route_accuracy": float(np.mean([r["route_accuracy"] for r, _ in pairs]))}
    return out


def core_policies(args, seed, assets, model, train, val, device, add, info, rows):
    """Reference rungs, the repo's policies, ours, PaFu-KV and the compressor (all need the predictor)."""
    from . import gate_policies as gp, pacc, pafu

    info["metric_crosscheck_rows"] = crosscheck(assets, model, val[:16])
    add("recent_only", gp.recent_only(val))
    add("oracle_keyframe", gp.oracle_keyframe(val))
    add("full_history", [list(range(e.candidates)) for e in val])
    for name, selected in gp.heuristics(model, val, seed, device).items():
        add(name, selected)

    if args.students:
        scorers, info["ours"] = gp.frozen_scorers(args.students, seed, device)
    else:
        scorers, info["ours"] = gp.train_scorers(model, train, seed, device)
    for name, scorer in scorers.items():
        add(name, gp.scorer_selections(scorer, val))
    if args.reference:
        info["original_validation_check"] = reference_check(args.reference, seed, rows)
        check = info["original_validation_check"]
        print(f"[seed {seed}] original validation rows compared: "
              f"{sum(v['rows'] for v in check.values())}, route mismatches "
              f"{sum(v['route_mismatches'] for v in check.values())}, max |mse diff| "
              f"{max(v['max_abs_mse_diff'] for v in check.values()):.2e}", flush=True)

    head, info["pafu_kv"] = pafu.train(model, train, seed, device, steps=args.pafu_steps)
    add("pafu_kv", pafu.select(head, val, device))
    add("pafu_teacher_top2", pafu.teacher_select(model, val, device))

    for variant in ("full", "recurrent"):
        compressor, meta = pacc.train(model, train, seed, device, variant=variant, steps=args.pacc_steps)
        prediction, gap = pacc.predict(model, compressor, val, device, variant=variant)
        info[f"pacc_{variant}"] = {**meta, "validation_gap_to_full_history": gap}
        add(f"pacc_{variant}", prediction=prediction)



def fm_policies(args, seed, assets, train, val, device, add, info, visual_name, variant):
    """One FrameMorrow selector. The selector never sees the predictor; only scoring does."""
    from . import fm

    if visual_name == "ijepa":
        visual = {e.id: e.z for e in assets.episodes}
    else:
        if not args.dinov2:
            raise SystemExit("--dinov2 is required for the fm_dinov2 parts")
        with np.load(args.dinov2) as f:
            visual = {e.id: f[e.id] for e in assets.episodes}
        if any(len(visual[e.id]) != len(e.z) for e in assets.episodes):
            raise ValueError("DINOv2 features do not cover every frame")
    if variant == "published":
        add(f"fm_{visual_name}_teacher_top2", fm.select_teacher(val, visual, device))
    selector, meta = fm.train(train, visual, seed, device, variant=variant, epochs=args.fm_epochs,
                              hidden=args.fm_hidden, layers=args.fm_layers)
    info[f"fm_{visual_name}_{variant}"] = meta
    select = fm.select_published if variant == "published" else fm.select_online
    add(f"fm_{visual_name}_{variant}", select(selector, val, visual, device))


def aggregate(args):
    out = Path(args.out)
    rows, info, seeds = [], {}, set()
    for path in sorted(out.glob("rows_seed*.json")):
        data = json.loads(path.read_text())
        rows.extend(data["rows"]); info.setdefault(str(data["seed"]), {}).update(data["info"]); seeds.add(data["seed"])
        episodes, assets_note = data["validation_episodes"], data.get("assets")
    seeds = sorted(seeds)
    if not rows:
        raise SystemExit("no rows_seed*.json to aggregate")
    scores = gd.summarise(rows)
    for name, value in scores.items():
        value["label"] = LABELS.get(name, name)
    summary = {"task": "gate", "split": "validation", "validation_episodes": episodes, "seeds": seeds,
               "horizons": list(gd.HORIZONS), "recent": gd.R, "budget": gd.B,
               "aggregation": "mean over episodes, seeds and horizons; repeated rows are not independent samples",
               "assets": assets_note,
               "scores": scores, "training_info": info}
    checks = [v["original_validation_check"] for v in info.values() if "original_validation_check" in v]
    if checks:
        summary["original_validation_check"] = {
            policy: {"rows": sum(c[policy]["rows"] for c in checks if policy in c),
                     "route_mismatches": sum(c[policy]["route_mismatches"] for c in checks if policy in c),
                     "max_abs_mse_diff": max(c[policy]["max_abs_mse_diff"] for c in checks if policy in c),
                     "original_route_accuracy": float(np.mean([c[policy]["original_route_accuracy"] for c in checks if policy in c]))}
            for policy in sorted({k for c in checks for k in c})}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    width = max(len(v["label"]) for v in scores.values())
    for name, v in sorted(scores.items(), key=lambda kv: -kv[1]["route_accuracy"]):
        cue = "   n/a" if v["cue_kept"] is None else f"{100 * v['cue_kept']:6.2f}"
        print(f"{v['label']:{width}s}  route {100 * v['route_accuracy']:6.2f}%  cue kept {cue}%  mse {v['normalized_mse']:.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config"); ap.add_argument("--snapshot"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, choices=gd.SEEDS); ap.add_argument("--device", default="cpu")
    ap.add_argument("--dinov2", help="npz of DINOv2 class tokens keyed by episode id")
    ap.add_argument("--aggregate", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--pafu-steps", type=int, default=1000); ap.add_argument("--pacc-steps", type=int, default=3000)
    ap.add_argument("--fm-epochs", type=int, default=10); ap.add_argument("--fm-hidden", type=int, default=512)
    ap.add_argument("--fm-layers", type=int, default=4)
    ap.add_argument("--part", default="core", choices=("core", "fm_dinov2_published", "fm_dinov2_online",
                                                       "fm_ijepa_published", "fm_ijepa_online", "pafu_repo"),
                    help="core: everything that needs the predictor to train; fm_<visual>_<variant>: one FrameMorrow selector")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--students", help="team model bundle folder: use its frozen student_<seed>.pt scorers instead of retraining ours")
    ap.add_argument("--reference", help="the team's original validation_results.json, to compare shared policies row by row")
    ap.add_argument("--assets-note", default="Gate dataset, I-JEPA features and predictors regenerated on torch from the repo "
                    "generator (seed 6301); not the team's original HPC checkpoints")
    args = ap.parse_args()
    if args.aggregate:
        return aggregate(args)
    if not (args.config and args.snapshot and args.seed):
        ap.error("--config, --snapshot and --seed are required unless --aggregate")
    gd.add_paths(args.snapshot)
    torch.set_num_threads(args.threads)
    run_seed(args, args.seed)


if __name__ == "__main__":
    main()
