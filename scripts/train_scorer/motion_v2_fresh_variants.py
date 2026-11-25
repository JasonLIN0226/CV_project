"""Motion v2, deletion-cost scorer on fresh scenes: label and state-distribution variants, chosen on fresh dev.

    BM_MOTION_ROOT=/scratch/$USER python scripts/train_scorer/motion_v2_fresh_variants.py \\
        --config configs/motion_v2/frozen.yaml --seed 0 --device cuda --out <dir> \\
        --fresh <scenes dir> --tables <scorer dir>/tables [--validate]

Reuses the fresh scenes and all-pairs loss tables of motion_v2_deletion_cost_v2.py --fresh (same split: the
first 512 scene pairs for training, the next 96 for dev). Its first run kept step 0 (the attention policy) on
all seeds: every trained checkpoint was worse on dev. Two causes are tested, alone and together:

    soft      labels: soft minimum over the reachable pairs (temperature 0.002, in mean +3/+6 position MSE)
              instead of the minimum, so a frame is credited for being in many good pairs, not one lucky pair
    dagger    states: after each training round, the trained scorer is rolled out on the training episodes and
              the states it visits are labelled from the tables and added (3 rounds); the first round trains
              on the behaviour policies' states only, as before

Each variant x seed (17, 29, 43) keeps the round and checkpoint with the lowest fresh-dev position MSE (mean of
+3 and +6; step 0, the attention policy, is always a candidate). The variant with the lowest mean over its seeds
is chosen (summary.json) before validation is read. Per-state diagnostics (regret against the cheapest deletion,
top-1) on the dev episodes' behaviour states are recorded for every round.

--validate replays the chosen variant once on both validation intervals, scored by FrozenAssets.losses, and
appends the result to <out>/validation_ledger.jsonl. Validation is not read otherwise.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import deletion_cost_v2 as dc, fresh, frozen, literature  # noqa: E402

SEEDS = (17, 29, 43)
VARIANTS = {                       # name: (label temperature, on-policy rounds after the first)
    "min": (0.0, 0),
    "min_dagger": (0.0, 3),
    "soft": (0.002, 0),
    "soft_dagger": (0.002, 3),
}


def concat(parts):
    parts = [p for p in parts if len(p["cost"])]
    if not parts:
        return {}
    return {k: np.concatenate([np.asarray(p[k], np.float32 if k != "cost" else np.float64) for p in parts])
            for k in ("items", "context", "prior", "cost")}


def labelled(assets, episodes, tables, states, temperature):
    return concat([dc.state_rows(assets, ep, tables[ep["rec"]["id"]], states[ep["rec"]["id"]], temperature)
                   for ep in episodes])


def fit_variant(assets, fit, dev, tables, base_rows, dev_rows, behaviour, temperature, rounds, seed, device, steps, log):
    """Train on behaviour-state rows, then `rounds` times add the trained scorer's own states and retrain.
    Returns the per-round history and (best dev mean, round, state dict, norm, meta) of the best round."""
    seen = {ep["rec"]["id"]: set(behaviour[ep["rec"]["id"]]) for ep in fit}
    data, history, best = base_rows, [], None
    for r in range(rounds + 1):
        model, norm, meta = dc.train(data, tables, dev, assets, seed, device, steps=steps,
                                     log=lambda m, r=r: log(f"round {r} {m}"))
        final = dc.ScorerV2(meta["width"]).to(device)
        final.load_state_dict(meta.pop("final_state"))
        metrics = {"selected": dc.state_metrics(model, norm, dev_rows, device),
                   "final": dc.state_metrics(final, norm, dev_rows, device)}
        entry = {"round": r, "states": int(len(data["cost"])), "selected_step": meta["selected_step"],
                 "selected_dev_mean": meta["selected_dev_mean"],
                 "final_dev_mean": float(np.mean([meta["history"][-1]["dev_position_+3"],
                                                  meta["history"][-1]["dev_position_+6"]])),
                 "dev_state_metrics": metrics}
        history.append(entry)
        log(f"round {r}: selected step {meta['selected_step']} dev {meta['selected_dev_mean']:.6f} final dev "
            f"{entry['final_dev_mean']:.6f} | dev-state top1 (informative) selected "
            f"{metrics['selected']['top1_informative']:.3f} final {metrics['final']['top1_informative']:.3f}")
        if best is None or meta["selected_dev_mean"] < best[0]:
            best = (meta["selected_dev_mean"], r, {k: v.cpu() for k, v in model.state_dict().items()}, norm, meta)
        if r == rounds:
            break
        _, visited = dc.rollout(assets, fit, final, norm, device, record=True)
        new = {k: [s for s in v if s not in seen[k]] for k, v in visited.items()}
        for k, v in new.items():
            seen[k].update(v)
        added = labelled(assets, fit, tables, new, temperature)
        log(f"round {r}: +{len(added['cost']) if added else 0} on-policy states")
        if added:
            data = concat([data, added])
    return history, best


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--fresh", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--fresh-pairs", type=int, default=512)
    ap.add_argument("--fresh-dev-pairs", type=int, default=96)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    task = runs.Task(args, "fresh_variants" + ("_validated" if args.validate else ""))
    if task.done():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    intervals = list(frozen.INTERVALS) if args.validate else ["ordinary"]
    frozen.verify(paths, intervals)
    assets = frozen.load(paths, args.device)

    records = fresh.generate(args.fresh, args.fresh_pairs + args.fresh_dev_pairs, log=lambda *_: None)
    episodes = [fresh.episode(assets, args.fresh, r) for r in records]
    tables = {}
    for ep in episodes:
        path = args.tables / f"{ep['rec']['id']}.npy"
        if not path.exists():
            raise SystemExit(f"missing loss table {path}: run motion_v2_deletion_cost_v2.py --fresh first")
        tables[ep["rec"]["id"]] = np.load(path)
    fit = [ep for ep in episodes if ep["rec"]["scene_index"] < args.fresh_pairs]
    dev = [ep for ep in episodes if ep["rec"]["scene_index"] >= args.fresh_pairs]
    task.log(f"fresh episodes: fit {len(fit)}, dev {len(dev)}")

    behaviour = {ep["rec"]["id"]: dc.behaviour_states(assets, ep, 20261124) for ep in episodes}
    identity = {"items_mean": np.zeros(dc.ITEM_DIM), "items_std": np.ones(dc.ITEM_DIM),
                "context_mean": np.zeros(dc.CONTEXT_DIM), "context_std": np.ones(dc.CONTEXT_DIM)}
    attention = dc.table_loss(tables, dev, dc.rollout(assets, dev, dc.ScorerV2().to(args.device), identity, args.device))
    reference = {"attention_+3": float(attention[0]), "attention_+6": float(attention[1]),
                 "attention_mean": float(attention.mean()),
                 "best_pair_mean": float(np.mean([tables[ep["rec"]["id"]].mean(0).min() for ep in dev]))}
    task.log(f"dev reference: {reference}")
    task.info["dev_reference"] = reference

    results, rows = {}, {}
    for name in args.variants:
        temperature, rounds = VARIANTS[name]
        if temperature not in rows:      # behaviour-state rows, shared by the seeds and variants of one temperature
            rows[temperature] = (labelled(assets, fit, tables, behaviour, temperature),
                                 labelled(assets, dev, tables, behaviour, temperature))
        base_rows, dev_rows = rows[temperature]
        for seed in SEEDS:
            target = out / f"{name}_seed{seed}.json"
            if target.exists():
                results.setdefault(name, {})[seed] = json.loads(target.read_text())
                continue
            history, best = fit_variant(assets, fit, dev, tables, base_rows, dev_rows, behaviour, temperature, rounds,
                                        seed, args.device, args.steps,
                                        log=lambda m, s=seed: task.log(f"{name} seed {s} {m}"))
            torch.save({"model": best[2], "norm": best[3], "round": best[1], "width": best[4]["width"]},
                       out / f"{name}_seed{seed}.pt")
            record = {"variant": name, "seed": seed, "temperature": temperature, "rounds": rounds,
                      "best_round": best[1], "best_dev_mean": best[0], "history": history}
            target.write_text(json.dumps(record, indent=1) + "\n")
            results.setdefault(name, {})[seed] = record

    summary = {"reference": reference, "variants": {
        name: {"dev_mean_per_seed": {s: r["best_dev_mean"] for s, r in per.items()},
               "dev_mean": float(np.mean([r["best_dev_mean"] for r in per.values()])),
               "best_rounds": {s: r["best_round"] for s, r in per.items()},
               "selected_steps": {s: r["history"][r["best_round"]]["selected_step"] for s, r in per.items()}}
        for name, per in results.items()}}
    chosen = min(summary["variants"], key=lambda n: summary["variants"][n]["dev_mean"])
    summary["chosen"] = chosen
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for name, s in summary["variants"].items():
        task.log(f"variant {name:12s} dev mean {s['dev_mean']:.6f} (attention {reference['attention_mean']:.6f}) "
                 f"rounds {s['best_rounds']} steps {s['selected_steps']}")
    task.log(f"chosen on fresh dev: {chosen}")
    task.info["summary"] = summary

    validation = None
    if args.validate:
        validation = {}
        for interval in intervals:
            recs, external = frozen.interval_inputs(assets, paths, interval)
            raw, _ = literature.split_episodes(assets, recs, "validation", external)
            scores = {}
            for seed in SEEDS:
                state = torch.load(out / f"{chosen}_seed{seed}.pt", map_location="cpu", weights_only=False)
                model = dc.ScorerV2(state["width"]).to(args.device)
                model.load_state_dict(state["model"])
                kept = dc.rollout(assets, raw, model, state["norm"], args.device)
                scores[seed] = literature.score_frames(assets, raw, kept)
                target = out / "raw" / interval / f"{chosen}_seed{seed}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, kept=np.asarray(kept), episode_ids=np.array([r["rec"]["id"] for r in raw]), **scores[seed])
            validation[interval] = literature.summarize(scores, list(SEEDS))
            task.log(f"validation {interval} ({chosen}): +3 {validation[interval]['position_mse_+3']:.6f} "
                     f"+6 {validation[interval]['position_mse_+6']:.6f}")
        with open(out / "validation_ledger.jsonl", "a") as ledger:
            ledger.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_sha256": runs.source_hash(),
                                     "variant": chosen, "seeds": list(SEEDS), "results": validation}) + "\n")
    task.write(environment="motion_v2", method="deletion_cost_v2_fresh_variants", chosen=chosen, validation=validation)


if __name__ == "__main__":
    main()
