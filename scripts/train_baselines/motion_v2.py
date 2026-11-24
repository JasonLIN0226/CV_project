"""Train and score one literature baseline on Motion v2 for one reader seed, on the frozen study assets.

    BM_MOTION_ROOT=/scratch/xy3257 python scripts/train_baselines/motion_v2.py --config configs/motion_v2/frozen.yaml \\
        --seed 0 --seeds 17 29 43 --device cuda --out <dir> --method pafu_kv fm_ijepa_online pacc_recurrent
    python scripts/train_baselines/motion_v2.py --config configs/motion_v2/frozen.yaml --seed 0 --out <dir> --aggregate

Methods (each method x seed is one unit with its own JSON; finished units are skipped, so rerunning the same
command after an interruption continues; the implementations are the Gate ones):

    pafu_kv              PaFu-KV as published: one stored score per frame (bounded). Also its teacher's top 2 (privileged)
    pacc_full            Compress to Remember as published: reads the whole history at the query (unbounded store)
    pacc_recurrent       Compress to Remember held to two persistent slots (bounded; not in the paper)
    fm_ijepa_published   FrameMorrow as published, on the project's I-JEPA features: ranks the whole history
                         (unbounded store). Also its teacher's top 2 (privileged)
    fm_ijepa_online      FrameMorrow held to the three resident candidates (bounded; not in the paper)

The baseline is trained against the reader of --seed (17, 29 or 43), on the 1,024 training episodes, and
scored on the 96 validation scene pairs of each interval under all three readers. Raw losses go to
<out>/raw/<interval>/<label>_seed<seed>.npz, the summary to <out>/<method>_seed<seed>.json. --aggregate pools
whatever seeds exist next to the tracked project baselines, students and references (literature_summary.json).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import evidence, frozen, literature  # noqa: E402

METHODS = ("pafu_kv", "pacc_full", "pacc_recurrent", "fm_ijepa_published", "fm_ijepa_online")
SEEDS = (17, 29, 43)
MEMORY = {"pafu_kv": "bounded", "pafu_teacher_top2": "privileged (reads the future)", "pacc_full": "full history",
          "pacc_recurrent": "bounded", "fm_ijepa_published": "full history",
          "fm_ijepa_teacher_top2": "privileged (reads the future)", "fm_ijepa_online": "bounded"}


def train_and_select(args, hyper, assets, train, evals, task):
    """Returns {label: {interval: kept frames or ('tokens', compressor, variant)}}."""
    device, seed = assets.device, args.seed
    model = assets.models[SEEDS.index(seed)]
    out = {}
    if args.method == "pafu_kv":
        from bounded_memory.methods.pafu_kv import gate as pafu
        head, task.info["pafu_kv"] = pafu.train(model, train, seed, device, steps=hyper["pafu_kv"]["steps"])
        out["pafu_kv"] = {i: pafu.select(head, eps, device) for i, (_, eps) in evals.items()}
        out["pafu_teacher_top2"] = {i: pafu.teacher_select(model, eps, device) for i, (_, eps) in evals.items()}
    elif args.method.startswith("pacc_"):
        from bounded_memory.methods.pacc import gate as pacc
        variant = args.method.split("_", 1)[1]
        checkpoints = Path(args.out) / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        compressor, task.info[args.method] = pacc.train(
            model, train, seed, device, variant=variant, steps=hyper["pacc"]["steps"],
            checkpoint=checkpoints / f"{args.method}_seed{seed}_{device.type}.pt", tag="motion_v2_frozen_16614")
        out[args.method] = {i: ("tokens", compressor, variant) for i in evals}
    else:
        from bounded_memory.methods.framemorrow import gate as fm
        variant = args.method.rsplit("_", 1)[1]
        visual = {e.id: e.z for e in train}
        for _, eps in evals.values():
            visual.update({e.id: e.z for e in eps})
        if variant == "published":
            out["fm_ijepa_teacher_top2"] = {i: fm.select_teacher(eps, visual, device) for i, (_, eps) in evals.items()}
        h = hyper["framemorrow"]
        selector, task.info[args.method] = fm.train(train, visual, seed, device, variant=variant, epochs=h["epochs"],
                                                    hidden=h["hidden"], layers=h["layers"])
        select = fm.select_published if variant == "published" else fm.select_online
        out[args.method] = {i: select(selector, eps, visual, device) for i, (_, eps) in evals.items()}
    return out


def run(args, config, units):
    """Load the frozen assets once, then train and score every (method, seed) unit not done yet."""
    pending = [(m, s) for m, s in units if not runs.Task(argparse.Namespace(**{**vars(args), "seed": s}), m).done()]
    if not pending:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    paths = frozen.Paths.from_config(config["assets"])
    verification = {k: v.get("checked", v.get("ok")) for k, v in frozen.verify(paths, args.intervals).items()}
    assets = frozen.load(paths, args.device)
    _, train = literature.split_episodes(assets, assets.store.manifest["records"], "train")
    evals = {}
    for interval in args.intervals:
        records, external = frozen.interval_inputs(assets, paths, interval)
        evals[interval] = literature.split_episodes(assets, records, "validation", external)
    for method, seed in pending:
        unit = argparse.Namespace(**{**vars(args), "method": method, "seed": seed})
        task = runs.Task(unit, method)
        task.info["verification"] = verification
        raw, eps = evals[args.intervals[0]]
        task.info["batch_equivalence_max_abs"] = literature.batch_equivalence(
            assets, raw[:32], eps[:32], [[e.q - 5, e.q - 4] for e in eps[:32]], reader=SEEDS.index(seed))
        task.log(f"{len(train)} training episodes; Gate vs Motion batch max |diff| {task.info['batch_equivalence_max_abs']:.2e}")
        results = {}
        for label, per_interval in train_and_select(unit, config["baselines"], assets, train, evals, task).items():
            for interval, choice in per_interval.items():
                raw, eps = evals[interval]
                if isinstance(choice, tuple):
                    scores = literature.score_tokens(assets, choice[1], raw, eps, choice[2])
                    kept = np.zeros((0, 2), np.int64)
                else:
                    scores = literature.score_frames(assets, raw, choice)
                    kept = np.asarray(choice, np.int64)
                target = Path(args.out) / "raw" / interval / f"{label}_seed{seed}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, kept=kept, episode_ids=np.array([r["rec"]["id"] for r in raw]), **scores)
                summary = literature.summarize({seed: scores}, [seed])
                results.setdefault(interval, {})[label] = {"memory": MEMORY[label], **summary}
                task.log(f"{interval:9s} {label:24s} +3 position MSE {summary['position_mse_+3']:.6f} "
                         f"(matched reader {summary['position_mse_+3_matched_reader']:.6f})")
        task.write(environment="motion_v2", method=method, reader_seed=seed, results=results)


def aggregate(args, config):
    """Pool the literature seeds that exist and put them next to the tracked online table."""
    paths = frozen.Paths.from_config(config["assets"])
    table = {}
    for interval in frozen.INTERVALS:
        folder = Path(args.out) / "raw" / interval
        rows = {}
        for label in sorted({p.stem.rsplit("_seed", 1)[0] for p in folder.glob("*_seed*.npz")}):
            scores, seeds = {}, []
            for s in SEEDS:
                f = folder / f"{label}_seed{s}.npz"
                if f.exists():
                    with np.load(f) as z:
                        scores[s] = {k: z[k] for k in ("position", "feature", "success")}
                    seeds.append(s)
            rows[f"literature/{label}"] = {"memory": MEMORY.get(label), **literature.summarize(scores, seeds)}
        try:
            tracked = evidence.online_table(paths.evidence, interval)
        except FileNotFoundError:
            tracked = {}
        for name, r in tracked.items():
            rows[name] = {"memory": {"reference/posthoc_best": "privileged (best pair in hindsight)",
                                     "reference/b": "privileged (the two cue frames)",
                                     "reference/last_visible": "privileged (reads visibility)"}.get(name, "bounded"),
                          "position_mse_+3": r["position_mse"]["3"], "position_mse_+6": r["position_mse"]["6"],
                          "success_rate": r["success_rate"], "source": "tracked step-3/4 evidence"}
        table[interval] = rows
    (Path(args.out) / "literature_summary.json").write_text(json.dumps(table, indent=2) + "\n")
    for interval, rows in table.items():
        print(f"\n{interval}: +3 position MSE (lower is better)")
        for name, r in sorted(rows.items(), key=lambda kv: kv[1]["position_mse_+3"]):
            print(f"  {name:48s} {r['position_mse_+3']:.6f}  {r['memory']}")


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--method", nargs="+", choices=METHODS, help="one or more methods, run in this order")
    ap.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, help="reader seeds to run (default: --seed)")
    ap.add_argument("--intervals", nargs="+", default=list(frozen.INTERVALS), choices=frozen.INTERVALS)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    config = runs.load_config(args.config)
    if args.aggregate:
        aggregate(args, config)
    elif args.method:
        seeds = args.seeds or [args.seed]
        if any(s not in SEEDS for s in seeds):
            raise SystemExit(f"reader seeds are {SEEDS}")
        run(args, config, [(m, s) for m in args.method for s in seeds])
    else:
        raise SystemExit("give --method or --aggregate")


if __name__ == "__main__":
    main()
