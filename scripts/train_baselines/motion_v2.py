"""Train and score one literature baseline on Motion v2 for one reader seed, on the frozen study assets.

    BM_MOTION_ROOT=/scratch/$USER python scripts/train_baselines/motion_v2.py --config configs/motion_v2/frozen.yaml \\
        --seed 17 --device cuda --out <dir> --method pacc_recurrent [--checkpoints <dir>]

Methods (one per call, so each is its own job; the implementations are the Gate ones):

    pafu_kv              PaFu-KV as published: one stored score per frame (bounded). Also its teacher's top 2 (privileged)
    pacc_full            Compress to Remember as published: reads the whole history at the query (unbounded store)
    pacc_recurrent       Compress to Remember held to two persistent slots (bounded; not in the paper)
    fm_ijepa_published   FrameMorrow as published, on the project's I-JEPA features: ranks the whole history
                         (unbounded store). Also its teacher's top 2 (privileged)
    fm_ijepa_online      FrameMorrow held to the three resident candidates (bounded; not in the paper)

The baseline is trained against the reader of --seed (17, 29 or 43) on the 1,024 training episodes and scored
on the 96 validation scene pairs of each interval under all three readers. Hyperparameters come from the
config's `baselines` block; --checkpoints makes the compressor training resumable. Writes
<out>/<method>_seed<seed>.json and the raw losses to <out>/raw/<interval>/<label>_seed<seed>.npz, which
scripts/aggregate_results/motion_v2_literature.py pools.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import episodes, frozen  # noqa: E402
from bounded_memory.methods.pacc import motion_v2 as pacc_motion  # noqa: E402

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
        checkpoint = None
        if args.checkpoints:
            args.checkpoints.mkdir(parents=True, exist_ok=True)
            checkpoint = args.checkpoints / f"{args.method}_seed{seed}_{device.type}.pt"
        compressor, task.info[args.method] = pacc.train(
            model, train, seed, device, variant=variant, steps=hyper["pacc"]["steps"],
            checkpoint=checkpoint, tag="motion_v2_frozen_16614")
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


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--checkpoints", type=Path)
    ap.add_argument("--intervals", nargs="+", default=list(frozen.INTERVALS), choices=frozen.INTERVALS)
    args = ap.parse_args()
    if args.seed not in SEEDS:
        raise SystemExit(f"--seed is the reader seed: one of {SEEDS}")
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    task = runs.Task(args, args.method)
    if task.done():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    task.info["verification"] = {k: v.get("checked", v.get("ok")) for k, v in frozen.verify(paths, args.intervals).items()}
    assets = frozen.load(paths, args.device)
    _, train = episodes.split_episodes(assets, assets.store.manifest["records"], "train")
    evals = {}
    for interval in args.intervals:
        records, external = frozen.interval_inputs(assets, paths, interval)
        evals[interval] = episodes.split_episodes(assets, records, "validation", external)
    raw, eps = evals[args.intervals[0]]
    task.info["batch_equivalence_max_abs"] = episodes.batch_equivalence(
        assets, raw[:32], eps[:32], [[e.q - 5, e.q - 4] for e in eps[:32]], reader=SEEDS.index(args.seed))
    task.log(f"{len(train)} training episodes; Gate vs Motion batch max |diff| {task.info['batch_equivalence_max_abs']:.2e}")

    results = {}
    for label, per_interval in train_and_select(args, config["baselines"], assets, train, evals, task).items():
        for interval, choice in per_interval.items():
            raw, eps = evals[interval]
            if isinstance(choice, tuple):
                scores = pacc_motion.score_tokens(assets, choice[1], raw, eps, choice[2])
                kept = np.zeros((0, 2), np.int64)
            else:
                scores = episodes.score_frames(assets, raw, choice)
                kept = np.asarray(choice, np.int64)
            target = Path(args.out) / "raw" / interval / f"{label}_seed{args.seed}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target, kept=kept, episode_ids=np.array([r["rec"]["id"] for r in raw]), **scores)
            summary = episodes.summarize({args.seed: scores}, [args.seed])
            results.setdefault(interval, {})[label] = {"memory": MEMORY[label], **summary}
            task.log(f"{interval:9s} {label:24s} +3 position MSE {summary['position_mse_+3']:.6f} "
                     f"(matched reader {summary['position_mse_+3_matched_reader']:.6f})")
    task.write(environment="motion_v2", method=args.method, reader_seed=args.seed, results=results)


if __name__ == "__main__":
    main()
