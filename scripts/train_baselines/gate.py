"""Train and score one literature baseline on Gate for one predictor seed.

    python scripts/train_baselines/gate.py --config configs/gate/team_bundle.yaml --seed 17 --device cpu \
        --out <dir> --method pafu_kv [--checkpoints <dir>]

Methods (one per call, so each is its own job):

    pafu_kv               PaFu-KV as published: one stored score per frame. Also scores its teacher.
    pafu_kv_repo          the repo's earlier PaFu-KV adaptation (re-scores the residents at each eviction)
    pacc_full             Compress to Remember as published: reads the whole history at the query
    pacc_recurrent        Compress to Remember held to two persistent slots (not in the paper)
    fm_dinov2_published   FrameMorrow as published on DINOv2 features: ranks the whole history. Also its teacher.
    fm_dinov2_online      FrameMorrow held to two slots
    fm_ijepa_published / fm_ijepa_online   the same on the project's I-JEPA features

Hyperparameters come from the config's `baselines` block. --checkpoints makes the compressor
training resumable: rerunning a stopped task continues from its last checkpoint.
Writes <out>/<method>_seed<seed>.json.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.gate.experiment import Scorer, open_assets  # noqa: E402

METHODS = ("pafu_kv", "pafu_kv_repo", "pacc_full", "pacc_recurrent", "fm_dinov2_published", "fm_dinov2_online",
           "fm_ijepa_published", "fm_ijepa_online")


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--checkpoints", type=Path)
    args = ap.parse_args()
    task = runs.Task(args, args.method)
    if task.done():
        return
    config = runs.load_config(args.config)
    hyper = config["baselines"]
    scorer = Scorer(open_assets(config, args.device), args.seed, task)
    model, train, val, device = scorer.model, scorer.train, scorer.val, scorer.assets.device

    if args.method == "pafu_kv":
        from bounded_memory.methods.pafu_kv import gate as pafu
        head, task.info["pafu_kv"] = pafu.train(model, train, args.seed, device, steps=hyper["pafu_kv"]["steps"])
        scorer.add("pafu_kv", pafu.select(head, val, device))
        scorer.add("pafu_teacher_top2", pafu.teacher_select(model, val, device))
    elif args.method == "pafu_kv_repo":
        from bounded_memory.methods.pafu_kv import repo_adaptation
        student, task.info["pafu_kv_repo"] = repo_adaptation.train(model, train, args.seed, device,
                                                                   max_states=hyper["pafu_kv_repo"]["max_states"])
        scorer.add("pafu_kv_repo", repo_adaptation.select(student, val))
    elif args.method.startswith("pacc_"):
        from bounded_memory.methods.pacc import gate as pacc
        variant = args.method.split("_", 1)[1]
        checkpoint = None
        if args.checkpoints:
            args.checkpoints.mkdir(parents=True, exist_ok=True)
            checkpoint = args.checkpoints / f"pacc_{variant}_seed{args.seed}_{device.type}.pt"
        compressor, meta = pacc.train(model, train, args.seed, device, variant=variant, steps=hyper["pacc"]["steps"],
                                      checkpoint=checkpoint, tag=f"{config['assets']['root']}")
        prediction, gap = pacc.predict(model, compressor, val, device, variant=variant)
        task.info[args.method] = {**meta, "validation_gap_to_full_history": gap}
        scorer.add(args.method, prediction=prediction)
    else:
        from bounded_memory.methods.framemorrow import gate as fm
        _, visual_name, variant = args.method.split("_")
        if visual_name == "ijepa":
            visual = {e.id: e.z for e in scorer.assets.episodes}
        else:
            path = config["assets"].get("dinov2")
            if not path:
                raise SystemExit("the config names no DINOv2 features (assets.dinov2); run scripts/encode/dinov2.py first")
            with np.load(path) as f:
                visual = {e.id: f[e.id] for e in scorer.assets.episodes}
            if any(len(visual[e.id]) != len(e.z) for e in scorer.assets.episodes):
                raise ValueError("DINOv2 features do not cover every frame")
        if variant == "published":
            scorer.add(f"fm_{visual_name}_teacher_top2", fm.select_teacher(val, visual, device))
        h = hyper["framemorrow"]
        selector, task.info[args.method] = fm.train(train, visual, args.seed, device, variant=variant,
                                                    epochs=h["epochs"], hidden=h["hidden"], layers=h["layers"])
        select = fm.select_published if variant == "published" else fm.select_online
        scorer.add(args.method, select(selector, val, visual, device))
    task.write(**scorer.payload())


if __name__ == "__main__":
    main()
