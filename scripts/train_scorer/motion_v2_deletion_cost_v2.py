"""Train our deletion-cost scorer (v2) on Motion v2 for one training seed and one variant, selected on dev.

    BM_MOTION_ROOT=/scratch/$USER python scripts/train_scorer/motion_v2_deletion_cost_v2.py \\
        --config configs/motion_v2/frozen.yaml --seed 17 --device cuda --out <dir> --variant min_dagger \\
        [--episodes fresh|registered]

--seed is the scorer's training seed (17, 29 or 43); the labels and the dev score average the three frozen
readers. Variants (methods/deletion_cost/motion_v2.VARIANTS) set the label (minimum or soft minimum over the
reachable pairs) and the number of on-policy rounds. --episodes fresh (default) trains on fresh scene pairs
0-511 and selects on 512-607 (configs/motion_v2/frozen.yaml, block `fresh`); registered trains on the
readers' own training episodes with 1 scene pair in 10 as dev, the setting the readers memorised.

Needs the loss tables of scripts/generate_teacher_labels/motion_v2.py. Writes <out>/<variant>_seed<seed>.json
(the dev attention policy, every round: states, selected step and dev loss, final dev loss, per-state metrics)
and the selected checkpoint to <scorer.checkpoints>/<variant>_seed<seed>.pt. Validation is not read.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import frozen  # noqa: E402
from bounded_memory.methods.deletion_cost import motion_v2 as scorer  # noqa: E402

SEEDS = (17, 29, 43)


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--variant", required=True, choices=list(scorer.VARIANTS))
    ap.add_argument("--episodes", default="fresh", choices=("fresh", "registered"))
    args = ap.parse_args()
    if args.seed not in SEEDS:
        raise SystemExit(f"--seed is the scorer's training seed: one of {SEEDS}")
    name = args.variant if args.episodes == "fresh" else f"{args.variant}_registered"
    task = runs.Task(args, name)
    if task.done():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    task.info["verification"] = {k: v.get("checked", v.get("ok")) for k, v in frozen.verify(paths, ["ordinary"]).items()}
    assets = frozen.load(paths, args.device)

    if args.episodes == "fresh":
        parts = scorer.fresh_episodes(assets, config["fresh"])
        tables = scorer.load_tables(config["fresh"]["tables"], parts["fit"] + parts["dev"])
    else:
        parts = scorer.registered_episodes(assets)
        tables = scorer.load_tables(config["scorer"]["registered_tables"], parts["fit"] + parts["dev"])
    fit, dev = parts["fit"], parts["dev"]
    temperature, rounds = scorer.VARIANTS[args.variant]
    behaviour = {ep["rec"]["id"]: scorer.behaviour_states(assets, ep, scorer.BEHAVIOUR_SEED) for ep in fit + dev}
    base_rows = scorer.labelled(assets, fit, tables, behaviour, temperature)
    dev_rows = scorer.labelled(assets, dev, tables, behaviour, temperature)
    attention = scorer.table_loss(tables, dev, scorer.rollout(assets, dev, scorer.ScorerV2().to(args.device),
                                                              scorer.identity_norm(), args.device))
    task.log(f"{args.episodes} episodes: fit {len(fit)}, dev {len(dev)}; {len(base_rows['cost'])} behaviour states; "
             f"dev attention policy +3 {attention[0]:.6f} +6 {attention[1]:.6f}")

    history, best = scorer.fit_variant(assets, fit, dev, tables, base_rows, dev_rows, behaviour, temperature, rounds,
                                       args.seed, args.device, steps=config["scorer"]["steps"], log=task.log)
    folder = Path(config["scorer"]["checkpoints"])
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best[2], "norm": best[3], "round": best[1], "width": best[4]["width"], "variant": args.variant,
                "episodes": args.episodes, "seed": args.seed}, folder / f"{name}_seed{args.seed}.pt")
    task.write(environment="motion_v2", method="deletion_cost_v2", variant=args.variant, episodes=args.episodes,
               temperature=temperature, rounds=rounds, best_round=best[1], best_dev_mean=best[0],
               dev_attention={"+3": float(attention[0]), "+6": float(attention[1]), "mean": float(attention.mean())},
               history=history)


if __name__ == "__main__":
    main()
