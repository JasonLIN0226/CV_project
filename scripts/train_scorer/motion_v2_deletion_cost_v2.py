"""Train our deletion-cost scorer (v2) on Motion v2 for one training seed and one variant, selected on dev.

    BM_MOTION_ROOT=<study folders> BM_SCRATCH=<scratch> python scripts/train_scorer/motion_v2_deletion_cost_v2.py \\
        --config configs/motion_v2/fresh.yaml --seed 17 --device cuda --out <dir> --variant min_dagger \\
        [--episodes fresh|registered] [--checkpoints <dir>]

--seed is the scorer's training seed (17, 29 or 43); the labels and the dev score average the three frozen
readers. Variants (methods/deletion_cost/motion_v2.VARIANTS) set the label (minimum or soft minimum over the
reachable pairs) and the number of on-policy rounds. --episodes fresh (default) trains on fresh scene pairs
0-511 and selects on 512-607 (configs/motion_v2/fresh.yaml, block `fresh`); registered trains on the
readers' own training episodes with 1 scene pair in 10 as dev, the setting the readers memorised.

Needs the loss tables of scripts/generate_teacher_labels/motion_v2.py. Writes, with <name> = <variant> for fresh
episodes and <variant>_registered for registered ones: <out>/<name>_seed<seed>.json (the dev attention policy,
every round: states, selected step and dev loss, last-step dev loss, per-state metrics; the checkpoint's SHA256)
and the selected checkpoint <checkpoints>/<name>_seed<seed>.pt (--checkpoints, default <out>/checkpoints).
Validation is not read.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import episodes, fresh, frozen  # noqa: E402
from bounded_memory.environments.motion_v2.step34_core import SEEDS  # noqa: E402
from bounded_memory.methods.deletion_cost import motion_v2 as scorer  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--variant", required=True, choices=list(scorer.VARIANTS))
    ap.add_argument("--episodes", default="fresh", choices=("fresh", "registered"))
    ap.add_argument("--checkpoints", type=Path, help="folder for the selected checkpoint (default: <out>/checkpoints)")
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
        parts, folder = fresh.load(assets, config["fresh"]), config["fresh"]["tables"]
    else:
        parts, folder = episodes.registered_parts(assets), config["scorer"]["registered_tables"]
    fit, dev = parts["fit"], parts["dev"]
    tables = scorer.load_tables(folder, fit + dev)
    temperature, rounds = scorer.VARIANTS[args.variant]
    behaviour = {ep["rec"]["id"]: scorer.behaviour_states(assets, ep, scorer.BEHAVIOUR_SEED) for ep in fit + dev}
    base_rows = scorer.labelled(assets, fit, tables, behaviour, temperature)
    dev_rows = scorer.labelled(assets, dev, tables, behaviour, temperature)
    attention = scorer.attention_policy_losses(assets, dev, tables, args.device).mean(0)
    task.log(f"{args.episodes} episodes: fit {len(fit)}, dev {len(dev)}; {len(base_rows['cost'])} behaviour states; "
             f"dev attention policy +3 {attention[0]:.6f} +6 {attention[1]:.6f}")

    history, best = scorer.fit_variant(assets, fit, dev, tables, base_rows, dev_rows, behaviour, temperature, rounds,
                                       args.seed, args.device, steps=config["scorer"]["steps"], log=task.log)
    checkpoint = (args.checkpoints or Path(args.out) / "checkpoints") / f"{name}_seed{args.seed}.pt"
    digest = scorer.save_scorer(checkpoint, best, variant=args.variant, episodes=args.episodes, seed=args.seed)
    task.write(environment="motion_v2", method="deletion_cost_v2", variant=args.variant, episodes=args.episodes,
               temperature=temperature, rounds=rounds, best_round=best.round, best_dev_mean=best.dev_mean,
               checkpoint=str(checkpoint), checkpoint_sha256=digest,
               dev_attention={"+3": float(attention[0]), "+6": float(attention[1]), "mean": float(attention.mean())},
               history=history)


if __name__ == "__main__":
    main()
