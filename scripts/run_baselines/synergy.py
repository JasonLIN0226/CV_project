"""The project's non-learned deletion rules on Synergy for one reader: delete the oldest card,
reader attention, full-set sensitivity and the prediction medoid.

    python scripts/run_baselines/synergy.py --config configs/synergy/release.yaml --seed 17 --device cpu \\
        --out <dir> --reader multihead --stage-a <dir with <reader>_confirmation.npz>

Writes <out>/rules_<reader>_seed<seed>.json. Nothing is trained.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.synergy import experiment  # noqa: E402
from bounded_memory.evaluation.heuristics import synergy as heuristics  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--reader", required=True); ap.add_argument("--stage-a", type=Path, required=True)
    args = ap.parse_args()
    task = runs.Task(args, f"rules_{args.reader}")
    if task.done():
        return
    config = runs.load_config(args.config)
    records = experiment.confirmation_store(config).records
    arrays = np.load(args.stage_a / f"{args.reader}_confirmation.npz")
    if arrays["ids"].tolist() != [r["id"] for r in records]:
        raise ValueError("stage-A arrays are not aligned with the confirmation records")
    policies = {name: experiment.policy(records, arrays["eval_cost"], drop=drop)
                for name, drop in heuristics.rules(arrays).items()}
    for name, record in policies.items():
        experiment.describe(task, name, record)
    task.write(environment="synergy", reader=args.reader, records=len(records), policies=policies)


if __name__ == "__main__":
    main()
