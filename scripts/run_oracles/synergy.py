"""Synergy reference points for one reader: the lowest-error deletion and the reader given all three cards.

    python scripts/run_oracles/synergy.py --config configs/synergy/release.yaml --seed 17 --device cpu \\
        --out <dir> --reader multihead --stage-a <dir with <reader>_confirmation.npz>

The cost table comes from this run's stage A (scripts/generate_teacher_labels/synergy.py).
Also checks that a plain deletion scored through the path used for non-deletion memories
reproduces that table. Writes <out>/references_<reader>_seed<seed>.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.synergy import experiment  # noqa: E402
from bounded_memory.evaluation.oracles import synergy as oracles  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--reader", required=True); ap.add_argument("--stage-a", type=Path, required=True)
    args = ap.parse_args()
    task = runs.Task(args, f"references_{args.reader}")
    if task.done():
        return
    config = runs.load_config(args.config)
    store = experiment.confirmation_store(config)
    records = store.records
    arrays = np.load(args.stage_a / f"{args.reader}_confirmation.npz")
    if arrays["ids"].tolist() != [r["id"] for r in records]:
        raise ValueError("stage-A arrays are not aligned with the confirmation records")
    cost = arrays["eval_cost"]
    three, worst = oracles.reader_checks(store, experiment.reader(config, args.reader, store), records, cost)
    task.info["deletion_path_check"] = {"records": len(records), "deletions": 3, "max_abs_diff_from_cost_table": worst}
    policies = {"oracle": experiment.policy(records, cost, drop=oracles.oracle(cost)),
                "three_cards": experiment.policy(records, cost, loss=three)}
    for name, record in policies.items():
        experiment.describe(task, name, record)
    task.write(environment="synergy", reader=args.reader, records=len(records), policies=policies)


if __name__ == "__main__":
    main()
