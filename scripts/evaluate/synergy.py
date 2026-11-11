"""Our deletion-cost student and its ablations on the Synergy confirmation set, from student archives.

    python scripts/evaluate/synergy.py --config configs/synergy/release.yaml --seed 17 --device cpu \\
        --out <dir> --reader multihead --stage-a <dir> [--students <dir>] [--arms ours|ablations|all]

<students>/<reader>_standard and <reader>_distilled are student archives as written by
scripts/train_scorer/synergy.py; without --students the released archives named by the config are
used. --seed selects the student seed. "ours" is the set-conditioned student trained on deletion
cost; "ablations" are independent scoring, future-attention and shuffled labels, the
appearance-only control and the distilled-sensitivity students.
Writes <out>/<arms>_<reader>_seed<seed>.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.synergy import experiment  # noqa: E402
from bounded_memory.methods.deletion_cost import terminal, terminal_students  # noqa: E402

OURS = "deletion_cost_student_set"


def main(default_arms="ours"):
    ap = runs.parser(__doc__)
    ap.add_argument("--reader", required=True); ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--students", type=Path); ap.add_argument("--arms", default=default_arms, choices=("ours", "ablations", "all"))
    args = ap.parse_args()
    task = runs.Task(args, f"students_{args.arms}_{args.reader}")
    if task.done():
        return
    config = runs.load_config(args.config)
    store = experiment.confirmation_store(config)
    records = store.records
    arrays = np.load(args.stage_a / f"{args.reader}_confirmation.npz")
    if arrays["ids"].tolist() != [r["id"] for r in records]:
        raise ValueError("stage-A arrays are not aligned with the confirmation records")
    students = args.students or Path(config["assets"]["students"])
    appearance = np.load(Path(config["assets"]["released"]) / "conf/appearance.npz", allow_pickle=False)
    if appearance["ids"].tolist() != [r["id"] for r in records]:
        raise ValueError("appearance rows do not align with the confirmation records")
    inputs = terminal.student_inputs(store, records)
    app = {"items": appearance["items"], "context": np.empty((len(records), 0), np.float32)}
    policies = {}
    for mode in ("standard", "distilled"):
        folder = students / f"{args.reader}_{mode}"
        if not folder.is_dir():
            raise SystemExit(f"no student archive at {folder}")
        prediction = terminal_students.predict_students(folder, inputs, app if mode == "standard" else None)
        for key, value in prediction.items():
            named = terminal.arm_name(mode, key)
            if named is None or named[1] != args.seed:
                continue
            name = named[0]
            if (args.arms == "ours") != (name == OURS) and args.arms != "all":
                continue
            policies[name] = experiment.policy(records, arrays["eval_cost"], drop=value["drop"], seed=args.seed)
    if not policies:
        raise SystemExit(f"no arm for seed {args.seed} in {students}")
    for name, record in sorted(policies.items()):
        experiment.describe(task, name, record)
    task.info["students"] = str(students)
    task.write(environment="synergy", reader=args.reader, records=len(records), policies=policies)


if __name__ == "__main__":
    main()
