"""Our deletion-cost scorer on Gate for one predictor seed, from a student checkpoint.

    python scripts/evaluate/gate.py --config configs/gate/team_bundle.yaml --seed 17 --device cpu --out <dir> \
        [--students <folder with student_<seed>.pt>]

Without --students the folder named by the config (assets.students) is used: for the team bundle
that is the published frozen scorer. Point --students at the output of scripts/train_scorer/gate.py
to score a retrained one. Writes <out>/deletion_cost_seed<seed>.json. When the config names the
team's original validation output, every row written so far in <out> for this seed is also
compared with it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.gate.experiment import Scorer, open_assets, reference_check  # noqa: E402
from bounded_memory.methods.deletion_cost import retention, training  # noqa: E402

ARM = "deletion_cost_student"


def main(arm=ARM, name="deletion_cost", doc=__doc__):
    ap = runs.parser(doc)
    ap.add_argument("--students", type=Path)
    args = ap.parse_args()
    task = runs.Task(args, name)
    if task.done():
        return
    config = runs.load_config(args.config)
    scorer = Scorer(open_assets(config, args.device), args.seed, task)
    students = args.students or config["assets"].get("students")
    if not students:
        raise SystemExit("no student checkpoint folder: pass --students or set assets.students in the config")
    scorers, task.info["students"] = training.frozen_scorers(students, args.seed, scorer.assets.device)
    scorer.add(arm, retention.scorer_selections(scorers[arm], scorer.val))
    reference = config["assets"].get("reference_rows")
    if reference:
        rows = list(scorer.rows)
        for other in sorted(Path(args.out).glob(f"*_seed{args.seed}.json")):
            rows.extend(json.loads(other.read_text())["rows"])
        task.info["original_validation_check"] = reference_check(reference, args.seed, rows)
        check = task.info["original_validation_check"]
        task.log(f"original validation rows compared: {sum(v['rows'] for v in check.values())}, route mismatches "
                 f"{sum(v['route_mismatches'] for v in check.values())}, max |mse diff| "
                 f"{max(v['max_abs_mse_diff'] for v in check.values()):.2e}")
    task.write(**scorer.payload())


if __name__ == "__main__":
    main()
