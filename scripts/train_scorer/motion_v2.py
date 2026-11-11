"""Retrain the 54 Motion v2 students from saved teacher labels and compare with the recorded run.

    python scripts/train_scorer/motion_v2.py --config configs/motion_v2/evidence.yaml --seed 0 --device cpu \
        --out <dir> --students <new dir>

The student stage needs only the scorer inputs and teacher labels of the 16,384 training and
3,072 validation states, which the completed study committed (labels/{train,validation}/ under
the step-3/4 evidence folder), not the frozen readers. This refits every arm with the original
student code (two architectures x nine labels x three seeds, 1,000 updates each) and compares each
arm with its recorded history: the validation cross-entropy at every check and the step that was
kept. Step 0 means the student kept its untrained initial weights.

Scoring the retrained students online still needs the frozen readers and features, which are not
available to this account. --seed is unused. Writes the students to <students> and
<out>/train_students_seed0.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import step34_students as students  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--students", type=Path, required=True, help="new folder for the retrained students")
    args = ap.parse_args()
    task = runs.Task(args, "train_students")
    if task.done():
        return
    evidence = Path(runs.load_config(args.config)["assets"]["step34"])
    arrays = {split: {kind: dict(np.load(evidence / f"labels/{split}/{kind}.npz", allow_pickle=False))
                      for kind in ("inputs", "labels")} for split in ("train", "validation")}
    ids = json.loads((evidence / "students/source_ids.json").read_text())
    students.fit_students(arrays["train"]["inputs"], arrays["validation"]["inputs"], arrays["train"]["labels"],
                          arrays["validation"]["labels"], ids["train"], ids["validation"], args.students,
                          args.device, budget=lambda **progress: None)
    arms, same_step, worst = {}, 0, 0.0
    for path in sorted(args.students.glob("*_history.json")):
        mine = json.loads(path.read_text())
        theirs = json.loads((evidence / "students" / path.name).read_text())
        a = [v["own_validation_label_cross_entropy"] for v in mine["validation"]]
        b = [v["own_validation_label_cross_entropy"] for v in theirs["validation"]]
        diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        m, t = mine["metadata"], theirs["metadata"]
        arms[path.name[:-len("_history.json")]] = {
            "selected_step": m["selected_step"], "recorded_selected_step": t["selected_step"],
            "kept_initial_weights": not m["is_trained"], "recorded_kept_initial_weights": not t["is_trained"],
            "max_abs_validation_ce_diff": diff, "step0_validation_ce_diff": abs(a[0] - b[0])}
        same_step += m["selected_step"] == t["selected_step"]
        worst = max(worst, diff)
    untrained = sum(v["kept_initial_weights"] for v in arms.values())
    recorded = sum(v["recorded_kept_initial_weights"] for v in arms.values())
    task.log(f"{len(arms)} arms; same kept step as recorded: {same_step}; kept initial weights: {untrained} (recorded {recorded}); "
             f"largest validation cross-entropy difference {worst:.2e}")
    task.info.update(arms=len(arms), same_selected_step=same_step, kept_initial_weights=untrained,
                     recorded_kept_initial_weights=recorded, max_abs_validation_ce_diff=worst)
    task.write(environment="motion_v2", students=str(args.students), per_arm=arms)


if __name__ == "__main__":
    main()
