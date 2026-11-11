"""Train every student arm on the terminal (Synergy) tasks for one reader and label mode.

    python scripts/train_scorer/synergy.py --config configs/synergy/release.yaml --seed 17 --device cpu \\
        --out <dir> --reader multihead --mode standard --stage-a <dir> --students <dir>

mode "standard": set-conditioned and independent scorers on deletion cost, future attention and
shuffled labels, plus the appearance-only control; three student seeds each (17, 29, 43).
mode "distilled": the same scorers with the cost label replaced by the full-set sensitivity.
1,000 steps of 128; each arm keeps the step with the lowest validation cross-entropy on its own
label, step 0 included. Needs this run's stage A for train and validation. --seed is the reader's
seed (17). Writes the archive <students>/<reader>_<mode>/ and <out>/train_students_<reader>_<mode>_seed<seed>.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json  # noqa: E402

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.synergy import experiment  # noqa: E402
from bounded_memory.methods.deletion_cost import terminal, terminal_students  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--reader", required=True); ap.add_argument("--mode", required=True, choices=("standard", "distilled"))
    ap.add_argument("--stage-a", type=Path, required=True); ap.add_argument("--students", type=Path, required=True)
    args = ap.parse_args()
    task = runs.Task(args, f"train_students_{args.reader}_{args.mode}")
    folder = args.students / f"{args.reader}_{args.mode}"
    if task.done() and (folder / "manifest.json").exists():
        return
    config = runs.load_config(args.config)
    store = experiment.train_store(config)
    appearance = None
    if args.mode == "standard":
        with np.load(config["assets"]["appearance_trainvaldev"], allow_pickle=False) as f:
            appearance = dict(zip(f["ids"].tolist(), f["items"]))
    exports = {}
    for split in ("train", "validation"):
        records = store.split(split)
        arrays = dict(np.load(args.stage_a / f"{args.reader}_{split}.npz"))
        app = None if appearance is None else np.stack([appearance[r["id"]] for r in records])
        exports[split] = terminal.student_export(store, records, arrays, args.mode, split == "train", app)
    student_config = terminal_students.StudentConfig(**{**config["students"], "seeds": tuple(config["students"]["seeds"])})
    manifest = terminal_students.fit_students(exports["train"], exports["validation"], folder, student_config)
    task.info["arms"] = {k: {"selected_step": v["selected_step"], "parameters": v["parameters"]} for k, v in manifest["arms"].items()}
    summary = Path(config["assets"]["released"]) / f"students/{args.reader}_student_summary.json"
    if summary.exists():
        theirs = json.loads(summary.read_text())[args.mode]
        task.info["selected_step_matches_released"] = {k: v["selected_step"] == theirs[k]["selected_step"]
                                                       for k, v in task.info["arms"].items() if k in theirs}
        same = sum(task.info["selected_step_matches_released"].values())
        task.log(f"{same}/{len(task.info['selected_step_matches_released'])} arms select the same step as the released students")
    task.write(environment="synergy", reader=args.reader, mode=args.mode, archive=str(folder))


if __name__ == "__main__":
    main()
