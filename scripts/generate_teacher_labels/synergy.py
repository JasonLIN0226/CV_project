"""Stage A on the terminal (Synergy) tasks: the frozen reader's loss after every possible deletion.

    python scripts/generate_teacher_labels/synergy.py --config configs/synergy/release.yaml --seed 17 \\
        --device cpu --out <dir> --reader multihead --split confirmation --labels <dir>

For each three-card decision the reader is run on every deletion and on the full set, over the
action-plan bank of the split (16 reference plans on train, 32 evaluation plans elsewhere). The
mean loss per deletion is the deletion-cost label of our method; the same pass gives the reader's
attention, the future-attention teacher, the full-set sensitivity and the medoid score used by the
ablations and project baselines. --seed is the reader's seed (17; one reader per architecture).
Writes <labels>/<reader>_<split>.npz and <out>/stage_a_<reader>_<split>_seed<seed>.json, which
records the largest difference from the released arrays for the same split.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.data.io import file_hash, write_npz  # noqa: E402
from bounded_memory.environments.synergy import experiment  # noqa: E402
from bounded_memory.methods.deletion_cost import terminal  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--split", required=True, choices=("train", "validation", "development", "confirmation"))
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="first N records only (mechanics checks)")
    args = ap.parse_args()
    task = runs.Task(args, f"stage_a_{args.reader}_{args.split}")
    path = args.labels / f"{args.reader}_{args.split}.npz"
    if task.done() and path.exists():
        return
    config = runs.load_config(args.config)
    store = experiment.confirmation_store(config) if args.split == "confirmation" else experiment.train_store(config)
    records = store.split(args.split)[:args.limit]
    model = experiment.reader(config, args.reader, store)
    arrays = terminal.stage_a(store, model, records, train=args.split == "train", log=task.log)
    write_npz(path, **arrays)
    released = Path(config["assets"]["released"])
    theirs = released / ("conf/stageA" if args.split == "confirmation" else "stageA") / f"{args.reader}_{args.split}.npz"
    if theirs.exists() and not args.limit:
        with np.load(theirs) as ref:
            if ref["ids"].tolist() != arrays["ids"].tolist():
                raise ValueError("released stage-A arrays are not aligned with this split")
            task.info["against_released"] = {
                k: {"max_abs_diff": float(np.abs(arrays[k] - ref[k]).max()),
                    **({"same_argmin": float((arrays[k].argmin(-1) == ref[k].argmin(-1)).mean())} if arrays[k].ndim == 2 else {})}
                for k in ref.files if k != "ids" and k in arrays}
        for k, v in task.info["against_released"].items():
            task.log(f"{k:14s} max |difference from released| {v['max_abs_diff']:.2e}  same choice {v.get('same_argmin', float('nan')):.4f}")
    task.write(environment="synergy", reader=args.reader, split=args.split, records=len(records),
               labels=str(path), labels_sha256=file_hash(path))


if __name__ == "__main__":
    main()
