"""Train the Gate eviction scorer and its attention-proxy twin for one predictor seed.

    python scripts/train_scorer/gate.py --config configs/gate/team_bundle.yaml --seed 17 --device cpu \
        --out <dir> --labels <dir with teacher_labels_seed<seed>.npz> --checkpoints <dir>

1,000 AdamW steps of 64 states, identical initial weights and minibatches for the two arms; no
validation selection. Writes <checkpoints>/student_<seed>.pt in the published layout and
<out>/train_scorer_seed<seed>.json. With --compare <folder> the trained weights are compared with
another student checkpoint (for example the team's frozen one).

--train-seed N draws the initial weights and minibatches from N instead of the predictor seed (seed
sweeps; the labels stay those of the predictor seed) and writes student_<seed>_t<N>.pt and
train_scorer_t<N>_seed<seed>.json.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.data.io import file_hash  # noqa: E402
from bounded_memory.methods.deletion_cost import training  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--compare", type=Path, help="folder with another student_<seed>.pt to compare weights with")
    ap.add_argument("--train-seed", type=int, help="seed of the initial weights and minibatches (default: the predictor seed)")
    args = ap.parse_args()
    task = runs.Task(args, runs.seeded("train_scorer", args.train_seed))
    path = args.checkpoints / training.student_filename(args.train_seed).format(seed=args.seed)
    if task.done() and path.exists():
        return
    with np.load(args.labels / f"teacher_labels_seed{args.seed}.npz", allow_pickle=False) as f:
        x, ya, yd = f["features"], f["attention_target"], f["deletion_target"]
    if args.train_seed is None:
        states = training.fit_scorers(x, ya, yd, args.seed, args.device)
    else:
        states = training.fit_scorers_with_training_seed(x, ya, yd, args.train_seed, args.device)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    training.save_students(path, {k: {n: v.cpu() for n, v in s.items()} for k, s in states.items()}, args.seed)
    task.info.update(training_states=int(len(x)), checkpoint=str(path), checkpoint_sha256=file_hash(path),
                     train_seed=args.seed if args.train_seed is None else args.train_seed)
    if args.compare:
        other = torch.load(args.compare / f"student_{args.seed}.pt", map_location="cpu", weights_only=True)["models"]
        for name, theirs in zip(training.ARMS, other):
            diff = max(float((states[name][k].cpu() - theirs[k]).abs().max()) for k in theirs)
            task.info[f"max_abs_weight_diff_vs_compare/{name}"] = diff
            task.log(f"{name}: max |weight difference| against {args.compare.name} = {diff:.3e}")
    task.write(environment="gate")


if __name__ == "__main__":
    main()
