"""Teacher labels for the Gate eviction scorer, for one frozen predictor seed.

    python scripts/generate_teacher_labels/gate.py --config configs/gate/team_bundle.yaml --seed 17 \
        --device cpu --out <dir>

For every training state reached by the four behaviour policies (4,096 on the full train split),
the frozen predictor is run once per possible deletion; its future error gives the deletion-cost
target. The attention-proxy target of the ablation is written alongside. Writes
<out>/teacher_labels_seed<seed>.npz and a small JSON summary.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.data.io import file_hash, write_npz  # noqa: E402
from bounded_memory.environments.gate.experiment import open_assets  # noqa: E402
from bounded_memory.methods.deletion_cost import training  # noqa: E402


def main():
    args = runs.parser(__doc__).parse_args()
    task = runs.Task(args, "teacher_labels")
    labels = Path(args.out) / f"teacher_labels_seed{args.seed}.npz"
    if task.done() and labels.exists():
        return
    assets = open_assets(runs.load_config(args.config), args.device)
    x, ya, yd, meta = training.teacher_labels(assets.predictor(args.seed), assets.split("train"), args.seed, assets.device)
    write_npz(labels, features=x, attention_target=ya, deletion_target=yd,
              attention_energy=np.stack([m["attention_energy"] for m in meta]),
              deletion_energy=np.stack([m["deletion_energy"] for m in meta]),
              candidates=np.array([m["candidates"] for m in meta]),
              episode=np.array([m["episode"] for m in meta]), behavior=np.array([m["behavior"] for m in meta]))
    task.log(f"{len(x)} training states")
    task.write(environment="gate", states=int(len(x)), labels=str(labels), labels_sha256=file_hash(labels))


if __name__ == "__main__":
    main()
