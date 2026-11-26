"""All-pairs loss tables of the frozen Motion v2 readers: the teacher of our scorer.

    BM_MOTION_ROOT=<study folders> BM_SCRATCH=<scratch> python scripts/generate_teacher_labels/motion_v2.py \\
        --config configs/motion_v2/fresh.yaml --seed 0 --device cuda --out <dir> [--episodes fresh|registered]

One <episode id>.npy per episode: the reader-mean decoded-position MSE at +3 and +6 of every legal memory pair,
[2, n, n] with n = q - 3 (methods/deletion_cost/motion_v2.pair_table). A table gives the deletion-cost label of
any state and scores any policy's final pair. fresh: every fresh scene pair (training, dev, holdout) into
fresh.tables; registered: the registered training episodes into scorer.registered_tables. Finished tables are
skipped after a shape check, so a rerun continues; a table written by other code is not detected, so delete the
folder after changing pair_table. <out>/teacher_tables_<episodes>_seed<seed>.json records the code, the
frozen-input check, the table folder and how many tables were built or reused. --seed is unused (the tables
average the three readers).
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import episodes, fresh, frozen  # noqa: E402
from bounded_memory.methods.deletion_cost import motion_v2 as scorer  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--episodes", default="fresh", choices=("fresh", "registered"))
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    task = runs.Task(args, f"teacher_tables_{args.episodes}")
    if task.done():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    task.info["verification"] = {k: v.get("checked", v.get("ok")) for k, v in frozen.verify(paths, ["ordinary"]).items()}
    assets = frozen.load(paths, args.device)
    if args.episodes == "fresh":
        parts = fresh.load(assets, config["fresh"], parts=("fit", "dev", "holdout"))
        folder = Path(config["fresh"]["tables"])
    else:
        parts = episodes.registered_parts(assets)
        folder = Path(config["scorer"]["registered_tables"])
    todo_all = [ep for part in parts.values() for ep in part]
    folder.mkdir(parents=True, exist_ok=True)
    todo = [ep for ep in todo_all if not (folder / f"{ep['rec']['id']}.npy").exists()]
    for ep in todo_all:
        path, n = folder / f"{ep['rec']['id']}.npy", ep["rec"]["query"] - 3
        if path.exists() and np.load(path, mmap_mode="r").shape != (2, n, n):
            raise SystemExit(f"{path} is not a [2, {n}, {n}] table for this episode; remove the folder and rerun")
    started = time.time()
    for i, ep in enumerate(todo):
        tmp = folder / f"{ep['rec']['id']}.tmp.npy"
        np.save(tmp, scorer.pair_table(assets, ep))
        os.replace(tmp, folder / f"{ep['rec']['id']}.npy")
        if i % 100 == 0 or i == len(todo) - 1:
            task.log(f"tables {i + 1}/{len(todo)} new ({len(todo_all) - len(todo)} cached), {time.time() - started:.0f}s")
    task.write(environment="motion_v2", episodes=args.episodes, tables=str(folder),
               counts={part: len(eps) for part, eps in parts.items()},
               built=len(todo), reused=len(todo_all) - len(todo))


if __name__ == "__main__":
    main()
