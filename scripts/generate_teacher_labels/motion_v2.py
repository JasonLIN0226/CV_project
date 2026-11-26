"""All-pairs loss tables of the frozen Motion v2 readers: the teacher of our scorer.

    BM_MOTION_ROOT=/scratch/$USER python scripts/generate_teacher_labels/motion_v2.py \\
        --config configs/motion_v2/frozen.yaml --device cuda [--episodes fresh|registered]

One <episode id>.npy per episode: the reader-mean decoded-position MSE at +3 and +6 of every legal memory pair,
[2, n, n] with n = q - 3 (methods/deletion_cost/motion_v2.pair_table). A table gives the deletion-cost label of
any state and scores any policy's final pair. fresh: every fresh scene pair (training, dev, holdout) into
fresh.tables; registered: the registered training episodes into scorer.registered_tables. Finished tables are
skipped, so a rerun continues.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import frozen  # noqa: E402
from bounded_memory.methods.deletion_cost import motion_v2 as scorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", default="fresh", choices=("fresh", "registered"))
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    frozen.verify(paths, ["ordinary"])
    assets = frozen.load(paths, args.device)
    if args.episodes == "fresh":
        parts = scorer.fresh_episodes(assets, config["fresh"], parts=("fit", "dev", "holdout"))
        folder = Path(config["fresh"]["tables"])
    else:
        parts = scorer.registered_episodes(assets)
        folder = Path(config["scorer"]["registered_tables"])
    episodes = [ep for part in parts.values() for ep in part]
    folder.mkdir(parents=True, exist_ok=True)
    todo = [ep for ep in episodes if not (folder / f"{ep['rec']['id']}.npy").exists()]
    started = time.time()
    for i, ep in enumerate(todo):
        tmp = folder / f"{ep['rec']['id']}.tmp.npy"
        np.save(tmp, scorer.pair_table(assets, ep))
        os.replace(tmp, folder / f"{ep['rec']['id']}.npy")
        if i % 100 == 0 or i == len(todo) - 1:
            print(f"tables {i + 1}/{len(todo)} new ({len(episodes) - len(todo)} cached), "
                  f"{time.time() - started:.0f}s", flush=True)
    print(f"{len(episodes)} loss tables in {folder}")


if __name__ == "__main__":
    main()
