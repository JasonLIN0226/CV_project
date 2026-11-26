"""Generate and encode fresh Motion v2 scenes for training our scorer (environments/motion_v2/fresh.py).

    BM_MOTION_ROOT=/scratch/$USER python scripts/prepare_data/motion_v2_fresh.py --config configs/motion_v2/frozen.yaml \\
        --device cuda

Writes the scene pairs the config's `fresh` block asks for (training states, dev, holdout) under fresh.root:
dataset/ (observations, evaluation-only arrays, manifest.json) and features/ (raw I-JEPA features with the
frozen step-2 recipe, index.json). Generated files are never rewritten, so a rerun only adds what is missing.
The frozen assets are read for the encoder settings; validation is not read.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import fresh, frozen  # noqa: E402
from bounded_memory.methods.deletion_cost import motion_v2 as scorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    frozen.verify(paths, ["ordinary"])
    assets = frozen.load(paths, args.device)
    pairs = max(r.stop for r in scorer.fresh_ranges(config["fresh"]).values())
    records = fresh.generate(config["fresh"]["root"], pairs, log=print)
    fresh.encode(assets, config["fresh"]["root"], records, args.device, log=print)
    print(json.dumps({"root": config["fresh"]["root"], "split": fresh.FRESH_SPLIT, "pairs": pairs,
                      "episodes": len(records)}, indent=1))


if __name__ == "__main__":
    main()
