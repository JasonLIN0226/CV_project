"""Collect a hidden-CoG PushT corpus, using the repository's v4 or v5 simulator."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

PHASE2 = Path(__file__).resolve().parents[2] / "experiments/phase2"
SPLITS = ("train", "validation", "evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v4", "v5"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--splits", nargs="+", choices=SPLITS)
    args, extra = parser.parse_known_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if args.version == "v4":
        if args.splits:
            parser.error("--splits is supported for v5 only")
        output = args.output_root / "pusht-cog-v4-20261107"
        subprocess.run([sys.executable, str(PHASE2 / "prepare_pusht_cog_v4_data.py"),
                        "--output", str(output), "--processes", str(args.processes),
                        *[arg for arg in extra if arg != "--"]], check=True)
        return
    if extra:
        parser.error(f"unknown v5 arguments: {extra}")
    sys.path.insert(0, str(PHASE2))
    from pusht_marker_collection import run
    output = args.output_root / "pusht-cog-v5-marker-20261109"
    run(output, args.processes, args.splits or SPLITS)
    print(output / "preparation_manifest.json")


if __name__ == "__main__":
    main()
