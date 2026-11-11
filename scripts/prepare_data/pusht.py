"""Collect a hidden-centre-of-gravity PushT corpus.

    python scripts/prepare_data/pusht.py --version v5 --output-root <dir> --processes 32

    v5  marker: the centre of gravity is drawn as a dot for 3 early frames (the Gate-like testbed)
        -> <output-root>/pusht-cog-v5-marker-20261109, generator src/bounded_memory/environments/pusht/prepare_pusht_cog_marker_data.py
    v4  natural: the cue is the block's response to a shared probe push (the harder follow-up)
        -> <output-root>/pusht-cog-v4-20261107, generator src/bounded_memory/environments/pusht/prepare_pusht_cog_v4_data.py

Runs the generators in src/bounded_memory/environments/pusht, which were moved there unchanged.
Needs the PushT simulator packages (gym-pusht, pymunk, pygame). v4 also needs its probe library;
pass extra generator arguments after `--`.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {"v5": ("src/bounded_memory/environments/pusht/prepare_pusht_cog_marker_data.py", "pusht-cog-v5-marker-20261109"),
            "v4": ("src/bounded_memory/environments/pusht/prepare_pusht_cog_v4_data.py", "pusht-cog-v4-20261107")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, choices=sorted(VERSIONS))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args, extra = ap.parse_known_args()
    script, corpus = VERSIONS[args.version]
    try:
        import gym_pusht  # noqa: F401
    except ImportError:
        sys.exit("the PushT simulator packages are not installed in this environment (gym-pusht, pymunk, pygame)")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str((ROOT / script).parent))        # the PushT modules import each other by file name
    sys.argv = [script, "--output", str(args.output_root / corpus), "--processes", str(args.processes), *[e for e in extra if e != "--"]]
    runpy.run_path(str(ROOT / script), run_name="__main__")


if __name__ == "__main__":
    main()
