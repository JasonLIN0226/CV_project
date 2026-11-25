"""Collect a hidden-centre-of-gravity PushT corpus.

    python scripts/prepare_data/pusht.py --version v5c --output-root <dir> --processes 32

    v5   marker: the centre of gravity is drawn as a 2-px dot for 3 early frames
         -> <output-root>/pusht-cog-v5-marker-20261109
    v5b  the same with a 4-px dot and 960 training families
         -> <output-root>/pusht-cog-v5b-marker-r4-t960-20261120
    v5c  v5b plus 1-2 distractor dots in every history frame (the main testbed)
         -> <output-root>/pusht-cog-v5c-marker-r4-d2-t960-20261123
    v4   natural: the cue is the block's response to a shared probe push (the harder follow-up)
         -> <output-root>/pusht-cog-v4-20261107

Runs the generators in src/bounded_memory/environments/pusht (versions and arguments in corpora.py):
v5 and v4 with the generators moved there unchanged, v5b and v5c with their successor
prepare_pusht_cog_marker_v5c_data.py. Needs the PushT simulator packages (gym-pusht, pymunk,
pygame). v4 also needs its probe library; pass extra generator arguments after `--`.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUSHT = ROOT / "src/bounded_memory/environments/pusht"
sys.path.insert(0, str(PUSHT))
from corpora import VERSIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, choices=sorted(VERSIONS))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args, extra = ap.parse_known_args()
    version = VERSIONS[args.version]
    script = PUSHT / version["generator"]
    try:
        import gym_pusht  # noqa: F401
    except ImportError:
        sys.exit("the PushT simulator packages are not installed in this environment (gym-pusht, pymunk, pygame)")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.argv = [str(script), "--output", str(args.output_root / version["corpus"]), "--processes", str(args.processes),
                *version["args"], *[e for e in extra if e != "--"]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
