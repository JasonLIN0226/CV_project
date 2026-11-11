"""Run a historical script against the frozen clean-future predictor source.

    python run_with_snapshot.py <snapshot_src> <script.py> [script args...]

The historical launchers insert experiments/predictor/control/src at the front of
sys.path, which would pick up the older Phase 1 probes.py. Importing the snapshot's
phase1 package first pins every later `from phase1...` import to the frozen source.
"""
import runpy
import sys
from pathlib import Path


def main():
    snapshot, script = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if not (snapshot / "phase1" / "__init__.py").is_file():
        raise SystemExit(f"no phase1 package under {snapshot}")
    sys.path.insert(0, str(snapshot))
    import phase1

    if Path(phase1.__file__).resolve().parent != snapshot / "phase1":
        raise SystemExit("phase1 resolved outside the frozen snapshot")
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
