"""The pre-refactor code, for old-versus-new checks.

experiments/ was removed from the working tree after the refactor. The migration checks compare
the package with the code at the base commit. BM_LEGACY_TREE names a folder that holds that
commit's experiments/ (and results/); without it, experiments/ is extracted from git into a
temporary folder on first use:

    git archive fe69074 experiments | tar -x -C <folder>; ln -s <repo>/results <folder>/results
"""
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "fe69074"


def path():
    """Folder holding experiments/ as of the base commit, or None when it cannot be provided."""
    given = os.environ.get("BM_LEGACY_TREE")
    if given:
        return Path(given).resolve()
    target = Path(tempfile.gettempdir()) / f"bounded_memory_legacy_{BASE}"
    if not (target / "experiments").is_dir():
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", BASE, "experiments"], capture_output=True)
        if archive.returncode != 0:
            return None
        staging = Path(tempfile.mkdtemp(prefix="legacy_", dir=target.parent))
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(staging)
        (staging / "results").symlink_to(ROOT / "results")       # the old code reads a few files under results/, which did not change
        try:
            staging.rename(target)
        except OSError:                                          # another process extracted it first
            pass
    return target


def predictor_snapshot(tree):
    """The clean-future predictor source the old Gate code imports (BM_LEGACY_SNAPSHOT overrides)."""
    given = os.environ.get("BM_LEGACY_SNAPSHOT")
    if given:
        return Path(given).resolve()
    out = tree / "clean_future_control"
    if not (out / "src/phase1").is_dir():
        subprocess.run([sys.executable, str(tree / "experiments/gate_clean_future/frozen_control/materialize_snapshot.py"),
                        "--output", str(out)], check=True, capture_output=True)
    return out / "src"


def use(tree):
    """Put the old code first on the import path and check that `experiments` resolves inside it."""
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    import experiments.memory_baselines as probe
    assert Path(list(probe.__path__)[0]).resolve().is_relative_to(tree.resolve()), "experiments resolved outside the legacy tree"
