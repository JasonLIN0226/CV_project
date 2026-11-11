"""Natural hidden-CoG PushT (v4): class-neutral start, contact-only motion, sound counterfactuals."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/bounded_memory/environments/pusht"))
spec = importlib.util.spec_from_file_location("pusht_cog_natural", ROOT / "src/bounded_memory/environments/pusht/pusht_cog_natural.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
LIBRARY = ROOT / "build/pusht_cog_probe_library_v3.json"


@pytest.fixture(scope="module")
def family():
    if not LIBRARY.exists():
        pytest.skip("probe library not built (pusht_cog_matched.build_library)")
    pool = m.probe_pool(json.loads(LIBRARY.read_text()))
    for seed in range(2026149000, 2026149050):  # test-only seeds
        result, runs, actions, phase = m.check_family(m.sample_family(seed, pool))
        if result["valid"]:
            return result, runs, actions, phase
    pytest.fail("no valid family among 50 test seeds")


def test_start_and_probe_are_class_neutral(family):
    result, runs, actions, _ = family
    assert result["reset_frames_identical"] and result["pre_contact_frames_identical"]
    probe_end = result["probe_end_frame"]
    assert all(np.array_equal(a[:probe_end], actions["center"][:probe_end]) for a in actions.values())


def test_block_moves_only_under_contact(family):
    _, runs, _, _ = family
    for r in runs.values():
        moved = np.abs(np.diff(r["keypoints"], axis=0)).max(axis=(1, 2)) > 1e-9
        assert not (moved & (r["contacts"][1:] == 0)).any()


def test_counterfactual_replay_reproduces_the_run_and_diverges(family):
    result, _, _, _ = family
    for entry in result["per_class"].values():
        assert entry["counterfactual_true_matches_run_px"] < 1e-6
        assert entry["counterfactual_min_pair_px"] > 5
