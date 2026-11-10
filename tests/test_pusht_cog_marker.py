"""Marker hidden-CoG PushT (v5): the dot sits on the block, and siblings differ only where they should."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/phase2"))
spec = importlib.util.spec_from_file_location("pusht_cog_marker", ROOT / "experiments/phase2/pusht_cog_marker.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
BLOCK_RGB = np.array([119, 136, 153])


@pytest.fixture(scope="module")
def family():
    for seed in range(2026159000, 2026159030):  # test-only seeds
        out = m.check_family(m.sample_family(seed))
        if out[0]["valid"]:
            return out
    pytest.fail("no valid family among 30 test seeds")


def test_siblings_share_actions_and_differ_only_in_marker_frames(family):
    result, runs, actions, phase = family
    assert result["pre_test_contacts"] == 0
    assert result["frames_differing_before_test_contact"] == result["marker_frames"]
    assert result["marker_frames"][-1] < result["query_frame"] - m.RECENT_LENGTH + 1


def test_marker_lands_on_the_block(family):
    result, runs, _, _ = family
    t = result["marker_frames"][0]
    for r in runs.values():
        x, y = (int(round(v)) for v in r["marker_px"][t])
        # The unmarked frame just before the marker shows the block under the dot position.
        before = r["rgb"][t - 1][y, x].astype(int)
        assert np.abs(before - BLOCK_RGB).max() < 40
        assert tuple(r["rgb"][t][y, x]) == m.MARKER_COLOR


def test_future_diverges_after_test_contact(family):
    result, _, _, _ = family
    assert result["first_test_contact_frame"] > result["query_frame"]
    assert result["target_min_pair_px"] > 12
