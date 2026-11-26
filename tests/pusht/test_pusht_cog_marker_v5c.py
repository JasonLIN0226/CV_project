"""Marker hidden-CoG PushT (v5/v5b/v5c successor module): the dot sits on the block, siblings differ
only where they should, the radius sets the dot size, and distractors are shared and off the block."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/bounded_memory/environments/pusht"))
spec = importlib.util.spec_from_file_location("pusht_cog_marker_v5c",
                                              ROOT / "src/bounded_memory/environments/pusht/pusht_cog_marker_v5c.py")
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


def test_marker_radius_sets_dot_size(family, monkeypatch):
    result, runs, _, _ = family
    xy = runs["center"]["marker_px"][result["marker_frames"][0]]
    blank = np.zeros((96, 96, 3), np.uint8)
    area = {}
    for radius in (2, 4):
        monkeypatch.setattr(m, "MARKER_RADIUS_PX", radius)
        area[radius] = int((m.draw_marker(blank, xy) == np.array(m.MARKER_COLOR)).all(-1).sum())
    assert area[4] > 3 * area[2]


def test_distractors_shared_by_siblings_and_off_the_block(monkeypatch):
    monkeypatch.setattr(m, "MARKER_RADIUS_PX", 4)
    monkeypatch.setattr(m, "DISTRACTOR_DOTS", 2)
    for seed in range(2026159000, 2026159030):
        result, runs, _, phase = m.check_family(m.sample_family(seed))
        if result["valid"]:
            break
    # Siblings still differ only in the marker frames before the test contact.
    assert result["frames_differing_before_test_contact"] == result["marker_frames"]
    history_end = result["query_frame"] - m.RECENT_LENGTH
    d = runs["center"]["distractors"]
    assert all(1 <= len(d[t]) <= 2 for t in range(history_end + 1))
    assert all(len(d[t]) == 0 for t in range(history_end + 1, len(d)))
    assert all(runs[c]["distractors"] == d for c in runs)
    # Distractors sit on background: the block's pixels are untouched by them.
    for t in range(history_end + 1):
        for x, y, _ in d[t]:
            assert np.abs(runs["center"]["keypoints"][t].mean(0) / 512 * 96 - [x, y]).max() > 2
