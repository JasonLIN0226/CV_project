"""Continuous hidden-CoG PushT: shared present, differing history, diverging future."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/phase2"))
spec = importlib.util.spec_from_file_location(
    "pusht_cog_coherent", Path(__file__).resolve().parents[1]/"experiments/phase2/pusht_cog_coherent.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def family():
    return m.check_family(m.sample_family(2026120000))


def test_probe_lands_every_class_on_the_shared_pose(family):
    result, runs, _ = family
    assert result["valid"] and result["post_probe_keypoint_mismatch_px"] < 1e-6
    turns = result["probe_turn_deg"]
    assert turns["left"] < -10 and turns["right"] > 10 and abs(turns["center"]) < 5
    # Start angles differ by the turn, so the reset frames differ between siblings.
    assert not np.array_equal(runs["left"]["rgb"][0], runs["right"]["rgb"][0])


def test_present_identical_and_future_diverges(family):
    result, runs, plans = family
    c, q = result["converge_frame"], result["query_frame"]
    assert c <= q - m.RECENT_LENGTH + 1 and result["first_test_contact_frame"] > q
    assert result["transit_gap_contacts"] == 0
    for r in runs.values():
        assert np.array_equal(r["rgb"][c:q + 1], runs["center"]["rgb"][c:q + 1])
    shared = result["shared_actions_from_frame"] - 1
    assert all(np.array_equal(p.actions[shared:], plans["center"].actions[shared:]) for p in plans.values())
    assert result["target_min_pair_px"] > 12


def test_no_teleport_after_reset(family):
    _, runs, _ = family
    # Block moves only while touched: frame-to-frame keypoint change is zero without contact.
    for r in runs.values():
        moved = np.abs(np.diff(r["keypoints"], axis=0)).max(axis=(1, 2)) > 1e-9
        assert not (moved & (r["contacts"][1:] == 0)).any()
