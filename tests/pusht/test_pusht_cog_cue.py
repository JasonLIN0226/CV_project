"""Hidden-CoG PushT cue: sibling identity, cue visibility and future divergence."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
spec = importlib.util.spec_from_file_location(
    "pusht_cog_cue", Path(__file__).resolve().parents[2]/"src/bounded_memory/environments/pusht/pusht_cog_cue.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def family():
    return m.check_family(m.sample_family(2026110000))


def test_plan_is_class_independent():
    a = m.sample_family(7)
    assert m.sample_family(7) == a
    assert np.array_equal(m.build_plan(a).actions, m.build_plan(a).actions)


def test_siblings_identical_outside_cue_and_future(family):
    result, runs, plan = family
    assert result["valid"] and result["gap_contacts"] == 0
    cut, q = result["cut_frame"], result["query_frame"]
    for r in runs.values():
        # Reset frame and the whole recent window before the query match exactly.
        assert np.array_equal(r["rgb"][0], runs["center"]["rgb"][0])
        assert np.array_equal(r["rgb"][cut:q + 1], runs["center"]["rgb"][cut:q + 1])


def test_probe_reveals_class_and_test_diverges(family):
    result, runs, _ = family
    turn = result["probe_turn_deg"]
    assert turn["left"] < -10 and turn["right"] > 10 and abs(turn["center"]) < 5
    assert min(result["probe_pair_keypoint_px"].values()) > 20
    assert result["target_frame"] == result["query_frame"] + m.HORIZON
    assert result["first_test_contact_frame"] > result["query_frame"]
    assert result["target_min_pair_px"] > 12


def test_place_leaves_no_residual_motion():
    s = m.sample_family(3)
    plan = m.build_plan(s)
    for cog in m.COG_CLASSES.values():
        env = m.make_env(cog)
        env.reset(seed=s.seed)
        m.place(env, plan.probe_start, s.probe_center, s.probe_angle)
        before = m.keypoints(env)
        env.space.step(env.dt)
        assert np.abs(m.keypoints(env) - before).max() < 1e-9
        env.close()
