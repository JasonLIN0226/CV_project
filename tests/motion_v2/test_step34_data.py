"""Pure/small synthetic software tests; no real dataset or classifier fitting.

Run only after the Step3/4 protocol and these exact source hashes are committed.
"""

import importlib.util
from pathlib import Path

MOTION = Path(__file__).resolve().parents[2] / "src/bounded_memory/environments/motion_v2"   # moved from experiments/motion_v2

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location("motion_step34_data_tests", MOTION.joinpath("step34_data.py"))
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def manual_pair():
    """One hand-constructed scene; never invoke the 96-pair data generator."""
    tc, end, query, length = 5, 8, 32, 39
    visible = np.zeros(length, dtype=bool)
    visible[[1, 2, 4, 5, 6, 8]] = True
    visible[query + 2:] = True
    actions = np.zeros((length - 1, 2), dtype=np.float32)
    actions[:, 0] = 0.125
    distractors = np.full((length, 8, 3), 0.5, dtype=np.float64)
    distractors[:, :, 0] = np.linspace(0.1, 0.9, length)[:, None]
    scene = {"seed": data._scene_seed(data.DATA_SEED, 0), "task": "motion", "split": "validation",
             "query": query, "gap": 20, "length": length, "size": 96, "t_on": 1, "t_c": tc,
             "end": end, "L0": 4, "L1": 4, "x_end": 0.5, "speed": 0.009, "x_pre": 0.45,
             "v0": 0.006, "visible": visible, "actions": actions, "label_order": [0, 1],
             "distractors": distractors, "shift": False}
    pair_id = data._pair_id(data.DATA_SEED, "validation", 0)
    ids = [data._digest([pair_id, slot, "opaque-id"])[:24] for slot in range(2)]
    records = [{"id": eid, "pair": pair_id, "sibling": ids[1 - slot], "split": "validation",
                "task": "motion", "seed": scene["seed"], "scene_index": 0, "query": query,
                "gap": 20, "frames": length, "obs": f"observations/validation/{eid}.npz",
                "eval": f"evaluation_only/validation/{eid}.npz", "obs_sha256": "0" * 64,
                "eval_sha256": "0" * 64} for slot, eid in enumerate(ids)]
    episodes = [data._envs().episode(scene, label) for label in (0, 1)]
    return records, [e["observations"] for e in episodes], [e["evaluation_only"] for e in episodes]


def test_manual_pair_obeys_longgap_and_observable_boundaries():
    records, observations, metas = manual_pair()
    detail = data._audit_pair(records, observations, metas)
    assert detail == {"q": 32, "t_on": 1, "t_c": 5, "end": 8, "visible_old": 3, "visible_new": 3}
    assert all(set(o) == {"rgb", "actions", "query"} for o in observations)
    for meta in metas:
        assert not data._online_success(meta, 32, "fifo")
        assert not data._online_success(meta, 32, "keep_first")
        assert data._online_success(meta, 32, "latest_visible_reference")


@pytest.mark.parametrize("split", ["test", "test_long_gap", "conf", "confirmation", "train"])
def test_new_record_split_rejected_before_any_file_read(split, tmp_path):
    records, _, _ = manual_pair()
    record = dict(records[0], split=split)
    with pytest.raises(ValueError, match="split/task"):
        data._read_arrays(tmp_path, record, "validation", "eval")


def test_path_escape_rejected_before_file_read(tmp_path):
    records, _, _ = manual_pair()
    record = dict(records[0], eval="../outside.npz")
    with pytest.raises(ValueError, match="authorized split layout"):
        data._read_arrays(tmp_path, record, "validation", "eval")


def test_extra_hidden_observation_key_is_rejected():
    records, observations, metas = manual_pair()
    observations[0]["t_c"] = metas[0]["t_c"]
    with pytest.raises(ValueError, match="Hidden fields"):
        data._audit_pair(records, observations, metas)


def test_recent_or_old_rgb_difference_is_rejected():
    for frame in (0, 32):
        records, observations, metas = manual_pair()
        observations[1]["rgb"][frame, 0, 0, 0] ^= np.uint8(1)
        with pytest.raises(ValueError, match="segment differs|Recent four"):
            data._audit_pair(records, observations, metas)


def test_missing_post_visibility_fails_without_probe_fitting():
    records, observations, metas = manual_pair()
    metas[0]["visible"][5:8] = False
    with pytest.raises(ValueError, match="visibility constraint"):
        data._audit_pair(records, observations, metas)


def test_action_transition_is_audited():
    records, observations, metas = manual_pair()
    observations[0]["actions"][8, 0] = 0.75
    with pytest.raises(ValueError, match="transition/action"):
        data._audit_pair(records, observations, metas)


def test_random_post_selection_is_sorted_distinct_and_sibling_shared():
    records, _, metas = manual_pair()
    original = metas[0]["visible"].copy()
    a = data.random_post_pair(metas[0]["visible"], 5, 8, records[0]["pair"])
    b = data.random_post_pair(metas[1]["visible"], 5, 8, records[1]["pair"])
    assert a.dtype == np.int64 and a.shape == (2,)
    assert a[0] < a[1] and set(a) <= {5, 6, 8}
    assert np.array_equal(a, b)
    assert np.array_equal(a, data.random_post_pair(original, 5, 8, records[0]["pair"]))
    assert np.array_equal(original, metas[0]["visible"])


def test_random_post_selection_returns_the_only_two_frames():
    visible = np.zeros(14, dtype=bool)
    visible[[0, 6, 9, 13]] = True
    assert data.random_post_pair(visible, 5, 9, "a" * 24).tolist() == [6, 9]


@pytest.mark.parametrize("change", ["one_frame", "end_hidden", "bad_bounds", "bad_pair"])
def test_random_post_selection_rejects_invalid_inputs(change):
    visible = np.zeros(14, dtype=bool)
    visible[[6, 9]] = True
    tc, end, pair_id = 5, 9, "b" * 24
    if change == "one_frame":
        visible[6] = False
    elif change == "end_hidden":
        visible[9] = False
        visible[[5, 7]] = True
    elif change == "bad_bounds":
        tc = end
    else:
        pair_id = "../../test"
    with pytest.raises(ValueError):
        data.random_post_pair(visible, tc, end, pair_id)


def test_probe_features_are_sibling_identical_for_direction_free_controls():
    records, _, metas = manual_pair()
    values, labels, choices = data._probe_features([(records[0]["pair"], records[0]["query"], metas)])
    assert labels.tolist() == [0, 1]
    assert values["end_only"].shape == (2, 4)
    assert values["two_post"].shape == (2, 8)
    assert np.array_equal(values["end_only"][0], values["end_only"][1])
    assert np.array_equal(values["stale_plus_end"][0], values["stale_plus_end"][1])
    assert not np.array_equal(values["two_post"][0], values["two_post"][1])
    assert choices[0]["one_post_non_end"][0] != 8
    assert all(t in {5, 6, 8} for t in choices[0]["two_post"])
    assert len(set(choices[0]["two_post"])) == 2


def test_probe_subsets_do_not_depend_on_sibling_order():
    records, _, metas = manual_pair()
    a = data._probe_features([(records[0]["pair"], 32, metas)])
    b = data._probe_features([(records[0]["pair"], 32, list(reversed(metas)))])
    for key in a[0]:
        assert np.array_equal(a[0][key], b[0][key])
    assert np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_existing_output_is_rejected_without_opening_original_data(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "preserved.txt"
    marker.write_text("preserve exactly", encoding="utf-8")
    with pytest.raises(ValueError, match="Output already exists"):
        data.generate_longgap(output, tmp_path / "missing-original-dataset")
    assert marker.read_text(encoding="utf-8") == "preserve exactly"


def test_registered_scope_is_fixed_validation_only():
    assert data.DATA_CONFIG["pairs"] == {"validation": 96}
    assert data.DATA_CONFIG["seed"] == 20261101
    assert data.DATA_CONFIG["gap"] == [20, 32]
    assert data.LEAKAGE_LIMITS == {"end_only": (0.47, 0.53), "one_post_non_end": (0.0, 0.58),
                                   "stale_plus_end": (0.47, 0.53), "two_post": (0.95, 1.0)}
    with pytest.raises(ValueError, match="scene index"):
        data._scene_longgap(96)
