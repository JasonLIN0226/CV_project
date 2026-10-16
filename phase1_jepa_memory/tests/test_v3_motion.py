"""Checks for the single motion-policy change and its continuous-speed diagnostic."""
import copy

import numpy as np
import pytest

from phase1.config import DEFAULT
from phase1.inputs import FeatureStore, selected_evidence
from phase1.probes import motion_precision_input, visible_signed_speed, probe_inputs


@pytest.fixture
def motion_store(monkeypatch):
    import phase1.inputs as inputs
    import phase1.probes as probes
    c = copy.deepcopy(DEFAULT)
    c["experiment"] = {"arm": "motion_wide_span", "stage": "development"}
    rec = {"id": "left", "sibling": "right", "task": "motion", "query": 18, "pair": "pair"}
    sibling = dict(rec, id="right", sibling="left")
    q = rec["query"]
    observations = {"actions": np.arange((q+8)*2, dtype=np.float32).reshape(-1, 2)/100}
    metadata = {"evidence": np.array([3, 8], np.int64), "label": np.int64(0),
                "states": np.column_stack([.6-.006*np.arange(q+9), np.full(q+9, .5)]),
                "visible": np.arange(q+9) <= 8}
    z = np.column_stack([np.arange(q+9), np.arange(q+9)**2]).astype(np.float32)
    features = {"left": z, "right": z+np.array([50, 100], np.float32)}
    # Match the recent buffer and second motion observation, as in real pairs.
    features["right"][q-c["data"]["recent"]+1:q+1] = z[q-c["data"]["recent"]+1:q+1]
    features["right"][8] = z[8]
    basis = {"input_mean": np.zeros(2, np.float32), "projection": np.eye(2, dtype=np.float32)}
    store = FeatureStore.__new__(FeatureStore)
    store.c, store.records, store.bases = c, {r["id"]: r for r in [rec, sibling]}, {"motion": basis}
    store.features = lambda r: features[r["id"]]
    store.projected_features = lambda r: features[r["id"]].copy()
    def load(c, r, metadata=False, keys=None):
        source = globals_metadata if metadata else observations
        return {k: np.array(source[k], copy=True) for k in (source if keys is None else keys)}
    globals_metadata = metadata
    monkeypatch.setattr(inputs, "load_episode", load)
    monkeypatch.setattr(probes, "load_episode", load)
    return store, rec, observations, metadata, features, basis


def test_wide_timestamps_change_only_motion_privileged_policy(motion_store):
    store, rec, _, meta, _, _ = motion_store
    assert selected_evidence(store.c, rec, meta["evidence"]).tolist() == [0, 8]
    gate = dict(rec, task="gate")
    assert selected_evidence(store.c, gate, np.array([3])).tolist() == [3]
    base = copy.deepcopy(store.c)
    base.pop("experiment")
    assert selected_evidence(base, rec, meta["evidence"]).tolist() == [3, 8]
    for condition in ["recent_only", "full_history", "random_small_budget"]:
        expected = store.sample(rec, condition)[0]
        store.c = base
        actual = store.sample(rec, condition)[0]
        store.c = dict(base, experiment={"arm": "motion_wide_span"})
        assert all(np.array_equal(actual[k], expected[k]) for k in expected)
    sample, _, info = store.sample(rec, "privileged_small_budget")
    assert info["indices"].tolist() == [0, 8]
    assert sample["age"][-2:].tolist() == [18., 10.]


@pytest.mark.parametrize("evidence", [np.array([3, 15]), np.array([-1, 8]), np.array([8, 3]), np.array([3, 3]), np.array([3., 8.]), np.array([8])])
def test_selected_evidence_rejects_invalid_times(motion_store, evidence):
    store, rec, *_ = motion_store
    with pytest.raises(ValueError):
        selected_evidence(store.c, rec, evidence)


def test_wrong_evidence_swaps_actual_new_images_only(motion_store):
    store, rec, _, _, features, _ = motion_store
    priv, target, info = store.sample(rec, "privileged_small_budget")
    wrong, wrong_target, wrong_info = store.sample(rec, "wrong_evidence")
    R = store.c["data"]["recent"]
    assert np.array_equal(wrong["z"][R:], features["right"][[0, 8]])
    assert np.array_equal(wrong["projected_z"][R:], features["right"][[0, 8]])
    assert not np.array_equal(priv["z"][R], wrong["z"][R])
    for key in ["age", "type", "past_actions", "actions", "current", "projected_current"]:
        assert np.array_equal(priv[key], wrong[key])
    assert np.array_equal(priv["z"][:R], wrong["z"][:R])
    assert np.array_equal(target, wrong_target)
    assert np.array_equal(info["indices"], wrong_info["indices"])


def test_hidden_values_and_future_features_cannot_enter_wide_inputs(motion_store):
    store, rec, _, meta, features, _ = motion_store
    expected = store.sample(rec, "privileged_small_budget")[0]
    meta["label"] = np.int64(999)
    meta["states"][:] = np.nan
    meta["visible"][:] = False
    features["left"][rec["query"]+1:] = -999
    actual = store.sample(rec, "privileged_small_budget")[0]
    assert all(np.array_equal(actual[k], expected[k]) for k in expected)


def test_precision_controls_share_temporal_context_and_actual_selected_features(motion_store):
    store, rec, observations, meta, features, basis = motion_store
    ev = selected_evidence(store.c, rec, meta["evidence"])
    two = motion_precision_input(store, rec, ev, "two_frame", basis)
    first = motion_precision_input(store, rec, ev, "single_start", basis)
    last = motion_precision_input(store, rec, ev, "single_end", basis)
    assert np.array_equal(first[:2], features["left"][0])
    assert np.array_equal(last[:2], features["left"][8])
    assert np.array_equal(two[2:4], (features["left"][8]-features["left"][0])/8)
    assert np.array_equal(two[-9:], first[-9:]) and np.array_equal(two[-9:], last[-9:])
    expected = two.copy()
    observations["actions"][rec["query"]:] = 999
    features["left"][rec["query"]+1:] = 999
    meta["label"] = np.int64(999)
    meta["states"][:] = 999
    assert np.array_equal(motion_precision_input(store, rec, ev, "two_frame", basis), expected)
    actual_probe = probe_inputs(store, rec, "motion_two_frame", basis)
    assert np.array_equal(actual_probe[:2], features["left"][8])
    assert actual_probe[-1] == 8


def test_signed_speed_uses_visible_positions_not_binary_class(motion_store):
    store, rec, _, meta, _, _ = motion_store
    assert visible_signed_speed(store.c, rec, meta)[0] == pytest.approx(-.006)
    meta["label"] = 1
    assert visible_signed_speed(store.c, rec, meta)[0] == pytest.approx(-.006)
    meta["visible"][0] = False
    with pytest.raises(ValueError, match="visible"):
        visible_signed_speed(store.c, rec, meta)


def test_precision_pipeline_fits_train_selects_validation_and_checks_both_singles(tmp_path, monkeypatch):
    import phase1.probes as probes
    from phase1.utils import write_npz, read_json
    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path)
    c["experiment"] = {"arm": "motion_wide_span", "stage": "development"}
    c["evaluation"]["bootstrap"] = 100
    c["probe"]["ridge"] = [.01, .1]
    basis = {"input_mean": np.zeros(1, np.float32), "projection": np.eye(1, dtype=np.float32)}
    write_npz(tmp_path/"predictor/motion/basis.npz", **basis)
    write_npz(tmp_path/"features/normalization.npz", mean=np.zeros(1), std=np.ones(1))
    records, observations, metadata, features = [], {}, {}, {}
    for split in ["train", "validation", "test_id"]:
        for i in range(4):
            end, ref = 6+i%3, .38+.07*i
            q = end+9
            for label in [0, 1]:
                eid = f"{split}_{i}_{label}"
                rec = {"id": eid, "sibling": f"{split}_{i}_{1-label}", "task": "motion",
                       "split": split, "pair": f"{split}_{i}", "query": q}
                records.append(rec)
                t = np.arange(q+9)
                x = ref+(2*label-1)*(.005+i*.0004)*(t-end)
                features[eid] = x[:, None].astype(np.float32)
                metadata[eid] = {"evidence": np.array([3, end], np.int64),
                                 "states": np.column_stack([x, np.full(len(t), .5)]),
                                 "visible": t <= end}
                observations[eid] = {"actions": np.zeros((len(t)-1, 2), np.float32)}
    class Store:
        def __init__(self, base):
            assert "experiment" not in base
            self.c, self.ix, self.manifest = base, {"signature": "fixed-cache"}, {"records": records}
        def features(self, rec):
            return features[rec["id"]]
    def load(base, rec, metadata=False, keys=None):
        source = globals_metadata[rec["id"]] if metadata else observations[rec["id"]]
        return {k: source[k] for k in keys}
    globals_metadata = metadata
    choose = probes.choose_readout
    fitted_sizes = []
    def watched_choose(base, train, val, classification):
        fitted_sizes.append((len(train[0]), len(val[0]), classification))
        return choose(base, train, val, classification)
    monkeypatch.setattr(probes, "FeatureStore", Store)
    monkeypatch.setattr(probes, "fit_basis", lambda *args: basis)
    monkeypatch.setattr(probes, "load_episode", load)
    monkeypatch.setattr(probes, "choose_readout", watched_choose)
    output = probes.motion_precision(c, splits=["validation", "test_id"])
    assert fitted_sizes == [(8, 8, False)]*6
    assert output["fit_split"] == "train" and output["selection_split"] == "validation"
    assert output["cache_signature"] == "fixed-cache"
    assert len(output["results"]) == 12 and len(output["checkpoints"]) == 6
    assert len(output["comparisons"]) == 10
    assert all(r["signed_speed_mse"]["pairs"] == 4 for r in output["results"])
    for split, gate in output["gates"].items():
        wide_single_checks = [r["passed"] for r in output["comparisons"]
                              if r["split"] == split and r["comparison"].startswith("wide_span_two_vs_")]
        assert gate["policies"]["wide_span"]["better_than_each_single"] == all(wide_single_checks)
        assert gate["passed"] == (gate["wide_better_than_current"] and all(wide_single_checks))
    assert read_json(tmp_path/"probes/motion_precision.json")["passed"] == output["passed"]
