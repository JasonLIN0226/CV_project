"""Diagnostic controls are evaluator-only; no real model/data downloads."""
import copy

import numpy as np
import pytest
import torch

from experiments.phase2.diagnose_history import (
    DEFAULT_MODES, compare_initial_replay, control_sample, paired_summary,
    validate_matched_sibling, _predict,
)


@pytest.fixture
def case():
    config = {"data": {"recent": 4, "budget": 2, "horizons": [3, 6]}}
    rec = {"id": "a", "sibling": "b", "pair": "p", "split": "validation", "task": "gate", "query": 12}
    sibling = {**rec, "id": "b", "sibling": "a"}
    rng = np.random.default_rng(20261020)
    features = rng.normal(size=(19, 5)).astype(np.float32)
    other = features.copy()
    other[:9] += 10
    other[13:] -= 3
    actions = rng.normal(size=(18, 2)).astype(np.float32)
    basis = {"input_mean": np.zeros(5, np.float32), "projection": np.eye(5, dtype=np.float32)}
    return config, rec, sibling, features, other, actions, basis


def test_wrong_evidence_changes_only_selected_history(case):
    c, r, sibling, z, other, actions, basis = case
    validate_matched_sibling(r, sibling, z, other, actions, actions)
    correct, target, info = control_sample(c, r, z, actions, np.array([2]), basis, "privileged_small_budget")
    swapped, swapped_target, swap_info = control_sample(c, r, z, actions, np.array([2]), basis, "wrong_evidence", sibling_features=other)
    assert info["history_indices"] == [2, 8]
    assert swap_info["replaced_timestamps"] == [2]
    assert np.array_equal(correct["z"][:4], swapped["z"][:4])
    assert np.array_equal(correct["z"][5], swapped["z"][5])
    assert np.array_equal(swapped["z"][4], other[2])
    assert np.array_equal(target, swapped_target)
    for key in ("actions", "past_actions", "current", "age", "type", "projected_current"):
        assert np.array_equal(correct[key], swapped[key])
    assert set(correct) == {"z", "age", "type", "past_actions", "actions", "current", "projected_z", "projected_current"}


def test_full_history_swap_preserves_recent_and_never_reads_future_sibling(case):
    c, r, _, z, other, actions, basis = case
    full, target, info = control_sample(c, r, z, actions, np.array([2]), basis, "full_history")
    swap, _, swap_info = control_sample(c, r, z, actions, np.array([2]), basis, "sibling_full_history", sibling_features=other)
    assert info["exceeds_training_memory_budget"] and info["history_indices"] == list(range(9))
    assert swap_info["replaced_timestamps"] == list(range(9))
    assert np.array_equal(swap["z"][:4], full["z"][:4])
    assert np.array_equal(swap["z"][4:], other[:9])
    altered = other.copy()
    altered[9:] = 1e9
    again, again_target, _ = control_sample(c, r, z, actions, np.array([2]), basis, "sibling_full_history", sibling_features=altered)
    assert np.array_equal(swap["z"], again["z"]) and np.array_equal(target, again_target)


def test_no_history_and_prediction_target_separation(case):
    c, r, _, z, _, actions, basis = case
    first, target, info = control_sample(c, r, z, actions, np.array([2]), basis, "recent_only")
    altered = z.copy()
    altered[13:] += 1000
    second, target_second, _ = control_sample(c, r, altered, actions, np.array([2]), basis, "recent_only")
    assert info["history_count"] == 0 and len(first["z"]) == 4
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert not np.array_equal(target, target_second)


@pytest.mark.parametrize("change", ["split", "pair", "query", "recent", "action"])
def test_reject_unmatched_intervention(case, change):
    _, rec, sibling, z, other, actions, _ = case
    sibling, other, other_actions = copy.deepcopy(sibling), other.copy(), actions.copy()
    if change in ("split", "pair"):
        sibling[change] = "different"
    elif change == "query":
        sibling[change] += 1
    elif change == "recent":
        other[12, 0] += 1
    else:
        other_actions[16, 0] += 1
    with pytest.raises(ValueError, match="requires"):
        validate_matched_sibling(rec, sibling, z, other, actions, other_actions)


def test_reject_test_split_and_future_privileged_timestamp(case):
    c, r, _, z, _, actions, basis = case
    with pytest.raises(ValueError, match="test splits"):
        control_sample(c, {**r, "split": "test_id"}, z, actions, np.array([2]), basis, "full_history")
    with pytest.raises(ValueError, match="precede recent"):
        control_sample(c, r, z, actions, np.array([9]), basis, "privileged_small_budget")


def test_paired_summary_uses_pair_not_individual_std():
    rows = [{"pair": "p1", "episode_id": "a", "mse": 0.}, {"pair": "p1", "episode_id": "b", "mse": 2.},
            {"pair": "p2", "episode_id": "c", "mse": 2.}, {"pair": "p2", "episode_id": "d", "mse": 4.}]
    summary = paired_summary(rows)
    assert summary["pairs"] == 2 and summary["episodes"] == 4
    assert summary["mean"] == 2 and summary["pair_std_ddof1"] == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="complete"):
        paired_summary(rows[:-1])


def test_initial_reconstruction_cannot_claim_exact_when_numeric_mismatch():
    logged = dict.fromkeys(DEFAULT_MODES, .25)
    actual = {**logged, DEFAULT_MODES[0]: .25000001}
    result = compare_initial_replay(actual, logged)
    assert result["matches_within_tolerance"] and not result["bit_exact_scores"]
    actual[DEFAULT_MODES[0]] = .3
    assert compare_initial_replay(actual, logged)["reconstruction_status"] == "UNVERIFIED_INITIALIZATION"


def test_predict_rejects_training_or_trainable_model(case):
    model = torch.nn.Linear(5, 5)
    with pytest.raises(ValueError, match="frozen eval"):
        _predict(model, [])
    model.eval()
    with pytest.raises(ValueError, match="frozen eval"):
        _predict(model, [])
