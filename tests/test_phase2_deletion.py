"""Independent arithmetic/causality fixtures, not a real deletion training set."""
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from experiments.phase2.deletion import (
    CausalContext, CausalParent, EvaluatorProvenance, OfflineFuture,
    generate_deletion_supervision,
)
from experiments.phase2.memory import MemoryEntry


PROVENANCE = EvaluatorProvenance("1" * 64, "test fixture, no trained model")


def entry(timestamp, values=None, *, episode="episode"):
    return MemoryEntry(episode, timestamp, np.array(values if values is not None else [timestamp + 1., 2.], np.float32))


def parent(k=2):
    return CausalParent(k, tuple(entry(t) for t in range(k + 1)), CausalContext(11, (entry(10), entry(11))))


def future(*, targets=None, available=(True, True), actions=None):
    return OfflineFuture((1, 3), np.array([[1., 2.], [4., 8.]] if targets is None else targets, np.float32),
                         np.ones((3, 2), np.float32) if actions is None else actions,
                         np.array(available, bool))


@pytest.mark.parametrize("k", [2, 3])
def test_all_deletions_equal_capacity_and_independent_mse(k):
    p, f, calls = parent(k), future(), []

    def evaluator(kept, context, horizon, actions):
        calls.append((tuple(e.entry_id for e in kept), context, horizon, actions.copy()))
        return np.array([sum(e.timestamp + 1 for e in kept) + horizon, 2. * k], np.float32)

    result = generate_deletion_supervision(p, f, evaluator, provenance=PROVENANCE)
    assert len(calls) == (k + 1) * 2
    assert result.labels.feature_mse.shape == (k + 1, 2)
    all_ids = set(e.entry_id for e in p.candidates)
    total_first_feature = (k + 1) * (k + 2) / 2
    for deleted in range(k + 1):
        kept = set(result.labels.kept_ids[deleted])
        assert len(kept) == k and all_ids - kept == {result.labels.deleted_ids[deleted]}
        prediction1 = [total_first_feature - (deleted + 1) + 1, 2 * k]
        prediction3 = [total_first_feature - (deleted + 1) + 3, 2 * k]
        expected = [((prediction1[0] - 1) ** 2 + (prediction1[1] - 2) ** 2) / 2,
                    ((prediction3[0] - 4) ** 2 + (prediction3[1] - 8) ** 2) / 2]
        np.testing.assert_allclose(result.labels.feature_mse[deleted], expected, rtol=0, atol=0)
    for _, context, horizon, actions in calls:
        assert context is p.context
        np.testing.assert_array_equal(actions, f.actions[:horizon])


def test_scorer_has_only_causal_features_and_future_changes_do_not_change_inputs():
    p = parent()
    evaluator = lambda kept, context, horizon, actions: kept[0].representation
    a = generate_deletion_supervision(p, future(), evaluator, provenance=PROVENANCE)
    b = generate_deletion_supervision(p, future(targets=[[99., 99.], [88., 88.]], actions=np.full((3, 2), 7., np.float32)), evaluator, provenance=PROVENANCE)
    assert set(vars(a.scorer_inputs)) == {"capacity", "current_time", "candidate_ids", "candidate_timestamps",
                                         "candidate_representations", "recent_ids", "recent_timestamps", "recent_representations"}
    for field in fields(a.scorer_inputs):
        left, right = getattr(a.scorer_inputs, field.name), getattr(b.scorer_inputs, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
            assert not np.shares_memory(left, b.scorer_inputs.candidate_representations)
        else:
            assert left == right
    assert not np.array_equal(a.labels.feature_mse, b.labels.feature_mse)
    assert not np.shares_memory(a.scorer_inputs.candidate_representations, p.candidates[0].representation)
    assert not np.shares_memory(a.scorer_inputs.candidate_representations, a.labels.feature_mse)


def test_missing_horizon_is_masked_not_zero_and_unevaluated():
    calls = []
    f = future(targets=[[1., 2.], [np.nan, np.nan]], available=(True, False), actions=np.ones((1, 2), np.float32))

    def evaluator(kept, context, horizon, prefix):
        calls.append(horizon)
        return np.array([1., 2.], np.float64)

    result = generate_deletion_supervision(parent(), f, evaluator, provenance=PROVENANCE)
    assert calls == [1, 1, 1]
    np.testing.assert_array_equal(result.labels.horizon_mask, [True, False])
    assert (result.labels.feature_mse[:, 0] == 0).all()
    assert np.isnan(result.labels.feature_mse[:, 1]).all()


def test_all_missing_has_no_model_calls_and_no_usable_labels():
    f = future(targets=[[np.nan] * 2] * 2, available=(False, False), actions=np.empty((0, 2), np.float32))

    def forbidden(*args):
        raise AssertionError("no valid targets, no predictor call")

    result = generate_deletion_supervision(parent(), f, forbidden, provenance=PROVENANCE)
    assert not result.labels.horizon_mask.any() and np.isnan(result.labels.feature_mse).all()


def test_inputs_and_action_prefix_are_immutable_and_disjoint():
    original = np.array([4., 5.], np.float32)
    initial = MemoryEntry("episode", 0, original, surprise=9.)
    p = CausalParent(2, (initial, entry(1), entry(2)), CausalContext(11, (entry(10), entry(11))))
    targets, actions = np.ones((2, 2), np.float32), np.arange(6, dtype=np.float32).reshape(3, 2)
    f = future(targets=targets, actions=actions)
    original[:] = 100
    targets[:] = 200
    actions[:] = 300
    snapshots = [e.representation.copy() for e in p.candidates]
    assert all(e.surprise is None for e in p.candidates)
    with pytest.raises(FrozenInstanceError):
        p.capacity = 5

    def evaluator(kept, context, horizon, prefix):
        assert prefix.shape == (horizon, 2)
        assert prefix.nbytes == horizon * 2 * np.dtype("float32").itemsize
        # Underlying storage has only the prefix, not the future suffix.
        assert prefix.base.nbytes == prefix.nbytes
        for arr in [prefix, kept[0].representation, context.recent[0].representation]:
            with pytest.raises(ValueError):
                arr.setflags(write=True)
        np.testing.assert_array_equal(prefix, np.arange(6, dtype=np.float32).reshape(3, 2)[:horizon])
        return np.zeros(2, np.float32)

    a = generate_deletion_supervision(p, f, evaluator, provenance=PROVENANCE)
    b = generate_deletion_supervision(p, f, evaluator, provenance=PROVENANCE)
    for item, snapshot in zip(p.candidates, snapshots):
        np.testing.assert_array_equal(item.representation, snapshot)
    np.testing.assert_array_equal(a.labels.feature_mse, b.labels.feature_mse)
    for arr in [a.labels.feature_mse, a.labels.horizon_mask, a.scorer_inputs.candidate_representations,
                a.scorer_inputs.candidate_timestamps, f.targets, f.actions]:
        with pytest.raises(ValueError):
            arr.setflags(write=True)


@pytest.mark.parametrize("candidates, capacity, message", [
    ((entry(0), entry(0), entry(2)), 2, "distinct"),
    ((entry(0), entry(1)), 2, "K\\+1"),
    ((entry(0), entry(1), entry(10)), 2, "precede"),
    ((entry(0), entry(1), entry(20)), 2, "precede"),
    ((entry(0), entry(1), entry(2, episode="other")), 2, "one episode"),
    ((entry(0), entry(1), entry(2, [1., 2., 3.])), 2, "shape/dtype"),
])
def test_invalid_parent_rejected(candidates, capacity, message):
    with pytest.raises(ValueError, match=message):
        CausalParent(capacity, candidates, CausalContext(11, (entry(10), entry(11))))


def test_future_recent_and_extra_oracle_fields_rejected():
    with pytest.raises(ValueError, match="end at current_time"):
        CausalContext(11, (entry(10), entry(12)))
    with pytest.raises(TypeError):
        CausalParent(2, parent().candidates, parent().context, evidence_roles=[0, 1, 0])


@pytest.mark.parametrize("invalid", [
    {"targets": [[np.inf, 2.], [4., 8.]]},
    {"actions": np.ones((2, 2), np.float32)},
    {"actions": np.full((3, 2), np.nan, np.float32)},
])
def test_bad_future_rejected(invalid):
    with pytest.raises(ValueError):
        future(**invalid)


def test_mask_and_dimensions_are_checked():
    with pytest.raises(ValueError, match="boolean"):
        OfflineFuture((1, 3), np.zeros((2, 2), np.float32), np.zeros((3, 2), np.float32), np.array([1, 0]))
    with pytest.raises(ValueError, match="distinct"):
        OfflineFuture((1, 1), np.zeros((2, 2), np.float32), np.zeros((3, 2), np.float32), np.array([True, True]))
    with pytest.raises(ValueError, match="dimension"):
        generate_deletion_supervision(parent(), future(targets=[[1., 2., 3.], [1., 2., 3.]]), lambda *args: None, provenance=PROVENANCE)


@pytest.mark.parametrize("prediction", [np.array([np.nan, 1.]), np.ones(3), np.ones((1, 2)), np.array([1, 2]), np.array([1e308, 1e308])])
def test_invalid_prediction_or_overflow_cannot_be_clean_label(prediction):
    with pytest.raises(ValueError):
        generate_deletion_supervision(parent(), future(), lambda *args: prediction, provenance=PROVENANCE)


def test_provenance_requires_checkpoint_digest_but_does_not_claim_model_inspection():
    with pytest.raises(ValueError, match="SHA-256"):
        EvaluatorProvenance("checkpoint.pt", "test")
    with pytest.raises(ValueError, match="nonempty"):
        EvaluatorProvenance("1" * 64, " ")
    with pytest.raises(ValueError, match="provenance"):
        generate_deletion_supervision(parent(), future(), lambda *args: np.zeros(2), provenance=None)
    result = generate_deletion_supervision(parent(), future(), lambda *args: np.zeros(2), provenance=PROVENANCE)
    assert result.provenance == PROVENANCE
    assert not hasattr(result, "model_freeze_verified")
