"""Causal boundary, independent arithmetic and teacher/student separation."""

import numpy as np
import pytest
import torch

from experiments.phase2.memory import MemoryEntry, PolicyContext, StreamingMemory
from experiments.phase2.literature_baselines import (
    CausalSurpriseTracker, SalienceStudent, SalienceTrainingRecord,
    balanced_attention_targets, make_student_inputs, student_accounting,
    student_probabilities, train_salience_student,
)


def entry(time, value, episode="e"):
    return MemoryEntry(episode, time, np.asarray([value, 2 * value], np.float32))


def advance(tracker, stream, value):
    time = stream.observations_seen
    current = entry(time, value)
    record = tracker.observe(current, stream.snapshot.recent)
    snapshot = stream.push(current)
    tracker.retain_only(snapshot.recent + snapshot.retained)
    return record, snapshot


def test_wr_standardized_l1_and_mad_exclude_current_observation():
    tracker = CausalSurpriseTracker(window=2, score_window=4, sigma_floor=.1)
    stream = StreamingMemory(2, 2, "attention", score_callback=tracker.score)
    records = [advance(tracker, stream, value)[0] for value in [0., 2., 4., 6., 10.]]
    # At t2 past dimensions are [0,2] and [0,4], means 1,2, std 1,2.
    assert records[2].raw_score == 3.
    assert records[2].threshold is None
    # Before t4 both valid previous scores are 3: MAD=0, threshold=3.
    # Current [10,20] is not included in Gaussian fit or threshold history.
    assert records[4].raw_score == 5.
    assert records[4].threshold == 3.


def test_wr_peak_confirmation_is_delayed_and_nms_does_not_revise_past():
    tracker = CausalSurpriseTracker(window=2, nms_radius=3)
    stream = StreamingMemory(2, 20, "attention", score_callback=tracker.score)
    for value in [0., 0., 0., 0., 0., 10.]:
        advance(tracker, stream, value)
    assert not next(r for r in tracker.records if r.timestamp == 5).confirmed_event
    advance(tracker, stream, 0.)
    assert next(r for r in tracker.records if r.timestamp == 5).confirmed_event
    for value in [0., 20., 0.]:
        advance(tracker, stream, value)
    # Both peaks exceed their frozen past threshold, but t8 is within radius3.
    assert next(r for r in tracker.records if r.timestamp == 8).raw_score > 1000
    assert [r.timestamp for r in tracker.records if r.confirmed_event] == [5]


def test_wr_event_admission_recency_fallback_and_irreversibility():
    tracker = CausalSurpriseTracker(window=4)
    stream = StreamingMemory(4, 2, "attention", score_callback=tracker.score)
    for value in [0.] * 5 + [10.] + [0.] * 20:
        _, snapshot = advance(tracker, stream, value)
        assert len(tracker.records) <= 6
    assert 5 in [e.timestamp for e in snapshot.retained]
    assert snapshot.retained[-1].timestamp == 21  # the other slot fills by recency
    with pytest.raises(ValueError, match="resurrected"):
        tracker.score((entry(0, 0.),), PolicyContext(25, snapshot.recent))
    with pytest.raises(ValueError, match="resurrected|contiguous"):
        tracker.observe(entry(0, 0.), snapshot.recent)


@pytest.mark.parametrize("score_window", [7, 64])
def test_wr_score_ring_is_bounded_numeric_only_and_fully_accounted(score_window):
    tracker = CausalSurpriseTracker(window=4, score_window=score_window)
    stream = StreamingMemory(4, 2, "attention", score_callback=tracker.score)
    for t in range(100):
        advance(tracker, stream, np.sin(t))
    stats = tracker.accounting()
    assert stats["feature_archive_bytes"] == 0
    assert stats["score_ring_length"] == score_window and stats["score_ring_bytes"] == score_window * 8
    assert stats["resident_scalar_records"] == 6
    assert stats["auxiliary_logical_bytes"] == sum(stats[name] for name in (
        "resident_scalar_record_bytes", "score_ring_bytes", "configuration_and_state_bytes"))
    assert not any(isinstance(value, np.ndarray) for value in vars(tracker).values())
    assert all(not any(isinstance(value, np.ndarray) for value in vars(record).values()) for record in tracker.records)


def test_wr_requires_pruning_and_rejects_nonresident_and_future_window():
    tracker = CausalSurpriseTracker(window=2)
    first = entry(0, 0.)
    tracker.observe(first, ())
    with pytest.raises(ValueError, match="retain_only"):
        tracker.observe(entry(1, 1.), (first,))
    tracker.retain_only((first,))
    with pytest.raises(ValueError, match="strictly past"):
        tracker.observe(entry(1, 1.), (entry(2, 1.),))
    with pytest.raises(ValueError, match="reset"):
        tracker.observe(entry(1, 1., "other"), ())
    tracker.observe(entry(1, 1.), (first,))
    with pytest.raises(ValueError, match="resurrected"):
        tracker.retain_only((first, entry(1, 1.), entry(3, 1.)))
    tracker.retain_only((first, entry(1, 1.)))


def test_wr_prefix_decisions_cannot_depend_on_later_suffix():
    def decisions(suffix):
        tracker = CausalSurpriseTracker(window=2)
        stream = StreamingMemory(2, 2, "attention", score_callback=tracker.score)
        output = []
        for value in [0., 0., 0., 0., 0., 10., 0., 0.] + suffix:
            _, snapshot = advance(tracker, stream, value)
            output.append(([e.timestamp for e in snapshot.retained], tuple(tracker.records)))
        return output
    assert decisions([0., 0.])[:8] == decisions([1000., -1000.])[:8]


def test_student_inputs_exact_causal_features_and_permutation_equivariance():
    candidates = np.array([[1., 2.], [3., 6.]], np.float32)
    recent = np.array([[4., 8.], [5., 10.]], np.float32)
    x = make_student_inputs(candidates, recent, [0, 2], [3, 4], 4)
    assert x.shape == (2, 11)
    np.testing.assert_allclose(x[0], [1, 2, 5, 10, -4, -8, 2, 4, 4.5, 9, np.log(5)])
    xp = make_student_inputs(candidates[::-1], recent, [2, 0], [3, 4], 4)
    np.testing.assert_array_equal(xp, x[::-1])
    assert not x.flags.writeable
    with pytest.raises(ValueError):
        x.setflags(write=True)
    for times in ([0, 3], [0, 5], [0, 0]):
        with pytest.raises(ValueError, match="timestamps"):
            make_student_inputs(candidates, recent, times, [3, 4], 4)
    with pytest.raises(ValueError, match="timestamps"):
        make_student_inputs(candidates, recent, [0, 2], [3, 5], 4)


def test_balanced_teacher_uses_key_relative_block_max_not_query_mean():
    attention = np.array([[.8, .2], [.4, .6], [.7, .3], [.1, .9], [.2, .8]], np.float32)
    p = balanced_attention_targets(attention, [0, 1, 2, 3, 4], [1, 3])
    # key1: max(past)=.8, same=.4, max(future)=.7 ->19/30
    # key3: max(past)=.6, same=.9, max(future)=.8 ->23/30
    np.testing.assert_allclose(p, [19 / 42, 23 / 42], atol=1e-7)
    np.testing.assert_array_equal(balanced_attention_targets(attention[:, ::-1], [0, 1, 2, 3, 4], [3, 1]), p[::-1])
    assert not p.flags.writeable


def test_balanced_teacher_omits_absent_groups_and_requires_future_queries():
    a = np.array([[.8, .2], [.2, .8]], np.float32)
    np.testing.assert_allclose(balanced_attention_targets(a, [2, 3], [0, 1]), [.5, .5])
    with pytest.raises(ValueError, match="future attention"):
        balanced_attention_targets(a, [2, 3], [0, 3])
    with pytest.raises(ValueError, match="normalized reader"):
        balanced_attention_targets(a * 5, [2, 3], [0, 1])


def test_teacher_future_is_after_online_decision_not_merely_after_old_key():
    a = np.array([[.8, .2], [.2, .8]], np.float32)
    with pytest.raises(ValueError, match="after the online decision"):
        balanced_attention_targets(a, [2, 3], [0, 1], decision_time=3)
    np.testing.assert_allclose(balanced_attention_targets(a, [2, 3], [0, 1], decision_time=2), [.5, .5])
    with pytest.raises(ValueError, match="after the online decision"):
        balanced_attention_targets(a, [2, 3], [0, 2], decision_time=1)


def training_records():
    return [SalienceTrainingRecord(np.array([[1., 0.], [0., 1.]], np.float32),
                                   np.array([.9, .1], np.float32), "train-pair-0")]


def test_student_trains_attention_target_only_fixed_final_epoch_and_rng_isolation():
    before = torch.random.get_rng_state().clone()
    model, curve = train_salience_student(training_records(), seed=17, epochs=30, lr=.03,
                                         train_group_ids={"train-pair-0"})
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(curve) == 31 and curve[-1] < curve[0] * .1
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    p = student_probabilities(model, training_records()[0].inputs)
    assert p[0] > .85 and np.isclose(p.sum(), 1.)
    other, other_curve = train_salience_student(training_records(), seed=17, epochs=30, lr=.03)
    assert curve == other_curve
    np.testing.assert_array_equal(p, student_probabilities(other, training_records()[0].inputs))
    stats = student_accounting(model)
    assert stats["student_parameter_bytes"] == sum(p.numel() * p.element_size() for p in model.parameters())
    assert stats["stream_archive_bytes"] == 0


def test_student_rejects_heldout_groups_and_deletion_targets():
    original = training_records()[0]
    with pytest.raises(ValueError, match="fixed train"):
        train_salience_student([original], seed=17, train_group_ids={"different"})
    for split in ("validation", "test"):
        heldout = SalienceTrainingRecord(original.inputs, original.teacher_probability, "held-out", split)
        with pytest.raises(ValueError, match="train-split"):
            train_salience_student([heldout], seed=17)
    with pytest.raises(ValueError, match="never deletion"):
        SalienceTrainingRecord(original.inputs, original.teacher_probability, "group", target_kind="deletion_loss")
    with pytest.raises(ValueError, match="summing to one"):
        SalienceTrainingRecord(original.inputs, np.array([1., 2.], np.float32), "group")


def test_student_logits_are_candidate_permutation_equivariant():
    torch.manual_seed(5)
    model = SalienceStudent(3)
    inputs = np.arange(12, dtype=np.float32).reshape(4, 3)
    p = student_probabilities(model, inputs)
    permutation = [2, 0, 3, 1]
    np.testing.assert_allclose(student_probabilities(model, inputs[permutation]), p[permutation], rtol=1e-6)
