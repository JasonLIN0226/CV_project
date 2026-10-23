"""Independent stream/capacity/causality checks; no research assets or GPU."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from experiments.phase2.memory import BoundedMemory, MemoryEntry, PolicyContext, POLICIES, StreamingMemory


def entry(t, z=(1., 0.), *, episode="e", surprise=None):
    return MemoryEntry(episode, t, np.asarray(z, np.float32), surprise)


def feed(memory, times, **kwargs):
    return [memory.observe(entry(t, **kwargs), PolicyContext(t + 4)) for t in times]


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_is_irreversible_bounded_and_resets(policy):
    callback = (lambda candidates, context: [e.representation[0] for e in candidates]) if policy == "attention" else None
    memory = BoundedMemory(2, policy, seed=3, score_callback=callback)
    previous = set()
    for t in range(20):
        candidates = previous | {("e", t)}
        transition = memory.observe(entry(t, (float(t), 1.)), PolicyContext(t + 4))
        assert len(memory.entries) == min(t + 1, 2)
        assert set(transition.kept_ids) <= candidates
        if t >= 2:
            assert transition.deleted_id in candidates
            assert len(candidates - set(transition.kept_ids)) == 1
        previous = set(transition.kept_ids)
    with pytest.raises(ValueError, match="duplicate|resurrected"):
        memory.observe(entry(0), PolicyContext(30))
    with pytest.raises(ValueError, match="reset"):
        memory.observe(entry(21, episode="new"), PolicyContext(30))
    memory.reset("new")
    assert memory.arrivals_seen == 0 and memory.entries == ()
    memory.observe(entry(0, episode="new"), PolicyContext(4))
    assert memory.arrivals_seen == 1


def test_fifo_exact_stream_and_no_original_array_alias():
    raw = np.array([1., 2.], np.float32)
    first = MemoryEntry("e", 0, raw)
    raw[:] = 99
    np.testing.assert_array_equal(first.representation, [1., 2.])
    assert not first.representation.flags.writeable
    with pytest.raises(ValueError):
        first.representation.setflags(write=True)
    memory = BoundedMemory(2, "fifo")
    memory.observe(first, PolicyContext(4))
    feed(memory, range(1, 10))
    assert [e.timestamp for e in memory.entries] == [8, 9]


def test_reservoir_uniform_inclusion_and_rejection_count():
    # Algorithm R should include each of 20 arrivals with probability 3/20.
    # Repeated K+1-uniform deletion strongly favors late arrivals and fails this.
    trials, length, capacity = 5000, 20, 3
    inclusions = np.zeros(length, np.int64)
    for seed in range(trials):
        memory = BoundedMemory(capacity, "reservoir", seed=seed)
        feed(memory, range(length))
        assert memory.arrivals_seen == length
        for e in memory.entries:
            inclusions[e.timestamp] += 1
    expected = trials * capacity / length
    assert np.max(np.abs(inclusions - expected)) < 5 * np.sqrt(expected * (1 - capacity / length))


def test_reservoir_seed_and_reset_reproduce_decisions():
    memory = BoundedMemory(2, "reservoir", seed=72)
    first = feed(memory, range(30))
    memory.reset()
    assert first == feed(memory, range(30))


def test_temporal_buckets_keep_covered_old_and_recent_bucket():
    memory = BoundedMemory(2, "temporal_buckets", temporal_bucket_width=8)
    memory.observe(entry(0), PolicyContext(1))
    memory.observe(entry(8), PolicyContext(9))
    memory.observe(entry(9), PolicyContext(10))
    assert [e.timestamp for e in memory.entries] == [0, 9]


def test_diversity_retains_distinct_content_and_zero_vectors_are_not_diverse():
    memory = BoundedMemory(2, "visual_diversity")
    for t, z in enumerate([(1., 0.), (-1., 0.), (-1., 0.)]):
        memory.observe(entry(t, z), PolicyContext(t + 4))
    assert [e.timestamp for e in memory.entries] == [0, 2]
    memory.reset()
    for t, z in enumerate([(0., 0.), (0., 0.), (1., 0.)]):
        memory.observe(entry(t, z), PolicyContext(t + 4))
    assert [e.timestamp for e in memory.entries] == [1, 2]


def test_surprise_uses_stored_past_errors_and_explicit_warmup():
    memory = BoundedMemory(2, "prediction_surprise")
    for t, score in enumerate([None, 2., 1., 0., 3.]):
        memory.observe(entry(t, surprise=score), PolicyContext(t + 4))
    assert [e.timestamp for e in memory.entries] == [1, 4]


def test_attention_requires_callback_and_receives_only_causal_boundary():
    with pytest.raises(ValueError, match="explicit.*attention"):
        BoundedMemory(2, "attention")
    calls = []

    def attention(candidates, context):
        calls.append(([e.timestamp for e in candidates], context.current_time))
        assert set(vars(context)) == {"current_time", "recent"}
        assert all(e.timestamp < context.recent[0].timestamp for e in candidates)
        return [e.representation[0] for e in candidates]

    memory = BoundedMemory(2, "attention", score_callback=attention)
    for t, score in enumerate([10., 1., 3.]):
        memory.observe(entry(t, (score, 0.)), PolicyContext(t + 4, (entry(t + 1), entry(t + 4))))
    assert calls == [([0, 1, 2], 6)]
    assert [e.timestamp for e in memory.entries] == [0, 2]


@pytest.mark.parametrize("scores", [[1.], [1., np.nan, 3.], [1., np.inf, 3.]])
def test_invalid_callback_does_not_advance_stream(scores):
    memory = BoundedMemory(2, "attention", score_callback=lambda candidates, context: scores)
    feed(memory, [0, 1])
    with pytest.raises(ValueError, match="finite score"):
        feed(memory, [2])
    assert memory.arrivals_seen == 2
    assert [e.timestamp for e in memory.entries] == [0, 1]


def test_no_future_recent_overlap_or_feature_change():
    memory = BoundedMemory(2, "fifo")
    with pytest.raises(ValueError, match="future"):
        memory.observe(entry(2), PolicyContext(1))
    with pytest.raises(ValueError, match="roles"):
        memory.observe(entry(2), PolicyContext(4, (entry(2), entry(3))))
    memory.observe(entry(2), PolicyContext(4))
    with pytest.raises(ValueError, match="shape/dtype"):
        memory.observe(entry(3, (1., 2., 3.)), PolicyContext(5))
    assert memory.arrivals_seen == 1


def test_payload_accounts_features_metadata_rng_and_recent_separately():
    memory = BoundedMemory(2, "prediction_surprise")
    feed(memory, [0, 1], surprise=0.3)
    result = memory.accounting(PolicyContext(5, (entry(4), entry(5))))
    assert result["feature_bytes"] == 2 * 2 * 4
    assert result["entry_metadata_bytes"] == 2 * (8 + 1 + 8)
    assert result["recent_payload_bytes"] == 2 * (8 + 8 + 1)
    assert result["logical_payload_bytes"] == sum(result[k] for k in ("feature_bytes", "entry_metadata_bytes", "policy_metadata_bytes", "recent_payload_bytes"))
    assert result["includes_python_allocator_overhead"] is False


def test_streaming_recent4_history2_warmup_and_automatic_admission():
    memory = StreamingMemory(4, 2, "fifo")
    assert memory.snapshot.current is None
    for t in range(4):
        snapshot = memory.push(entry(t))
        assert [e.timestamp for e in snapshot.recent] == list(range(t + 1))
        assert snapshot.retained == () and snapshot.transition is None
    first_full = snapshot
    for t in range(4, 15):
        snapshot = memory.push(entry(t))
        assert [e.timestamp for e in snapshot.recent] == list(range(t - 3, t + 1))
        expected = list(range(max(0, t - 5), t - 3))
        assert [e.timestamp for e in snapshot.retained] == expected
        assert snapshot.current is snapshot.recent[-1]
        assert snapshot.current_time == t and snapshot.observations_seen == t + 1
        assert snapshot.transition.arrivals_seen == t - 3
        assert not {e.entry_id for e in snapshot.recent} & {e.entry_id for e in snapshot.retained}
        assert memory.accounting()["resident_observations"] <= 6
    # A returned value stays frozen, but the stream itself owns no snapshot log.
    assert [e.timestamp for e in first_full.recent] == [0, 1, 2, 3]
    with pytest.raises(FrozenInstanceError):
        first_full.current_time = 100
    assert memory.snapshot.transition is None


def test_streaming_callback_sees_only_k_plus_one_and_current_recent():
    calls = []

    def attention(candidates, context):
        assert len(candidates) == 3 and len(context.recent) == 4
        assert max(e.timestamp for e in candidates) < min(e.timestamp for e in context.recent)
        assert context.recent[-1].timestamp == context.current_time
        assert len({e.entry_id for e in candidates + context.recent}) == 7
        calls.append(context.current_time)
        return [e.timestamp for e in candidates]

    memory = StreamingMemory(4, 2, "attention", score_callback=attention)
    for t in range(15):
        snapshot = memory.push(entry(t))
        assert len(snapshot.recent) + len(snapshot.retained) <= 6
    assert calls == list(range(6, 15))
    accounting = memory.accounting()
    assert accounting["resident_observations"] == 6
    assert accounting["max_transient_observations"] == 7
    assert accounting["current_shares_recent_representation"] is True
    assert accounting["logical_payload_bytes"] == sum(accounting[k] for k in (
        "feature_bytes", "entry_metadata_bytes", "policy_metadata_bytes",
        "recent_payload_bytes", "stream_metadata_bytes"))


def test_streaming_surprise_is_preserved_from_observation_to_admission():
    memory = StreamingMemory(4, 2, "prediction_surprise")
    inputs = [entry(t, surprise=score) for t, score in enumerate([10., 1., 5., 0., 0., 0., 0.])]
    for observation in inputs:
        snapshot = memory.push(observation)
    assert [e.timestamp for e in snapshot.retained] == [0, 2]
    assert [e.surprise for e in snapshot.retained] == [10., 5.]
    assert snapshot.retained[0] is inputs[0]
    assert snapshot.transition.deleted_id == ("e", 1)


def test_streaming_reset_clears_recent_history_and_reproduces_reservoir():
    memory = StreamingMemory(4, 2, "reservoir", seed=17)
    first = [memory.push(entry(t)).transition for t in range(30)]
    memory.reset("e")
    assert memory.snapshot.recent == () and memory.snapshot.retained == ()
    assert memory.snapshot.observations_seen == 0
    assert memory.snapshot.current_time is None
    assert first == [memory.push(entry(t)).transition for t in range(30)]
    memory.reset("new")
    assert memory.push(entry(10, episode="new")).current_time == 10


def test_streaming_rejects_bad_arrivals_before_recent_warmup_completes():
    memory = StreamingMemory(4, 2, "fifo")
    memory.push(entry(5))
    for invalid, pattern in [(entry(5), "duplicate"), (entry(7), "contiguous"),
                             (entry(6, episode="other"), "reset"),
                             (entry(6, (1., 2., 3.)), "shape/dtype")]:
        with pytest.raises(ValueError, match=pattern):
            memory.push(invalid)
    assert memory.snapshot.observations_seen == 1
    assert memory.snapshot.current_time == 5
    sparse = StreamingMemory(2, 1, "fifo", require_contiguous=False)
    for t in [0, 4, 10, 20]:
        snapshot = sparse.push(entry(t))
    assert [e.timestamp for e in snapshot.recent] == [10, 20]
    assert [e.timestamp for e in snapshot.retained] == [4]


def test_streaming_invalid_callback_preserves_both_buffers_for_retry():
    state = {"valid": False}

    def attention(candidates, context):
        return [float(e.timestamp) for e in candidates] if state["valid"] else [np.nan] * len(candidates)

    memory = StreamingMemory(4, 2, "attention", score_callback=attention)
    for t in range(6):
        memory.push(entry(t))
    before = memory.snapshot
    with pytest.raises(ValueError, match="finite score"):
        memory.push(entry(6))
    after = memory.snapshot
    assert after.recent == before.recent and after.retained == before.retained
    assert after.observations_seen == before.observations_seen
    state["valid"] = True
    snapshot = memory.push(entry(6))
    assert snapshot.observations_seen == 7 and snapshot.current_time == 6
