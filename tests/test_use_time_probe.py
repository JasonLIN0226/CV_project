"""Counterfactual continuation semantics, not model-performance assertions."""
import numpy as np
import pytest

from experiments.phase2.memory import MemoryEntry, StreamingMemory
from experiments.phase2.run_use_time_probe import AttentionContinuation


def test_deleted_cue_cannot_return_even_when_later_callback_wants_it():
    z = np.arange(20, dtype=np.float32)[:, None]
    inspected = []

    def prefer_cue(candidates, context):
        times = [e.timestamp for e in candidates]
        inspected.append((context.current_time, times))
        assert all(t < context.current_time-3 for t in times)
        assert [e.timestamp for e in context.recent] == list(range(context.current_time-3, context.current_time+1))
        return [1000. if t == 5 else -float(t) for t in times]

    teacher = AttentionContinuation("fixture", z, 17, prefer_cue)
    kept_if_deleted = teacher.final_kept(9, (0, 1))
    assert 5 not in kept_if_deleted
    assert all(5 not in times for _, times in inspected)
    assert 5 in teacher.final_kept(9, (0, 5))
    before = teacher.calls
    assert teacher.final_kept(9, (0, 1)) == kept_if_deleted
    assert teacher.calls == before


def test_continuation_matches_uninterrupted_stream_with_exact_ties():
    z = np.arange(24, dtype=np.float32)[:, None]

    def salience(candidates, context):
        # Repeated scores exercise the oldest-candidate tie convention.
        return [float(e.timestamp % 3) for e in candidates]

    stream = StreamingMemory(4, 2, "attention", score_callback=salience)
    parents = []
    for now in range(18):
        snapshot = stream.push(MemoryEntry("fixture", now, z[now]))
        if now in (6, 9, 14):
            parents.append((now, tuple(e.timestamp for e in snapshot.retained)))
    expected = tuple(e.timestamp for e in snapshot.retained)
    teacher = AttentionContinuation("fixture", z, 17, salience)
    for t, kept in parents:
        assert teacher.final_kept(t, kept) == expected


def test_continuation_rejects_overcapacity_and_recent_candidates():
    teacher = AttentionContinuation("fixture", np.zeros((20, 1), np.float32), 17, lambda c, x: [0.] * len(c))
    with pytest.raises(ValueError):
        teacher.final_kept(9, (0, 1, 5))
    with pytest.raises(ValueError):
        teacher.final_kept(9, (0, 6))
    with pytest.raises(ValueError):
        teacher.final_kept(18, (0, 1))
