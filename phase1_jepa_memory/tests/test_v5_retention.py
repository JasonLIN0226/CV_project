import sys
from pathlib import Path
import random
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v5_retention import HistoryState, insert, deletion_forks, stream_prefix


@pytest.mark.parametrize("policy", ["FIFO", "Reservoir", "Uniform"])
def test_irreversibility_equal_budget_and_no_recent_overlap(policy):
    rng = random.Random(17)
    draws = [rng.random() for _ in range(120 - 8)]
    replay = stream_prefix(120, 8, 16, policy, draws)
    deleted = set()
    for time, (recent, history, discarded) in enumerate(replay):
        if discarded is not None:
            deleted.add(discarded)
        assert not deleted.intersection(history)
        assert not set(recent).intersection(history)
        assert all(t < recent[0] for t in history)
        assert len(history) == min(16, max(0, time - 8 + 1))
        assert len(recent) <= 8


def test_longer_future_does_not_change_any_earlier_decision():
    rng = random.Random(29)
    draws = [rng.random() for _ in range(92)]
    long = stream_prefix(100, 8, 16, "Reservoir", draws)
    short = stream_prefix(40, 8, 16, "Reservoir", draws[:32])
    assert long[:40] == short


def test_all17_forks_keep_same_clock_and_can_reject_incoming():
    state = HistoryState(16, tuple(range(16)), 16, 15)
    forks = deletion_forks(state, 16)
    assert len(forks) == 17
    assert {deleted for deleted, _ in forks} == set(range(17))
    assert all(s.seen == 17 and s.last_incoming == 16 and len(s.retained) == 16 for _, s in forks)
    next_states = [insert(s, 17, "Reservoir", .99)[0] for _, s in forks]
    assert all(s.last_incoming == 17 and s.seen == 18 for s in next_states)
    # A common draw rejects the new incoming for every coupled continuation.
    assert all(17 not in s.retained for s in next_states)
    assert all(deleted not in s.retained for (deleted, _), s in zip(forks, next_states))


def test_skipped_or_reintroduced_identity_fails():
    state = HistoryState(2, (0, 1), 2, 1)
    with pytest.raises(ValueError):
        insert(state, 0, "FIFO", .5)
    with pytest.raises(ValueError):
        insert(state, 3, "FIFO", .5)
    with pytest.raises(ValueError):
        insert(state, 2, "learned", .5)
