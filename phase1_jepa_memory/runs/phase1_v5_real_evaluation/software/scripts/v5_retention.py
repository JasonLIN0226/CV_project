"""Model-independent irreversible retention primitives for v5.

State holds retained identities and counters only. There is no archive/retrieval,
future target, label, learned selector, or predictor interface here.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HistoryState:
    capacity: int
    retained: tuple = ()
    seen: int = 0
    last_incoming: int = -1

    def __post_init__(self):
        if self.capacity <= 0 or len(self.retained) > self.capacity:
            raise ValueError("invalid capacity")
        if tuple(sorted(set(self.retained))) != self.retained:
            raise ValueError("retained identities must be unique chronological indices")
        if self.seen < len(self.retained):
            raise ValueError("eligible history counter is inconsistent")
        if self.retained and self.retained[-1] > self.last_incoming:
            raise ValueError("retained future identity")


def candidate_ids(state, incoming):
    if incoming != state.last_incoming + 1:
        raise ValueError("incoming history must advance exactly once in time")
    return state.retained + (incoming,)


def insert(state, incoming, policy, draw):
    """One admission/deletion; random draw is supplied externally for coupling.

    Reservoir uses Algorithm R's eligible-history count. Slot order is actual
    timestamp order, so deletion never relabels observation ages.
    """
    if policy not in ("FIFO", "Reservoir", "Uniform"):
        raise ValueError("unregistered retention policy")
    if not math.isfinite(draw) or not 0 <= draw < 1:
        raise ValueError("draw must lie in [0,1)")
    candidates = candidate_ids(state, incoming)
    seen = state.seen + 1
    deleted = None
    if len(candidates) <= state.capacity:
        retained = candidates
    else:
        if policy == "FIFO":
            drop = 0
        elif policy == "Uniform":
            drop = min(int(draw * len(candidates)), len(candidates) - 1)
        else:
            reservoir_index = min(int(draw * seen), seen - 1)
            drop = reservoir_index if reservoir_index < state.capacity else state.capacity
        deleted = candidates[drop]
        retained = tuple(x for i, x in enumerate(candidates) if i != drop)
    return HistoryState(state.capacity, retained, seen, incoming), deleted


def deletion_forks(state, incoming):
    """All B+1-to-B alternatives, without scoring or choosing an oracle."""
    candidates = candidate_ids(state, incoming)
    if len(state.retained) != state.capacity:
        raise ValueError("controlled deletion requires a full memory")
    return tuple((deleted, HistoryState(state.capacity,
                 tuple(x for x in candidates if x != deleted), state.seen + 1, incoming))
                 for deleted in candidates)


def stream_prefix(length, recent, capacity, policy, draws):
    """Artificial/software replay helper; only the supplied chronological prefix.

    Each returned tuple exposes recent and currently retained history. No identity
    that was previously deleted can re-enter memory.
    """
    if length < 0 or recent <= 0 or len(draws) != max(0, length - recent):
        raise ValueError("invalid stream or per-update random draw count")
    state = HistoryState(capacity)
    output = []
    for time in range(length):
        deleted = None
        if time >= recent:
            incoming = time - recent
            state, deleted = insert(state, incoming, policy, draws[incoming])
        recent_ids = tuple(range(max(0, time - recent + 1), time + 1))
        if set(recent_ids) & set(state.retained):
            raise ValueError("recent/history overlap")
        output.append((recent_ids, state.retained, deleted))
    return tuple(output)
