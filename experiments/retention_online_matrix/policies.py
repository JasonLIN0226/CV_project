"""Strict irreversible online retention rules for capacity B=2.

Every update sees only the two retained timestamps and the single timestamp
that has just left the recent buffer.  No rule may inspect the historical
archive, restore an evicted item, or use a future observation/target.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


POLICIES = (
    "fifo_online_b2",
    "reservoir_online_b2",
    "temporal_buckets_online_b2",
    "visual_diversity_online_b2",
    "prediction_surprise_online_b2",
    "attention_online_b2",
)


def checked_candidates(previous, arrival, capacity: int = 2) -> list[int]:
    """Return the only legal update pool: prior memory plus one arrival."""
    previous = [int(v) for v in previous]
    arrival = int(arrival)
    if len(previous) > capacity:
        raise ValueError("Retained memory exceeded capacity")
    if len(set(previous)) != len(previous) or arrival in previous:
        raise ValueError("Retained timestamps must be distinct")
    return previous + [arrival]


def choose_subset(candidates, selected, capacity: int = 2) -> list[int]:
    """Validate and sort a selected subset of the bounded candidate pool."""
    candidates = {int(v) for v in candidates}
    selected = [int(v) for v in selected]
    if len(selected) != min(capacity, len(candidates)):
        raise ValueError("Selection must preserve the fixed capacity")
    if len(set(selected)) != len(selected) or not set(selected).issubset(candidates):
        raise ValueError("Selection attempted to resurrect or invent a timestamp")
    return sorted(selected)


def warm_start(previous, arrival, capacity: int = 2) -> list[int]:
    candidates = checked_candidates(previous, arrival, capacity)
    return choose_subset(candidates, candidates, capacity)


def fifo(previous, arrival, capacity: int = 2) -> list[int]:
    candidates = checked_candidates(previous, arrival, capacity)
    return choose_subset(candidates, sorted(candidates)[-capacity:], capacity)


def reservoir(
    previous,
    arrival,
    arrival_ordinal: int,
    rng: np.random.Generator,
    capacity: int = 2,
) -> list[int]:
    """One-pass Algorithm R state update using only previous state and arrival."""
    candidates = checked_candidates(previous, arrival, capacity)
    if len(previous) < capacity:
        return choose_subset(candidates, candidates, capacity)
    if arrival_ordinal <= 0:
        raise ValueError("Arrival ordinal must be positive")
    replacement = int(rng.integers(0, arrival_ordinal + 1))
    chosen = list(previous)
    if replacement < capacity:
        chosen[replacement] = int(arrival)
    return choose_subset(candidates, chosen, capacity)


def temporal_bucket(age: int, recent: int) -> int:
    """Logarithmic age bucket anchored in units of the recent-buffer length."""
    if age < recent:
        raise ValueError("History age cannot precede the recent buffer")
    return int(math.floor(math.log2(max(1.0, age / float(recent)))))


def temporal_buckets(
    previous,
    arrival,
    clock: int,
    recent: int,
    capacity: int = 2,
) -> list[int]:
    """Keep the pair covering the most logarithmic age buckets.

    This is an explicitly bounded online adaptation of temporal stratification.
    Ties prefer the pair whose timestamps span the largest nonnegative gap, then
    the newer pair.
    """
    candidates = checked_candidates(previous, arrival, capacity)
    if len(candidates) <= capacity:
        return choose_subset(candidates, candidates, capacity)

    def score(pair):
        ages = [int(clock) - int(v) for v in pair]
        buckets = {temporal_bucket(age, recent) for age in ages}
        return (
            len(buckets),
            max(ages) - min(ages),
            sum(int(v) for v in pair),
        )

    pairs = list(itertools.combinations(candidates, capacity))
    chosen = max(pairs, key=lambda pair: (score(pair), tuple(sorted(pair))))
    return choose_subset(candidates, chosen, capacity)


def visual_diversity(
    previous,
    arrival,
    projected_history: np.ndarray,
    projected_anchor: np.ndarray,
    capacity: int = 2,
) -> list[int]:
    """Bounded anchored diversity using only old memory and the new arrival.

    Among the three possible retained pairs, maximize the sum of anchor-to-item
    and item-to-item Euclidean distances.  The anchor is the currently observed
    frame at the update time.
    """
    candidates = checked_candidates(previous, arrival, capacity)
    if len(candidates) <= capacity:
        return choose_subset(candidates, candidates, capacity)
    projected_history = np.asarray(projected_history, dtype=np.float64)
    anchor = np.asarray(projected_anchor, dtype=np.float64)

    def score(pair):
        first, second = (projected_history[timestamp] for timestamp in pair)
        return float(
            np.linalg.norm(anchor - first)
            + np.linalg.norm(anchor - second)
            + np.linalg.norm(first - second)
        )

    pairs = list(itertools.combinations(candidates, capacity))
    chosen = max(pairs, key=lambda pair: (score(pair), tuple(sorted(pair))))
    return choose_subset(candidates, chosen, capacity)


def top_scores(
    previous,
    arrival,
    scores: dict[int, float],
    capacity: int = 2,
) -> list[int]:
    """Keep the highest-scoring bounded candidates, breaking ties toward newer."""
    candidates = checked_candidates(previous, arrival, capacity)
    if len(candidates) <= capacity:
        return choose_subset(candidates, candidates, capacity)
    ranked = sorted(
        candidates,
        key=lambda timestamp: (
            -float(scores.get(int(timestamp), -np.inf)),
            -int(timestamp),
        ),
    )
    return choose_subset(candidates, ranked[:capacity], capacity)

