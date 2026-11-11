"""Irreversible online retention with B = 2: the replay loop, the behaviour policies that visit
training states, and the selections a trained scorer makes.

`rollout`, `behavior_update` and `BEHAVIORS` are extracted unchanged from
experiments/gate_supervision_comparison_design_v2/core.py and `deterministic_seed` from its
states.py. `scorer_states` and `scorer_selections` are moved from
experiments/memory_baselines/gate_policies.py; the behaviour rules they replay are the audited
ones in evaluation/heuristics/gate.py.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ...evaluation.heuristics.gate import update as behavior_rule
from .scorer import B, DIM, R, OnlineState, score_update

BEHAVIORS = ("fifo", "reservoir", "temporal_buckets", "visual_diversity")


def rollout(projected, actions, query, update, *, capture_final=False):
    """Replay an irreversible stream. update only receives OnlineState.

    Input arrays are offline replay storage; the returned trace is an audit, not a
    physical B=2 streaming memory measurement. capture_final returns the pool
    before the query-time eviction, without calling update at that event.
    """
    p = np.asarray(projected)
    a = np.asarray(actions)
    if p.shape != (query + 1, DIM) or a.shape != (query, 2):
        raise ValueError("rollout accepts exactly query-time observation prefixes")
    memory, trace = [], []
    for clock in range(R, query + 1):
        arrival = clock - R
        pool = memory + [arrival]
        if len(pool) <= B:
            selected = sorted(pool)
        else:
            state = OnlineState(clock, tuple(pool), p[:clock + 1], a[:clock])
            if capture_final and clock == query:
                return state, trace
            selected = sorted(update(state))
            if len(selected) != B or len(set(selected)) != B or not set(selected) <= set(pool):
                raise ValueError("selector attempted invalid selection or resurrection")
        trace.append({"clock": clock, "previous": memory.copy(),
                      "arrival": arrival, "selected": selected.copy()})
        memory = selected
    if capture_final:
        raise ValueError("query has no three-candidate eviction")
    return memory, trace


def behavior_update(name, rng, update_impl):
    """Reuse exact frozen v1 rules, supplying no learned scores or model."""
    if name not in BEHAVIORS:
        raise ValueError("unknown or task-trained behavior")
    def decide(state):
        pool = state.candidates
        vectors = {t: state.projected_prefix[t].copy() for t in pool} if name == "visual_diversity" else {}
        return update_impl(name, list(pool[:2]), pool[2], state.clock, R,
                           vectors, state.projected_prefix[state.clock], {}, rng)
    return decide


def deterministic_seed(seed, episode_id, policy):
    raw = f"gate-v2\0{seed}\0{episode_id}\0{policy}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def scorer_states(episodes, seed):
    """Query-time three-candidate states reached by the four behaviour policies (the scorer's training set)."""
    for e in episodes:
        for policy in BEHAVIORS:
            rng = np.random.default_rng(deterministic_seed(seed, e.id, policy))
            state, _ = rollout(e.p[:e.q + 1].copy(), e.actions[:e.q].copy(), e.q,
                               behavior_update(policy, rng, behavior_rule), capture_final=True)
            yield e, policy, state


def scorer_selections(scorer, episodes):
    out = []
    for e in episodes:
        kept, _ = rollout(e.p[:e.q + 1].copy(), e.actions[:e.q].copy(), e.q, lambda s: score_update(scorer, s))
        out.append(sorted(kept))
    return out
