"""Candidate-pool preflight only; no predictor, labels-to-model path or policy."""
import numpy as np


def gate_age_counterbalance(cue_start, cue_end, query, recent=4):
    """Privileged Gate diagnostic: put its cue at each chronological rank.

    Cue timing is evaluation-only pool-construction metadata, not a deployable
    retention rule. Non-cue RGB must be checked against paired siblings first.
    """
    stop = query-recent+1
    if not (2 <= cue_start <= cue_end and cue_end+2 < stop):
        raise ValueError('Need two genuine pre-cue and post-cue observations')
    return {'cue_oldest': (cue_start, cue_end+1, stop-1),
            'cue_middle': (cue_start-1, cue_start, cue_end+1),
            'cue_newest': (0, cue_start-1, cue_start)}


def clock_only_static_pools(query, seed, recent=4, count=6):
    """Real past timestamps only, without cue, visibility, hidden-state or future access.

    These are static headroom diagnostics, not sequential streaming eviction.
    Uninformative pools must remain in the denominator.
    """
    stop = query-recent+1
    if stop < 3 or count < 1 or count > stop*(stop-1)*(stop-2)//6:
        raise ValueError('Insufficient distinct legal triples')
    rng = np.random.default_rng(seed)
    chosen = set()
    while len(chosen) < count:
        chosen.add(tuple(sorted(int(x) for x in rng.choice(stop, 3, replace=False))))
    return sorted(chosen)
