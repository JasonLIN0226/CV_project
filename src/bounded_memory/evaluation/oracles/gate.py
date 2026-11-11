"""Gate reference points: no history, the privileged cue keyframe, and the unbounded full history.

`recent_only` and `oracle_keyframe` are moved unchanged from
experiments/memory_baselines/gate_policies.py. None of them is a bounded online policy:
the oracle knows the cue timestamps and `full_history` keeps every eligible frame.
"""
from ...environments.gate.protocol import B, R


def recent_only(episodes):
    return [[] for _ in episodes]


def oracle_keyframe(episodes):
    """Privileged reference: the cue timestamps, padded with the newest eligible frames."""
    out = []
    for e in episodes:
        kept = [int(t) for t in e.evidence if int(t) <= e.q - R][:B]
        for t in range(e.q - R, -1, -1):
            if len(kept) == B:
                break
            if t not in kept:
                kept.append(t)
        out.append(sorted(kept))
    return out


def full_history(episodes):
    """Every eligible frame: a reference with no memory bound."""
    return [list(range(e.candidates)) for e in episodes]
