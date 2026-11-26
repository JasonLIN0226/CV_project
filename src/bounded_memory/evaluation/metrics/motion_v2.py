"""Motion v2 scores computed from all-pairs loss tables (methods/deletion_cost/motion_v2.pair_table).

A table holds the reader-mean decoded-position MSE at +3 and +6 of every legal memory pair of one episode,
so any policy's final pair is scored by a lookup, the same number the frozen readers give it online.

    final_pair_losses   [episodes, 2 horizons] loss of each episode's final kept pair
    paired              a policy against a baseline per scene pair (mean of siblings and horizons), with a
                        95% interval that resamples scene pairs
"""
from __future__ import annotations

import numpy as np


def final_pair_losses(tables, episodes, kept):
    """[episodes, 2 horizons] position loss of each episode's final pair, looked up in the tables."""
    return np.array([tables[ep["rec"]["id"]][:, a, b] for ep, (a, b) in zip(episodes, kept)])


def paired(policy, baseline, episodes, resamples=10000, seed=20261126, tolerance=1e-12):
    """policy, baseline: [episodes, horizons] losses. Scene-pair means of (policy - baseline); negative = policy lower.
    A pair counts as lower only by more than `tolerance`, so float rounding of equal choices is not a win."""
    pairs = sorted({ep["rec"]["pair"] for ep in episodes})
    index = {p: [i for i, ep in enumerate(episodes) if ep["rec"]["pair"] == p] for p in pairs}
    diff = np.array([(policy[index[p]] - baseline[index[p]]).mean() for p in pairs])
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(pairs), size=(resamples, len(pairs)))].mean(1)
    return {"pairs": len(pairs), "difference": float(diff.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "share_of_pairs_lower": float((diff < -tolerance).mean())}
