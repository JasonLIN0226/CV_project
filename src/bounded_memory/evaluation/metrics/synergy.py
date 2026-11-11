"""Synergy metrics: how often a needed card is deleted, and prediction regret.

Regret is a policy's expected in-span prediction loss minus that of the best single deletion for
the same record, so it is zero for the lowest-error deletion and can be negative for a memory
that is not a deletion. Moved unchanged from experiments/memory_baselines/synergy_data.py; the
definitions follow results/.../qualified-reader-retention-20261025/code/evaluate_dev.py.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

TASKS = "RDS"


def family_mean(values, records, index):
    groups = defaultdict(list)
    for i in index:
        groups[records[i]["family"]].append(values[i])
    return float(np.mean([np.mean(v) for v in groups.values()]))


def score(records, cost, drops=None, losses=None):
    """Per task: needed-evidence loss and prediction regret, as in evaluate_dev.py.

    drops: list of per-seed drop arrays (deletion policies). losses: list of per-seed [N] expected
    losses (memories that are not a deletion, e.g. compressed tokens); regret is measured against the
    best single deletion, so it can be negative.
    """
    correct = [set(r["semantic_correct_delete_set"]) for r in records]
    n = len(records)
    if drops is not None:
        lost = np.mean([[float(d[i] not in correct[i]) for i in range(n)] for d in drops], 0)
        regret = np.mean([[cost[i, d[i]] - cost[i].min() for i in range(n)] for d in drops], 0)
    else:
        lost = None
        regret = np.mean([np.asarray(l) - cost.min(1) for l in losses], 0)
    out = {}
    for task in TASKS:
        index = [i for i, r in enumerate(records) if r["task_family"] == task]
        out[task] = {"n": len(index),
                     "needed_evidence_lost": None if lost is None else float(lost[index].mean()),
                     "regret": family_mean(regret, records, index)}
    return out
