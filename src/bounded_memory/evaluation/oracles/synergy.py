"""Synergy reference points: the lowest-error deletion, and the reader given all three cards.

`oracle` is privileged: it picks, per record, the deletion with the lowest realised loss.
`three_cards` is not a bounded memory; it is the teacher that Compress to Remember copies.
`deletion_path_check` verifies that a plain deletion passed through the path used to score
non-deletion memories reproduces the released cost of that deletion. Moved from
experiments/memory_baselines/synergy_released.py.
"""
import numpy as np
import torch

from ...environments.synergy.assets import qform, reader_batch, reader_coefficients


def oracle(cost):
    return cost.argmin(1)


def reader_checks(store, model, records, cost, chunk=64):
    """Returns (per-record loss with all three cards, max |difference| of the deletion path from `cost`)."""
    evaluation = store.banks["evaluation"]
    three, worst = np.empty(len(records)), 0.0
    with torch.inference_mode():
        for b in range(0, len(records), chunk):
            rr = records[b:b + chunk]
            cs = [store.capsule(r) for r in rr]
            rep = [c for c in cs for _ in evaluation]
            plans = np.tile(evaluation, (len(cs), 1, 1))
            resid = np.stack([store.target_residual(r, "evaluation") for r in rr])

            def loss(memory=None):
                coeff = reader_coefficients(model, reader_batch(rep, plans, memory)).numpy().astype(np.float64)
                return qform(coeff.reshape(len(cs), len(evaluation), 2, -1) - resid, store.G).mean(1)

            three[b:b + len(rr)] = loss()
            for d in range(3):
                keep = [j for j in range(3) if j != d]
                kept = tuple(torch.from_numpy(np.stack([c[k][4:][keep] for c in rep])) for k in ("pz", "past", "age"))
                worst = max(worst, float(np.abs(loss(kept) - cost[b:b + len(rr), d]).max()))
    return three, worst
