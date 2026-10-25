"""Target-free scoring kernels for the proposed terminal three-family study.

These kernels do not run a reader or establish its quality. Callers must supply
forecasts from the same frozen reader and the same TRAIN reference plans, never
the evaluation episode's unexecuted plan. The old frozen Gate2 runner is unchanged.
"""
import numpy as np


def _deletions(predictions):
    p = np.asarray(predictions, dtype=np.float64)
    if p.ndim != 4 or p.shape[1] != 3 or any(n == 0 for n in p.shape):
        raise ValueError('Expected [reference_plan, 3 deletions, horizon, feature]')
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite prediction')
    return p


def _decision(scores):
    return {'drop': int(np.argmin(scores)), 'scores': scores.tolist(),
            'exact_min_set': np.flatnonzero(scores == scores.min()).tolist(),
            'tolerant_min_set': np.flatnonzero(scores <= scores.min()+1e-6).tolist()}


def medoid(deletion_predictions):
    """Average per-plan squared distances; exact earliest-index tie breaking."""
    p = _deletions(deletion_predictions)
    distances = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i+1, 3):
            distances[i, j] = distances[j, i] = np.mean((p[:, i]-p[:, j])**2)
    return {**_decision(distances.sum(1)), 'pairwise_mse': distances.tolist()}


def full_set_sensitivity(deletion_predictions, full_predictions):
    """Compare every K-retained prediction with one common K+1 reference.

    This measures output preservation, not ground-truth loss. A future trained
    reader must be qualified on both K and K+1 before using this comparator.
    """
    p = _deletions(deletion_predictions)
    full = np.asarray(full_predictions, dtype=np.float64)
    if full.shape != (p.shape[0], p.shape[2], p.shape[3]):
        raise ValueError('Full reference must use identical plans, horizons and features')
    if not np.isfinite(full).all():
        raise ValueError('Nonfinite full-set reference')
    scores = np.mean((p-full[:, None])**2, axis=(0, 2, 3))
    return _decision(scores)
