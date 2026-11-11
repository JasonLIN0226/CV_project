"""Ablation: the feature-attention-proxy teacher.

Same scorer, same initial weights, same minibatches as ours; only the training label differs.
The label is a softmax over cosine similarity between each candidate and the realised future
frames, in the predictor's projected space, in place of the predictor's error after each deletion.

Extracted unchanged from experiments/gate_supervision_comparison_design_v2/teachers.py.
"""
from __future__ import annotations

import numpy as np

from ..scorer import DIM


def attention_energy(candidate, future):
    """Feature-similarity attention proxy; future allowed only offline."""
    x = np.asarray(candidate, dtype=np.float64)
    y = np.asarray(future, dtype=np.float64)
    if x.shape != (3, DIM) or y.shape != (2, DIM):
        raise ValueError("expected [3,128] candidates and [2,128] futures")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite features")
    denom = np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(y, axis=1)[None, :]
    valid = (np.linalg.norm(x, axis=1)[:, None] > 1e-8) & (np.linalg.norm(y, axis=1)[None, :] > 1e-8)
    similarity = np.zeros((3, 2), dtype=np.float64)
    np.divide(x @ y.T, denom, out=similarity, where=valid)
    exponent = np.exp(similarity - similarity.max(axis=0, keepdims=True))
    return (exponent / exponent.sum(axis=0, keepdims=True)).mean(axis=1)
