"""Gate evaluation: route accuracy from the frozen linear state decoder, and normalised latent MSE.

`LinearReadout` is extracted unchanged from experiments/gate_clean_future/frozen_control/probes.py
(the clean-future version of phase1/probes.py). `metric_rows` and `summarise` are moved unchanged
from experiments/memory_baselines/gate_data.py; they follow
retention_online_matrix/run_online_matrix.py::append_metric_rows.
"""
from __future__ import annotations

import numpy as np
import torch

from ...data.io import write_npz
from ...environments.gate.protocol import HORIZONS


class LinearReadout:
    def fit(self, x, y, ridge):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.mean, self.std = x.mean(0), np.maximum(x.std(0), .05)
        x = (x-self.mean)/self.std
        self.bias = y.mean(0)
        xx, yy = torch.from_numpy(x), torch.from_numpy(y-self.bias)
        if len(x) <= x.shape[1]:
            gram = xx @ xx.T
            gram.diagonal().add_(ridge)
            w = xx.T @ torch.linalg.solve(gram, yy)
        else:
            gram = xx.T @ xx
            gram.diagonal().add_(ridge)
            w = torch.linalg.solve(gram, xx.T @ yy)
        self.weight = w.numpy()
        self.ridge = ridge
        return self

    def predict(self, x):
        return ((np.asarray(x)-self.mean)/self.std) @ self.weight + self.bias

    def save(self, path):
        write_npz(path, mean=self.mean, std=self.std, weight=self.weight, bias=self.bias, ridge=np.float64(self.ridge))

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as f:
            for k in f.files:
                setattr(obj, k, f[k])
        return obj


def metric_rows(assets, episodes, prediction, selected, policy, seed):
    """One row per episode and horizon. `selected` is a list of kept frame lists, or None for token memories."""
    prediction = np.asarray(prediction, np.float32)
    if prediction.shape[:2] != (len(episodes), len(HORIZONS)) or not np.isfinite(prediction).all():
        raise ValueError("invalid predictor output")
    rows = []
    for i, e in enumerate(episodes):
        state = assets.decoder.predict(prediction[i])
        kept = None if selected is None else sorted(int(t) for t in selected[i])
        for h_index, h in enumerate(HORIZONS):
            target = e.z[e.q + h]
            rows.append({
                "policy": policy, "seed": seed, "episode": e.id, "pair": e.pair, "horizon": h,
                "route_accuracy": float((state[h_index, 0] > 0) == (e.classes[e.q + h] > 0)),
                "normalized_mse": float(np.mean((prediction[i, h_index] - target) ** 2)),
                "cue_kept": None if kept is None else float(any(e.cue[0] <= t <= e.cue[1] for t in kept)),
                "kept": None if kept is None else kept})
    return rows


def summarise(rows):
    """Pooled means per policy over episodes, seeds and horizons (the README's aggregation)."""
    out = {}
    for policy in sorted({r["policy"] for r in rows}):
        group = [r for r in rows if r["policy"] == policy]
        cue = [r["cue_kept"] for r in group if r["cue_kept"] is not None]
        per_seed = {}
        for seed in sorted({r["seed"] for r in group}):
            per_seed[str(seed)] = float(np.mean([r["route_accuracy"] for r in group if r["seed"] == seed]))
        out[policy] = {"route_accuracy": float(np.mean([r["route_accuracy"] for r in group])),
                       "normalized_mse": float(np.mean([r["normalized_mse"] for r in group])),
                       "cue_kept": float(np.mean(cue)) if cue else None,
                       "rows": len(group), "route_accuracy_by_seed": per_seed}
    return out
