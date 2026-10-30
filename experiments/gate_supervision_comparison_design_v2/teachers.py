"""Offline train/validation teacher labels; never pass this object to a selector."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch

from .core import DIM, R, OnlineState


@dataclass(frozen=True)
class OfflineTargets:
    split: str
    clean_future: np.ndarray  # normalized latent, [2, latent_dim]
    future_actions: np.ndarray  # exactly six recorded actions

    def __post_init__(self):
        if self.split not in {"train", "validation"}:
            raise ValueError("only train/validation teacher data permitted")
        future = np.asarray(self.clean_future)
        acts = np.asarray(self.future_actions)
        if future.ndim != 2 or future.shape[0] != 2 or acts.shape != (6, 2):
            raise ValueError("expected two clean targets and six future actions")
        if not np.isfinite(future).all() or not np.isfinite(acts).all():
            raise ValueError("nonfinite teacher inputs")


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


def deletion_distribution(energy):
    values = np.asarray(energy, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("three finite teacher energies required")
    logits = -values / max(float(values.std(ddof=0)), 1e-6)
    weights = np.exp(logits - logits.max())
    return (weights / weights.sum()).astype(np.float32)


def predictor_samples(state, latent_prefix, future_actions):
    """Three equal-B requests, compatible with phase1.inputs.collate."""
    q = state.clock
    z = np.asarray(latent_prefix, dtype=np.float32)
    actions = np.asarray(state.actions_prefix, dtype=np.float32)
    future_actions = np.asarray(future_actions, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != q + 1 or future_actions.shape != (6, 2):
        raise ValueError("latent prefix and action continuation mismatch")
    if not np.isfinite(z).all() or not np.isfinite(future_actions).all():
        raise ValueError("nonfinite predictor input")
    prefix = np.concatenate((np.zeros((1, 2), np.float32),
                             np.cumsum(actions, axis=0)), axis=0)
    p = np.asarray(state.projected_prefix, dtype=np.float32)
    samples = []
    for removed in state.candidates:
        kept = sorted(t for t in state.candidates if t != removed)
        times = np.array(list(range(q - R + 1, q + 1)) + kept, dtype=np.int64)
        samples.append({"z": z[times].copy(), "age": (q - times).astype(np.float32),
            "type": np.array([0] * R + [1] * 2, np.int64),
            "past_actions": prefix[q] - prefix[times], "actions": future_actions.copy(),
            "current": z[q].copy(), "projected_z": p[times].copy(),
            "projected_current": p[q].copy()})
    return samples


@torch.inference_mode()
def deletion_energy(state: OnlineState, targets: OfflineTargets, latent_prefix, model, collate, device):
    """Frozen predictor on three deletions; returns mean normalized latent MSE."""
    samples = predictor_samples(state, latent_prefix, targets.future_actions)
    model.eval()
    pred = model(collate(samples, device)).detach().cpu().numpy()
    truth = np.asarray(targets.clean_future, dtype=np.float32)
    if pred.shape != (3,) + truth.shape or not np.isfinite(pred).all():
        raise ValueError("invalid frozen predictor output")
    return np.mean((pred.astype(np.float64) - truth[None]) ** 2, axis=(1, 2))


def teacher_pair(state, targets, latent_prefix, model, collate, device):
    """Produces target distributions; no training or persistence is performed."""
    if tuple(getattr(model, "horizons", ())) != (3, 6):
        raise ValueError("frozen predictor must use horizons 3 and 6")
    start_a = perf_counter()
    with torch.inference_mode():
        future = torch.as_tensor(targets.clean_future, device=device)
        projected = ((future - model.input_mean) @ model.projection).cpu().numpy()
    candidates = state.projected_prefix[list(state.candidates)]
    a = attention_energy(candidates, projected)
    elapsed_a = perf_counter() - start_a
    start_d = perf_counter()
    d = deletion_energy(state, targets, latent_prefix, model, collate, device)
    elapsed_d = perf_counter() - start_d
    return {"attention_energy": a, "deletion_energy": d,
            "attention_target": deletion_distribution(a),
            "deletion_target": deletion_distribution(d),
            "attention_seconds": elapsed_a, "deletion_seconds": elapsed_d}
