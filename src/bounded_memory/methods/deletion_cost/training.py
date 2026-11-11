"""Training of the eviction scorer and loading of the frozen published scorers.

`minibatch_indices` and `fit_pair` are extracted unchanged from
experiments/gate_supervision_comparison_design_v2/fit.py: the deletion-cost scorer and its
attention-proxy twin start from identical weights and see identical minibatches for 1,000 steps.
`teacher_labels`, `fit_scorers`, `train_scorers` and `frozen_scorers` are moved from
experiments/memory_baselines/gate_policies.py.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from ...predictors.inputs import collate
from .retention import scorer_states
from .scorer import EvictionScorer, paired_initialization
from .teacher import OfflineTargets, teacher_pair

ARMS = ("attention_proxy_student", "deletion_cost_student")   # order of the two models in a student checkpoint


def minibatch_indices(seed, n_states):
    if n_states < 1:
        raise ValueError("empty training states")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_states, size=(1000, 64))


def fit_pair(features, attention_targets, deletion_targets, *, seed, split, device="cpu"):
    """Explicit call performs the authorized 1000-step paired fit.

    Callers must separately verify source hashes, budgets, and authorization;
    this function performs no I/O, no validation selection and no test access.
    """
    if split != "train":
        raise ValueError("optimizer accepts only training split")
    if seed not in (17, 29, 43):
        raise ValueError("unexpected frozen predictor/student seed")
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y_a = torch.as_tensor(attention_targets, dtype=torch.float32, device=device)
    y_d = torch.as_tensor(deletion_targets, dtype=torch.float32, device=device)
    if x.ndim != 3 or x.shape[1:] != (3, 517) or y_a.shape != (len(x), 3) or y_d.shape != y_a.shape:
        raise ValueError("invalid training tensor dimensions")
    if len(x) != 4096:
        raise ValueError("expected the complete 4096-record training state pack")
    if not torch.isfinite(x).all() or not torch.isfinite(y_a).all() or not torch.isfinite(y_d).all():
        raise ValueError("nonfinite training data")
    for target in (y_a, y_d):
        if (target < 0).any() or not torch.allclose(target.sum(dim=1), torch.ones(len(x), device=device), atol=1e-6):
            raise ValueError("invalid target distribution")
    a, d = paired_initialization(seed)
    a, d = a.to(device), d.to(device)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001) for m in (a, d)]
    batches = minibatch_indices(seed, len(x))
    for batch in batches:
        indices = torch.as_tensor(batch, device=device)
        for model, optimizer, target in zip((a, d), optimizers, (y_a, y_d)):
            optimizer.zero_grad(set_to_none=True)
            loss = -(target[indices] * torch.log_softmax(model(x[indices]), dim=-1)).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {"attention": copy.deepcopy(a.state_dict()), "deletion": copy.deepcopy(d.state_dict()),
            "seed": seed, "steps_per_arm": 1000, "batch_size": 64}


def teacher_labels(model, train_episodes, seed, device):
    """Features and both target distributions for every training state of this predictor seed."""
    x, ya, yd, meta = [], [], [], []
    for e, policy, state in scorer_states(train_episodes, seed):
        q = e.q
        targets = OfflineTargets("train", e.z[[q + 3, q + 6]].copy(), e.actions[q:q + 6].copy())
        labels = teacher_pair(state, targets, e.z[:q + 1].copy(), model, collate, device)
        x.append(state.features()); ya.append(labels["attention_target"]); yd.append(labels["deletion_target"])
        meta.append({"episode": e.id, "behavior": policy, "candidates": list(state.candidates),
                     "attention_energy": labels["attention_energy"], "deletion_energy": labels["deletion_energy"]})
    x, ya, yd = (np.stack(v).astype(np.float32) for v in (x, ya, yd))
    return x, ya, yd, meta


def fit_scorers(x, ya, yd, seed, device):
    """State dicts of the two matched scorers trained on these labels."""
    if len(x) == 4096:
        fit = fit_pair(x, ya, yd, seed=seed, split="train", device=device)  # the project's exact fit
        return {"attention_proxy_student": fit["attention"], "deletion_cost_student": fit["deletion"]}
    # Same update rule for other dataset sizes (local smoke runs only).
    xt = torch.as_tensor(x, device=device)
    a, d = paired_initialization(seed)
    models = [a.to(device), d.to(device)]
    opts = [torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001) for m in models]
    labels = [torch.as_tensor(y, device=device) for y in (ya, yd)]
    for batch in minibatch_indices(seed, len(x)):
        idx = torch.as_tensor(batch, device=device)
        for m, opt, y in zip(models, opts, labels):
            opt.zero_grad(set_to_none=True)
            loss = -(y[idx] * torch.log_softmax(m(xt[idx]), dim=-1)).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return {"attention_proxy_student": models[0].state_dict(), "deletion_cost_student": models[1].state_dict()}


def as_models(states, device):
    out = {}
    for name, state in states.items():
        scorer = EvictionScorer().to(device)
        scorer.load_state_dict(state, strict=True)
        if not all(torch.isfinite(p).all() for p in scorer.parameters()):
            raise ValueError("scorer contains nonfinite parameters")
        out[name] = scorer.eval()
    return out


def train_scorers(model, train_episodes, seed, device):
    """Deletion-cost scorer and its matched feature-attention-proxy twin (same init, same batches)."""
    x, ya, yd, _ = teacher_labels(model, train_episodes, seed, device)
    return as_models(fit_scorers(x, ya, yd, seed, device), device), {"training_states": int(len(x))}


def save_students(path, states, seed):
    """Same layout as the published student_<seed>.pt: both arms, attention proxy first."""
    from ...data.io import save_torch
    save_torch(path, {"seed": seed, "step": 1000, "models": [states[name] for name in ARMS]})


def frozen_scorers(model_root, seed, device, filename="student_{seed}.pt"):
    """Load a matched scorer checkpoint (the team's published one, or one written by save_students)."""
    path = Path(model_root) / filename.format(seed=seed)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    models = checkpoint.get("models")
    if checkpoint.get("seed") != seed or checkpoint.get("step") != 1000 or not isinstance(models, (list, tuple)) or len(models) != 2:
        raise ValueError("not the completed matched scorer checkpoint for this seed")
    return as_models(dict(zip(ARMS, models)), device), {
        "source": "frozen checkpoint, not retrained", "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
