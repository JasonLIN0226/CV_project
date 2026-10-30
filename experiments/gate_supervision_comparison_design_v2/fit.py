"""Paired optimizer implementation. No command entrypoint or automatic execution."""
from __future__ import annotations
import copy
import numpy as np
import torch
from .core import paired_initialization


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
