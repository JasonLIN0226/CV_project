"""Our method: a past-only, permutation-equivariant eviction scorer.

At each arrival the memory holds two frames and a third arrives. The scorer reads the three
candidates' features, which are built only from already observed frames and executed actions,
and returns one deletion logit per candidate; the largest is evicted.

Extracted unchanged from experiments/gate_supervision_comparison_design_v2/core.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn

R = 4
B = 2
DIM = 128


class EvictionScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(517, 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (3, 517):
            raise ValueError("expected [batch,3,517] past-only candidate features")
        hidden = self.phi(features)
        pooled = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
        return self.head(torch.cat((hidden, pooled), dim=-1)).squeeze(-1)


def paired_initialization(seed):
    """Identical parameters; independent optimizers, without using predictor.score."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        a = EvictionScorer()
        d = EvictionScorer()
    d.load_state_dict(a.state_dict())
    return a, d


@dataclass(frozen=True)
class OnlineState:
    clock: int
    candidates: tuple[int, int, int]
    # Prefix only: no target, ID, label, cue, future action, or end-of-query.
    projected_prefix: np.ndarray
    actions_prefix: np.ndarray

    def __post_init__(self):
        c = self.clock
        p = np.asarray(self.projected_prefix)
        a = np.asarray(self.actions_prefix)
        if c < R or p.shape != (c + 1, DIM) or a.shape != (c, 2):
            raise ValueError("expected exactly the observed feature/action prefixes")
        if len(self.candidates) != 3 or len(set(self.candidates)) != 3:
            raise ValueError("three distinct candidates required")
        if any(t < 0 or t > c - R for t in self.candidates):
            raise ValueError("candidate outside legal history")
        if not np.isfinite(p).all() or not np.isfinite(a).all():
            raise ValueError("nonfinite observations or actions")

    def features(self):
        p = np.asarray(self.projected_prefix, dtype=np.float32)
        actions = np.asarray(self.actions_prefix, dtype=np.float32)
        c = self.clock
        current = p[c]
        recent_mean = p[c - R + 1:c + 1].mean(axis=0)
        prefix = np.concatenate([np.zeros((1, 2), np.float32),
                                 np.cumsum(actions, axis=0)], axis=0)
        out = []
        for t in self.candidates:
            out.append(np.concatenate((p[t], p[t] - current,
                np.array([math.log1p(c - t) / math.log(65)], np.float32),
                prefix[c] - prefix[t], current, recent_mean, actions[c - 1])))
        return np.stack(out).astype(np.float32)


def choose(scores, candidates):
    """Largest deletion logit; exact ties delete oldest timestamp."""
    vals = np.asarray(scores, dtype=np.float64)
    if vals.shape != (3,) or not np.isfinite(vals).all():
        raise ValueError("three finite deletion logits required")
    candidates = tuple(candidates)
    if len(candidates) != 3 or len(set(candidates)) != 3:
        raise ValueError("three distinct candidate timestamps required")
    evicted = min(candidates[i] for i in np.flatnonzero(vals == vals.max()))
    return sorted(t for t in candidates if t != evicted)


@torch.inference_mode()
def score_update(model, state):
    model.eval()
    device = next(model.parameters()).device
    features = torch.from_numpy(state.features()).to(device).unsqueeze(0)
    return choose(model(features).squeeze(0).cpu().numpy(), state.candidates)
