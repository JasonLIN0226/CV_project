"""PaFu-KV (Chen et al., arXiv 2601.21896) as a frame-level bounded-memory policy, on Gate.

Moved unchanged from experiments/memory_baselines/pafu.py.

No official code is public (the authors' repository holds only a README), so this follows
the paper text. What is kept, and what had to change for a predictor that is not a
causal video diffusion transformer:

  kept     Teacher salience from attention of a model that sees past and future (Sec. 3.2):
           for every key, the maximum attention over queries in earlier / same / later
           blocks, averaged over the blocks that exist (Eq. 4-5).
  kept     Student: a two-layer MLP with SiLU that regresses the salience with SmoothL1
           (Sec. 3.3, Eq. 6, App. A).
  kept     Policy: each token is scored once when it is written; when the cache is full,
           the lowest stored scores are evicted (Sec. 3.4).
  changed  A block is one frame. Attention is the frozen predictor's own pooling weight
           (one head), probed with queries placed at every frame up to query+6. The paper's
           teacher is a separate bidirectional 14B model; here the reader is the only model.
  changed  The student reads the frame's projected feature and its causal context when it
           enters the memory pool, in place of the token's Q, K, V from the last layer.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...environments.gate.protocol import B, R, pooling_attention


@torch.inference_mode()
def teacher_salience(model, e, device, future=6):
    """Salience of every frame 0..q of one episode. Offline only: uses frames after the query."""
    q = e.q
    keys = torch.from_numpy(e.p[:q + 1]).to(device)
    query_times = np.arange(q + future + 1)
    queries = torch.from_numpy(e.p[query_times]).to(device)
    attention = pooling_attention(model, keys[None].expand(len(queries), -1, -1), queries).cpu().numpy()
    salience = np.empty(q + 1, np.float64)
    for k in range(q + 1):
        groups = (query_times < k, query_times == k, query_times > k)
        salience[k] = np.mean([attention[g, k].max() for g in groups if g.any()])
    return salience.astype(np.float32)


def write_features(e, k):
    """Causal inputs for frame k at the clock it leaves the recent window (t = k + R)."""
    t = k + R
    return np.concatenate([e.p[k], e.p[k] - e.p[t], e.p[t - R + 1:t + 1].mean(0)]).astype(np.float32)


class SalienceHead(nn.Module):
    """S-MLP: two linear layers with SiLU in between (App. A)."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(model, train_episodes, seed, device, steps=1000, batch=256, lr=1e-3):
    x, y = [], []
    for e in train_episodes:
        s = teacher_salience(model, e, device)
        for k in range(e.candidates):
            x.append(write_features(e, k)); y.append(s[k])
    x = torch.as_tensor(np.stack(x), device=device)
    y = torch.as_tensor(np.asarray(y, np.float32), device=device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = SalienceHead(x.shape[1])
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    curve = []
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(x), size=batch), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(head(x[idx]), y[idx])
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            curve.append(float(loss))
    return head.eval(), {"tokens": int(len(x)), "loss_first": curve[0], "loss_last": curve[-1],
                         "target_mean": float(y.mean()), "target_std": float(y.std())}


@torch.inference_mode()
def select(head, episodes, device):
    """Online replay: one stored scalar per resident frame, evict the lowest (ties: oldest)."""
    out = []
    for e in episodes:
        scores, kept = {}, []
        for t in range(R, e.q + 1):
            k = t - R
            scores[k] = float(head(torch.from_numpy(write_features(e, k)).to(device)[None])[0])
            pool = kept + [k]
            if len(pool) > B:
                evict = min(pool, key=lambda f: (scores[f], f))
                pool.remove(evict)
                del scores[evict]
            kept = pool
        out.append(sorted(kept))
    return out


@torch.inference_mode()
def teacher_select(model, episodes, device):
    """Diagnostic upper reference for the student: keep the two frames with the highest teacher salience."""
    out = []
    for e in episodes:
        s = teacher_salience(model, e, device)[:e.candidates]
        out.append(sorted(np.argsort(-s, kind="stable")[:B].tolist()))
    return out
