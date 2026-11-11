"""PaFu-KV on the terminal (Synergy) tasks: one stored score per card, the lowest is deleted.

Moved unchanged from experiments/memory_baselines/synergy_run.py. Each record is one decision
over three candidates, so "ranks the whole history" and "online" coincide here.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ...environments.synergy.assets import H, frame, head_attention, reader_batch
from .gate import SalienceHead


@torch.inference_mode()
def pafu_teacher(model, store, records, bank="reference", chunk=64):
    """Salience of the three candidates: per attention path, the max over queries placed before / at /
    after each candidate, averaged over paths and over the groups that exist (PaFu Eq. 4-5)."""
    out = np.zeros((len(records), 3), np.float32)
    for a in range(0, len(records), chunk):
        rows, owner, times = [], [], []
        for i, r in enumerate(records[a:a + chunk]):
            q = r["query"]
            cap = store.capsule(r)
            ids = list(r["rgb_indices"][:q + 1]) + [x for plan in r["plan_targets"][bank] for x in plan]
            tt = list(range(q + 1)) + [q + h for _ in r["plan_targets"][bank] for h in H]
            seen = set()
            for frame_id, t in zip(ids, tt):
                if (frame_id, t) in seen:      # identical frame at the same time gives the same row
                    continue
                seen.add((frame_id, t))
                rows.append((cap, store.P[frame_id])); owner.append(i); times.append(t)
        batch = reader_batch([c for c, _ in rows], np.zeros((len(rows), 6, 2), np.float32))
        batch["projected_current"] = torch.from_numpy(np.stack([f for _, f in rows]).astype(np.float32))
        w = head_attention(model, batch).numpy()            # [rows, paths, 3]
        owner, times = np.asarray(owner), np.asarray(times)
        for i, r in enumerate(records[a:a + chunk]):
            mine = owner == i
            for j in range(3):
                k = r["query"] - 6 + j
                groups = (times[mine] < k, times[mine] == k, times[mine] > k)
                out[a + i, j] = np.mean([w[mine][g][:, :, j].max(0).mean() for g in groups if g.any()])
    return out


def pafu_features(store, r):
    """Each candidate when it leaves the recent window (clock = its time + 4)."""
    out = []
    for j in range(3):
        k = r["query"] - 6 + j
        t = k + 4
        recent = np.stack([frame(store, r, s) for s in range(t - 3, t + 1)])
        out.append(np.concatenate([frame(store, r, k), frame(store, r, k) - frame(store, r, t), recent.mean(0)]))
    return np.stack(out).astype(np.float32)


def pafu(model, train_store, train, conf_store, conf, seed, steps=1000, batch=256):
    y = torch.from_numpy(pafu_teacher(model, train_store, train).reshape(-1))
    x = torch.from_numpy(np.concatenate([pafu_features(train_store, r) for r in train]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = SalienceHead(x.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(x), size=batch))
        opt.zero_grad(set_to_none=True)
        F.smooth_l1_loss(head(x[idx]), y[idx]).backward()
        opt.step()
    with torch.inference_mode():
        scores = head(torch.from_numpy(np.concatenate([pafu_features(conf_store, r) for r in conf]))).numpy()
    return scores.reshape(-1, 3).argmin(1)  # ties: argmin returns the oldest candidate
