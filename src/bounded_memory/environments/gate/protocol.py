"""Gate protocol constants and the frozen predictor as a function of memory items.

R = 4 recent frames, B = 2 memory items, horizons +3 and +6, predictor seeds 17 / 29 / 43.
The batch builders are moved unchanged from experiments/memory_baselines/gate_data.py.
"""
from __future__ import annotations

import numpy as np
import torch

R, B, HORIZONS = 4, 2, (3, 6)
SEEDS = (17, 29, 43)


def item_batch(episodes, clocks, memory_times, device):
    """Predictor inputs for [R recent frames + the given memory frames] at each clock."""
    n = R + max((len(m) for m in memory_times), default=0)
    rows = len(episodes)
    d, dim = episodes[0].p.shape[1], episodes[0].z.shape[1]
    out = {"projected_z": np.zeros((rows, n, d), np.float32), "age": np.zeros((rows, n), np.float32),
           "type": np.zeros((rows, n), np.int64), "past_actions": np.zeros((rows, n, 2), np.float32),
           "mask": np.zeros((rows, n), bool), "actions": np.zeros((rows, max(HORIZONS), 2), np.float32),
           "current": np.zeros((rows, dim), np.float32), "projected_current": np.zeros((rows, d), np.float32)}
    for i, (e, t, mem) in enumerate(zip(episodes, clocks, memory_times)):
        mem = sorted(int(m) for m in mem)
        if len(set(mem)) != len(mem) or any(m < 0 or m > t - R for m in mem):
            raise ValueError("memory frame outside the legal history")
        times = np.array(list(range(t - R + 1, t + 1)) + mem, np.int64)
        k = len(times)
        out["projected_z"][i, :k] = e.p[times]
        out["age"][i, :k] = t - times
        out["type"][i, R:k] = 1
        out["past_actions"][i, :k] = e.prefix[t] - e.prefix[times]
        out["mask"][i, :k] = True
        out["actions"][i] = e.actions[t:t + max(HORIZONS)]
        out["current"][i] = e.z[t]
        out["projected_current"][i] = e.p[t]
    batch = {k: torch.from_numpy(v).to(device) for k, v in out.items()}
    batch["z"] = torch.zeros(rows, n, 1, device=device)  # unused once projected_z is supplied
    return batch


def with_memory_tokens(batch, token_p, token_past, token_age):
    """Replace the memory part of a recent-only batch by synthetic memory items (differentiable)."""
    rows, m = token_p.shape[:2]
    recent = {k: batch[k][:, :R] for k in ("projected_z", "age", "type", "past_actions", "mask")}
    new = dict(batch)
    new["projected_z"] = torch.cat([recent["projected_z"], token_p], 1)
    new["past_actions"] = torch.cat([recent["past_actions"], token_past], 1)
    new["age"] = torch.cat([recent["age"], token_age], 1)
    new["type"] = torch.cat([recent["type"], torch.ones(rows, m, dtype=torch.long, device=token_p.device)], 1)
    new["mask"] = torch.cat([recent["mask"], torch.ones(rows, m, dtype=torch.bool, device=token_p.device)], 1)
    new["z"] = torch.zeros(rows, R + m, 1, device=token_p.device)
    return new


def pooling_attention(model, keys_p, current_p):
    """The frozen predictor's own pooling weights over `keys_p` for a query at `current_p`.

    Same operations as DynamicsPredictor.forward: learned score plus the novelty term, softmax over keys.
    keys_p [rows, K, d], current_p [rows, d] -> [rows, K].
    """
    delta = keys_p - current_p[:, None]
    score = model.score(torch.cat([keys_p, delta], -1)).squeeze(-1) + torch.log1p(delta.square().mean(-1))
    return torch.softmax(score, 1)


@torch.inference_mode()
def predict_selection(model, episodes, selected, device, chunk=256):
    preds = []
    for i in range(0, len(episodes), chunk):
        eps = episodes[i:i + chunk]
        batch = item_batch(eps, [e.q for e in eps], selected[i:i + chunk], device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)
