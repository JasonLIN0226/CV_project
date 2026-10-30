"""CPU pipeline for terminal R/D/S retention on qualified readers.

Mirrors experiments/phase2/run_terminal_structures.py (causal_capsule, reader_sample,
causal_features, medoid/full_set_sensitivity) but feeds the reader TRAIN-projected
features that were computed from the hash-verified local feature cache. Raw 5120-d
features are never needed: the reader consumes projected_z/projected_current, and every
loss/distance we use is a within-parent comparison inside the output-basis span.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from path_config import REPO, RUN, XFER, AGG, RECORDS, ACTIONS
sys.path[:0] = [str(REPO), str(REPO / "experiments/predictor/control/src")]
from experiments.phase2.reader_aggregation import build_reader  # noqa: E402

H = (3, 6)
DIM = 5120


class Store:
    def __init__(self):
        self.records = json.load(open(RECORDS))
        self.by_id = {r["id"]: r for r in self.records}
        self.actions = np.load(ACTIONS)
        self.banks = {k: v for k, v in np.load(RUN / "data/action_banks.npz").items()}
        b = np.load(RUN / "basis.npz")
        self.basis = {k: b[k] for k in b.files}
        self.P = np.load(XFER / "projected_features.npy")          # [U,128] float32
        self.FB = np.load(XFER / "output_basis_coords.npy")         # [U,128] float64, F @ B
        self.tmB = np.load(XFER / "target_mean_coords.npy")         # [128]
        B = self.basis["output_basis"].astype(np.float64)
        self.G = B.T @ B                                            # ~identity

    def capsule(self, r):
        q = r["query"]
        times = np.r_[np.arange(q - 3, q + 1), np.arange(q - 6, q - 3)]
        rgb = np.asarray(r["rgb_indices"])
        ac = self.actions[r["id"]][:q].astype(np.float32)
        prefix = np.concatenate([np.zeros((1, 2), np.float32), ac.cumsum(0)])
        return {"pz": self.P[rgb[times]], "age": (q - times).astype(np.float32),
                "past": (prefix[q] - prefix[times]).astype(np.float32), "pcur": self.P[rgb[q]],
                "rgb_times": rgb[times], "rgb_cur": rgb[q]}

    def student_features(self, r):
        cap = self.capsule(r)
        p, cur = cap["pz"], cap["pcur"]
        items = np.concatenate([p[4:], p[4:] - cur, np.log1p(cap["age"][4:, None]), cap["past"][4:]], 1)
        context = np.concatenate([p[:4].reshape(-1), cap["age"][:4], cap["past"][:4].reshape(-1)])
        return items.astype(np.float32), context.astype(np.float32)

    def target_residual(self, r, bank):
        """B^T(target - current - target_mean) per plan/horizon: [plans,2,128]."""
        cur = r["rgb_indices"][r["query"]]
        tgt = np.asarray(r["plan_targets"][bank])                   # [plans,2] unique-frame ids
        return self.FB[tgt] - self.FB[cur][None, None] - self.tmB[None, None]

    def actual_target_residual(self, r):
        rgb = np.asarray(r["rgb_indices"]); q = r["query"]
        return (self.FB[rgb[q + np.asarray(H)]] - self.FB[rgb[q]] - self.tmB)[None]


def load_reader(name, store):
    state = torch.load(AGG / f"readers/{name}/reader_selected.pt", map_location="cpu", weights_only=False)
    model = build_reader(DIM, state["model_config"], H, store.basis, state.get("variant", name), state.get("heads", 4))
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    return model


KEEPS = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [0, 1, 2]}


def batch_tensors(caps, keep, plans):
    """caps: list of capsules; plans: [N,6,2] (one plan per sample)."""
    ix = [0, 1, 2, 3] + [4 + i for i in keep]
    n = len(caps)
    pz = np.stack([c["pz"][ix] for c in caps])
    age = np.stack([c["age"][ix] for c in caps])
    past = np.stack([c["past"][ix] for c in caps])
    pcur = np.stack([c["pcur"] for c in caps])
    m = len(ix)
    return {"z": torch.zeros((n, m, 1)), "projected_z": torch.from_numpy(pz), "age": torch.from_numpy(age),
            "past_actions": torch.from_numpy(past), "type": torch.from_numpy(np.asarray([[0] * 4 + [1] * len(keep)] * n, np.int64)),
            "mask": torch.ones((n, m), dtype=torch.bool), "actions": torch.from_numpy(np.asarray(plans, np.float32)),
            "current": torch.zeros((n, 1)), "projected_current": torch.from_numpy(pcur)}


def coefficients(model, caps, keep, plans, chunk=4096):
    out = []
    for a in range(0, len(caps), chunk):
        cap = []
        hook = model.output.register_forward_hook(lambda mod, i, o: cap.append(o.detach().numpy()))
        try:
            with torch.inference_mode():
                model(batch_tensors(caps[a:a + chunk], keep, plans[a:a + chunk]))
        finally:
            hook.remove()
        assert len(cap) == 2
        out.append(np.stack(cap, 1))
    return np.concatenate(out).astype(np.float64)                    # [N,2,128]


def pooled_attention(model, caps):
    """Head-mean pooling weights over the three candidates with the full K+1 set at query."""
    b = batch_tensors(caps, [0, 1, 2], np.zeros((len(caps), 6, 2), np.float32))
    with torch.inference_mode():
        if hasattr(model, "attention_weights"):
            w = model.attention_weights(b).mean(1)                   # [N, items]
        else:
            p, q = b["projected_z"], b["projected_current"]
            hist = b["type"] == 1
            s = model.score(torch.cat([p, p - q[:, None]], -1)).squeeze(-1) + torch.log1p((p - q[:, None]).square().mean(-1))
            w = torch.softmax(s.masked_fill(~hist, -1e4), 1) * hist
            w = w / w.sum(1, keepdim=True)
    return w.numpy()[:, 4:].astype(np.float64)


def teacher_attention(model, caps, query_proj):
    """Future-attention teacher: pooling weights with future frames as the query.
    query_proj: [N,Q,128]; returns mean over queries [N,3]. Ages kept at observed values."""
    n, qn = query_proj.shape[:2]
    rep = [c for c in caps for _ in range(qn)]
    b = batch_tensors(rep, [0, 1, 2], np.zeros((len(rep), 6, 2), np.float32))
    b["projected_current"] = torch.from_numpy(query_proj.reshape(n * qn, -1).astype(np.float32))
    with torch.inference_mode():
        if hasattr(model, "attention_weights"):
            w = model.attention_weights(b).mean(1)
        else:
            p, q = b["projected_z"], b["projected_current"]
            hist = b["type"] == 1
            s = model.score(torch.cat([p, p - q[:, None]], -1)).squeeze(-1) + torch.log1p((p - q[:, None]).square().mean(-1))
            w = torch.softmax(s.masked_fill(~hist, -1e4), 1) * hist
            w = w / w.sum(1, keepdim=True)
    return w.numpy()[:, 4:].reshape(n, qn, 3).mean(1).astype(np.float64)


def qform(d, G):
    """mean over horizons and 5120 dims of squared forecast difference; d [...,2,128]."""
    return np.einsum("...hr,rs,...hs->...", d, G, d) / (d.shape[-2] * DIM)
