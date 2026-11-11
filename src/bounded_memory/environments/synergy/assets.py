"""Synergy (terminal R/D/S) assets: records, features, the frozen readers and the in-span loss.

Each record is one terminal decision: four recent frames and three candidate cards (at query-6,
-5 and -4), of which two may be kept. R is redundancy, D is decoy, S is XOR synergy.

`Store` methods, `KEEPS`, `batch_tensors`, `coefficients`, `pooled_attention`, `teacher_attention`
and `qform` are extracted unchanged from
results/literature_diagnosis/qualified-reader-retention-20261025/code/pipe.py, which the active
code used to import from results/. The store takes its paths explicitly instead of from
environment variables, so train and confirmation sets can coexist in one process. `reader_batch`,
`reader_coefficients` and `head_attention` are moved from experiments/memory_baselines.

Reader checkpoints hold pickled state. They are loaded only when their SHA256 equals the value
the release's own reader_locked.json records, pinned below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ...data.io import require_sha256
from ...predictors.readers import build_reader

H = (3, 6)
DIM = 5120
READERS = ("multihead", "item_attention")
TASKS = "RDS"
READER_SHA256 = {   # reader-aggregation-20261025, readers/<name>/reader_locked.json
    "multihead": "fb0ea31b20e5aa53c4f3adbaa1b261d82826cda068ad23fa5d1a9edd56221543",
    "item_attention": "15428e52128c16887238bf4a424691b0785966001c680a0a409f13cf8026a597",
}
BASIS_SHA256 = "922bc4a676893d0ba5974cd20dbcac09c461bc908549966f197dfbbe38f4f05d"


class Store:
    """Records plus TRAIN-projected features for one set of splits."""

    def __init__(self, records, actions, xfer, run):
        records, xfer, run = Path(records), Path(xfer), Path(run)
        self.records = json.loads(records.read_text())
        self.by_id = {r["id"]: r for r in self.records}
        self.actions = np.load(actions)
        self.banks = dict(np.load(run / "data/action_banks.npz"))
        require_sha256(run / "basis.npz", BASIS_SHA256, "reader basis")
        b = np.load(run / "basis.npz")
        self.basis = {k: b[k] for k in b.files}
        self.P = np.load(xfer / "projected_features.npy")          # [U,128] float32
        self.FB = np.load(xfer / "output_basis_coords.npy").astype(np.float64)   # [U,128], F @ B
        self.tmB = np.load(xfer / "target_mean_coords.npy")         # [128]
        B = self.basis["output_basis"].astype(np.float64)
        self.G = B.T @ B                                            # ~identity

    def split(self, name):
        return [r for r in self.records if r["split"] == name]

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


def load_reader(name, store, agg, sha256=None):
    """One frozen reader from a reader folder `agg` (readers/<name>/reader_selected.pt).

    `sha256` defaults to the pinned checksum of the released reader. A reader trained in a run is
    loaded by passing the checksum that its own reader_locked.json recorded when it was written.
    """
    path = require_sha256(Path(agg) / f"readers/{name}/reader_selected.pt", sha256 or READER_SHA256[name], "reader checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_reader(DIM, state["model_config"], H, store.basis, state.get("variant", name), state.get("heads", 4))
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


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


def frame(store, r, t):
    return store.P[r["rgb_indices"][t]]


def reader_batch(caps, plans, memory=None):
    """Reader inputs for recent frames + either all three candidates or synthetic memory items."""
    n = len(caps)
    recent = {k: torch.from_numpy(np.stack([c[k][:4] for c in caps])) for k in ("pz", "age", "past")}
    if memory is None:
        extra = {k: torch.from_numpy(np.stack([c[k][4:] for c in caps])) for k in ("pz", "age", "past")}
    else:
        extra = dict(zip(("pz", "past", "age"), memory))
    m = 4 + extra["pz"].shape[1]
    return {"z": torch.zeros(n, m, 1), "projected_z": torch.cat([recent["pz"], extra["pz"]], 1),
            "age": torch.cat([recent["age"], extra["age"]], 1),
            "past_actions": torch.cat([recent["past"], extra["past"]], 1),
            "type": torch.tensor([[0] * 4 + [1] * (m - 4)] * n), "mask": torch.ones(n, m, dtype=torch.bool),
            "actions": torch.as_tensor(np.asarray(plans, np.float32)), "current": torch.zeros(n, 1),
            "projected_current": torch.from_numpy(np.stack([c["pcur"] for c in caps]))}


def reader_coefficients(model, batch):
    """Differentiable version of pipe.coefficients: output-basis coefficients [N, 2, 128]."""
    captured = []
    hook = model.output.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        model(batch)
    finally:
        hook.remove()
    if len(captured) != 2:
        raise RuntimeError("expected one readout per horizon")
    return torch.stack(captured, 1)


def head_attention(model, batch):
    """Pooling weights over the three candidates, one row per attention path: [N, paths, 3]."""
    return model.attention_weights(batch)[:, :, 4:]
