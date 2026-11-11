"""Synergy (terminal R/D/S) assets and metrics, on top of the frozen project pipeline.

Each record is one terminal decision: four recent frames and three candidates (query-6, -5, -4),
of which two may be kept. Stores, capsules and the in-span loss come unchanged from
results/literature_diagnosis/qualified-reader-retention-20261025/code/pipe.py; the metric
follows that folder's evaluate_dev.py.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "results/literature_diagnosis/qualified-reader-retention-20261025/code"
for _p in (CODE, ROOT, ROOT / "experiments/predictor/control/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import pipe  # noqa: E402

READERS = ("multihead", "item_attention")
TASKS = "RDS"


class Store(pipe.Store):
    """pipe.Store with explicit paths, so the train and confirmation sets can coexist in one process."""

    def __init__(self, records, actions, xfer, run):
        records, xfer, run = Path(records), Path(xfer), Path(run)
        self.records = json.loads(records.read_text())
        self.by_id = {r["id"]: r for r in self.records}
        self.actions = np.load(actions)
        self.banks = dict(np.load(run / "data/action_banks.npz"))
        b = np.load(run / "basis.npz")
        self.basis = {k: b[k] for k in b.files}
        self.P = np.load(xfer / "projected_features.npy")
        self.FB = np.load(xfer / "output_basis_coords.npy").astype(np.float64)
        self.tmB = np.load(xfer / "target_mean_coords.npy")
        B = self.basis["output_basis"].astype(np.float64)
        self.G = B.T @ B

    def split(self, name):
        return [r for r in self.records if r["split"] == name]


def load_reader(name, store, agg):
    state = torch.load(Path(agg) / f"readers/{name}/reader_selected.pt", map_location="cpu", weights_only=False)
    model = pipe.build_reader(pipe.DIM, state["model_config"], pipe.H, store.basis,
                              state.get("variant", name), state.get("heads", 4))
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


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


def family_mean(values, records, index):
    groups = defaultdict(list)
    for i in index:
        groups[records[i]["family"]].append(values[i])
    return float(np.mean([np.mean(v) for v in groups.values()]))


def score(records, cost, drops=None, losses=None):
    """Per task: needed-evidence loss and prediction regret, as in evaluate_dev.py.

    drops: list of per-seed drop arrays (deletion policies). losses: list of per-seed [N] expected
    losses (memories that are not a deletion, e.g. compressed tokens); regret is measured against the
    best single deletion, so it can be negative.
    """
    correct = [set(r["semantic_correct_delete_set"]) for r in records]
    n = len(records)
    if drops is not None:
        lost = np.mean([[float(d[i] not in correct[i]) for i in range(n)] for d in drops], 0)
        regret = np.mean([[cost[i, d[i]] - cost[i].min() for i in range(n)] for d in drops], 0)
    else:
        lost = None
        regret = np.mean([np.asarray(l) - cost.min(1) for l in losses], 0)
    out = {}
    for task in TASKS:
        index = [i for i, r in enumerate(records) if r["task_family"] == task]
        out[task] = {"n": len(index),
                     "needed_evidence_lost": None if lost is None else float(lost[index].mean()),
                     "regret": family_mean(regret, records, index)}
    return out
