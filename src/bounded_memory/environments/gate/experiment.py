"""What every Gate script shares: open the asset tree named by a config and score a policy.

A Gate config (configs/gate/*.yaml) names one asset tree and how its predictors are verified:

    assets:
      root: <tree with resolved_config.json>
      predictor_sha256: team | {17: ..., 29: ..., 43: ...} | <json file written by training>
      students: <folder with student_<seed>.pt>          (optional)
      reference_rows: <validation_results.json>           (optional, the team's original output)
      dinov2: <features.npz>                              (optional, FrameMorrow's visual features)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...evaluation.metrics.gate import metric_rows
from . import bundle
from .assets import load_gate
from .protocol import predict_selection

ORIGINAL_NAMES = {"oracle_keyframe": "manual_keyframe"}   # this package's name -> the 29 Oct run's name


def open_assets(config, device):
    spec = config["assets"]
    pins, basis = spec.get("predictor_sha256"), None
    if pins == "team":
        pins, basis = bundle.TEAM_PREDICTOR_SHA256, bundle.TEAM_BASIS_SHA256
    elif isinstance(pins, str):
        pins = json.loads(Path(pins).read_text())
    if not pins:
        raise ValueError("the config must say how predictor checkpoints are verified (assets.predictor_sha256)")
    return load_gate(spec["root"], device, predictor_sha256={int(k): v for k, v in pins.items()}, basis_sha256=basis)


class Scorer:
    """Collects metric rows for one predictor seed on the validation episodes."""

    def __init__(self, assets, seed, task):
        self.assets, self.seed, self.task = assets, seed, task
        self.model = assets.predictor(seed)
        self.train, self.val = assets.split("train"), assets.split("validation")
        self.rows = []

    def add(self, policy, selected=None, prediction=None):
        if prediction is None:
            prediction = predict_selection(self.model, self.val, selected, self.assets.device)
        new = metric_rows(self.assets, self.val, prediction, selected, policy, self.seed)
        self.rows.extend(new)
        self.task.log(f"{policy:32s} route {np.mean([r['route_accuracy'] for r in new]):.4f}  "
                      f"mse {np.mean([r['normalized_mse'] for r in new]):.5f}")

    def payload(self):
        return {"environment": "gate", "validation_episodes": len(self.val), "train_episodes": len(self.train),
                "rows": self.rows}


def reference_check(path, seed, rows):
    """Compare this run with the team's original validation output, row by row (same seed, episode, horizon)."""
    original = {}
    for r in json.loads(Path(path).read_text())["rows"]:
        if r["seed"] == seed:
            original[(r["condition"], r["episode"], r["horizon"])] = (r["state_accuracy"], r["normalized_mse"])
    out = {}
    for policy in sorted({r["policy"] for r in rows}):
        theirs = ORIGINAL_NAMES.get(policy, policy)
        pairs = [(r, original[(theirs, r["episode"], r["horizon"])]) for r in rows
                 if r["policy"] == policy and (theirs, r["episode"], r["horizon"]) in original]
        if not pairs:
            continue
        out[policy] = {"rows": len(pairs),
                       "route_mismatches": int(sum(r["route_accuracy"] != o[0] for r, o in pairs)),
                       "max_abs_mse_diff": float(max(abs(r["normalized_mse"] - o[1]) for r, o in pairs)),
                       "original_route_accuracy": float(np.mean([o[0] for _, o in pairs])),
                       "this_run_route_accuracy": float(np.mean([r["route_accuracy"] for r, _ in pairs]))}
    return out
