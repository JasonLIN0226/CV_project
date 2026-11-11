"""Motion v2: recompute the published tables from the loss arrays tracked in the repository.

The frozen Motion v2 readers, features and dataset are not available to this account, so no
Motion policy can be run again. What is tracked is the evidence of the completed study:

    step 3/4 (17925)   online/{ordinary,long_gap}/raw_losses/<kind>_<method>.npz
                       position, feature [student-or-RNG seed 3, reader seed 3, scene pair 96, sibling 2, horizon 2]
                       success [seed 3, scene pair 96, sibling 2]
                       students/<arm>_seed<s>_history.json   which training step each student kept
    step 2 (v2.3.1)    metrics/pair_averaged_losses.npz
                       position, feature [scene pair 96, condition a|b|c|d, horizon 2]

The published numbers are plain means of these arrays (siblings, seeds and pairs averaged), which
is what this module recomputes. Deterministic baselines are stored once per seed slot; those
copies are not independent samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HORIZONS = (3, 6)
CONDITIONS = {"a": "recent frames only", "b": "first new-segment frame + end (the two cue frames)",
              "c": "first and last old-segment frames", "d": "last old-segment frame + end"}


def online_table(step34, interval):
    """method -> metrics for one interval ("ordinary" or "long_gap")."""
    folder = Path(step34) / "online" / interval / "raw_losses"
    if not folder.is_dir():
        raise FileNotFoundError(f"Motion v2 evidence missing: {folder}")
    out = {}
    for path in sorted(folder.glob("*.npz")):
        kind, name = path.stem.split("_", 1)
        with np.load(path, allow_pickle=False) as z:
            if z["position"].shape != (3, 3, 96, 2, 2) or z["success"].shape != (3, 96, 2):
                raise ValueError(f"unexpected raw-loss shape in {path}")
            position, feature = z["position"].mean(axis=(0, 1, 2, 3)), z["feature"].mean(axis=(0, 1, 2, 3))
            success = float(z["success"].mean())
        out[f"{kind}/{name}"] = {
            "kind": kind, "method": name, "success_rate": success,
            "position_mse": {str(h): float(position[i]) for i, h in enumerate(HORIZONS)},
            "feature_mse": {str(h): float(feature[i]) for i, h in enumerate(HORIZONS)}}
    return out


def reference_arms(step2):
    """condition -> metrics from the step-2 reader qualification (ordinary interval)."""
    path = Path(step2) / "metrics/pair_averaged_losses.npz"
    if not path.exists():
        raise FileNotFoundError(f"Motion v2 evidence missing: {path}")
    with np.load(path, allow_pickle=False) as z:
        conditions, horizons = [str(c) for c in z["conditions"]], [int(h) for h in z["horizons"]]
        position, feature = z["position"].mean(0), z["feature"].mean(0)
    if tuple(horizons) != HORIZONS:
        raise ValueError("unexpected horizons in the step-2 evidence")
    return {c: {"history": CONDITIONS[c],
                "position_mse": {str(h): float(position[i, j]) for j, h in enumerate(horizons)},
                "feature_mse": {str(h): float(feature[i, j]) for j, h in enumerate(horizons)}}
            for i, c in enumerate(conditions)}


def student_selection(step34):
    """arm -> per-seed record of the checkpoint each student kept (step 0 means untrained initial weights)."""
    out = {}
    for path in sorted((Path(step34) / "students").glob("*_history.json")):
        meta = json.loads(path.read_text())["metadata"]
        arm = f"{meta['architecture']}/{meta['label']}"
        out.setdefault(arm, {})[str(meta["seed"])] = {
            "selected_step": int(meta["selected_step"]), "is_trained": bool(meta["is_trained"]),
            "updates_executed": int(meta["training_updates_executed"]),
            "selected_validation_cross_entropy": float(meta["selected_validation_label_cross_entropy"])}
    return out
