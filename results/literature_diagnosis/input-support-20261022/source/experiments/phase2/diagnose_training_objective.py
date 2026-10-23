"""Reproduce fixed development diagnostics of the fresh pilot's training failure.

Four epochs, original paired-batch/mode order, three objective arms and one
unit-column input-projection arm. The separate inactive-column intervention is
posthoc and evaluation-only. Original data, bases and checkpoints are read-only;
no test arrays, scorer training, deployment decision or model selection occurs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from experiments.phase2.diagnose_history import control_sample
from experiments.phase2.portable_snapshot import PortableSnapshot, state_hash
from experiments.phase2.training_samples import (
    DEFAULT_MODES, EpisodeIdentity, FeatureEpisode, SamplingConfig, SplitRegistry,
    phase1_arrays, sample_predictor,
)
from phase1.inputs import collate
from phase1.model import DynamicsPredictor, project
from phase1.training import paired_rows, regression_loss
from phase1.utils import seed_all


def components(prediction, target):
    """Exact original diagnostic float32 means and pair/difference decomposition."""
    if prediction.shape != target.shape or prediction.ndim != 3 or not len(prediction) or len(prediction) % 2:
        raise ValueError("Require matching nonempty adjacent-pair [N,H,D] tensors")
    mse = (prediction - target).square().mean()
    delta = (prediction[::2] - prediction[1::2]) - (target[::2] - target[1::2])
    pair = delta.square().mean()
    other = target.reshape(-1, 2, *target.shape[1:]).flip(1).reshape_as(target)
    own = (prediction - target).square().mean(-1)
    opposite = (prediction - other).square().mean(-1)
    separation = (target - other).square().mean(-1).clamp_min(1e-6)
    rank = torch.nn.functional.softplus(-4 * (opposite - own) / separation).mean()
    common = (((prediction[::2] + prediction[1::2]) - (target[::2] + target[1::2])) / 2).square().mean()
    return {"mse": mse, "pair_mse": pair, "rank": rank, "common_mse": common,
            "difference_mse": pair / 4, "objective": mse + 2 * pair + .5 * rank,
            "pair_choice_accuracy": (own < opposite).float().mean()}


def gradient_vector(value, model):
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([g.flatten() if g is not None else torch.zeros_like(p).flatten()
                      for g, p in zip(gradients, parameters)])


def cosine(first, second):
    return float(torch.dot(first, second) / (first.norm() * second.norm()).clamp_min(1e-20))


def unit_column_basis(basis):
    """Remove saved train-fit PCA whitening scales without fitting anything."""
    norms = np.linalg.norm(basis["projection"], axis=0, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Projection must have finite, nonzero columns")
    result = copy.deepcopy(basis)
    result["projection"] = result["projection"] / norms
    return result


def inactive_columns(train_projected, threshold=1e-4):
    if train_projected.ndim != 2 or not len(train_projected) or not np.isfinite(train_projected).all():
        raise ValueError("Require finite nonempty training-only projected inputs")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Require a positive finite threshold")
    return np.mean(train_projected ** 2, axis=0) < threshold


class AuditInputs:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.config = copy.deepcopy(snapshot.config)
        if (self.config["model"]["dropout"] != 0 or self.config["training"]["batch_size"] != 32
                or self.config["training"]["lr"] != .001 or self.config["training"]["weight_decay"] != .0001
                or self.config["training"]["clip"] != 1.):
            raise ValueError("This fixed diagnostic reproduces only the original pilot configuration")
        self.records = snapshot.records
        if any(r["split"] not in ("train", "validation") for r in self.records):
            raise ValueError("Test records are prohibited in this development diagnostic")
        self.registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in self.records])
        self.episodes, self.evidence = {}, {}
        for rec in self.records:
            self.episodes[rec["id"]] = FeatureEpisode(
                EpisodeIdentity.from_record(rec), snapshot.features(rec), snapshot.actions(rec),
                "additive_controls", snapshot.ix["signature"])
            # Privileged timestamps support coverage and intervention diagnostics
            # only; neither hidden state nor timestamps enter training tensors.
            self.evidence[rec["id"]] = snapshot.evidence(rec)

    def rows(self, task):
        return {split: [r for r in self.records if r["task"] == task and r["split"] == split]
                for split in ("train", "validation")}

    def batch(self, rows, mode, epoch, split, basis, dropout=.15):
        arrays, targets, metadata = [], [], []
        for rec in rows:
            sample = sample_predictor(self.episodes[rec["id"]], self.registry,
                SamplingConfig(memory_dropout=dropout), query=rec["query"], base_seed=17,
                epoch=epoch if split == "train" else 0, expected_split=split, history_mode=mode)
            arrays_one, target = phase1_arrays(sample)
            arrays_one["projected_z"] = project(arrays_one["z"], basis)
            arrays_one["projected_current"] = project(arrays_one["current"], basis)
            arrays.append(arrays_one)
            targets.append(target)
            metadata.append(dict(sample.metadata))
        return collate(arrays, "cpu"), torch.from_numpy(np.stack(targets)), metadata

    def extra_batch(self, rows, mode, basis):
        arrays, targets = [], []
        for rec in rows:
            episode = self.episodes[rec["id"]]
            sample, target, _ = control_sample(self.config, rec, episode.features,
                episode.actions, self.evidence[rec["id"]], basis, mode,
                sibling_features=self.episodes[rec["sibling"]].features)
            arrays.append(sample)
            targets.append(target)
        return collate(arrays, "cpu"), torch.from_numpy(np.stack(targets))

    def evaluate(self, model, rows, basis, split="validation", epoch=0):
        model.eval()
        values = []
        with torch.no_grad():
            for mode in DEFAULT_MODES:
                x, target, _ = self.batch(rows, mode, epoch, split, basis)
                values.append({k: float(v) for k, v in components(model(x), target).items()})
        return {key: float(np.mean([value[key] for value in values])) for key in values[0]}

    def new_model(self, basis):
        seed_all(17, 4)
        return DynamicsPredictor(len(self.snapshot.norm["mean"]), self.config["model"], (3, 6), basis).cpu()

    def train_four_epochs(self, task, basis, pair_weight, rank_weight):
        """Unselected diagnostic trajectory, exactly six updates per epoch."""
        rows = self.rows(task)
        pairs = paired_rows(rows["train"])
        model = self.new_model(basis)
        initial = {split: self.evaluate(model, part, basis, split) for split, part in rows.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        logs = []
        for epoch in range(4):
            model.train()
            rng = np.random.default_rng(17 + 100003 * epoch)
            order = rng.permutation(len(pairs))
            for start in range(0, len(order), 16):
                part = [rec for i in order[start:start + 16] for rec in pairs[i]]
                for mode in rng.permutation(DEFAULT_MODES):
                    x, target, _ = self.batch(part, str(mode), epoch, "train", basis)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _ = regression_loss(model(x), target, pair_weight, rank_weight)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
            logs.append({"epoch": epoch + 1,
                         "train_fixed_epoch_zero": self.evaluate(model, rows["train"], basis, "train"),
                         "validation": self.evaluate(model, rows["validation"], basis)})
        return initial, logs


def objective_task(inputs, task, basis):
    rows = inputs.rows(task)
    model = inputs.new_model(basis)
    result = {"initial": {split: inputs.evaluate(model, part, basis, split) for split, part in rows.items()}}
    logged = inputs.snapshot.training_log(task)
    result["initial_validation_mse_abs_replay_error"] = abs(result["initial"]["validation"]["mse"]
                                                           - np.mean(list(logged["initial_validation_by_mode"].values())))
    for kind in ("frozen", "latest"):
        saved, _ = inputs.snapshot.model(task, kind)
        before = state_hash(saved)
        result[kind] = {split: inputs.evaluate(saved, part, basis, split) for split, part in rows.items()}
        if state_hash(saved) != before:
            raise ValueError("Read-only checkpoint evaluation changed state")
    support = {}
    for split, part in rows.items():
        targets = np.stack([inputs.episodes[r["id"]].features[r["query"] + np.array([3, 6])] for r in part])
        current = np.stack([inputs.episodes[r["id"]].features[r["query"]] for r in part])[:, None]
        residual = targets - current - basis["target_mean"]
        output_basis = basis["output_basis"]
        projected = (residual @ output_basis) @ output_basis.T
        pair_delta = targets[::2] - targets[1::2]
        pair_projected = (pair_delta @ output_basis) @ output_basis.T
        visual = np.concatenate([project(inputs.episodes[r["id"]].features[:r["query"] + 1], basis) for r in part])
        support[split] = {"persistence_mse": float(np.mean((targets - current) ** 2)),
            "current_plus_train_mean_mse": float(np.mean(residual ** 2)),
            "oracle_in_output_basis_mse": float(np.mean((residual - projected) ** 2)),
            "output_residual_variance_fraction_representable": float(np.sum(projected ** 2) / np.sum(residual ** 2)),
            "pair_difference_fraction_representable": float(np.sum(pair_projected ** 2) / np.sum(pair_delta ** 2)),
            "projected_input_rms": float(np.sqrt(np.mean(visual ** 2))),
            "projected_input_maxabs": float(np.max(np.abs(visual)))}
    result["basis_support"] = support
    # Differentiate the average of six original training modes. Validation uses
    # no dropout; training uses exactly the original epoch-zero dropout draws.
    model.train()
    gradients, values = {}, {}
    for split, part in rows.items():
        components_by_mode = []
        for mode in DEFAULT_MODES:
            x, target, _ = inputs.batch(part, mode, 0, split, basis)
            components_by_mode.append(components(model(x), target))
        average = {key: sum(c[key] for c in components_by_mode) / len(components_by_mode)
                   for key in ("mse", "pair_mse", "rank")}
        gradients[split] = {key: gradient_vector(value, model) for key, value in average.items()}
        values[split] = {key: float(value.detach()) for key, value in average.items()}
    gm, gp, gr = (gradients["train"][key] for key in ("mse", "pair_mse", "rank"))
    gv = gradients["validation"]["mse"]
    combined = gm + 2 * gp + .5 * gr
    result["initial_gradients"] = {"components": values,
        "norms": {"mse": float(gm.norm()), "weighted_pair": float((2 * gp).norm()),
                  "weighted_rank": float((.5 * gr).norm()), "combined": float(combined.norm())},
        "cosines": {"train_mse_pair": cosine(gm, gp), "train_mse_rank": cosine(gm, gr),
                    "train_mse_validation_mse": cosine(gm, gv), "train_pair_validation_mse": cosine(gp, gv),
                    "train_rank_validation_mse": cosine(gr, gv), "train_combined_validation_mse": cosine(combined, gv)}}
    coverage = {}
    for mode in DEFAULT_MODES:
        events = []
        for epoch in range(4):
            _, _, metadata = inputs.batch(rows["train"], mode, epoch, "train", basis)
            for rec, meta in zip(rows["train"], metadata):
                indices = meta["history_after_dropout"]
                z, sibling = inputs.episodes[rec["id"]].features, inputs.episodes[rec["sibling"]].features
                different = [i for i in indices if not np.array_equal(z[i], sibling[i])]
                events.append({"paired_difference_observable": bool(different),
                    "all_evidence_retained": set(inputs.evidence[rec["id"]]).issubset(indices),
                    "history_count": len(indices), "dropout": meta["dropout_event"]})
        coverage[mode] = {key: float(np.mean([e[key] for e in events])) for key in events[0]}
    result["training_coverage"] = coverage
    arms = {}
    for arm, pw, rw in (("original", 2., .5), ("no_rank", 2., 0.), ("mse_only", 0., 0.)):
        _, arms[arm] = inputs.train_four_epochs(task, basis, pw, rw)
        print(json.dumps({"task": task, "arm": arm,
                          "validation_mse": [row["validation"]["mse"] for row in arms[arm]]}), flush=True)
    result["four_epoch_arms"] = arms
    return result


def input_basis_task(inputs, task, basis):
    projection = basis["projection"]
    scales = np.linalg.norm(projection, axis=0)
    unit_projection = unit_column_basis(basis)["projection"]
    arrays = {}
    for split, part in inputs.rows(task).items():
        visual = np.concatenate([inputs.episodes[r["id"]].features[:r["query"] + 1] for r in part])
        centered = visual - basis["input_mean"]
        projected, raw = centered @ projection, centered @ unit_projection
        arrays[split] = {"projected": projected, "raw": raw, "norm": np.mean(centered ** 2),
                         "second_moment": np.mean(projected ** 2, axis=0)}
    train, val = arrays["train"], arrays["validation"]
    inactive = inactive_columns(train["projected"])
    return {"projection_scale_min": float(scales.min()), "projection_scale_max": float(scales.max()),
            "nearly_inactive_training_dimensions": int(inactive.sum()),
            "validation_projected_energy_fraction_from_nearly_inactive_training_dimensions":
                float(np.sum(val["projected"][:, inactive] ** 2) / np.sum(val["projected"] ** 2)),
            "train_unwhitened_projected_rms": float(np.sqrt(np.mean(train["raw"] ** 2))),
            "validation_unwhitened_projected_rms": float(np.sqrt(np.mean(val["raw"] ** 2))),
            "train_feature_centered_rms": float(np.sqrt(train["norm"])),
            "validation_feature_centered_rms": float(np.sqrt(val["norm"])),
            "validation_whitened_mean_shift_energy_fraction":
                float(np.sum(val["projected"].mean(axis=0) ** 2) / np.sum(np.mean(val["projected"] ** 2, axis=0))),
            "per_dimension_train_secondmoment": train["second_moment"].tolist(),
            "per_dimension_validation_secondmoment": val["second_moment"].tolist()}


def masked_input_task(inputs, task, basis):
    rows = inputs.rows(task)
    projected = np.concatenate([project(inputs.episodes[r["id"]].features[:r["query"] + 1], basis) for r in rows["train"]])
    inactive = inactive_columns(projected)
    altered = copy.deepcopy(basis)
    altered["projection"][:, inactive] = 0
    model, _ = inputs.snapshot.model(task, "frozen")
    initial_state = state_hash(model)
    result = {"masked_dimensions": int(inactive.sum()), "original": {}, "masked": {}}
    for arm, arm_basis in (("original", basis), ("masked", altered)):
        for split, part in rows.items():
            by_mode = {}
            with torch.inference_mode():
                for mode in DEFAULT_MODES + ("recent_only", "full_history", "privileged_small_budget", "wrong_evidence"):
                    if mode in DEFAULT_MODES:
                        x, target, _ = inputs.batch(part, mode, 0, split, arm_basis)
                    else:
                        x, target = inputs.extra_batch(part, mode, arm_basis)
                    by_mode[mode] = {key: float(value) for key, value in components(model(x), target).items()}
            result[arm][split] = {"six_mode_average": {key: float(np.mean([by_mode[m][key] for m in DEFAULT_MODES]))
                                  for key in by_mode[DEFAULT_MODES[0]]}, "by_mode": by_mode}
    if state_hash(model) != initial_state:
        raise ValueError("Posthoc input intervention modified original predictor state")
    result["model_state_bitwise_unchanged"] = True
    return result


def run(snapshot, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refuse to overwrite an existing diagnostic result directory")
    output.mkdir(parents=True, exist_ok=True)
    before = snapshot.audit()
    started = time.monotonic()
    inputs = AuditInputs(snapshot)
    objective = {"purpose": "Fixed exploratory objective/capacity/gradient diagnostic; no test arrays or model selection",
                 "torch": torch.__version__, "device": "cpu", "tasks": {}}
    basis_audit = {}
    unwhitened = {"purpose": "Unit-normalize saved train-fit PCA columns only; output basis unchanged; exploratory, not adopted",
                  "torch": torch.__version__, "tasks": {}}
    masked = {"purpose": "Posthoc development input intervention on frozen-best; training-only near-inactive columns masked, not adopted or confirmatory",
              "threshold": 1e-4, "tasks": {}}
    for task in ("gate", "motion"):
        basis = snapshot.basis(task)
        objective["tasks"][task] = objective_task(inputs, task, basis)
        basis_audit[task] = input_basis_task(inputs, task, basis)
        initial, logs = inputs.train_four_epochs(task, unit_column_basis(basis), 2., .5)
        unwhitened["tasks"][task] = {"initial": initial, "logs": logs,
            "original_whitened_initial": objective["tasks"][task]["initial"],
            "original_whitened_four_epoch_arm": objective["tasks"][task]["four_epoch_arms"]["original"]}
        masked["tasks"][task] = masked_input_task(inputs, task, basis)
    reports = {"objective_audit.json": objective, "input_basis_audit.json": basis_audit,
               "no_whitening_ablation.json": unwhitened, "mask_inactive_diagnostic.json": masked}
    for filename, value in reports.items():
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    sources = [Path(__file__), ROOT / "experiments/phase2/portable_snapshot.py",
               ROOT / "experiments/phase2/diagnose_history.py", ROOT / "experiments/phase2/training_samples.py"]
    sources += [ROOT / "experiments/predictor/control/src/phase1" / name for name in ("model.py", "inputs.py", "training.py")]
    manifest = {"status": "completed_fixed_training_objective_diagnostic",
                "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
                "device": "cpu", "torch_version": torch.__version__, "test_arrays_opened": 0,
                "numpy_version": np.__version__,
                "numpy_blas_build": getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas", {}),
                "numeric_thread_environment": {name: os.environ.get(name) for name in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                "protocol": {"epochs": 4, "seed": 17, "threads": 4, "pair_batch_size": 16,
                    "mode_order": "numpy default_rng(17+100003*epoch), original pair permutation then mode permutation",
                    "training_dropout": .15, "evaluation_train_dropout": "fixed original epoch-zero draws",
                    "validation_dropout": 0., "objective_arms": [["original", 2., .5], ["no_rank", 2., 0.], ["mse_only", 0., 0.]],
                    "same_architecture_and_train_only_basis": True,
                    "unit_projection_arm": "only rescale input PCA columns to norm one; original objective",
                    "masking": "posthoc train-derived threshold1e-4; frozen model state unchanged",
                    "no_checkpoint_selection_no_original_mutation": True,
                    "all_results_exploratory_reused_development_validation": True},
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "report_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in reports},
                "artifact_audit_before": before, "artifact_audit_after": snapshot.audit()}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(PortableSnapshot(args.snapshot), args.output)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
