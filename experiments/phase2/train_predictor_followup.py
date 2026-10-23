"""Three predeclared train-only projection diagnostics for the fresh predictor.

This bounded development followup retains the supplied architecture, original
output basis and paired objective. It is not a hyperparameter search, a repaired
teammate-model reproduction, or evidence that a learned retention policy works.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import os
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from experiments.phase2.diagnose_history import (
    CONTROL_CONDITIONS, control_sample, validate_matched_sibling,
)
from experiments.phase2.portable_snapshot import PortableSnapshot, state_hash
from experiments.phase2.training_samples import (
    DEFAULT_MODES, EpisodeIdentity, FeatureEpisode, SamplingConfig, SplitRegistry,
    phase1_arrays, sample_predictor,
)
from experiments.phase2.train_predictor_pilot import sha256
from phase1.inputs import collate
from phase1.model import DynamicsPredictor, project
from phase1.training import paired_rows, regression_loss
from phase1.utils import seed_all

ARMS = ("original_projection", "unit_columns", "train_support_mask")
METRICS = ("mse", "common_mse", "difference_mse", "pair_mse", "rank_loss", "combined_loss")
PROTOCOL = {
    "name": "three_fixed_projection_arms_v1", "tasks": ["gate", "motion"],
    "arms": list(ARMS), "seed": 17, "epochs": 40, "batch_size": 32,
    "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1.,
    "lr_step": 40, "lr_gamma": .5, "threads": 4, "timeout_seconds": 600,
    "memory_dropout": .15, "training_modes": list(DEFAULT_MODES),
    "pair_weight": 2., "rank_weight": .5, "inactive_train_secondmoment_threshold": 1e-4,
    "selection": "earliest minimum plain equal-six-mode validation MSE, including epoch zero",
    "evaluation_splits": ["train", "validation"], "test_arrays_permitted": False,
    "training_metric_evaluation": "fixed epoch-zero samples with dropout disabled, distinct from online train loss",
    "controls": list(CONTROL_CONDITIONS), "control_evaluation": ["selected", "terminal"],
    "all_arms_exploratory": True, "adopt_best_arm": False,
    "limitations": ["Same small reused validation split; no independent generalization evidence.",
                    "Full/privileged/sibling controls are evaluator-only and do not enter training.",
                    "Full history exceeds this K=2 training mixture's memory budget.",
                    "Epoch-zero selection means no learned improvement under the selection metric."],
}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def loss_components(prediction, target):
    """Keep paired common/difference errors explicit under the original loss."""
    if prediction.shape != target.shape or prediction.ndim != 3 or len(prediction) % 2:
        raise ValueError("Require equally shaped paired [episode,horizon,feature] tensors")
    error = prediction - target
    common = ((error[::2] + error[1::2]) / 2).square().mean()
    difference = ((error[::2] - error[1::2]) / 2).square().mean()
    total, mse, pair_mse = regression_loss(prediction, target, 2., .5)
    # Compute directly, rather than cancelling terms in a large scalar loss.
    other = target.reshape(-1, 2, *target.shape[1:]).flip(1).reshape_as(target)
    own_error = error.square().mean(-1)
    other_error = (prediction - other).square().mean(-1)
    separation = (target - other).square().mean(-1).clamp_min(1e-6)
    rank = torch.nn.functional.softplus(-4 * (other_error-own_error) / separation).mean()
    return dict(zip(METRICS, (mse, common, difference, pair_mse, rank, total)))


def projection_variant(basis, train_prefixes, arm, threshold=1e-4):
    """Fit only the projection diagnostic; leave all other fitted arrays intact."""
    if arm not in ARMS or not train_prefixes:
        raise ValueError("Require one fixed arm and nonempty training prefixes")
    result = {k: v.copy() for k, v in basis.items()}
    prefix = np.concatenate(train_prefixes)
    if prefix.ndim != 2 or not np.isfinite(prefix).all():
        raise ValueError("Training prefixes must be finite feature matrices")
    projection = result["projection"]
    norms = np.linalg.norm(projection, axis=0)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Projection columns must have finite nonzero norms")
    second = np.mean(project(prefix, basis).astype(np.float64) ** 2, axis=0)
    inactive = second < threshold
    if arm == "unit_columns":
        result["projection"] = (projection / norms[None]).astype(np.float32)
    elif arm == "train_support_mask":
        result["projection"][:, inactive] = 0.
    return result, {"arm": arm, "fit_split": "train", "train_frames": len(prefix),
                    "threshold": threshold, "inactive_dimensions": int(inactive.sum()),
                    "masked_dimensions": int(inactive.sum()) if arm == "train_support_mask" else 0,
                    "original_column_norms": norms.tolist(),
                    "training_projected_second_moment": second.tolist()}


class Samples:
    """Verified train/validation cache; privileged data is loaded only by controls."""
    def __init__(self, snapshot, task, basis):
        self.snapshot, self.basis, self.task = snapshot, basis, task
        self.records = [r for r in snapshot.records if r["task"] == task]
        if any(r["split"] not in ("train", "validation") for r in self.records):
            raise ValueError("No test samples allowed in this followup")
        self.registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in self.records])
        self.pairs = {s: paired_rows([r for r in self.records if r["split"] == s])
                      for s in ("train", "validation")}
        if not all(self.pairs.values()):
            raise ValueError("Both train and validation matched sibling pairs are required")
        self.rows = {s: [r for pair in groups for r in pair] for s, groups in self.pairs.items()}
        self.episodes = {r["id"]: FeatureEpisode(EpisodeIdentity.from_record(r), snapshot.features(r),
                         snapshot.actions(r), "additive_controls", snapshot.ix["signature"])
                         for r in self.records}
        self.cached_eval = {}
        self.sampling_counts = {mode: {"examples": 0, "dropout_events": 0} for mode in DEFAULT_MODES}

    def batch(self, rows, mode, epoch, split, training=False):
        arrays, targets = [], []
        for rec in rows:
            sample = sample_predictor(self.episodes[rec["id"]], self.registry,
                       SamplingConfig(memory_dropout=.15 if training else 0.), query=rec["query"],
                       base_seed=17, epoch=epoch if split == "train" else 0,
                       expected_split=split, history_mode=mode)
            value, target = phase1_arrays(sample)
            value["projected_z"] = project(value["z"], self.basis)
            value["projected_current"] = project(value["current"], self.basis)
            arrays.append(value)
            targets.append(target)
            if training:
                self.sampling_counts[mode]["examples"] += 1
                self.sampling_counts[mode]["dropout_events"] += int(sample.metadata["dropout_event"])
        return collate(arrays, "cpu"), torch.from_numpy(np.stack(targets))

    def evaluate(self, model, split):
        model.eval()
        by_mode = {}
        with torch.inference_mode():
            for mode in DEFAULT_MODES:
                key = (split, mode)
                if key not in self.cached_eval:
                    self.cached_eval[key] = self.batch(self.rows[split], mode, 0, split)
                inputs, target = self.cached_eval[key]
                by_mode[mode] = {k: float(v) for k, v in loss_components(model(inputs), target).items()}
        return {"equal_mode_mean": {k: float(np.mean([v[k] for v in by_mode.values()])) for k in METRICS},
                "by_mode": by_mode, "pairs": len(self.pairs[split]), "episodes": len(self.rows[split])}

    def controls(self, model):
        before = state_hash(model)
        model.eval().requires_grad_(False)
        results, vectors = {}, {}
        for split, rows in self.rows.items():
            metrics, y = {}, None
            for condition in CONTROL_CONDITIONS:
                arrays, targets = [], []
                for rec in rows:
                    ep = self.episodes[rec["id"]]
                    sibling = self.snapshot.by_id[rec["sibling"]]
                    other = self.episodes[rec["sibling"]]
                    validate_matched_sibling(rec, sibling, ep.features, other.features, ep.actions, other.actions)
                    value, target, _ = control_sample(self.snapshot.config, rec, ep.features, ep.actions,
                           self.snapshot.evidence(rec), self.basis, condition, sibling_features=other.features)
                    arrays.append(value)
                    targets.append(target)
                y = torch.from_numpy(np.stack(targets))
                with torch.inference_mode():
                    prediction = model(collate(arrays, "cpu"))
                    metrics[condition] = {k: float(v) for k, v in loss_components(prediction, y).items()}
                vectors[f"{split}_{condition}"] = prediction.numpy().copy()
            current = torch.from_numpy(np.stack([self.episodes[r["id"]].features[r["query"]] for r in rows]))[:, None]
            base = (current + torch.from_numpy(self.basis["target_mean"])[None, None]).expand_as(y)
            metrics["current_plus_train_target_mean"] = {k: float(v) for k, v in loss_components(base, y).items()}
            vectors[f"{split}_targets"] = y.numpy()
            recent = metrics["recent_only"]["mse"]
            constant = metrics["current_plus_train_target_mean"]["mse"]
            results[split] = {"metrics": metrics, "pairs": len(self.pairs[split]),
                "history_benefit": {c: {"mse_minus_recent": metrics[c]["mse"] - recent,
                    "mse_minus_current_plus_mean": metrics[c]["mse"] - constant,
                    "beats_recent_and_current_plus_mean": bool(metrics[c]["mse"] < min(recent, constant))}
                    for c in ("full_history", "privileged_small_budget")},
                "wrong_minus_correct": {"privileged": metrics["wrong_evidence"]["mse"] - metrics["privileged_small_budget"]["mse"],
                                        "full": metrics["sibling_full_history"]["mse"] - metrics["full_history"]["mse"]}}
        if state_hash(model) != before:
            raise ValueError("Diagnostic changed model state")
        return results, vectors


def save_checkpoint(path, model, task, arm, epoch, score, basis_sha, protocol_sha, state_kind):
    torch.save({"model": model.state_dict(), "task": task, "arm": arm, "epoch": epoch,
        "validation_mse": score, "state_kind": state_kind, "model_config": copy.deepcopy(model._followup_config),
        "horizons": (3, 6), "feature_dimension": len(model.input_mean), "basis_sha256": basis_sha,
        "protocol_sha256": protocol_sha, "state_sha256": state_hash(model),
        "checkpoint_kind": "exploratory_fixed_projection_followup"}, path)


def runtime_threads():
    """Record effective Torch and loaded NumPy BLAS thread settings."""
    info = {"environment": {key: os.environ.get(key) for key in
            ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(), "numpy_version": np.__version__,
            "blas_libraries": []}
    numpy_root = Path(np.__file__).resolve().parent
    for directory in (numpy_root / ".dylibs", numpy_root.parent / "numpy.libs"):
        for path in directory.glob("*openblas*"):
            try:
                library = ctypes.CDLL(str(path))
                for symbol in ("openblas_get_num_threads64_", "openblas_get_num_threads"):
                    if hasattr(library, symbol):
                        function = getattr(library, symbol)
                        function.restype = ctypes.c_int
                        info["blas_libraries"].append({"library": path.name, "query_symbol": symbol,
                                                       "effective_threads": int(function())})
                        break
            except OSError as error:
                info["blas_libraries"].append({"library": path.name, "query_error": str(error)})
    return info


def run(snapshot, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refuse to overwrite followup results")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    execution_threads = runtime_threads()
    source_paths = [Path(__file__).resolve(), ROOT / "experiments/phase2/portable_snapshot.py",
        ROOT / "experiments/phase2/diagnose_history.py", ROOT / "experiments/phase2/training_samples.py",
        ROOT / "experiments/phase2/train_predictor_pilot.py", ROOT / "experiments/phase2/memory.py",
        ROOT / "experiments/phase2/deletion.py"]
    source_paths += list((ROOT / "experiments/predictor/control/src/phase1").glob("*.py"))
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(source_paths))}
    declaration = {"protocol": PROTOCOL, "declared_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sources, "backup_index_sha256": snapshot.audit()["backup_index_sha256"],
        "train_episode_ids": sorted(r["id"] for r in snapshot.records if r["split"] == "train"),
        "validation_episode_ids": sorted(r["id"] for r in snapshot.records if r["split"] == "validation"),
        "declared_before_feature_fitting_or_evaluation": True, "execution_threads": execution_threads}
    protocol_path = output / "predeclared_protocol.json"
    write(protocol_path, declaration)
    protocol_sha = sha256(protocol_path)
    for source in source_paths:
        destination = output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    result = {"status": "running", "protocol_sha256": protocol_sha, "protocol": PROTOCOL,
        "source_sha256": sources, "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "execution_threads": execution_threads, "arms": [], "scientific_phase2_completed": False,
        "learned_retention_scorer_trained": False, "test_arrays_read": 0}
    write(output / "summary.json", result)
    torch.set_num_threads(4)
    for task in PROTOCOL["tasks"]:
        original_basis = snapshot.basis(task)
        train = [r for r in snapshot.records if r["task"] == task and r["split"] == "train"]
        prefixes = [snapshot.features(r)[:r["query"] + 1] for r in train]
        for arm in ARMS:
            basis, transform = projection_variant(original_basis, prefixes, arm)
            path = output / task / arm
            path.mkdir(parents=True)
            np.savez_compressed(path / "basis.npz", **basis)
            basis_sha = sha256(path / "basis.npz")
            write(path / "projection_transform.json", transform)
            samples = Samples(snapshot, task, basis)
            seed_all(17, 4)
            model = DynamicsPredictor(len(snapshot.norm["mean"]), snapshot.config["model"], (3, 6), basis).cpu()
            model._followup_config = copy.deepcopy(snapshot.config["model"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, .5)
            train_scores, val_scores = samples.evaluate(model, "train"), samples.evaluate(model, "validation")
            score = val_scores["equal_mode_mean"]["mse"]
            best, best_epoch, step = score, 0, 0
            logs = [{"epoch": 0, "optimizer_steps": 0, "train_fixed": train_scores, "validation": val_scores}]
            for name in ("initial", "selected"):
                save_checkpoint(path / f"{name}.pt", model, task, arm, 0, score, basis_sha, protocol_sha, name)
            for epoch in range(40):
                if time.monotonic() - started > 600:
                    raise TimeoutError("Followup exceeded predeclared 600-second limit")
                model.train()
                rng = np.random.default_rng(17 + 100003 * epoch)
                order = rng.permutation(len(samples.pairs["train"]))
                epoch_losses = []
                for start in range(0, len(order), 16):
                    rows = [r for i in order[start:start + 16] for r in samples.pairs["train"][i]]
                    for mode in rng.permutation(DEFAULT_MODES):
                        inputs, target = samples.batch(rows, str(mode), epoch, "train", training=True)
                        optimizer.zero_grad(set_to_none=True)
                        loss, _, _ = regression_loss(model(inputs), target, 2., .5)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training objective")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        epoch_losses.append(float(loss.detach()))
                        step += 1
                train_scores, val_scores = samples.evaluate(model, "train"), samples.evaluate(model, "validation")
                score = val_scores["equal_mode_mean"]["mse"]
                if score < best:
                    best, best_epoch = score, epoch + 1
                    save_checkpoint(path / "selected.pt", model, task, arm, best_epoch, best, basis_sha, protocol_sha, "selected")
                scheduler.step()
                logs.append({"epoch": epoch + 1, "optimizer_steps": step,
                    "online_training_objective": float(np.mean(epoch_losses)), "train_fixed": train_scores,
                    "validation": val_scores, "selected_epoch": best_epoch})
                write(path / "learning_curve.json", logs)
                if (epoch + 1) % 10 == 0:
                    print(json.dumps({"task": task, "arm": arm, "epoch": epoch + 1,
                        "validation_mse": score, "selected_epoch": best_epoch}), flush=True)
            save_checkpoint(path / "terminal.pt", model, task, arm, 40, score, basis_sha, protocol_sha, "terminal")
            diagnostics, checkpoint_proofs = {}, {}
            for kind in ("selected", "terminal"):
                checkpoint = torch.load(path / f"{kind}.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["model"])
                diagnostic, vectors = samples.controls(model)
                np.savez_compressed(path / f"{kind}_control_vectors.npz", **vectors)
                diagnostics[kind] = diagnostic
                checkpoint_proofs[kind] = {"epoch": checkpoint["epoch"], "state_sha256": state_hash(model),
                    "checkpoint_sha256": sha256(path / f"{kind}.pt"), "all_eval": not any(m.training for m in model.modules()),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            write(path / "diagnostics.json", diagnostics)
            entry = {"task": task, "arm": arm, "projection_transform": transform,
                "selected_epoch": best_epoch, "selected_validation_mse": best,
                "initial_validation_mse": logs[0]["validation"]["equal_mode_mean"]["mse"],
                "terminal_validation_mse": score, "training_improved_selection_metric": best_epoch > 0,
                "epochs": 40, "optimizer_steps": step, "sampling_counts": samples.sampling_counts,
                "checkpoint_proofs": checkpoint_proofs, "diagnostics": diagnostics,
                "artifact_sha256": {p.name: sha256(p) for p in path.iterdir() if p.is_file()}}
            result["arms"].append(entry)
            write(output / "summary.json", result)
    result.update(status="completed_three_fixed_projection_arms", elapsed_seconds=time.monotonic() - started,
                  artifact_audit=snapshot.audit(), protocol_unchanged=sha256(protocol_path) == protocol_sha)
    if not result["protocol_unchanged"]:
        raise ValueError("Declared protocol changed during execution")
    result["selected_arms_with_validation_history_benefit_and_baseline_improvement"] = [
        {"task": entry["task"], "arm": entry["arm"], "selected_epoch": entry["selected_epoch"], "condition": condition}
        for entry in result["arms"] for condition, metric in entry["diagnostics"]["selected"]["validation"]["history_benefit"].items()
        if metric["beats_recent_and_current_plus_mean"]]
    write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def timeout(signum, frame):
        raise TimeoutError("Followup hard wall-clock limit reached (600 seconds)")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 600)
    owns_output = not args.output.exists() or not any(args.output.iterdir())
    try:
        result = run(PortableSnapshot(args.snapshot), args.output)
        print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
            "selected_arms_with_history_benefit": result["selected_arms_with_validation_history_benefit_and_baseline_improvement"]}), flush=True)
    except BaseException as error:
        if owns_output and args.output.exists():
            write(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
            summary_path = args.output / "summary.json"
            if summary_path.exists():
                partial = json.loads(summary_path.read_text())
                partial.update(status="failed_or_interrupted_partial", failure_type=type(error).__name__,
                               failure_message=str(error))
                write(summary_path, partial)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
