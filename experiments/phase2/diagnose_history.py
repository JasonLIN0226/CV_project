"""Read-only, train/validation-only history-use diagnostic for the fresh pilot.

Privileged timestamps and sibling interventions are evaluator-only controls.
They are never a deployable retention policy, and full history exceeds the
predictor's training memory budget. No fitting or checkpoint selection occurs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from experiments.phase2.training_samples import (
    DEFAULT_MODES, EpisodeIdentity, FeatureEpisode, SamplingConfig, SplitRegistry,
    phase1_arrays, sample_predictor,
)
from phase1.inputs import collate, memory_indices, selected_evidence
from phase1.model import project
from phase1.training import paired_rows


CONTROL_CONDITIONS = (
    "recent_only", "full_history", "privileged_small_budget",
    "wrong_evidence", "sibling_full_history",
)
HORIZONS = (3, 6)


def state_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        result.update(name.encode() + b"\0" + str(a.dtype).encode() + repr(a.shape).encode())
        result.update(a.tobytes())
    return result.hexdigest()


def validate_matched_sibling(record, sibling, features, sibling_features, actions, sibling_actions):
    """Ensure swaps preserve recent observations, timestamp schedule and actions."""
    if (record["sibling"] != sibling["id"] or sibling["sibling"] != record["id"]
            or record["pair"] != sibling["pair"] or record["task"] != sibling["task"]
            or record["split"] != sibling["split"] or record["query"] != sibling["query"]):
        raise ValueError("Sibling intervention requires a matched within-split pair")
    q = record["query"]
    if (features.shape != sibling_features.shape
            or not np.array_equal(features[q - 3:q + 1], sibling_features[q - 3:q + 1])
            or not np.array_equal(actions[:q + 6], sibling_actions[:q + 6])):
        raise ValueError("Sibling intervention requires exactly matching recent context and actions")


def control_sample(config, record, features, actions, evidence, basis, condition,
                   *, sibling_features=None):
    """Create a legal-time control; evaluation targets are returned separately.

    ``wrong_evidence`` replaces only privileged evidence slots, preserving any
    budget filler. ``sibling_full_history`` replaces every old observation.
    Neither swaps current/recent context, action summaries, or target identity.
    """
    if record["split"] not in ("train", "validation"):
        raise ValueError("History diagnostic cannot access test splits")
    if condition not in CONTROL_CONDITIONS:
        raise ValueError("Unknown diagnostic condition")
    q, recent, budget = record["query"], config["data"]["recent"], config["data"]["budget"]
    if recent != 4 or budget != 2 or tuple(config["data"]["horizons"]) != HORIZONS:
        raise ValueError("Require the audited recent=4, budget=2, horizons=(3,6) contract")
    if features.ndim != 2 or features.dtype != np.float32 or len(features) <= q + 6:
        raise ValueError("Require float32 features with both future targets")
    if actions.shape[1:] != (2,) or actions.dtype != np.float32 or len(actions) < q + 6:
        raise ValueError("Require executed/planned additive actions for trained horizons")
    evidence = selected_evidence(config, record, evidence)
    index_condition = "full_history" if condition == "sibling_full_history" else condition
    indices = memory_indices(q, recent, budget, evidence, index_condition, np.random.default_rng(0))
    times = np.r_[np.arange(q - 3, q + 1), indices]
    z = features[times].copy()
    replaced = []
    if condition in ("wrong_evidence", "sibling_full_history"):
        if sibling_features is None or sibling_features.shape != features.shape:
            raise ValueError("A matched sibling feature sequence is required for a swap")
        for slot, timestamp in enumerate(indices):
            if condition == "sibling_full_history" or timestamp in evidence:
                z[recent + slot] = sibling_features[timestamp]
                replaced.append(int(timestamp))
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions[:q], axis=0)])
    arrays = {"z": z, "age": (q - times).astype(np.float32),
              "type": np.r_[np.zeros(4, np.int64), np.ones(len(indices), np.int64)],
              "past_actions": prefix[q] - prefix[times],
              "actions": actions[q:q + 6].copy(), "current": features[q].copy()}
    arrays["projected_z"] = project(arrays["z"], basis)
    arrays["projected_current"] = project(arrays["current"], basis)
    targets = features[q + np.asarray(HORIZONS)].copy()
    return arrays, targets, {"history_indices": indices.tolist(), "replaced_timestamps": replaced,
                             "evidence_timestamps": evidence.tolist(), "history_count": len(indices),
                             "evaluator_privileged": condition in ("privileged_small_budget", "wrong_evidence"),
                             "exceeds_training_memory_budget": len(indices) > budget}


def paired_summary(rows, metric="mse"):
    """Group the two siblings before computing means, SD, or sign counts."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pair"], []).append(row)
    if not groups or any(len(group) != 2 or len({r["episode_id"] for r in group}) != 2
                         for group in groups.values()):
        raise ValueError("Metrics require complete, distinct two-episode pairs")
    values = np.array([np.mean([r[metric] for r in group]) for group in groups.values()], np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic metric")
    return {"mean": float(values.mean()), "pair_std_ddof1": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            "pairs": len(values), "episodes": 2 * len(values),
            "pair_values": {pair: float(value) for pair, value in zip(groups, values)}}


def compare_initial_replay(actual, logged, tolerance=1e-6):
    if set(actual) != set(DEFAULT_MODES) or set(logged) != set(DEFAULT_MODES):
        raise ValueError("Initial replay must contain all six original sampling modes")
    differences = {mode: float(actual[mode] - logged[mode]) for mode in DEFAULT_MODES}
    maximum = max(abs(value) for value in differences.values())
    return {"logged": logged, "reconstructed": actual, "difference": differences,
            "maximum_absolute_difference": maximum, "tolerance": tolerance,
            "matches_within_tolerance": maximum <= tolerance,
            "bit_exact_scores": maximum == 0.,
            "reconstruction_status": "numerically_verified" if maximum <= tolerance else "UNVERIFIED_INITIALIZATION"}


def _predict(model, arrays, batch_size=32):
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Diagnostic requires a frozen eval model")
    out = []
    with torch.inference_mode():
        for start in range(0, len(arrays), batch_size):
            out.append(model(collate(arrays[start:start + batch_size], "cpu")).cpu().numpy())
    prediction = np.concatenate(out)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite diagnostic prediction")
    return prediction


def run(snapshot, output, batch_size=32):
    """Evaluate an audited PortableSnapshot without modifying its artifacts."""
    began = time.monotonic()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refuse to overwrite an existing diagnostic result directory")
    output.mkdir(parents=True, exist_ok=True)
    records = snapshot.records
    if any(r["split"] not in ("train", "validation") for r in records):
        raise ValueError("Portable snapshot exposed a test record for evaluation")
    config = copy.deepcopy(snapshot.config)
    by_id = {r["id"]: r for r in records}
    registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in records],
                             split_roles={"train": "training", "validation": "validation"})
    artifacts_before = snapshot.audit()
    all_rows, comparisons, aggregates, model_proofs, replays, references, vectors = [], [], [], {}, {}, [], {}
    sampling = SamplingConfig(memory_dropout=0.)

    for task in ("gate", "motion"):
        basis = snapshot.basis(task)
        task_records = [r for r in records if r["task"] == task]
        cache = {r["id"]: (snapshot.features(r), snapshot.actions(r), snapshot.evidence(r)) for r in task_records}
        for rec in task_records:
            sibling = by_id[rec["sibling"]]
            z, a, _ = cache[rec["id"]]
            sz, sa, _ = cache[sibling["id"]]
            validate_matched_sibling(rec, sibling, z, sz, a, sa)
        task_log = snapshot.training_log(task)
        for version in ("initial", "best", "latest"):
            model, metadata = snapshot.model(task, kind=version)
            original_hash = state_hash(model)
            mode_scores = {}
            for split in ("train", "validation"):
                rows = [r for pair in paired_rows([r for r in task_records if r["split"] == split]) for r in pair]
                predictions, targets, condition_info = {}, None, {}
                for condition in CONTROL_CONDITIONS:
                    arrays, target_rows, infos = [], [], []
                    for rec in rows:
                        z, actions, evidence = cache[rec["id"]]
                        sample, target, info = control_sample(config, rec, z, actions, evidence, basis, condition,
                                                              sibling_features=cache[rec["sibling"]][0])
                        arrays.append(sample)
                        target_rows.append(target)
                        infos.append(info)
                    targets = np.stack(target_rows)
                    prediction = _predict(model, arrays, batch_size)
                    predictions[condition], condition_info[condition] = prediction, infos
                    vector_key = f"{task}_{version}_{split}_{condition}"
                    vectors[vector_key] = prediction
                    vectors[f"{task}_{split}_targets"] = targets
                    for index, rec in enumerate(rows):
                        sibling_index = next(j for j, other in enumerate(rows) if other["id"] == rec["sibling"])
                        for hi, horizon in enumerate(HORIZONS):
                            delta = prediction[index, hi].astype(np.float64) - targets[index, hi]
                            other_delta = prediction[index, hi].astype(np.float64) - targets[sibling_index, hi]
                            row = {"task": task, "version": version, "split": split, "condition": condition,
                                   "episode_id": rec["id"], "sibling": rec["sibling"], "pair": rec["pair"],
                                   "query": rec["query"], "horizon": horizon, "mse": float(np.mean(delta ** 2)),
                                   "sibling_target_mse": float(np.mean(other_delta ** 2)), **infos[index]}
                            row["own_target_preference"] = int(row["mse"] < row["sibling_target_mse"])
                            all_rows.append(row)
                    for hi, horizon in enumerate(HORIZONS):
                        selected = [r for r in all_rows if r["task"] == task and r["version"] == version
                                    and r["split"] == split and r["condition"] == condition and r["horizon"] == horizon]
                        aggregates.append({"task": task, "version": version, "split": split, "condition": condition,
                                           "horizon": horizon, "mse": paired_summary(selected),
                                           "own_target_preference": paired_summary(selected, "own_target_preference")})
                for correct, intervention in (("recent_only", "privileged_small_budget"),
                                              ("recent_only", "full_history"),
                                              ("privileged_small_budget", "wrong_evidence"),
                                              ("full_history", "sibling_full_history")):
                    for hi, horizon in enumerate(HORIZONS):
                        values = []
                        for index, rec in enumerate(rows):
                            first = predictions[correct][index, hi].astype(np.float64)
                            second = predictions[intervention][index, hi].astype(np.float64)
                            target = targets[index, hi].astype(np.float64)
                            values.append({"pair": rec["pair"], "episode_id": rec["id"],
                                           "error_second_minus_first": float(np.mean((second - target) ** 2) - np.mean((first - target) ** 2)),
                                           "prediction_change_mse": float(np.mean((second - first) ** 2))})
                        comparisons.append({"task": task, "version": version, "split": split, "first": correct,
                                            "second": intervention, "horizon": horizon,
                                            "error_second_minus_first": paired_summary(values, "error_second_minus_first"),
                                            "prediction_change_mse": paired_summary(values, "prediction_change_mse"),
                                            "episode_values": values})
                # Same six history modes and validation epoch=0 as the recorded run.
                for mode in DEFAULT_MODES:
                    arrays, target_rows = [], []
                    for rec in rows:
                        z, actions, _ = cache[rec["id"]]
                        episode = FeatureEpisode(EpisodeIdentity.from_record(rec), z, actions, "additive_controls", "audited-portable-snapshot")
                        sample = sample_predictor(episode, registry, sampling, query=rec["query"], base_seed=17,
                                                  epoch=0, expected_split=split, history_mode=mode)
                        a, target = phase1_arrays(sample)
                        a["projected_z"], a["projected_current"] = project(a["z"], basis), project(a["current"], basis)
                        arrays.append(a)
                        target_rows.append(target)
                    prediction = _predict(model, arrays, batch_size)
                    target = np.stack(target_rows)
                    # torch float32 reduction and weighted batches match original validation.
                    totals, count = 0., 0
                    for start in range(0, len(rows), batch_size):
                        p, t = torch.from_numpy(prediction[start:start + batch_size]), torch.from_numpy(target[start:start + batch_size])
                        totals += float((p - t).square().mean()) * len(p)
                        count += len(p)
                    mode_scores.setdefault(split, {})[mode] = totals / count
                if version == "initial":
                    for hi, horizon in enumerate(HORIZONS):
                        for reference in ("current_persistence", "current_plus_train_target_mean"):
                            items = []
                            for index, rec in enumerate(rows):
                                z = cache[rec["id"]][0]
                                pred = z[rec["query"]].astype(np.float64)
                                if reference == "current_plus_train_target_mean":
                                    pred = pred + basis["target_mean"]
                                items.append({"pair": rec["pair"], "episode_id": rec["id"],
                                              "mse": float(np.mean((pred - targets[index, hi]) ** 2))})
                            references.append({"task": task, "split": split, "reference": reference,
                                               "horizon": horizon, "mse": paired_summary(items)})
            if state_hash(model) != original_hash:
                raise ValueError("Evaluation changed model values")
            model_proofs[f"{task}/{version}"] = {"metadata": metadata, "state_sha256": original_hash,
                                                 "state_unchanged": True, "all_eval": not any(m.training for m in model.modules()),
                                                 "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                                 "six_mode_eval_mse": mode_scores,
                                                 "six_mode_equal_mean": {s: float(np.mean(list(v.values()))) for s, v in mode_scores.items()}}
            if version == "initial":
                replays[task] = compare_initial_replay(mode_scores["validation"], task_log["initial_validation_by_mode"])

    vectors_path = output / "prediction_vectors.npz"
    np.savez_compressed(vectors_path, **vectors)
    rows_path = output / "episode_horizon_results.json"
    rows_path.write_text(json.dumps(all_rows, indent=2) + "\n")
    report = {"status": "completed_read_only_history_diagnostic", "created_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - began, "device": "cpu", "torch_version": torch.__version__,
              "protocol": {"tasks": ["gate", "motion"], "splits": ["train", "validation"],
                           "versions": ["initial", "best", "latest"], "horizons": list(HORIZONS),
                           "conditions": list(CONTROL_CONDITIONS), "batch_size": batch_size,
                           "no_training_no_selection": True, "test_arrays_read": 0,
                           "independent_unit": "matched sibling pair", "privileged_evaluator_only": True,
                           "full_history_exceeds_training_budget": True,
                           "six_mode_train_evaluation": "epoch 0 draws with dropout disabled; distinct from stochastic training objective",
                           "limitations": ["Small reused development validation split; no untouched test/generalization claim.",
                                           "Initial state is reconstructed and only numerically certified when logged mode scores match.",
                                           "A response to history is distinct from a reduction in prediction error."]},
              "initialization_replay": replays, "model_proofs": model_proofs,
              "aggregate_metrics": aggregates, "contrasts": comparisons, "reference_predictions": references,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "prediction_vectors_sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
              "episode_horizon_results_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
              "artifact_audit_before": artifacts_before, "artifact_audit_after": snapshot.audit()}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("batch-size must be positive, even and at least two")
    from experiments.phase2.portable_snapshot import PortableSnapshot
    result = run(PortableSnapshot(args.snapshot), args.output, args.batch_size)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "initialization_replay": result["initialization_replay"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
