"""Fixed train-support intervention on the existing frozen Gate predictor.

This is an exploratory diagnostic on already observed development trajectories.
The input mask is fitted only from old training prefixes; no new model or policy
is trained, and no intervention is adopted as benchmark performance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments/predictor/control/src"):
    sys.path.insert(0, str(directory))
import numpy as np
import torch
from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.diagnose_gate_oracle_metric import (
    decode, fit_readout, group_ci, paired_components,
)
from experiments.phase2.memory import MemoryEntry
from experiments.phase2.portable_snapshot import PortableSnapshot, state_hash
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from phase1.model import project

THRESHOLD = 1e-4  # Existing diagnose_training_objective.inactive_columns rule.
ARMS = ("original", "train_support_mask")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def training_support(train_projected):
    x = np.asarray(train_projected)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Require finite nonempty TRAIN-prefix projected features")
    # Preserve the old float32 mean-square arithmetic, not a std threshold.
    second_moment = np.mean(x ** 2, axis=0)
    return second_moment < THRESHOLD, second_moment


def masked_batch(batch, unsupported):
    """Copy only projected inputs; retain the original raw-current output skip."""
    mask = torch.as_tensor(np.asarray(unsupported, dtype=bool))
    if mask.ndim != 1 or mask.numel() != batch["projected_current"].shape[-1]:
        raise ValueError("Projection mask has the wrong dimension")
    result = dict(batch)
    for name in ("projected_z", "projected_current"):
        result[name] = batch[name].clone()
        result[name][..., mask] = 0
    return result


def masked_readout(readout, unsupported):
    """Zero existing coefficient rows; no refitting or standardizer changes."""
    result = {key: np.asarray(value).copy() for key, value in readout.items()}
    result["weight"][unsupported] = 0
    return result


def predict_pair(scores, rec, z, actions, kept, unsupported):
    q = rec["query"]
    recent = tuple(MemoryEntry(rec["id"], t, z[t]) for t in range(q-3, q+1))
    history = tuple(MemoryEntry(rec["id"], int(t), z[int(t)]) for t in kept)
    past = {e.entry_id: actions[e.timestamp:q].sum(0, dtype=np.float32)
            for e in recent + history}
    batch = scores._batch(recent, history, past, q, actions[q:q+6].copy())
    original_projected = batch["projected_z"].clone()
    scores._evaluator._ensure_frozen()
    changed = masked_batch(batch, unsupported)
    if changed["current"] is not batch["current"] or changed["z"] is not batch["z"]:
        raise AssertionError("The raw feature/skip path changed")
    result = {}
    with torch.inference_mode():
        for arm, inputs in (("original", batch), ("train_support_mask", changed)):
            pred = scores._model(inputs)[0].numpy().copy()
            history_mask = inputs["mask"] & (inputs["type"] == 1)
            weight = scores._weights(inputs["projected_z"], inputs["projected_current"], history_mask)
            result[arm] = (pred, weight[0, 4:].numpy().tolist())
    if not torch.equal(batch["projected_z"], original_projected):
        raise AssertionError("Mask mutated the original projected input")
    scores._evaluator._ensure_frozen()
    return result


def unsupported_fraction(projected, unsupported):
    x = np.asarray(projected, np.float64)
    total = np.sum(x*x)
    return float(np.sum(x[..., unsupported] ** 2) / total) if total else 0.


def summaries(rows, field="mse"):
    return {condition: group_ci([r for r in rows if condition == "all" or r["condition"] == condition], field)
            for condition in ("all", "id", "long_gap", "distractor_shift")
            if any(condition == "all" or r["condition"] == condition for r in rows)}


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Require a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    source_files = [Path(__file__), ROOT / "experiments/phase2/diagnose_training_objective.py",
                    ROOT / "experiments/phase2/diagnose_gate_oracle_metric.py",
                    ROOT / "experiments/phase2/diagnose_gate_reader_local.py"]
    input_files = [args.evaluation / "episode_results.json",
                   args.global_run / "episode_rows.json", args.global_run / "summary.json",
                   args.local_run / "parents.json", args.local_run / "episode_rows.json",
                   args.local_run / "cue_probe.json", args.local_run / "summary.json"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files + input_files}
    protocol = {
        "version": "gate-train-prefix-support-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "posthoc diagnostic on already observed development trajectories; not confirmation",
        "support_fit": "only z[:query+1] from all32 old Gate TRAIN episodes in original unit-column projection",
        "threshold": THRESHOLD, "threshold_semantics": "mean(projected**2,axis=0)<1e-4, float32; existing inactive_columns definition",
        "predictor_intervention": "zero unsupported coordinates of projected_z and projected_current; raw current output skip, learned weights/actions, and all other inputs unchanged",
        "fixed_sets": "all192 actual attention K2, all192 original global best-with/without cue, all72 available original local actual-parent cue comparisons; no set reoptimization",
        "cue_probe": "reconstruct prior cue and cue-minus-current lambda1 ridge exactly on old32train, assert every prior saved score; zero unsupported coefficient rows with means/std/other weights frozen",
        "statistics": "5000 scene-pair bootstrap samples, seed20261022; full sets and fixed originally cue-harmed cohorts",
        "model_refitting": False, "policy_refitting": False, "test_arrays": False,
        "runtime": {"python": sys.executable, "numpy": np.__version__, "torch": torch.__version__,
                    "blas": getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas", {})},
        "rank": "descriptive TRAIN-prefix distinct frames and numerical projected rank at source-float32 tolerance; no fitted replacement basis",
        "hashes": hashes,
        "limitations": ["Masking can itself be out of distribution, not an adopted repair.",
                        "Unsupported means low training second moment, not proof a coordinate is semantically useless.",
                        "Global best sets are hindsight; local comparisons hold one original candidate parent.",
                        "Cue probes are privileged evaluation probes, not retention scorers or proof of unavailable encoder information."]}
    save(output / "protocol.json", protocol)
    for p in source_files:
        destination = output / "source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())

    old = PortableSnapshot(args.snapshot)
    evaluator = FollowupGateEvaluator(old, args.followup)
    scores = CausalModelScores(evaluator)
    original_state = state_hash(scores._model)
    fresh = FreshGateDataset(args.fresh)
    fresh.verify_old_normalization(old.norm)
    basis = scores._basis
    train = [r for r in old.records if r["task"] == "gate" and r["split"] == "train"]
    assert len(train) == 32 and len(fresh.records) == 192
    train_raw = np.concatenate([old.features(r)[:r["query"]+1] for r in train])
    train_projected = project(train_raw, basis)
    unsupported, second_moment = training_support(train_projected)
    singular = np.linalg.svd(train_projected.astype(np.float64), compute_uv=False)
    rank_tol32 = float(singular[0] * max(train_projected.shape) * np.finfo(np.float32).eps)
    rank_tol64 = float(singular[0] * max(train_projected.shape) * np.finfo(np.float64).eps)
    unique = np.unique(train_raw, axis=0)
    train_summary = {
        "train_episode_ids": [r["id"] for r in train], "prefix_frames": len(train_raw),
        "distinct_raw_prefix_feature_rows": len(unique), "projection_dimensions": len(unsupported),
        "unsupported_count": int(unsupported.sum()), "unsupported_indices": np.flatnonzero(unsupported).tolist(),
        "second_moment": second_moment.tolist(), "std": train_projected.std(0).tolist(),
        "mean": train_projected.mean(0).tolist(), "singular_values": singular.tolist(),
        "projected_rank_at_float32_tolerance": int(np.sum(singular > rank_tol32)), "float32_rank_tolerance": rank_tol32,
        "projected_rank_at_float64_tolerance": int(np.sum(singular > rank_tol64)), "float64_rank_tolerance": rank_tol64,
        "centered_unique_frame_rank_upper_bound": min(len(unique)-1, train_projected.shape[1]),
        "rank_caveat": "Inputs were computed in float32; double precision rank can count projection-rounding noise. Unique-frame bound assumes exact centering in exact arithmetic.",
        "train_unsupported_energy_fraction": unsupported_fraction(train_projected, unsupported)}
    save(output / "training_support.json", train_summary)
    np.savez_compressed(output / "training_support.npz", unsupported=unsupported,
                        second_moment=second_moment, mean=train_projected.mean(0), std=train_projected.std(0), singular=singular)

    baseline = {r["episode_id"]: r for r in json.loads(input_files[0].read_text()) if r["policy"] == "attention"}
    global_rows = [r for r in json.loads((args.global_run / "episode_rows.json").read_text()) if r["arm"] == "original"]
    local_rows = [r for r in json.loads((args.local_run / "episode_rows.json").read_text()) if r["arm"] == "original"]
    fixed = defaultdict(list)
    for rec in fresh.records:
        r = baseline[rec["id"]]
        fixed[rec["id"]].append(("attention", r["kept_times"], r["mean_mse"]))
    for family, selected in (("global", global_rows), ("local", local_rows)):
        for r in selected:
            fixed[r["episode_id"]].append((family + ("_cue" if r["keeps_cue"] else "_no_cue"), r["kept_times"], r["metrics"]["mse"]))

    # Reconstruct the already published cue probes with identical old TRAIN data.
    cue_train, cue_y, cue_inputs = defaultdict(list), [], []
    for rec in old.records:
        if rec["task"] != "gate":
            continue
        z = old.features(rec)
        hidden = old.read_npz("data/dataset/" + rec["eval"], ("evidence", "label"))
        t, label, q = int(hidden["evidence"][0]), int(hidden["label"]), rec["query"]
        inputs = {"cue": project(z[t], basis), "cue_minus_current": project(z[t], basis)-project(z[q], basis)}
        cue_inputs.append({"id": rec["id"], "pair": rec["pair"], "split": rec["split"], "label": label, "inputs": inputs})
        if rec["split"] == "train":
            for name, x in inputs.items():
                cue_train[name].append(x)
            cue_y.append([label])
    readouts = {name: fit_readout(x, cue_y) for name, x in cue_train.items()}
    for name, readout in readouts.items():
        np.savez_compressed(output / (name + "_readout.npz"), **readout)

    rows, vectors, names, input_diagnostics = [], [], [], []
    for i, rec in enumerate(fresh.records):
        z, actions = fresh.episode(rec)
        sibling, _ = fresh.episode(fresh.by_id[rec["sibling"]])
        q, times = rec["query"], rec["query"] + np.array([3, 6])
        target = z[times]
        with np.load(fresh.path(rec["eval_path"]), allow_pickle=False) as archive:
            t, label = int(archive["evidence"][0]), int(archive["label"])
        cue_inputs.append({"id": rec["id"], "pair": rec["pair"], "split": rec["condition"], "label": label,
                           "inputs": {"cue": project(z[t], basis), "cue_minus_current": project(z[t], basis)-project(z[q], basis)}})
        input_diagnostics.append({"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"],
                                  "prefix_unsupported_energy_fraction": unsupported_fraction(project(z[:q+1], basis), unsupported),
                                  "cue_unsupported_energy_fraction": unsupported_fraction(project(z[t], basis), unsupported),
                                  "current_unsupported_energy_fraction": unsupported_fraction(project(z[q], basis), unsupported)})
        for family, kept, prior_loss in fixed[rec["id"]]:
            predictions = predict_pair(scores, rec, z, actions, kept, unsupported)
            for arm, (pred, attention) in predictions.items():
                measure = paired_components(pred, target, sibling[times])
                measure["raw_feature_mse"] = float(np.mean(((pred.astype(np.float64)-target)*old.norm["std"])**2))
                if arm == "original" and abs(measure["mse"]-prior_loss) > 1e-12:
                    raise AssertionError("Original fixed-set prediction loss changed")
                rows.append({"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"],
                             "family": family, "arm": arm, "kept_times": kept, "attention": attention, **measure})
                vectors.append(pred); names.append(f"{rec['id']}:{family}:{arm}")
        if (i+1) % 32 == 0:
            print(json.dumps({"episodes": i+1, "total": 192}), flush=True)
    save(output / "episode_rows.json", rows)
    save(output / "input_diagnostics.json", input_diagnostics)
    np.savez_compressed(output / "predictions.npz", predictions=np.stack(vectors), case_names=np.array(names))

    old_probe = {(r["id"], r["arm"]): r for r in json.loads((args.local_run / "cue_probe.json").read_text())}
    probe = []
    for rec in cue_inputs:
        for name, readout in readouts.items():
            value = float(decode(rec["inputs"][name], readout)[0])
            if value != old_probe[rec["id"], name]["score"]:
                raise AssertionError("Reconstructed original cue-probe score differs")
            for arm, model in (("original", readout), ("train_support_mask", masked_readout(readout, unsupported))):
                value = float(decode(rec["inputs"][name], model)[0])
                probe.append({k: v for k, v in rec.items() if k != "inputs"} |
                             {"probe": name, "arm": arm, "score": value, "abs_score": abs(value), "correct": (value >= .5) == rec["label"]})
    save(output / "cue_probe.json", probe)

    diffs = []
    for a, b in zip(rows[::2], rows[1::2]):
        assert a["arm"] == "original" and b["arm"] == "train_support_mask"
        diffs.append({k: a[k] for k in ("episode_id", "pair", "condition", "family")} |
                     {key: b[key]-a[key] for key in ("mse", "contrast_mse", "orthogonal_mse", "raw_feature_mse", "feature_branch_correct")})
    save(output / "mask_minus_original.json", diffs)
    by_key = {(r["episode_id"], r["family"], r["arm"]): r for r in rows}
    cue_gaps = []
    for rec in fresh.records:
        for scope in ("global", "local"):
            if (rec["id"], scope + "_no_cue", "original") not in by_key:
                continue
            for arm in ARMS:
                a, b = (by_key[rec["id"], scope + suffix, arm] for suffix in ("_cue", "_no_cue"))
                cue_gaps.append({"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"], "scope": scope, "arm": arm,
                                 **{key: a[key]-b[key] for key in ("mse", "contrast_mse", "orthogonal_mse", "raw_feature_mse", "feature_branch_correct")}})
    save(output / "cue_gaps.json", cue_gaps)
    cue_summary = {}
    for scope in ("global", "local"):
        part = [r for r in cue_gaps if r["scope"] == scope]
        bad_ids = {r["episode_id"] for r in part if r["arm"] == "original" and r["mse"] > 1e-8}
        cue_summary[scope] = {"available_episodes": len(part)//2, "original_strict_cue_harm_episodes": len(bad_ids)}
        for arm in ARMS:
            subset = [r for r in part if r["arm"] == arm]
            bad = [r for r in subset if r["episode_id"] in bad_ids]
            cue_summary[scope][arm] = {
                "all_cue_penalty": summaries(subset), "all_strict_cue_harm_episodes": sum(r["mse"] > 1e-8 for r in subset),
                "fixed_original_harm_cohort": {key: summaries(bad, key) for key in ("mse", "contrast_mse", "orthogonal_mse", "raw_feature_mse")},
                "fixed_original_harm_cohort_rescued": sum(r["mse"] <= 1e-8 for r in bad)}
    families = sorted({r["family"] for r in rows})
    probe_summary = {name: {arm: {split: {
        "accuracy": group_ci([r for r in probe if r["probe"] == name and r["arm"] == arm and r["split"] == split], "correct"),
        "mean_absolute_score": group_ci([r for r in probe if r["probe"] == name and r["arm"] == arm and r["split"] == split], "abs_score"),
        "max_absolute_score": max(r["abs_score"] for r in probe if r["probe"] == name and r["arm"] == arm and r["split"] == split)
    } for split in ("train", "validation", "id", "long_gap", "distractor_shift")} for arm in ARMS} for name in readouts}
    if state_hash(scores._model) != original_state:
        raise AssertionError("Frozen predictor state changed")
    for relative, expected in hashes.items():
        if sha(ROOT / relative) != expected:
            raise AssertionError("Source or fixed-set result changed during run: " + relative)
    summary = {
        "status": "completed", "elapsed_seconds": time.monotonic()-started, "episodes": len(fresh.records),
        "prediction_rows": len(rows), "original_fixed_set_replays_verified": len(rows)//2,
        "prior_cue_probe_scores_replayed_exactly": len(old_probe), "model_state_bitwise_unchanged": True,
        "protocol_sha256": sha(output / "protocol.json"), "source_sha256": sha(__file__),
        "overall_mse": {family: {arm: summaries([r for r in rows if r["family"] == family and r["arm"] == arm]) for arm in ARMS} for family in families},
        "mask_minus_original": {family: {key: summaries([r for r in diffs if r["family"] == family], key) for key in ("mse", "contrast_mse", "raw_feature_mse")} for family in families},
        "cue_penalty": cue_summary, "cue_probe": probe_summary,
        "input_energy": {key: summaries(input_diagnostics, key) for key in ("prefix_unsupported_energy_fraction", "cue_unsupported_energy_fraction", "current_unsupported_energy_fraction")},
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "frozen_predictor_audit": scores.audit(), "old_input_audit": old.audit(), "fresh_input_audit": fresh.audit()}
    save(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "episodes", "prediction_rows", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "build/hpc-20261020/development-assets")
    parser.add_argument("--followup", type=Path, default=ROOT / "build/history-diagnosis-20261020/followup-blas1")
    parser.add_argument("--fresh", type=Path, default=ROOT / "build/literature-baselines-20261022/fresh-data")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "build/literature-baselines-20261022/fresh-eval")
    parser.add_argument("--global-run", type=Path, default=ROOT / "build/literature-diagnosis-20261022/metric-reader-v2")
    parser.add_argument("--local-run", type=Path, default=ROOT / "build/literature-diagnosis-20261022/reader-local-v2")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
