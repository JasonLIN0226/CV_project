"""Evaluate frozen literature adaptations on newly encoded Gate scenes.

No fitting, teacher-label training, checkpoint selection, or protocol tuning.
The old snapshot binds the frozen predictor/normalization; --fresh supplies only
new seed-20261022 development scenes. Training and evaluation source hashes are
reported separately. Oracles and sibling interventions remain evaluator-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import torch

from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.encode_fresh_gate_diagnostic import (
    CONDITIONS, DATA_CONFIG, ENCODER_CONFIG, EXPECTED_ENCODER_METADATA_SHA256,
    EXPECTED_SOURCE_SHA256, OLD_NORMALIZATION_SHA256, PAIRS_PER_CONDITION, SEED,
    sha256, write_json,
)
from experiments.phase2.literature_baselines import SalienceStudent
from experiments.phase2.portable_snapshot import PortableSnapshot, state_hash
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.run_literature_baselines import QueryOracle, aggregate, diagnose_episode


def validate_fresh_records(records):
    if not isinstance(records, list) or len(records) != len(CONDITIONS) * PAIRS_PER_CONDITION * 2:
        raise ValueError("Require all 192 fresh episodes, not an outcome-selected subset")
    by_id, pairs, seeds = {}, {}, {}
    for rec in records:
        if (not re.fullmatch(r"[0-9a-f]{24}", str(rec.get("id", "")))
                or rec["id"] in by_id or rec.get("split") != "fresh_development"
                or rec.get("task") != "gate" or rec.get("condition") not in CONDITIONS
                or rec.get("generation_condition") != CONDITIONS[rec["condition"]]
                or rec.get("feature_path") != f"features/{rec['id']}.npz"
                or rec.get("eval_path") != f"evaluation_only/{rec['id']}.npz"):
            raise ValueError("Invalid fresh identity/split/condition/path")
        if (rec["query"] < 6 or rec["frames"] != rec["query"] + 7
                or len(rec["rgb_indices"]) != rec["frames"] or min(rec["rgb_indices"]) < 0):
            raise ValueError("Invalid fresh frame/query indexing")
        by_id[rec["id"]] = rec
        pairs.setdefault(rec["pair"], []).append(rec)
        if seeds.setdefault(rec["seed"], rec["pair"]) != rec["pair"]:
            raise ValueError("Independent pairs share a scene seed")
    if len(pairs) != 96:
        raise ValueError("Require 96 independent scene pairs")
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("Incomplete sibling group")
        a, b = pair
        if (a["sibling"] != b["id"] or b["sibling"] != a["id"]
                or any(a[key] != b[key] for key in ("condition", "seed", "query", "frames", "scene_index"))):
            raise ValueError("Sibling condition/timing/identity mismatch")
        q = a["query"]
        if a["rgb_indices"][q-3:q+1] != b["rgb_indices"][q-3:q+1]:
            raise ValueError("Paired recent RGB index mismatch")
    for condition in CONDITIONS:
        subset = [r for r in records if r["condition"] == condition]
        if len(subset) != 64 or {r["scene_index"] for r in subset} != set(range(PAIRS_PER_CONDITION)):
            raise ValueError("Incomplete fixed fresh generation condition")
    return by_id


class FreshGateDataset:
    """Offline hashed data archive. Never owned by a retention-policy object."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        helper = ROOT / "experiments/phase2/encode_fresh_gate_diagnostic.py"
        if (m.get("status") != "completed" or m.get("scope") != "fresh_development"
                or m.get("seed") != SEED or m.get("data_config") != DATA_CONFIG
                or m.get("conditions") != CONDITIONS or m.get("pairs_per_condition") != PAIRS_PER_CONDITION
                or m.get("episodes") != 192 or m.get("independent_pairs") != 96
                or m.get("old_normalization_sha256") != OLD_NORMALIZATION_SHA256
                or m.get("encoder_metadata_sha256") != EXPECTED_ENCODER_METADATA_SHA256
                or m.get("source_sha256") != EXPECTED_SOURCE_SHA256
                or m.get("helper_sha256") != sha256(helper)
                or m.get("old_test_arrays_opened") != 0 or not m.get("exact_shared_rgb_features")
                or any(m.get(key) is not False for key in ("normalization_fitted", "basis_fitted", "predictor_trained", "student_trained"))):
            raise ValueError("Fresh encoder manifest is incomplete or violates the frozen protocol")
        self.loaded = {}
        self.array_files_opened = []
        for relative in m["artifact_sha256"]:
            self.path(relative)  # verify all generated bytes, including saved RGB audit bank
        audit = self.read_json("rgb_pair_audit.json")
        hardware = self.read_json("hardware.json")
        encoder = self.read_json("frozen_encoder_audit.json")
        if (not audit.get("passed") or audit.get("old_scene_seed_overlap") != 0
                or hardware != m["hardware"] or "A100" not in hardware["torch_device_name"]
                or encoder.get("actual_gpu_uuid") != hardware["actual_gpu_uuid"]
                or encoder.get("metadata_sha256") != EXPECTED_ENCODER_METADATA_SHA256
                or encoder.get("trainable_parameters") != 0 or not encoder.get("all_modules_eval")
                or encoder["metadata"].get("resolved_revision") != ENCODER_CONFIG["revision"]):
            raise ValueError("Fresh RGB/GPU/frozen encoder audit failed")
        self.records = self.read_json("records.json")
        self.by_id = validate_fresh_records(self.records)
        if any(max(record["rgb_indices"]) >= m["unique_rgb_frames"] for record in self.records):
            raise ValueError("Fresh RGB expansion index exceeds the unique-frame bank")
        self._episodes = {}
        self._normalized_bank = None

    def verify_old_normalization(self, old_norm):
        """Recompute saved normalization arithmetic; never estimate statistics."""
        relative = "unique_features.npz"
        with np.load(self.path(relative), allow_pickle=False) as archive:
            if set(archive.files) != {"raw", "normalized"}:
                raise ValueError("Unexpected unique feature-bank fields")
            raw, normalized = archive["raw"].copy(), archive["normalized"].copy()
        shape = (self.manifest["unique_rgb_frames"], 5120)
        if (raw.shape != shape or normalized.shape != shape or raw.dtype != np.float32
                or normalized.dtype != np.float32 or not np.isfinite(raw).all() or not np.isfinite(normalized).all()
                or not np.array_equal(normalized, ((raw-old_norm["mean"])/old_norm["std"]).astype(np.float32))):
            raise ValueError("Fresh features do not reproduce the original training normalization arithmetic")
        self._normalized_bank = normalized
        self.array_files_opened.append(relative + "::raw_and_normalized")

    def path(self, relative):
        path = (self.root / relative).resolve()
        expected = self.manifest["artifact_sha256"].get(relative)
        if not path.is_relative_to(self.root) or expected is None or sha256(path) != expected:
            raise ValueError("Fresh artifact checksum/path mismatch: " + str(relative))
        self.loaded[relative] = expected
        return path

    def read_json(self, relative):
        return json.loads(self.path(relative).read_text())

    def episode(self, record):
        if self.by_id.get(record.get("id")) != record:
            raise ValueError("Require an unchanged fresh record")
        if record["id"] not in self._episodes:
            relative = record["feature_path"]
            if record.get("feature_sha256") != self.manifest["artifact_sha256"].get(relative):
                raise ValueError("Fresh feature identity mismatch")
            with np.load(self.path(relative), allow_pickle=False) as archive:
                if set(archive.files) != {"z", "actions", "query"}:
                    raise ValueError("Fresh model input archive contains forbidden fields")
                z, actions, q = archive["z"].copy(), archive["actions"].copy(), archive["query"]
            if (z.shape != (record["frames"], 5120) or z.dtype != np.float32 or not np.isfinite(z).all()
                    or actions.shape != (len(z)-1, 2) or actions.dtype != np.float32 or not np.isfinite(actions).all()
                    or q.shape != () or int(q) != record["query"]):
                raise ValueError("Fresh feature/action/query shape or value mismatch")
            if self._normalized_bank is None or not np.array_equal(z, self._normalized_bank[record["rgb_indices"]]):
                raise ValueError("Episode features are not exact expansions of the verified old-normalized RGB bank")
            self.array_files_opened.append(relative)
            self._episodes[record["id"]] = (z, actions)
        return self._episodes[record["id"]]

    def evidence(self, record):
        if self.by_id.get(record.get("id")) != record:
            raise ValueError("Require an unchanged fresh record")
        relative = record["eval_path"]
        if record.get("eval_sha256") != self.manifest["artifact_sha256"].get(relative):
            raise ValueError("Fresh evidence identity mismatch")
        with np.load(self.path(relative), allow_pickle=False) as archive:
            evidence = archive["evidence"].copy()  # deliberately do not load hidden state/labels
        if evidence.shape != (1,) or not np.issubdtype(evidence.dtype, np.integer) or not 0 <= evidence[0] < record["query"]-3:
            raise ValueError("Invalid privileged historical evidence timestamp")
        self.array_files_opened.append(relative + "::evidence_only")
        return evidence

    def audit(self):
        if sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("Fresh manifest changed during evaluation")
        for relative, expected in self.loaded.items():
            if sha256(self.root / relative) != expected:
                raise ValueError("Fresh artifact changed during evaluation")
        return {"manifest_sha256": self.manifest_sha256, "verified_file_sha256": dict(self.loaded),
                "array_files_opened": sorted(set(self.array_files_opened)), "old_test_arrays_opened": 0,
                "data_seed": SEED, "normalization_fitted": False, "arrays_already_old_normalized": True}


def load_frozen_students(root, old_snapshot, fresh_started_utc):
    root = Path(root).resolve()
    protocol_path, summary_path = root / "protocol.json", root / "summary.json"
    protocol, summary = json.loads(protocol_path.read_text()), json.loads(summary_path.read_text())
    protocol_sha = sha256(protocol_path)
    old_train_groups = sorted({r["pair"] for r in old_snapshot.records if r["task"] == "gate" and r["split"] == "train"})
    if (summary.get("status") != "completed_development_diagnostic" or summary.get("protocol_sha256") != protocol_sha
            or protocol.get("train_pairs") != old_train_groups or protocol.get("scope") != "existing_gate_development_only"
            or protocol.get("student") != {"seeds": [17, 23, 41], "epochs": 100, "hidden": 32, "lr": .001,
                                           "selection": "fixed final epoch; no validation selection", "targets": "attention only"}
            or summary.get("test_arrays_read") != 0 or summary.get("utility_scorer_trained") is not False):
        raise ValueError("Student training metadata violates the frozen train-only protocol")
    definition = "experiments/phase2/literature_baselines.py"
    if protocol["source_sha256"].get(definition) != sha256(ROOT / definition):
        raise ValueError("Student feature/network definition changed after training")
    training_completed = datetime.fromisoformat(protocol["created_utc"]) + timedelta(seconds=summary["elapsed_seconds"])
    if training_completed > datetime.fromisoformat(fresh_started_utc):
        raise ValueError("Student development run was not complete before fresh encoding began")
    models, hashes, states = {}, {}, {}
    width = 5 * old_snapshot.config["model"]["rank"] + 1
    for seed in (17, 23, 41):
        path = root / f"pafu_seed_{seed}.pt"
        hashes[path.name] = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("seed") != seed or checkpoint.get("protocol_sha256") != protocol_sha
                or checkpoint.get("target_kind") != "future_attention" or checkpoint.get("trained_utility_scorer") is not False
                or checkpoint.get("hidden") != 32 or checkpoint.get("input_dim") != width):
            raise ValueError("Frozen student/checkpoint/protocol binding mismatch")
        model = SalienceStudent(width, hidden=32)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
        models[seed], states[seed] = model, state_hash(model)
    proof = {"training_directory": str(root), "protocol_sha256": protocol_sha,
             "summary_sha256": sha256(summary_path), "checkpoint_sha256": hashes,
             "model_state_sha256": states, "training_source_sha256": protocol["source_sha256"],
             "training_completed_before_fresh_encoding": True, "fresh_training_calls": 0}
    return models, proof


def history_control(record, z, actions, sibling_z, evidence, oracle, scores):
    """Replace only the privileged historical cue, preserving real targets."""
    q = record["query"]
    kept = [int(t) for t in evidence]
    for time_index in reversed(range(q-3)):
        if len(kept) == 2:
            break
        if time_index not in kept:
            kept.append(time_index)
    wrong_z = z.copy()
    wrong_z[evidence] = sibling_z[evidence]
    wrong_oracle = QueryOracle(record["id"], wrong_z, actions, q, scores)
    wrong_loss, wrong_prediction = wrong_oracle.evaluate(kept)
    recent = float(oracle.evaluate(())[0].mean())
    privileged = float(oracle.evaluate(kept)[0].mean())
    result = {"episode_id": record["id"], "pair": record["pair"], "condition": record["condition"],
              "recent_only_mse": recent, "privileged_evidence_mse": privileged,
              "wrong_sibling_evidence_mse": float(wrong_loss.mean()), "privileged_kept_times": sorted(kept),
              "privileged_minus_recent": privileged-recent,
              "wrong_minus_privileged": float(wrong_loss.mean())-privileged,
              "privileged_is_information_control_not_loss_optimum": True}
    return result, wrong_prediction


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty fresh-evaluation directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    fresh = FreshGateDataset(args.fresh)
    old = PortableSnapshot(args.snapshot)
    evaluator = FollowupGateEvaluator(old, args.followup)
    scores = CausalModelScores(evaluator)
    if evaluator.binding["normalization_sha256"] != OLD_NORMALIZATION_SHA256:
        raise ValueError("Fresh normalization differs from frozen predictor normalization")
    fresh.verify_old_normalization(old.norm)
    models, student_proof = load_frozen_students(args.students, old, fresh.manifest["started_utc"])
    source_paths = [Path(__file__), ROOT / "experiments/phase2/encode_fresh_gate_diagnostic.py"]
    source_paths += [ROOT / "experiments/phase2" / name for name in
                     ("run_literature_baselines.py", "literature_baselines.py", "memory.py", "causal_model_scores.py",
                      "run_retention_baseline_pilot.py", "run_gate_followup_phase2.py", "portable_snapshot.py")]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    protocol = {"version": "fresh-gate-literature-diagnostic-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "fresh_development", "fresh_data_seed": SEED, "conditions": list(CONDITIONS),
                "pairs_per_condition": 32, "episodes": 192, "recent": 4, "capacity": 2, "horizons": [3, 6],
                "frozen_students": student_proof, "fresh_source_sha256": source_hashes,
                "fresh_manifest_sha256": fresh.manifest_sha256, "predictor_binding": evaluator.binding,
                "student_training": False, "checkpoint_selection": False, "normalization_fitting": False,
                "oracle_objective": "equal mean h3/h6 feature MSE; same deletion choice for both horizons",
                "oracle_scopes": ["local K+1 deletion", "exhaustive observed-history K2 evaluator-only hindsight"],
                "condition_scope": "inherited ID / longer gap / distractor appearance shift; not three isolated failure types"}
    write_json(output / "protocol.json", protocol)
    rows, predictions, targets, oracle_rows, controls, wrong_predictions = [], [], [], [], [], []
    with (output / "online_trace.jsonl").open("x") as trace:
        for position, record in enumerate(fresh.records):
            z, actions = fresh.episode(record)
            sibling = fresh.by_id[record["sibling"]]
            sibling_z, sibling_actions = fresh.episode(sibling)
            q = record["query"]
            if not np.array_equal(z[q-3:q+1], sibling_z[q-3:q+1]) or not np.array_equal(actions, sibling_actions):
                raise ValueError("Fresh paired recent features or action identity failed")
            evidence = fresh.evidence(record)
            oracle = QueryOracle(record["id"], z, actions, q, scores)
            arms = [(policy, 17, None) for policy in
                    ("fifo", "reservoir", "visual_diversity", "attention", "prediction_surprise", "wr")]
            arms += [("pafu", seed, model) for seed, model in models.items()]
            for policy, seed, model in arms:
                row, prediction, events = diagnose_episode(record, z, actions, evidence, scores, policy, oracle,
                                                            seed=seed, student=model)
                row["condition"] = record["condition"]
                rows.append(row)
                predictions.append(prediction)
                targets.append(z[q + np.array([3, 6])])
                for event in events:
                    trace.write(json.dumps({"episode_id": record["id"], "condition": record["condition"],
                                            "policy": policy, "seed": seed, **event}, allow_nan=False) + "\n")
            control, wrong = history_control(record, z, actions, sibling_z, evidence, oracle, scores)
            controls.append(control)
            wrong_predictions.append(wrong)
            oracle_rows.extend({"episode_id": record["id"], "condition": record["condition"],
                                "kept_times": list(key), "feature_mse": value[0].tolist()}
                               for key, value in oracle.cache.items())
            if (position + 1) % 16 == 0:
                print(json.dumps({"episodes_evaluated": position + 1, "total_episodes": len(fresh.records)}), flush=True)
    write_json(output / "episode_results.json", rows)
    write_json(output / "oracle_subsets.json", oracle_rows)
    write_json(output / "history_controls.json", controls)
    np.savez_compressed(output / "prediction_vectors.npz", predictions=np.stack(predictions), targets=np.stack(targets))
    np.savez_compressed(output / "wrong_history_predictions.npz", predictions=np.stack(wrong_predictions))
    by_condition = {condition: aggregate([row for row in rows if row["condition"] == condition]) for condition in CONDITIONS}
    control_summary = {}
    for condition in CONDITIONS:
        part = [row for row in controls if row["condition"] == condition]
        control_summary[condition] = {key: float(np.mean([row[key] for row in part])) for key in
                                      ("recent_only_mse", "privileged_evidence_mse", "wrong_sibling_evidence_mse",
                                       "privileged_minus_recent", "wrong_minus_privileged")}
        control_summary[condition]["privileged_improves_recent_episode_fraction"] = float(np.mean(
            [row["privileged_minus_recent"] < 0 for row in part]))
    for seed, model in models.items():
        if state_hash(model) != student_proof["model_state_sha256"][seed]:
            raise ValueError("Frozen student state changed during fresh evaluation")
    for filename, expected in student_proof["checkpoint_sha256"].items():
        if sha256(args.students / filename) != expected:
            raise ValueError("Student checkpoint changed during fresh evaluation")
    if any(sha256(ROOT / relative) != expected for relative, expected in source_hashes.items()):
        raise ValueError("Fresh evaluation source changed during the run")
    summary = {"status": "completed_fresh_development_diagnostic", "elapsed_seconds": time.monotonic()-started,
               "device": "cpu_frozen_prediction_and_oracles", "visual_encoder_device": fresh.manifest["hardware"],
               "episodes": 192, "independent_pairs": 96, "pairs_per_condition": 32,
               "aggregate": aggregate(rows), "aggregate_by_condition": by_condition,
               "history_controls_by_condition": control_summary, "frozen_students": student_proof,
               "fresh_source_sha256": source_hashes, "protocol_sha256": sha256(output / "protocol.json"),
               "fresh_input_audit": fresh.audit(), "old_input_audit": old.audit(), "predictor_audit": scores.audit(),
               "student_training_calls": 0, "normalization_fitting_calls": 0, "new_visual_scenes": 96,
               "old_test_arrays_read": 0, "equal_retained_K": True, "equal_total_bytes": False,
               "limitations": ["fresh development scenarios, not untouched benchmark confirmation",
                               "three inherited conditions do not isolate three distinct failure mechanisms",
                               "PaFu bidirectional additive-pooling proxy differs from original DiT QKV/DMD",
                               "WR resident window4 differs from original64; causal first-peak NMS",
                               "hindsight oracle uses evaluator-only future information; gap is not causal learnability proof",
                               "feature prediction, not model-driven PushT planning"]}
    summary["result_artifact_sha256"] = {str(path.relative_to(output)): sha256(path)
                                         for path in sorted(output.rglob("*")) if path.is_file() and path.name != "summary.json"}
    write_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"],
                      "aggregate_by_condition": by_condition, "history_controls": control_summary}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
