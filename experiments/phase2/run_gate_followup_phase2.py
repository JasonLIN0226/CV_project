"""Gate-only development baselines and deletion labels for the verified followup.

A distinct evaluator constructor binds the unit-column epoch-35 checkpoint.
The original pilot evaluator's artifact checks and constructor remain unchanged.
No scorer training, test-array access, equal-byte claim or planning result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import numpy as np
import torch

from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.deletion import CausalContext, CausalParent, EvaluatorProvenance
from experiments.phase2.frozen_predictor import FrozenPredictorEvaluator, _immutable, _versions
from experiments.phase2.generate_deletion_pilot import generate_task, sha256, write
from experiments.phase2.memory import MemoryEntry, POLICIES
from experiments.phase2.portable_snapshot import PortableSnapshot, state_hash, BACKUP_INDEX_SHA256
from experiments.phase2.run_retention_baseline_pilot import replay_episode, aggregate
from experiments.phase2.training_samples import EpisodeIdentity, SplitRegistry
from experiments.phase2.train_predictor_followup import runtime_threads
from phase1.model import DynamicsPredictor


def verify_selected_binding(checkpoint, basis, original_basis, protocol_sha, expected_config):
    """Reject another arm/epoch/model space; do not reinterpret historical files."""
    if (checkpoint.get("task") != "gate" or checkpoint.get("arm") != "unit_columns"
            or checkpoint.get("epoch") != 35 or checkpoint.get("state_kind") != "selected"
            or checkpoint.get("checkpoint_kind") != "exploratory_fixed_projection_followup"
            or checkpoint.get("protocol_sha256") != protocol_sha
            or checkpoint.get("model_config") != expected_config or checkpoint.get("horizons") != (3, 6)):
        raise ValueError("Require the explicitly bound Gate unit-column selected epoch-35 checkpoint")
    if checkpoint.get("feature_dimension") != len(basis["input_mean"]):
        raise ValueError("Checkpoint feature space dimension differs")
    for key in ("input_mean", "target_mean", "output_basis", "train_ids"):
        if not np.array_equal(basis[key], original_basis[key]):
            raise ValueError("Followup changed a preserved fitted basis array: " + key)
    expected_projection = original_basis["projection"] / np.linalg.norm(original_basis["projection"], axis=0, keepdims=True)
    if not np.array_equal(basis["projection"], expected_projection):
        raise ValueError("Projection is not the exact train-fitted unit-column transform")
    for key in ("input_mean", "projection", "target_mean", "output_basis"):
        if not np.array_equal(checkpoint["model"][key].numpy(), basis[key]):
            raise ValueError("Checkpoint model/basis buffer mismatch: " + key)


class FollowupGateEvaluator(FrozenPredictorEvaluator):
    """Separate provenance binding; inherits only frozen evaluation contracts.

    This object retains no PortableSnapshot, episode cache, future observations,
    diagnostic evidence, or archive of deleted entries. Only verified frozen
    model/basis values and artifact paths for integrity checks are retained.
    """
    def __init__(self, snapshot, followup_root):
        followup_root = Path(followup_root).resolve()
        paths = {"summary": followup_root / "summary.json", "protocol": followup_root / "predeclared_protocol.json"}
        summary, protocol = (json.loads(paths[name].read_text()) for name in ("summary", "protocol"))
        protocol_sha = sha256(paths["protocol"])
        if (summary.get("status") != "completed_three_fixed_projection_arms"
                or summary.get("protocol_sha256") != protocol_sha or not summary.get("protocol_unchanged")
                or summary.get("test_arrays_read") != 0
                or protocol.get("backup_index_sha256") != BACKUP_INDEX_SHA256):
            raise ValueError("Require a completed verified followup with unchanged protocol")
        for split in ("train", "validation"):
            ids = sorted(r["id"] for r in snapshot.records if r["split"] == split)
            if protocol[split + "_episode_ids"] != ids:
                raise ValueError("Followup split identity differs from verified snapshot")
        entries = [entry for entry in summary["arms"] if entry["task"] == "gate" and entry["arm"] == "unit_columns"]
        if len(entries) != 1 or entries[0]["selected_epoch"] != 35:
            raise ValueError("Missing unique Gate unit-column selected epoch-35 result")
        entry = entries[0]
        arm_root = followup_root / "gate/unit_columns"
        artifacts = {path: sha256(path) for path in paths.values()}
        for name, expected in entry["artifact_sha256"].items():
            path = arm_root / name
            if path.parent != arm_root or sha256(path) != expected:
                raise ValueError("Followup arm artifact checksum mismatch: " + name)
            artifacts[path] = expected
        checkpoint_path, basis_path = arm_root / "selected.pt", arm_root / "basis.npz"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        with np.load(basis_path, allow_pickle=False) as archive:
            basis = {key: archive[key].copy() for key in archive.files}
        if (checkpoint["basis_sha256"] != sha256(basis_path)
                or checkpoint["validation_mse"] != entry["selected_validation_mse"]
                or checkpoint["state_sha256"] != entry["checkpoint_proofs"]["selected"]["state_sha256"]):
            raise ValueError("Selected checkpoint/basis/score binding mismatch")
        verify_selected_binding(checkpoint, basis, snapshot.basis("gate"), protocol_sha, snapshot.config["model"])
        curve = json.loads((arm_root / "learning_curve.json").read_text())
        best = min(curve, key=lambda row: row["validation"]["equal_mode_mean"]["mse"])
        if best["epoch"] != 35 or best["validation"]["equal_mode_mean"]["mse"] != checkpoint["validation_mse"]:
            raise ValueError("Checkpoint differs from declared earliest-minimum validation selection")
        for name in ("model.py", "inputs.py", "training.py"):
            relative = "experiments/predictor/control/src/phase1/" + name
            path = ROOT / relative
            if sha256(path) != protocol["source_sha256"][relative]:
                raise ValueError("Retained model source differs from trained checkpoint")
            artifacts[path] = sha256(path)
        used = summary["artifact_audit"]["loaded_file_sha256"]
        for relative in ("data/features/index.json", "data/features/normalization.npz",
                         "data/dataset/manifest.json", "runs/cpu-predictor-pilot/predictor/gate/basis.npz"):
            path = snapshot.verified_path(relative)
            if sha256(path) != used[relative]:
                raise ValueError("Followup and current snapshot input provenance differ")
            artifacts[path] = sha256(path)
        model = DynamicsPredictor(checkpoint["feature_dimension"], checkpoint["model_config"], (3, 6), basis)
        model.load_state_dict(checkpoint["model"], strict=True)
        self._model = model.cpu().eval().requires_grad_(False)
        self._basis = MappingProxyType({key: _immutable(basis[key]) for key in ("input_mean", "projection")})
        self.feature_dimension, self.task, self.seed, self.horizons = checkpoint["feature_dimension"], "gate", 17, (3, 6)
        self._artifacts = MappingProxyType(artifacts)
        self._initial_hash, self._initial_versions, self._calls = state_hash(model), _versions(model), {"count": 0}
        if self._initial_hash != checkpoint["state_sha256"]:
            raise ValueError("Loaded model state hash differs")
        self.provenance = EvaluatorProvenance(sha256(checkpoint_path), "fixed_projection_followup:gate:unit_columns:epoch35")
        self.binding = {"followup_protocol_sha256": protocol_sha, "selected_epoch": 35, "arm": "unit_columns",
                        "basis_sha256": sha256(basis_path), "cache_signature": snapshot.ix["signature"],
                        "normalization_sha256": snapshot.loaded["data/features/normalization.npz"]}
        self.audit()

    def audit(self):
        proof = super().audit()
        proof["explicit_followup_binding"] = dict(self.binding)
        return proof


def verify_saved_deletions(snapshot, evaluator, output):
    """Recompute every saved fork via the frozen model and retain prediction vectors."""
    checks = []
    for split in ("train", "validation"):
        with np.load(output / "scorer_inputs" / f"gate_{split}.npz", allow_pickle=False) as archive:
            inputs = {key: archive[key].copy() for key in archive.files}
        with np.load(output / "labels" / f"gate_{split}.npz", allow_pickle=False) as archive:
            labels = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads((output / "metadata" / f"gate_{split}.json").read_text())
        predictions, target_rows, losses = [], [], []
        for i, meta in enumerate(metadata):
            rec = snapshot.by_id[meta["episode_id"]]
            if rec["task"] != "gate" or rec["split"] != split:
                raise ValueError("Saved deletion metadata crossed task/split boundary")
            z, actions, q = snapshot.features(rec), snapshot.actions(rec), int(inputs["current_time"][i])
            candidates = tuple(MemoryEntry(rec["id"], int(t), value) for t, value in
                zip(inputs["candidate_timestamps"][i], inputs["candidate_representations"][i]))
            recent = tuple(MemoryEntry(rec["id"], int(t), value) for t, value in
                zip(inputs["recent_timestamps"][i], inputs["recent_representations"][i]))
            for item in candidates + recent:
                if not np.array_equal(item.representation, z[item.timestamp]):
                    raise ValueError("Saved scorer observation differs from source feature")
            parent = CausalParent(2, candidates, CausalContext(q, recent))
            past = {item.entry_id: value for item, value in zip(candidates + recent,
                np.concatenate([inputs["candidate_past_actions"][i], inputs["recent_past_actions"][i]]))}
            callback = evaluator.bind(parent, past)
            forks = []
            for deleted in range(3):
                kept = tuple(item for j, item in enumerate(candidates) if j != deleted)
                forks.append(np.stack([callback(kept, parent.context, h, actions[q:q+h].copy()) for h in (3, 6)]))
            pred = np.stack(forks)
            target = z[q + np.array([3, 6])]
            mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)[None]) ** 2, axis=-1)
            if not np.array_equal(mse, labels["feature_mse"][i]):
                raise ValueError("Saved deletion loss differs from complete frozen recomputation")
            predictions.append(pred); target_rows.append(target); losses.append(mse)
        pred_path = output / "labels" / f"gate_{split}_prediction_vectors.npz"
        np.savez_compressed(pred_path, parent_id=inputs["parent_id"], predictions=np.stack(predictions),
                            targets=np.stack(target_rows), recomputed_feature_mse=np.stack(losses))
        checks.append({"split": split, "parents": len(metadata), "forks": 3 * len(metadata),
                       "loss_values": 6 * len(metadata), "all_losses_recomputed_exactly": True,
                       "prediction_vectors_sha256": sha256(pred_path)})
    return checks


def run(snapshot, followup, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refuse to overwrite Gate continuation outputs")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    files = [Path(__file__).resolve()] + [ROOT / "experiments/phase2" / name for name in (
        "frozen_predictor.py", "causal_model_scores.py", "generate_deletion_pilot.py", "run_retention_baseline_pilot.py",
        "memory.py", "deletion.py", "training_samples.py", "portable_snapshot.py", "train_predictor_followup.py")]
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": "gate", "arm": "unit_columns",
        "selected_epoch": 35, "recent": 4, "capacity": 2, "horizons": [3, 6], "seed": 17,
        "baseline_split": "validation", "baseline_policies": list(POLICIES),
        "deletion_splits": ["train", "validation"], "deletion_parent_policies": ["fifo", "reservoir", "temporal_buckets", "visual_diversity"],
        "query_rule": "same declared query following continuous replay", "timeout_seconds": 300,
        "scorer_training": False, "test_arrays": False, "equal_auxiliary_bytes": False,
        "source_sha256": sources, "runtime_threads": runtime_threads()}
    write(output / "protocol.json", protocol)
    for path in files:
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    evaluator = FollowupGateEvaluator(snapshot, followup)
    scores = CausalModelScores(evaluator)
    records = [r for r in snapshot.records if r["task"] == "gate"]
    registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in records])
    baseline = output / "six_baselines"
    baseline.mkdir()
    rows, predictions = [], []
    with (baseline / "online_trace.jsonl").open("x") as stream:
        def trace(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        for rec in records:
            if rec["split"] != "validation":
                continue
            identity = EpisodeIdentity.from_record(rec)
            features, actions = snapshot.features(rec), snapshot.actions(rec)
            for policy in POLICIES:
                row, prediction = replay_episode(identity, features, actions, rec["query"], policy, scores, base_seed=17, trace=trace)
                row["checkpoint_sha256"] = evaluator.provenance.frozen_predictor_sha256
                row["followup_protocol_sha256"] = evaluator.binding["followup_protocol_sha256"]
                rows.append(row); predictions.append(prediction)
    np.savez_compressed(baseline / "prediction_vectors.npz", predictions=np.stack(predictions),
                        episode_id=np.array([r["episode_id"] for r in rows]), policy=np.array([r["policy"] for r in rows]))
    write(baseline / "episode_results.json", rows)
    baseline_summary = {"episode_policy_rows": len(rows), "horizon_losses": len(rows) * 2,
        "episodes": len({r["episode_id"] for r in rows}), "independent_pairs": len({r["group_id"] for r in rows}),
        "aggregate": aggregate(rows), "equal_retained_capacity": True, "equal_auxiliary_bytes": False,
        "surprise_semantics": "realized-action delayed h3 error, bounded pending queue; no future target as input",
        "memory_scope": "logical payload; allocator overhead and temporary workspaces excluded",
        "shared_predictor_bytes": sum(v.numel() * v.element_size() for v in list(evaluator._model.parameters()) + list(evaluator._model.buffers())),
        "model_audit": scores.audit()}
    write(baseline / "summary.json", baseline_summary)
    deletion = output / "deletion"
    deletion.mkdir()
    def load_episode(rec):
        if rec["task"] != "gate" or rec["split"] not in ("train", "validation"):
            raise ValueError("Only verified Gate train/validation inputs allowed")
        return snapshot.features(rec), snapshot.actions(rec)
    labels = generate_task(records, registry, load_episode, evaluator, "gate", deletion, base_seed=17)
    labels["complete_fork_recomputation"] = verify_saved_deletions(snapshot, evaluator, deletion)
    write(deletion / "summary.json", labels)
    final = {"status": "completed_gate_followup_phase2_development", "elapsed_seconds": time.monotonic() - started,
        "device": "cpu", "protocol_sha256": sha256(output / "protocol.json"), "binding": evaluator.binding,
        "checkpoint_sha256": evaluator.provenance.frozen_predictor_sha256,
        "six_baselines": baseline_summary, "deletion": labels, "final_model_audit": scores.audit(),
        "input_audit": snapshot.audit(), "source_sha256": sources,
        "scorer_trained": False, "test_arrays_read": 0, "scientific_phase2_completed": False,
        "baseline_model_planning_verified": False, "equal_auxiliary_bytes": False,
        "limitations": ["Gate only; Motion followup has no established validation history advantage.",
            "Small validation split reused for checkpoint/arm inspection; exploratory development only.",
            "Deleting an entry is evaluated by this predictor, not an independent ground-truth memory utility.",
            "K=2 is equal across policies; surprise carries extra bounded numeric context state."]}
    final["result_artifact_sha256"] = {str(path.relative_to(output)): sha256(path) for directory in (baseline, deletion)
                                        for path in directory.rglob("*") if path.is_file()}
    write(output / "summary.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def timeout(signum, frame):
        raise TimeoutError("Gate continuation exceeded 300 seconds")
    old = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 300)
    owns_output = not args.output.exists() or not any(args.output.iterdir())
    try:
        result = run(PortableSnapshot(args.snapshot), args.followup, args.output)
        print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                          "baseline_rows": result["six_baselines"]["episode_policy_rows"],
                          "deletion_parents": result["deletion"]["unique_parents"]}), flush=True)
    except BaseException as error:
        if owns_output and args.output.exists():
            write(args.output / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


if __name__ == "__main__":
    main()
