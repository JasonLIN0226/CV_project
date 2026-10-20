"""Verified fresh-pilot predictor binding for offline equal-K deletion labels.

The callback retains only its frozen model, the immutable bounded parent, and
per-resident executed-action summaries. No episode feature/action store or
future targets enter that closure. No weights are fitted here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from .deletion import CausalParent, EvaluatorProvenance
from .train_predictor_pilot import ROOT, read, sha256, verify_inputs, verify_predictor_contract
from phase1.inputs import collate
from phase1.model import DynamicsPredictor, project


def _immutable(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0" + str(a.dtype).encode() + repr(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _versions(model):
    return tuple((name, id(value), value._version) for name, value in
                 list(model.named_parameters()) + list(model.named_buffers()))


class FrozenPredictorEvaluator:
    """Load a completed new CPU training run, not a historical external model.

    predictor_root is the top-level directory containing run_manifest.json and
    training_result.json. ``provenance`` is an EvaluatorProvenance instance.
    Call audit() again after label generation to verify state/artifact identity.
    """

    def __init__(self, predictor_root: Path, encoded_report: Path, data_root: Path,
                 task: str, seed: int = 17):
        if task not in ("gate", "motion") or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Require gate/motion and a nonnegative integer model seed")
        predictor_root, encoded_report, data_root = map(lambda p: Path(p).resolve(),
                                                       (predictor_root, encoded_report, data_root))
        run_path, result_path = predictor_root / "run_manifest.json", predictor_root / "training_result.json"
        run, result = read(run_path), read(result_path)
        if (run.get("status") != "completed" or run.get("worker_exit_code") != 0
                or result.get("status") != "new_cpu_predictor_pilot_completed"
                or not result.get("predictor_trained") or not result.get("originals_preserved")):
            raise ValueError("Require a completed, successful new predictor pilot")
        if sha256(result_path) != run.get("training_result_sha256"):
            raise ValueError("Training result checksum differs from completed run manifest")
        config, observation_index, _, cache = verify_inputs(data_root, encoded_report)
        contract = verify_predictor_contract(config)
        rows = [row for row in result["tasks"] if row.get("task") == task and row.get("seed") == seed]
        if len(rows) != 1:
            raise ValueError("Completed result must identify exactly one requested task/seed")
        row = rows[0]
        if row.get("retained_predictor_contract") != contract or not row.get("basis_train_only"):
            raise ValueError("Training predictor source/contract or basis provenance differs")
        artifacts = {run_path: sha256(run_path), result_path: sha256(result_path)}
        for name in ("manifest.json", "encoder_config.json", "sample_summary.json"):
            path = encoded_report / name
            artifacts[path] = sha256(path)
        sources = run.get("source_sha256", {})
        if not all("experiments/predictor/control/src/phase1/" + name in sources for name in
                   ("model.py", "inputs.py", "training.py")):
            raise ValueError("Training manifest is missing retained predictor source hashes")
        for relative, expected in sources.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or sha256(path) != expected:
                raise ValueError("Training source changed: " + relative)
            artifacts[path] = expected
        for relative in ("resolved_config.json", "dataset/manifest.json", "features/index.json", "features/normalization.npz"):
            path = data_root / relative
            expected = result.get("original_artifact_sha256", {}).get(str(path))
            if expected is None or sha256(path) != expected:
                raise ValueError("Training input provenance changed: " + relative)
            artifacts[path] = expected
        path = predictor_root / "predictor" / task / f"seed_{seed}" / "frozen.pt"
        expected = row["frozen_checkpoint_sha256"]
        if sha256(path) != expected:
            raise ValueError("Frozen checkpoint checksum mismatch")
        artifacts[path] = expected
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        signature = checkpoint["signature"]
        basis_path = predictor_root / "predictor" / task / "basis.npz"
        training_config_path = predictor_root / "training_config.json"
        training_config = read(training_config_path)
        artifacts[training_config_path] = sha256(training_config_path)
        if (checkpoint.get("checkpoint_kind") != "new_frozen_predictor_for_future_phase2_use"
                or signature.get("architecture") != "phase1.model.DynamicsPredictor"
                or signature.get("task") != task or signature.get("seed") != seed
                or signature.get("cache_signature") != cache["signature"]
                or signature.get("normalization_sha256") != artifacts[data_root / "features/normalization.npz"]
                or signature.get("basis_sha256") != sha256(basis_path)
                or tuple(checkpoint["horizons"]) != (3, 6) or signature.get("horizons") != [3, 6]
                or checkpoint["model_config"] != signature["model_config"]
                or checkpoint["model_config"] != config["model"]
                or training_config["model"] != checkpoint["model_config"]
                or training_config["training"] != signature["training"]
                or checkpoint["selected_epoch"] != row["best_epoch"]
                or checkpoint["validation_mse"] != row["best_validation_mse"]):
            raise ValueError("Frozen predictor signature/config/cache/basis binding mismatch")
        artifacts[basis_path] = signature["basis_sha256"]
        records = observation_index["records"]
        for split, key in (("train", "training_episode_ids"), ("validation", "validation_episode_ids")):
            expected_ids = sorted(r["id"] for r in records if r["task"] == task and r["split"] == split)
            if signature.get(key) != expected_ids:
                raise ValueError("Frozen predictor split membership mismatch")
        with np.load(basis_path, allow_pickle=False) as archive:
            basis = {name: archive[name].copy() for name in archive.files}
        if sorted(basis["train_ids"].tolist()) != signature["training_episode_ids"]:
            raise ValueError("Stored basis is not bound to the declared train split")
        self.feature_dimension = int(checkpoint["feature_dimension"])
        if self.feature_dimension != len(basis["input_mean"]):
            raise ValueError("Feature dimension differs from the stored basis")
        for name in ("input_mean", "projection", "target_mean", "output_basis"):
            if not torch.equal(checkpoint["model"][name], torch.from_numpy(basis[name])):
                raise ValueError("Checkpoint basis buffer differs from verified basis artifact")
        model = DynamicsPredictor(self.feature_dimension, checkpoint["model_config"], (3, 6), basis)
        model.load_state_dict(checkpoint["model"], strict=True)
        self._model = model.cpu().eval().requires_grad_(False)
        self._basis = MappingProxyType({name: _immutable(basis[name]) for name in ("input_mean", "projection")})
        self._artifacts = MappingProxyType(artifacts)
        self._initial_hash = _state_hash(model)
        self._initial_versions = _versions(model)
        self._calls = {"count": 0}
        self.task, self.seed, self.horizons = task, seed, (3, 6)
        self.provenance = EvaluatorProvenance(expected, "fresh_cpu_dynamics_predictor_v1:" + task + ":seed_" + str(seed))
        self.audit()

    def _ensure_frozen(self):
        if (any(module.training for module in self._model.modules())
                or any(p.requires_grad or p.device.type != "cpu" for p in self._model.parameters())
                or _versions(self._model) != self._initial_versions):
            raise ValueError("Frozen predictor mode, device, parameters, or buffers changed")

    def audit(self) -> dict:
        self._ensure_frozen()
        if _state_hash(self._model) != self._initial_hash:
            raise ValueError("Frozen predictor state values changed")
        if any(sha256(path) != expected for path, expected in self._artifacts.items()):
            raise ValueError("Verified predictor/source/data artifact changed")
        return {"passed": True, "task": self.task, "seed": self.seed, "device": "cpu",
                "checkpoint_sha256": self.provenance.frozen_predictor_sha256,
                "state_sha256": self._initial_hash, "parameter_buffer_identity_unchanged": True,
                "all_modules_eval": True, "trainable_parameters": 0, "model_calls": self._calls["count"],
                "feature_dimension": self.feature_dimension, "horizons": list(self.horizons),
                "artifact_sha256": {str(path): value for path, value in self._artifacts.items()}}

    def bind(self, parent: CausalParent, past_action_summaries: Mapping) -> callable:
        """Bind float32 shape-(2,) executed additive-action sums for every resident.

        Summary keys must be exactly all recent and K+1 candidate entry IDs.
        Values are copied into immutable storage. This method does not receive
        the full executed-action sequence, feature cache, or future targets.
        """
        self._ensure_frozen()
        if not isinstance(parent, CausalParent) or parent.capacity != 2 or len(parent.context.recent) != 4:
            raise ValueError("Require the audited K=2, recent=4 bounded parent")
        entries = parent.context.recent + parent.candidates
        if any(e.representation.dtype != np.float32 or e.representation.shape != (self.feature_dimension,) for e in entries):
            raise ValueError("Parent must use normalized float32 checkpoint feature vectors")
        expected = {entry.entry_id for entry in entries}
        if set(past_action_summaries) != expected:
            raise ValueError("Executed-action summaries must match exactly the bounded resident entries")
        summaries = {}
        for key in expected:
            value = np.asarray(past_action_summaries[key])
            if value.dtype != np.float32 or value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("Executed-action summaries require finite float32 vectors of shape (2,)")
            summaries[key] = _immutable(value)
        if not np.array_equal(summaries[parent.context.recent[-1].entry_id], np.zeros(2, np.float32)):
            raise ValueError("Current observation must have zero executed-action suffix")
        summaries = MappingProxyType(summaries)
        by_id = MappingProxyType({entry.entry_id: entry for entry in parent.candidates})
        model, basis, calls, state_versions = self._model, self._basis, self._calls, self._initial_versions
        dimension = self.feature_dimension
        recent, current_time = parent.context.recent, parent.context.current_time

        def callback(kept, context, horizon, action_prefix):
            if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon not in (3, 6):
                raise ValueError("Only the trained horizons 3 and 6 are supported")
            if (context.current_time != current_time or len(context.recent) != len(recent)
                    or any(a.entry_id != b.entry_id or not np.array_equal(a.representation, b.representation)
                           for a, b in zip(context.recent, recent))):
                raise ValueError("Callback recent context differs from its immutable parent")
            kept = tuple(kept)
            if len(kept) != 2 or len({entry.entry_id for entry in kept}) != 2:
                raise ValueError("Every deletion must retain exactly two distinct real entries")
            for entry in kept:
                original = by_id.get(entry.entry_id)
                if (original is None or entry.representation.dtype != original.representation.dtype
                        or not np.array_equal(entry.representation, original.representation)):
                    raise ValueError("Retained entry differs from the bound candidate set")
            prefix = np.asarray(action_prefix)
            if prefix.shape != (horizon, 2) or prefix.dtype != np.float32 or not np.isfinite(prefix).all():
                raise ValueError("Callback must receive only the finite float32 horizon-length action prefix")
            if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()) or _versions(model) != state_versions:
                raise ValueError("Predictor changed after binding")
            tokens = recent + kept
            z = np.stack([entry.representation for entry in tokens])
            actions = np.zeros((6, 2), np.float32)
            actions[:horizon] = prefix
            sample = {"z": z, "age": np.array([current_time - e.timestamp for e in tokens], np.float32),
                      "type": np.array([0] * 4 + [1] * 2, np.int64),
                      "past_actions": np.stack([summaries[e.entry_id] for e in tokens]),
                      "actions": actions, "current": recent[-1].representation.copy(),
                      "projected_z": project(z, basis), "projected_current": project(recent[-1].representation, basis)}
            with torch.inference_mode():
                value = model(collate([sample], "cpu"))[0, 0 if horizon == 3 else 1].cpu().numpy().copy()
            if _versions(model) != state_versions:
                raise ValueError("Predictor state changed during forward")
            if value.shape != (dimension,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Invalid frozen predictor output")
            calls["count"] += 1
            return value

        return callback
