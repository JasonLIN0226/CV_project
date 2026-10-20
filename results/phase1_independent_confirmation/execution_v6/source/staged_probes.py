"""Preparation: original train-only readouts, frozen before confirmation tests."""
import hashlib
from pathlib import Path

import numpy as np

from phase1 import probes
from phase1.utils import read_json, write_json, file_hash
from staged_data import (bind, TRAIN_SPLITS, StageFeatureStore,
                         fit_stage_normalization, verify_permit,
                         verify_training_subset)


def array_identity(a):
    x = np.ascontiguousarray(a)
    return {"dtype": str(x.dtype), "shape": list(x.shape),
            "sha256": hashlib.sha256(x.tobytes()).hexdigest()}


def fitting_identity(train, validation, classification):
    return {"classification": bool(classification),
            "train": [array_identity(a) for a in train[:2]],
            "validation": [array_identity(a) for a in validation[:2]]}


def role_names(c):
    names = []
    for task in c["data"]["tasks"]:
        variants = ["gate_cue", "recent_only"] if task == "gate" else [
            "motion_two_frame", "motion_single_start", "motion_single_end", "recent_only"]
        names += [f"{task}_{v}.npz" for v in variants]
        names.append(f"{task}_state_decoder.npz")
    return names


def fit_training_readouts(c, permit=None, *, fixture=False):
    verify_permit(c, permit, fixture=fixture)
    root = Path(c["output"])/"probes"
    if root.exists():
        raise ValueError("Refuse implicit readout refitting/retry")
    inputs = []
    def choose(train, validation, classification):
        inputs.append(fitting_identity(train, validation, classification))
        return probes.choose_readout(c, train, validation, classification)
    def write_stage(path, value):
        if Path(path) != root/"metrics.json":
            raise ValueError("Unexpected original probe write")
        write_json(root/"train_validation_metrics.json", value)
    result = bind(probes.probe, SPLITS=TRAIN_SPLITS,
                  fit_normalization=fit_stage_normalization,
                  FeatureStore=StageFeatureStore,
                  choose_readout=lambda _c, a, b, classify: choose(a, b, classify),
                  write_json=write_stage)(c)
    names = role_names(c)
    if len(inputs) != len(names) or set(result["checkpoints"]) != set(names):
        raise ValueError("Incomplete original readout role order")
    lock = {"status": "TRAIN_VALIDATION_READOUTS_FROZEN",
            "test_opened": False, "fitting_inputs": dict(zip(names, inputs)),
            "checkpoints": result["checkpoints"],
            "basis_sha256": result["input_basis_sha256"],
            "normalization_sha256": result["normalization_sha256"],
            "cache_signature": result["cache_signature"],
            "original_probe_code_hash": result["code_hash"]}
    write_json(root/"train_validation_readout_lock.json", lock)
    return lock


class FrozenReadout:
    """Allow original evaluation loop to call save without rewriting fit weights."""
    def __init__(self, path, sha):
        self.path, self.sha = Path(path), sha
        if file_hash(self.path) != sha:
            raise ValueError("Frozen readout changed")
        self.model = probes.LinearReadout.load(path)
        self.ridge = self.model.ridge

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        if Path(path) != self.path or file_hash(self.path) != self.sha:
            raise ValueError("Evaluation attempted to replace a frozen readout")


def evaluate_frozen_readouts(c, subset, permit=None, *, fixture=False):
    verify_permit(c, permit, models_locked=True, fixture=fixture)
    verify_training_subset(c, subset)
    root = Path(c["output"])/"probes"
    lock = read_json(root/"train_validation_readout_lock.json")
    if lock["status"] != "TRAIN_VALIDATION_READOUTS_FROZEN" or lock["test_opened"]:
        raise ValueError("Original training-stage lock absent")
    if (file_hash(Path(c["output"])/"features/normalization.npz") != lock["normalization_sha256"]
            or probes.code_hash("probes.py") != lock["original_probe_code_hash"]):
        raise ValueError("Transform or original probe changed")
    for task, sha in lock["basis_sha256"].items():
        if file_hash(Path(c["output"])/"predictor"/task/"basis.npz") != sha:
            raise ValueError("Frozen train-only basis changed")
    names, selected = role_names(c), []
    def load_frozen(_c, train, validation, classification):
        name = names[len(selected)]
        if fitting_identity(train, validation, classification) != lock["fitting_inputs"][name]:
            raise ValueError("Training/validation fit inputs changed")
        selected.append(name)
        return FrozenReadout(root/name, lock["checkpoints"][name])
    # Exact original metrics, all original splits; selection is replaced only by
    # loading already fitted train/validation readouts after input-hash equality.
    result = bind(probes.probe, choose_readout=load_frozen)(c)
    if selected != names or result["checkpoints"] != lock["checkpoints"]:
        raise ValueError("Missing or changed frozen final readouts")
    write_json(root/"frozen_readout_final_verification.json",
               {"passed": True, "readout_count": len(names), "refitted": False,
                "all_fitting_input_hashes_exact": True,
                "training_lock_sha256": file_hash(root/"train_validation_readout_lock.json")})
    return result
