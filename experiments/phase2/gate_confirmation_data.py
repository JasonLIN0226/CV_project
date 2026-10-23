"""Strict, read-only loader for the separately seeded literature confirmation.

The development loader remains unchanged. Only its verified array-loading and
normalization methods are reused; this constructor binds the new scope/seed and
checks disjointness against both old and previously inspected scene metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.phase2.encode_fresh_gate_diagnostic import (
    CONDITIONS, DATA_CONFIG, ENCODER_CONFIG, EXPECTED_ENCODER_METADATA_SHA256,
    EXPECTED_SOURCE_SHA256, OLD_NORMALIZATION_SHA256, PAIRS_PER_CONDITION, sha256,
)
from experiments.phase2.run_fresh_literature_diagnostic import (
    FreshGateDataset, validate_fresh_records,
)

SEED = 20261023
SCOPE = "literature_confirmation"
ROOT = Path(__file__).resolve().parents[2]


def validate_confirmation_records(records, prior_records):
    """Reuse structural checks, while requiring the new scope before mapping."""
    if not isinstance(records, list) or any(r.get("split") != SCOPE for r in records):
        raise ValueError("Require literature_confirmation records")
    # The old validator is a pure metadata check. Do not mutate archived records.
    validate_fresh_records([{**r, "split": "fresh_development"} for r in records])
    for field in ("id", "pair", "seed"):
        if {r[field] for r in records}.intersection(r[field] for r in prior_records):
            raise ValueError("Confirmation overlaps prior scenes: " + field)
    return {r["id"]: r for r in records}


class ConfirmationGateDataset(FreshGateDataset):
    def __init__(self, root, *, encoder_sha256, prior_records):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if (m.get("status") != "completed" or m.get("scope") != SCOPE
                or m.get("seed") != SEED or m.get("data_config") != {**DATA_CONFIG, "seed": SEED}
                or m.get("conditions") != CONDITIONS or m.get("pairs_per_condition") != PAIRS_PER_CONDITION
                or m.get("episodes") != 192 or m.get("independent_pairs") != 96
                or m.get("old_normalization_sha256") != OLD_NORMALIZATION_SHA256
                or m.get("encoder_metadata_sha256") != EXPECTED_ENCODER_METADATA_SHA256
                or m.get("source_sha256") != EXPECTED_SOURCE_SHA256
                or m.get("helper_sha256") != encoder_sha256
                or m.get("old_test_arrays_opened") != 0 or not m.get("exact_shared_rgb_features")
                or any(m.get(k) is not False for k in
                       ("normalization_fitted", "basis_fitted", "predictor_trained", "student_trained"))):
            raise ValueError("Confirmation manifest violates frozen protocol")
        self.loaded = {}
        self.array_files_opened = []
        for relative in m["artifact_sha256"]:
            self.path(relative)
        audit = self.read_json("rgb_pair_audit.json")
        hardware = self.read_json("hardware.json")
        encoder = self.read_json("frozen_encoder_audit.json")
        if (not audit.get("passed") or audit.get("old_scene_seed_overlap") != 0
                or audit.get("prior_development_seed_overlap") != 0
                or hardware != m["hardware"] or "A100" not in hardware["torch_device_name"]
                or encoder.get("actual_gpu_uuid") != hardware["actual_gpu_uuid"]
                or encoder.get("metadata_sha256") != EXPECTED_ENCODER_METADATA_SHA256
                or encoder.get("trainable_parameters") != 0 or not encoder.get("all_modules_eval")
                or encoder["metadata"].get("resolved_revision") != ENCODER_CONFIG["revision"]):
            raise ValueError("Confirmation RGB/GPU/frozen-encoder audit failed")
        self.records = self.read_json("records.json")
        self.by_id = validate_confirmation_records(self.records, prior_records)
        if any(max(r["rgb_indices"]) >= m["unique_rgb_frames"] for r in self.records):
            raise ValueError("RGB expansion index exceeds confirmation bank")
        self._episodes = {}
        self._normalized_bank = None

    def audit(self):
        audit = super().audit()
        audit["data_seed"] = SEED
        audit["scope"] = SCOPE
        audit["prior_scene_identity_pair_seed_overlap"] = 0
        return audit
