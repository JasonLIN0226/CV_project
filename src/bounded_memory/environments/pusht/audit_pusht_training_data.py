"""Summarize saved TRAIN-only PushT interactions; no simulator/model execution.

Validation and evaluation diagnostics are never opened. The collection protocol
and each training diagnostic checksum are checked against the saved manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_training_data(root: Path) -> dict:
    started = time.perf_counter()
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "preparation_manifest.json").read_text())
    if manifest["status"] != "CPU_RGB_action_preparation_completed":
        raise ValueError("Require a completed preparation manifest")
    if sha256(root / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Protocol checksum mismatch")
    records = [spec for spec in protocol["episodes"] if spec["split"] == "train"]
    if len(records) != 32 or len({r["id"] for r in records}) != 32:
        raise ValueError("Expected the fixed 32 distinct training episodes")
    rows, inputs = [], {}
    for spec in records:
        if re.fullmatch(r"pusht_train_\d{3}", spec["id"]) is None:
            raise ValueError("Training diagnostic ID is not in the declared train namespace")
        relative = f"evaluation_only/{spec['id']}.json"
        path = root / relative
        artifact = manifest["artifacts"][relative]
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError("Training diagnostic checksum mismatch")
        inputs[relative] = artifact["sha256"]
        trace = json.loads(path.read_text())
        if trace["episode_id"] != spec["id"]:
            raise ValueError("Training diagnostic episode identity mismatch")
        steps = trace["steps"]
        poses = np.array([s["info"]["block_pose"] for s in steps], dtype=np.float64)
        if poses.shape != (len(steps), 3) or not np.isfinite(poses).all():
            raise ValueError("Invalid diagnostic block poses")
        displacement = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
        angle = np.abs((poses[:, 2] - poses[0, 2] + np.pi) % (2*np.pi) - np.pi)
        contacts = [int(s["info"].get("n_contacts", 0)) for s in steps]
        coverage = [float(s["info"].get("coverage", 0)) for s in steps]
        rows.append({
            "episode_id": spec["id"], "observations": len(steps),
            "any_contact": bool(max(contacts) > 0), "steps_with_contact": sum(v > 0 for v in contacts),
            "nonzero_coverage": bool(max(coverage) > 0), "max_coverage": max(coverage),
            "success_seen": any(bool(s["info"].get("is_success", False)) for s in steps),
            "max_block_translation_from_initial": float(displacement.max()),
            "max_wrapped_block_rotation_from_initial_radians": float(angle.max()),
            "paired_diagnostic_max_abs_difference": trace["paired_diagnostic_max_abs_difference"],
            "later_visible_targets": len(trace["later_visible_targets"]),
        })
    return {
        "status": "training_diagnostics_only", "protocol_sha256": sha256(root / "protocol.json"),
        "preparation_manifest_sha256": sha256(root / "preparation_manifest.json"),
        "audit_source_sha256": sha256(Path(__file__)), "inputs_sha256": inputs,
        "read_splits": ["train"], "validation_diagnostics_read": False,
        "evaluation_diagnostics_read": False, "protocol_changed": False, "actions_retuned": False,
        "episodes": len(rows), "episodes_with_contact": sum(r["any_contact"] for r in rows),
        "episodes_with_nonzero_coverage": sum(r["nonzero_coverage"] for r in rows),
        "episodes_with_block_translation_gt_1e_6": sum(r["max_block_translation_from_initial"] > 1e-6 for r in rows),
        "episodes_with_block_rotation_gt_1e_6": sum(r["max_wrapped_block_rotation_from_initial_radians"] > 1e-6 for r in rows),
        "episodes_with_success_seen": sum(r["success_seen"] for r in rows),
        "total_steps_with_contact": sum(r["steps_with_contact"] for r in rows),
        "median_max_block_translation_from_initial": float(np.median([r["max_block_translation_from_initial"] for r in rows])),
        "median_max_wrapped_block_rotation_from_initial_radians": float(np.median([r["max_wrapped_block_rotation_from_initial_radians"] for r in rows])),
        "max_paired_diagnostic_abs_difference_on_train": max(r["paired_diagnostic_max_abs_difference"] for r in rows),
        "train_episodes_with_no_later_visible_target": sum(r["later_visible_targets"] == 0 for r in rows),
        "rows": rows, "elapsed_seconds": time.perf_counter()-started,
        "interpretation": "Describes interactions under the fixed collection policy only. It does not measure learned prediction/planning, prove action identifiability, or certify sufficient dataset diversity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; existing audit evidence is preserved")
    result = audit_training_data(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "inputs_sha256")}, indent=2))


if __name__ == "__main__":
    main()
