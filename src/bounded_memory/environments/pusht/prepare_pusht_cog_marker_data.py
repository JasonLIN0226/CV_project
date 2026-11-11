"""Collect the marker hidden-CoG PushT corpus (design v5).

Each family runs one shared action plan under every CoG class (see
pusht_cog_marker.py). Siblings differ only in the marker frames before the first
test contact. Siblings are stored as s0/s1/s2 under a seeded shuffle; the class
map, the marker frames (where the cue is), simulator state and checks live only
in evaluation_only/.

Selection: candidate seeds are tried in order per split until the split has its
family count; a family is rejected if any class fails validity (contact before the
test push, leaving the arena, siblings differing outside the marker frames, test
contact before the query). Class-symmetric; never on test outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_cog_marker as cue
from prepare_pusht_cog_v4_data import (FIELDS, _runtime_provenance, _safe_path, array_sha, json_sha,
                                       sha256, sibling_classes, write_json)
from pusht import normalize_actions

SPLIT_COUNTS = {"train": 128, "validation": 32, "evaluation": 32}  # families
SPLIT_BASE_SEEDS = {"train": 2026161000, "validation": 2026162000, "evaluation": 2026163000}
TEST_ONLY_SEEDS = (2026159000, 2026159029)  # used by tests/test_pusht_cog_marker.py
MAX_CANDIDATES_PER_SPLIT = 999
RECENT_LENGTH = cue.RECENT_LENGTH


def collect_candidate(job):
    split, seed = job
    spec = cue.sample_family(seed)
    result, runs, actions, phase = cue.check_family(spec)
    if not result["valid"]:
        reasons = [k for k, bad in {
            "pre_test_contact": result["pre_test_contacts"] > 0,
            "siblings_differ_outside_marker": result["frames_differing_before_test_contact"] != result["marker_frames"],
            "out_of_bounds": not (result["block_in_bounds"] and result["actions_in_arena"]),
            "test_contact_before_query": result["first_test_contact_frame"] <= result["query_frame"],
        }.items() if bad]
        return {"split": split, "seed": seed, "valid": False, "reasons": reasons}
    return {"split": split, "seed": seed, "valid": True, "result": result, "runs": runs,
            "actions": actions, "phase": phase}


def write_family(output: Path, family_id: str, cand: dict, classes: list[str]):
    result, actions = cand["result"], cand["actions"].astype(np.float32)
    frames = {k: result[k] for k in ("test_start_frame", "query_frame", "target_frame")}
    write_json(output / "evaluation_only" / f"{family_id}_checks.json", result)
    records = []
    for k, cog in enumerate(classes):
        sibling_id = f"{family_id}_s{k}"
        run = cand["runs"][cog]
        archive = {"rgb": run["rgb"], "timestamps": np.arange(len(actions) + 1, dtype=np.int64),
                   "actions_absolute": actions, "actions_normalized": normalize_actions(actions)}
        path = output / "observations" / f"{sibling_id}.npz"
        np.savez_compressed(path, **archive)
        write_json(output / "evaluation_only" / f"{sibling_id}.json", {
            "episode_id": sibling_id, "family_id": family_id, "cog_class": cog,
            "cog_body_local": list(cue.COG_CLASSES[cog]), "phase": cand["phase"],
            "marker_frames": result["marker_frames"], "marker_px": run["marker_px"],
            "block_pose": run["pose"].tolist(), "block_keypoints": run["keypoints"].tolist(),
            "pusher_position": run["agent"].tolist(), "contact_points": run["contacts"].tolist(),
        })
        records.append({
            "id": sibling_id, "family_id": family_id, "split": cand["split"],
            "environment_seed": cand["seed"], "frames": len(run["rgb"]), **frames,
            "observations": str(path.relative_to(output)), "observations_sha256": sha256(path),
            "actions_sha256": array_sha(actions),
        })
    return records


def audit_dataset(output: Path, protocol: dict, records: list[dict]):
    """Structural checks on model-facing archives only; never reads evaluation_only."""
    by_family, counts, model_bytes = {}, {}, 0
    for record in records:
        path = _safe_path(output, record["observations"], "observations")
        if sha256(path) != record["observations_sha256"]:
            raise ValueError("Observation checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FIELDS:
                raise ValueError("Forbidden/missing fields in model archive")
            data = {k: archive[k] for k in FIELDS}
        frames = record["frames"]
        if data["rgb"].shape != (frames, *protocol["observation_shape"]) or data["rgb"].dtype != np.uint8:
            raise ValueError("Bad RGB shape/dtype")
        if not np.array_equal(data["timestamps"], np.arange(frames, dtype=np.int64)):
            raise ValueError("Bad timestamps")
        if data["actions_absolute"].dtype != np.float32 or array_sha(data["actions_absolute"]) != record["actions_sha256"]:
            raise ValueError("Actions differ from the index")
        if not np.array_equal(data["actions_normalized"], normalize_actions(data["actions_absolute"])):
            raise ValueError("Wrong action normalization")
        model_bytes += sum(v.nbytes for v in data.values())
        by_family.setdefault(record["family_id"], []).append((record, data))
        c = counts.setdefault(record["split"], {"families": 0, "episodes": 0, "frames": 0})
        c["episodes"] += 1
        c["frames"] += frames
    if len({r["environment_seed"] for r in records}) != len(by_family):
        raise ValueError("Environment seeds repeat across families")
    for family_id, items in by_family.items():
        r0, d0 = items[0]
        counts[r0["split"]]["families"] += 1
        if len(items) != len(cue.COG_CLASSES):
            raise ValueError(f"Family {family_id} does not have one episode per class")
        window_start = r0["query_frame"] - RECENT_LENGTH + 1
        for r, d in items[1:]:
            if any(r[k] != r0[k] for k in ("frames", "test_start_frame", "query_frame", "target_frame", "split")):
                raise ValueError(f"Sibling frame indices differ in {family_id}")
            if not np.array_equal(d0["actions_absolute"], d["actions_absolute"]):
                raise ValueError(f"Sibling actions differ in {family_id}")
            differ = [t for t in range(r0["test_start_frame"]) if not np.array_equal(d0["rgb"][t], d["rgb"][t])]
            if len(differ) > cue.MARKER_FRAMES or any(t >= window_start for t in differ):
                raise ValueError(f"Siblings differ outside a short early window in {family_id}")
    if any(counts.get(s, {}).get("families") != n for s, n in protocol["split_family_counts"].items()):
        raise ValueError("Split family counts differ from protocol")
    return {"passed": True, "counts": counts, "model_archive_uncompressed_bytes": model_bytes,
            "diagnostics_loaded": False,
            "checks": ["archive_field_whitelist", "rgb_shape_dtype", "timestamps", "actions_match_index_hash",
                       "absolute_action_normalization", "checksums", "family_seed_unique",
                       "one_episode_per_class", "sibling_frame_indices_equal", "sibling_actions_identical",
                       "siblings_differ_in_at_most_MARKER_FRAMES_frames_before_the_recent_window", "split_counts"]}


def run(output: Path, processes: int):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing data will not be overwritten")
    protocol = {
        "schema_version": 5, "protocol_id": "pusht-hidden-cog-marker-v5-20261109",
        "purpose": "Gate-like PushT: the hidden CoG is shown as a dot for a few early frames; the future "
                   "depends on it through a later test push",
        "design": "no probe; block untouched until the test push; one shared action plan per family; "
                  "siblings differ only in the marker frames before the first test contact",
        "split_family_counts": SPLIT_COUNTS.copy(), "siblings_per_family": len(cue.COG_CLASSES),
        "split_base_seeds": SPLIT_BASE_SEEDS, "test_only_seeds_excluded": list(TEST_ONLY_SEEDS),
        "selection": "candidate seeds base, base+1, ... per split until the count is reached; a family is "
                     "rejected if any class fails validity (class-symmetric); never on test outcomes",
        "environment": "gym_pusht/PushT-v0", "observation_shape": [96, 96, 3],
        "observation_source": "pixels returned by reset/step; marker drawn on the 96x96 observation afterwards",
        "control_hz": 10, "physics_dt_seconds": .01,
        "action_semantics": "absolute agent target XY in [0,512], ordered; not delta or velocity",
        "normalization": "actions_absolute / float32(256) - float32(1)",
        "cog_classes_body_local": cue.COG_CLASSES,
        "marker": {"frames": cue.MARKER_FRAMES, "start_range_inclusive": list(cue.MARKER_START_RANGE),
                   "radius_px_at_96": cue.MARKER_RADIUS_PX, "rgb": list(cue.MARKER_COLOR),
                   "position": "block CoG in world coordinates mapped to observation pixels"},
        "start_center_range": list(cue.START_RANGE), "wander_steps": cue.WANDER_STEPS,
        "test": f"crossbar top face pushed toward the stem; offset uniform +-{cue.TEST_OFFSET_MAX:g} px",
        "recent_length": RECENT_LENGTH, "horizon": cue.HORIZON,
        "query_definition": "last gap frame; the recent window is query-3..query",
        "class_assignment": "seeded shuffle per family seed; map stored only in evaluation_only/sibling_classes.json",
        "privileged_fields": "marker frames and positions are oracle information; evaluation_only only",
        "evaluation_role": "reserved families; validity checks only, no outcome-based tuning",
        "encoder_run": False, "predictor_trained": False, "planning_run": False,
    }
    provenance = _runtime_provenance()
    provenance["source_sha256"]["experiments/phase2/pusht_cog_marker.py"] = sha256(
        Path(__file__).with_name("pusht_cog_marker.py"))
    provenance["source_sha256"]["experiments/phase2/prepare_pusht_cog_marker_data.py"] = sha256(Path(__file__))
    output.mkdir(parents=True)
    for name in ("observations", "evaluation_only"):
        (output / name).mkdir()
    write_json(output / "protocol.json", protocol)
    protocol_hash = sha256(output / "protocol.json")
    write_json(output / "protocol_frozen_before_collection.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
        "before_environment_creation": True, **provenance})

    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    records, rejected, assignment, train_checks = [], [], {}, []
    with multiprocessing.get_context("fork").Pool(processes) as pool:
        for split, count in SPLIT_COUNTS.items():
            accepted, next_k = 0, 0
            while accepted < count:
                batch = [(split, SPLIT_BASE_SEEDS[split] + k) for k in range(next_k, next_k + processes)]
                next_k += processes
                if next_k > MAX_CANDIDATES_PER_SPLIT:
                    raise AssertionError(f"Too many rejections in {split}")
                for cand in pool.map(collect_candidate, batch):  # seed order -> deterministic acceptance
                    if accepted == count:
                        break
                    if not cand["valid"]:
                        rejected.append({k: cand[k] for k in ("split", "seed", "reasons")})
                        continue
                    family_id = f"cog_{split}_{accepted:03d}"
                    classes = sibling_classes(cand["seed"])
                    assignment[family_id] = classes
                    records += write_family(output, family_id, cand, classes)
                    if split == "train":
                        train_checks.append(cand["result"])
                    accepted += 1
            print(f"{split}: {count} families, {sum(r['split'] == split for r in rejected)} rejected, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    collection_seconds = time.perf_counter() - started
    write_json(output / "evaluation_only" / "sibling_classes.json", assignment)
    write_json(output / "rejected_candidates.json", rejected)
    if sha256(output / "protocol.json") != protocol_hash:
        raise AssertionError("Frozen protocol changed during collection")
    write_json(output / "observation_index.json", {
        "schema_version": 5, "protocol_sha256": protocol_hash,
        "model_archive_field_whitelist": sorted(FIELDS),
        "identity_fields_not_model_inputs": ["id", "family_id", "split", "environment_seed"],
        "sample_frames": "test_start/query/target frames are shared by the siblings of a family; marker frames "
                         "are privileged and only in evaluation_only/",
        "recent_length": RECENT_LENGTH, "horizon": cue.HORIZON,
        "class_labels": "not in this index; evaluation_only/sibling_classes.json",
        "class_assignment_sha256": json_sha(assignment),
        "records": records,
    })
    audit = audit_dataset(output, protocol, records)
    write_json(output / "structural_audit.json", audit)
    train_summary = {
        "families": len(train_checks),
        "target_min_pair_keypoint_px_q10_50_90": np.quantile(
            [r["target_min_pair_px"] for r in train_checks], [.1, .5, .9]).tolist(),
        "query_minus_marker_end_frames_min": min(r["query_frame"] - r["marker_frames"][-1] for r in train_checks),
        "query_frame_range": [min(r["query_frame"] for r in train_checks), max(r["query_frame"] for r in train_checks)],
    }
    artifacts = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = {
        "schema_version": 5, "status": "CPU_RGB_action_preparation_completed",
        "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "collection_seconds": collection_seconds, "processes": processes, "device": "cpu",
        "protocol_sha256": protocol_hash, **provenance, "counts": audit["counts"],
        "rejected_candidates": len(rejected),
        "rejected_by_reason": {k: sum(k in r["reasons"] for r in rejected)
                               for k in {k for r in rejected for k in r["reasons"]}},
        "structural_audit_passed": True, "train_outcome_summary": train_summary,
        "stored_artifact_bytes_excluding_this_receipt": sum(a["bytes"] for a in artifacts.values()),
        "artifacts": artifacts, "encoder_run": False, "normalization_fitted": False,
        "predictor_trained": False,
        "limitations": ["The marker is a drawn symbol, not physics; the dynamics enter only through the test push.",
                        "Moment of inertia is the gym-pusht default for every class."],
    }
    write_json(output / "preparation_manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    result = run(args.output, args.processes)
    print(json.dumps({k: result[k] for k in ("status", "counts", "rejected_candidates", "rejected_by_reason",
                                             "train_outcome_summary", "collection_seconds",
                                             "stored_artifact_bytes_excluding_this_receipt")}, indent=2))


if __name__ == "__main__":
    main()
