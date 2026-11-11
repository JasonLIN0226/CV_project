"""Collect the hidden-CoG PushT corpus (probe -> cut -> gap -> test push).

Each family is one sampled spec run under every CoG class with identical
actions (see pusht_cog_cue.py). Siblings are stored as s0/s1/s2 under a seeded
shuffle, so neither file names nor model-facing archives reveal the class.
The class map, simulator state and outcome checks live only in evaluation_only/.

The protocol (family specs, action hashes, sample frames, a hash of the class
map) is written before any simulator is created. A family that fails the
validity checks stops collection; families are never resampled on outcomes.
No blackout is applied: the scene cut and the no-contact gap make every frame
from the cut through the query pixel-identical across siblings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_cog_cue as cue
from pusht import normalize_actions

SPLIT_COUNTS = {"train": 128, "validation": 32, "evaluation": 32}  # families
# 2026110000..2026110127 were inspected while designing the task; splits start
# at fresh blocks and must stay disjoint from that range.
SPLIT_BASE_SEEDS = {"train": 2026111000, "validation": 2026112000, "evaluation": 2026113000}
DEV_INSPECTED_SEEDS = (2026110000, 2026110127)
FIELDS = {"rgb", "timestamps", "actions_absolute", "actions_normalized"}
RECENT_LENGTH = 4


def write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan_actions(spec: cue.FamilySpec) -> np.ndarray:
    return cue.build_plan(spec).actions.astype(np.float32)


def sibling_classes(family_seed: int) -> list[str]:
    """Seeded order of CoG classes over siblings s0, s1, s2."""
    order = np.random.default_rng(family_seed + 3_000_000).permutation(len(cue.COG_CLASSES))
    names = list(cue.COG_CLASSES)
    return [names[i] for i in order]


def sample_frames(spec: cue.FamilySpec) -> dict:
    """Class-independent frame indices; frame i follows action i-1."""
    phase = ["reset"] + cue.build_plan(spec).phase
    query = phase.index("test") - 1
    return {"cut_frame": phase.index("cut"), "query_frame": query,
            "target_frame": query + cue.HORIZON, "frames": len(phase)}


def make_protocol():
    """Freeze family specs, actions and sample frames without any simulator."""
    families, assignment = [], {}
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            seed = SPLIT_BASE_SEEDS[split] + index
            if DEV_INSPECTED_SEEDS[0] <= seed <= DEV_INSPECTED_SEEDS[1]:
                raise AssertionError("Split seed overlaps the inspected development range")
            spec = cue.sample_family(seed)
            family_id = f"cog_{split}_{index:03d}"
            assignment[family_id] = sibling_classes(seed)
            families.append({
                "id": family_id, "split": split, "environment_seed": seed,
                "spec": spec.__dict__ | {"probe_center": list(spec.probe_center),
                                         "test_center": list(spec.test_center)},
                "siblings": [f"{family_id}_s{k}" for k in range(len(cue.COG_CLASSES))],
                "planned_actions_sha256": array_sha(plan_actions(spec)),
                **sample_frames(spec),
            })
    protocol = {
        "schema_version": 1, "protocol_id": "pusht-hidden-cog-probe-cut-test-v1-20261101",
        "purpose": "history-dependent PushT: a hidden discrete CoG is visible only through the probe push",
        "split_family_counts": SPLIT_COUNTS.copy(), "siblings_per_family": len(cue.COG_CLASSES),
        "dev_inspected_seed_range_excluded": list(DEV_INSPECTED_SEEDS),
        "environment": "gym_pusht/PushT-v0", "observation_shape": [96, 96, 3],
        "observation_source": "pixels returned by reset/step and get_obs after the cut; never render()",
        "control_hz": 10, "physics_dt_seconds": .01,
        "action_semantics": "absolute agent target XY in [0,512], ordered; not delta or velocity",
        "normalization": "actions_absolute / float32(256) - float32(1)",
        "cut_action": "the action recorded at the cut step is the teleported pusher position; "
                      "no control is executed for it, so models should not treat frames "
                      "cut_frame-1 -> cut_frame as a controlled transition",
        "cog_classes_body_local": cue.COG_CLASSES,
        "probe": "stem tip pushed toward the crossbar; contact offset uniform +-4 px",
        "test": f"crossbar top face pushed toward the stem; offset uniform +-{cue.TEST_OFFSET_MAX:g} px",
        "push_speed_px_per_step": 6.0, "push_depth_px": 100.0,
        "occlusion": "none; the cut and no-contact gap hide the cue from the recent window",
        "recent_length": RECENT_LENGTH, "horizon": cue.HORIZON,
        "query_definition": "last gap frame; the recent window is query-3..query",
        "class_assignment": "seeded shuffle per family; map stored only in evaluation_only/sibling_classes.json",
        "class_assignment_sha256": json_sha(assignment),
        "validity_policy": "any family failing pixel identity, no-gap-contact, arena or "
                           "contact-after-query checks stops collection; no resampling",
        "evaluation_role": "reserved families; validity checks only, no outcome-based tuning",
        "encoder_run": False, "predictor_trained": False, "planning_run": False,
        "families": families,
    }
    return protocol, assignment


def collect_family(family: dict, classes: list[str]):
    spec = cue.FamilySpec(**{**family["spec"], "probe_center": tuple(family["spec"]["probe_center"]),
                             "test_center": tuple(family["spec"]["test_center"])})
    actions = plan_actions(spec)
    if array_sha(actions) != family["planned_actions_sha256"]:
        raise ValueError("Plan no longer matches the frozen protocol")
    result, runs, plan = cue.check_family(spec, classes=tuple(cue.COG_CLASSES))
    if not result["valid"]:
        raise AssertionError(f"Family {family['id']} failed validity checks: {result}")
    for key in ("cut_frame", "query_frame", "target_frame"):
        if result[key] != family[key]:
            raise AssertionError(f"{key} differs from the frozen protocol")
    common = {"timestamps": np.arange(len(actions) + 1, dtype=np.int64),
              "actions_absolute": actions, "actions_normalized": normalize_actions(actions)}
    phase = ["reset"] + plan.phase
    episodes = []
    for sibling_id, cog_class in zip(family["siblings"], classes):
        run = runs[cog_class]
        episodes.append((sibling_id, {"rgb": run["rgb"], **common}, {
            "episode_id": sibling_id, "family_id": family["id"], "cog_class": cog_class,
            "cog_body_local": list(cue.COG_CLASSES[cog_class]), "phase": phase,
            "block_pose": run["pose"].tolist(), "block_keypoints": run["keypoints"].tolist(),
            "contact_points": run["contacts"].tolist(),
        }))
    return episodes, result


def _safe_path(root: Path, relative: str, directory: str):
    path = (root / relative).resolve()
    if not path.is_relative_to((root / directory).resolve()):
        raise ValueError(f"Artifact must remain in {directory}")
    return path


def audit_dataset(output: Path, protocol: dict, records: list[dict]):
    """Structural checks on model-facing archives only; never reads evaluation_only."""
    families = {f["id"]: f for f in protocol["families"]}
    seeds = [f["environment_seed"] for f in families.values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Environment seeds repeat across families")
    expected_ids = {s for f in families.values() for s in f["siblings"]}
    if {r["id"] for r in records} != expected_ids or len(records) != len(expected_ids):
        raise ValueError("Missing or duplicate episodes")
    counts = {s: {"families": n, "episodes": 0, "frames": 0} for s, n in protocol["split_family_counts"].items()}
    by_family, model_bytes = {}, 0
    for record in records:
        family = families[record["family_id"]]
        if record["split"] != family["split"] or record["id"] not in family["siblings"]:
            raise ValueError("Episode identity/split differs from frozen protocol")
        path = _safe_path(output, record["observations"], "observations")
        if sha256(path) != record["observations_sha256"]:
            raise ValueError("Observation checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FIELDS:
                raise ValueError("Forbidden/missing fields in model archive")
            data = {k: archive[k] for k in FIELDS}
        frames = family["frames"]
        if data["rgb"].shape != (frames, *protocol["observation_shape"]) or data["rgb"].dtype != np.uint8:
            raise ValueError("Bad RGB shape/dtype")
        if not np.array_equal(data["timestamps"], np.arange(frames, dtype=np.int64)):
            raise ValueError("Bad timestamps")
        action = data["actions_absolute"]
        if action.dtype != np.float32 or array_sha(action) != family["planned_actions_sha256"]:
            raise ValueError("Actions differ from the frozen plan")
        if not np.array_equal(data["actions_normalized"], normalize_actions(action)):
            raise ValueError("Wrong action normalization")
        model_bytes += sum(v.nbytes for v in data.values())
        by_family.setdefault(family["id"], []).append(data["rgb"])
        counts[record["split"]]["episodes"] += 1
        counts[record["split"]]["frames"] += frames
    for family_id, rgbs in by_family.items():
        f = families[family_id]
        window = slice(f["cut_frame"], f["query_frame"] + 1)
        if any(not np.array_equal(rgbs[0][0], r[0]) or not np.array_equal(rgbs[0][window], r[window])
               for r in rgbs[1:]):
            raise ValueError(f"Sibling pixels differ before the cue/future in {family_id}")
    return {
        "passed": True, "counts": counts, "unique_environment_seeds": len(seeds),
        "model_archive_uncompressed_bytes": model_bytes, "diagnostics_loaded": False,
        "checks": ["family_seed_disjoint", "sibling_ids_match_protocol", "archive_field_whitelist",
                   "rgb_shape_dtype", "timestamps", "actions_equal_frozen_plan_and_across_siblings",
                   "absolute_action_normalization", "checksums",
                   "sibling_pixels_identical_at_reset_and_cut_through_query"],
    }


def _runtime_provenance():
    import gym_pusht
    import gym_pusht.envs.pusht as installed
    here = Path(__file__)
    source_paths = [here, here.with_name("pusht_cog_cue.py"), here.with_name("pusht.py"),
                    Path(gym_pusht.__file__), Path(installed.__file__)]
    try:
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        base = None

    def name(p):
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else f"site-packages/{p.parent.name}/{p.name}"
    return {
        "git_base": base,
        "source_sha256": {name(p): sha256(p) for p in source_paths},
        "python": platform.python_version(),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("numpy", "gym-pusht", "gymnasium", "pygame", "pymunk", "shapely",
                      "opencv-python-headless")},
    }


def run(output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing data will not be overwritten")
    protocol, assignment = make_protocol()
    provenance = _runtime_provenance()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("observations", "evaluation_only"):
        (output / name).mkdir()
    write_json(output / "protocol.json", protocol)
    protocol_hash = sha256(output / "protocol.json")
    write_json(output / "evaluation_only" / "sibling_classes.json", assignment)
    write_json(output / "protocol_frozen_before_collection.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
        "before_environment_creation": True, **provenance,
    })
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    records, train_checks = [], []
    for family in protocol["families"]:
        episodes, result = collect_family(family, assignment[family["id"]])
        write_json(output / "evaluation_only" / f"{family['id']}_checks.json", result)
        if family["split"] == "train":
            train_checks.append(result)
        for sibling_id, archive, diagnostics in episodes:
            path = output / "observations" / f"{sibling_id}.npz"
            np.savez_compressed(path, **archive)
            write_json(output / "evaluation_only" / f"{sibling_id}.json", diagnostics)
            records.append({
                "id": sibling_id, "family_id": family["id"], "split": family["split"],
                "environment_seed": family["environment_seed"],
                **{k: family[k] for k in ("frames", "cut_frame", "query_frame", "target_frame")},
                "observations": str(path.relative_to(output)), "observations_sha256": sha256(path),
            })
    collection_seconds = time.perf_counter() - started
    if sha256(output / "protocol.json") != protocol_hash:
        raise AssertionError("Frozen protocol changed during collection")
    write_json(output / "observation_index.json", {
        "schema_version": 1, "protocol_sha256": protocol_hash,
        "model_archive_field_whitelist": sorted(FIELDS),
        "identity_fields_not_model_inputs": ["id", "family_id", "split", "environment_seed"],
        "sample_frames": "cut/query/target frames are shared by all siblings of a family",
        "recent_length": RECENT_LENGTH, "horizon": cue.HORIZON,
        "class_labels": "not in this index; evaluation_only/sibling_classes.json",
        "records": records,
    })
    audit = audit_dataset(output, protocol, records)
    write_json(output / "structural_audit.json", audit)
    # Outcome statistics from TRAIN families only; held-out checks are stored, not summarized.
    target_px = [r["target_min_pair_px"] for r in train_checks]
    train_summary = {
        "families": len(train_checks),
        "probe_turn_deg_median": {c: float(np.median([r["probe_turn_deg"][c] for r in train_checks]))
                                  for c in cue.COG_CLASSES},
        "target_min_pair_keypoint_px_quantiles_10_50_90": np.quantile(target_px, [.1, .5, .9]).tolist(),
        "first_test_contact_minus_query": sorted({r["first_test_contact_frame"] - r["query_frame"]
                                                  for r in train_checks}),
    }
    artifacts = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = {
        "schema_version": 1, "status": "CPU_RGB_action_preparation_completed",
        "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "collection_seconds": collection_seconds, "total_seconds": time.perf_counter() - started,
        "device": "cpu", "slurm_jobs_submitted": 0, "gpu_seconds": 0,
        "protocol_sha256": protocol_hash, **provenance,
        "counts": audit["counts"], "structural_audit_passed": True,
        "all_families_valid": True, "train_outcome_summary": train_summary,
        "stored_artifact_bytes_excluding_this_receipt": sum(a["bytes"] for a in artifacts.values()),
        "artifacts": artifacts, "encoder_run": False, "normalization_fitted": False,
        "predictor_trained": False, "history_necessity_validated": False,
        "limitations": ["The cut is a teleport, not continuous physics.",
                        "Quasi-static PushT: the cue is rotation direction under a push, not momentum.",
                        "Moment of inertia is the gym-pusht default for every class.",
                        "Simulator separability does not show that a frozen encoder or predictor resolves it."],
    }
    write_json(output / "preparation_manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ("status", "counts", "train_outcome_summary",
                                             "collection_seconds", "stored_artifact_bytes_excluding_this_receipt")},
                     indent=2))


if __name__ == "__main__":
    main()
