"""Collect the hidden-CoG PushT corpus as natural continuous episodes (design v4).

Each family shares the start pose, pusher start and one probe push across the
three CoG classes (see pusht_cog_natural.py); afterwards each episode follows its
own block. Siblings are stored as s0/s1/s2 under a seeded shuffle; neither file
names nor model-facing archives reveal the class. The class map, simulator state,
phase labels and counterfactual checks live only in evaluation_only/.

Selection: candidate seeds are tried in a fixed order per split until the split
has its family count. A family is rejected if any of its three episodes fails a
validity check (contact before the probe or during transit/gap, leaving the
arena, test contact before the query). The rule looks at all classes jointly, so
it is class-symmetric; rejected seeds and reasons are recorded. Test outcomes
and class readability are never used for selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_cog_matched as v3
import pusht_cog_natural as cue
from pusht import normalize_actions

SPLIT_COUNTS = {"train": 128, "validation": 32, "evaluation": 32}  # families
# Dev seeds 2026140000..2026140191 were inspected while designing v4.
SPLIT_BASE_SEEDS = {"train": 2026151000, "validation": 2026152000, "evaluation": 2026153000}
DEV_INSPECTED_SEEDS = (2026140000, 2026140191)
MAX_CANDIDATES_PER_SPLIT = 999
FIELDS = {"rgb", "timestamps", "actions_absolute", "actions_normalized"}
RECENT_LENGTH = cue.RECENT_LENGTH

POOL = OUTPUT = None


def write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sibling_classes(family_seed: int) -> list[str]:
    order = np.random.default_rng(family_seed + 3_000_000).permutation(len(cue.COG_CLASSES))
    names = list(cue.COG_CLASSES)
    return [names[i] for i in order]


def _init(pool, output):
    global POOL, OUTPUT
    POOL, OUTPUT = pool, output


def collect_candidate(job):
    """Run one candidate family; write its files only if valid. Returns a record."""
    split, seed = job
    spec = cue.sample_family(seed, POOL)
    result, runs, actions, phase = cue.check_family(spec)
    if not result["valid"]:
        reasons = [k for k, bad in {
            "approach_contact": result["approach_contacts"] > 0,
            "transit_gap_contact": result["transit_gap_contacts"] > 0,
            "out_of_bounds": not (result["block_in_bounds"] and result["actions_in_arena"]),
            "test_contact_before_query": result["first_test_contact_frame"] <= result["query_frame"],
            "reset_differs": not result["reset_frames_identical"],
        }.items() if bad]
        return {"split": split, "seed": seed, "valid": False, "reasons": reasons}
    return {"split": split, "seed": seed, "valid": True, "spec": asdict(spec), "result": result,
            "episodes": {c: (runs[c], actions[c]) for c in runs}, "phase": phase}


def write_family(output: Path, family_id: str, candidate: dict, classes: list[str]):
    result, phase = candidate["result"], candidate["phase"]
    frames = {k: result[k] for k in ("probe_start_frame", "probe_end_frame", "query_frame", "target_frame")}
    records = []
    write_json(output / "evaluation_only" / f"{family_id}_checks.json", result)
    for k, cog in enumerate(classes):
        sibling_id = f"{family_id}_s{k}"
        run, actions = candidate["episodes"][cog]
        actions = actions.astype(np.float32)
        archive = {"rgb": run["rgb"], "timestamps": np.arange(len(actions) + 1, dtype=np.int64),
                   "actions_absolute": actions, "actions_normalized": normalize_actions(actions)}
        path = output / "observations" / f"{sibling_id}.npz"
        np.savez_compressed(path, **archive)
        write_json(output / "evaluation_only" / f"{sibling_id}.json", {
            "episode_id": sibling_id, "family_id": family_id, "cog_class": cog,
            "cog_body_local": list(cue.COG_CLASSES[cog]), "phase": phase,
            "block_pose": run["pose"].tolist(), "block_keypoints": run["keypoints"].tolist(),
            "pusher_position": run["agent"].tolist(), "contact_points": run["contacts"].tolist(),
            "per_class_check": result["per_class"][cog],
        })
        records.append({
            "id": sibling_id, "family_id": family_id, "split": candidate["split"],
            "environment_seed": candidate["seed"], "frames": len(run["rgb"]), **frames,
            "observations": str(path.relative_to(output)), "observations_sha256": sha256(path),
            "actions_sha256": array_sha(actions),
        })
    return records


def _safe_path(root: Path, relative: str, directory: str):
    path = (root / relative).resolve()
    if not path.is_relative_to((root / directory).resolve()):
        raise ValueError(f"Artifact must remain in {directory}")
    return path


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
        action = data["actions_absolute"]
        if action.dtype != np.float32 or action.shape != (frames - 1, 2) or \
                array_sha(action) != record["actions_sha256"]:
            raise ValueError("Actions differ from the index")
        if not np.array_equal(data["actions_normalized"], normalize_actions(action)):
            raise ValueError("Wrong action normalization")
        if not record["probe_end_frame"] < record["query_frame"] - RECENT_LENGTH + 1:
            raise ValueError("Recent window overlaps the probe")
        model_bytes += sum(v.nbytes for v in data.values())
        by_family.setdefault(record["family_id"], []).append((record, data))
        c = counts.setdefault(record["split"], {"families": 0, "episodes": 0, "frames": 0})
        c["episodes"] += 1
        c["frames"] += frames
    seeds = {r["environment_seed"] for r in records}
    if len(seeds) != len(by_family):
        raise ValueError("Environment seeds repeat across families")
    for family_id, items in by_family.items():
        r0, d0 = items[0]
        counts[r0["split"]]["families"] += 1
        if len(items) != len(cue.COG_CLASSES):
            raise ValueError(f"Family {family_id} does not have one episode per class")
        shared = r0["probe_end_frame"]  # probe actions (and frames before contact) are shared
        for r, d in items[1:]:
            if any(r[k] != r0[k] for k in ("frames", "probe_start_frame", "probe_end_frame", "query_frame",
                                           "target_frame", "split")):
                raise ValueError(f"Sibling frame indices differ in {family_id}")
            if not np.array_equal(d0["rgb"][0], d["rgb"][0]):
                raise ValueError(f"Sibling reset frames differ in {family_id}")
            if not np.array_equal(d0["actions_absolute"][:shared], d["actions_absolute"][:shared]):
                raise ValueError(f"Sibling probe actions differ in {family_id}")
    if any(counts.get(s, {}).get("families") != n for s, n in protocol["split_family_counts"].items()):
        raise ValueError("Split family counts differ from protocol")
    return {
        "passed": True, "counts": counts, "model_archive_uncompressed_bytes": model_bytes,
        "diagnostics_loaded": False,
        "checks": ["archive_field_whitelist", "rgb_shape_dtype", "timestamps", "actions_match_index_hash",
                   "absolute_action_normalization", "checksums", "recent_window_after_probe",
                   "family_seed_unique", "one_episode_per_class", "sibling_frame_indices_equal",
                   "sibling_reset_frames_identical", "sibling_probe_actions_identical", "split_counts"],
    }


def _runtime_provenance():
    import gym_pusht
    import gym_pusht.envs.pusht as installed
    here = Path(__file__)
    source_paths = [here] + [here.with_name(n) for n in ("pusht_cog_natural.py", "pusht_cog_matched.py",
                                                          "pusht_cog_coherent.py", "pusht_cog_cue.py", "pusht.py")]
    source_paths += [Path(gym_pusht.__file__), Path(installed.__file__)]
    try:
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        base = None

    def name(p):
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else f"site-packages/{p.parent.name}/{p.name}"
    return {"git_base": base, "source_sha256": {name(p): sha256(p) for p in source_paths},
            "python": platform.python_version(),
            "packages": {n: importlib.metadata.version(n) for n in
                         ("numpy", "gym-pusht", "gymnasium", "pygame", "pymunk", "shapely",
                          "opencv-python-headless")}}


def run(output: Path, library_path: Path, processes: int):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing data will not be overwritten")
    library = v3.build_library(library_path, processes=processes)
    pool_rows = cue.probe_pool(library)
    protocol = {
        "schema_version": 4, "protocol_id": "pusht-hidden-cog-natural-v4-20261107",
        "purpose": "history-dependent PushT for retention and, later, planning: a hidden discrete CoG is "
                   "revealed by how the block responds to a probe push",
        "design": "family shares start pose, pusher start and probe push; afterwards continuous physics, "
                  "pusher moves planned from the block's actual pose; no teleport, no matching",
        "split_family_counts": SPLIT_COUNTS.copy(), "siblings_per_family": len(cue.COG_CLASSES),
        "split_base_seeds": SPLIT_BASE_SEEDS, "dev_inspected_seed_range_excluded": list(DEV_INSPECTED_SEEDS),
        "selection": "candidate seeds base, base+1, ... per split until the count is reached; a family is "
                     "rejected if any class fails validity (class-symmetric); never on test outcomes",
        "environment": "gym_pusht/PushT-v0", "observation_shape": [96, 96, 3],
        "observation_source": "pixels returned by reset/step; never render()",
        "control_hz": 10, "physics_dt_seconds": .01,
        "action_semantics": "absolute agent target XY in [0,512], ordered; not delta or velocity",
        "normalization": "actions_absolute / float32(256) - float32(1)",
        "cog_classes_body_local": cue.COG_CLASSES,
        "start_center_range": list(cue.START_RANGE), "probe_shift_max_px": cue.PROBE_SHIFT_MAX,
        "probe_turn_spread_min_deg": cue.PROBE_TURN_SPREAD_MIN,
        "probe_library_sha256": sha256(library_path), "probe_pool_size": len(pool_rows),
        "test": f"crossbar top face pushed toward the stem; offset uniform +-{cue.TEST_OFFSET_MAX:g} px",
        "recent_length": RECENT_LENGTH, "horizon": cue.HORIZON,
        "query_definition": "last gap frame; the recent window is query-3..query",
        "class_assignment": "seeded shuffle per family seed; map stored only in evaluation_only/sibling_classes.json",
        "evaluation_role": "reserved families; validity checks only, no outcome-based tuning",
        "encoder_run": False, "predictor_trained": False, "planning_run": False,
    }
    provenance = _runtime_provenance()
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
    with multiprocessing.get_context("fork").Pool(processes, initializer=_init,
                                                  initargs=(pool_rows, output)) as pool:
        for split, count in SPLIT_COUNTS.items():
            accepted, next_k = 0, 0
            while accepted < count:
                batch = [(split, SPLIT_BASE_SEEDS[split] + k) for k in range(next_k, next_k + processes)]
                next_k += processes
                if next_k > MAX_CANDIDATES_PER_SPLIT:
                    raise AssertionError(f"Too many rejections in {split}")
                # Results come back in seed order, so acceptance is deterministic.
                for cand in pool.map(collect_candidate, batch):
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
            print(f"{split}: {count} families, {len([r for r in rejected if r['split'] == split])} rejected, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    collection_seconds = time.perf_counter() - started
    write_json(output / "evaluation_only" / "sibling_classes.json", assignment)
    write_json(output / "rejected_candidates.json", rejected)
    if sha256(output / "protocol.json") != protocol_hash:
        raise AssertionError("Frozen protocol changed during collection")
    write_json(output / "observation_index.json", {
        "schema_version": 4, "protocol_sha256": protocol_hash,
        "model_archive_field_whitelist": sorted(FIELDS),
        "identity_fields_not_model_inputs": ["id", "family_id", "split", "environment_seed"],
        "sample_frames": "probe_start/probe_end/query/target frames are shared by the siblings of a family; "
                         "frames before the probe contact and the probe actions are identical across siblings",
        "recent_length": RECENT_LENGTH, "horizon": cue.HORIZON,
        "class_labels": "not in this index; evaluation_only/sibling_classes.json",
        "class_assignment_sha256": json_sha(assignment),
        "records": records,
    })
    audit = audit_dataset(output, protocol, records)
    write_json(output / "structural_audit.json", audit)
    cf = [e["counterfactual_min_pair_px"] for r in train_checks for e in r["per_class"].values()]
    turns = {c: [r["per_class"][c]["probe_turn_deg"] for r in train_checks] for c in cue.COG_CLASSES}
    train_summary = {
        "families": len(train_checks),
        "probe_turn_deg_q10_50_90": {c: np.quantile(v, [.1, .5, .9]).round(1).tolist() for c, v in turns.items()},
        "counterfactual_min_pair_px_q10_50_90": np.quantile(cf, [.1, .5, .9]).tolist(),
    }
    artifacts = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = {
        "schema_version": 4, "status": "CPU_RGB_action_preparation_completed",
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
        "limitations": ["Quasi-static PushT: the cue is the block's response to a push, not momentum.",
                        "Moment of inertia is the gym-pusht default for every class.",
                        "Post-probe poses differ by class and persist into the recent frames; dev runs measured "
                        "present-only state classifiers near chance, but this is a distributional property.",
                        "Pusher moves after the probe are planned from the simulator's block pose (scripted "
                        "collection policy), not from a model."],
    }
    write_json(output / "preparation_manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    result = run(args.output, args.library, args.processes)
    print(json.dumps({k: result[k] for k in ("status", "counts", "rejected_candidates", "rejected_by_reason",
                                             "train_outcome_summary", "collection_seconds",
                                             "stored_artifact_bytes_excluding_this_receipt")}, indent=2))


if __name__ == "__main__":
    main()
