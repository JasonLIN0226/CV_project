"""Collect a fixed CPU-only PushT development corpus for later learned planning.

The collection policy is fully determined before reset and never reads pixels,
reward, state, or diagnostics. No encoder, predictor, planner or goal rendering is
run. Model observations, unoccluded controls and privileged diagnostics are
separate artifacts; no model-facing file contains control pixels or state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from pusht import (Occlusion, PushTVisualAdapter, VisualObservation, absolute_actions,
                   later_visible_targets, normalize_actions)

SPLIT_COUNTS = {"train": 32, "validation": 8, "evaluation": 8}
FIELDS = {"rgb", "timestamps", "actions_absolute", "actions_normalized"}
BASE_SEED = 2026102000
MAX_STEPS = 96
DIAGNOSTIC_ATOL = 1e-12


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _json(value):
    return json.dumps(value, default=_json_default, sort_keys=True, allow_nan=False)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, default=_json_default, sort_keys=True,
                               indent=2, allow_nan=False) + "\n")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def paired_diagnostic_difference(masked, control):
    """Compare states/rewards with a tight float tolerance; flags/counts exact.

    Identical Shapely coverage geometries can yield ~1e-16 roundoff in separate
    environments. This does not relax the exact RGB or action checks.
    """
    left, right = json.loads(_json(asdict(masked))), json.loads(_json(asdict(control)))

    def compare(a, b):
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            return max((compare(a[k], b[k]) for k in a), default=0.)
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return max((compare(x, y) for x, y in zip(a, b)), default=0.)
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            if type(a) is not type(b) or a != b:
                raise AssertionError("Mask changed paired physical diagnostics")
            return 0.
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = abs(a-b)
            tolerance = 0. if isinstance(a, int) and isinstance(b, int) else DIAGNOSTIC_ATOL
            if not np.isfinite(delta) or delta > tolerance:
                raise AssertionError("Mask changed paired physical diagnostics")
            return float(delta)
        if type(a) is not type(b) or a != b:
            raise AssertionError("Mask changed paired physical diagnostics")
        return 0.

    return compare(left, right)


def absolute_action_schedule(policy_seed: int) -> np.ndarray:
    """96 precomputed target positions; no observation/state-dependent choices.

    First 48 controls: three back-and-forth workspace sweeps, 16 steps each.
    Last 48: a seeded random walk, with each absolute target held four controls.
    Orientation and direction are seeded; targets stay inside [40,472]^2.
    This is data collection, not model-based planning.
    """
    rng = np.random.default_rng(policy_seed)
    horizontal = bool(rng.integers(2))
    reverse = bool(rng.integers(2))
    lines = np.array([104., 256., 408.]) + rng.uniform(-32., 32., 3)
    if reverse:
        lines = lines[::-1]
    targets = []
    for number, line in enumerate(lines):
        sweep = np.linspace(40., 472., 16, dtype=np.float32)
        if bool(number % 2) ^ reverse:
            sweep = sweep[::-1]
        targets.extend(([float(v), float(line)] if horizontal else [float(line), float(v)])
                       for v in sweep)
    current = np.asarray(targets[-1], dtype=np.float32)
    for _ in range(12):
        current = np.clip(current + rng.uniform(-144., 144., 2), 40., 472.).astype(np.float32)
        targets.extend([current.copy()] * 4)
    result = absolute_actions(np.asarray(targets, dtype=np.float32))
    if result.shape != (MAX_STEPS, 2):
        raise AssertionError("Unexpected policy length")
    return result


def make_protocol():
    """Freeze membership, actions and masks without touching an environment."""
    episodes = []
    for split_index, (split, count) in enumerate(SPLIT_COUNTS.items()):
        for index in range(count):
            seed = BASE_SEED + split_index * 1000 + index
            policy_seed = seed + 1_000_000
            mask_seed = seed + 2_000_000
            start = int(np.random.default_rng(mask_seed).choice([8, 16, 24, 32]))
            episodes.append({
                "id": f"pusht_{split}_{index:03d}", "split": split,
                "environment_seed": seed, "policy_seed": policy_seed,
                "occlusion": asdict(Occlusion(start=start, stop=start+5)),
                "planned_actions_sha256": array_sha(absolute_action_schedule(policy_seed)),
            })
    return {
        "schema_version": 1, "protocol_id": "pusht-absolute-sweep-walk-v1-20261020",
        "purpose": "new_development_data_for_future_learned_planning",
        "split_counts": SPLIT_COUNTS.copy(), "max_control_steps": MAX_STEPS,
        "environment": "gym_pusht/PushT-v0", "observation_shape": [96, 96, 3],
        "observation_source": "pixels returned by reset/step; never render()",
        "control_hz": 10, "physics_dt_seconds": .01,
        "action_semantics": "absolute agent target XY in [0,512], ordered; not delta or velocity",
        "normalization": "actions_absolute / float32(256) - float32(1)",
        "policy": "seeded absolute sweeps (48 controls) then random walk (12 targets held 4 controls)",
        "policy_inputs": ["predeclared policy_seed"],
        "paired_control": "separate same-seed same-action simulator; mask changes observation RGB only",
        "paired_diagnostic_float_absolute_tolerance": DIAGNOSTIC_ATOL,
        "paired_pixels_actions_flags_and_integer_counts": "exact equality",
        "occlusion": "full RGB black for five controls; start chosen before reset from 8,16,24,32",
        "recent_length_for_offline_target_audit": 4, "target_horizons": [3, 6],
        "evaluation_role": "reserved episodes; structural audit only, no diagnostic-based tuning or quality aggregation",
        "diagnostics_policy": "reward/state/coverage/contact and target audits stored only in evaluation_only",
        "future_actions_policy": "recorded actions are offline training continuations only; future retention/planning must not read them",
        "goal_image_status": "pending_not_rendered",
        "goal_image_reason": "No verified independent goal RGB; reset_to_state legacy pose ordering needs explicit separate validation.",
        "encoder_run": False, "predictor_trained": False, "planning_run": False,
        "episodes": episodes,
    }


def collect_episode(spec: dict, *, factory: Callable = PushTVisualAdapter.create):
    """Collect real paired observations; policy is computed before either reset."""
    actions = absolute_action_schedule(spec["policy_seed"])
    if array_sha(actions) != spec["planned_actions_sha256"]:
        raise ValueError("Policy no longer matches the frozen protocol")
    schedule = Occlusion(**spec["occlusion"])
    masked = factory(schedule)
    control = factory(Occlusion(enabled=False))
    observations, controls, diagnostics, control_diagnostics, executed = [], [], [], [], []
    diagnostic_differences = []

    def append(mo, md, co, cd):
        if mo.timestamp != co.timestamp or not np.array_equal(mo.rgb, schedule.apply(co.rgb, co.timestamp)):
            raise AssertionError("Paired RGB differs outside the prescribed observation mask")
        diagnostic_differences.append(paired_diagnostic_difference(md, cd))
        observations.append(mo)
        controls.append(co)
        diagnostics.append(asdict(md))
        control_diagnostics.append(asdict(cd))

    try:
        mo, md = masked.reset(seed=spec["environment_seed"])
        co, cd = control.reset(seed=spec["environment_seed"])
        append(mo, md, co, cd)
        for action in actions:
            mo, md = masked.step(action)
            co, cd = control.step(action)
            append(mo, md, co, cd)
            executed.append(action.copy())
            if md.terminated or md.truncated:
                break
    finally:
        masked.close()
        control.close()
    steps = np.asarray(executed, dtype=np.float32).reshape(-1, 2)
    timestamps = np.array([o.timestamp for o in observations], dtype=np.int64)
    common = {"timestamps": timestamps, "actions_absolute": steps,
              "actions_normalized": normalize_actions(steps)}
    return {
        "observations": {"rgb": np.stack([o.rgb for o in observations]), **common},
        "control": {"rgb": np.stack([o.rgb for o in controls]), **common},
        "diagnostics": {"episode_id": spec["id"], "steps": diagnostics,
                        "control_steps": control_diagnostics,
                        "later_visible_targets": later_visible_targets(
                            observations, schedule, horizons=(3, 6), recent_length=4),
                        "paired_physical_diagnostics_equal_within_absolute_tolerance": True,
                        "paired_diagnostic_float_absolute_tolerance": DIAGNOSTIC_ATOL,
                        "paired_diagnostic_max_abs_difference": max(diagnostic_differences)},
    }


def _safe_path(root: Path, relative: str, directory: str):
    path = (root / relative).resolve()
    if not path.is_relative_to((root / directory).resolve()):
        raise ValueError(f"Artifact must remain in {directory}")
    return path


def audit_dataset(output: Path, protocol: dict, records: list[dict]):
    """Structural inspection only; never load evaluation-only diagnostics."""
    expected = {e["id"]: e for e in protocol["episodes"]}
    if len(records) != len(expected) or len({r["id"] for r in records}) != len(records):
        raise ValueError("Missing or duplicate episode identities")
    all_seeds = [s["environment_seed"] for s in expected.values()]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Environment seeds cross episode/split boundaries")
    counts = {split: {"episodes": 0, "frames": 0, "actions": 0} for split in SPLIT_COUNTS}
    model_bytes = control_bytes = 0
    for record in records:
        spec = expected.get(record["id"])
        if spec is None or any(record[k] != spec[k] for k in ("split", "environment_seed", "policy_seed")):
            raise ValueError("Episode identity/split differs from frozen protocol")
        observation_path = _safe_path(output, record["observations"], "observations")
        control_path = _safe_path(output, record["unoccluded_control"], "unoccluded_control")
        if sha256(observation_path) != record["observations_sha256"] or sha256(control_path) != record["unoccluded_control_sha256"]:
            raise ValueError("Observation/control checksum mismatch")
        with np.load(observation_path, allow_pickle=False) as masked, np.load(control_path, allow_pickle=False) as control:
            if set(masked.files) != FIELDS or set(control.files) != FIELDS:
                raise ValueError("Forbidden/missing fields in RGB/action archive")
            frames = len(masked["rgb"])
            if not 2 <= frames <= MAX_STEPS+1 or frames != record["frames"]:
                raise ValueError("Bad observation length")
            for archive in (masked, control):
                rgb = archive["rgb"]
                if rgb.shape != (frames, *protocol["observation_shape"]) or rgb.dtype != np.uint8:
                    raise ValueError("Bad RGB shape/dtype")
                ts, action, norm = (archive[k] for k in ("timestamps", "actions_absolute", "actions_normalized"))
                if ts.dtype != np.int64 or not np.array_equal(ts, np.arange(frames, dtype=np.int64)):
                    raise ValueError("Bad timestamp sequence")
                if action.dtype != np.float32 or action.shape != (frames-1, 2):
                    raise ValueError("Bad action shape/dtype")
                if not np.array_equal(action, absolute_action_schedule(spec["policy_seed"])[:frames-1]):
                    raise ValueError("Action sequence is not the protocol prefix")
                if norm.dtype != np.float32 or not np.array_equal(norm, normalize_actions(action)):
                    raise ValueError("Wrong absolute-action normalization")
            for key in FIELDS - {"rgb"}:
                if not np.array_equal(masked[key], control[key]):
                    raise ValueError("Paired control actions/timestamps differ")
            schedule = Occlusion(**spec["occlusion"])
            if any(not np.array_equal(masked["rgb"][t], schedule.apply(control["rgb"][t], t)) for t in range(frames)):
                raise ValueError("RGB mask mismatch")
            model_bytes += sum(masked[k].nbytes for k in masked.files)
            control_bytes += sum(control[k].nbytes for k in control.files)
        counts[record["split"]]["episodes"] += 1
        counts[record["split"]]["frames"] += frames
        counts[record["split"]]["actions"] += frames-1
    if any(counts[s]["episodes"] != n for s, n in protocol["split_counts"].items()):
        raise ValueError("Split counts differ from protocol")
    return {
        "passed": True, "counts": counts, "unique_environment_seeds": len(all_seeds),
        "frames_per_variant": sum(v["frames"] for v in counts.values()),
        "actions_per_variant": sum(v["actions"] for v in counts.values()),
        "model_archive_uncompressed_bytes": model_bytes,
        "control_archive_uncompressed_bytes": control_bytes,
        "diagnostics_loaded": False, "evaluation_quality_measured": False,
        "checks": ["identity_seed_split_disjoint", "protocol_action_prefix", "absolute_action_normalization",
                   "timestamp_and_rgb_shape_dtype", "archive_field_whitelist", "checksums",
                   "pre_encoding_RGB_mask", "paired_control_actions_equal", "diagnostic_paths_excluded"],
    }


def _runtime_provenance():
    source_paths = [Path(__file__), Path(__file__).with_name("pusht.py")]
    import gym_pusht
    import gym_pusht.envs.pusht as installed
    source_paths += [Path(gym_pusht.__file__), Path(installed.__file__)]
    try:
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        base = None
    return {
        "git_base": base,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "gym-pusht", "gymnasium", "pygame", "pymunk", "shapely", "opencv-python")},
    }


def run(output: Path, *, factory: Callable = PushTVisualAdapter.create):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing data/protocol will not be overwritten")
    protocol = make_protocol()
    provenance = _runtime_provenance()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("observations", "unoccluded_control", "evaluation_only"):
        (output / name).mkdir()
    # This immutable receipt is written before either simulator is created/reset.
    write_json(output / "protocol.json", protocol)
    protocol_hash = sha256(output / "protocol.json")
    write_json(output / "protocol_frozen_before_collection.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
        "before_environment_creation": True, **provenance,
    })
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    records = []
    for spec in protocol["episodes"]:
        data = collect_episode(spec, factory=factory)
        observation_path = output / "observations" / f"{spec['id']}.npz"
        control_path = output / "unoccluded_control" / f"{spec['id']}.npz"
        diagnostic_path = output / "evaluation_only" / f"{spec['id']}.json"
        np.savez_compressed(observation_path, **data["observations"])
        np.savez_compressed(control_path, **data["control"])
        write_json(diagnostic_path, data["diagnostics"])
        records.append({
            **{key: spec[key] for key in ("id", "split", "environment_seed", "policy_seed")},
            "frames": len(data["observations"]["rgb"]),
            "observations": str(observation_path.relative_to(output)),
            "observations_sha256": sha256(observation_path),
            "unoccluded_control": str(control_path.relative_to(output)),
            "unoccluded_control_sha256": sha256(control_path),
        })
    collection_seconds = time.perf_counter()-started
    if sha256(output / "protocol.json") != protocol_hash:
        raise AssertionError("Frozen protocol changed during collection")
    write_json(output / "observation_index.json", {
        "schema_version": 1, "protocol_sha256": protocol_hash,
        "model_archive_field_whitelist": sorted(FIELDS),
        "identity_fields_not_model_inputs": ["id", "split", "environment_seed", "policy_seed"],
        "unoccluded_control_is_separate_evaluation_or_control_training_arm": True,
        "future_actions_are_offline_targets_not_online_policy_inputs": True,
        "records": records,
    })
    audit = audit_dataset(output, protocol, records)
    write_json(output / "structural_audit.json", audit)
    # Hash diagnostics as opaque bytes; do not inspect held-out outcome quality.
    artifacts = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = {
        "schema_version": 1, "status": "CPU_RGB_action_preparation_completed",
        "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "collection_seconds": collection_seconds, "total_seconds": time.perf_counter()-started,
        "device": "cpu", "slurm_jobs_submitted": 0, "gpu_seconds": 0,
        "protocol_sha256": protocol_hash, **provenance,
        "counts": audit["counts"], "frames_per_variant": audit["frames_per_variant"],
        "actions_per_variant": audit["actions_per_variant"], "paired_simulator_variants": 2,
        "structural_audit_passed": True,
        "stored_artifact_bytes_excluding_this_receipt": sum(a["bytes"] for a in artifacts.values()),
        "artifacts": artifacts, "encoder_run": False, "normalization_fitted": False,
        "predictor_trained": False, "goal_image_status": "pending_not_rendered",
        "history_necessity_validated": False, "baseline_model_planning_verified": False,
        "evaluation_quality_measured": False,
        "limitations": ["A collection policy is not model-based planning.",
                        "RGB masking alone does not establish the need for history.",
                        "No hidden-state/action-independent model or learned PushT dynamics is claimed.",
                        "A fixed exogenous goal RGB still needs separate preparation/verification.",
                        "No evaluation reward/contact/success summaries were used to tune this protocol."],
    }
    write_json(output / "preparation_manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ("status", "counts", "frames_per_variant",
                                             "collection_seconds", "total_seconds",
                                             "goal_image_status", "baseline_model_planning_verified")}, indent=2))


if __name__ == "__main__":
    main()
