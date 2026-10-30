"""Prepare NEW CPU Gate/Motion development data with the retained generator.

No encoder, normalization, basis, checkpoint, or training is run. This is a
small structural pipeline dataset, not reproduction of inherited results or
the reserved independent-confirmation experiment.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE1_SRC = ROOT / "experiments/predictor/control/src"
REFERENCE = ROOT / "results/reference/gate/resolved_config.json"
SPLIT_ROLES = {
    "train": "training",
    "validation": "validation",
    "test_id": "development_smoke",
    "test_long_gap": "development_smoke",
    "test_distractor_shift": "development_smoke",
}
OBSERVATION_FIELDS = {"rgb", "actions", "query"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_config(reference: dict, output: Path, *, seed=20261020, train_pairs=16, eval_pairs=4) -> dict:
    """Preserve source dynamics; copy rather than edit the inherited configuration."""
    if seed in (6301, 9137):
        raise ValueError("Use a new data seed; inherited 6301 and reserved 9137 are excluded")
    if seed < 0 or train_pairs < 2 or eval_pairs < 2:
        raise ValueError("Require nonnegative new seed and at least two pairs per split")
    c = copy.deepcopy(reference)
    c.pop("experiment", None)
    c.update(mode="pilot", output=str(output.resolve()), device="cpu", threads=2)
    c["data"]["seed"] = seed
    c["data"]["pairs"] = {name: train_pairs if name == "train" else eval_pairs for name in SPLIT_ROLES}
    return c


def audit_observation_boundary(config: dict, records: list[dict]) -> dict:
    """Check model-readable archives and group identities without loading labels."""
    root = Path(config["output"]) / "dataset"
    observation_root = (root / "observations").resolve()
    episode_ids, group_splits, seed_groups = set(), {}, {}
    index, counts = [], {}
    frames = 0
    for rec in records:
        if rec["id"] in episode_ids:
            raise ValueError("Duplicate episode identity")
        episode_ids.add(rec["id"])
        if group_splits.setdefault(rec["pair"], rec["split"]) != rec["split"]:
            raise ValueError("Sibling group crosses splits")
        if seed_groups.setdefault(rec["seed"], rec["pair"]) != rec["pair"]:
            raise ValueError("Environment seed reused by a different scene group")
        if rec["split"] not in SPLIT_ROLES:
            raise ValueError("Unrecognized split role")
        path = (root / rec["obs"]).resolve()
        if not path.is_relative_to(observation_root):
            raise ValueError("Model-readable path must remain in observations, not diagnostics")
        if sha256(path) != rec["obs_sha256"]:
            raise ValueError("Observation checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != OBSERVATION_FIELDS:
                raise ValueError("Observation archive contains forbidden or missing fields")
            rgb, actions, query = archive["rgb"], archive["actions"], archive["query"]
            expected = (rec["frames"], config["data"]["image_size"], config["data"]["image_size"], 3)
            if rgb.shape != expected or rgb.dtype != np.uint8:
                raise ValueError("Invalid observation RGB shape or dtype")
            if actions.shape != (len(rgb)-1, 2) or actions.dtype != np.float32 or not np.isfinite(actions).all():
                raise ValueError("Invalid action sequence shape, dtype, or finite values")
            if query.shape != () or not np.issubdtype(query.dtype, np.integer) or int(query) != rec["query"]:
                raise ValueError("Query timestamp mismatch")
        clean = {k: rec[k] for k in ("id", "pair", "sibling", "split", "task", "seed", "query",
                                     "frames", "obs", "obs_sha256")}
        index.append(clean)
        key = (rec["task"], rec["split"])
        counts[key] = counts.get(key, 0) + 1
        frames += rec["frames"]
    by_id = {r["id"]: r for r in index}
    for rec in index:
        sibling = by_id.get(rec["sibling"])
        if sibling is None or sibling["sibling"] != rec["id"] or any(
            sibling[k] != rec[k] for k in ("pair", "split", "task", "seed", "query")):
            raise ValueError("Incomplete or inconsistent sibling group")
    return {
        "passed": True, "episodes": len(index), "pairs": len(group_splits),
        "unique_environment_seeds": len(seed_groups), "frames": frames,
        "checks": ["observation_archive_field_whitelist", "rgb_action_shape_dtype",
                   "observation_checksum", "query_consistency", "episode_identity_unique",
                   "pair_split_disjoint", "environment_seed_disjoint", "complete_sibling_groups",
                   "diagnostic_paths_excluded_from_observation_index"],
        "counts": [{"task": task, "split": split, "role": SPLIT_ROLES[split],
                    "episodes": number, "pairs": number//2}
                   for (task, split), number in sorted(counts.items())],
        "observation_index": index,
    }


def run(output: Path, *, seed=20261020, train_pairs=16, eval_pairs=4):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing data/evidence is preserved")
    sys.path.insert(0, str(PHASE1_SRC))
    from phase1.config import validate
    from phase1.data import generate, validate_data
    from phase1.utils import digest, write_json

    reference = json.loads(REFERENCE.read_text())
    config = make_config(reference, output, seed=seed, train_pairs=train_pairs, eval_pairs=eval_pairs)
    validate(config)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    started = time.perf_counter()
    manifest = generate(config)
    native_audit = validate_data(config)
    boundary = audit_observation_boundary(config, manifest["records"])
    index = boundary.pop("observation_index")
    write_json(output / "observation_index.json", {
        "schema_version": 1, "dataset_signature": manifest["signature"], "records": index,
        "data_root": "dataset", "model_inputs": ["rgb", "actions"],
        "identity_fields_not_model_features": ["id", "pair", "sibling", "split", "task", "seed", "query"],
        "action_semantics": "inherited controlled-task exogenous additive controls in [-1,1]",
        "future_recorded_actions": "offline target/action block only; not retention input",
    })
    write_json(output / "split_roles.json", {
        "schema_version": 1, "roles": SPLIT_ROLES,
        "note": "All test_* files are development smoke data. Structural audits inspect all splits; no untouched-test claim.",
        "train_only_fitting": "Any later normalization, basis, or model fitting must read train membership only.",
        "diagnostics": "dataset/evaluation_only is excluded from observation_index; states/labels/evidence never become model features.",
    })
    write_json(output / "observation_boundary_audit.json", boundary)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_paths = [PHASE1_SRC / "phase1" / name for name in ("envs.py", "data.py", "config.py", "utils.py")]
    source_paths += [Path(__file__), REFERENCE]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    outputs = [output/"resolved_config.json", output/"dataset/manifest.json", output/"dataset/split_audit.json",
               output/"observation_boundary_audit.json", output/"observation_index.json", output/"split_roles.json"]
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    provenance = {
        "schema_version": 1, "status": "new_cpu_dataset_generated_and_structurally_audited",
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "new_development_smoke_data_only",
        "data_seed": seed, "data_config_sha256": digest(config["data"]),
        "base_data_seed": reference["data"]["seed"], "reserved_confirmation_seed_accessed": False,
        "base_config": str(REFERENCE.relative_to(ROOT)), "source_commit": commit, "source_sha256": source_hashes,
        "changes_from_inherited_data_config": {"seed": seed, "pairs": config["data"]["pairs"]},
        "unchanged_dynamics": {k: config["data"][k] for k in ("image_size", "recent", "budget", "horizons", "gap",
                                                          "long_gap", "cue_start", "cue_duration", "motion_dt")},
        "counts": {k: boundary[k] for k in ("episodes", "pairs", "frames", "unique_environment_seeds")},
        "native_audit_passed": native_audit["passed"], "observation_boundary_passed": boundary["passed"],
        "elapsed_seconds": time.perf_counter()-started, "device": "cpu",
        "slurm_job_id": slurm_job_id,
        "gpu_allocation_seconds": None if slurm_job_id else 0,
        "resource_accounting_note": (
            "CPU execution inside a Slurm allocation; GPU allocation time is unknown here. "
            "Use scheduler accounting for this job, including CPU-only work and idle time."
            if slurm_job_id else
            "Local CPU execution; this command did not request or hold a Slurm GPU allocation."
        ),
        "encoder_run": False, "normalization_fitted": False, "basis_fitted": False, "predictor_trained": False,
        "baseline_results_generated": False, "deletion_supervision_generated": False,
        "inactive_config_sections": ["encoder", "normalization", "probe", "model", "training", "evaluation", "decision"],
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "PyYAML", "torch")},
        "output_sha256": {str(p.relative_to(output)): sha256(p) for p in outputs},
        "dataset_bytes": sum(p.stat().st_size for p in (output/"dataset").rglob("*") if p.is_file()),
        "reproduction": [".venv/bin/python", "experiments/phase2/prepare_controlled_data.py", "--output", "NEW_EMPTY_OUTPUT",
                         "--seed", str(seed), "--train-pairs", str(train_pairs), "--eval-pairs", str(eval_pairs)],
        "limitations": ["New data are not the teammate's missing original dataset or checkpoints.",
                        "Structural RGB checks do not establish that the frozen encoder/predictor can use the evidence.",
                        "No learned retention scorer or baseline model planning was run.",
                        "Inherited decision/training sections are copied but inactive; they do not create new proposal gates."],
    }
    write_json(output / "preparation_manifest.json", provenance)
    print(json.dumps({"output": str(output.resolve()), "status": provenance["status"],
                      **provenance["counts"], "native_audit_passed": True, "observation_boundary_passed": True,
                      "encoder_run": False, "predictor_trained": False}, indent=2))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"build/phase2-data-20261020")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--train-pairs", type=int, default=16)
    parser.add_argument("--eval-pairs", type=int, default=4)
    args = parser.parse_args()
    run(args.output, seed=args.seed, train_pairs=args.train_pairs, eval_pairs=args.eval_pairs)


if __name__ == "__main__":
    main()
