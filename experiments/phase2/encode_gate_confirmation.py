"""Standalone, CUDA/A100-only encoder for a locked independent Gate confirmation set.

Upload this one file to the existing HPC workspace. It imports only the pinned
phase1 envs/encoders/utils source from --repo-root, reuses the pinned OLD training
normalization, and performs no model/statistic fitting. Legacy ``test_*`` names
are passed only to the pure scene generator to select a condition. Every output
record is literature_confirmation; no old test observation/feature array is opened.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np


SEED = 20261023
PAIRS_PER_CONDITION = 32
PRIOR_DEVELOPMENT_RECORDS_SHA256 = "59b2de60df645277f2131299ca423f4a5bbb5aef12ce60ef1e2cf8f5e2efe8e0"
CONDITIONS = {"id": "test_id", "long_gap": "test_long_gap", "distractor_shift": "test_distractor_shift"}
DATA_CONFIG = {"seed": SEED, "tasks": ["gate"], "image_size": 96, "recent": 4,
               "horizons": [3, 6], "budget": 2, "gap": [5, 10], "long_gap": [20, 32],
               "cue_start": [2, 7], "cue_duration": [1, 3], "motion_dt": [3, 6]}
ENCODER_CONFIG = {"backend": "ijepa", "repository": "facebook/ijepa_vith14_1k",
                  "revision": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
                  "pooling": "grid2", "dtype": "float32", "batch_size": 2,
                  "smoke_size": 12, "local_files_only": True, "cache_dir": None}
EXPECTED_SOURCE_SHA256 = {
    "envs.py": "64b07f972fc5144d5d13c7bd614c1c796d67a73521043efc57073454ac4c12a7",
    "encoders.py": "7a407ad8f03ecf58bac6dd0bd0fceca0aaabae1201298083fe7ccf3cb5aee871",
    "utils.py": "03b0c1fb7f52fc822523125e5fe813cf48883459d48a038ffb22aa7ff87aef14",
}
OLD_NORMALIZATION_SHA256 = "2ecbe84ae1555e2e8f5fae49e49bab02fd5475f31ff7b4e4694e19791c61a690"
OLD_MANIFEST_SHA256 = "7188bd6de6b54eaaff75d22c8d3c631a1467f1d6e5b14c19bd1c269c37ee0d05"
EXPECTED_ENCODER_METADATA_SHA256 = "caaf35f0bbb97992cc49450dafdc6cfbe0e84395650bd61a48f3718f175e2fa4"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def verify_sources(repo_root):
    root = Path(repo_root).resolve() / "experiments/predictor/control/src"
    actual = {name: sha256(root / "phase1" / name) for name in EXPECTED_SOURCE_SHA256}
    if actual != EXPECTED_SOURCE_SHA256:
        raise ValueError("Inherited phase1 source hash mismatch: " + json.dumps(actual))
    sys.path.insert(0, str(root))
    envs = importlib.import_module("phase1.envs")
    if Path(envs.__file__).resolve() != root / "phase1/envs.py":
        raise ValueError("A shadow phase1 module was already imported")
    return envs, actual


def load_old_normalization(path):
    if sha256(path) != OLD_NORMALIZATION_SHA256:
        raise ValueError("Require the exact OLD train-only normalization artifact")
    with np.load(path, allow_pickle=False) as archive:
        mean, std = archive["mean"].copy(), archive["std"].copy()
    if (mean.shape != (5120,) or std.shape != mean.shape or mean.dtype != np.float32
            or std.dtype != np.float32 or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0)):
        raise ValueError("Invalid old I-JEPA normalization")
    return mean, std


def make_pair(envs, condition, index):
    """Pure generator plus RGB-level paired audit; no model or feature access."""
    if condition not in CONDITIONS or not isinstance(index, int) or not 0 <= index < PAIRS_PER_CONDITION:
        raise ValueError("Unknown fixed fresh condition/index")
    legacy_condition = CONDITIONS[condition]
    config = {"data": copy.deepcopy(DATA_CONFIG)}
    scene = envs.scene(config, "gate", legacy_condition, index)
    pair_id = digest(["gate-confirmation", SEED, condition, index])[:24]
    ids = [digest([pair_id, j, "opaque-id"])[:24] for j in range(2)]
    episodes = [envs.episode(scene, label) for label in scene["label_order"]]
    a, b = episodes
    q, recent, horizons = scene["query"], DATA_CONFIG["recent"], DATA_CONFIG["horizons"]
    if (not np.array_equal(a["rgb"][q-recent+1:q+1], b["rgb"][q-recent+1:q+1])
            or not np.array_equal(a["actions"], b["actions"])
            or not np.array_equal(a["evidence"], b["evidence"])
            or not all(t < q-recent+1 for t in a["evidence"])
            or np.array_equal(a["rgb"][a["evidence"]], b["rgb"][b["evidence"]])
            or {a["label"], b["label"]} != {0, 1}):
        raise ValueError("Paired evidence/recent/action RGB contract failed")
    for horizon in horizons:
        if (q + horizon >= len(a["rgb"]) or len(a["actions"][q:q+horizon]) != horizon
                or not a["visible"][q+horizon] or not b["visible"][q+horizon]
                or np.array_equal(a["rgb"][q+horizon], b["rgb"][q+horizon])
                or np.array_equal(a["states"][q+horizon], b["states"][q+horizon])):
            raise ValueError("Paired future/action-index contract failed")
    if condition == "long_gap" and not 20 <= scene["gap"] <= 32:
        raise ValueError("Long-gap range changed")
    if condition != "long_gap" and not 5 <= scene["gap"] <= 10:
        raise ValueError("Ordinary gap range changed")
    # Re-run both siblings to verify deterministic rendering and shared nuisance.
    regenerated_scene = envs.scene(config, "gate", legacy_condition, index)
    for episode in episodes:
        regenerated = envs.episode(regenerated_scene, episode["label"])
        if not np.array_equal(regenerated["rgb"], episode["rgb"]) or not np.array_equal(regenerated["actions"], episode["actions"]):
            raise ValueError("Fresh scene generation is not deterministic")
    records = [{"id": ids[j], "pair": pair_id, "sibling": ids[1-j], "split": "literature_confirmation",
                "condition": condition, "generation_condition": legacy_condition, "task": "gate",
                "seed": int(scene["seed"]), "scene_index": index, "query": q,
                "gap": int(scene["gap"]), "frames": int(scene["length"]),
                "feature_path": f"features/{ids[j]}.npz", "eval_path": f"evaluation_only/{ids[j]}.npz"}
               for j in range(2)]
    return records, episodes


def collect_dataset(envs, output, old_manifest, prior_records):
    if sha256(old_manifest) != OLD_MANIFEST_SHA256:
        raise ValueError("Old dataset metadata manifest changed")
    # Metadata only: do NOT resolve or open any old test_* observation files.
    old_records = json.loads(Path(old_manifest).read_text())["records"]
    old_seeds, old_ids = {r["seed"] for r in old_records}, {r["id"] for r in old_records}
    if sha256(prior_records) != PRIOR_DEVELOPMENT_RECORDS_SHA256:
        raise ValueError("Prior development records changed")
    prior = json.loads(Path(prior_records).read_text())
    prior_seeds = {r["seed"] for r in prior}
    prior_ids = {r["id"] for r in prior}
    records, payloads, unique_rgb, unique_hashes, locations = [], {}, [], [], {}
    scene_seeds = set()
    for condition in CONDITIONS:
        for index in range(PAIRS_PER_CONDITION):
            pair_records, pair_episodes = make_pair(envs, condition, index)
            seed = pair_records[0]["seed"]
            if seed in old_seeds or seed in prior_seeds or seed in scene_seeds:
                raise ValueError("Fresh scene seed overlaps old metadata or another fresh pair")
            scene_seeds.add(seed)
            for record, episode in zip(pair_records, pair_episodes):
                if record["id"] in old_ids or record["id"] in prior_ids:
                    raise ValueError("Fresh identity overlaps old data")
                indices = []
                for frame in episode["rgb"]:
                    key = hashlib.sha256(frame.tobytes()).hexdigest()
                    if key not in locations:
                        locations[key] = len(unique_rgb)
                        unique_rgb.append(frame.copy())
                        unique_hashes.append(key)
                    location = locations[key]
                    if not np.array_equal(unique_rgb[location], frame):
                        raise ValueError("RGB hash collision")
                    indices.append(location)
                record["rgb_indices"] = indices
                payloads[record["id"]] = {"actions": episode["actions"], "query": np.int64(record["query"])}
                np.savez_compressed(output / record["eval_path"], evidence=np.asarray(episode["evidence"], np.int64),
                                    label=np.int64(episode["label"]), states=episode["states"],
                                    classes=episode["classes"], visible=episode["visible"])
                record["eval_sha256"] = sha256(output / record["eval_path"])
                records.append(record)
    rgb = np.stack(unique_rgb)
    np.savez_compressed(output / "unique_rgb.npz", rgb=rgb)
    write_json(output / "unique_rgb_sha256.json", unique_hashes)
    audit = {"passed": True, "episodes": len(records), "independent_pairs": len(scene_seeds),
             "counts": {condition: {"pairs": PAIRS_PER_CONDITION, "episodes": 2 * PAIRS_PER_CONDITION}
                        for condition in CONDITIONS}, "total_frames": sum(r["frames"] for r in records),
             "unique_rgb_frames": len(rgb), "old_scene_seed_overlap": 0, "prior_development_seed_overlap": 0, "prior_development_records_sha256": PRIOR_DEVELOPMENT_RECORDS_SHA256, "old_test_arrays_opened": 0,
             "old_manifest_sha256": OLD_MANIFEST_SHA256,
             "checks": ["identical_recent_rgb", "identical_all_actions", "distinct_historical_evidence",
                        "distinct_visible_future_rgb_and_state", "exact_horizon_action_slices",
                        "balanced_opaque_siblings", "deterministic_regeneration", "nonoverlapping_scene_seeds",
                        "exact_rgb_hash_deduplication"],
             "limitation": "Independent new seeds from the same generator; confirms frozen literature/probe claims only, not the registered Phase1 protocol or new failure mechanisms."}
    write_json(output / "rgb_pair_audit.json", audit)
    return records, payloads, rgb, audit


def smi_rows(kind, fields):
    command = ["nvidia-smi", "--query-" + kind + "=" + ",".join(fields), "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
    return [{key: value.strip() for key, value in zip(fields, row)}
            for row in csv.reader(result.stdout.splitlines()) if len(row) == len(fields)]


def verify_a100():
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run inside a Slurm job with exactly one visible CUDA device")
    torch.cuda.set_device(0)
    marker = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    prop = torch.cuda.get_device_properties(0)
    inventory = smi_rows("gpu", ["index", "name", "uuid", "memory.total", "driver_version"])
    used_uuid = getattr(prop, "uuid", None)
    uuid_source = "torch.cuda.get_device_properties(0).uuid"
    if used_uuid is None:
        matches = {r["gpu_uuid"] for r in smi_rows("compute-apps", ["pid", "gpu_uuid"]) if r["pid"] == str(os.getpid())}
        if len(matches) != 1:
            raise RuntimeError("Cannot uniquely map current Torch CUDA process to a physical GPU UUID")
        used_uuid, uuid_source = matches.pop(), "nvidia-smi compute-apps current PID mapping"
    key = lambda value: str(value).lower().removeprefix("gpu-")
    matches = [r for r in inventory if key(r["uuid"]) == key(used_uuid)]
    if len(matches) != 1 or "A100" not in prop.name.upper() or "A100" not in matches[0]["name"].upper():
        raise RuntimeError("Actual Torch GPU must uniquely match an A100 nvidia-smi UUID")
    del marker
    return {"hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "pid": os.getpid(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "torch_device_name": prop.name, "actual_gpu_uuid": matches[0]["uuid"],
            "uuid_source": uuid_source, "matched_nvidia_smi_device": matches[0], "visible_devices": 1}


def run(args):
    output, repo = args.output.resolve(), args.repo_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; never replace an earlier run")
    if not isinstance(args.batch_size, int) or not 1 <= args.batch_size <= 16:
        raise ValueError("batch-size must be between 1 and 16")
    output.mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir()
    (output / "evaluation_only").mkdir()
    started = time.monotonic()
    manifest = {"schema_version": 1, "status": "running", "scope": "literature_confirmation",
                "seed": SEED, "data_config": DATA_CONFIG, "pairs_per_condition": PAIRS_PER_CONDITION,
                "conditions": CONDITIONS, "normalization_fitted": False, "basis_fitted": False,
                "predictor_trained": False, "student_trained": False, "old_test_arrays_opened": 0,
                "helper_sha256": sha256(__file__), "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "manifest.json", manifest)
    try:
        envs, source_hashes = verify_sources(repo)
        mean, std = load_old_normalization(args.normalization)
        old_manifest = args.old_manifest or args.normalization.resolve().parent.parent / "dataset/manifest.json"
        manifest.update(source_sha256=source_hashes, old_normalization_sha256=OLD_NORMALIZATION_SHA256,
                        old_manifest_sha256=OLD_MANIFEST_SHA256, prior_development_records_sha256=PRIOR_DEVELOPMENT_RECORDS_SHA256)
        # This config must be set before the first CUDA context is initialized.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        hardware = verify_a100()
        write_json(output / "hardware.json", hardware)
        print(json.dumps({"hardware_verified": hardware}), flush=True)
        records, payloads, rgb, rgb_audit = collect_dataset(envs, output, old_manifest, args.prior_development_records)
        write_json(output / "records.json", records)
        print(json.dumps({"rgb_audit": rgb_audit}), flush=True)

        encoders = importlib.import_module("phase1.encoders")
        if Path(encoders.__file__).resolve() != repo / "experiments/predictor/control/src/phase1/encoders.py":
            raise ValueError("Unexpected encoder module origin")
        config = copy.deepcopy(ENCODER_CONFIG)
        config.update(cache_dir=str(args.cache_dir.resolve()), batch_size=args.batch_size)
        write_json(output / "encoder_config.json", config)
        encoder = encoders.FrozenIJepa(config, torch.device("cuda:0"))
        metadata_hash = digest(encoder.metadata)
        if metadata_hash != EXPECTED_ENCODER_METADATA_SHA256:
            raise ValueError("Encoder/processor metadata differs from the old feature definition: " + metadata_hash)
        values = list(encoder.model.parameters()) + list(encoder.model.buffers())
        if (any(module.training for module in encoder.model.modules())
                or any(p.requires_grad for p in encoder.model.parameters())
                or any(p.device.type != "cuda" or p.device.index != 0 for p in values)):
            raise ValueError("Encoder must be frozen/eval on the verified GPU")
        versions = tuple(value._version for value in values)
        audit = {"all_modules_eval": True, "trainable_parameters": 0, "device": "cuda:0",
                 "actual_gpu_uuid": hardware["actual_gpu_uuid"], "metadata_sha256": metadata_hash,
                 "metadata": encoder.metadata, "batch_size": args.batch_size}
        write_json(output / "frozen_encoder_audit.json", audit)
        encoded = []
        encoding_started = time.monotonic()
        for start in range(0, len(rgb), args.batch_size):
            z = encoder.encode(rgb[start:start + args.batch_size])
            if z.shape != (min(args.batch_size, len(rgb) - start), 5120) or z.dtype != np.float32 or not np.isfinite(z).all():
                raise ValueError("Invalid I-JEPA output shape/dtype/value")
            encoded.append(z)
            if start % (args.batch_size * 25) == 0:
                print(json.dumps({"unique_frames_encoded": start + len(z), "total_unique_frames": len(rgb)}), flush=True)
        torch.cuda.synchronize()
        encoding_seconds = time.monotonic() - encoding_started
        if versions != tuple(value._version for value in values):
            raise ValueError("Frozen encoder tensor version changed during inference")
        raw = np.concatenate(encoded)
        normalized = ((raw - mean) / std).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Nonfinite old-normalized fresh features")
        # Deduplicated frames are encoded once, then expanded from exact same
        # vector, guaranteeing bitwise-equal features for bitwise-equal RGB.
        np.savez_compressed(output / "unique_features.npz", raw=raw, normalized=normalized)
        for record in records:
            z = normalized[record["rgb_indices"]]
            np.savez_compressed(output / record["feature_path"], z=z, **payloads[record["id"]])
            record["feature_sha256"] = sha256(output / record["feature_path"])
        by_id = {r["id"]: r for r in records}
        for record in records:
            q = record["query"]
            sibling = by_id[record["sibling"]]
            a, b = np.asarray(record["rgb_indices"]), np.asarray(sibling["rgb_indices"])
            if not np.array_equal(normalized[a[q-3:q+1]], normalized[b[q-3:q+1]]):
                raise ValueError("Exact paired recent feature identity failed")
        write_json(output / "records.json", records)
        if sha256(args.normalization) != OLD_NORMALIZATION_SHA256:
            raise ValueError("Old normalization changed during encoding")
        verify_sources(repo)
        manifest.update(status="completed", episodes=len(records), independent_pairs=96,
                        total_frames=rgb_audit["total_frames"], unique_rgb_frames=len(rgb),
                        encoding_seconds=encoding_seconds, exact_shared_rgb_features=True,
                        encoder_metadata_sha256=metadata_hash, hardware=hardware,
                        peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()),
                        elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(output / "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = {str(path.relative_to(output)): sha256(path)
                                     for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output),
                      "episodes": manifest["episodes"], "unique_rgb_frames": manifest["unique_rgb_frames"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path)
    parser.add_argument("--prior-development-records", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
