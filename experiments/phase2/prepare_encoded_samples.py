"""Encode fresh controlled data and audit predictor samples; no model training.

The CLI uses only the official pinned frozen I-JEPA float32/grid2 encoder. It
preserves the dataset preparation configuration, resumes the retained feature
cache by its exact signature, fits train-only normalization, and prepares
train/validation examples using the independent Phase 2 sampler. No predictor,
learned retention policy, deletion labels, or planning result is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE1_SRC = ROOT / "experiments/predictor/control/src"
for path in (ROOT, PHASE1_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.phase2.prepare_controlled_data import audit_observation_boundary
from experiments.phase2.training_samples import (
    DEFAULT_MODES, EpisodeIdentity, FeatureEpisode, SamplingConfig, SplitRegistry,
    phase1_arrays, sample_predictor,
)


OFFICIAL_REPOSITORY = "facebook/ijepa_vith14_1k"
OFFICIAL_REVISION = "f157467ea509bc356ff9f61fd3c0d840eec5e04e"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    from phase1.utils import write_json
    write_json(path, value)


def make_encoder_config(prepared: dict, data_root: Path, cache_dir: Path, *, device: str,
                        batch_size: int = 2, local_files_only: bool = False) -> dict:
    """Rebase execution paths only; preserve source data and normalization."""
    if device not in ("cpu", "cuda"):
        raise ValueError("explicit device must be cpu or cuda")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if prepared["data"]["seed"] in (6301, 9137):
        raise ValueError("This runner is for the new prepared development dataset, not old/reserved seeds")
    config = copy.deepcopy(prepared)
    expected = {"backend": "ijepa", "repository": OFFICIAL_REPOSITORY,
                "revision": OFFICIAL_REVISION, "pooling": "grid2", "dtype": "float32"}
    if any(config["encoder"].get(key) != value for key, value in expected.items()):
        raise ValueError("prepared data must declare the official pinned I-JEPA float32/grid2 encoder")
    config.update(output=str(data_root.resolve()), device=device)
    config["encoder"].update(cache_dir=str(cache_dir.resolve()), batch_size=batch_size,
                              local_files_only=bool(local_files_only))
    from phase1.config import validate
    validate(config)
    return config


def verify_retained_source() -> dict:
    control = ROOT / "experiments/predictor/control"
    lock = _read(control / "source_lock.json")["files"]
    names = ("envs.py", "data.py", "config.py", "utils.py", "encoders.py", "cache.py")
    hashes = {}
    for name in names:
        relative = "src/phase1/" + name
        actual = sha256(control / relative)
        if lock[relative] != actual:
            raise ValueError(f"retained Phase 1 scientific source changed: {name}")
        hashes[relative] = actual
    return {"source_lock_sha256": sha256(control / "source_lock.json"), "files": hashes}


def verify_prepared_inputs(data_root: Path, config: dict) -> tuple[dict, dict, dict]:
    from phase1.data import require_audit
    preparation = _read(data_root / "preparation_manifest.json")
    if preparation.get("status") != "new_cpu_dataset_generated_and_structurally_audited":
        raise ValueError("a successful new controlled-data preparation manifest is required")
    for name, expected in preparation["output_sha256"].items():
        candidate = (data_root / name).resolve()
        if not candidate.is_relative_to(data_root.resolve()) or sha256(candidate) != expected:
            raise ValueError(f"prepared input checksum differs: {name}")
    manifest = require_audit(config)
    index = _read(data_root / "observation_index.json")
    roles = _read(data_root / "split_roles.json")["roles"]
    boundary = audit_observation_boundary(config, manifest["records"])
    if index["dataset_signature"] != manifest["signature"] or index["records"] != boundary["observation_index"]:
        raise ValueError("observation index differs from the audited native dataset")
    if preparation["data_seed"] != config["data"]["seed"]:
        raise ValueError("preparation seed and runtime data config differ")
    if any(roles.get(split) != expected for split, expected in (("train", "training"), ("validation", "validation"))):
        raise ValueError("training and validation split roles must remain explicit")
    if any(roles.get(split) != "development_smoke" for split in config["data"]["pairs"] if split.startswith("test_")):
        raise ValueError("prepared test_* roles must remain development smoke")
    return manifest, index, roles


def _array_hash(array):
    a = np.ascontiguousarray(array)
    header = json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def _sample_fingerprints(sample):
    arrays, targets = phase1_arrays(sample)
    return {"model_arrays": {key: _array_hash(value) for key, value in sorted(arrays.items())},
            "target_array": _array_hash(targets),
            "causal_executed_actions": _array_hash(sample.causal.executed_actions)}


def summarize_encoded_samples(config: dict, observation_index: dict, split_roles: dict,
                              cache_index: dict, normalization: dict, report_dir: Path, *,
                              epochs: int = 2, base_seed: int = 17,
                              dropout: float = 0.15, expected_dimension: int = 5120) -> dict:
    """Audit actual cache arrays and emit compact TRAIN/VALIDATION sample rows.

    This helper can be tested with an explicitly labeled pixel fixture; it does
    not certify an encoder. The public run() separately enforces official I-JEPA.
    Every cached episode is verified/loaded, but test_* data are never used to
    generate predictor examples, choose a policy, or compute evaluation metrics.
    """
    if epochs < 1 or base_seed < 0:
        raise ValueError("positive sample epochs and nonnegative base seed required")
    from phase1.cache import load_feature, normalize
    from phase1.utils import digest
    records = observation_index["records"]
    identities = [EpisodeIdentity.from_record(rec) for rec in records]
    registry = SplitRegistry(identities, split_roles=split_roles)
    normalization_path = Path(config["output"]) / "features/normalization.npz"
    feature_space = digest({"cache_signature": cache_index["signature"],
                            "encoder_hash": cache_index["encoder_hash"],
                            "normalization_sha256": sha256(normalization_path)})
    sampling = SamplingConfig(recent_length=config["data"]["recent"], capacity=config["data"]["budget"],
                              horizons=tuple(config["data"]["horizons"]), memory_dropout=dropout)
    episodes = {}
    total, squares, frames, train_ids = None, None, 0, []
    cache_files = {}
    for record, identity in zip(records, identities):
        raw = load_feature(config, record, cache_index)
        if raw.shape != (record["frames"], expected_dimension):
            raise ValueError("encoded episode frame count or feature dimension mismatch")
        feature_path = Path(config["output"]) / "features" / cache_index["episodes"][record["id"]]["path"]
        cache_files[record["id"]] = {"path": str(feature_path.relative_to(config["output"])),
                                     "sha256": sha256(feature_path), "shape": list(raw.shape)}
        observation = Path(config["output"]) / "dataset" / record["obs"]
        if sha256(observation) != record["obs_sha256"]:
            raise ValueError("observed actions/query source checksum mismatch")
        with np.load(observation, allow_pickle=False) as archive:
            if set(archive.files) != {"rgb", "actions", "query"} or int(archive["query"]) != record["query"]:
                raise ValueError("observed archive schema or query mismatch")
            actions = archive["actions"]
        episodes[record["id"]] = FeatureEpisode(identity, normalize(raw, normalization), actions,
                                                 "additive_controls", feature_space)
        if record["split"] == "train":
            values = raw.astype(np.float64)
            total = values.sum(0) if total is None else total + values.sum(0)
            squares = (values * values).sum(0) if squares is None else squares + (values * values).sum(0)
            frames += len(values)
            train_ids.append(record["id"])
    if not train_ids or set(normalization["train_ids"].tolist()) != set(train_ids) or int(normalization["count"]) != frames:
        raise ValueError("normalization membership/count does not match training episodes")
    mean = total / frames
    std = np.maximum(np.sqrt(np.maximum(squares / frames - mean * mean, 0)), config["normalization"]["std_floor"])
    if not np.array_equal(normalization["mean"], mean.astype(np.float32)) or not np.array_equal(normalization["std"], std.astype(np.float32)):
        raise ValueError("normalization moments differ from independent train-only replay")

    rows = 0
    cells = defaultdict(lambda: {"examples": 0, "dropout_events": 0, "removed_entries": 0,
                                 "history_counts": Counter()})
    deterministic_validation = 0
    membership = {"train": set(), "validation": set()}
    groups = {"train": set(), "validation": set()}
    rows_path = report_dir / "sample_rows.jsonl"
    with rows_path.open("x") as stream:
        for record in records:
            split = record["split"]
            if split not in membership:
                continue
            ep = episodes[record["id"]]
            for epoch in range(epochs if split == "train" else 1):
                for mode in DEFAULT_MODES:
                    example = sample_predictor(ep, registry, sampling, query=record["query"], base_seed=base_seed,
                                               epoch=epoch, expected_split=split, history_mode=mode)
                    fingerprints = _sample_fingerprints(example)
                    if split == "validation":
                        repeated = sample_predictor(ep, registry, sampling, query=record["query"], base_seed=base_seed,
                                                    expected_split=split, history_mode=mode)
                        if dict(example.metadata) != dict(repeated.metadata) or fingerprints != _sample_fingerprints(repeated):
                            raise ValueError("validation sample changed on exact replay")
                        if example.metadata["dropout_event"] or example.metadata["dropout_probability"] != 0:
                            raise ValueError("validation unexpectedly applied training dropout")
                        deterministic_validation += 1
                    causal = example.causal
                    if any(e.timestamp >= causal.recent[0].timestamp for e in causal.history) or len(causal.history) > sampling.capacity:
                        raise ValueError("sample history crossed causal/capacity boundary")
                    if causal.executed_actions.shape != (record["query"], 2):
                        raise ValueError("sample includes wrong executed-action prefix")
                    if not example.future.available.all():
                        raise ValueError("registered controlled query does not supply both target horizons")
                    row = {**dict(example.metadata), "history_count": len(causal.history),
                           "recent_count": len(causal.recent), "horizon_mask": example.future.available.tolist(),
                           "horizons": list(example.future.horizons), "array_sha256": fingerprints,
                           "feature_cache_sha256": cache_files[record["id"]]["sha256"]}
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    rows += 1
                    membership[split].add(record["id"])
                    groups[split].add(record["pair"])
                    cell = cells[(split, record["task"], mode)]
                    cell["examples"] += 1
                    cell["dropout_events"] += int(example.metadata["dropout_event"])
                    cell["removed_entries"] += example.metadata["dropout_removed_entries"]
                    cell["history_counts"][len(causal.history)] += 1
    if membership["train"] & membership["validation"] or groups["train"] & groups["validation"]:
        raise ValueError("generated examples overlap across train/validation groups")
    summary = {
        "status": "encoded_predictor_samples_audited_without_training", "feature_space_id": feature_space,
        "encoder_backend": config["encoder"]["backend"], "encoder_name": cache_index["encoder"].get("name"),
        "feature_dimension": expected_dimension, "feature_episodes_loaded": len(episodes),
        "cache_files": cache_files, "sample_rows": rows, "sample_rows_sha256": sha256(rows_path),
        "sample_schedule": "balanced six independent sample modes per train episode per sample epoch; each validation mode once",
        "sample_epochs_not_training_epochs": epochs, "base_seed": base_seed, "memory_dropout": dropout,
        "sample_modes": list(DEFAULT_MODES), "capacity": sampling.capacity, "recent_length": sampling.recent_length,
        "normalization_train_only_verified": True, "normalization_train_episodes": len(train_ids),
        "normalization_train_frames": frames, "normalization_sha256": sha256(normalization_path),
        "normalization_moments_recomputed_exactly": True,
        "sample_episode_ids": {key: sorted(value) for key, value in membership.items()},
        "sample_pair_counts": {key: len(value) for key, value in groups.items()},
        "deterministic_validation_replays": deterministic_validation,
        "test_examples_generated": 0, "predictor_training": False, "basis_fitted": False,
        "baseline_prediction_results": False, "deletion_supervision_generated": False,
        "cells": [{"split": key[0], "task": key[1], "mode": key[2], **value,
                   "history_counts": dict(sorted(value["history_counts"].items()))}
                  for key, value in sorted(cells.items())],
    }
    _write(report_dir / "sample_summary.json", summary)
    return summary


def run(data_root: Path, cache_dir: Path, output: Path, *, device: str,
        batch_size=2, local_files_only=False, max_episodes=None,
        sample_epochs=2, base_seed=17, dropout=.15) -> dict:
    data_root, cache_dir, output = data_root.resolve(), cache_dir.resolve(), output.resolve()
    if output == data_root or (output.exists() and any(output.iterdir())):
        raise ValueError("output must be a new empty report directory, separate from prepared data")
    if max_episodes is not None and (isinstance(max_episodes, bool) or max_episodes < 1):
        raise ValueError("max_episodes must be positive")
    if isinstance(sample_epochs, bool) or not isinstance(sample_epochs, int) or sample_epochs < 1:
        raise ValueError("sample_epochs must be a positive integer")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    SamplingConfig(memory_dropout=dropout)
    config_path = data_root / "resolved_config.json"
    before_config_sha = sha256(config_path)
    config = make_encoder_config(_read(config_path), data_root, cache_dir, device=device,
                                 batch_size=batch_size, local_files_only=local_files_only)
    source_proof = verify_retained_source()
    manifest, observation_index, roles = verify_prepared_inputs(data_root, config)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write(output / "encoder_config.json", config)
    started = time.monotonic()
    progress = {"status": "encoding_started", "started_utc": datetime.now(timezone.utc).isoformat(),
                "data_root": str(data_root), "device": device, "dataset_signature": manifest["signature"],
                "prepared_config_sha256": before_config_sha, "source_proof": source_proof,
                "download_transport": {"HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET"),
                                       "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                                       "local_files_only": bool(local_files_only)},
                "predictor_training": False, "scientific_phase2_completed": False}
    _write(output / "progress.json", progress)
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from phase1.cache import encode, fit_normalization, load_index
        from phase1.utils import environment
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA explicitly requested but unavailable")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        previous = data_root / "features/index.json"
        progress["resume_existing_index"] = previous.exists()
        index = encode(config, max_episodes=max_episodes)
        progress.update(encoded_episodes=len(index["episodes"]), expected_episodes=len(manifest["records"]),
                        cache_signature=index["signature"], encoder_hash=index["encoder_hash"],
                        environment=environment(device))
        if len(index["episodes"]) != len(manifest["records"]):
            progress["status"] = "partial_cache_saved_resume_required"
            progress["normalization_fitted"] = False
        else:
            index = load_index(config)
            if index["encoder"].get("name") != OFFICIAL_REPOSITORY or index["encoder"].get("resolved_revision") != OFFICIAL_REVISION:
                raise ValueError("actual encoder identity differs from the pinned official model")
            progress["status"] = "normalization_and_sampling_started"
            _write(output / "progress.json", progress)
            normalization = fit_normalization(config)
            summary = summarize_encoded_samples(config, observation_index, roles, index, normalization, output,
                                                epochs=sample_epochs, base_seed=base_seed, dropout=dropout)
            progress.update(status="official_encoding_and_sample_audit_completed",
                            normalization_fitted=True, sample_rows=summary["sample_rows"],
                            feature_dimension=summary["feature_dimension"],
                            sample_summary_sha256=sha256(output / "sample_summary.json"))
        if sha256(config_path) != before_config_sha:
            raise ValueError("original preparation configuration was modified")
        progress.update(elapsed_seconds=time.monotonic() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        prepared_config_preserved=True,
                        feature_index_sha256=sha256(data_root / "features/index.json"),
                        encoder_config_sha256=sha256(output / "encoder_config.json"),
                        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
                            Path(__file__), ROOT / "experiments/phase2/training_samples.py",
                            ROOT / "experiments/phase2/memory.py", ROOT / "experiments/phase2/deletion.py")},
                        no_test_target_examples=True, predictor_trained=False,
                        learned_retention_scorer_trained=False, baseline_model_planning_verified=False)
        _write(output / "progress.json", progress)
        _write(output / "manifest.json", progress)
        print(json.dumps({key: progress.get(key) for key in ("status", "encoded_episodes", "expected_episodes", "sample_rows", "elapsed_seconds")}), flush=True)
        return progress
    except Exception:
        progress.update(status="failed", error=traceback.format_exc(), elapsed_seconds=time.monotonic() - started,
                        prepared_config_preserved=sha256(config_path) == before_config_sha)
        _write(output / "failure.json", progress)
        _write(output / "progress.json", progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--sample-epochs", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=.15)
    args = parser.parse_args()
    run(args.data_root, args.cache_dir, args.output, device=args.device, batch_size=args.batch_size,
        local_files_only=args.local_files_only, max_episodes=args.max_episodes,
        sample_epochs=args.sample_epochs, base_seed=args.base_seed, dropout=args.dropout)


if __name__ == "__main__":
    main()
