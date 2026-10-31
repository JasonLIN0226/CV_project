"""Frozen, one-shot Motion v2.3 evaluation of the v2.1 predictor experiment.

This command requires a committed protocol and an existing single-A100 Slurm
allocation. It does not log in, submit a job, resume, retry, or delete files.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback


sys.dont_write_bytecode = True


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "results/phase1_independent_confirmation/execution_v10_checkpoint_resume/source/original_src/phase1"
CONDITIONS = ("recent_only", "full_history", "privileged_small_budget", "random_small_budget")
EVAL_CONDITIONS = ("a", "b", "c", "d")
SEEDS = (17, 29, 43)
REVISION = "f157467ea509bc356ff9f61fd3c0d840eec5e04e"


class BudgetStop(RuntimeError):
    pass


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def command(arguments, cwd=None):
    return subprocess.run(arguments, cwd=cwd, check=True, text=True, capture_output=True,
                          timeout=30).stdout.strip()


def local_module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("motion_v2_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def committed_file(path):
    path = Path(path).resolve()
    relative = path.relative_to(REPO).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO,
                            check=True, capture_output=True, timeout=30)
    if result.stdout != path.read_bytes():
        raise ValueError(f"File is not identical to the committed HEAD: {relative}")
    return relative


def checked_repo_file(relative, expected_hash):
    path = (REPO / relative).resolve()
    path.relative_to(REPO)
    if sha256(path) != expected_hash:
        raise ValueError(f"Frozen source/input SHA256 mismatch: {relative}")
    committed_file(path)
    return path


def validate_registration(args, config, protocol):
    if protocol.get("status") != "FROZEN_STEP2_PROTOCOL" or protocol.get("registered_before_run") is not True:
        raise ValueError("Step 2 requires its committed, pre-run protocol")
    committed_file(args.protocol)
    committed_file(args.config)
    if sha256(args.config) != protocol["config_sha256"]:
        raise ValueError("Config differs from registered SHA256")
    source_hashes = protocol["source_sha256"]
    required = {Path(__file__).resolve().relative_to(REPO).as_posix(),
                "experiments/motion_v2/reader_v2.py", "experiments/motion_v2/step2_metrics.py",
                "experiments/motion_v2/prepare_direction_bound_v2_3.py"}
    required.update(p.relative_to(REPO).as_posix() for p in REFERENCE.glob("*.py"))
    if not required <= set(source_hashes):
        raise ValueError("Protocol does not lock the complete runtime source set")
    for relative, expected in source_hashes.items():
        checked_repo_file(relative, expected)
    expected_bound = {"method": "paired_midpoint_position_mse", "split": "validation",
                      "pairs": 96, "horizons": [3, 6], "prepared_before_gpu": True}
    if protocol.get("direction_bound") != expected_bound:
        raise ValueError("The scientific protocol must register CPU preparation of the v2.3 direction bound")
    audit = read_json(checked_repo_file(protocol["step1_audit_path"], protocol["step1_audit_sha256"]))
    status = read_json(checked_repo_file(protocol["step1_status_path"], protocol["step1_status_sha256"]))
    if audit.get("status") != "passed" or status.get("status") != "passed":
        raise ValueError("Step 1 has not passed")
    if audit["structure"]["pairs_checked"] != 608 or audit["structure"]["pairs_passed"] != 608:
        raise ValueError("Step 1 audit does not cover all 608 pairs")
    manifest_path = args.dataset / "manifest.json"
    actual_hash = sha256(manifest_path)
    if actual_hash != protocol["dataset_manifest_sha256"] or actual_hash != audit["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the passing Step 1 audit")
    manifest = read_json(manifest_path)
    records = manifest["records"]
    expected_counts = {("motion", "train"): 1024, ("motion", "validation"): 192}
    if Counter((r["task"], r["split"]) for r in records) != expected_counts:
        raise ValueError("Only the registered Motion train and validation records are allowed")
    if len({r["id"] for r in records}) != 1216:
        raise ValueError("Duplicate or missing episode IDs")
    data = config["data"]
    if (data["seed"] != 20261030 or data["tasks"] != ["motion"] or
            data["pairs"] != {"train": 512, "validation": 96} or
            data["image_size"] != 96 or data["recent"] != 4 or data["budget"] != 2 or
            data["horizons"] != [3, 6]):
        raise ValueError("Unexpected Step 2 dataset settings")
    if config["generator_version"] != "motion_v2.1" or manifest["generator_version"] != "motion_v2.1":
        raise ValueError("Unexpected generator version")
    if manifest["config_hash"] != canonical_digest(data):
        raise ValueError("Runtime data configuration differs from the audited generator configuration")
    encoder = config["encoder"]
    if (encoder["backend"] != "ijepa" or encoder["repository"] != "facebook/ijepa_vith14_1k" or
            encoder["revision"] != REVISION or encoder["dtype"] != "float32" or
            encoder["pooling"] != "grid2" or encoder["batch_size"] != 2):
        raise ValueError("Encoder settings differ from the frozen official I-JEPA recipe")
    expected_training = {"seeds": list(SEEDS), "epochs": 120, "batch_size": 64, "lr": 0.001,
                         "weight_decay": 0.0001, "clip": 1.0, "lr_step": 40, "lr_gamma": 0.5,
                         "pair_weight": 2.0, "rank_weight": 0.5}
    if any(config["training"].get(key) != value for key, value in expected_training.items()):
        raise ValueError("Training architecture/budget must retain the registered Phase 1 recipe")
    if config["model"] != {"hidden": 128, "heads": 4, "layers": 2, "dropout": 0.0, "rank": 128}:
        raise ValueError("Predictor architecture differs from Phase 1")
    if config["normalization"]["std_floor"] != 0.05:
        raise ValueError("Unexpected normalization floor")
    if config["threads"] != 4:
        raise ValueError("Exactly four CPU science threads are registered")
    if config["evaluation"]["bootstrap"] != 2000 or config["evaluation"]["random_seed"] != 901:
        raise ValueError("Unexpected bootstrap settings")
    execution = config["execution"]
    if (execution["gpu_count"] != 1 or execution["estimated_gpu_seconds"] != 7200 or
            execution["hard_cap_gpu_seconds"] != 14400 or execution["memory_dropout"] != 0.25 or
            execution["encoding_frames_per_second"] != 27.15545 or
            execution["training_seconds_per_epoch"] != 5.72894 or
            execution["remaining_overhead_seconds"] != 3600 or execution["max_frames"] != 52288):
        raise ValueError("Execution budget differs from the frozen estimate")
    frames = sum(r["frames"] for r in records)
    if frames != execution["expected_frames"] or frames > execution["max_frames"]:
        raise ValueError("Actual frame count differs from the registered resource estimate")
    estimate = (frames / execution["encoding_frames_per_second"] +
                360 * execution["training_seconds_per_epoch"] + execution["remaining_overhead_seconds"])
    if estimate > execution["estimated_gpu_seconds"]:
        raise BudgetStop("The registered work cannot fit its GPU estimate")
    # Verify existing source artifacts without opening any unregistered split.
    for rec in records:
        for kind in ("obs", "eval"):
            path = (args.dataset / rec[kind]).resolve()
            path.relative_to(args.dataset.resolve())
            if sha256(path) != rec[kind + "_sha256"]:
                raise ValueError(f"Step 1 artifact SHA256 changed: {rec['id']} {kind}")
    return manifest, {"head_commit": command(["git", "rev-parse", "HEAD"], REPO),
                      "protocol_sha256": sha256(args.protocol), "config_sha256": sha256(args.config),
                      "manifest_sha256": actual_hash, "frames": frames,
                      "estimated_work_seconds": estimate, "source_sha256": source_hashes}


def slurm_seconds(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, minutes, seconds = (int(v) for v in clock.split(":"))
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def native_allocation():
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError("A native Slurm allocation is required; no local or login fallback")
    output = command(["sacct", "-j", job_id, "--noheader", "--parsable2",
                      "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES"])
    rows = [line.split("|") for line in output.splitlines() if line.split("|", 1)[0] == job_id]
    if len(rows) != 1 or len(rows[0]) < 5:
        raise RuntimeError("Cannot establish native current-job allocation accounting")
    row = rows[0]
    if row[1] != "RUNNING" or not row[3].isdigit():
        raise RuntimeError("The current Slurm allocation is not confirmed RUNNING")
    gpu_tres = {key: int(v) for key, v in
                (part.split("=", 1) for part in row[4].split(",") if "=" in part)
                if key == "gres/gpu" or key.startswith("gres/gpu:")}
    gpu_counts = list(gpu_tres.values())
    typed_gpu_count = sum(count for key, count in gpu_tres.items() if key.startswith("gres/gpu:"))
    if (not gpu_counts or any(count != 1 for count in gpu_counts) or
            typed_gpu_count not in (0, 1)):
        raise RuntimeError("Native AllocTRES must confirm exactly one allocated GPU")
    if any("a100" not in key.lower() for key in gpu_tres if key.startswith("gres/gpu:")):
        raise RuntimeError("Native typed GPU allocation is not an A100")
    control = command(["scontrol", "show", "job", "--oneliner", job_id])
    fields = dict(item.split("=", 1) for item in control.split() if "=" in item)
    if fields.get("JobId") != job_id or fields.get("JobState") != "RUNNING" or fields.get("NumNodes") != "1":
        raise RuntimeError("Current job must be RUNNING on one node")
    if fields.get("Restarts") != "0":
        raise RuntimeError(f"Scientific job must have native Restarts=0; scontrol={control}")
    if not fields.get("StartTime") or fields["StartTime"] == "Unknown":
        raise RuntimeError("Native scientific job start identity is unavailable")
    elapsed = max(int(row[3]), slurm_seconds(fields["RunTime"]))
    if fields.get("TimeLimit") not in ("UNLIMITED", "Partition_Limit"):
        time_limit = slurm_seconds(fields["TimeLimit"])
        if time_limit > 7200:
            raise RuntimeError("Native Slurm time limit must be no more than the registered 7200 seconds")
    else:
        raise RuntimeError("A finite registered native Slurm time limit is required")
    return {"job_id": job_id, "start": row[2], "elapsed_seconds_at_probe": elapsed,
            "alloc_tres": row[4], "scontrol": control, "sacct": output,
            "native_start_time": fields["StartTime"], "restarts": 0,
            "requeue": fields.get("Requeue"),
            "native_time_limit_seconds": time_limit, "probe_utc": datetime.now(timezone.utc).isoformat()}


class BudgetGuard:
    def __init__(self, config, allocation, frames, output):
        self.config = config["execution"]
        self.allocation = allocation
        self.started = time.monotonic()
        self.initial_elapsed = allocation["elapsed_seconds_at_probe"]
        self.phase = "overhead"
        self.phase_started = self.started
        self.totals = {"overhead": float(self.initial_elapsed), "encoding": 0.0, "training": 0.0}
        self.total_frames, self.frames = frames, 0
        self.epoch_seconds = []
        self.interrupted = None
        self.last_native_probe = self.started
        self.visible_gpu_probe = None
        self.native_probe_count = 1
        self.native_probe_path = Path(output) / "native_allocation_checks.jsonl"
        with self.native_probe_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "allocation_start", "allocation": allocation}, allow_nan=False) + "\n")

    def check_native(self, stage, force=False):
        if not force and time.monotonic() - self.last_native_probe < 30:
            return
        evidence = {"stage": stage, "probe_utc": datetime.now(timezone.utc).isoformat()}
        try:
            allocation = native_allocation()
            evidence["allocation"] = allocation
            for key in ("job_id", "start", "native_start_time", "alloc_tres", "native_time_limit_seconds"):
                if allocation[key] != self.allocation[key]:
                    raise RuntimeError(f"Native scientific allocation identity changed: {key}")
            if self.visible_gpu_probe is not None:
                evidence["visible_gpu"] = self.visible_gpu_probe()
            evidence["status"] = "passed"
            self.last_native_probe = time.monotonic()
            self.native_probe_count += 1
        except Exception as exc:
            evidence.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            with self.native_probe_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evidence, allow_nan=False) + "\n")

    def phase_totals(self):
        totals = dict(self.totals)
        totals[self.phase] += time.monotonic() - self.phase_started
        return totals

    def set_phase(self, phase):
        self.totals = self.phase_totals()
        self.phase, self.phase_started = phase, time.monotonic()

    def elapsed(self):
        return self.initial_elapsed + time.monotonic() - self.started

    def remaining_estimate(self):
        totals = self.phase_totals()
        rate = self.config["encoding_frames_per_second"]
        if self.frames >= 64:
            rate = self.frames / max(totals["encoding"], 1e-9)
        epoch_rate = (sum(self.epoch_seconds) / len(self.epoch_seconds)
                      if self.epoch_seconds else self.config["training_seconds_per_epoch"])
        overhead = max(0.0, self.config["remaining_overhead_seconds"] - totals["overhead"])
        return (self.total_frames - self.frames) / rate + (360 - len(self.epoch_seconds)) * epoch_rate + overhead

    def check(self, stage="unspecified", force_native=False):
        if self.interrupted:
            raise BudgetStop(self.interrupted)
        self.check_native(stage, force=force_native)
        elapsed = self.elapsed()
        limit = min(self.config["estimated_gpu_seconds"], self.config["hard_cap_gpu_seconds"],
                    self.allocation["native_time_limit_seconds"])
        if elapsed >= limit:
            raise BudgetStop(f"Allocated GPU wall time reached its registered limit during {stage}")
        if elapsed + self.remaining_estimate() > limit:
            raise BudgetStop(f"Registered work is projected to exceed allocated GPU estimate during {stage}")

    def snapshot(self):
        return {"actual_allocated_gpu_seconds_so_far": self.elapsed(),
                "estimated_remaining_seconds": self.remaining_estimate(),
                "phase_wall_seconds": self.phase_totals(), "encoded_frames": self.frames,
                "completed_epochs": len(self.epoch_seconds),
                "native_allocation_checks": self.native_probe_count,
                "native_restarts_required": 0, "native_check_interval_seconds": 30,
                "gpu_count": 1, "estimate_limit_seconds": self.config["estimated_gpu_seconds"],
                "hard_cap_seconds": self.config["hard_cap_gpu_seconds"]}


def pair_groups(records, split):
    groups = {}
    for rec in records:
        if rec["split"] == split:
            groups.setdefault(rec["pair"], []).append(rec)
    pairs = list(groups.values())
    if any(len(pair) != 2 or pair[0]["sibling"] != pair[1]["id"] or
           pair[1]["sibling"] != pair[0]["id"] for pair in pairs):
        raise ValueError("All batches require complete reciprocal sibling pairs")
    return pairs


def encode_all(config, manifest, dataset, output, reader, guard):
    import numpy as np
    import torch
    guard.check("before_encoder_load")
    encoder = reader.load_reference("encoders").FrozenIJepa(config["encoder"], torch.device("cuda:0"))
    if encoder.metadata["resolved_revision"] != REVISION:
        raise RuntimeError("Resolved official I-JEPA revision does not match the frozen commit")
    if encoder.metadata["pooling"] != "grid2" or encoder.metadata["dtype"] != "float32":
        raise RuntimeError("Encoder metadata differs from FP32 four-cell pooling")
    write_json(output / "encoder_metadata.json", encoder.metadata)
    guard.check("after_encoder_load")
    root = output / "features"
    episodes = {}
    total = np.zeros(5120, dtype=np.float64)
    square = np.zeros(5120, dtype=np.float64)
    count = 0
    train_ids = []
    guard.set_phase("encoding")
    for rec in manifest["records"]:
        with np.load(dataset / rec["obs"], allow_pickle=False) as file:
            if set(file.files) != {"rgb", "actions", "query"}:
                raise ValueError("Observation schema is not the registered RGB/actions/query boundary")
            rgb = file["rgb"]
        if rgb.shape != (rec["frames"], 96, 96, 3) or rgb.dtype != np.uint8:
            raise ValueError("Unexpected source RGB shape/dtype")
        chunks = []
        for start in range(0, len(rgb), 2):
            guard.check("encoder_batch_before")
            features = encoder.encode(rgb[start:start + 2])
            torch.cuda.synchronize()
            if features.shape != (min(2, len(rgb) - start), 5120) or features.dtype != np.float32 or not np.isfinite(features).all():
                raise RuntimeError("Encoder returned invalid 5120-dimensional FP32 features")
            chunks.append(features)
            guard.frames += len(features)
            guard.check("encoder_batch_after")
        values = np.concatenate(chunks)
        relative = f"episodes/{rec['id']}.npz"
        path = root / relative
        write_npz(path, features=values)
        episodes[rec["id"]] = {"path": relative, "sha256": sha256(path), "frames": len(values)}
        if rec["split"] == "train":
            as_double = values.astype(np.float64)
            total += as_double.sum(axis=0)
            square += np.square(as_double).sum(axis=0)
            count += len(values)
            train_ids.append(rec["id"])
    guard.set_phase("overhead")
    # Releasing RAM/VRAM does not remove any saved model or user file.
    del encoder
    torch.cuda.empty_cache()
    signature = canonical_digest({"dataset_manifest_sha256": sha256(dataset / "manifest.json"),
                                  "encoder": config["encoder"], "episodes": episodes})
    index = {"signature": signature, "manifest_sha256": sha256(dataset / "manifest.json"),
             "feature_dim": 5120, "episodes": episodes}
    write_json(root / "index.json", index)
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(square / count - mean * mean, 0)), 0.05)
    write_npz(root / "normalization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32),
              train_ids=np.asarray(train_ids), count=np.int64(count),
              signature=np.asarray(signature), floor=np.float32(0.05))
    write_json(root / "normalization_provenance.json", {"fit_split": "train", "episodes": len(train_ids),
               "frames": count, "all_train_frames": True, "float64_accumulation": True,
               "std_floor": 0.05, "sha256": sha256(root / "normalization.npz")})
    guard.check("after_train_only_normalization")
    return index


def batch_samples(store, rows, condition, rng=None):
    import numpy as np
    samples = []
    if rng is None:
        samples = [store.sample(rec, condition) for rec in rows]
    else:
        for start in range(0, len(rows), 2):
            sample_seed = int(rng.integers(2**32))
            samples.extend(store.sample(rec, condition, np.random.default_rng(sample_seed))
                           for rec in rows[start:start + 2])
    return [sample[0] for sample in samples], np.stack([sample[1] for sample in samples])


def validation_score(model, store, pairs, collate, guard):
    import numpy as np
    import torch
    store.training = False
    model.eval()
    summary = {}
    with torch.inference_mode():
        for condition in CONDITIONS:
            total, count = 0.0, 0
            for start in range(0, len(pairs), 32):
                guard.check("validation_batch")
                rows = [rec for pair in pairs[start:start + 32] for rec in pair]
                samples, target = batch_samples(store, rows, condition)
                prediction = model(collate(samples, torch.device("cuda:0")))
                truth = torch.from_numpy(target).to("cuda:0")
                per_example_horizon = (prediction - truth).square().mean(-1)
                if not torch.isfinite(per_example_horizon).all():
                    raise RuntimeError("Non-finite validation loss")
                total += float(per_example_horizon.sum())
                count += per_example_horizon.numel()
            summary[condition] = total / count
    return float(np.mean(list(summary.values()))), summary


def train_predictors(config, store, basis, reader, output, guard, provenance):
    import numpy as np
    import torch
    model_module = reader.load_reference("model")
    training = reader.load_reference("training")
    collate = reader.load_reference("inputs").collate
    utils = reader.load_reference("utils")
    train_pairs = pair_groups(store.manifest["records"], "train")
    validation_pairs = pair_groups(store.manifest["records"], "validation")
    selected = {}
    for seed in SEEDS:
        utils.seed_all(seed, config["threads"])
        root = output / "predictor" / "motion" / f"seed_{seed}"
        root.mkdir(parents=True, exist_ok=False)
        model = model_module.DynamicsPredictor(5120, config["model"], [3, 6], basis).to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        best_score, best_path, steps = math.inf, None, 0
        guard.set_phase("training")
        with (root / "training.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(120):
                epoch_start = time.monotonic()
                store.training = True
                model.train()
                rng = np.random.default_rng(seed + 100003 * epoch)
                order = rng.permutation(len(train_pairs))
                counts = {name: 0 for name in CONDITIONS}
                train_mse_sum, train_examples = 0.0, 0
                epoch_steps = 0
                for start in range(0, len(order), 32):
                    rows = [rec for pair_index in order[start:start + 32] for rec in train_pairs[pair_index]]
                    for condition in rng.permutation(CONDITIONS):
                        condition = str(condition)
                        guard.check("optimizer_step_before")
                        samples, targets = batch_samples(store, rows, condition, rng)
                        target = torch.from_numpy(targets).to("cuda:0")
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(collate(samples, torch.device("cuda:0")))
                        loss, mse, _ = training.regression_loss(prediction, target, 2.0, 0.5)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        torch.cuda.synchronize()
                        counts[condition] += len(rows)
                        train_mse_sum += float(mse.detach()) * len(rows)
                        train_examples += len(rows)
                        steps += 1
                        epoch_steps += 1
                        guard.check("optimizer_step_after")
                score, components = validation_score(model, store, validation_pairs, collate, guard)
                scheduler.step()
                if epoch_steps != 64 or any(value != 1024 for value in counts.values()):
                    raise RuntimeError("Registered training budget was not executed exactly")
                if score < best_score:
                    best_score = score
                    best_path = root / f"best_epoch_{epoch + 1:03d}.pt"
                    checkpoint = {"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                                  "epoch": epoch + 1, "seed": seed, "selection_score": score,
                                  "selection_rule": "earliest strict minimum of four-condition/two-horizon validation feature MSE",
                                  "protocol_sha256": provenance["protocol_sha256"],
                                  "dataset_manifest_sha256": provenance["manifest_sha256"]}
                    with best_path.open("xb") as handle:
                        torch.save(checkpoint, handle)
                duration = time.monotonic() - epoch_start
                guard.epoch_seconds.append(duration)
                row = {"epoch": epoch + 1, "seed": seed, "steps": steps,
                       "training_mse": train_mse_sum / train_examples, "condition_episode_counts": counts,
                       "validation_mse": score, "validation_conditions": components,
                       "selected_checkpoint": best_path.name, "epoch_wall_seconds": duration,
                       "allocated_gpu_seconds": guard.elapsed()}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps({"stage": "training", **row}, allow_nan=False), flush=True)
                guard.check("epoch_complete")
        guard.set_phase("overhead")
        if steps != 7680 or best_path is None:
            raise RuntimeError("Incomplete seed training")
        selected[seed] = best_path
        write_json(root / "selection.json", {"seed": seed, "checkpoint": best_path.name,
                   "sha256": sha256(best_path), "validation_mse": best_score,
                   "epochs_completed": 120, "optimizer_steps": steps, "test_used": False})
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        guard.check("seed_complete")
    return selected


def validate_decoder_on_true_targets(store, decoder, output, guard):
    """Report frozen-readout error on true validation features, without a gate.

    This diagnosis does not fit a decoder, select a ridge/checkpoint, alter a
    threshold, or change the predictor training/evaluation sequence.
    """
    import numpy as np
    pairs = pair_groups(store.manifest["records"], "validation")
    if len(pairs) != 96:
        raise ValueError("Frozen-decoder diagnosis requires all 96 validation pairs")
    losses = np.empty((96, 2, 2), dtype=np.float64)
    for start in range(0, len(pairs), 32):
        guard.check("decoder_true_validation_targets_before")
        rows = [rec for pair in pairs[start:start + 32] for rec in pair]
        feature_rows, true_positions = [], []
        for rec in rows:
            times = rec["query"] + np.asarray([3, 6])
            features = store.features(rec)[times]
            meta = store.read_evaluation(rec, keys=("states", "visible"))
            if not np.all(meta["visible"][times]):
                raise ValueError("Frozen-decoder diagnostic target must be visible")
            feature_rows.append(features)
            true_positions.append(meta["states"][times])
        true_features = np.stack(feature_rows)
        decoded = np.asarray(decoder.predict(true_features.reshape(-1, 5120)))
        decoded = decoded.reshape(len(rows), 2, 2).astype(np.float64)
        truth = np.stack(true_positions).astype(np.float64)
        error = np.square(decoded - truth).mean(-1)
        if error.shape != (len(rows), 2) or not np.isfinite(error).all():
            raise RuntimeError("Invalid frozen-decoder diagnostic position MSE")
        count = len(rows) // 2
        losses[start:start + count] = error.reshape(count, 2, 2)
        guard.check("decoder_true_validation_targets_after")
    raw_path = output / "decoder" / "validation_true_target_losses.npz"
    write_npz(raw_path, position_mse=losses,
              pair_ids=np.asarray([pair[0]["pair"] for pair in pairs]), horizons=np.asarray([3, 6]))
    report = {
        "status": "diagnostic_only", "decoder_fit_split": "train", "evaluation_split": "validation",
        "feature_input": "true normalized I-JEPA target features, not predictor outputs",
        "position_mse_definition": "mean squared error over decoded x/y coordinates, averaged over 192 episodes",
        "pairs": 96, "episodes": 192, "raw_loss_shape": [96, 2, 2],
        "horizon_position_mse": {str(h): float(losses[:, :, i].mean()) for i, h in enumerate((3, 6))},
        "decoder_refit": False, "checkpoint_or_ridge_selection_uses_this": False,
        "gate_uses_this": False, "threshold": None,
        "raw_loss_file": str(raw_path.relative_to(output)), "raw_loss_sha256": sha256(raw_path),
    }
    write_json(output / "decoder" / "validation_true_targets.json", report)
    guard.check("decoder_true_validation_targets_complete")
    return losses


def evaluate(config, store, basis, selected, decoder, reader, output, guard, metrics,
             true_positions, decoder_true_target_losses):
    import numpy as np
    import torch
    pairs = pair_groups(store.manifest["records"], "validation")
    if len(pairs) != 96:
        raise ValueError("Validation must contain exactly 96 pairs")
    feature_loss = np.empty((3, 96, 2, 4, 2), dtype=np.float64)
    position_loss = np.empty_like(feature_loss)
    collate = reader.load_reference("inputs").collate
    model_class = reader.load_reference("model").DynamicsPredictor
    store.training = False
    for si, seed in enumerate(SEEDS):
        guard.check("evaluation_model_load")
        model = model_class(5120, config["model"], [3, 6], basis).to("cuda:0")
        checkpoint = torch.load(selected[seed], map_location="cpu", weights_only=True)
        if checkpoint["seed"] != seed:
            raise RuntimeError("Selected checkpoint seed mismatch")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.inference_mode():
            for ci, condition in enumerate(EVAL_CONDITIONS):
                for start in range(0, len(pairs), 32):
                    guard.check("evaluation_batch")
                    rows = [rec for pair in pairs[start:start + 32] for rec in pair]
                    samples, targets = batch_samples(store, rows, condition)
                    prediction = model(collate(samples, torch.device("cuda:0"))).cpu().numpy()
                    if not np.isfinite(prediction).all():
                        raise RuntimeError("Non-finite evaluation prediction")
                    positions = np.asarray(decoder.predict(prediction.reshape(-1, 5120))).reshape(len(rows), 2, 2)
                    truth = np.stack([store.read_evaluation(rec, keys=("states",))["states"]
                                      [rec["query"] + np.asarray([3, 6])] for rec in rows])
                    fl = np.square(prediction.astype(np.float64) - targets.astype(np.float64)).mean(-1)
                    pl = np.square(positions.astype(np.float64) - truth.astype(np.float64)).mean(-1)
                    if not np.isfinite(fl).all() or not np.isfinite(pl).all():
                        raise RuntimeError("Non-finite feature/decoded-position metric")
                    count = len(rows) // 2
                    feature_loss[si, start:start + count, :, ci, :] = fl.reshape(count, 2, 2)
                    position_loss[si, start:start + count, :, ci, :] = pl.reshape(count, 2, 2)
        del model, checkpoint
        torch.cuda.empty_cache()
    losses = {"feature": feature_loss, "position": position_loss}
    write_npz(output / "validation_losses.npz", **losses,
              seed_values=np.asarray(SEEDS), pair_ids=np.asarray([pair[0]["pair"] for pair in pairs]),
              conditions=np.asarray(EVAL_CONDITIONS), horizons=np.asarray([3, 6]))
    guard.check("before_metrics")
    result = metrics.evaluate_metrics(losses, output_dir=output / "metrics", seed_values=SEEDS,
                                      pair_ids=[pair[0]["pair"] for pair in pairs],
                                      true_positions=true_positions,
                                      decoder_true_target_losses=decoder_true_target_losses)
    guard.check("after_metrics")
    return result


def execute(args):
    config, protocol = read_json(args.config), read_json(args.protocol)
    manifest, provenance = validate_registration(args, config, protocol)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output root must be new; no resume, replacement, or retry")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(config)
    config["encoder"] = dict(config["encoder"])
    config["encoder"]["cache_dir"] = str(output / "hf_cache")
    config["output"], config["device"] = str(output), "cuda"
    write_json(output / "resolved_config.json", config)
    write_json(output / "registration_verification.json", provenance)
    guard = None
    allocation = None
    state = {"status": "failed_before_gpu_verification", "step2_gate": "not_evaluated",
             "test_opened_or_generated": False, "retention_training_started": False,
             "auto_retry_or_resume": False, "protocol_sha256": provenance["protocol_sha256"]}
    previous_handlers = {}
    try:
        allocation = native_allocation()
        write_json(output / "allocation_start.json", allocation)
        guard = BudgetGuard(config, allocation, provenance["frames"], output)
        guard.check("allocation_start")
        def interrupted(signum, frame):
            guard.interrupted = f"Received signal {signum}; stopping without retry/resume"
            raise BudgetStop(guard.interrupted)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        remaining = min(7200, allocation["native_time_limit_seconds"]) - guard.elapsed()
        signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cache_environment = {
            "HF_HOME": str(output / "hf_home"), "HF_HUB_CACHE": str(output / "hf_cache"),
            "TORCH_HOME": str(output / "torch_home"), "XDG_CACHE_HOME": str(output / "cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(output / "torchinductor_cache"),
            "PYTORCH_KERNEL_CACHE_PATH": str(output / "torch_kernel_cache"),
        }
        os.environ.update(cache_environment)
        thread_variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        for name in thread_variables:
            os.environ[name] = "4"
        packages = {name: importlib.metadata.version(name) for name in (
            "numpy", "Pillow", "scikit-learn", "scipy", "torch", "transformers", "huggingface-hub")}
        expected_packages = protocol["runtime_package_versions"]
        def package_matches(name):
            # Official CUDA wheels may append their CUDA build as a local version.
            actual = packages[name].split("+", 1)[0] if name == "torch" else packages[name]
            return actual == expected_packages[name]
        if set(expected_packages) != set(packages) or not all(package_matches(key) for key in packages):
            write_json(output / "runtime_dependency_mismatch.json", {"expected": expected_packages, "actual": packages})
            raise RuntimeError("GPU runtime package versions differ from the committed protocol")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("The committed GPU runtime requires Python 3.12")
        import numpy as np
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Exactly one visible native NVIDIA A100 is required")
        initial_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        def visible_gpu_probe():
            if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
                    "A100" not in torch.cuda.get_device_name(0) or
                    os.environ.get("CUDA_VISIBLE_DEVICES") != initial_visible_devices):
                raise RuntimeError("The scientific job no longer exposes exactly its original single A100")
            return {"device_count": 1, "gpu_name": torch.cuda.get_device_name(0),
                    "cuda_visible_devices": initial_visible_devices,
                    "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"])}
        guard.visible_gpu_probe = visible_gpu_probe
        guard.check("before_scientific_work", force_native=True)
        torch.set_num_threads(config["threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        smi = command(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"])
        write_json(output / "runtime_environment.json", {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "package_versions": packages,
            "cuda_runtime": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
            "nvidia_smi": smi, "hostname": command(["hostname"]), "slurm_job_id": allocation["job_id"],
            "float32_tf32_disabled": True, "torch_threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "thread_environment": {name: os.environ[name] for name in thread_variables},
            "cache_environment": cache_environment,
        })
        reader, metrics = local_module("reader_v2"), local_module("step2_metrics")
        bound_helper = local_module("prepare_direction_bound_v2_3")
        true_positions, _direction_bound, bound_receipt = bound_helper.verify_prepared_bound(
            args.direction_bound, args.dataset, manifest, args.protocol, metrics)
        prepared_at = datetime.fromisoformat(bound_receipt["prepared_finished_utc"])
        allocation_started_at = datetime.fromisoformat(allocation["native_start_time"]).astimezone(timezone.utc)
        if prepared_at.tzinfo is None or prepared_at > allocation_started_at:
            raise ValueError("The validation direction bound must be prepared on CPU before GPU allocation starts")
        write_json(output / "direction_bound_verification.json", bound_receipt)
        guard.check("after_direction_bound_verification", force_native=True)
        encode_all(config, manifest, args.dataset, output, reader, guard)
        store = reader.V2FeatureStore(config, args.dataset, output / "features", manifest)
        store.training = False
        store.train_dropout = 0.25
        model_module = reader.load_reference("model")
        # Keep the original basis computation but use exclusive new-file writers;
        # there are no temporary-file deletions or replace operations in this run.
        model_module.write_npz, model_module.write_json = write_npz, write_json
        guard.check("before_train_only_basis")
        basis = model_module.fit_basis(config, store, "motion")
        guard.check("after_train_only_basis")
        store.bases = {"motion": basis}
        store.training = False
        for rec in manifest["records"]:
            guard.check("input_cache_preparation")
            store.sample(rec, "b")
        write_json(output / "input_cache_preparation.json", {
            "status": "completed", "episodes": 1216, "split_counts": {"train": 1024, "validation": 192},
            "condition": "b", "memory_dropout": False, "model_forward": False,
            "optimization_or_decoder_fit": False, "charged_phase": "overhead",
        })
        guard.check("before_control_input_identity_audit")
        identity = reader.assert_a_c_d_inputs_identical(store, pair_groups(manifest["records"], "validation"))
        write_json(output / "control_input_identity_audit.json", {"status": "passed", "details": identity,
                   "split": "validation", "pairs": 96, "conditions": ["a", "c", "d"]})
        guard.check("after_control_input_identity_audit")
        guard.check("before_train_only_decoder")
        decoder = metrics.fit_train_decoder(store, config, output / "decoder", guard.check)
        guard.check("after_train_only_decoder")
        decoder_true_target_losses = validate_decoder_on_true_targets(store, decoder, output, guard)
        guard.check("before_predictor_training", force_native=True)
        selected = train_predictors(config, store, basis, reader, output, guard, provenance)
        result = evaluate(config, store, basis, selected, decoder, reader, output, guard, metrics,
                          true_positions, decoder_true_target_losses)
        guard.check("scientific_work_complete", force_native=True)
        state.update(status="completed_step2_stop", step2_gate="passed" if result["gate"]["passed"] else "failed",
                     stop_after_step2=True, selected_checkpoints={str(k): str(v.relative_to(output)) for k, v in selected.items()})
    except Exception as exc:
        state.update(status="stopped_budget" if isinstance(exc, BudgetStop) else "stopped_execution_error",
                     error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if previous_handlers:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
        if guard is not None:
            state["budget"] = guard.snapshot()
            state["actual_a100_gpu_seconds_so_far"] = guard.elapsed()
        else:
            state["actual_a100_gpu_seconds_so_far"] = allocation["elapsed_seconds_at_probe"] if allocation else None
        state["allocation_final_settlement_required"] = allocation is not None
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", state)
        hashes = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
        write_json(output / "artifact_sha256.json", hashes)
    print(json.dumps(state, indent=2, allow_nan=False), flush=True)
    return 0 if state["status"] == "completed_step2_stop" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--direction-bound", required=True, type=Path,
                        help="Fresh CPU preparation directory produced by prepare_direction_bound_v2_3.py")
    parser.add_argument("--output", required=True, type=Path)
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
