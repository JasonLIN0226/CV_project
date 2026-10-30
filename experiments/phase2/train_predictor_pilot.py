"""Fresh CPU-only DynamicsPredictor pilot on verified new I-JEPA features.

Uses the unchanged supplied predictor architecture, train-only bases and the new
causal subset/dropout sampler. This is not the unavailable repaired Phase 1 model.
No test metrics, retention scorer, deletion labels, or planning are produced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from phase1.utils import write_json
    write_json(path, value)


def verify_predictor_contract(config):
    """Reject incompatible sampling settings or changed retained predictor code."""
    data = config["data"]
    if (data.get("recent") != 4 or data.get("budget") != 2
            or data.get("horizons") not in ([3, 6], (3, 6))):
        raise ValueError("This predictor pilot requires recent=4, budget=2, horizons=[3, 6]")
    control = ROOT / "experiments/predictor/control"
    lock_path = control / "source_lock.json"
    lock = read(lock_path)["files"]
    hashes = {}
    for name in ("model.py", "inputs.py", "training.py"):
        relative = "src/phase1/" + name
        actual = sha256(control / relative)
        if lock.get(relative) != actual:
            raise ValueError("Retained Phase 1 predictor source changed: " + name)
        hashes[relative] = actual
    return {"source_lock_sha256": sha256(lock_path), "files": hashes,
            "recent": 4, "budget": 2, "horizons": [3, 6]}


def verify_inputs(data_root, encoded_report):
    from experiments.phase2.prepare_encoded_samples import (
        OFFICIAL_REPOSITORY, OFFICIAL_REVISION, verify_prepared_inputs, verify_retained_source)
    from phase1.cache import load_index, load_normalization
    progress = read(encoded_report / "manifest.json")
    if progress.get("status") != "official_encoding_and_sample_audit_completed":
        raise ValueError("Require a completed real I-JEPA encoding/sample audit")
    for filename, key in (("encoder_config.json", "encoder_config_sha256"),
                          ("sample_summary.json", "sample_summary_sha256")):
        if sha256(encoded_report / filename) != progress.get(key):
            raise ValueError("Encoded report artifact hash mismatch: " + filename)
    config = read(encoded_report / "encoder_config.json")
    if Path(config["output"]).resolve() != data_root:
        raise ValueError("Encoded report and explicitly supplied data root differ")
    expected = {"backend": "ijepa", "repository": OFFICIAL_REPOSITORY,
                "revision": OFFICIAL_REVISION, "dtype": "float32", "pooling": "grid2"}
    if any(config["encoder"].get(k) != v for k, v in expected.items()):
        raise ValueError("Require the actual pinned official float32/grid2 I-JEPA cache")
    if config["data"]["seed"] in (6301, 9137):
        raise ValueError("This pilot only uses new development data, not inherited/reserved seeds")
    verify_retained_source()
    verify_predictor_contract(config)
    _, index, roles = verify_prepared_inputs(data_root, config)
    cache = load_index(config)
    load_normalization(config)
    summary = read(encoded_report / "sample_summary.json")
    if (sha256(data_root / "features/index.json") != progress["feature_index_sha256"]
            or sha256(data_root / "features/normalization.npz") != summary["normalization_sha256"]
            or not summary.get("normalization_train_only_verified")
            or not summary.get("normalization_moments_recomputed_exactly")
            or cache["encoder"].get("name") != OFFICIAL_REPOSITORY
            or cache["encoder"].get("resolved_revision") != OFFICIAL_REVISION):
        raise ValueError("Cache or train-only normalization provenance differs")
    return config, index, roles, cache


def train_task(config, store, task, identities, roles, output, *, epochs, seed, dropout, batch_size):
    """CPU training kernel; tests may supply explicitly labeled toy features."""
    import numpy as np
    import torch
    from experiments.phase2.training_samples import (
        DEFAULT_MODES, EpisodeIdentity, FeatureEpisode, SamplingConfig, SplitRegistry,
        phase1_arrays, sample_predictor)
    from phase1.inputs import collate
    from phase1.model import DynamicsPredictor, fit_basis, project
    from phase1.training import paired_rows, regression_loss
    from phase1.utils import rng_state, save_torch, seed_all

    predictor_contract = verify_predictor_contract(config)
    seed_all(seed, config["threads"])
    records = [r for r in identities if r["task"] == task and r["split"] in ("train", "validation")]
    pairs = {split: paired_rows([r for r in records if r["split"] == split]) for split in ("train", "validation")}
    if not all(pairs.values()):
        raise ValueError("Both train and validation sibling groups are required")
    registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in identities], split_roles=roles)
    sampling = SamplingConfig(memory_dropout=dropout)
    basis = fit_basis(config, store, task)
    expected_ids = {r["id"] for r in records if r["split"] == "train"}
    if set(basis["train_ids"].tolist()) != expected_ids:
        raise ValueError("Basis must contain exactly this task's training episodes")
    feature_id = store.ix["signature"] + ":normalization:" + sha256(Path(config["output"]) / "features/normalization.npz")
    episodes = {}
    for rec in records:
        with np.load(Path(config["output"]) / "dataset" / rec["obs"], allow_pickle=False) as archive:
            actions = archive["actions"]
        episodes[rec["id"]] = FeatureEpisode(EpisodeIdentity.from_record(rec), store.features(rec),
                                             actions, "additive_controls", feature_id)
    model = DynamicsPredictor(len(store.norm["mean"]), config["model"], (3, 6), basis).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                                 weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["training"]["lr_step"], config["training"]["lr_gamma"])
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    root = output / "predictor" / task / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    signature = {"architecture": "phase1.model.DynamicsPredictor", "new_random_initialization": True,
                 "task": task, "seed": seed, "cache_signature": store.ix["signature"],
                 "normalization_sha256": sha256(Path(config["output"]) / "features/normalization.npz"),
                 "basis_sha256": sha256(output / "predictor" / task / "basis.npz"),
                 "sampling_modes": list(DEFAULT_MODES), "memory_dropout": dropout,
                 "validation_selection": "earliest_minimum_equal_mode_mean_feature_mse",
                 "model_config": config["model"], "horizons": [3, 6], "training": config["training"],
                 "training_episode_ids": sorted(expected_ids),
                 "validation_episode_ids": sorted(r["id"] for r in records if r["split"] == "validation")}
    logs, step, best, best_epoch = [], 0, float("inf"), None
    counters = {mode: {"examples": 0, "dropout_events": 0, "removed_entries": 0} for mode in DEFAULT_MODES}

    def batch(rows, mode, epoch, split):
        arrays, targets = [], []
        for rec in rows:
            sample = sample_predictor(episodes[rec["id"]], registry, sampling, query=rec["query"],
                                     base_seed=seed, epoch=epoch if split == "train" else 0,
                                     expected_split=split, history_mode=mode)
            a, target = phase1_arrays(sample)
            a["projected_z"], a["projected_current"] = project(a["z"], basis), project(a["current"], basis)
            arrays.append(a)
            targets.append(target)
            if split == "train":
                counters[mode]["examples"] += 1
                counters[mode]["dropout_events"] += int(sample.metadata["dropout_event"])
                counters[mode]["removed_entries"] += sample.metadata["dropout_removed_entries"]
            elif sample.metadata["dropout_event"]:
                raise ValueError("Validation applied memory dropout")
        return collate(arrays, "cpu"), torch.from_numpy(np.stack(targets))

    def validate():
        model.eval()
        results = {}
        with torch.inference_mode():
            for mode in DEFAULT_MODES:
                total, count = 0., 0
                for start in range(0, len(pairs["validation"]), batch_size // 2):
                    rows = [r for pair in pairs["validation"][start:start + batch_size // 2] for r in pair]
                    inputs, targets = batch(rows, mode, 0, "validation")
                    value = (model(inputs) - targets).square().mean()
                    if not torch.isfinite(value):
                        raise ValueError("Nonfinite validation MSE")
                    total += float(value) * len(rows)
                    count += len(rows)
                results[mode] = total / count
        return results

    initial_validation = validate()
    began = time.monotonic()
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + 100003 * epoch)
        order = rng.permutation(len(pairs["train"]))
        losses = []
        for start in range(0, len(order), batch_size // 2):
            rows = [r for i in order[start:start + batch_size // 2] for r in pairs["train"][i]]
            for mode in rng.permutation(DEFAULT_MODES):
                inputs, targets = batch(rows, str(mode), epoch, "train")
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss, mse, _ = regression_loss(prediction, targets, config["training"]["pair_weight"], config["training"]["rank_weight"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                losses.append((float(loss.detach()), float(mse.detach()), len(rows)))
        validation = validate()
        score = float(np.mean(list(validation.values())))
        improved = score < best
        if improved:
            best, best_epoch = score, epoch + 1
        scheduler.step()
        logs.append({"epoch": epoch + 1, "optimizer_steps": step,
                     "train_loss": sum(loss * n for loss, _, n in losses) / sum(n for _, _, n in losses),
                     "train_mse": sum(mse * n for _, mse, n in losses) / sum(n for _, _, n in losses),
                     "validation_mse": score, "validation_by_mode": validation,
                     "best_epoch": best_epoch, "elapsed_seconds": time.monotonic() - began})
        checkpoint = {"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "rng": rng_state(), "epoch": epoch + 1,
                      "step": step, "best_validation_mse": best, "best_epoch": best_epoch, "logs": logs,
                      "checkpoint_kind": "new_development_predictor_training", "feature_dimension": len(store.norm["mean"])}
        save_torch(root / "latest.pt", checkpoint)
        if improved:
            save_torch(root / "best.pt", checkpoint)
        write(root / "training.json", {"signature": signature, "initial_validation_by_mode": initial_validation,
                                       "logs": logs, "sampling_counts": counters})
        print(json.dumps({"task": task, **logs[-1]}), flush=True)
    changed = any(not torch.equal(initial_parameters[name], value) for name, value in model.named_parameters())
    if not changed:
        raise ValueError("No predictor parameters changed during training")
    # This file was created above by this process, not an external checkpoint.
    selected = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    model.eval().requires_grad_(False)
    frozen_score = float(np.mean(list(validate().values())))
    if frozen_score != best or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Selected frozen checkpoint failed validation replay")
    save_torch(root / "frozen.pt", {"signature": signature, "model": model.state_dict(),
               "model_config": config["model"], "feature_dimension": len(store.norm["mean"]),
               "horizons": (3, 6), "selected_epoch": best_epoch, "validation_mse": best,
               "checkpoint_kind": "new_frozen_predictor_for_future_phase2_use", "frozen_verified": True})
    return {"task": task, "seed": seed, "epochs": epochs, "optimizer_steps": step,
            "parameter_count": sum(p.numel() for p in model.parameters()), "parameters_changed": changed,
            "input_rank": basis["projection"].shape[1], "output_rank": basis["output_basis"].shape[1],
            "best_epoch": best_epoch, "best_validation_mse": best, "frozen_validation_replay_exact": True,
            "retained_predictor_contract": predictor_contract,
            "basis_train_only": True, "sampling_counts": counters, "test_examples_or_metrics": 0,
            "frozen_checkpoint": str(root / "frozen.pt"), "frozen_checkpoint_sha256": sha256(root / "frozen.pt")}


def worker(args):
    from phase1.inputs import FeatureStore
    import torch
    data_root, report, output = args.data_root.resolve(), args.encoded_report.resolve(), args.output.resolve()
    raw, index, roles, cache = verify_inputs(data_root, report)
    c = copy.deepcopy(raw)
    c.update(output=str(output), device="cpu", threads=args.threads)
    c.pop("experiment", None)
    c["training"].update(epochs=args.epochs, seeds=[args.seed], batch_size=args.batch_size)
    for name in ("dataset", "features"):
        (output / name).symlink_to(data_root / name, target_is_directory=True)
    write(output / "training_config.json", c)
    preserved = {str(path): sha256(path) for path in (
        data_root / "resolved_config.json", data_root / "dataset/manifest.json",
        data_root / "features/index.json", data_root / "features/normalization.npz")}
    store = FeatureStore(c, max_cached=len(index["records"]))
    torch.set_num_threads(args.threads)
    result = {"status": "training", "device": "cpu", "new_initialization": True,
              "source_data_root": str(data_root), "encoded_report": str(report),
              "cache_signature": cache["signature"], "tasks": [], "test_metrics_generated": False,
              "latest_repaired_phase1_reproduced": False, "scientific_phase2_completed": False,
              "deletion_supervision_generated": False, "baseline_model_planning_verified": False}
    write(output / "training_result.json", result)
    for task in c["data"]["tasks"]:
        result["tasks"].append(train_task(c, store, task, index["records"], roles, output,
             epochs=args.epochs, seed=args.seed, dropout=args.dropout, batch_size=args.batch_size))
        write(output / "training_result.json", result)
    if any(sha256(path) != value for path, value in preserved.items()):
        raise ValueError("Original prepared-data/cache artifacts changed")
    result.update(status="new_cpu_predictor_pilot_completed", originals_preserved=True,
                  original_artifact_sha256=preserved, predictor_trained=True,
                  trained_retention_baseline_results=False, learned_retention_scorer_trained=False)
    write(output / "training_result.json", result)


def supervise(args):
    from experiments.phase2.run_hpc_pipeline import terminate_group
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_paths = [Path(__file__), ROOT / "experiments/phase2/training_samples.py",
                    ROOT / "experiments/phase2/memory.py", ROOT / "experiments/phase2/prepare_encoded_samples.py"]
    source_paths += [ROOT / "experiments/predictor/control/src/phase1" / name for name in
                     ("model.py", "training.py", "inputs.py", "cache.py", "config.py")]
    manifest = {"status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "new_cpu_predictor_path_pilot_not_repaired_phase1_reproduction",
                "timeout_seconds": args.timeout_seconds, "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                "device": "cpu", "gpu_allocation_requested": False, "scientific_phase2_completed": False,
                "settings": vars(args).copy()}
    manifest["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in manifest["settings"].items()}
    try:
        manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        manifest["git_head"] = None
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--data-root", str(args.data_root.resolve()),
               "--encoded-report", str(args.encoded_report.resolve()), "--output", str(output),
               "--epochs", str(args.epochs), "--seed", str(args.seed), "--dropout", str(args.dropout),
               "--batch-size", str(args.batch_size), "--threads", str(args.threads), "--timeout-seconds", str(args.timeout_seconds)]
    manifest["command"] = command
    write(output / "run_manifest.json", manifest)
    process = None
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(str(signum))
    for sig in handlers:
        signal.signal(sig, interrupt)
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(args.threads),
               "MKL_NUM_THREADS": str(args.threads), "PYTHONUNBUFFERED": "1"}
        with (output / "worker.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=max(1., args.timeout_seconds - (time.monotonic() - started)))
        manifest["worker_exit_code"] = code
        if code != 0:
            raise RuntimeError("CPU predictor worker failed; inspect worker.log")
        result = read(output / "training_result.json")
        if result.get("status") != "new_cpu_predictor_pilot_completed":
            raise ValueError("Worker did not complete the new predictor pilot")
        manifest.update(status="completed", predictor_trained=True,
                        training_result_sha256=sha256(output / "training_result.json"))
    except subprocess.TimeoutExpired:
        manifest.update(status="timed_out", error="20-minute-or-smaller CPU wall cap reached; completed-epoch snapshots preserved")
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error=traceback.format_exc())
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
    finally:
        terminate_group(process)
        manifest.update(ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started,
                        worker_exit_code=process.returncode if process else None,
                        worker_log_sha256=sha256(output / "worker.log") if (output / "worker.log").exists() else None)
        write(output / "run_manifest.json", manifest)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({"status": manifest["status"], "output": str(output), "scientific_phase2_completed": False}))
    return 0 if manifest["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--encoded-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.epochs <= 20 and args.seed >= 0 and 0 <= args.dropout <= 1 and
            args.batch_size >= 2 and args.batch_size % 2 == 0 and 1 <= args.threads <= 4 and
            1 <= args.timeout_seconds <= 1200):
        parser.error("Require epochs 1..20, nonnegative seed, dropout [0,1], even batch >=2, threads 1..4, timeout 1..1200")
    if args.worker:
        worker(args)
    else:
        raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()
