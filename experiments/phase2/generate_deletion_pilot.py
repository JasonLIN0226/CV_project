"""Bounded CPU deletion-label pilot on a new, actually frozen predictor.

Replay four independent policies from episode start to the predeclared query.
Capture the genuine K+1 overflow before its deletion, enumerate equal-K forks,
and save causal scorer arrays separately from future-derived labels. This is
not a six-policy baseline comparison or retention-scorer training. The four
parent policies and one-query sampling rule are explicit engineering choices.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import MappingProxyType

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.phase2.deletion import (
    CausalContext, CausalParent, OfflineFuture, generate_deletion_supervision)
from experiments.phase2.memory import MemoryEntry, StreamingMemory
from experiments.phase2.training_samples import EpisodeIdentity, SplitRegistry

PARENT_POLICIES = ("fifo", "reservoir", "temporal_buckets", "visual_diversity")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def immutable(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def policy_seed(base_seed, identity, policy):
    # Siblings share RNG. No query, target, future action or evidence dependency.
    key = [int(base_seed), identity.task, identity.group_id, policy]
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8], "little")


@dataclass(frozen=True)
class ParentCapture:
    parent: CausalParent
    past_actions: object
    accounting: dict


def capture_query_parent(episode_id, features, actions, query, *, policy, seed):
    """Offline reader; the policy sees only each arriving observation.

    Birth action-prefix values exist only for resident entries. A global
    two-float cumulative prefix plus per-resident prefixes exactly reproduce
    the inherited additive-action convention without retaining an action or
    evicted-observation archive. Targets and a_query onward are never read.
    """
    features, actions = np.asarray(features), np.asarray(actions)
    if (features.ndim != 2 or features.dtype != np.float32 or not features.shape[1]
            or actions.shape != (len(features) - 1, 2) or actions.dtype != np.float32):
        raise ValueError("Require float32 [T,D] features and [T-1,2] additive actions")
    if isinstance(query, bool) or not isinstance(query, (int, np.integer)) or not 6 <= query < len(features):
        raise ValueError("Query must permit a full R4/K2 overflow")
    if policy not in PARENT_POLICIES:
        raise ValueError("Only the four declared independent parent policies are allowed")
    stream = StreamingMemory(4, 2, policy, seed=seed)
    prefix = np.zeros(2, np.float32)
    births = {}
    maximum_resident = maximum_transient = 0
    began = time.perf_counter()
    capture = None
    for t in range(int(query) + 1):
        if t:
            if not np.isfinite(actions[t - 1]).all():
                raise ValueError("Nonfinite executed action")
            prefix += actions[t - 1]
        observation = MemoryEntry(episode_id, t, features[t])
        before = stream.snapshot
        births[observation.entry_id] = prefix.copy()
        maximum_transient = max(maximum_transient, len(births))
        if t == query:
            candidates = before.retained + (before.recent[0],)
            recent = before.recent[1:] + (observation,)
            parent = CausalParent(2, candidates, CausalContext(t, recent))
            summaries = MappingProxyType({entry.entry_id: immutable(prefix - births[entry.entry_id])
                                         for entry in candidates + recent})
            capture = (parent, summaries)
        snapshot = stream.push(observation)
        residents = snapshot.recent + snapshot.retained
        births = {entry.entry_id: births[entry.entry_id] for entry in residents}
        maximum_resident = max(maximum_resident, len(births))
        if len(births) > 6 or maximum_transient > 7:
            raise ValueError("Action metadata exceeded resident/transient memory bounds")
    if capture is None:
        raise ValueError("No full query overflow was captured")
    accounting = stream.accounting()
    accounting.update(persistent_action_prefix_entries_max=maximum_resident,
                      transient_action_prefix_entries_max=maximum_transient,
                      action_prefix_numeric_bytes_max=(maximum_transient + 1) * 2 * 4,
                      policy_stream_seconds=time.perf_counter() - began,
                      offline_label_buffers_excluded=True)
    return ParentCapture(*capture, accounting)


def parent_id(parent, task, checkpoint_hash):
    key = [task, checkpoint_hash, parent.context.current_time,
           [list(e.entry_id) for e in parent.candidates]]
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def label_parent(capture, features, actions, evaluator):
    q = capture.parent.context.current_time
    horizons = (3, 6)
    available = np.array([q + h < len(features) for h in horizons], bool)
    targets = np.full((2, features.shape[1]), np.nan, np.float32)
    for i, h in enumerate(horizons):
        if available[i]:
            targets[i] = features[q + h]
    future = OfflineFuture(horizons, targets, immutable(actions[q:q + 6]), available)
    callback = evaluator.bind(capture.parent, capture.past_actions)
    return generate_deletion_supervision(capture.parent, future, callback,
                                         provenance=evaluator.provenance)


def save_shard(output, task, split, rows):
    """Writer is an offline archive; neither memory nor evaluator can read it."""
    if not rows:
        return None
    inputs_path = output / "scorer_inputs" / f"{task}_{split}.npz"
    labels_path = output / "labels" / f"{task}_{split}.npz"
    metadata_path = output / "metadata" / f"{task}_{split}.json"
    for path in (inputs_path, labels_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
    parents = np.array([r["metadata"]["parent_id"] for r in rows])
    scorer = [r["supervision"].scorer_inputs for r in rows]
    labels = [r["supervision"].labels for r in rows]
    np.savez_compressed(inputs_path, parent_id=parents,
        current_time=np.array([r.current_time for r in scorer], np.int64),
        candidate_timestamps=np.stack([r.candidate_timestamps for r in scorer]),
        candidate_representations=np.stack([r.candidate_representations for r in scorer]),
        recent_timestamps=np.stack([r.recent_timestamps for r in scorer]),
        recent_representations=np.stack([r.recent_representations for r in scorer]),
        candidate_past_actions=np.stack([r["candidate_past_actions"] for r in rows]),
        recent_past_actions=np.stack([r["recent_past_actions"] for r in rows]))
    np.savez_compressed(labels_path, parent_id=parents,
        horizons=np.array((3, 6), np.int64),
        horizon_mask=np.stack([r.horizon_mask for r in labels]),
        feature_mse=np.stack([r.feature_mse for r in labels]),
        deleted_timestamps=np.array([[i[1] for i in r.deleted_ids] for r in labels], np.int64),
        kept_timestamps=np.array([[[i[1] for i in fork] for fork in r.kept_ids] for r in labels], np.int64))
    write(metadata_path, [r["metadata"] for r in rows])
    return {"task": task, "split": split, "unique_parents": len(rows),
            "deletion_forks": len(rows) * 3,
            "available_loss_values": sum(int(r.horizon_mask.sum()) * 3 for r in labels),
            "files": {str(p.relative_to(output)): sha256(p) for p in (inputs_path, labels_path, metadata_path)}}


def generate_task(records, registry, load_episode, evaluator, task, output, *, base_seed):
    """Only declared train/validation rows are passed to load_episode.

    All state-generation policies replay one persistent stream per episode.
    Identical parents reached by multiple policies are saved once with all
    provenance origins; their repeated forks are not independent examples.
    """
    shards = []
    total_attempted = total_unique = total_recomputed = 0
    exclusions = []
    for split in ("train", "validation"):
        rows, seen = [], {}
        selected = [r for r in records if r["task"] == task and r["split"] == split]
        for rec in selected:
            identity = EpisodeIdentity.from_record(rec)
            registry.require(identity, split)
            features, actions = load_episode(rec)
            q = int(rec["query"])
            if q < 6:
                exclusions.append({"episode_id": identity.episode_id, "reason": "no_full_query_overflow"})
                continue
            if q + 3 >= len(features):
                exclusions.append({"episode_id": identity.episode_id, "reason": "all_horizons_missing"})
                continue
            for policy in PARENT_POLICIES:
                seed = policy_seed(base_seed, identity, policy)
                capture = capture_query_parent(identity.episode_id, features, actions, q, policy=policy, seed=seed)
                total_attempted += 1
                uid = parent_id(capture.parent, task, evaluator.provenance.frozen_predictor_sha256)
                origin = {"policy": policy, "seed": seed, "rule": "continuous_prefix_replay_then_declared_query_only",
                          "selection_probability": 1., "accounting": capture.accounting}
                if uid in seen:
                    rows[seen[uid]]["metadata"]["parent_origins"].append(origin)
                    continue
                supervision = label_parent(capture, features, actions, evaluator)
                labels = supervision.labels
                if (labels.feature_mse.shape != (3, 2)
                        or not np.isfinite(labels.feature_mse[:, labels.horizon_mask]).all()
                        or any(len(ids) != 2 for ids in labels.kept_ids)):
                    raise ValueError("Invalid equal-K deletion labels")
                # Deterministic fixed subset of unique parents, never loss-selected.
                if len(rows) % 8 == 0:
                    repeated = label_parent(capture, features, actions, evaluator)
                    if not np.array_equal(labels.feature_mse, repeated.labels.feature_mse, equal_nan=True):
                        raise ValueError("Frozen deletion loss recomputation changed")
                    total_recomputed += 1
                parent = capture.parent
                metadata = {"parent_id": uid, "episode_id": identity.episode_id,
                    "group_id": identity.group_id, "split": split, "split_role": registry.roles[split],
                    "task": task, "environment_seed": identity.environment_seed, "query": q,
                    "candidate_ids": [list(e.entry_id) for e in parent.candidates],
                    "recent_ids": [list(e.entry_id) for e in parent.context.recent],
                    "parent_origins": [origin], "checkpoint_sha256": evaluator.provenance.frozen_predictor_sha256}
                seen[uid] = len(rows)
                rows.append({"metadata": metadata, "supervision": supervision,
                    "candidate_past_actions": np.stack([capture.past_actions[e.entry_id] for e in parent.candidates]),
                    "recent_past_actions": np.stack([capture.past_actions[e.entry_id] for e in parent.context.recent])})
        shard = save_shard(output, task, split, rows)
        if shard:
            shards.append(shard)
            total_unique += len(rows)
    return {"task": task, "parent_policy_records": total_attempted, "unique_parents": total_unique,
            "recomputed_parents": total_recomputed, "exclusions": exclusions, "shards": shards,
            "frozen_evaluator_audit": evaluator.audit()}


def worker(args):
    from experiments.phase2.train_predictor_pilot import verify_inputs
    from experiments.phase2.frozen_predictor import FrozenPredictorEvaluator
    from phase1.cache import load_feature, load_normalization, normalize
    import torch
    data_root, encoded_report = args.data_root.resolve(), args.encoded_report.resolve()
    output = args.output.resolve()
    config, index, roles, cache = verify_inputs(data_root, encoded_report)
    registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in index["records"]], split_roles=roles)
    norm = load_normalization(config)
    torch.set_num_threads(args.threads)
    loaded = []
    def load_episode(rec):
        if rec["split"] not in ("train", "validation"):
            raise ValueError("Test-feature access is forbidden in this label pilot")
        raw = load_feature(config, rec, cache)
        with np.load(data_root / "dataset" / rec["obs"], allow_pickle=False) as archive:
            if set(archive.files) != {"rgb", "actions", "query"}:
                raise ValueError("Observation archive schema changed")
            actions = archive["actions"].copy()
            if int(archive["query"]) != rec["query"]:
                raise ValueError("Query alignment mismatch")
        loaded.append(rec["id"])
        return normalize(raw, norm), actions
    summary = {"schema": "phase2_equal_k_deletion_pilot_v1", "status": "generating",
        "data_seed": config["data"]["seed"], "device": "cpu", "base_seed": args.seed,
        "capacity": 2, "recent_length": 4, "horizons": [3, 6], "parent_policies": list(PARENT_POLICIES),
        "parent_sampling_rule": "each_train_validation_episode_declared_query_after_continuous_prefix_replay",
        "deduplication": "same_task_checkpoint_episode_query_candidate_ids_saved_once_with_all_policy_origins",
        "loss": {"name": "raw_full_normalized_feature_MSE", "reduction": "mean_over_feature_dimensions",
                 "dtype": "float64_residual_reduction_from_float32_predictions_targets", "horizon_aggregation": None,
                 "ties": "raw_values_preserved_no_margin_or_ranking_threshold"},
        "cache_signature": cache["signature"], "encoder_hash": cache["encoder_hash"],
        "normalization_sha256": sha256(data_root / "features/normalization.npz"),
        "encoded_manifest_sha256": sha256(encoded_report / "manifest.json"),
        "split_roles": roles, "tasks": [], "six_retention_baseline_results": False,
        "learned_retention_scorer_trained": False, "baseline_model_planning_verified": False,
        "scientific_phase2_completed": False}
    write(output / "summary.json", summary)
    for task in ("gate", "motion"):
        evaluator = FrozenPredictorEvaluator(predictor_root=args.predictor_root.resolve(),
                    encoded_report=encoded_report, data_root=data_root, task=task, seed=args.seed)
        summary["tasks"].append(generate_task(index["records"], registry, load_episode, evaluator,
                                             task, output, base_seed=args.seed))
        write(output / "summary.json", summary)
    expected = {r["id"] for r in index["records"] if r["split"] in ("train", "validation")}
    if set(loaded) != expected or len(loaded) != len(expected):
        raise ValueError("Episode loading membership mismatch")
    summary.update(status="new_development_equal_k_deletion_pilot_completed", loaded_episode_ids=sorted(loaded),
                   test_feature_arrays_loaded=0, loss_labels_generated=True,
                   unique_parents=sum(t["unique_parents"] for t in summary["tasks"]),
                   parent_policy_records=sum(t["parent_policy_records"] for t in summary["tasks"]),
                   recomputed_parents=sum(t["recomputed_parents"] for t in summary["tasks"]))
    write(output / "summary.json", summary)


def supervise(args):
    from experiments.phase2.run_hpc_pipeline import terminate_group
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_names = ["generate_deletion_pilot.py", "frozen_predictor.py", "deletion.py", "memory.py", "training_samples.py"]
    manifest = {"status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "new_development_frozen_equal_K_deletion_pilot", "device": "cpu",
        "source_sha256": {"experiments/phase2/" + name: sha256(Path(__file__).parent / name) for name in source_names},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu_job_submitted": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_allocation_seconds": None if os.environ.get("SLURM_JOB_ID") else 0,
        "resource_note": "CPU worker; use scheduler accounting for any enclosing allocation.",
        "scientific_phase2_completed": False}
    manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--data-root", str(args.data_root.resolve()),
        "--encoded-report", str(args.encoded_report.resolve()), "--predictor-root", str(args.predictor_root.resolve()),
        "--output", str(output), "--seed", str(args.seed), "--threads", str(args.threads),
        "--timeout-seconds", str(args.timeout_seconds)]
    manifest["command"] = command
    write(output / "run_manifest.json", manifest)
    process = None
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(str(signum))
    for s in handlers:
        signal.signal(s, interrupt)
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(args.threads),
               "MKL_NUM_THREADS": str(args.threads), "PYTHONUNBUFFERED": "1"}
        with (output / "worker.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=max(1., args.timeout_seconds - (time.monotonic() - started)))
        if code:
            raise RuntimeError("Deletion worker failed; see worker.log")
        summary = json.loads((output / "summary.json").read_text())
        if summary.get("status") != "new_development_equal_k_deletion_pilot_completed":
            raise ValueError("Incomplete deletion output")
        manifest.update(status="completed", summary_sha256=sha256(output / "summary.json"),
                        loss_labels_generated=True)
    except subprocess.TimeoutExpired:
        manifest.update(status="timed_out", error="Bounded CPU pilot time limit reached; partial shards preserved")
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
        for s, handler in handlers.items():
            signal.signal(s, handler)
    print(json.dumps({"status": manifest["status"], "output": str(output), "scientific_phase2_completed": False}))
    return 0 if manifest["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "encoded-report", "predictor-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (args.seed >= 0 and 1 <= args.threads <= 4 and 1 <= args.timeout_seconds <= 1200):
        parser.error("Require nonnegative seed, 1..4 threads and 1..1200 seconds")
    if args.worker:
        worker(args)
    else:
        raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()
