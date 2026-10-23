"""Six real retention policies on one frozen predictor per task, validation only.

Surprise is a causal h=3 realized-action prediction error. Its bounded pending
context queue is measured explicitly and makes auxiliary memory unequal across
policies. This is a development prediction pilot, not an equal-byte comparison,
retention-scorer training, predictive-quality gate, or planning result.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.phase2.memory import MemoryEntry, StreamingMemory, POLICIES
from experiments.phase2.training_samples import EpisodeIdentity, SplitRegistry
from experiments.phase2.generate_deletion_pilot import sha256, write, immutable, policy_seed


@dataclass
class Pending:
    context: object
    actions: list


class OnlineRetention:
    """Receives one current feature and at most one just-executed action.

    No full episode, future action/target, output trace or evicted ID archive is
    reachable through the online object. Only R+K entry birth-prefix values and
    at most three model context summaries are retained. The old current feature
    needed to finish a 3-step prediction is still in the R=4 recent buffer.
    """

    def __init__(self, episode_id, policy, scores, *, seed):
        self.episode_id, self.policy, self.scores = episode_id, policy, scores
        self.stream = StreamingMemory(4, 2, policy, seed=seed,
            score_callback=scores.attention if policy == "attention" else None)
        self.prefix = np.zeros(2, np.float32)
        self.births = {}
        self.pending = deque()
        self.next_time = 0
        self.surprise_predictions = 0
        self.maxima = {}
        self.seconds = 0.

    def past_actions(self, entries):
        return {e.entry_id: immutable(self.prefix - self.births[e.entry_id]) for e in entries}

    def account(self):
        stream = self.stream.accounting()
        # Prepared numeric_bytes includes its int64 origin; report that scalar
        # separately without double-counting it in the total.
        pending_bytes = sum(int(p.context.numeric_bytes) - 8 for p in self.pending)
        pending_action_bytes = sum(a.nbytes for p in self.pending for a in p.actions)
        prefix_bytes = self.prefix.nbytes + sum(a.nbytes for a in self.births.values())
        prefix_ids = sum(len(key[0].encode()) + 8 for key in self.births)
        origin_bytes = 8 * len(self.pending)
        values = {"resident_observations": stream["resident_observations"],
            "birth_prefix_entries": len(self.births), "pending_contexts": len(self.pending),
            "memory_interface_logical_bytes": stream["logical_payload_bytes"],
            "action_prefix_numeric_bytes": prefix_bytes, "action_prefix_identifier_bytes": prefix_ids,
            "pending_context_numeric_bytes": pending_bytes, "pending_origin_time_bytes": origin_bytes,
            "pending_realized_action_bytes": pending_action_bytes,
            "total_logical_bytes": stream["logical_payload_bytes"] + prefix_bytes + prefix_ids
                                   + pending_bytes + pending_action_bytes + origin_bytes}
        for name, value in values.items():
            self.maxima[name] = max(value, self.maxima.get(name, 0))
        if len(self.pending) > 3 or len(self.births) > 6 or stream["resident_observations"] > 6:
            raise ValueError("Online auxiliary state exceeded its declared bounded state")
        return values

    def observe(self, feature, *, executed_action=None):
        began = time.perf_counter()
        t = self.next_time
        feature = np.asarray(feature)
        if feature.dtype != np.float32 or feature.ndim != 1 or not np.isfinite(feature).all():
            raise ValueError("Require one finite float32 current feature")
        if t == 0:
            if executed_action is not None:
                raise ValueError("First frame has no just-executed action")
        else:
            action = np.asarray(executed_action)
            if action.dtype != np.float32 or action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("Require one finite float32 just-executed additive action")
            self.prefix += action
            for pending in self.pending:
                pending.actions.append(immutable(action))
        self.account()
        surprise = None
        if self.pending and self.pending[0].context.origin_time + 3 == t:
            pending = self.pending.popleft()
            prior = [e for e in self.stream.snapshot.recent if e.timestamp == pending.context.origin_time]
            if len(prior) != 1 or len(pending.actions) != 3:
                raise ValueError("Delayed prediction origin/action prefix is not in the bounded recent context")
            prediction = self.scores.predict_context(pending.context, prior[0].representation,
                                                      np.stack(pending.actions))
            residual = prediction.astype(np.float64) - feature.astype(np.float64)
            surprise = float(np.mean(residual * residual))
            if not np.isfinite(surprise):
                raise ValueError("Nonfinite causal prediction surprise")
            self.surprise_predictions += 1
        if self.pending and self.pending[0].context.origin_time + 3 < t:
            raise ValueError("Missed pending prediction maturity")
        observation = MemoryEntry(self.episode_id, t, feature, surprise=surprise)
        self.births[observation.entry_id] = self.prefix.copy()
        self.maxima["transient_birth_prefix_entries"] = max(len(self.births), self.maxima.get("transient_birth_prefix_entries", 0))
        self.maxima["transient_action_prefix_numeric_bytes"] = max(
            (len(self.births) + 1) * 8, self.maxima.get("transient_action_prefix_numeric_bytes", 0))
        snapshot = self.stream.push(observation)
        entries = snapshot.recent + snapshot.retained
        self.births = {e.entry_id: self.births[e.entry_id] for e in entries}
        if self.policy == "prediction_surprise" and len(snapshot.recent) == 4:
            prepared = self.scores.prepare_context(snapshot.recent, snapshot.retained,
                                                   self.past_actions(entries), t)
            self.pending.append(Pending(prepared, []))
        self.account()
        self.next_time += 1
        self.seconds += time.perf_counter() - began
        return snapshot, {"time": t, "retained_timestamps": [e.timestamp for e in snapshot.retained],
            "recent_timestamps": [e.timestamp for e in snapshot.recent],
            "deleted_timestamp": snapshot.transition.deleted_id[1] if snapshot.transition and snapshot.transition.deleted_id else None,
            "arrivals_seen": snapshot.transition.arrivals_seen if snapshot.transition else 0,
            "current_surprise": surprise, "pending_contexts": len(self.pending)}


def replay_episode(identity, features, actions, query, policy, scores, *, base_seed, trace):
    """Offline harness owns targets; online.observe cannot read this closure."""
    if features.dtype != np.float32 or actions.dtype != np.float32 or actions.shape != (len(features) - 1, 2):
        raise ValueError("Expected normalized float32 features and additive controls")
    if not 6 <= query or query + 6 >= len(features):
        raise ValueError("Validation query must have full R4/K2 state and both requested horizons")
    seed = policy_seed(base_seed, identity, policy)
    online = OnlineRetention(identity.episode_id, policy, scores, seed=seed)
    prediction = None
    query_kept = None
    prediction_seconds = 0.
    for t, feature in enumerate(features):
        snapshot, event = online.observe(feature, executed_action=None if t == 0 else actions[t - 1])
        trace({"episode_id": identity.episode_id, "task": identity.task, "policy": policy, **event})
        if t == query:
            if len(snapshot.retained) != 2:
                raise ValueError("Query memory did not retain exactly K=2 entries")
            summaries = online.past_actions(snapshot.recent + snapshot.retained)
            began = time.perf_counter()
            # This isolated offline world-model scoring uses recorded actions.
            # Those arrays are never passed to online retention or its callbacks.
            prediction = np.stack([scores.predict(snapshot.recent, snapshot.retained, summaries,
                immutable(actions[query:query + horizon]), horizon) for horizon in (3, 6)])
            prediction_seconds = time.perf_counter() - began
            query_kept = [e.timestamp for e in snapshot.retained]
    if prediction is None or prediction.shape != (2, features.shape[1]) or not np.isfinite(prediction).all():
        raise ValueError("Invalid query prediction")
    target = features[query + np.array([3, 6])]
    mse = np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2, axis=1)
    row = {"episode_id": identity.episode_id, "group_id": identity.group_id, "split": identity.split,
        "task": identity.task, "environment_seed": identity.environment_seed, "policy": policy,
        "policy_seed": seed, "query": query, "horizons": [3, 6], "feature_mse": mse.tolist(),
        "retained_timestamps": query_kept, "frames_replayed": len(features),
        "surprise_predictions": online.surprise_predictions,
        "pending_contexts_discarded_at_episode_end": len(online.pending),
        "online_seconds": online.seconds, "offline_query_prediction_seconds": prediction_seconds,
        "logical_memory_maxima": online.maxima,
        "max_transient_real_observations": 7,
        "memory_scope": "logical_owned_payload_plus_pending_state; Python allocator and temporary tensor workspaces excluded",
        "equal_auxiliary_bytes_across_policies": False}
    return row, prediction


def aggregate(rows):
    cells = []
    for task in ("gate", "motion"):
        for policy in POLICIES:
            chosen = [r for r in rows if r["task"] == task and r["policy"] == policy]
            if not chosen:
                continue
            groups = sorted({r["group_id"] for r in chosen})
            pair_losses = np.stack([np.mean([r["feature_mse"] for r in chosen if r["group_id"] == group], axis=0)
                                    for group in groups])
            cells.append({"task": task, "policy": policy, "horizons": [3, 6],
                "episodes": len(chosen), "independent_pairs": len(groups),
                "mean_pair_feature_mse": pair_losses.mean(0).tolist(),
                "pair_sample_std": pair_losses.std(0, ddof=1).tolist() if len(groups) > 1 else None,
                "pair_feature_mse": {group: value.tolist() for group, value in zip(groups, pair_losses)},
                "total_online_seconds": sum(r["online_seconds"] for r in chosen),
                "max_total_logical_bytes": max(r["logical_memory_maxima"]["total_logical_bytes"] for r in chosen)})
    return cells


def worker(args):
    import torch
    from experiments.phase2.train_predictor_pilot import verify_inputs
    from experiments.phase2.frozen_predictor import FrozenPredictorEvaluator
    from experiments.phase2.causal_model_scores import CausalModelScores
    from phase1.cache import load_feature, load_normalization, normalize
    torch.set_num_threads(args.threads)
    data_root, report, predictor_root, output = (getattr(args, n).resolve() for n in
                                               ("data_root", "encoded_report", "predictor_root", "output"))
    config, index, roles, cache = verify_inputs(data_root, report)
    registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in index["records"]], split_roles=roles)
    norm = load_normalization(config)
    rows, predictions, audits, quality = [], [], [], []
    trained = json.loads((predictor_root / "training_result.json").read_text())
    with (output / "online_trace.jsonl").open("x") as trace_stream:
        def trace(event):
            trace_stream.write(json.dumps(event, allow_nan=False) + "\n")
        for task in ("gate", "motion"):
            evaluator = FrozenPredictorEvaluator(predictor_root, report, data_root, task, args.seed)
            scores = CausalModelScores(evaluator)
            training = json.loads((predictor_root / "predictor" / task / f"seed_{args.seed}" / "training.json").read_text())
            initial = float(np.mean(list(training["initial_validation_by_mode"].values())))
            best = next(r["best_validation_mse"] for r in trained["tasks"] if r["task"] == task and r["seed"] == args.seed)
            quality.append({"task": task, "initial_training_validation_mse": initial,
                "selected_training_validation_mse": best, "relative_change_percent": (best / initial - 1) * 100,
                "interpretation": "Pilot training execution does not imply predictive improvement; positive change means worse MSE."})
            records = [r for r in index["records"] if r["task"] == task and r["split"] == "validation"]
            for rec in records:
                identity = EpisodeIdentity.from_record(rec)
                registry.require(identity, "validation")
                features = normalize(load_feature(config, rec, cache), norm)
                with np.load(data_root / "dataset" / rec["obs"], allow_pickle=False) as archive:
                    if set(archive.files) != {"rgb", "actions", "query"} or int(archive["query"]) != rec["query"]:
                        raise ValueError("Observation schema/query differs")
                    actions = archive["actions"].copy()
                for policy in POLICIES:
                    row, prediction = replay_episode(identity, features, actions, rec["query"], policy,
                                                     scores, base_seed=args.seed, trace=trace)
                    row["checkpoint_sha256"] = evaluator.provenance.frozen_predictor_sha256
                    rows.append(row)
                    predictions.append(prediction)
            audit = scores.audit()
            audit["shared_predictor_parameter_buffer_bytes"] = sum(v.numel() * v.element_size() for v in
                list(evaluator._model.parameters()) + list(evaluator._model.buffers()))
            audits.append(audit)
    if len(rows) != 6 * sum(r["split"] == "validation" for r in index["records"]):
        raise ValueError("Six-policy validation coverage mismatch")
    np.savez_compressed(output / "prediction_vectors.npz", predictions=np.stack(predictions),
                        episode_id=np.array([r["episode_id"] for r in rows]), policy=np.array([r["policy"] for r in rows]))
    write(output / "episode_results.json", rows)
    summary = {"schema": "phase2_six_retention_validation_pilot_v1", "status": "completed",
        "scope": "new_development_validation_prediction_pilot", "data_seed": config["data"]["seed"],
        "model_seed": args.seed, "policies": list(POLICIES), "capacity": 2, "recent_length": 4,
        "horizons": [3, 6], "episode_policy_rows": len(rows),
        "unique_episodes": len({r["episode_id"] for r in rows}),
        "unique_pairs": len({r["group_id"] for r in rows}),
        "query_rule": "existing declared query; entire episode replayed continuously",
        "attention_definition": "actual frozen DynamicsPredictor content salience softmax over current candidate memories, conditioned on current feature; not transformer multihead attention",
        "surprise_definition": "MSE(F(context_at_t_minus_3,realized_actions_t_minus_3_through_t_minus_1),observed_feature_t); immutable when observed",
        "surprise_warmup": "No score until the first full recent context at t=3 matures at t=6; missing score has existing lowest priority, deterministic oldest tie-break",
        "surprise_pending_bound": 3, "equal_real_memory_capacity": True,
        "equal_total_memory_bytes": False, "auxiliary_budget_difference_reported": True,
        "loss": "raw float64 mean squared residual in full normalized feature space; per horizon",
        "checkpoint_selection_used_this_validation_split": True,
        "untouched_test_claim": False, "test_feature_arrays_loaded": 0,
        "training_quality_context": quality, "aggregate": aggregate(rows), "frozen_model_audits": audits,
        "cache_signature": cache["signature"], "encoder_hash": cache["encoder_hash"],
        "normalization_sha256": sha256(data_root / "features/normalization.npz"),
        "learned_retention_scorer_trained": False, "baseline_model_planning_verified": False,
        "scientific_phase2_completed": False,
        "files": {name: sha256(output / name) for name in ("online_trace.jsonl", "prediction_vectors.npz", "episode_results.json")}}
    write(output / "summary.json", summary)


def supervise(args):
    from experiments.phase2.run_hpc_pipeline import terminate_group
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_names = ["run_retention_baseline_pilot.py", "causal_model_scores.py", "frozen_predictor.py", "memory.py"]
    manifest = {"status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "new_six_policy_development_validation_prediction_pilot", "device": "cpu",
        "source_sha256": {"experiments/phase2/" + name: sha256(Path(__file__).parent / name) for name in source_names},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu_job_submitted": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_allocation_seconds": None if os.environ.get("SLURM_JOB_ID") else 0,
        "resource_note": "CPU worker; enclosing allocation requires scheduler accounting.",
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
            raise RuntimeError("Baseline worker failed; see worker.log")
        summary = json.loads((output / "summary.json").read_text())
        if summary.get("status") != "completed":
            raise ValueError("Incomplete baseline pilot")
        manifest.update(status="completed", summary_sha256=sha256(output / "summary.json"),
                        six_retention_prediction_pilot_completed=True)
    except subprocess.TimeoutExpired:
        manifest.update(status="timed_out", error="CPU pilot time cap reached; partial evidence preserved")
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
