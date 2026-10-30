"""WR / PaFu mechanism adaptations and exact small-K development oracles.

Offline labels and targets live in this harness, never in the online policy.
This does not reproduce either original paper's complete video/robot system.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np
import torch

from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.generate_deletion_pilot import sha256, write
from experiments.phase2.literature_baselines import (
    CausalSurpriseTracker, SalienceTrainingRecord, make_student_inputs,
    train_salience_student, balanced_attention_targets,
)
from experiments.phase2.memory import MemoryEntry, PolicyContext, StreamingMemory
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.run_retention_baseline_pilot import OnlineRetention
from phase1.model import project


def student_inputs(candidates, recent, t, basis):
    return make_student_inputs(
        project(np.stack([e.representation for e in candidates]), basis),
        project(np.stack([e.representation for e in recent]), basis),
        np.array([e.timestamp for e in candidates]),
        np.array([e.timestamp for e in recent]), t)


class LiteratureRetention(OnlineRetention):
    """Only current frames, executed actions, frozen weights and bounded state."""
    def __init__(self, episode_id, policy, scores, *, seed=17, student=None):
        if policy not in ("wr", "pafu"):
            raise ValueError("Unknown literature adaptation")
        super().__init__(episode_id, "attention", scores, seed=seed)
        self.policy = policy
        self.student = student
        self.tracker = CausalSurpriseTracker() if policy == "wr" else None
        if policy == "pafu" and student is None:
            raise ValueError("PaFu adaptation requires a trained attention student")
        self.stream = StreamingMemory(4, 2, "attention", seed=seed, score_callback=self.score)

    def score(self, candidates, context):
        if self.tracker is not None:
            return self.tracker.score(candidates, context)
        x = student_inputs(candidates, context.recent, context.current_time, self.scores._basis)
        with torch.inference_mode():
            return self.student(torch.tensor(x)).softmax(-1).numpy()

    def observe(self, feature, *, executed_action=None):
        began, previous_seconds = time.perf_counter(), self.seconds
        if self.tracker is not None:
            self.tracker.observe(MemoryEntry(self.episode_id, self.next_time, feature),
                                 past_window=self.stream.snapshot.recent)
        snap, event = super().observe(feature, executed_action=executed_action)
        if self.tracker is not None:
            self.tracker.retain_only(snap.recent + snap.retained)
            event["wr_state"] = self.tracker.accounting()
            event["wr_scalar_records"] = [{"time":r.timestamp,"score":r.raw_score,
                "threshold":r.threshold,"valid":r.valid,"event":r.confirmed_event}
                for r in self.tracker.records]
        self.account()
        self.seconds = previous_seconds + time.perf_counter() - began
        return snap, event

    def account(self):
        values = super().account()
        if getattr(self, "tracker", None) is not None:
            extra = self.tracker.accounting()
            # Module contract: logical auxiliary bytes, no feature archive.
            b = extra["auxiliary_logical_bytes"]
            values["controller_auxiliary_bytes"] = b
            values["total_logical_bytes"] += b
            for k, v in values.items():
                self.maxima[k] = max(v, self.maxima.get(k, 0))
        return values


def online_policy(episode_id, policy, scores, seed=17, student=None):
    if policy in ("wr", "pafu"):
        return LiteratureRetention(episode_id, policy, scores, seed=seed, student=student)
    return OnlineRetention(episode_id, policy, scores, seed=seed)


def overflow_parent(snapshot, feature, episode_id, t):
    if len(snapshot.recent) != 4 or len(snapshot.retained) != 2:
        return None
    candidates = snapshot.retained + (snapshot.recent[0],)
    recent = snapshot.recent[1:] + (MemoryEntry(episode_id, t, feature),)
    return candidates, recent


def attention_teacher(scores, projected_episode, times, decision_time):
    """Offline bidirectional probe of REAL frozen additive pooling attention.

    Queries before a key are outside the causal reader's original training
    domain. This is an explicitly named teacher-probe adaptation, not DiT QKV.
    Per-key past/same/future grouping and max follow PaFu's salience rule.
    """
    qt = np.arange(decision_time + 7)
    if qt[-1] >= len(projected_episode):
        raise ValueError("Teacher requires all declared future queries")
    candidates = torch.tensor(projected_episode[times])
    queries = torch.tensor(projected_episode[qt])
    p = candidates[None].expand(len(qt), -1, -1)
    with torch.inference_mode():
        a = scores._weights(p, queries, torch.ones(p.shape[:2], dtype=torch.bool)).numpy()
    return balanced_attention_targets(a, qt, np.asarray(times), decision_time=decision_time), a


def training_records(snapshot, scores, output):
    records, metadata, arrays, targets, seen = [], [], [], [], set()
    for rec in snapshot.records:
        if rec["task"] != "gate" or rec["split"] != "train":
            continue
        z, actions = snapshot.features(rec), snapshot.actions(rec)
        projected = project(z, scores._basis)
        for policy in ("fifo", "reservoir", "visual_diversity", "attention", "wr"):
            online = online_policy(rec["id"], policy, scores)
            for t in range(rec["query"] + 1):
                parent = overflow_parent(online.stream.snapshot, z[t], rec["id"], t)
                if parent is not None:
                    candidates, recent = parent
                    order = tuple(e.timestamp for e in candidates)
                    key = (rec["id"], t, tuple(sorted(order)))
                    if key not in seen:
                        seen.add(key)
                        x = student_inputs(candidates, recent, t, scores._basis)
                        target, _ = attention_teacher(scores, projected, list(order), t)
                        records.append(SalienceTrainingRecord(x, target, rec["pair"], split="train",
                                                              target_kind="future_attention"))
                        arrays.append(x); targets.append(target)
                        metadata.append({"episode_id": rec["id"], "pair": rec["pair"], "split": "train",
                            "time": t, "candidate_times": list(order), "parent_policy": policy,
                            "teacher_query_max": t + 6})
                online.observe(z[t], executed_action=None if t == 0 else actions[t-1])
    np.savez_compressed(output / "attention_training_inputs.npz", inputs=np.stack(arrays))
    np.savez_compressed(output / "attention_training_targets.npz", targets=np.stack(targets))
    write(output / "attention_training_metadata.json", metadata)
    return records


class QueryOracle:
    """Offline only. Equal-K losses cached by subset, fixed h3/h6 mean."""
    def __init__(self, episode_id, z, actions, q, scores):
        self.episode_id, self.z, self.actions, self.q, self.scores = episode_id, z, actions, q, scores
        self.recent = tuple(MemoryEntry(episode_id, t, z[t]) for t in range(q-3, q+1))
        self.cache = {}

    def evaluate(self, times):
        key = tuple(sorted(int(t) for t in times))
        if len(key) != len(set(key)) or len(key) > 2 or any(t < 0 or t >= self.q-3 for t in key):
            raise ValueError("Oracle requires zero to two distinct historical entries")
        if key not in self.cache:
            kept = tuple(MemoryEntry(self.episode_id, t, self.z[t]) for t in key)
            past = {e.entry_id: self.actions[e.timestamp:self.q].sum(0, dtype=np.float32)
                    for e in self.recent + kept}
            pred = np.stack([self.scores.predict(self.recent, kept, past,
                             self.actions[self.q:self.q+h].copy(), h) for h in (3, 6)])
            target = self.z[self.q + np.array([3, 6])]
            losses = np.mean((pred.astype(np.float64)-target.astype(np.float64))**2, axis=1)
            self.cache[key] = (losses, pred)
        return self.cache[key]

    def best(self, candidate_times):
        subsets = list(combinations(sorted(candidate_times), 2))
        if not subsets:
            raise ValueError("Need at least K candidates")
        values = [(float(self.evaluate(k)[0].mean()), k) for k in subsets]
        loss, subset = min(values)
        return {"mean_mse": loss, "kept_times": list(subset), "subsets_evaluated": len(subsets)}


def evidence_equivalence(z, evidence, historical_stop):
    """Evaluator-only exact feature copies; distinguish evidence from its ID."""
    return [[t for t in range(historical_stop) if np.array_equal(z[t], z[int(e)])]
            for e in evidence]


def diagnose_episode(rec, z, actions, evidence, scores, policy, oracle, *, seed=17, student=None):
    q = rec["query"]
    online = online_policy(rec["id"], policy, scores, seed, student)
    events, query_parent = [], None
    for t in range(q+1):
        parent = overflow_parent(online.stream.snapshot, z[t], rec["id"], t)
        if t == q:
            query_parent = parent
        _, event = online.observe(z[t], executed_action=None if t == 0 else actions[t-1])
        events.append(event)
    snap = online.stream.snapshot
    kept = [e.timestamp for e in snap.retained]
    losses, pred = oracle.evaluate(kept)
    local = oracle.best([e.timestamp for e in query_parent[0]])
    hindsight = oracle.best(range(q-3))
    recent_loss = float(oracle.evaluate(())[0].mean())
    # Exact task evidence reference with deterministic newest filler, not optimum.
    priv = sorted(set(int(t) for t in evidence))
    for t in reversed(range(q-3)):
        if len(priv) >= 2:
            break
        if t not in priv:
            priv.append(t)
    priv_loss = float(oracle.evaluate(priv)[0].mean())
    evicted = {int(t): next((e["time"] for e in events if e["deleted_timestamp"] == int(t)), None)
               for t in evidence}
    groups = evidence_equivalence(z, evidence, q-3)
    survived = all(any(t in kept for t in group) for group in groups)
    information_loss = [next((event["time"] for event in events
        if event["time"] > max(group) and not set(group).intersection(
            event["retained_timestamps"]+event["recent_timestamps"])), None) for group in groups]
    teacher_diagnostic = None
    if policy == "pafu":
        candidate_times = [e.timestamp for e in query_parent[0]]
        probabilities, _ = attention_teacher(scores, project(z, scores._basis), candidate_times, q)
        teacher_kept = [t for i,t in enumerate(candidate_times) if i != int(np.argmin(probabilities))]
        teacher_loss = float(oracle.evaluate(teacher_kept)[0].mean())
        teacher_diagnostic = {"probabilities":probabilities.tolist(), "kept_times":teacher_kept,
            "same_deletion_as_student":set(teacher_kept)==set(kept), "mean_mse":teacher_loss,
            "teacher_local_regret":teacher_loss-local["mean_mse"],
            "student_minus_teacher_mse":float(losses.mean())-teacher_loss,
            "future_query_count":6,"offline_only":True}
    row = {"episode_id": rec["id"], "pair": rec["pair"], "split": rec["split"],
        "policy": policy, "seed": seed, "query": q, "kept_times": kept,
        "feature_mse": losses.tolist(), "mean_mse": float(losses.mean()),
        "local_oracle": local, "hindsight_oracle": hindsight,
        "local_regret": float(losses.mean())-local["mean_mse"],
        "hindsight_regret": float(losses.mean())-hindsight["mean_mse"],
        "recent_only_mse": recent_loss, "privileged_evidence_mse": priv_loss,
        "evidence_times": [int(t) for t in evidence], "evidence_survived": survived,
        "evidence_timestamp_survived":all(int(t) in kept for t in evidence),
        "evidence_equivalent_feature_times":groups,
        "first_evidence_information_loss":information_loss,
        "first_evidence_eviction": evicted, "logical_memory_maxima": online.maxima,
        "online_seconds": online.seconds,
        "controller_parameter_bytes": 0 if student is None else sum(p.numel()*p.element_size() for p in student.parameters()),
        "teacher_diagnostic":teacher_diagnostic,
        "evidence_usefulness": "positive" if priv_loss < recent_loss else "not_established",
        "causal_opportunity": "cue_observed_but_future_action_unknown_to_retention; oracle_gap_not_learnability_proof",
        "failure_stage": "retained_evidence" if survived else "evidence_evicted_before_query"}
    if min(row["local_regret"], row["hindsight_regret"]) < -1e-10:
        raise AssertionError("Oracle missed baseline's feasible subset")
    return row, pred, events


def aggregate(rows):
    result = []
    for policy in sorted({r["policy"] for r in rows}):
        part = [r for r in rows if r["policy"] == policy]
        pair_values = {}
        for r in part:
            pair_values.setdefault(r["pair"], []).append(r["hindsight_regret"])
        means = np.array([np.mean(v) for v in pair_values.values()])
        rng = np.random.default_rng(20261022)
        boot = means[rng.integers(len(means), size=(2000, len(means)))].mean(1)
        result.append({"policy": policy, "episode_seed_rows": len(part), "independent_pairs": len(means),
            **{k: float(np.mean([r[k] for r in part])) for k in
               ("mean_mse", "local_regret", "hindsight_regret", "evidence_survived")},
            "local_positive_regret_fraction": float(np.mean([r["local_regret"] > 1e-8 for r in part])),
            "hindsight_positive_regret_fraction": float(np.mean([r["hindsight_regret"] > 1e-8 for r in part])),
            "pair_bootstrap_hindsight_regret_ci95": np.quantile(boot, [.025,.975]).tolist(),
            "max_logical_state_bytes": max(r["logical_memory_maxima"]["total_logical_bytes"] for r in part),
            "controller_parameter_bytes":max(r["controller_parameter_bytes"] for r in part),
            "online_seconds_per_episode":float(np.mean([r["online_seconds"] for r in part]))})
    return result


def run(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refuse to overwrite experiment output")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    snapshot = PortableSnapshot(args.snapshot)
    evaluator = FollowupGateEvaluator(snapshot, args.followup)
    scores = CausalModelScores(evaluator)
    records = [r for r in snapshot.records if r["task"] == "gate"]
    train_pairs = {r["pair"] for r in records if r["split"] == "train"}
    val_pairs = {r["pair"] for r in records if r["split"] == "validation"}
    if train_pairs & val_pairs:
        raise ValueError("Pair leakage")
    sources = [Path(__file__), ROOT/"experiments/phase2/literature_baselines.py",
               ROOT/"experiments/phase2/causal_model_scores.py", ROOT/"experiments/phase2/memory.py",
               ROOT/"experiments/phase2/run_retention_baseline_pilot.py"]
    protocol = {"version": "literature-bank-dev-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "existing_gate_development_only", "recent": 4, "capacity": 2, "horizons": [3,6],
        "objective": "mean of h3 and h6 feature MSE; one shared deletion decision",
        "teacher": "real frozen additive reader pooling; offline bidirectional query probe 0..t+6; per-key temporal-group max then equal group mean; normalized targets",
        "teacher_limit": "not original DiT/QKV; earlier-than-key probe outside causal reader training domain",
        "student": {"seeds": [17,23,41], "epochs":100, "hidden":32, "lr":.001, "selection":"fixed final epoch; no validation selection", "targets":"attention only"},
        "wr": {"gaussian_window":4,"scalar_score_window":64,"gamma":1.,"nms_radius":3,"confirmation_delay":1},
        "train_pairs":sorted(train_pairs),"validation_pairs":sorted(val_pairs),
        "test_arrays":False,"own_utility_scorer":False,"independent_generalization":False,
        "oracle_archive": "evaluation only; no callback or online object holds it",
        "source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in sources}}
    write(out/"protocol.json", protocol)
    train = training_records(snapshot, scores, out)
    students, training = {}, []
    for seed in (17,23,41):
        model, curve = train_salience_student(train, seed=seed, epochs=100, hidden=32, lr=.001,
                                             train_group_ids=train_pairs)
        students[seed] = model
        torch.save({"model":model.state_dict(), "input_dim":train[0].inputs.shape[-1],
                    "hidden":32,"seed":seed,"protocol_sha256":sha256(out/"protocol.json"),
                    "target_kind":"future_attention","trained_utility_scorer":False},out/f"pafu_seed_{seed}.pt")
        training.append({"seed":seed,"loss_curve":curve,"parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters())})
    write(out/"training.json", {"records":len(train), "training_pairs":len(train_pairs),"students":training})
    rows, preds, targets, oracle_rows = [], [], [], []
    with (out/"online_trace.jsonl").open("x") as trace:
        for rec in records:
            if rec["split"] != "validation":
                continue
            z, actions, evidence = snapshot.features(rec), snapshot.actions(rec), snapshot.evidence(rec)
            oracle = QueryOracle(rec["id"], z, actions, rec["query"], scores)
            arms = [(p,17,None) for p in ("fifo","reservoir","visual_diversity","attention","prediction_surprise","wr")]
            arms += [("pafu",seed,m) for seed,m in students.items()]
            for policy,seed,model in arms:
                row,pred,events = diagnose_episode(rec,z,actions,evidence,scores,policy,oracle,seed=seed,student=model)
                rows.append(row); preds.append(pred); targets.append(z[rec["query"]+np.array([3,6])])
                for event in events:
                    trace.write(json.dumps({"episode_id":rec["id"],"policy":policy,"seed":seed,**event},allow_nan=False)+"\n")
            oracle_rows.extend({"episode_id":rec["id"],"kept_times":list(k),"feature_mse":v[0].tolist()}
                               for k,v in oracle.cache.items())
    write(out/"episode_results.json",rows)
    write(out/"oracle_subsets.json",oracle_rows)
    np.savez_compressed(out/"prediction_vectors.npz",predictions=np.stack(preds),targets=np.stack(targets))
    summary={"status":"completed_development_diagnostic","elapsed_seconds":time.monotonic()-started,
        "device":"cpu","aggregate":aggregate(rows),"episodes":len(val_pairs)*2,"independent_pairs":len(val_pairs),
        "student_training_records":len(train),"protocol_sha256":sha256(out/"protocol.json"),
        "source_sha256":protocol["source_sha256"],"input_audit":snapshot.audit(),"predictor_audit":scores.audit(),
        "equal_retained_K":True,"equal_total_bytes":False,"test_arrays_read":0,
        "new_visual_scenes":0,"fresh_generalization_claim":False,"utility_scorer_trained":False,
        "oracle_definition":"local feasible K+1 deletion; exhaustive all-observed-history K2 evaluator-only reference",
        "limitations":["four reused validation pairs", "PaFu bidirectional pooling proxy not original architecture", "oracle future actions may not be causally predictable", "WR window4 instead of original64", "no new visual encoding or planning"]}
    write(out/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("status","elapsed_seconds","aggregate")},indent=2))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--followup",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
