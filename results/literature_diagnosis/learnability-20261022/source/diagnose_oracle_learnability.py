"""Fixed CPU feasibility probes for causal deletion decisions, not a final selector.

Fit three identically sized ridge probes on old TRAIN-only parents. Future loss
and attention are labels only. Fresh scenes are already-inspected development
data; no model, regularizer, checkpoint or threshold is selected on them.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import torch
from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.generate_deletion_pilot import sha256, write
from experiments.phase2.memory import MemoryEntry, PolicyContext, StreamingMemory, BoundedMemory
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.run_literature_baselines import QueryOracle, attention_teacher, student_inputs, overflow_parent
from experiments.phase2.run_retention_baseline_pilot import OnlineRetention
from phase1.model import project

ARMS = ("cost_ridge", "attention_ridge", "shuffled_cost_ridge")
TOL = 1e-8


def hash_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.asarray(array)
        h.update(str(a.dtype).encode() + repr(a.shape).encode() + a.tobytes(order="C"))
    return h.hexdigest()


def causal_features(candidates, recent, current_time, basis, past_actions):
    """Only bounded resident observations and executed-action suffix sums.

    Shared candidate features plus candidate*current/candidate*set-mean terms
    allow a fixed linear head to react to context/redundancy. No future or task
    evidence argument exists. All three label arms share this exact mapping.
    """
    entries = tuple(candidates) + tuple(recent)
    if set(past_actions) != {e.entry_id for e in entries}:
        raise ValueError("Executed-action summaries must match resident entries")
    ca = np.stack([past_actions[e.entry_id] for e in candidates])
    ra = np.stack([past_actions[e.entry_id] for e in recent])
    if ca.shape != (len(candidates), 2) or ra.shape != (len(recent), 2) or not np.isfinite(ca).all() or not np.isfinite(ra).all():
        raise ValueError("Require finite executed-action summaries")
    base = student_inputs(candidates, recent, current_time, basis).astype(np.float64)
    p = project(np.stack([e.representation for e in candidates]), basis).astype(np.float64)
    current = project(recent[-1].representation, basis).astype(np.float64)
    x = np.concatenate((base, p * current, p * p.mean(0), ca,
                        np.broadcast_to(ra.mean(0), ca.shape), np.broadcast_to(ca.mean(0), ca.shape)), axis=1)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite causal probe feature")
    return x


def parent_from_arrays(rec, z, actions, t, times):
    times = tuple(sorted(int(v) for v in times))
    if len(times) != 3 or len(set(times)) != 3 or times[-1] >= t-3:
        raise ValueError("Require exactly K+1 historical candidates")
    candidates = tuple(MemoryEntry(rec["id"], k, z[k]) for k in times)
    recent = tuple(MemoryEntry(rec["id"], k, z[k]) for k in range(t-3, t+1))
    past = {e.entry_id: actions[e.timestamp:t].sum(0, dtype=np.float32) for e in candidates + recent}
    return candidates, recent, past


@dataclass
class RidgeProbe:
    scale: np.ndarray
    weights: np.ndarray
    label_scale: float
    target: str

    @classmethod
    def fit(cls, inputs, labels, metadata, target):
        x, y = np.asarray(inputs, np.float64), np.asarray(labels, np.float64)
        if x.ndim != 3 or x.shape[:2] != y.shape or x.shape[1] != 3 or len(metadata) != len(x):
            raise ValueError("Require aligned three-candidate parent arrays")
        if any(row.get("split") != "train" for row in metadata):
            raise ValueError("Probe fitting is restricted to old TRAIN parents")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite fit input")
        # Per-parent centering is invariant to candidate permutation and removes
        # common score offsets, which cannot influence a single deletion.
        xc = x-x.mean(1, keepdims=True)
        yc = y-y.mean(1, keepdims=True)
        scale = np.maximum(np.sqrt(np.mean(xc*xc, axis=(0, 1))), 1e-6)
        label_scale = max(float(np.sqrt(np.mean(yc*yc))), 1e-12)
        a, b = (xc/scale).reshape(-1, x.shape[-1]), (yc/label_scale).ravel()
        # Predeclared lambda=1 for mean squared error + lambda * ||w||^2.
        weights = np.linalg.solve(a.T@a/len(a)+np.eye(a.shape[1]), a.T@b/len(a))
        return cls(scale, weights, label_scale, target)

    def predict(self, inputs):
        x = np.asarray(inputs, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.weights) or not np.isfinite(x).all():
            raise ValueError("Require finite candidate-by-feature input")
        return ((x-x.mean(0))/self.scale)@self.weights*self.label_scale

    @property
    def bytes(self):
        return self.scale.nbytes+self.weights.nbytes+8


class ProbeRetention(OnlineRetention):
    """Bounded replay; no future labels, data archive or episode is stored."""
    def __init__(self, episode_id, scores, probe):
        super().__init__(episode_id, "attention", scores, seed=17)
        self.probe = probe
        self.stream = StreamingMemory(4, 2, "attention", seed=17, score_callback=self.score)

    def score(self, candidates, context):
        x = causal_features(candidates, context.recent, context.current_time, self.scores._basis,
                            self.past_actions(tuple(candidates)+context.recent))
        # A larger deletion loss OR attention salience means retain this item.
        return self.probe.predict(x)


def parent_hashes(z, actions, t, times, x):
    """Exact equality audits, not approximate similarity or hidden-state labels."""
    times = np.array(sorted(times), np.int64)
    recent_times = np.arange(t-3, t+1, dtype=np.int64)
    suffixes = np.stack([actions[k:t].sum(0, dtype=np.float32) for k in list(times)+list(recent_times)])
    return {
        "full_observed_prefix": hash_arrays(np.array([t], np.int64), z[:t+1], actions[:t]),
        "bounded_observed_state": hash_arrays(np.array([t], np.int64), times, z[times], recent_times,
                                             z[recent_times], suffixes, actions[:t].sum(0, dtype=np.float32)),
        "probe_inputs": hash_arrays(x),
        "future_actions": hash_arrays(actions[t:t+6]),
        "future_targets": hash_arrays(z[t+np.array([3, 6])]),
    }


def ambiguity_report(rows, hash_key):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["hashes"][hash_key]].append(row)
    examples, repeated, conflicts, divergent, seen_episodes = [], 0, 0, 0, set()
    # Full-prefix groups may contain different candidate sets: compare the same
    # three available actions only. A common past does not make parents equal.
    if hash_key == "full_observed_prefix":
        buckets = {(key, tuple(row["candidate_times"])): [] for key, group in buckets.items() for row in group}
        for row in rows:
            buckets[(row["hashes"][hash_key], tuple(row["candidate_times"]))].append(row)
    bound_sum, bound_weight = 0., 0
    for key, group in buckets.items():
        # Repeated methods reaching the same episode/parent are not new futures.
        group = list({(r["episode_id"], tuple(r["candidate_times"])): r for r in group}.values())
        if len({r["episode_id"] for r in group}) < 2:
            continue
        repeated += 1
        seen_episodes.update(r["episode_id"] for r in group)
        costs = np.array([r["losses"] for r in group])
        optimal = costs <= costs.min(1, keepdims=True)+TOL
        conflict = not np.any(optimal.all(0))
        divergent += len({r["hashes"]["future_targets"] for r in group}) > 1
        conflicts += conflict
        bound = float(costs.mean(0).min()-costs.min(1).mean())
        bound_sum += bound*len(group); bound_weight += len(group)
        if conflict:
            examples.append({"state_hash": str(key), "episodes": [r["episode_id"] for r in group],
                             "candidate_times": group[0]["candidate_times"], "losses": costs.tolist(),
                             "optimal_deletion_indices": [np.flatnonzero(a).tolist() for a in optimal],
                             "future_action_hashes": sorted({r["hashes"]["future_actions"] for r in group}),
                             "empirical_shared_action_regret": bound})
    return {"state_definition": hash_key, "unique_states": len(buckets), "cross_episode_repeated_states": repeated,
            "episodes_in_repeated_states": len(seen_episodes), "repeated_states_with_different_future_targets": int(divergent),
            "strict_optimal_action_conflicts": int(conflicts), "comparison_tolerance": TOL,
            "empirical_shared_action_regret_on_repeated_rows": bound_sum/max(1, bound_weight),
            "rows_in_repeated_states": bound_weight, "conflict_examples": examples,
            "scope": "Empirical ambiguity of this exact represented state; not a distribution-wide Bayes bound or proof about full-history policies."}


def ci(values):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(20261022)
    samples = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "low": float(np.quantile(samples, .025)),
            "high": float(np.quantile(samples, .975)), "pairs": len(values)}


def paired_summary(rows, fields, conditions):
    answer = {}
    for condition in conditions:
        chosen = rows if condition == "all" else [r for r in rows if r["condition"] == condition]
        by_pair = defaultdict(lambda: defaultdict(list))
        for row in chosen:
            by_pair[row["pair"]][row["episode_id"]].append(row)
        answer[condition] = {field: ci([np.mean([np.mean([r[field] for r in episode])
                                                for episode in group.values()]) for group in by_pair.values()])
                             for field in fields}
    return answer


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty new diagnostic output")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    old, fresh = PortableSnapshot(args.snapshot), FreshGateDataset(args.fresh)
    scores = CausalModelScores(FollowupGateEvaluator(old, args.followup))
    fresh.verify_old_normalization(old.norm)
    training_metadata = json.loads((args.students/"attention_training_metadata.json").read_text())
    train_ids = {r["id"] for r in old.records if r["task"] == "gate" and r["split"] == "train"}
    train_pairs = {old.by_id[e]["pair"] for e in train_ids}
    if len(train_ids) != 32 or len(train_pairs) != 16 or len(training_metadata) != 1282:
        raise ValueError("Unexpected fixed training population")
    if any(m["episode_id"] not in train_ids or m["split"] != "train" or m["pair"] not in train_pairs for m in training_metadata):
        raise ValueError("Training parent crosses old TRAIN split")
    if train_pairs.intersection(r["pair"] for r in fresh.records) or train_ids.intersection(fresh.by_id):
        raise ValueError("Train/development overlap")
    source_paths = [Path(__file__)] + [ROOT/"experiments/phase2"/name for name in
        ("run_literature_baselines.py", "literature_baselines.py", "memory.py", "causal_model_scores.py",
         "run_retention_baseline_pilot.py", "run_gate_followup_phase2.py", "run_fresh_literature_diagnostic.py")]
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    inputs_to_pin = [args.students/"attention_training_metadata.json", args.students/"attention_training_targets.npz",
                     args.evaluation/"episode_results.json", args.evaluation/"online_trace.jsonl", args.evaluation/"oracle_subsets.json"]
    protocol = {"version": "oracle-learnability-diagnostic-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "training_parents": 1282, "train_pairs": sorted(train_pairs), "fresh_pairs": sorted({r["pair"] for r in fresh.records}),
        "source_sha256": sources, "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs_to_pin},
        "fresh_manifest_sha256": fresh.manifest_sha256, "predictor_binding": scores._evaluator.binding,
        "features": "Causal student features; candidate*current and candidate*set-mean; executed-action suffixes, their candidate/recent means. No IDs, split/condition, evidence labels, future actions or future observations.",
        "fit": {"architecture": "shared scalar linear ridge, parent-centered inputs/labels", "lambda": 1.,
                "scaling": "RMS fitted only on old TRAIN parents", "seed_for_within_parent_null_permutation": 17,
                "targets": ["mean h3/h6 deletion cost", "future reader attention", "within-parent shuffled deletion cost"],
                "hyperparameter_or_checkpoint_selection": False},
        "evaluation": "All frozen fresh baseline query parents, deduplicated by episode/three timestamps; then all three probes in irreversible R4/K2 replay",
        "oracle": "One common deletion for mean h3/h6 MSE; fresh query subset cache reused without new target-dependent tuning",
        "alias_audit": "Exact full observed normalized-feature/action prefix versus bounded resident observations/times/action sums versus projected probe inputs; no hidden-state equality proxy",
        "bootstrap": {"unit": "scene pair, average parents and siblings first", "replicates": 5000, "seed": 20261022},
        "scope": "Exploratory feasibility diagnostic on already inspected fresh development scenes; not production selector or untouched confirmation",
        "test_arrays": False, "new_encoder_or_predictor_training": False}
    write(output/"protocol.json", protocol)  # Before cost labels or probe fitting.
    protocol_sha = sha256(output/"protocol.json")
    xtrain, cost, attention, train_meta, oracles = [], [], [], [], {}
    saved_attention = np.load(args.students/"attention_training_targets.npz")["targets"]
    for index, meta in enumerate(training_metadata):
        rec = old.by_id[meta["episode_id"]]
        z, actions, t = old.features(rec), old.actions(rec), int(meta["time"])
        order = np.argsort(meta["candidate_times"])
        times = [meta["candidate_times"][i] for i in order]
        candidates, recent, past = parent_from_arrays(rec, z, actions, t, times)
        xtrain.append(causal_features(candidates, recent, t, scores._basis, past))
        oracle = oracles.setdefault((rec["id"], t), QueryOracle(rec["id"], z, actions, t, scores))
        cost.append([float(oracle.evaluate([k for j, k in enumerate(times) if j != d])[0].mean()) for d in range(3)])
        attention.append(saved_attention[index][order])
        train_meta.append({**meta, "candidate_times": times})
    xtrain, cost, attention = np.array(xtrain), np.array(cost), np.array(attention)
    rng = np.random.default_rng(17)
    null = np.array([row[rng.permutation(3)] for row in cost])
    fit_started = time.monotonic()
    probes = {name: RidgeProbe.fit(xtrain, labels, train_meta, name)
              for name, labels in zip(ARMS, (cost, attention, null))}
    fit_seconds = time.monotonic()-fit_started
    np.savez_compressed(output/"training_inputs_labels.npz", inputs=xtrain, cost=cost, attention=attention, shuffled_cost=null)
    np.savez_compressed(output/"probe_parameters.npz", **{name+"_"+key: getattr(p, key)
        for name, p in probes.items() for key in ("scale", "weights", "label_scale")})
    write(output/"training_metadata.json", train_meta)
    print(json.dumps({"stage": "fit_complete", "parents": len(train_meta), "feature_width": xtrain.shape[-1], "seconds": fit_seconds}), flush=True)
    # Cache includes every feasible K2 subset at each final query.
    loss_cache = defaultdict(dict)
    for row in json.loads((args.evaluation/"oracle_subsets.json").read_text()):
        loss_cache[row["episode_id"]][tuple(row["kept_times"])] = float(np.mean(row["feature_mse"]))
    baseline_rows = json.loads((args.evaluation/"episode_results.json").read_text())
    baseline_by_id = defaultdict(dict)
    for row in baseline_rows:
        baseline_by_id[row["episode_id"]][(row["policy"], row["seed"])] = row
    parents = defaultdict(set)
    with (args.evaluation/"online_trace.jsonl").open() as file:
        for line in file:
            row = json.loads(line); rec = fresh.by_id[row["episode_id"]]
            if row["time"] == rec["query"]:
                parents[rec["id"]].add(tuple(sorted(row["retained_timestamps"]+[row["deleted_timestamp"]])))
    common, online_rows, traces = [], [], []
    train_fit = {name: float(np.mean([np.mean((p.predict(x)-(y-y.mean()))**2)
        for x, y in zip(xtrain, labels)])) for (name, p), labels in zip(probes.items(), (cost, attention, null))}
    for index, rec in enumerate(fresh.records):
        z, actions = fresh.episode(rec); t = rec["query"]
        for times in sorted(parents[rec["id"]]):
            candidates, recent, past = parent_from_arrays(rec, z, actions, t, times)
            x = causal_features(candidates, recent, t, scores._basis, past)
            losses = np.array([loss_cache[rec["id"]][tuple(k for j, k in enumerate(times) if j != d)] for d in range(3)])
            teacher, _ = attention_teacher(scores, project(z, scores._basis), list(times), t)
            weights = scores.attention(candidates, PolicyContext(t, recent))
            diversity = BoundedMemory(2, "visual_diversity")._select_deletion(candidates, PolicyContext(t, recent), 3)
            choices = {name: int(np.argmin(p.predict(x))) for name, p in probes.items()}
            choices.update(fifo=0, attention=int(np.argmin(weights)), visual_diversity=int(diversity), attention_teacher=int(np.argmin(teacher)))
            row = {"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"],
                   "query": t, "candidate_times": list(times), "losses": losses.tolist(),
                   "margin": float(np.partition(losses, 1)[1]-losses.min()),
                   "hashes": parent_hashes(z, actions, t, times, x), "choices": choices}
            for name, deleted in choices.items():
                row[name+"_mse"] = float(losses[deleted])
                row[name+"_regret"] = float(losses[deleted]-losses.min())
                row[name+"_optimal"] = float(losses[deleted] <= losses.min()+TOL)
            row["cost_minus_attention"] = row["cost_ridge_mse"]-row["attention_ridge_mse"]
            row["cost_minus_null"] = row["cost_ridge_mse"]-row["shuffled_cost_ridge_mse"]
            common.append(row)
        for name, probe in probes.items():
            online = ProbeRetention(rec["id"], scores, probe)
            parent = None
            for now in range(t+1):
                if now == t:
                    parent = overflow_parent(online.stream.snapshot, z[now], rec["id"], now)
                snap, event = online.observe(z[now], executed_action=None if now == 0 else actions[now-1])
                traces.append({"episode_id": rec["id"], "arm": name, **event})
            kept = tuple(sorted(e.timestamp for e in snap.retained))
            parent_times = sorted(e.timestamp for e in parent[0])
            local = min(loss_cache[rec["id"]][tuple(k for j, k in enumerate(parent_times) if j != d)] for d in range(3))
            mse = loss_cache[rec["id"]][kept]
            baseline = baseline_by_id[rec["id"]]
            cue_groups = baseline[("attention", 17)]["evidence_equivalent_feature_times"]
            row = {"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"], "arm": name,
                   "kept_times": list(kept), "candidate_times": parent_times, "mse": mse, "local_regret": mse-local,
                   "hindsight_regret": mse-min(v for k,v in loss_cache[rec["id"]].items() if len(k)==2),
                   "cue_retained": float(all(any(k in kept for k in group) for group in cue_groups)),
                   "parameter_bytes": probe.bytes, "dynamic_logical_maxima": online.maxima,
                   "online_seconds": online.seconds}
            for control in ("fifo", "attention", "visual_diversity", "wr"):
                row["minus_"+control] = mse-baseline[(control, 17)]["mean_mse"]
            row["minus_pafu"] = mse-np.mean([baseline[("pafu", seed)]["mean_mse"] for seed in (17,23,41)])
            online_rows.append(row)
        if (index+1)%32 == 0:
            print(json.dumps({"stage": "evaluating", "episodes": index+1}), flush=True)
    online_by_id = defaultdict(dict)
    for row in online_rows:
        online_by_id[row["episode_id"]][row["arm"]] = row
    for arms in online_by_id.values():
        for row in arms.values():
            row["minus_attention_ridge"] = row["mse"]-arms["attention_ridge"]["mse"]
            row["minus_shuffled_cost_ridge"] = row["mse"]-arms["shuffled_cost_ridge"]["mse"]
    conditions = ["all", "id", "long_gap", "distractor_shift"]
    fields = [name+"_"+metric for name in (*ARMS, "fifo", "attention", "visual_diversity", "attention_teacher")
              for metric in ("mse", "regret", "optimal")]+["cost_minus_attention", "cost_minus_null"]
    common_stats = paired_summary(common, fields, conditions)
    online_stats = {name: paired_summary([r for r in online_rows if r["arm"]==name],
        ["mse", "local_regret", "hindsight_regret", "cue_retained", "minus_fifo", "minus_attention", "minus_visual_diversity",
         "minus_wr", "minus_pafu", "minus_attention_ridge", "minus_shuffled_cost_ridge"], conditions) for name in ARMS}
    ambiguity = {key: ambiguity_report(common, key) for key in ("full_observed_prefix", "bounded_observed_state", "probe_inputs")}
    write(output/"common_parent_results.json", common)
    write(output/"online_results.json", online_rows)
    write(output/"ambiguity.json", ambiguity)
    write(output/"statistics.json", {"common_parent": common_stats, "online": online_stats})
    with (output/"online_trace.jsonl").open("w") as file:
        for row in traces:
            file.write(json.dumps(row, allow_nan=False)+"\n")
    if sha256(output/"protocol.json") != protocol_sha or any(sha256(ROOT/p)!=h for p,h in sources.items()):
        raise ValueError("Frozen protocol or source changed")
    if any(sha256(ROOT/p)!=h for p,h in protocol["input_sha256"].items()):
        raise ValueError("Pinned training/evaluation input changed")
    summary = {"status": "completed_causal_learnability_diagnostic", "protocol_sha256": protocol_sha,
        "elapsed_seconds": time.monotonic()-started, "fit_seconds": fit_seconds, "training_parents": len(train_meta),
        "train_pairs": 16, "fresh_pairs": 96, "common_parent_rows": len(common), "online_rows": len(online_rows),
        "feature_width": xtrain.shape[-1], "parameters_per_arm": len(next(iter(probes.values())).weights),
        "probe_train_centered_mse": train_fit, "fixed_lambda": 1., "hyperparameter_selection": False,
        "margin_quantiles": {str(q):float(np.quantile([r['margin'] for r in common],q)) for q in (0,.25,.5,.75,.95,1)},
        "strict_margin_fraction": float(np.mean([r['margin']>TOL for r in common])),
        "common_parent_statistics": common_stats, "online_statistics": online_stats,
        "ambiguity_summary": {k:{a:b for a,b in v.items() if a!='conflict_examples'} for k,v in ambiguity.items()},
        "old_input_audit": old.audit(), "fresh_input_audit": fresh.audit(), "predictor_audit": scores.audit(),
        "new_GPU_seconds": 0, "production_selector": False, "past_only_input_contract": True,
        "limits": ["Already-inspected fresh development data; exploratory pair bootstrap without multiplicity correction",
                   "Small fixed ridge probes do not establish the ceiling of learnability or final method superiority",
                   "Feature-MSE oracle may disagree with physical task utility; probe does not repair the frozen predictor",
                   "Exact bounded-state aliasing after evidence eviction is not identical full-history ambiguity",
                   "Same K and matched probe parameters; full baselines do not have equal total byte/compute budgets"]}
    summary["result_sha256"] = {str(p.relative_to(output)):sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    write(output/"summary.json", summary)
    print(json.dumps({k:summary[k] for k in ("status","elapsed_seconds","common_parent_rows","online_rows","ambiguity_summary")}, indent=2), flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,default=ROOT/"build/hpc-20261020/development-assets")
    parser.add_argument("--followup",type=Path,default=ROOT/"build/history-diagnosis-20261020/followup-blas1")
    parser.add_argument("--students",type=Path,default=ROOT/"build/literature-baselines-20261022/dev-v3")
    parser.add_argument("--fresh",type=Path,default=ROOT/"build/literature-baselines-20261022/fresh-data")
    parser.add_argument("--evaluation",type=Path,default=ROOT/"build/literature-baselines-20261022/fresh-eval")
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
