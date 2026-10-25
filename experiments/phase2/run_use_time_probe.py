"""Exploratory fixed-reader use-time deletion labels with causal ridge students.

This is an offline teacher experiment, not an online oracle. Each candidate
deletion is followed by the SAME frozen attention policy until the recorded
query, then one common kept set is scored at q+3/q+6. The deployed student is
the unchanged bounded ProbeRetention interface and never receives the archive,
query time, future actions, future targets, or task-evidence annotations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.diagnose_gate_oracle_metric import paired_components
from experiments.phase2.diagnose_oracle_learnability import (
    ProbeRetention, RidgeProbe, causal_features, parent_from_arrays,
)
from experiments.phase2.gate_confirmation_data import ConfirmationGateDataset
from experiments.phase2.generate_deletion_pilot import sha256, write
from experiments.phase2.memory import MemoryEntry, PolicyContext
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.run_literature_baselines import QueryOracle, evidence_equivalence
from experiments.phase2.run_retention_baseline_pilot import OnlineRetention

TOL = 1e-8
ARMS = ("local_cost", "use_time_cost", "future_attention", "shuffled_local_cost", "shuffled_use_time_cost")


class AttentionContinuation:
    """Evaluator-only archive owner; its callback sees bounded causal context.

    The memoization table is teacher computation and is never credited as
    online memory. No previously evicted index can re-enter a continuation:
    at time s the sole incoming historical index is s-4 (> every initial
    candidate), and the remaining two entries are the previous kept set.
    """
    def __init__(self, episode_id, z, query, attention):
        self.episode_id, self.z, self.query = episode_id, z, int(query)
        self.attention, self.cache, self.calls = attention, {}, 0

    def final_kept(self, t, kept):
        t, kept = int(t), tuple(sorted(int(k) for k in kept))
        if len(kept) != 2 or len(set(kept)) != 2 or min(kept) < 0 or max(kept) >= t-3 or t > self.query:
            raise ValueError("Require exactly two valid retained items after a decision")
        key = (t, kept)
        if key not in self.cache:
            if t == self.query:
                self.cache[key] = kept
            else:
                now = t + 1
                times = kept + (now - 4,)
                candidates = tuple(MemoryEntry(self.episode_id, k, self.z[k]) for k in times)
                recent = tuple(MemoryEntry(self.episode_id, k, self.z[k]) for k in range(now-3, now+1))
                scores = np.asarray(self.attention(candidates, PolicyContext(now, recent)))
                if scores.shape != (3,) or not np.isfinite(scores).all():
                    raise ValueError("Invalid frozen attention continuation scores")
                # Same chronological candidate order and np.argmin tie rule as
                # BoundedMemory(policy='attention').
                dropped = int(np.argmin(scores))
                self.calls += 1
                self.cache[key] = self.final_kept(now, tuple(k for j, k in enumerate(times) if j != dropped))
        return self.cache[key]


def deletion_labels(rec, z, actions, t, times, scores, continuation=None, query_oracle=None):
    """Three deletion actions share q, continuation, and both final horizons."""
    times = tuple(sorted(int(k) for k in times))
    if len(times) != 3 or len(set(times)) != 3 or max(times) >= t-3:
        raise ValueError("Require one real K+1 parent")
    continuation = continuation or AttentionContinuation(rec["id"], z, rec["query"], scores.attention)
    query_oracle = query_oracle or QueryOracle(rec["id"], z, actions, rec["query"], scores)
    endpoints = [continuation.final_kept(t, tuple(k for j, k in enumerate(times) if j != d)) for d in range(3)]
    losses = [query_oracle.evaluate(kept)[0] for kept in endpoints]
    return np.mean(losses, axis=1), {"final_kept": [list(k) for k in endpoints],
                                   "horizon_losses": [v.tolist() for v in losses]}


def compare_case(args, old, fresh, scores):
    """Pre-existing inspected confirmation example; excluded from training."""
    encoder = ROOT / "experiments/phase2/encode_gate_confirmation.py"
    if not encoder.exists():
        # Confirmation manifest pins its helper; find its actual saved name.
        manifest = json.loads((args.confirmation / "manifest.json").read_text())
        candidates = list((ROOT / "experiments/phase2").glob("*confirmation*.py"))
        encoder = next(p for p in candidates if sha256(p) == manifest["helper_sha256"])
    data = ConfirmationGateDataset(args.confirmation, encoder_sha256=sha256(encoder), prior_records=fresh.records)
    data.verify_old_normalization(old.norm)
    rec = data.by_id["587c4cb6ee274da5e3a4b35d"]
    z, actions = data.episode(rec)
    t, times = 9, (0, 1, 5)
    local = QueryOracle(rec["id"], z, actions, t, scores)
    local_losses = [local.evaluate([k for j, k in enumerate(times) if j != d])[0].tolist() for d in range(3)]
    labels, detail = deletion_labels(rec, z, actions, t, times, scores)
    return {"episode_id": rec["id"], "query": rec["query"], "decision_time": t,
            "candidate_times": times, "local_target_times": [t+3, t+6],
            "use_time_target_times": [rec["query"]+3, rec["query"]+6],
            "local_horizon_losses": local_losses, "local_mean_losses": np.mean(local_losses, 1).tolist(),
            "use_time_mean_losses": labels.tolist(), **detail,
            "scope": "Posthoc selected illustrative confirmation case; not a fitting, tuning, or confirmation sample."}


def pair_ci(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair"]].append(row[key])
    vals = np.array([np.mean(v) for v in grouped.values()])
    if not len(vals):
        return None
    rng = np.random.default_rng(20261024)
    means = vals[rng.integers(len(vals), size=(2000, len(vals)))].mean(1)
    return {"mean": float(vals.mean()), "low": float(np.quantile(means, .025)),
            "high": float(np.quantile(means, .975)), "scene_pairs": len(vals), "episodes": len(rows)}


def diagnose_students(inputs, labels_by_arm, probes, records, old, old_arrays, scores):
    """Read fixed fits; no selection, new regularizer, or architecture search."""
    fits = {}
    for name, labels in labels_by_arm.items():
        predictions = np.array([probes[name].predict(x) for x in inputs])
        chosen = predictions.argmin(1)
        regret = labels[np.arange(len(labels)), chosen]-labels.min(1)
        subsets = {"all": np.arange(len(records)), "pre_use_single_cue_at_risk": np.array([
            i for i, r in enumerate(records) if r["complete_cue_loss_possible"] and r["timing_regime"] == "pre_use"])}
        fits[name] = {}
        for subset, ix in subsets.items():
            yc = labels[ix]-labels[ix].mean(1, keepdims=True)
            optimal = labels[ix] <= labels[ix].min(1, keepdims=True)+TOL
            fits[name][subset] = {"parents": len(ix), "optimal_choice_within_tolerance": int(np.sum(regret[ix] <= TOL)),
                "mean_label_regret": float(regret[ix].mean()),
                "centered_label_rms": float(np.sqrt(np.mean(yc**2))),
                "centered_fit_mse_over_zero_prediction_mse": float(np.mean((yc-predictions[ix])**2)/max(np.mean(yc**2), 1e-30)),
                "tied_minimum_parents": int(np.sum(optimal.sum(1)>1)),
                "all_three_tied_parents": int(np.sum(optimal.sum(1)==3)),
                "predicted_delete_cue": int(sum(int(chosen[i]) in records[i]["candidate_cue_indices"] for i in ix))}
    training_online = {}
    for name in ("local_cost", "use_time_cost", "future_attention"):
        values = []
        for episode_id, (z, actions) in old_arrays.items():
            rec = old.by_id[episode_id]
            online = ProbeRetention(episode_id, scores, probes[name])
            for t in range(rec["query"]+1):
                snap, _ = online.observe(z[t], executed_action=None if t == 0 else actions[t-1])
            kept = [e.timestamp for e in snap.retained]
            groups = evidence_equivalence(z, old.evidence(rec), rec["query"]-3)
            mse = float(QueryOracle(episode_id, z, actions, rec["query"], scores).evaluate(kept)[0].mean())
            values.append({"episode_id": episode_id, "mse": mse, "kept_times": kept,
                           "cue_retained": bool(all(set(g).intersection(kept) for g in groups))})
        training_online[name] = {"episodes": len(values), "cue_retained": sum(r["cue_retained"] for r in values),
                                "mse": float(np.mean([r["mse"] for r in values])), "rows": values}
    return {"training_parent_fit": fits, "training_online": training_online,
            "scope": "Posthoc inspection of unchanged fits; no parameter/architecture/hyperparameter selection",
            "tolerance": TOL}


def run(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    torch.set_num_threads(1)
    old, fresh = PortableSnapshot(args.snapshot), FreshGateDataset(args.fresh)
    fresh.verify_old_normalization(old.norm)
    scores = CausalModelScores(FollowupGateEvaluator(old, args.followup))
    meta = json.loads((args.training / "training_metadata.json").read_text())
    with np.load(args.training / "training_inputs_labels.npz") as saved:
        x, local, attention = [saved[k].copy() for k in ("inputs", "cost", "attention")]
    train_ids = {r["id"] for r in old.records if r["task"] == "gate" and r["split"] == "train"}
    if (len(meta) != 1282 or x.shape != (1282, 3, 903)
            or any(r["episode_id"] not in train_ids or r["split"] != "train" for r in meta)
            or train_ids.intersection(fresh.by_id)):
        raise ValueError("Training identities/shapes differ from archived TRAIN-only inputs")
    sources = [Path(__file__)] + [ROOT / "experiments/phase2" / f for f in (
        "diagnose_oracle_learnability.py", "memory.py", "run_literature_baselines.py",
        "run_retention_baseline_pilot.py", "causal_model_scores.py", "run_gate_followup_phase2.py",
        "train_predictor_followup.py")]
    protocol = {"version": "use-time-probe-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exploratory development experiment; not untouched confirmation or a new method claim",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "input_sha256": {str((args.training/f).relative_to(ROOT)): sha256(args.training/f)
                         for f in ("training_inputs_labels.npz", "training_metadata.json")},
        "predictor_binding": scores._evaluator.binding, "fresh_manifest_sha256": fresh.manifest_sha256,
        "teacher": "Enumerate one deletion; irreversible frozen attention continuation; average losses at query+3 and query+6",
        "reader_domain_confound": "Reader was trained only at rec.query. Local labels at earlier overflow times can be outside its training-time origin distribution. Use-time simultaneously changes target horizon and restores query-origin evaluation; these effects are not isolated.",
        "training": {"pairs": 16, "parents": 1282, "arms": ARMS, "lambda": 1., "fixed_support_mask": "zero weights for TRAIN RMS<=1e-6",
                     "permutation_seed": 17, "model_selection": False, "future_inputs_to_student": False},
        "evaluation": {"seed": 20261022, "pairs": 96, "horizons": [3, 6], "R": 4, "K": 2},
        "confirmation_case": "Previously inspected selected case only; excluded from fit and aggregate evaluation",
        "TD": "Not implemented; item survival is not a complete Markov value state or specified Bellman target"}
    write(out / "protocol.json", protocol)
    case = compare_case(args, old, fresh, scores)
    write(out / "selected_case.json", case)
    print(json.dumps({"stage": "case", **case}), flush=True)
    use_labels, records, teachers, old_arrays, started_labels = [], [], {}, {}, time.monotonic()
    feature_reproduction = []
    for i, m in enumerate(meta):
        rec = old.by_id[m["episode_id"]]
        if rec["id"] not in old_arrays:
            z, actions = old.features(rec), old.actions(rec)
            old_arrays[rec["id"]] = (z, actions)
            teachers[rec["id"]] = (AttentionContinuation(rec["id"], z, rec["query"], scores.attention),
                                    QueryOracle(rec["id"], z, actions, rec["query"], scores))
        z, actions = old_arrays[rec["id"]]
        cont, oracle = teachers[rec["id"]]
        t, times = int(m["time"]), tuple(m["candidate_times"])
        # Reconstruct a subset of stored student arrays to audit column order
        # and causal-feature identity without substituting a new representation.
        if i % 97 == 0:
            candidates, recent, past = parent_from_arrays(rec, z, actions, t, times)
            reproduced = causal_features(candidates, recent, t, scores._basis, past)
            difference = np.abs(x[i]-reproduced)
            feature_reproduction.append({"parent_index": i, "max_absolute_difference": float(difference.max()),
                                         "rtol": 1e-5, "atol": 1e-3})
            if not np.allclose(x[i], reproduced, rtol=1e-5, atol=1e-3):
                raise ValueError("Stored causal student inputs do not reproduce")
        labels, detail = deletion_labels(rec, z, actions, t, times, scores, cont, oracle)
        use_labels.append(labels)
        evidence = old.evidence(rec).tolist()
        groups = evidence_equivalence(z, evidence, rec["query"]-3)
        cue_times = sorted(set(k for group in groups for k in group))
        cue_indices = [j for j, k in enumerate(times) if k in cue_times]
        recent_has_cue = bool(set(range(t-3, t+1)).intersection(cue_times))
        records.append({**m, "query": rec["query"], "candidate_cue_indices": cue_indices,
            "recent_has_cue": recent_has_cue,
            "complete_cue_loss_possible": len(cue_indices) == 1 and not recent_has_cue,
            "timing_regime": ("pre_use" if t+6 <= rec["query"] else "post_use" if t+3 > rec["query"] else "straddle"),
            "local_cost": local[i].tolist(), "use_time_cost": labels.tolist(), "attention": attention[i].tolist(),
            "local_best_deletion": int(np.argmin(local[i])), "use_time_best_deletion": int(np.argmin(labels)), **detail})
        if (i+1) % 200 == 0:
            print(json.dumps({"stage": "labels", "parents": i+1, "seconds": time.monotonic()-started_labels}), flush=True)
    use_labels = np.asarray(use_labels)
    label_seconds = time.monotonic()-started_labels
    rng = np.random.default_rng(17)
    permutations = np.array([rng.permutation(3) for _ in meta])
    labels_by_arm = dict(zip(ARMS, (local, use_labels, attention,
        np.take_along_axis(local, permutations, 1), np.take_along_axis(use_labels, permutations, 1))))
    fit_start = time.monotonic()
    probes = {}
    for name, labels in labels_by_arm.items():
        p = RidgeProbe.fit(x, labels, meta, name)
        probes[name] = RidgeProbe(p.scale, np.where(p.scale <= 1e-6, 0., p.weights), p.label_scale, name)
    fit_seconds = time.monotonic()-fit_start
    np.savez_compressed(out / "use_time_training_labels.npz", labels=use_labels)
    np.savez_compressed(out / "probe_parameters.npz", **{name+"_"+key: getattr(p, key)
        for name, p in probes.items() for key in ("scale", "weights", "label_scale")})
    write(out / "training_parent_results.json", records)
    write(out / "student_fit_diagnosis.json", diagnose_students(x, labels_by_arm, probes, records, old, old_arrays, scores))
    rows, all_traces = [], []
    for i, rec in enumerate(fresh.records):
        z, actions = fresh.episode(rec)
        sibling, _ = fresh.episode(fresh.by_id[rec["sibling"]])
        q = rec["query"]
        oracle = QueryOracle(rec["id"], z, actions, q, scores)
        evidence = fresh.evidence(rec)
        groups = evidence_equivalence(z, evidence, q-3)
        episode_rows = []
        for name in (*ARMS, "attention", "visual_diversity"):
            online = (ProbeRetention(rec["id"], scores, probes[name]) if name in probes
                      else OnlineRetention(rec["id"], name, scores, seed=17))
            events = []
            for t in range(q+1):
                snap, event = online.observe(z[t], executed_action=None if t == 0 else actions[t-1])
                events.append(event)
            kept = [e.timestamp for e in snap.retained]
            loss, pred = oracle.evaluate(kept)
            metrics = paired_components(pred, z[q+np.array([3, 6])], sibling[q+np.array([3, 6])])
            first_loss = next((e["time"] for e in events if e["time"] > max(max(g) for g in groups)
                               and not all(set(g).intersection(e["retained_timestamps"]+e["recent_timestamps"]) for g in groups)), None)
            row = {"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"], "arm": name,
                "query": q, "kept_times": kept, "horizon_losses": loss.tolist(), **metrics,
                "cue_retained": float(all(set(g).intersection(kept) for g in groups)), "first_cue_loss": first_loss,
                "online_seconds": online.seconds, "logical_memory_maxima": online.maxima,
                "probe_parameter_bytes": probes[name].bytes if name in probes else 0}
            episode_rows.append(row)
            all_traces.extend({"episode_id": rec["id"], "arm": name, **e} for e in events)
        by_arm = {r["arm"]: r for r in episode_rows}
        for row in episode_rows:
            row["attention_reader_direction_correct_both"] = by_arm["attention"]["feature_branch_correct"] == 1.
            for control in ("local_cost", "future_attention", "shuffled_use_time_cost", "shuffled_local_cost", "attention", "visual_diversity"):
                row["minus_"+control] = row["mse"] - by_arm[control]["mse"]
        rows.extend(episode_rows)
        if (i+1) % 24 == 0:
            print(json.dumps({"stage": "evaluation", "episodes": i+1, "seconds": time.monotonic()-began}), flush=True)
    write(out / "online_results.json", rows)
    with (out / "online_trace.jsonl").open("w") as file:
        for event in all_traces:
            file.write(json.dumps(event, allow_nan=False)+"\n")
    stats = {}
    for name in (*ARMS, "attention", "visual_diversity"):
        chosen = [r for r in rows if r["arm"] == name]
        subsets = {"all": chosen, "cue_retained": [r for r in chosen if r["cue_retained"]],
            "attention_reader_direction_correct_both": [r for r in chosen if r["attention_reader_direction_correct_both"]],
            "attention_reader_direction_wrong_any": [r for r in chosen if not r["attention_reader_direction_correct_both"]]}
        fields = ("mse", "contrast_mse", "orthogonal_mse", "feature_branch_correct", "cue_retained",
                  "minus_local_cost", "minus_future_attention", "minus_shuffled_use_time_cost",
                  "minus_shuffled_local_cost", "minus_attention", "minus_visual_diversity")
        stats[name] = {subset: {key: pair_ci(values, key) for key in fields} for subset, values in subsets.items()}
    single = [r for r in records if r["complete_cue_loss_possible"]]
    early = [r for r in single if r["timing_regime"] == "pre_use"]
    train_audit = {}
    strata = [("all", records), ("single_cue_at_risk", single), ("single_cue_both_targets_prebranch", early)]
    strata += [("single_cue_at_risk_"+regime, [r for r in single if r["timing_regime"] == regime])
               for regime in ("pre_use", "straddle", "post_use")]
    for name, selected in strata:
        train_audit[name] = {"parents": len(selected),
            "different_argmin": sum(r["local_best_deletion"] != r["use_time_best_deletion"] for r in selected),
            "local_argmin_deletes_cue": sum(r["local_best_deletion"] in r["candidate_cue_indices"] for r in selected),
            "use_time_argmin_deletes_cue": sum(r["use_time_best_deletion"] in r["candidate_cue_indices"] for r in selected),
            "note": "Argmin uses oldest tie-break; consult per-parent label margins and cue multiplicity, not causal interpretation of tied choices."}
        for label_name in ("local_cost", "use_time_cost"):
            optimal = [set(np.flatnonzero(np.asarray(r[label_name]) <= min(r[label_name])+TOL).tolist()) for r in selected]
            destroy = [set(r["candidate_cue_indices"]) if r["complete_cue_loss_possible"] else set() for r in selected]
            train_audit[name][label_name+"_minimum_can_destroy_cue"] = sum(bool(a & d) for a, d in zip(optimal, destroy))
            train_audit[name][label_name+"_every_minimum_destroys_cue"] = sum(a <= d for a, d in zip(optimal, destroy))
            train_audit[name][label_name+"_tied_minimum_parents"] = sum(len(a) > 1 for a in optimal)
        train_audit[name]["disjoint_minimum_sets"] = sum(not set(np.flatnonzero(np.asarray(r["local_cost"]) <= min(r["local_cost"])+TOL)).intersection(
            np.flatnonzero(np.asarray(r["use_time_cost"]) <= min(r["use_time_cost"])+TOL)) for r in selected)
    summary = {"status": "completed_exploratory_use_time_probe", "elapsed_seconds": time.monotonic()-began,
        "label_seconds": label_seconds, "fit_seconds": fit_seconds, "new_gpu_seconds": 0,
        "training_parent_audit": train_audit, "statistics": stats,
        "teacher_continuation_states": sum(len(c.cache) for c,o in teachers.values()),
        "teacher_attention_calls": sum(c.calls for c,o in teachers.values()),
        "feature_width": x.shape[-1], "supported_features": int(np.count_nonzero(probes["local_cost"].scale>1e-6)),
        "stored_feature_reproduction_checks": feature_reproduction,
        "limits": ["Known seed-20261022 development data; no independent confirmation",
                   "Frozen reader feature-MSE utility may disagree with task evidence value",
                   "Fixed attention continuation is reference-policy dependent, not an optimal future policy",
                   "Offline teacher uses recorded query timing/future; causal students do not",
                   "Subgroups based on attention correctness are posthoc descriptions, not causal effects",
                   "Same slots and matched probe parameters; total model/compute budget differs from heuristic baselines"]}
    if any(sha256(ROOT/p) != h for p,h in protocol["source_sha256"].items()):
        raise ValueError("Source changed during run")
    if any(sha256(ROOT/p) != h for p,h in protocol["input_sha256"].items()):
        raise ValueError("Pinned training input changed during run")
    summary["result_sha256"] = {p.name:sha256(p) for p in out.iterdir() if p.is_file()}
    write(out / "summary.json", summary)
    print(json.dumps({"stage": "complete", "elapsed_seconds": summary["elapsed_seconds"],
        "training_parent_audit": train_audit,
        "online": {name: {key: stats[name]["all"][key]["mean"] for key in ("mse", "cue_retained", "feature_branch_correct")} for name in stats}}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT/"build/hpc-20261020/development-assets")
    parser.add_argument("--followup", type=Path, default=ROOT/"build/history-diagnosis-20261020/followup-blas1")
    parser.add_argument("--training", type=Path, default=ROOT/"build/literature-diagnosis-20261022/learnability-v2")
    parser.add_argument("--fresh", type=Path, default=ROOT/"build/literature-baselines-20261022/fresh-data")
    parser.add_argument("--confirmation", type=Path, default=ROOT/"build/literature-confirmation-20261023/data")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
