"""Matched, no-new-information prefix intervention for WR startup diagnostics.

This is a diagnostic counterfactual, not a replacement or retuned baseline.
Frozen runtime source is imported unchanged. All policies receive the identical
four-copy prefix, original relative times/actions and targets are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
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
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.run_literature_baselines import (
    QueryOracle, evidence_equivalence, online_policy, overflow_parent,
)

PREFIX = 4
POLICIES = ("wr", "fifo", "attention", "visual_diversity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def initial_prefix(z, actions, query, evidence, *, length=PREFIX):
    """Only reuse the first observed frame; add zero executed-action intervals."""
    if (not isinstance(length, int) or isinstance(length, bool) or length < 0
            or z.dtype != np.float32 or z.ndim != 2
            or actions.dtype != np.float32 or actions.shape != (len(z)-1, 2)
            or not np.isfinite(z).all() or not np.isfinite(actions).all()
            or not 6 <= query < len(z)-6):
        raise ValueError("Invalid prefix input")
    evidence = np.asarray(evidence, dtype=np.int64)
    if evidence.ndim != 1 or np.any(evidence < 0) or np.any(evidence >= query-3):
        raise ValueError("Evidence is evaluation-only historical timing")
    values = np.concatenate([np.repeat(z[:1], length, axis=0), z], axis=0)
    commands = np.concatenate([np.zeros((length, 2), np.float32), actions], axis=0)
    assert np.array_equal(values[length:], z)
    assert np.array_equal(commands[length:], actions)
    assert np.array_equal(values[query+length+np.array([3, 6])], z[query+np.array([3, 6])])
    return values, commands, query+length, evidence+length


def replay(record, z, actions, evidence, scores, policy, oracle, prefix):
    q = record["query"] + prefix
    online = online_policy(record["id"], policy, scores)
    groups = evidence_equivalence(z, evidence, q-3)
    events, parent, tracker_rows = [], None, []
    for t in range(q+1):
        if t == q:
            parent = overflow_parent(online.stream.snapshot, z[t], record["id"], t)
        _, event = online.observe(z[t], executed_action=None if t == 0 else actions[t-1])
        events.append(event)
        if policy == "wr":
            tracker_rows.append({"time": t, "original_time": t-prefix,
                "cue_resident": all(any(x in event["retained_timestamps"]+event["recent_timestamps"] for x in g) for g in groups),
                "retained": event["retained_timestamps"], "recent": event["recent_timestamps"],
                "deleted": event["deleted_timestamp"], "scalars": event["wr_scalar_records"]})
    kept = [e.timestamp for e in online.stream.snapshot.retained]
    losses, prediction = oracle.evaluate(kept)
    local = oracle.best([e.timestamp for e in parent[0]])
    survived = all(any(t in kept for t in group) for group in groups)
    first_loss = [next((event["time"] for event in events if event["time"] > max(group)
                       and not set(group).intersection(event["retained_timestamps"]+event["recent_timestamps"])), None)
                  for group in groups]
    first_event = None
    onset_record = None
    if policy == "wr":
        cue_times = set(groups[0])
        first_event = next((e["time"] for e in tracker_rows
            if any(r["event"] and r["time"] in cue_times for r in e["scalars"])), None)
        onset_record = next(r for event in tracker_rows if event["time"] == int(evidence[0])
                            for r in event["scalars"] if r["time"] == int(evidence[0]))
    return {"episode_id": record["id"], "pair": record["pair"],
        "split": record["split"], "condition": record.get("condition", "old_train"),
        "policy": policy, "prefix": prefix, "original_query": record["query"], "query": q,
        "cue_original_time": int(evidence[0])-prefix, "cue_times": groups,
        "kept_times": kept, "kept_original_times": [t-prefix for t in kept],
        "prefix_item_retained": any(t < prefix for t in kept),
        "cue_survived": survived, "cue_first_loss": first_loss,
        "cue_first_confirmed_event": first_event, "cue_onset_record": onset_record,
        "feature_mse": losses.tolist(), "mean_mse": float(losses.mean()),
        "local_oracle": local, "local_regret": float(losses.mean())-local["mean_mse"],
        "logical_memory_maxima": online.maxima, "online_seconds": online.seconds}, prediction, tracker_rows


def paired_ci(values, groups, *, repetitions=10000):
    """Sibling means first; scene pairs are the bootstrap units."""
    keys = sorted(set(groups))
    means = np.array([np.mean([v for v, g in zip(values, groups) if g == key]) for key in keys])
    rng = np.random.default_rng(20261022)
    boot = means[rng.integers(len(means), size=(repetitions, len(means)))].mean(1)
    return {"mean": float(means.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "pairs": len(keys), "rows": len(values)}


def summarize(rows):
    indexed = {(r["episode_id"], r["policy"], r["prefix"]): r for r in rows}
    result = []
    for condition in ("old_train", "id", "long_gap", "distractor_shift"):
        for policy in POLICIES:
            cold = [r for r in rows if r["condition"] == condition and r["policy"] == policy and not r["prefix"]]
            warm = [indexed[(r["episode_id"], policy, PREFIX)] for r in cold]
            groups = [r["pair"] for r in cold]
            result.append({"condition": condition, "policy": policy, "episodes": len(cold),
                "cold_mse": float(np.mean([r["mean_mse"] for r in cold])),
                "prefix_mse": float(np.mean([r["mean_mse"] for r in warm])),
                "prefix_minus_cold_mse": paired_ci([b["mean_mse"]-a["mean_mse"] for a,b in zip(cold,warm)], groups),
                "cold_cue_survived": sum(r["cue_survived"] for r in cold),
                "prefix_cue_survived": sum(r["cue_survived"] for r in warm),
                "same_original_kept_set": sum(a["kept_original_times"] == b["kept_original_times"] for a,b in zip(cold,warm)),
                "prefix_items_retained": sum(r["prefix_item_retained"] for r in warm)})
    wr_strata = []
    for split in ("train", "fresh_development"):
        for early in (True, False):
            cold = [r for r in rows if r["split"] == split and r["policy"] == "wr" and not r["prefix"]
                    and (r["cue_original_time"] <= 3) == early]
            warm = [indexed[(r["episode_id"], "wr", PREFIX)] for r in cold]
            wr_strata.append({"split": split, "early_cue_t2_t3": early, "episodes": len(cold),
                "cold_cue_survived": sum(r["cue_survived"] for r in cold),
                "prefix_cue_survived": sum(r["cue_survived"] for r in warm),
                "cold_threshold_missing": sum(r["cue_onset_record"]["threshold"] is None for r in cold),
                "prefix_threshold_missing": sum(r["cue_onset_record"]["threshold"] is None for r in warm),
                "cold_confirmed_events": sum(r["cue_first_confirmed_event"] is not None for r in cold),
                "prefix_confirmed_events": sum(r["cue_first_confirmed_event"] is not None for r in warm),
                "prefix_minus_cold_mse": paired_ci([b["mean_mse"]-a["mean_mse"] for a,b in zip(cold,warm)],
                                                    [r["pair"] for r in cold]) if cold else None})
    return {"by_condition_policy": result, "wr_strata": wr_strata}


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Require a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    source_names = ("diagnose_wr_startup.py", "literature_baselines.py", "run_literature_baselines.py",
                    "run_retention_baseline_pilot.py", "causal_model_scores.py", "memory.py",
                    "run_fresh_literature_diagnostic.py", "run_gate_followup_phase2.py", "portable_snapshot.py")
    source_hashes = {"experiments/phase2/"+name: sha(ROOT / "experiments/phase2" / name) for name in source_names}
    protocol = {"version": "wr-shared-initial-prefix-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Missing startup score/peak history prevents early-cue event admission",
        "scope": "causal no-new-information intervention on reused development data; not new held-out confirmation",
        "prefix_copies": PREFIX, "prefix_rule": "Repeat first already-observed feature four times, prepend four zero action intervals, shift all original timestamps/query by four",
        "prefix_choice": "fixed from Gaussian min_history2 plus MAD min_history2; no outcome tuning or sweep",
        "policies": POLICIES, "recent": 4, "capacity": 2, "wr_parameters_unchanged": True,
        "horizons": [3,6], "loss": "one retained set; mean normalized feature MSE across h3/h6",
        "episodes": {"old_train":32, "fresh_development":192}, "new_visual_encoding":False,
        "new_training":False, "model_changes":False, "utility_scorer_training":False,
        "evaluation_only": ["cue timestamps", "future targets", "local deletion oracle"],
        "paired_bootstrap": {"unit":"scene pair after sibling mean", "repetitions":10000,"seed":20261022},
        "source_sha256":source_hashes,
        "fresh_manifest_sha256":sha(args.fresh / "manifest.json"),
        "comparison_boundaries": ["Prefix changes initialization and number of decisions; shared-policy controls quantify these effects",
            "Original item relative ages/actions and query targets are unchanged",
            "Prefix is a diagnostic counterfactual, not a retuned replacement WR or an improvement claim for original WR paper"]}
    write(output / "protocol.json", protocol)
    protocol_hash = sha(output / "protocol.json")
    old = PortableSnapshot(args.snapshot)
    fresh = FreshGateDataset(args.fresh)
    fresh.verify_old_normalization(old.norm)
    evaluator = FollowupGateEvaluator(old, args.followup)
    scores = CausalModelScores(evaluator)
    train = [r for r in old.records if r["task"] == "gate" and r["split"] == "train"]
    assert len(train) == 32 and len(fresh.records) == 192
    assert not set(r["pair"] for r in train) & set(r["pair"] for r in fresh.records)
    rows, predictions, targets, invariance, example_traces = [], [], [], [], {}
    with gzip.open(output / "wr_trace.jsonl.gz", "wt") as tracefile:
        for position, rec in enumerate(train + fresh.records):
            z, actions = (old.features(rec), old.actions(rec)) if rec["split"] == "train" else fresh.episode(rec)
            evidence = old.evidence(rec) if rec["split"] == "train" else fresh.evidence(rec)
            pref_z, pref_a, pref_q, pref_e = initial_prefix(z, actions, rec["query"], evidence)
            oracles = {0:QueryOracle(rec["id"], z, actions, rec["query"], scores),
                       PREFIX:QueryOracle(rec["id"], pref_z, pref_a, pref_q, scores)}
            for prefix, values, commands, cue in ((0,z,actions,evidence), (PREFIX,pref_z,pref_a,pref_e)):
                for policy in POLICIES:
                    row, pred, traces = replay(rec, values, commands, cue, scores, policy, oracles[prefix], prefix)
                    rows.append(row);predictions.append(pred);targets.append(z[rec["query"]+np.array([3,6])])
                    if traces:
                        for event in traces:
                            tracefile.write(json.dumps({"episode_id":rec["id"],"prefix":prefix,**event},allow_nan=False)+"\n")
                        example_traces[(rec["id"],prefix)] = traces
            # No selection/model effect from the global clock shift itself.
            for row in rows[-8:-4]:
                cold_loss, cold_pred = oracles[0].evaluate(row["kept_times"])
                shifted_loss, shifted_pred = oracles[PREFIX].evaluate([t+PREFIX for t in row["kept_times"]])
                max_error = float(np.max(np.abs(cold_pred.astype(np.float64)-shifted_pred.astype(np.float64))))
                assert max_error == 0 and np.array_equal(cold_loss,shifted_loss), (rec["id"],policy,max_error)
                invariance.append({"episode_id":rec["id"],"policy":row["policy"],"prediction_max_abs_error":max_error})
            if (position+1) % 32 == 0:
                print(json.dumps({"episodes_completed":position+1,"total":224}),flush=True)
    predictions, targets = np.stack(predictions), np.stack(targets)
    recomputed = np.mean((predictions.astype(np.float64)-targets.astype(np.float64))**2,axis=-1)
    assert np.array_equal(recomputed,np.array([r["feature_mse"] for r in rows]))
    np.savez_compressed(output / "prediction_vectors.npz",predictions=predictions,targets=targets)
    write(output / "episode_results.json",rows)
    write(output / "shift_invariance.json",invariance)
    selected = []
    for early in (True,False):
        pool=[r for r in rows if r["split"]=="fresh_development" and r["policy"]=="wr" and r["prefix"]==0
              and (r["cue_original_time"]<=3)==early]
        # Fixed descriptive selection, not maximum error selection.
        rec=min(pool,key=lambda r:(r["cue_original_time"],r["episode_id"]))
        selected.append({"early_cue":early,"episode_id":rec["episode_id"],
                         "selection":"lowest cue time, then episode ID within startup stratum",
                         "rows":[r for r in rows if r["episode_id"]==rec["episode_id"]],
                         "wr_traces":{"cold":example_traces[(rec["episode_id"],0)],
                                      "prefix":example_traces[(rec["episode_id"],PREFIX)]}})
    write(output / "paired_examples.json",selected)
    summary={"status":"completed_fixed_protocol_wr_startup_diagnostic","elapsed_seconds":time.monotonic()-started,
        "protocol_sha256":protocol_hash,"device":"cpu","gpu_seconds":0,"source_sha256":source_hashes,
        "episodes":224,"episode_policy_arm_rows":len(rows),"frozen_model_shift_checks":len(invariance),
        "shift_prediction_max_abs_error":max(r["prediction_max_abs_error"] for r in invariance),
        "saved_prediction_mse_recomputation_max_error":0.,**summarize(rows),
        "old_input_audit":old.audit(),"fresh_input_audit":fresh.audit(),"predictor_audit":scores.audit(),
        "new_training_calls":0,"old_test_arrays_opened":0}
    assert sha(output / "protocol.json")==protocol_hash
    assert all(sha(ROOT/path)==expected for path,expected in source_hashes.items())
    write(output / "summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("status","elapsed_seconds","by_condition_policy","wr_strata")},indent=2))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,default=ROOT/"build/hpc-20261020/development-assets")
    parser.add_argument("--fresh",type=Path,default=ROOT/"build/literature-baselines-20261022/fresh-data")
    parser.add_argument("--followup",type=Path,default=ROOT/"build/history-diagnosis-20261020/followup-blas1")
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
