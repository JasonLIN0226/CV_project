"""Frozen-rule, posthoc Gate2 prediction-consensus analysis; no model calls.

Primary rule: for the three genuine terminal deletion predictions p_i[h,d],
compute D_ij=mean_(h,d)((p_i-p_j)^2), and delete argmin_i sum_j D_ij.
The diagnostic proxy applies exactly the same medoid rule to the four saved
distances from each prediction to the four REAL sibling future targets. That
proxy uses privileged outcome information and is not a deployable policy.

Both rules use float64 arithmetic and exact np.argmin (first chronological drop
index for exact ties). Exact and 1e-6 tolerant minimizer sets are saved separately;
the tolerance does not change the selected action. No variants are searched.
Only archived native predictions are used. Those predictions were produced with
the actual recorded future planned actions. This script does not implement an
online causal interface, test counterfactual action plans, rerun a reader, or
measure three-forward-pass deployment latency. All episodes are already-seen
development data. Family bootstrap: 2,000 draws, seed20261025.

First --freeze writes the rules and source/input hashes without loading outcomes.
Then --execute checks that receipt before loading predictions or result rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

STYLES = ("benign_duplicate", "distant_style_duplicate")
NEW_RULES = ("full_prediction_medoid", "future_prototype_distance_medoid")
BASELINES = ("fifo", "terminal_uniform", "visual_diversity", "reader_attention")
INPUT_FILES = ("evaluation_predictions.npz", "evaluation_rows.json", "terminal_decisions.json",
               "data/records.json", "data/actions.npz", "summary.json")
TOL = 1e-6


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def declaration():
    return {
        "name": "gate2-prediction-consensus-posthoc-v1",
        "status": "posthoc_rule_lock_on_already_inspected_development_data",
        "primary_rule": "D_ij=mean over both horizons and all features of (native_prediction_i-native_prediction_j)^2; choose argmin_i sum_j D_ij",
        "privileged_proxy_rule": "Apply identical squared-distance medoid to each prediction's four archived joint future-prototype MSE coordinates",
        "primary_selection_inputs": "Only the three archived native prediction vectors; no targets, bits, roles, loss or metadata in the medoid function",
        "proxy_selection_inputs": "Four REAL future-target distance coordinates; privileged diagnostic, not deployable",
        "candidate_order": "native_drop_0,native_drop_1,native_drop_2, chronological q-6,q-5,q-4",
        "tie_break": "Exact float64 np.argmin, first candidate on exact ties",
        "reported_tie_sets": {"exact": "score == min(score)", "tolerant": "score <= min(score)+1e-6", "tolerance_changes_choice": False},
        "baseline_arms": list(BASELINES), "horizons": [3, 6],
        "evaluation": "same native joint-horizon losses and semantic mask as frozen Gate2; no per-horizon action mixing",
        "attention_review_check": "Report all exact-unique attention minima separately from singleton B uniquely below both duplicate cues",
        "uniform_reference": "One uniformly random deletion per family, shared by both styles and four siblings; semantic-loss probability1/3; independent reference n=24 families, not192 episodes",
        "subgroups": "all, each style, each duplicated physical axis, each style-by-axis; no subgroup filtering",
        "bootstrap": {"cluster": "family", "draws": 2000, "seed": 20261025, "interval": "percentile95; descriptive exploratory"},
        "no_variant_sweep": True, "new_model_calls": 0, "new_training": False,
        "forecast_actions": "Archived predictions use actual recorded future planned controls; no deployed causal-policy implementation or action-plan intervention is tested",
        "scope": "Existing DEV only, no independent confirmation, no model changes"}


def medoid(vectors):
    """Pure selection function: no outcome/semantic information is accepted."""
    values = np.asarray(vectors, np.float64).reshape(3, -1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite medoid input")
    delta = values[:, None] - values[None, :]
    distances = np.mean(delta*delta, axis=2)
    scores = distances.sum(1)
    return {"drop": int(scores.argmin()), "pairwise_mse": distances.tolist(),
            "sum_distance_by_drop": scores.tolist(),
            "exact_min_set": np.flatnonzero(scores == scores.min()).tolist(),
            "tolerant_min_set": np.flatnonzero(scores <= scores.min()+TOL).tolist(),
            "second_best_score_margin": float(np.sort(scores)[1]-scores.min())}


def interval(values, families):
    grouped = defaultdict(list)
    for value, family in zip(values, families):
        grouped[family].append(float(value))
    means = np.asarray([np.mean(grouped[k]) for k in sorted(grouped)])
    rng = np.random.default_rng(20261025)
    samples = means[rng.integers(0, len(means), (2000, len(means)))].mean(1)
    return {"episodes": len(values), "families": len(means), "mean": float(np.mean(values)),
            "equal_family_mean": float(means.mean()), "family_cluster_percentile95": np.quantile(samples, [.025, .975]).tolist()}


def uniform_reference(rows):
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r["policies"]["terminal_uniform"]["semantic_loss"])
    if any(len(set(v)) != 1 or len(v) != 8 for v in by_family.values()):
        raise ValueError("Uniform family coupling differs from locked design")
    n = len(by_family)
    observed = sum(bool(v[0]) for v in by_family.values())
    probabilities = np.asarray([math.comb(n, k)*(1/3)**k*(2/3)**(n-k) for k in range(n+1)])
    cdf = probabilities.cumsum()
    central = [int(np.searchsorted(cdf, p)) for p in (.025, .975)]
    return {"independent_reference_unit": "family; eight episodes share each draw",
            "family_count": n, "observed_semantic_loss_families": observed,
            "observed_semantic_loss_episodes": sum(r["policies"]["terminal_uniform"]["semantic_loss"] for r in rows),
            "observed_fraction": observed/n, "theoretical_fraction": 1/3,
            "expected_semantic_loss_families": n/3, "expected_semantic_loss_episodes": len(rows)/3,
            "reference_binomial_central95_count": central,
            "reference_probability_count_at_most_observed": float(cdf[observed]),
            "interpretation": "Descriptive calibration against uniformly random deletion; fixed hash draws are not192 independent Bernoulli trials"}


def freeze(run, out):
    out.mkdir(parents=True, exist_ok=True)
    path = out/"rule_lock.json"
    if path.exists():
        raise ValueError("Rule lock already exists; do not overwrite")
    lock = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "declaration": declaration(), "script_sha256": sha(__file__),
            "input_sha256": {f: sha(run/f) for f in INPUT_FILES},
            "outcomes_loaded_during_freeze": False}
    write(path, lock)
    print(json.dumps({"status": "posthoc_rules_frozen_before_new_computation", "rule_lock_sha256": sha(path)}))


def execute(run, out):
    lock = read(out/"rule_lock.json")
    if lock["script_sha256"] != sha(__file__) or lock["declaration"] != declaration():
        raise ValueError("Source or rule changed after lock")
    if any(sha(run/f) != expected for f, expected in lock["input_sha256"].items()):
        raise ValueError("Archived input changed after lock")
    old_rows = read(run/"evaluation_rows.json")
    decisions = read(run/"terminal_decisions.json")
    records = {r["id"]: r for r in read(run/"data/records.json")}
    with np.load(run/"evaluation_predictions.npz", allow_pickle=False) as data:
        predictions = data["predictions"]
    lookup = {(r["id"], r["arm"]): (i, r) for i, r in enumerate(old_rows)}
    rows = []
    for d in decisions:
        record = records[d["id"]]
        native = [lookup[(d["id"], "native_drop_"+str(i))] for i in range(3)]
        vectors = predictions[[i for i, _ in native]]
        coordinates = np.asarray([r["class_distances"] for _, r in native], np.float64)
        full, proxy = medoid(vectors), medoid(coordinates)
        semantic = d["semantic_correct_delete_set"]
        singleton = next(i for i in range(3) if i not in semantic)
        losses = np.asarray(d["losses"], np.float64)
        if not np.array_equal(losses, [r["mse"] for _, r in native]):
            raise ValueError("Native losses do not match archive")
        attention = np.asarray(d["attention"], np.float64)
        attention_min = np.flatnonzero(attention == attention.min()).tolist()
        attention_tolerant = np.flatnonzero(attention <= attention.min()+TOL).tolist()
        row = {"id": d["id"], "family": d["family"], "quartet": d["quartet"], "bits": d["bits"],
               "condition": d["condition"], "duplicate_axis": record["scene"]["duplicate_axis"],
               "roles": record["scene"]["roles"], "candidate_times": record["evidence"],
               "native_prediction_row_indices": [i for i, _ in native], "losses": losses.tolist(),
               "semantic_correct_delete_set": semantic, "singleton_B_drop": singleton,
               "full_prediction_medoid": full, "future_prototype_distance_medoid": proxy,
               "future_prototype_distance_coordinates": coordinates.tolist(),
               "attention_exact_min_set": attention_min, "attention_tolerant_min_set": attention_tolerant,
               "attention_singleton_strictly_unique_minimum": attention_min == [singleton],
               "attention_singleton_minus_best_duplicate_score": float(attention[singleton]-attention[semantic].min()),
               "policies": {}}
        choices = dict(d["choices"], full_prediction_medoid=full["drop"], future_prototype_distance_medoid=proxy["drop"])
        for policy in BASELINES+NEW_RULES:
            drop = choices[policy]
            row["policies"][policy] = {"drop": drop, "semantic_loss": drop not in semantic,
                                       "mse": float(losses[drop]), "regret": float(losses[drop]-losses.min()),
                                       "feature_class_correct": native[drop][1]["feature_class_correct"]}
        rows.append(row)
    conditions = ["all"]+list(STYLES)+["axis/"+a for a in ("a", "b")]+[s+"/"+a for s in STYLES for a in ("a", "b")]
    summary = {}
    for condition in conditions:
        rr = [r for r in rows if condition == "all" or r["condition"] == condition
              or "axis/"+r["duplicate_axis"] == condition or r["condition"]+"/"+r["duplicate_axis"] == condition]
        families = [r["family"] for r in rr]
        policies = {}
        for policy in BASELINES+NEW_RULES:
            pp = [r["policies"][policy] for r in rr]
            policies[policy] = {"semantic_loss_count": sum(p["semantic_loss"] for p in pp),
                                "feature_class_correct_count": sum(p["feature_class_correct"] for p in pp),
                                "mean_mse": float(np.mean([p["mse"] for p in pp])),
                                "semantic_loss": interval([p["semantic_loss"] for p in pp], families),
                                "regret": interval([p["regret"] for p in pp], families),
                                "positive_regret_count_over1e-6": sum(p["regret"] > TOL for p in pp),
                                "regret_quantiles_0_50_95_100": np.quantile([p["regret"] for p in pp], [0, .5, .95, 1]).tolist()}
            if policy in NEW_RULES:
                policies[policy].update(exact_tie_count=sum(len(r[policy]["exact_min_set"]) > 1 for r in rr),
                                        tolerant_tie_count=sum(len(r[policy]["tolerant_min_set"]) > 1 for r in rr),
                                        exact_min_set_all_semantic_count=sum(set(r[policy]["exact_min_set"]) <= set(r["semantic_correct_delete_set"]) for r in rr),
                                        tolerant_min_set_all_semantic_count=sum(set(r[policy]["tolerant_min_set"]) <= set(r["semantic_correct_delete_set"]) for r in rr))
        comparisons = {}
        for other in BASELINES+("future_prototype_distance_medoid",):
            values = [r["policies"]["full_prediction_medoid"]["mse"]-r["policies"][other]["mse"] for r in rr]
            comparisons["full_prediction_medoid_minus_"+other+"_mse"] = interval(values, families)
        summary[condition] = {"episodes": len(rr), "families": len(set(families)), "policies": policies,
                              "attention_any_strict_unique_minimum_count": sum(len(r["attention_exact_min_set"]) == 1 for r in rr),
                              "attention_singleton_strictly_unique_minimum_count": sum(r["attention_singleton_strictly_unique_minimum"] for r in rr),
                              "full_and_proxy_same_action_count": sum(r[NEW_RULES[0]]["drop"] == r[NEW_RULES[1]]["drop"] for r in rr),
                              "paired_comparisons": comparisons}
    result = {"status": "completed_posthoc_no_new_models", "rule_lock_sha256": sha(out/"rule_lock.json"),
              "script_sha256": sha(__file__), "input_sha256": lock["input_sha256"],
              "summary": summary, "uniform_reference": uniform_reference(rows),
              "model_calls": 0, "training": False, "independent_confirmation": False,
              "deployment_causality_implemented": False,
              "limitations": ["All192episodes belong to previously inspected DEV; this is a new posthoc diagnostic.",
                              "The future-prototype coordinate proxy uses actual outcome features and is privileged.",
                              "Full-vector selection uses only archived predictions, but those predictions already use actual recorded planned future controls.",
                              "Three-native-prediction online compute and causal action-plan access have not been implemented or timed.",
                              "Family bootstrap is exploratory; no multiplicity-adjusted confirmatory claim is made.",
                              "Agreement among prediction variants is not proof that their shared prediction is correct."]}
    write(out/"episode_results.json", rows)
    write(out/"summary.json", result)
    write(out/"receipt.json", {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                              "files": {f: sha(out/f) for f in ("rule_lock.json", "episode_results.json", "summary.json")}})
    compact = {p: {"semantic_loss": summary["all"]["policies"][p]["semantic_loss_count"],
                    "mean_regret": summary["all"]["policies"][p]["regret"]["mean"]} for p in BASELINES+NEW_RULES}
    print(json.dumps({"status": result["status"], "episodes": len(rows), "policies": compact,
                      "full_and_proxy_same_action_count": summary["all"]["full_and_proxy_same_action_count"]}))


def self_test():
    rule = medoid(np.asarray([[0.], [1.], [4.]]))
    assert rule["drop"] == 1 and rule["sum_distance_by_drop"] == [17., 10., 25.]
    assert medoid(np.asarray([[13.], [14.], [17.]]))["drop"] == 1
    tied = medoid(np.zeros((3, 2, 5)))
    assert tied["drop"] == 0 and tied["exact_min_set"] == [0, 1, 2]
    print(json.dumps({"status": "passed_synthetic_medoid_sanity", "model_calls": 0}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--output", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.run is None or args.output is None:
        p.error("--run and --output required")
    elif args.freeze:
        freeze(args.run.resolve(), args.output.resolve())
    else:
        execute(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
