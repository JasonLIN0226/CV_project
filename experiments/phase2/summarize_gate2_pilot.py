"""Posthoc descriptive Gate2 decomposition from saved rows, with no model calls.

This does not change the frozen experiment or its stop decision. It separates
regret from losing the complementary axis from regret between two semantically
adequate native retained sets. The latter can reflect copy style and timestamp
effects, and must not be described as missing evidence. Four siblings and two
styles are clustered by their common family for exploratory intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

STYLES = ("benign_duplicate", "distant_style_duplicate")
POLICIES = ("fifo", "terminal_uniform", "visual_diversity", "reader_attention")


def read(path):
    return json.loads(path.read_text())


def hashes(run, files):
    return {f: hashlib.sha256((run/f).read_bytes()).hexdigest() for f in files}


def family_interval(values, families, *, seed=20261025, repetitions=2000):
    grouped = defaultdict(list)
    for value, family in zip(values, families):
        grouped[family].append(float(value))
    means = np.asarray([np.mean(grouped[k]) for k in sorted(grouped)])
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), (repetitions, len(means)))].mean(1)
    return {"family_count": len(means), "episode_count": len(values),
            "episode_mean": float(np.mean(values)), "equal_family_mean": float(means.mean()),
            "family_cluster_percentile_95": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed}


def stats(values):
    a = np.asarray(values, float)
    return {"n": len(a), "mean": float(a.mean()), "min": float(a.min()),
            "p05": float(np.quantile(a, .05)), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def interpretation(run, summary, decisions, records):
    out = {"scope": "descriptive decomposition of the locked development pilot; no new model or selection",
           "source_sha256": hashes(run, ["summary.json", "terminal_decisions.json", "data/records.json"]),
           "by_style": {}, "by_style_axis": {}}
    for style in ("all",)+STYLES:
        dd = [d for d in decisions if style == "all" or d["condition"] == style]
        cs = [c for c in summary["quartet_consistency"] if style == "all" or c["condition"] == style]
        res = {"episodes": len(dd), "quartets": len(cs),
               "same_exact_min_set": sum(c["same_exact_min_set"] for c in cs),
               "same_tolerant_min_set": sum(c["same_tolerant_min_set"] for c in cs),
               "common_optimum": sum(bool(c["common_optimal_action"]) for c in cs),
               "oracle_deleted_role_counts": dict(Counter(records[d["id"]]["scene"]["roles"][d["oracle_exact_min_delete_set"][0]] for d in dd)),
               "policies": {}}
        for policy in POLICIES:
            groups = {}
            for correctness in (True, False):
                pp = [d["policies"][policy] for d in dd if d["policies"][policy]["semantic_correct"] == correctness]
                total = sum(d["policies"][policy]["regret"] for d in dd)
                groups["semantic_correct" if correctness else "semantic_wrong"] = {
                    "n": len(pp), "class_correct": sum(p["feature_class_correct"] for p in pp),
                    "positive_regret": sum(p["regret"] > 1e-6 for p in pp),
                    "mean_regret": float(np.mean([p["regret"] for p in pp])) if pp else None,
                    "share_of_total_regret": sum(p["regret"] for p in pp)/total if total else None}
            res["policies"][policy] = groups
        out["by_style"][style] = res
    for style in STYLES:
        for axis in ("a", "b"):
            dd = [d for d in decisions if d["condition"] == style and records[d["id"]]["scene"]["duplicate_axis"] == axis]
            out["by_style_axis"][style+"/"+axis] = {"n": len(dd), "policies": {
                policy: {"semantic_wrong": sum(not d["policies"][policy]["semantic_correct"] for d in dd),
                         "mean_regret": float(np.mean([d["policies"][policy]["regret"] for d in dd])),
                         "class_correct": sum(d["policies"][policy]["feature_class_correct"] for d in dd)}
                for policy in ("visual_diversity", "reader_attention")}}
    return out


def decomposition(run, decisions, records):
    rows = []
    for d in decisions:
        semantic = d["semantic_correct_delete_set"]
        bad = next(i for i in range(3) if i not in semantic)
        losses = np.asarray(d["losses"])
        best = float(losses[semantic].min())
        # This identity is specific to the archived finding; do not silently
        # assume a reader whose oracle might prefer losing an independent bit.
        if abs(best-losses.min()) >= 1e-15:
            raise ValueError("Semantic complement is not oracle-optimal for " + d["id"])
        row = {"id": d["id"], "quartet": d["quartet"], "family": d["family"],
               "condition": d["condition"], "bits": d["bits"],
               "duplicate_axis": records[d["id"]]["scene"]["duplicate_axis"],
               "roles": records[d["id"]]["scene"]["roles"], "semantic_delete_set": semantic,
               "best_complementary_delete_set": np.asarray(semantic)[np.flatnonzero(losses[semantic] == best)].tolist(),
               "best_complementary_mse": best, "redundant_drop": bad,
               "redundant_minus_best_complementary_mse": float(losses[bad]-best),
               "absolute_complementary_loss_margin": float(abs(losses[semantic[0]]-losses[semantic[1]])),
               "policies": {}}
        for policy, p in d["policies"].items():
            redundant = p["drop"] == bad
            value = {"chose_redundant_pair": redundant,
                     "redundancy_component": float(losses[bad]-best) if redundant else 0.,
                     "complementary_choice_component": 0. if redundant else float(losses[p["drop"]]-best),
                     "regret": p["regret"]}
            if abs(value["redundancy_component"]+value["complementary_choice_component"]-p["regret"]) >= 1e-14:
                raise ValueError("Regret decomposition failed for " + d["id"])
            row["policies"][policy] = value
        rows.append(row)
    quartets = []
    for quartet in dict.fromkeys(r["quartet"] for r in rows):
        rr = [r for r in rows if r["quartet"] == quartet]
        sets = [set(r["best_complementary_delete_set"]) for r in rr]
        quartets.append({"quartet": quartet, "family": rr[0]["family"], "condition": rr[0]["condition"],
                         "duplicate_axis": rr[0]["duplicate_axis"],
                         "common_best_complementary_delete_set": sorted(set.intersection(*sets)),
                         "same_best_set_across_siblings": all(s == sets[0] for s in sets),
                         "complementary_margin_min": min(r["absolute_complementary_loss_margin"] for r in rr),
                         "complementary_margin_max": max(r["absolute_complementary_loss_margin"] for r in rr),
                         "bits_and_best_roles": [{"bits": r["bits"], "delete_roles": [r["roles"][i] for i in r["best_complementary_delete_set"]],
                                                  "margin": r["absolute_complementary_loss_margin"]} for r in rr]})
    summary = {}
    for condition in ("all",)+STYLES:
        rr = [r for r in rows if condition == "all" or r["condition"] == condition]
        qs = [q for q in quartets if condition == "all" or q["condition"] == condition]
        families = [r["family"] for r in rr]
        result = {"episodes": len(rr), "quartets": len(qs),
                  "quartets_with_common_best_complementary_action": sum(bool(q["common_best_complementary_delete_set"]) for q in qs),
                  "absolute_complementary_loss_margin": stats([r["absolute_complementary_loss_margin"] for r in rr]),
                  "redundant_minus_best_complementary_mse": stats([r["redundant_minus_best_complementary_mse"] for r in rr]),
                  "policies": {}}
        for policy in rr[0]["policies"]:
            pp = [r["policies"][policy] for r in rr]
            total = sum(p["regret"] for p in pp)
            result["policies"][policy] = {
                "redundant_pair_count": sum(p["chose_redundant_pair"] for p in pp),
                "complementary_pair_count": sum(not p["chose_redundant_pair"] for p in pp),
                "mean_regret": float(np.mean([p["regret"] for p in pp])),
                "redundancy_component_mean": float(np.mean([p["redundancy_component"] for p in pp])),
                "complementary_choice_component_mean": float(np.mean([p["complementary_choice_component"] for p in pp])),
                "redundancy_share_total_regret": sum(p["redundancy_component"] for p in pp)/total,
                "redundancy_component_family_interval": family_interval([p["redundancy_component"] for p in pp], families),
                "complementary_choice_component_family_interval": family_interval([p["complementary_choice_component"] for p in pp], families)}
        summary[condition] = result
    return {"analysis_status": "posthoc_descriptive_decomposition_after_locked_pilot",
            "definition": {
                "redundancy_component": "1[baseline deletes singleton B] * (L_AAprime - min(L_AB,L_AprimeB))",
                "complementary_choice_component": "1[baseline retains both axes] * (L_chosen - min(L_AB,L_AprimeB))",
                "identity": "Their sum is baseline-oracle regret here because every oracle is semantic-correct; checked per episode.",
                "ages": "All losses here use genuine native candidate timestamps. Semantic-complementary actions can have distinct ages/style, so their residual difference is not missing evidence.",
                "sibling_consistency": "Four observed histories differ in cue directions; a shared semantic-correct action set does not require a shared exact MSE minimizer and does not prove a learned causal rule."},
            "source_sha256": hashes(run, ["terminal_decisions.json", "data/records.json"]),
            "model_calls": 0, "protocol_or_experiment_files_changed": False,
            "bootstrap": {"family_cluster": True, "repetitions": 2000, "seed": 20261025,
                          "status": "exploratory component intervals, no confirmatory significance claim"},
            "summary": summary, "quartets": quartets, "episodes": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = read(args.run/"summary.json")
    if summary["decision"] == "stop_geometry_no_competition":
        raise ValueError("Geometry stop: no reader or deletion results to decompose")
    decisions = read(args.run/"terminal_decisions.json")
    records = {r["id"]: r for r in read(args.run/"data/records.json")}
    outputs = {"independent_interpretation.json": interpretation(args.run, summary, decisions, records),
               "posthoc_regret_decomposition.json": decomposition(args.run, decisions, records)}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (args.output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "completed_posthoc_descriptive_analysis", "model_calls": 0,
                      "outputs": {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in outputs}}))


if __name__ == "__main__":
    main()
