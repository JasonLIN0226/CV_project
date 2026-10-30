"""Evaluate the three predeclared CONF claims from saved predictions only.

No fitting, inference, threshold selection, or exploratory hypothesis selection.
Rates are fractions; intervals are paired family bootstrap descriptions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

READERS = ("multihead", "item_attention")
SEEDS = (17, 29, 43)
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_REPS = 2000
TIE_SCALE = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_family_ci(values, indices, families):
    grouped = defaultdict(list)
    for i in indices:
        grouped[families[i]].append(values[i])
    per = np.asarray([np.mean(v) for v in grouped.values()], np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = per[rng.integers(len(per), size=(BOOTSTRAP_REPS, len(per)))].mean(1)
    return [float(per.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def losses_from_drop(drop, correct):
    if drop.shape != (len(correct),) or not np.isin(drop, [0, 1, 2]).all():
        raise ValueError("Invalid saved deletion array")
    return np.asarray([float(int(d) not in ok) for d, ok in zip(drop, correct)])


def tie_average_loss(values, correct):
    if values.shape != (len(correct), 3) or not np.isfinite(values).all():
        raise ValueError("Invalid values for registered tie robustness")
    tolerance = TIE_SCALE * values.max(1)
    if (tolerance < 0).any():
        raise ValueError("Negative row-maximum tolerance")
    tied = values - values.min(1, keepdims=True) <= tolerance[:, None]
    if (tied.sum(1) == 0).any():
        raise ValueError("Empty tie set")
    wrong = np.asarray([[d not in ok for d in range(3)] for ok in correct])
    return (wrong * tied).sum(1) / tied.sum(1), tied.sum(1)


def claim_truth(evidence, by_task):
    rates = lambda reader, method, task: float(evidence[reader][method][by_task[task]].mean())
    return {
        "claim_1_synergy": all(rates(r, "student_set_cost", "S") < rates(r, m, "S")
                               for r in READERS for m in ("full_sensitivity", "attention", "student_set_future_attention")),
        "claim_2_rd_oracle_gap": all(abs(rates("item_attention", "full_sensitivity", t) -
                                           rates("item_attention", "oracle", t)) <= .10 for t in ("R", "D")),
        "claim_3_medoid": all(rates(r, "medoid", t) > rates(r, "full_sensitivity", t)
                              for r in READERS for t in ("D", "S")),
    }


def assess(records_path, stage_a, predictions, evaluations):
    recs = json.loads(Path(records_path).read_text())
    if not recs or {r["split"] for r in recs} != {"confirmation"}:
        raise ValueError("Only CONF records are accepted")
    families = [r["family"] for r in recs]
    if len(set(families)) != 48:
        raise ValueError("Expected 48 CONF families")
    ids = [r["id"] for r in recs]
    correct = [set(r["semantic_correct_delete_set"]) for r in recs]
    by_task = {t: [i for i, r in enumerate(recs) if r["task_family"] == t] for t in "RDS"}
    expected_n = {"R": 576, "D": 576, "S": 2304}
    if {t: len(v) for t, v in by_task.items()} != expected_n:
        raise ValueError("Unexpected CONF task counts")
    evidence, tie_evidence, tie_table = {}, {}, []
    provenance = {"records_sha256": sha(records_path)}
    for reader in READERS:
        a_path = Path(stage_a) / f"{reader}_confirmation.npz"
        p_path = Path(predictions) / f"{reader}_standard" / "confirmation_predictions.npz"
        e_path = Path(evaluations) / f"{reader}_conf.json"
        a = np.load(a_path, allow_pickle=False)
        p = np.load(p_path, allow_pickle=False)
        evaluation = json.loads(e_path.read_text())
        if a["ids"].tolist() != ids or p["ids"].tolist() != ids:
            raise ValueError("Saved outputs do not match record ordering")
        loss = {}
        for method, key in (("full_sensitivity", "sensitivity"), ("attention", "attention"),
                            ("medoid", "medoid_sum"), ("oracle", "eval_cost")):
            value = a[key]
            if value.shape != (len(recs), 3) or not np.isfinite(value).all():
                raise ValueError("Invalid Stage A scores")
            loss[method] = losses_from_drop(value.argmin(1), correct)
        for label in ("cost", "future_attention"):
            arrays = [losses_from_drop(p[f"set_{label}_seed{seed}__drop"], correct) for seed in SEEDS]
            loss[f"student_set_{label}"] = np.mean(arrays, axis=0)
            if evaluation["student_seeds"][f"student_set_{label}"] != 3:
                raise ValueError("Evaluation did not average exactly three student seeds")
        for method, values in loss.items():
            for task, indices in by_task.items():
                actual = paired_family_ci(values, indices, families)
                expected = evaluation["methods"][method][task]["lost_rate_ci"]
                if not np.allclose(actual, expected, atol=1e-12, rtol=0):
                    raise ValueError(f"Saved evaluation mismatch: {reader}/{method}/{task}")
        evidence[reader] = loss
        tie_loss = dict(loss)
        for method, key in (("full_sensitivity", "sensitivity"), ("medoid", "medoid_sum"), ("oracle", "eval_cost")):
            tie_loss[method], sizes = tie_average_loss(a[key], correct)
            for task, indices in by_task.items():
                exact_ci = paired_family_ci(loss[method], indices, families)
                tie_ci = paired_family_ci(tie_loss[method], indices, families)
                difference_ci = paired_family_ci(tie_loss[method] - loss[method], indices, families)
                tie_table.append({"reader": reader, "task": task, "method": method,
                                  "parents": len(indices), "families": len({families[i] for i in indices}),
                                  "tie_parents": int((sizes[indices] > 1).sum()),
                                  "tie_set_sizes": {str(k): int((sizes[indices] == k).sum()) for k in (1, 2, 3)},
                                  "argmin_lost_rate_ci": exact_ci, "tie_average_lost_rate_ci": tie_ci,
                                  "tie_minus_argmin_family_bootstrap_95_ci": difference_ci})
        tie_evidence[reader] = tie_loss
        provenance[reader] = {"stage_a_sha256": sha(a_path), "standard_predictions_sha256": sha(p_path),
                              "evaluation_sha256": sha(e_path)}

    claim1 = {"rule": "On S, mean-of-three-seed set_cost loss is strictly lower than each comparator, for both readers", "readers": {}}
    for reader in READERS:
        loss = evidence[reader]
        index = by_task["S"]
        student = loss["student_set_cost"]
        comparisons = {}
        for comparator in ("full_sensitivity", "attention", "student_set_future_attention"):
            ci = paired_family_ci(student - loss[comparator], index, families)
            comparisons[comparator] = {
                "student_lost_rate": float(student[index].mean()),
                "comparator_lost_rate": float(loss[comparator][index].mean()),
                "student_minus_comparator_family_bootstrap_95_ci": ci,
                "passes": bool(student[index].mean() < loss[comparator][index].mean()),
            }
        claim1["readers"][reader] = {"passes": all(x["passes"] for x in comparisons.values()),
                                      "comparisons": comparisons}
    claim1["passes"] = all(x["passes"] for x in claim1["readers"].values())

    claim2 = {"rule": "For item_attention, absolute full_sensitivity minus oracle loss-rate difference is at most 0.10 separately in R and D", "reader": "item_attention", "tasks": {}}
    loss = evidence["item_attention"]
    for task in ("R", "D"):
        index = by_task[task]
        sensitivity = float(loss["full_sensitivity"][index].mean())
        oracle = float(loss["oracle"][index].mean())
        difference = abs(sensitivity - oracle)
        claim2["tasks"][task] = {"full_sensitivity_lost_rate": sensitivity,
                                   "oracle_lost_rate": oracle,
                                   "absolute_difference": difference,
                                   "passes": bool(difference <= .10)}
    claim2["passes"] = all(x["passes"] for x in claim2["tasks"].values())

    claim3 = {"rule": "For both readers, medoid loss rate is strictly higher than full_sensitivity separately in D and S", "readers": {}}
    for reader in READERS:
        loss = evidence[reader]
        tasks = {}
        for task in ("D", "S"):
            index = by_task[task]
            medoid = float(loss["medoid"][index].mean())
            sensitivity = float(loss["full_sensitivity"][index].mean())
            tasks[task] = {"medoid_lost_rate": medoid,
                           "full_sensitivity_lost_rate": sensitivity,
                           "difference": medoid - sensitivity,
                           "passes": bool(medoid > sensitivity)}
        claim3["readers"][reader] = {"passes": all(x["passes"] for x in tasks.values()), "tasks": tasks}
    claim3["passes"] = all(x["passes"] for x in claim3["readers"].values())
    claims = {"claim_1_synergy": claim1, "claim_2_rd_oracle_gap": claim2, "claim_3_medoid": claim3}
    formal_truth = claim_truth(evidence, by_task)
    robust_truth = claim_truth(tie_evidence, by_task)
    for key, claim in claims.items():
        if formal_truth[key] != claim["passes"]:
            raise ValueError("Formal claim implementations disagree")
        claim["conclusion"] = "通过" if claim["passes"] else "不通过"
        claim["tie_average_passes"] = robust_truth[key]
        claim["truth_flips_under_tie_average"] = claim["passes"] != robust_truth[key]
        claim["tie_robustness"] = "不稳健" if claim["truth_flips_under_tie_average"] else "真值不变"
    return {"status": "completed_confirmation_claims", "split": "confirmation", "seed": 20261104,
            "families": 48, "task_counts": expected_n, "student_seeds": list(SEEDS),
            "units": "loss rates and CI endpoints are fractions; multiply by 100 for percentage points",
            "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPS,
                          "unit": "family", "paired": True, "role": "descriptive only; not a pass criterion"},
            "claims": claims,
            "tie_robustness": {"rule": "value - min(value) <= 1e-4 * max(value) per row",
                               "formal_decisions_unchanged": True, "rows": tie_table,
                               "replaced_methods_only": ["full_sensitivity", "medoid", "oracle"],
                               "student_and_attention_decisions_unchanged": True},
            "input_bindings": provenance, "claims_source_sha256": sha(__file__)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--evaluations", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite the first CONF claims")
    result = assess(args.records, args.stage_a, args.predictions, args.evaluations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v["conclusion"] for k, v in result["claims"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
