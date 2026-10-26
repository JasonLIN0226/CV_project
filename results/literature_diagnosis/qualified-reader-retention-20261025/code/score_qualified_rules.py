"""Score training-free terminal retention rules on qualified readers (CPU only).

Usage: python -I score_qualified_rules.py <agg_extract_dir> <terminal_extract_dir> <out_json>
Rules are locked in qualified_reader_rules_lock.md.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch

AGG, TERM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SPLITS = {"existing_development": TERM + "/run/data/records.json",
          "fresh_diagnostic": AGG + "/run/diagnostic_data/records.json"}
READERS = ("multihead", "item_attention", "original")
RULES = ("fifo", "attention", "medoid", "full_sensitivity", "oracle")


def output_basis(reader):
    state = torch.load(f"{AGG}/run/readers/{reader}/reader_selected.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    basis = state["output_basis"].double().numpy()
    return basis


def family_ci(values, families, seed=20261107, reps=2000):
    fam = defaultdict(list)
    for v, f in zip(values, families):
        fam[f].append(v)
    per = np.array([np.mean(v) for v in fam.values()])
    rng = np.random.default_rng(seed)
    boot = per[rng.integers(len(per), size=(reps, len(per)))].mean(1)
    return {"mean": float(np.mean(values)), "family_mean": float(per.mean()),
            "low": float(np.quantile(boot, .025)), "high": float(np.quantile(boot, .975)), "families": len(per)}


def score(split, reader, records, basis):
    base = f"{AGG}/run/evaluation/{split}/{reader}/"
    rows = json.load(open(base + "reader_qualification_rows.json"))
    coeff = np.load(base + "reader_qualification_coefficients.npy").astype(np.float64)
    weights = {r["id"]: np.array(r["weights"], np.float64) for r in json.load(open(base + "pooling_weights.json"))["rows"]}
    by = defaultdict(dict)
    for r in rows:
        if r["arm"] in ("native_0", "native_1", "native_2", "full"):
            by[r["id"]][r["arm"]] = r
    gram_cache = {}
    out = []
    dim = basis.shape[0]
    btb = basis.T @ basis  # handles non-orthonormal bases exactly

    def dist(ci, cj):
        d = ci - cj  # (2, rank)
        return float(np.einsum("hr,rs,hs->", d, btb, d) / (d.shape[0] * dim))

    for pid, arms in by.items():
        rec = records[pid]
        c = [coeff[arms["native_%d" % k]["coefficient_index"]] for k in range(3)]
        cf = coeff[arms["full"]["coefficient_index"]]
        mse = np.array([arms["native_%d" % k]["mse"] for k in range(3)])
        D = np.array([[dist(c[i], c[j]) for j in range(3)] for i in range(3)])
        sens = np.array([dist(cf, c[i]) for i in range(3)])
        choice = {"fifo": 0, "attention": int(np.argmin(weights[pid])), "medoid": int(np.argmin(D.sum(1))),
                  "full_sensitivity": int(np.argmin(sens)), "oracle": int(np.argmin(mse))}
        correct = set(rec["semantic_correct_delete_set"])
        row = {"id": pid, "family": rec["family_id"], "task": rec["task_family"], "condition": rec["condition"],
               "mse": mse.tolist(), "correct_delete_set": sorted(correct), "D_row_sums": D.sum(1).tolist(),
               "sensitivity": sens.tolist(), "attention": weights[pid].tolist(), "choices": choice,
               "full_correct": bool(arms["full"]["strict_correct"])}
        for k, v in choice.items():
            row[k + "_lost"] = float(v not in correct)
            row[k + "_regret"] = float(mse[v] - mse.min())
        # mechanism check for medoid: are the two incomplete forecasts closer to each other than to the complete one?
        wrong = [k for k in range(3) if k not in correct]
        if len(wrong) == 2:
            a, b = wrong
            comp = [k for k in range(3) if k in correct][0]
            row["incomplete_pair_closer"] = bool(D[a, b] < min(D[a, comp], D[b, comp]))
        out.append(row)
    return out


def summarize(rows):
    res = {}
    for task in ("R", "D", "S"):
        sub = [r for r in rows if r["task"] == task]
        fam = [r["family"] for r in sub]
        t = {"n": len(sub)}
        for rule in RULES:
            t[rule] = {"lost": int(sum(r[rule + "_lost"] for r in sub)),
                       "lost_ci": family_ci([r[rule + "_lost"] for r in sub], fam),
                       "regret_ci": family_ci([r[rule + "_regret"] for r in sub], fam)}
        t["uniform_expected_lost_fraction"] = float(np.mean([1 - len(r["correct_delete_set"]) / 3 for r in sub]))
        t["full_K3_correct"] = int(sum(r["full_correct"] for r in sub))
        if task in ("D", "S"):
            t["incomplete_pair_closer"] = int(sum(r.get("incomplete_pair_closer", False) for r in sub))
        conds = defaultdict(list)
        for r in sub:
            conds[r["condition"]].append(r)
        t["by_condition"] = {c: {rule: f"{int(sum(r[rule + '_lost'] for r in v))}/{len(v)}" for rule in RULES}
                             for c, v in sorted(conds.items())}
        res[task] = t
    return res


def main():
    result = {"rules_lock": "qualified_reader_rules_lock.md", "splits": {}}
    for split, rpath in SPLITS.items():
        records = {r["id"]: r for r in json.load(open(rpath))}
        result["splits"][split] = {}
        for reader in READERS:
            basis = output_basis(reader)
            rows = score(split, reader, records, basis)
            result["splits"][split][reader] = {"summary": summarize(rows), "rows": rows,
                                               "basis_shape": list(basis.shape),
                                               "basis_orthonormal_maxdev": float(np.abs(basis.T @ basis - np.eye(basis.shape[1])).max())}
    json.dump(result, open(OUT, "w"))
    for split, readers in result["splits"].items():
        for reader, v in readers.items():
            print(f"\n## {split} / {reader}  (basis {v['basis_shape']}, orth dev {v['basis_orthonormal_maxdev']:.2e})")
            for task, t in v["summary"].items():
                line = "  ".join(f"{rule} {t[rule]['lost']}/{t['n']}" for rule in RULES)
                extra = f"  | uniform E[lost]={t['uniform_expected_lost_fraction']:.2f}  K3 correct {t['full_K3_correct']}/{t['n']}"
                if "incomplete_pair_closer" in t:
                    extra += f"  incomplete-pair-closer {t['incomplete_pair_closer']}/{t['n']}"
                print(f"  {task}: {line}{extra}")


if __name__ == "__main__":
    main()
