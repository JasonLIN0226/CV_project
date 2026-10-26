"""Stage C: score every locked method on DEV for one reader.

Usage: python -I evaluate_dev.py <reader> <stageA_dir> <students_dir> <out_json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe  # noqa: E402
from path_config import SPLIT, BASELINE_DECISIONS

reader, A, S, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
st = pipe.Store()
recs = [r for r in st.records if r["split"] == SPLIT]
a = np.load(A / f"{reader}_{SPLIT}.npz")
assert a["ids"].tolist() == [r["id"] for r in recs]
cost = a["eval_cost"]                       # [N,3] expected in-span loss over 32 evaluation plans
TOL = 1e-6
term = {d["id"]: d for d in json.load(open(BASELINE_DECISIONS))} if SPLIT == "development" else None

choices = {
    "fifo": np.zeros(len(recs), int),
    "attention": a["attention"].argmin(1),
    "medoid": a["medoid_sum"].argmin(1),
    "full_sensitivity": a["sensitivity"].argmin(1),
    "oracle": cost.argmin(1),
}
if term is not None:
    choices["uniform"] = np.array([term[r["id"]]["policies"]["terminal_uniform"]["drop"] for r in recs])
    choices["visual_diversity"] = np.array([term[r["id"]]["policies"]["visual_diversity"]["drop"] for r in recs])
student_arms = defaultdict(list)            # arm -> list of per-seed drop arrays
for mode in ("standard", "distilled"):
    p = S / f"{reader}_{mode}" / ("dev_predictions.npz" if SPLIT == "development" else f"{SPLIT}_predictions.npz")
    if not p.exists():
        continue
    z = np.load(p)
    assert z["ids"].tolist() == [r["id"] for r in recs]
    for k in z.files:
        if not k.endswith("__drop"):
            continue
        name = k[:-6]                       # e.g. set_cost_seed17
        arch, label = name.split("_")[0], name.split("_")[1]
        if mode == "distilled":
            if label != "cost":
                continue
            arm = f"student_{arch}_distilled_sensitivity"
        else:
            arm = f"student_{arch}_{'future_attention' if label == 'future' else label}"
        student_arms[arm].append(z[k])

fam = [r["family"] for r in recs]
correct = [set(r["semantic_correct_delete_set"]) for r in recs]


def boot(values, idx, seed=20261107, reps=2000):
    g = defaultdict(list)
    for i in idx:
        g[fam[i]].append(values[i])
    per = np.array([np.mean(v) for v in g.values()])
    rng = np.random.default_rng(seed)
    b = per[rng.integers(len(per), size=(reps, len(per)))].mean(1)
    return [float(per.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def metrics(drop_sets):
    """drop_sets: list of per-seed drop arrays (len 1 for deterministic methods)."""
    lost = np.mean([[float(d[i] not in correct[i]) for i in range(len(recs))] for d in drop_sets], 0)
    regret = np.mean([[cost[i, d[i]] - cost[i].min() for i in range(len(recs))] for d in drop_sets], 0)
    res = {}
    for task in "RDS":
        idx = [i for i, r in enumerate(recs) if r["task_family"] == task]
        conds = defaultdict(list)
        for i in idx:
            conds[recs[i]["condition"]].append(i)
        res[task] = {"n": len(idx), "lost_count": float(lost[idx].sum()), "lost_rate_ci": boot(lost, idx),
                     "regret_ci": boot(regret, idx),
                     "by_condition_lost": {c: f"{lost[v].sum():.1f}/{len(v)}" for c, v in sorted(conds.items())}}
    return res


table = {m: metrics([d]) for m, d in choices.items()}
table.update({m: metrics(ds) for m, ds in sorted(student_arms.items())})
expected_uniform = {t: float(np.mean([1 - len(correct[i]) / 3 for i, r in enumerate(recs) if r["task_family"] == t])) for t in "RDS"}
json.dump({"reader": reader, "methods": table, "uniform_expected_lost_rate": expected_uniform,
           "not_computed": [] if term is not None else ["uniform", "visual_diversity"],
           "student_seeds": {k: len(v) for k, v in student_arms.items()},
           "oracle_tie_parents": int(sum((cost[i] <= cost[i].min() + TOL).sum() > 1 for i in range(len(recs))))},
          open(OUT, "w"), indent=1)
print(f"## {SPLIT} / {reader}   (lost necessary evidence: count / n ; regret mean)")
for m, t in table.items():
    print(f"{m:42s} " + "  ".join(f"{task}: {t[task]['lost_count']:6.1f}/{t[task]['n']:<4d} reg {t[task]['regret_ci'][0]:.4f}" for task in "RDS"))
