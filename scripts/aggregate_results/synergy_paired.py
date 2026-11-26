"""Paired comparison of our Synergy students with one literature baseline on the locked confirmation set.

    python scripts/aggregate_results/synergy_paired.py --config configs/synergy/release.yaml --seed 0 --out <dir> \\
        --reader item_attention --method pacc --stage-a <dir with <reader>_confirmation.npz> \\
        --ours <dir> --baselines <dir>

Ours: <ours>/students_ours_<reader>_seed{17,29,43}.json (scripts/evaluate/synergy.py, policy
deletion_cost_student_set). Baseline: <baselines>/<method>_<reader>_seed{17,29,43}.json
(scripts/train_baselines/synergy.py; policy pacc, or fm_published for framemorrow).

Per record, a policy's regret is its loss (or the cost of the card it deleted) minus the best single
deletion, averaged over its seeds; per task (R, D, S) the statistic is the family mean of regret A minus
regret B, the definition of evaluation/metrics/synergy.py. The 95% interval resamples the 48 families with replacement
(10,000 draws, fixed seed). A negative difference means ours has the lower regret.
Writes <out>/paired_<reader>_ours_vs_<baseline policy>.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.synergy import experiment  # noqa: E402

TASKS = "RDS"
SEEDS = (17, 29, 43)
POLICY = {"pacc": "pacc", "framemorrow": "fm_published"}
OURS = "deletion_cost_student_set"


def per_record_regret(files, policy, cost):
    seeds = []
    for f in files:
        record = json.loads(Path(f).read_text())["policies"][policy]
        if record["loss"] is not None:
            seeds.append(np.asarray(record["loss"], float) - cost.min(1))
        else:
            drop = np.asarray(record["drop"], int)
            seeds.append(cost[np.arange(len(drop)), drop] - cost.min(1))
    return np.mean(seeds, 0), len(seeds)


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--method", required=True, choices=list(POLICY))
    ap.add_argument("--ours", type=Path, required=True, help="folder with students_ours_<reader>_seed<s>.json")
    ap.add_argument("--baselines", type=Path, required=True, help="folder with <method>_<reader>_seed<s>.json")
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()
    args.policy_a, args.policy_b = OURS, POLICY[args.method]
    args.a = [args.ours / f"students_ours_{args.reader}_seed{s}.json" for s in SEEDS]
    args.b = [args.baselines / f"{args.method}_{args.reader}_seed{s}.json" for s in SEEDS]
    config = runs.load_config(args.config)
    records = experiment.confirmation_store(config).records
    stage = np.load(args.stage_a / f"{args.reader}_confirmation.npz")
    if "ids" in stage.files and stage["ids"].tolist() != [r["id"] for r in records]:
        raise SystemExit("stage-A arrays do not align with the confirmation records")
    cost = stage["eval_cost"]
    a, na = per_record_regret(args.a, args.policy_a, cost)
    b, nb = per_record_regret(args.b, args.policy_b, cost)
    rng = np.random.default_rng(20261124)
    out = {"reader": args.reader, "a": args.policy_a, "b": args.policy_b, "seeds_a": na, "seeds_b": nb,
           "resamples": args.resamples, "tasks": {}}
    for task in TASKS:
        index = [i for i, r in enumerate(records) if r["task_family"] == task]
        families = sorted({records[i]["family"] for i in index})
        by_family = {f: [i for i in index if records[i]["family"] == f] for f in families}
        fa = np.array([a[by_family[f]].mean() for f in families])
        fb = np.array([b[by_family[f]].mean() for f in families])
        diff = fa - fb
        draws = rng.integers(0, len(families), size=(args.resamples, len(families)))
        boot = diff[draws].mean(1)
        out["tasks"][task] = {"records": len(index), "families": len(families), "regret_a": float(fa.mean()),
                              "regret_b": float(fb.mean()), "difference_a_minus_b": float(diff.mean()),
                              "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                              "share_of_families_a_better": float((diff < 0).mean())}
        t = out["tasks"][task]
        print(f"{args.reader:15s} {task}: {args.policy_a} {t['regret_a']:+.4f}  {args.policy_b} {t['regret_b']:+.4f}  "
              f"diff {t['difference_a_minus_b']:+.4f}  95% CI [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]  "
              f"A better in {100 * t['share_of_families_a_better']:.0f}% of {t['families']} families")
    target = Path(args.out) / f"paired_{args.reader}_ours_vs_{args.policy_b}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
