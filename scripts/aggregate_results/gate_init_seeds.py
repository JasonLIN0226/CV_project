"""Gate: summarise the extra-initialisation-seed runs of our scorer and Compress to Remember (two slots).

    python scripts/aggregate_results/gate_init_seeds.py --runs <dir> [--reference configs/reference_values.json] \\
        --out <file.json>

<dir> holds the unit files of scripts/train_baselines/gate_init_seeds.py, <method>_init<init>_seed<predictor>.json
with method in {ours, pacc_recurrent}. A unit's route accuracy is the mean over its validation rows (192 episodes x
horizons 3 and 6), as in the panel. Per method the summary gives the mean, spread and per-predictor values over its
units, and the number of units at or below 0.60 route accuracy ("collapsed": chance is 0.50; the registered
Compress to Remember run for predictor 17 scored 0.505). Paired, over the (predictor, init) units both methods have:
ours minus Compress to Remember, with two 95% intervals (10,000 draws, seed 20261125): one resampling the units
(training-seed variation) and one resampling the 192 validation episodes with every unit kept (evaluation-set
variation). The registered seed-17/29/43 values are copied from the reference file for comparison only.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = {"ours": "deletion_cost_student", "pacc_recurrent": "pacc_recurrent"}
NAME = re.compile(r"^(ours|pacc_recurrent)_init(\d+)_seed(\d+)\.json$")
COLLAPSE = 0.60


def load_units(folder):
    """{method: {(predictor, init): {"route": float, "mse": float, "episode_route": {episode: float}}}}"""
    units = defaultdict(dict)
    for path in sorted(Path(folder).glob("*.json")):
        m = NAME.match(path.name)
        if not m:
            continue
        method, init, predictor = m.group(1), int(m.group(2)), int(m.group(3))
        policy = f"{METHODS[method]}_init{init}"
        rows = [r for r in json.loads(path.read_text())["rows"] if r["policy"] == policy]
        if not rows:
            raise SystemExit(f"{path} has no rows for policy {policy}")
        by_episode = defaultdict(list)
        for r in rows:
            by_episode[r["episode"]].append(float(r["route_accuracy"]))
        mse = [float(r["normalized_mse"]) for r in rows if r.get("normalized_mse") is not None]
        units[method][(predictor, init)] = {
            "route": float(np.mean([float(r["route_accuracy"]) for r in rows])),
            "mse": float(np.mean(mse)) if mse else None,
            "rows": len(rows),
            "episode_route": {e: float(np.mean(v)) for e, v in by_episode.items()},
        }
    return units


def method_summary(units):
    route = np.array([u["route"] for u in units.values()])
    by_predictor = defaultdict(dict)
    for (p, i), u in sorted(units.items()):
        by_predictor[str(p)][str(i)] = round(u["route"], 6)
    return {"units": len(units), "route_mean": float(route.mean()), "route_std": float(route.std(ddof=1)) if len(route) > 1 else 0.0,
            "route_min": float(route.min()), "route_max": float(route.max()),
            "collapsed_units": int((route <= COLLAPSE).sum()), "route_by_predictor_and_init": by_predictor}


def paired(a, b, resamples=10000, seed=20261125):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diff = np.array([a[k]["route"] - b[k]["route"] for k in keys])
    rng = np.random.default_rng(seed)
    unit_boot = diff[rng.integers(0, len(keys), size=(resamples, len(keys)))].mean(1)
    episodes = sorted(set.intersection(*[set(a[k]["episode_route"]) & set(b[k]["episode_route"]) for k in keys]))
    per_episode = np.array([np.mean([a[k]["episode_route"][e] - b[k]["episode_route"][e] for k in keys]) for e in episodes])
    episode_boot = per_episode[rng.integers(0, len(episodes), size=(resamples, len(episodes)))].mean(1)
    return {"units": len(keys), "episodes": len(episodes), "difference_ours_minus_pacc": float(diff.mean()),
            "ci95_units": [float(np.quantile(unit_boot, 0.025)), float(np.quantile(unit_boot, 0.975))],
            "ci95_episodes": [float(np.quantile(episode_boot, 0.025)), float(np.quantile(episode_boot, 0.975))],
            "units_ours_higher": int((diff > 0).sum()), "units_tied": int((diff == 0).sum()),
            "per_unit": {f"seed{p}_init{i}": round(float(d), 6) for (p, i), d in zip(keys, diff)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[2] / "configs/reference_values.json")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    units = load_units(args.runs)
    out = {"runs": str(args.runs), "collapse_threshold": COLLAPSE, "methods": {}, "registered": {}}
    reference = json.loads(args.reference.read_text()).get("gate", {}) if args.reference.exists() else {}
    for method, policy in METHODS.items():
        if policy in reference:
            out["registered"][method] = {"route_accuracy": reference[policy]["route_accuracy"],
                                         "route_accuracy_by_seed": reference[policy]["route_accuracy_by_seed"]}
        if method in units:
            out["methods"][method] = s = method_summary(units[method])
            print(f"{method:15s} {s['units']} units  route mean {s['route_mean']:.4f}  sd {s['route_std']:.4f}  "
                  f"min {s['route_min']:.4f}  max {s['route_max']:.4f}  collapsed {s['collapsed_units']}")
    out["paired"] = paired(units.get("ours", {}), units.get("pacc_recurrent", {}), args.resamples)
    if out["paired"]:
        p = out["paired"]
        print(f"ours - pacc_recurrent over {p['units']} units: {p['difference_ours_minus_pacc']:+.4f}  "
              f"95% CI units [{p['ci95_units'][0]:+.4f}, {p['ci95_units'][1]:+.4f}]  "
              f"episodes [{p['ci95_episodes'][0]:+.4f}, {p['ci95_episodes'][1]:+.4f}]  "
              f"ours higher in {p['units_ours_higher']}/{p['units']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
