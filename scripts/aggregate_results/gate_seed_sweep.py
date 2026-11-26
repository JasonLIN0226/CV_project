"""Pool a Gate seed sweep: two-slot Compress to Remember against our scorer over training seeds.

    python scripts/aggregate_results/gate_seed_sweep.py --run runs/<run_id> [--sweep sweep] [--collapse-below 0.75]

Reads <run>/gate/<sweep>/<task>_t<N>_seed<P>.json (--sweep sweep_gpu for the GPU sweep, gate.sweep_gpu.*),
written by scripts/train_baselines/gate.py and scripts/evaluate/gate.py with --train-seed: one run per method, predictor seed P and training seed N.
A run's route accuracy is the mean over its rows (validation episodes x horizons); a run below
--collapse-below collapsed, i.e. it did not learn to keep the cue. Per method: the collapse count, the
mean route accuracy over all runs and over the runs that trained, and the panel-style value of each
training seed (mean over the predictor seeds). Per (P, N) cell: which method is higher, and the mean of ours
minus Compress to Remember with a 95% interval that resamples the cells (10,000 draws). A score of our
scorer must come from the checkpoint of its own training seed (student_<P>_t<N>.pt), and every file
must record the training seed its name gives; anything else stops the summary.
Writes <run>/aggregate/seed_<sweep>.json and .md (seed_sweep.json and seed_sweep.md for the default).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

POLICY = {"pacc_recurrent": "pacc_recurrent", "deletion_cost": "deletion_cost_student"}
LABEL = {"pacc_recurrent": "Compress to Remember, two slots", "deletion_cost": "ours, deletion-cost scorer"}


def load(folder):
    """method -> {(P, N): run}, checking that each file is the run its name says."""
    found = defaultdict(dict)
    for path in sorted(folder.glob("*_t*_seed*.json")):
        record = json.loads(path.read_text())
        method, _, n = record["task"].rpartition("_t")
        if method not in POLICY:
            continue
        p, n = int(record["seed"]), int(n)
        if record["info"].get("train_seed") != n:
            raise SystemExit(f"{path.name}: records training seed {record['info'].get('train_seed')}, not {n}")
        if method == "deletion_cost":
            expected, got = f"student_{p}_t{n}.pt", record["info"].get("students", {}).get("file")
            if got != expected:
                raise SystemExit(f"{path.name}: scored {got}, not {expected}")
        accuracy = [r["route_accuracy"] for r in record["rows"] if r["policy"] == POLICY[method]]
        if not accuracy:
            raise SystemExit(f"{path.name}: no rows of {POLICY[method]}")
        found[method][(p, n)] = {"route_accuracy": float(np.mean(accuracy)), "rows": len(accuracy),
                                 "device": record.get("device"), "threads": record.get("threads"), "host": record.get("host")}
    return found


def describe(cells, threshold):
    accuracy = np.array([c["route_accuracy"] for c in cells.values()])
    trained = accuracy[accuracy >= threshold]
    predictors = sorted({p for p, _ in cells})
    by_seed = defaultdict(dict)
    for (p, n), c in cells.items():
        by_seed[n][p] = c["route_accuracy"]
    pooled = {str(n): float(np.mean(list(v.values()))) for n, v in sorted(by_seed.items()) if sorted(v) == predictors}
    return {"runs": len(accuracy), "collapsed": int((accuracy < threshold).sum()),
            "collapse_rate": float((accuracy < threshold).mean()),
            "route_accuracy_mean": float(accuracy.mean()),
            "route_accuracy_mean_trained": float(trained.mean()) if len(trained) else None,
            "route_accuracy_min": float(accuracy.min()), "route_accuracy_max": float(accuracy.max()),
            "predictor_seeds": predictors, "training_seeds": sorted(by_seed),
            "pooled_by_training_seed": pooled,
            "incomplete_training_seeds": sorted(n for n, v in by_seed.items() if sorted(v) != predictors),
            "collapsed_cells": sorted([p, n] for (p, n), c in cells.items() if c["route_accuracy"] < threshold),
            "devices": sorted({str(c["device"]) for c in cells.values()}),
            "threads": sorted({c["threads"] for c in cells.values() if c["threads"] is not None}),
            "hosts": len({c["host"] for c in cells.values()})}


def head_to_head(found):
    a, b = found.get("pacc_recurrent", {}), found.get("deletion_cost", {})
    shared = sorted(set(a) & set(b))
    return {"cells": len(shared),
            "pacc_recurrent_higher": sum(a[c]["route_accuracy"] > b[c]["route_accuracy"] for c in shared),
            "deletion_cost_higher": sum(a[c]["route_accuracy"] < b[c]["route_accuracy"] for c in shared),
            "tied": sum(a[c]["route_accuracy"] == b[c]["route_accuracy"] for c in shared)}


def paired_difference(found, resamples=10000, seed=20261125):
    """Ours minus Compress to Remember per shared (P, N) cell: mean and a 95% interval over the cells."""
    a, b = found.get("pacc_recurrent", {}), found.get("deletion_cost", {})
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diff = np.array([b[c]["route_accuracy"] - a[c]["route_accuracy"] for c in shared])
    draws = diff[np.random.default_rng(seed).integers(0, len(diff), size=(resamples, len(diff)))].mean(1)
    return {"cells": len(shared), "ours_minus_pacc_recurrent": float(diff.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}"


def markdown(run, threshold, methods, found, versus, paired=None, sweep="sweep"):
    title = f"# Gate seed sweep: run {run}" + ("" if sweep == "sweep" else f", {sweep}")
    lines = [title, "",
             f"Route accuracy (%) on the 192 validation episodes, both horizons. A run below {percent(threshold)} collapsed.", "",
             "| Method | Runs | Collapsed | Mean, all runs | Mean, runs that trained | Min | Max | Panel-style value per training seed: mean (min to max) | Threads |",
             "|---|---|---|---|---|---|---|---|---|"]
    for method, s in methods.items():
        pooled = list(s["pooled_by_training_seed"].values())
        spread = f"{percent(np.mean(pooled))} ({percent(min(pooled))} to {percent(max(pooled))})" if pooled else "n/a"
        lines.append(f"| {LABEL[method]} | {s['runs']} | {s['collapsed']} | {percent(s['route_accuracy_mean'])} | "
                     f"{percent(s['route_accuracy_mean_trained'])} | {percent(s['route_accuracy_min'])} | {percent(s['route_accuracy_max'])} | "
                     f"{spread} | {', '.join(map(str, s['threads'])) or 'n/a'} |")
    lines += ["", f"Same predictor and training seed, higher route accuracy: Compress to Remember in {versus['pacc_recurrent_higher']}, "
              f"ours in {versus['deletion_cost_higher']}, tied in {versus['tied']} of {versus['cells']} cells."]
    if paired:
        lines.append(f"Ours minus Compress to Remember, mean over cells: {100 * paired['ours_minus_pacc_recurrent']:+.2f} points "
                     f"(95% interval over cells {100 * paired['ci95'][0]:+.2f} to {100 * paired['ci95'][1]:+.2f}).")
    lines.append("")
    for method, cells in found.items():
        predictors = sorted({p for p, _ in cells})
        seeds = sorted({n for _, n in cells})
        lines += [f"## {LABEL[method]}", "", "| Training seed | " + " | ".join(f"predictor {p}" for p in predictors) + " |",
                  "|---|" + "---|" * len(predictors)]
        for n in seeds:
            values = [percent(cells[(p, n)]["route_accuracy"]) if (p, n) in cells else "" for p in predictors]
            lines.append(f"| {n} | " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--sweep", default="sweep", help="folder under <run>/gate with the sweep files (sweep, sweep_gpu)")
    ap.add_argument("--collapse-below", type=float, default=0.75, help="route accuracy below which a run collapsed")
    args = ap.parse_args()
    found = load(args.run / "gate" / args.sweep)
    if not found:
        raise SystemExit(f"no sweep results under {args.run / 'gate' / args.sweep}")
    methods = {m: describe(found[m], args.collapse_below) for m in POLICY if m in found}
    versus, paired = head_to_head(found), paired_difference(found)
    out = args.run / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    name = "seed_sweep" if args.sweep == "sweep" else f"seed_{args.sweep}"
    (out / f"{name}.json").write_text(json.dumps(
        {"run": args.run.name, "sweep": args.sweep, "collapse_below": args.collapse_below, "methods": methods,
         "head_to_head": versus, "paired_difference": paired,
         "cells": {m: {f"{p}/{n}": c for (p, n), c in sorted(cells.items())} for m, cells in found.items()}}, indent=1) + "\n")
    (out / f"{name}.md").write_text(markdown(args.run.name, args.collapse_below, methods, found, versus, paired, args.sweep))
    for method, s in methods.items():
        print(f"{method:16s} runs {s['runs']:3d}  collapsed {s['collapsed']:3d}  mean {percent(s['route_accuracy_mean'])}  "
              f"trained mean {percent(s['route_accuracy_mean_trained'])}")
    print(f"wrote {out / (name + '.json')} and {out / (name + '.md')}")


if __name__ == "__main__":
    main()
