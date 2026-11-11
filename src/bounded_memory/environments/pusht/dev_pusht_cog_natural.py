"""Development run for design v4 (natural episodes): validity, leak and history gain.

CPU only. For dev seeds, runs families in a process pool (appending to
families.jsonl), then fits classifiers on *simulator state* (not pixels):

  present           block pose relative to the arena + pusher, at the query
  reset             the same, at the first frame
  probe_end         the same, right after the probe
  reset+present     both, plus the relative motion between them
  probe_end+present both, plus the relative motion between them

Accuracy is family-grouped 5-fold CV (chance 1/3). The memory value of the task
is the gap between `present` and the history+present rows. These seeds are
inspected and must stay out of any split.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_cog_matched as v3
import pusht_cog_natural as m

POOL = None


def _init(pool):
    global POOL
    POOL = pool


def run_family(seed: int):
    started = time.perf_counter()
    spec = m.sample_family(seed, POOL)
    result, _, _, _ = m.check_family(spec)
    result.update(seed=seed, seconds=time.perf_counter() - started)
    return result


def pose_features(s):
    """Arena-relative block pose in the block frame, its angle, and the pusher in the block frame."""
    c, a = np.array(s["center"]), s["angle"]
    off = m.rotate(c - 256.0, -a)
    pusher = m.rotate(np.array(s["pusher"]) - c, -a)
    return [*off, np.sin(a), np.cos(a), *pusher]


def relative_features(s0, s1):
    da = s1["angle"] - s0["angle"]
    shift = m.rotate(np.array(s1["center"]) - np.array(s0["center"]), -s0["angle"])
    return [np.sin(da), np.cos(da), *shift]


def leak_table(rows):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score
    names = list(m.COG_CLASSES)
    X = {k: [] for k in ("present", "reset", "probe_end", "reset+present", "probe_end+present")}
    y, groups = [], []
    for r in rows:
        for c, e in r["per_class"].items():
            p, s0, s1 = pose_features(e["query"]), pose_features(e["reset"]), pose_features(e["probe_end"])
            X["present"].append(p)
            X["reset"].append(s0)
            X["probe_end"].append(s1)
            X["reset+present"].append(s0 + p + relative_features(e["reset"], e["query"]))
            X["probe_end+present"].append(s1 + p + relative_features(e["probe_end"], e["query"]))
            y.append(names.index(c))
            groups.append(r["seed"])
    y, groups = np.array(y), np.array(groups)
    out = {}
    for k, v in X.items():
        scores = cross_val_score(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
                                 np.array(v), y, groups=groups, cv=GroupKFold(5))
        out[k] = {"accuracy": float(scores.mean()), "fold_std": float(scores.std())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=192)
    parser.add_argument("--base-seed", type=int, default=2026140000)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "families.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines() if line.strip()}

    library = v3.build_library(args.library, processes=args.processes)
    pool_rows = m.probe_pool(library)
    print(f"probe pool: {len(pool_rows)} library probes", flush=True)
    seeds = [args.base_seed + k for k in range(args.families) if args.base_seed + k not in done]
    t = time.perf_counter()
    with multiprocessing.get_context("fork").Pool(args.processes, initializer=_init,
                                                  initargs=(pool_rows,)) as pool, out.open("a") as f:
        for k, result in enumerate(pool.imap_unordered(run_family, seeds), 1):
            f.write(json.dumps(result) + "\n")
            f.flush()
            if k % 16 == 0:
                print(f"{k}/{len(seeds)} families, {time.perf_counter() - t:.0f}s", flush=True)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    valid = [r for r in rows if r["valid"]]
    cf = [e["counterfactual_min_pair_px"] for r in valid for e in r["per_class"].values()]
    replay = [e["counterfactual_true_matches_run_px"] for r in valid for e in r["per_class"].values()]
    turns = {c: [r["per_class"][c]["probe_turn_deg"] for r in valid] for c in m.COG_CLASSES}
    summary = {
        "families": len(rows), "valid": len(valid), "probe_pool_size": len(pool_rows),
        "invalid_reasons": {k: sum(1 for r in rows if not r["valid"] and bad(r)) for k, bad in {
            "reset_differs": lambda r: not r["reset_frames_identical"],
            "approach_contact": lambda r: r["approach_contacts"] > 0,
            "transit_gap_contact": lambda r: r["transit_gap_contacts"] > 0,
            "out_of_bounds": lambda r: not (r["block_in_bounds"] and r["actions_in_arena"]),
            "test_contact_before_query": lambda r: r["first_test_contact_frame"] <= r["query_frame"],
        }.items()},
        "pre_contact_frames_identical": sum(r["pre_contact_frames_identical"] for r in valid),
        "probe_turn_deg_q10_50_90": {c: np.quantile(v, [.1, .5, .9]).round(1).tolist() for c, v in turns.items()}
        if valid else None,
        "counterfactual_min_pair_px_q10_50_90": np.quantile(cf, [.1, .5, .9]).tolist() if cf else None,
        "counterfactual_replay_error_px_max": max(replay) if replay else None,
        "state_classifier_accuracy": leak_table(valid) if len(valid) >= 10 else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
