"""Development run for design v3 (matched probes): library, then design + check families.

CPU only. Families run in a process pool; each result is appended to
`families.jsonl` as it finishes, so an interrupted job keeps what it did.
Dev seeds only: these families are inspected and must stay out of any split.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_cog_matched as m

LIBRARY = POOL = None


def _init(library, pool):
    global LIBRARY, POOL
    LIBRARY, POOL = library, pool


def run_family(seed: int):
    started = time.perf_counter()
    spec = m.sample_family(seed)
    draws = []
    design = m.design_family(spec, LIBRARY, pool=POOL, log=draws)
    if design is None:
        return {"seed": seed, "matched": False, "draws": draws, "seconds": time.perf_counter() - started}
    result, _, _ = m.check_family(spec, design)
    result.update(seed=seed, matched=True, anchor_class=design["anchor_class"], draws=draws,
                  seconds=time.perf_counter() - started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=2026130000)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "families.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines() if line.strip()}

    t = time.perf_counter()
    library = m.build_library(args.library, processes=args.processes)
    pool_rows = m.anchor_pool(library)
    print(f"library: {len(library)} rows, anchor pool {len(pool_rows)}, {time.perf_counter() - t:.0f}s", flush=True)

    seeds = [args.base_seed + k for k in range(args.families) if args.base_seed + k not in done]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(args.processes, initializer=_init, initargs=(library, pool_rows)) as pool, out.open("a") as f:
        for result in pool.imap_unordered(run_family, seeds):
            f.write(json.dumps(result) + "\n")
            f.flush()
            if result["matched"]:
                print(result["seed"], f"{result['seconds']:.0f}s valid={result['valid']}",
                      f"draw={result['anchor_draw']}", f"turn={result['shared_turn_deg']:.0f}",
                      f"err={max(result['match_error'].values()):.1e}",
                      f"tgt_px={result['target_min_pair_px']:.1f}", flush=True)
            else:
                print(result["seed"], f"{result['seconds']:.0f}s NO MATCH", flush=True)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    matched = [r for r in rows if r["matched"]]
    valid = [r for r in matched if r["valid"]]
    all_draws = [d for r in rows for d in r.get("draws", [])]
    reasons = {}
    for d in all_draws:
        reasons[d.get("rejected", "matched")] = reasons.get(d.get("rejected", "matched"), 0) + 1
    summary = {
        "families": len(rows), "matched": len(matched), "valid": len(valid),
        "anchor_pool_size": len(pool_rows), "draw_outcomes": reasons,
        "failed_draw_best_error_median": (float(np.median([v for d in all_draws for k, v in d.items()
                                                           if k.endswith("_best_error") and v is not None
                                                           and v >= m.MATCH_TOL]))
                                          if any(d.get("rejected", "").startswith("no_match") for d in all_draws)
                                          else None),
        "anchor_draws_median": float(np.median([r["anchor_draw"] for r in matched])) if matched else None,
        "seconds_median": float(np.median([r["seconds"] for r in rows])) if rows else None,
        "max_match_error": max((max(r["match_error"].values()) for r in matched), default=None),
        "reset_frames_identical": sum(r["reset_frames_identical"] for r in matched),
        "converge_slack_min": min((r["query_frame"] - m.RECENT_LENGTH + 1 - r["converge_frame"] for r in valid),
                                  default=None),
        "target_min_pair_px_q10_50_90": (np.quantile([r["target_min_pair_px"] for r in valid], [.1, .5, .9]).tolist()
                                         if valid else None),
        "invalid_reasons": {k: sum(1 for r in matched if not r["valid"] and bad(r)) for k, bad in {
            "approach_contact": lambda r: r["approach_contacts"] > 0,
            "transit_gap_contact": lambda r: r["transit_gap_contacts"] > 0,
            "reset_differs": lambda r: not r["reset_frames_identical"],
            "late_converge": lambda r: r["converge_frame"] > r["query_frame"] - m.RECENT_LENGTH + 1,
            "out_of_bounds": lambda r: not (r["block_in_bounds"] and r["actions_in_arena"]),
        }.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
