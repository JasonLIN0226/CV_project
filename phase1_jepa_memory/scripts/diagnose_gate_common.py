#!/usr/bin/env python3
"""Audit frozen gate common replacement using compact scalar artifacts only.

No model, cache arrays, or checkpoint is loaded. Marginal calibration summaries
identify the hybrid mean but cannot identify a paired-bootstrap interval. An
optional gate_pair_calibration.json supplies the missing complete-pair scalars.
Its schema is described in the output JSON even when it has not been exported.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ARMS = ("reference", "gate_no_rank")
SEEDS = (17, 29, 43)
HORIZONS = (3, 6)
SPLITS = ("validation", "test_id", "test_long_gap", "test_distractor_shift")
CONDITIONS = ("recent_only", "full_history", "privileged_small_budget",
              "random_small_budget", "wrong_evidence", "persistence")
METRICS = ("common_mse", "contrast_mse", "total_mse", "decomposition_residual",
           "alpha", "target_contrast_mse", "orthogonal_contrast_mse")
ATOL = 1e-7  # Original raw MSE is float32; calibration is computed in float64.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=ATOL)


def mean(values):
    return float(np.mean(values))


def key(row):
    return tuple(row[k] for k in ("seed", "split", "horizon", "condition"))


def load_stats():
    path = ROOT / "src/phase1/stats.py"
    spec = importlib.util.spec_from_file_location("gate_common_original_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_shared(dev, source, sources):
    paths = {
        "reuse_manifest": dev / "shared/reuse_manifest.json",
        "manifest": dev / "shared/dataset/manifest.json",
        "data_audit": dev / "shared/dataset/split_audit.json",
        "feature_audit": source / "features/pair_feature_audit.json",
        "feature_index": source / "features/index.json",
        "normalization": source / "features/normalization.json",
        "verification": dev / "verification.json",
    }
    values = {name: read_json(path) for name, path in paths.items() if name != "feature_index"}
    sources.update({name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()})
    reuse, audit, feature = (values[n] for n in ("reuse_manifest", "data_audit", "feature_audit"))
    require(audit["passed"] and {"exact_recent_rgb", "exact_all_actions", "pair_split_disjoint"}.issubset(audit["checks"]),
            "Data audit does not establish exact shared recent observations/actions and pair isolation")
    require(feature["passed"] and feature["exact_recent_feature_equality"], "Exact recent feature audit is missing")
    require(feature["signature"] == reuse["cache_signature"], "Feature audit cache signature mismatch")
    require(feature["index_sha256"] == reuse["index_sha256"] == sources["feature_index"]["sha256"], "Feature index checksum mismatch")
    norm = values["normalization"]
    require(norm["sha256"] == reuse["normalization_sha256"] and norm["cache_signature"] == reuse["cache_signature"] and
            norm["fit_split"] == "train", "Normalization metadata provenance mismatch")
    require(audit["signature"] == values["manifest"]["signature"] == reuse["dataset_signature"], "Dataset signature mismatch")
    records = values["manifest"]["records"]
    require(feature["pairs"] == len({r["pair"] for r in records}), "Feature audit did not cover every base pair")
    by = {r["id"]: r for r in records}
    require(len(by) == len(records), "Duplicate manifest episode")
    for r in records:
        sibling = by[r["sibling"]]
        require(sibling["sibling"] == r["id"] and sibling["id"] != r["id"] and
                all(sibling[k] == r[k] for k in ("task", "split", "pair", "query", "gap")),
                "Manifest sibling/pair/timing mismatch")
    values["records_by_id"] = by
    return values


def load_arm(dev, arm, shared, sources):
    root = dev / "gate" / arm
    paths = {"config": root / "resolved_config.json", "metrics": root / "evaluation/metrics.json",
             "calibration": root / "evaluation/calibration_summary.json", "raw": root / "evaluation/raw_records.csv"}
    sources[arm] = {n: {"path": str(p), "sha256": sha(p)} for n, p in paths.items()}
    config, metrics, calibration = (read_json(paths[n]) for n in ("config", "metrics", "calibration"))
    known = shared["verification"]["actual_hpc_compact_sha256"]
    for p in paths.values():
        require(known.get(str(p.relative_to(dev))) == sha(p), f"{arm}: artifact differs from verified remote compact mirror: {p.name}")
    require(config["experiment"] == {"stage": "development", "arm": arm, "tasks": ["gate"]}, "Unexpected experiment scope")
    require(config["training"]["seeds"] == list(SEEDS) and config["training"]["epochs"] == 120 and
            config["data"]["horizons"] == list(HORIZONS), "Unexpected seed/epoch/horizon protocol")
    require(config["evaluation"]["bootstrap"] == 2000 and config["decision"]["latent_relative_improvement"] == .1,
            "Original bootstrap repeats or improvement threshold changed")
    require(config["model"]["dropout"] == 0, "Recent equality proof requires deterministic evaluation")
    require(metrics["raw_sha256"] == sha(paths["raw"]), "Raw records checksum mismatch")
    eval_hash = digest({n: sha(ROOT / "src/phase1" / n) for n in ("evaluation.py", "stats.py")})
    require(metrics["evaluation_code_hash"] == eval_hash, "Original evaluation/statistics implementation changed")
    require(metrics["cache_signature"] == calibration["cache_signature"] == shared["reuse_manifest"]["cache_signature"], "Arm cache signature mismatch")
    completion = metrics["training_completion"]
    require(completion["complete"] and completion["required_epochs"] == 120, "Missing complete training proof")
    for seed in SEEDS:
        proof, ck = completion["models"][f"gate/{seed}"], metrics["checkpoints"][f"gate/{seed}"]
        logpath = root / "predictor/gate" / f"seed_{seed}" / "training.json"
        log = read_json(logpath)
        require(sha(logpath) == proof["training_json_sha256"], "Training log checksum mismatch")
        require(proof["complete"] and proof["completed_epochs"] == len(log["logs"]) == 120 and
                proof["selection_metric"] == "validation_mse" and log["signature"] == proof["signature"], "Training completion/provenance mismatch")
        signature, reuse = proof["signature"], shared["reuse_manifest"]
        require(signature["task"] == "gate" and signature["experiment"] == config["experiment"] and
                signature["cache_signature"] == reuse["cache_signature"] and
                signature["normalization_sha256"] == reuse["normalization_sha256"] and
                signature["basis_sha256"] == reuse["basis_sha256"]["gate"], "Checkpoint cache/normalization/basis/task provenance mismatch")
        require(ck["best_sha256"] == proof["best_sha256"] and ck["best_epoch"] == proof["best_epoch"] ==
                min(log["logs"], key=lambda r: r["validation_mse"])["epoch"], "Selected checkpoint does not match fixed validation criterion")
    raw = defaultdict(dict)
    numeric_int = {"seed", "horizon", "gap", "query", "history_count"}
    strings = {"task", "split", "condition", "episode", "pair"}
    with paths["raw"].open(newline="") as stream:
        for item in csv.DictReader(stream):
            r = {k: v if k in strings else int(v) if k in numeric_int else float(v) for k, v in item.items()}
            require(all(math.isfinite(v) for v in r.values() if isinstance(v, (int, float))), "Nonfinite raw record")
            require(r["task"] == "gate" and r["episode"] not in raw[key(r)], "Duplicate or wrong-task record")
            m = shared["records_by_id"][r["episode"]]
            require(all(r[k] == m[k] for k in ("task", "split", "pair", "query", "gap")), "Raw/manifest pair mismatch")
            raw[key(r)][r["episode"]] = r
    summary = {key(r): r for r in calibration["per_seed"]}
    metric_summary = {key(r): r for r in metrics["per_seed"]}
    expected = {(seed, split, h, cond) for seed in SEEDS for split in SPLITS for h in HORIZONS for cond in CONDITIONS}
    require(set(raw) == set(summary) == expected and len(summary) == len(calibration["per_seed"]), "Missing or extra seed/split/horizon/condition")
    for k, rows in raw.items():
        seed, split, h, cond = k
        episodes = {r["id"] for r in shared["records_by_id"].values() if r["task"] == "gate" and r["split"] == split}
        require(set(rows) == episodes, "Raw slice lacks an expected sibling/episode")
        pairs = defaultdict(list)
        for r in rows.values():
            pairs[r["pair"]].append(r)
        require(len(pairs) == config["data"]["pairs"][split] and all(len(v) == 2 for v in pairs.values()), "Incomplete raw pair")
        sm = summary[k]["metrics"]
        for metric in METRICS:
            require(sm[metric] is not None and all(math.isfinite(sm[metric][v]) for v in ("mean", "low", "high")), "Missing/nonfinite calibration metric")
            require(sm[metric]["pairs"] == len(pairs), "Calibration summary pair count mismatch")
        require(close(sm["total_mse"]["mean"], mean([r["normalized_mse"] for r in rows.values()])) and
                close(sm["total_mse"]["mean"], metric_summary[k]["metrics"]["normalized_mse"]["mean"]), "Calibration/raw normalized MSE mismatch")
        require(close(sm["total_mse"]["mean"], sm["common_mse"]["mean"] + sm["contrast_mse"]["mean"]) and
                abs(sm["decomposition_residual"]["mean"]) <= ATOL, "Calibration decomposition mismatch")
        require(close(sm["target_contrast_mse"]["mean"], mean([r["target_pair_distance"] / 4 for r in rows.values()])), "Raw target separation has wrong normalization")
        if cond == "recent_only":
            require(sm["alpha"]["mean"] == 0 and sm["orthogonal_contrast_mse"]["mean"] == 0 and
                    close(sm["contrast_mse"]["mean"], sm["target_contrast_mse"]["mean"]), "Recent contrast is not zero in original calibration")
            for a, b in pairs.values():
                require(a["history_count"] == b["history_count"] == 0 and
                        a["state_pred_x"] == b["state_pred_x"] and a["state_pred_y"] == b["state_pred_y"] and
                        close(a["pair_margin"], -b["pair_margin"]), "Recent raw sibling equality corroboration failed")
    return {"config": config, "metrics": metrics, "raw": raw, "summary": summary,
            "checks": ["verified remote file hashes", "all 120-epoch selected checkpoint/log proofs",
                       "all raw episodes matched to complete manifest siblings", "all slice summaries match raw normalized MSE",
                       "exact paired recent RGB/actions/features audit", "zero recent contrast and equal decoded outputs corroborate shared predictions"]}


def load_pair_rows(export, arm, data, stats):
    require(export["schema_version"] == 1 and export["scope"] == "frozen_gate_common_diagnostic", "Wrong pair export schema/scope")
    part, metrics = export["arms"][arm], data["metrics"]
    for k in ("raw_sha256", "cache_signature", "evaluation_code_hash"):
        require(part[k] == metrics[k], f"Pair export {k} does not match original evaluation")
    for name, ck in metrics["checkpoints"].items():
        require(all(part["checkpoints"][name][k] == ck[k] for k in ("best_sha256", "best_epoch")), "Pair export checkpoint mismatch")
    grouped = defaultdict(dict)
    for r in part["rows"]:
        require(r["task"] == "gate" and r["pair"] not in grouped[key(r)], "Duplicate or wrong-task pair export row")
        require(all(r[m] is not None and math.isfinite(r[m]) for m in METRICS), "Missing/nonfinite pair calibration scalar")
        require(not r["degenerate_target_contrast"], "Degenerate target pair cannot satisfy the diagnostic proof")
        require(abs(r["decomposition_residual"]) <= ATOL and close(r["total_mse"], r["common_mse"] + r["contrast_mse"]), "Pair calibration identity failed")
        grouped[key(r)][r["pair"]] = r
    require(set(grouped) == set(data["raw"]), "Pair export slice coverage differs")
    for k, pairs in grouped.items():
        rr = defaultdict(list)
        order = []
        for r in data["raw"][k].values():
            rr[r["pair"]].append(r)
            if len(rr[r["pair"]]) == 2:
                order.append(r["pair"])
        require(set(pairs) == set(rr), "Pair export lacks exact original pair IDs")
        for pid, r in pairs.items():
            require(close(r["total_mse"], mean([v["normalized_mse"] for v in rr[pid]])) and
                    close(r["target_contrast_mse"], mean([v["target_pair_distance"] / 4 for v in rr[pid]])), "Pair scalar differs from original raw predictions/targets")
            if k[3] == "recent_only":
                require(r["recent_prediction_equal"] is True and r["recent_prediction_max_abs_difference"] == 0,
                        "Direct frozen recent sibling output equality proof missing")
                require(r["alpha"] == 0 and r["orthogonal_contrast_mse"] == 0 and
                        close(r["contrast_mse"], r["target_contrast_mse"]), "Recent predicted contrast is not zero")
        # Original calibration insertion order follows completion of each pair
        # during the original raw-record traversal, not sorted pair IDs.
        for m in METRICS:
            got = stats.clustered_interval([pairs[p][m] for p in order], order, data["config"]["evaluation"]["bootstrap"], seed=0)
            want = data["summary"][k]["metrics"][m]
            require(all(close(got[v], want[v]) for v in ("mean", "low", "high")), f"Export cannot reproduce original calibration {m} interval")
    return grouped


def effect_point(base, alternative, rules):
    absolute = base < rules["baseline_floor"]
    value = base - alternative if absolute else (base - alternative) / max(base, rules["baseline_floor"])
    return {"mean": value, "low": None, "high": None,
            "scale": "absolute_error_reduction" if absolute else "relative_error_reduction",
            "absolute_threshold": rules["absolute_latent_gain"] if absolute else None,
            "baseline_error": base}


def summarize_arm(data, pair_rows, stats):
    rows, rules = [], data["config"]["decision"]
    repeats = data["config"]["evaluation"]["bootstrap"]
    for split in SPLITS:
        for h in HORIZONS:
            for seed in (*SEEDS, "all_seeds_mean"):
                seeds = SEEDS if seed == "all_seeds_mean" else [seed]
                for condition in CONDITIONS:
                    own = [data["summary"][(s, split, h, condition)]["metrics"] for s in seeds]
                    recent = [data["summary"][(s, split, h, "recent_only")]["metrics"] for s in seeds]
                    base = mean([r["total_mse"]["mean"] for r in recent])
                    original = mean([r["total_mse"]["mean"] for r in own])
                    rc = mean([r["common_mse"]["mean"] for r in recent])
                    hc = mean([r["contrast_mse"]["mean"] for r in own])
                    hybrid = rc + hc
                    interval = effect_point(base, hybrid, rules)
                    reduction = {"mean": original - hybrid, "low": None, "high": None}
                    threshold = rules["latent_relative_improvement"] if interval["scale"].startswith("relative") else rules["absolute_latent_gain"]
                    if pair_rows is not None:
                        # Match original comparison order: sorted episodes, first
                        # encounter of a pair. Average seeds within each pair.
                        raw_slice = data["raw"][(SEEDS[0], split, h, "recent_only")]
                        order = list(dict.fromkeys(raw_slice[e]["pair"] for e in sorted(raw_slice)))
                        per_pair = []
                        for p in order:
                            r = [pair_rows[(s, split, h, "recent_only")][p] for s in seeds]
                            a = [pair_rows[(s, split, h, condition)][p] for s in seeds]
                            per_pair.append([mean([v["total_mse"] for v in r]),
                                             mean([v["common_mse"] for v in r]) + mean([v["contrast_mse"] for v in a]),
                                             mean([v["total_mse"] for v in a])])
                        v = np.asarray(per_pair)
                        require(close(mean(v[:, 1]), hybrid), "Pair and marginal hybrid means differ")
                        interval = stats.paired_improvement(v[:, 0], v[:, 1], order, repeats,
                                                           rules["baseline_floor"], rules["absolute_latent_gain"], seed=0)
                        reduction = stats.clustered_interval(v[:, 2] - v[:, 1], order, repeats, seed=0)
                    passed = interval["mean"] >= threshold and interval["low"] > 0 if interval["low"] is not None else None
                    rows.append({"seed": seed, "task": "gate", "split": split, "horizon": h, "condition": condition,
                                 "pairs": data["config"]["data"]["pairs"][split], "recent_normalized_mse": base,
                                 "original_history_normalized_mse": original, "recent_common_mse": rc,
                                 "history_common_mse": mean([r["common_mse"]["mean"] for r in own]),
                                 "unchanged_history_contrast_mse": hc, "hybrid_normalized_mse": hybrid,
                                 "original_vs_recent_point": effect_point(base, original, rules),
                                 "hybrid_vs_recent": interval, "absolute_reduction_vs_original_history": reduction,
                                 "hybrid_worse_than_original_history": hybrid > original + ATOL,
                                 "point_threshold": threshold, "point_threshold_met": interval["mean"] >= threshold,
                                 "diagnostic_rule_passed": passed,
                                 "status": "INCOMPLETE" if passed is None else "SUPPORTED" if passed else "NOT_SUPPORTED"})
    target = [r for r in rows if r["split"] == "test_distractor_shift" and r["condition"] == "privileged_small_budget" and r["seed"] != "all_seeds_mean"]
    require(len(target) == 6, "Diagnostic core must retain every seed and horizon")
    status = "INCOMPLETE" if any(r["diagnostic_rule_passed"] is None for r in target) else "SUPPORTED" if all(r["diagnostic_rule_passed"] for r in target) else "NOT_SUPPORTED"
    return {"status": status, "core_slice_count": len(target), "core_point_thresholds_met": sum(r["point_threshold_met"] for r in target),
            "core_passed_count": None if status == "INCOMPLETE" else sum(r["diagnostic_rule_passed"] for r in target),
            "id_regressions_vs_original_history": [r for r in rows if r["split"] == "test_id" and r["condition"] == "privileged_small_budget" and r["hybrid_worse_than_original_history"]],
            "slices": rows}


def markdown(result):
    lines = ["# Frozen gate common replacement diagnostic", "", f"**{result['status']} — diagnostic only; not Phase 1 GO.**", "",
             "For paired targets m±d and frozen history predictions c±v, replace only c by the same checkpoint's shared recent prediction r. "
             "The hypothetical pair-average normalized MSE is recent common MSE + unchanged history contrast MSE. No new predictions are created.", "",
             "This is an oracle pair diagnostic: evaluating the construction requires both sibling predictions. It is not a deployable single-trajectory model, "
             "does not establish that common error is background error, and does not predict that an anchor-training loss will work.", "",
             "Core criterion: privileged B2 distractor shift must meet the original 10% point improvement (or original low-baseline absolute rule) "
             "and a strictly positive 95% paired CI lower bound in all three seeds × both horizons. Bootstrap: 2,000 resamples, seed 0, complete base pairs; "
             "pooled seeds are averaged within each pair, and pooled CIs are conditional on those seeds.", ""]
    for issue in result.get("issues", []):
        lines.append(f"- {issue}")
    for arm, output in result["arms"].items():
        lines += [f"## {arm}: {output['status']}", ""]
        for issue in output.get("issues", []):
            lines.append(f"- {issue}")
        if not output.get("slices"):
            continue
        lines += ["", "| Split | Seed | h | Recent MSE | History MSE | Hybrid MSE | Hybrid gain | 95% paired CI | History→hybrid MSE reduction | Status |",
                  "|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
        for r in output["slices"]:
            if r["condition"] != "privileged_small_budget":
                continue
            v = r["hybrid_vs_recent"]
            relative = v["scale"].startswith("relative")
            fmt = (lambda x: f"{100*x:.3f}%") if relative else (lambda x: f"{x:.8f}")
            ci = "unavailable" if v["low"] is None else f"[{fmt(v['low'])}, {fmt(v['high'])}]"
            lines.append(f"| {r['split']} | {r['seed']} | {r['horizon']} | {r['recent_normalized_mse']:.8f} | {r['original_history_normalized_mse']:.8f} | "
                         f"{r['hybrid_normalized_mse']:.8f} | {fmt(v['mean'])} | {ci} | {r['absolute_reduction_vs_original_history']['mean']:.8f} | {r['status']} |")
        regression = output["id_regressions_vs_original_history"]
        lines += ["", f"ID slices where replacement increases privileged error: {len(regression)} (including pooled slices)."]
        for r in regression:
            lines.append(f"- seed {r['seed']}, h{r['horizon']}: {r['original_history_normalized_mse']:.8f} → {r['hybrid_normalized_mse']:.8f}.")
    lines += ["", "All six conditions, validation/ID/long-gap/distractor splits, per-seed and pooled values are retained in gate_common.json. "
              "Missing pair calibration is INCOMPLETE: adding/subtracting marginal confidence limits would not recover the required paired interval.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=ROOT / "runs/phase1_v3_dev")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/phase1_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/phase1_v4_diagnostics")
    parser.add_argument("--pair-calibration", type=Path)
    args = parser.parse_args()
    pair_path = args.pair_calibration or args.output / "gate_pair_calibration.json"
    result = {"schema_version": 1, "status": "INCOMPLETE", "scope": "frozen pair-oracle mechanism diagnostic; not GO or a deployable model",
              "formula": "hybrid_pair_normalized_mse = same_checkpoint_recent_common_mse + unchanged_history_contrast_mse",
              "statistics": {"bootstrap_repeats": 2000, "bootstrap_seed": 0, "confidence": .95,
                             "unit": "complete base pair", "pooled": "mean within pair across fixed seeds, then bootstrap pairs",
                             "note": "config evaluation.random_seed=901 controls memory sampling, not the original bootstrap seed"},
              "inputs": {}, "arms": {}, "limitations": ["No semantic attribution of the common component", "No single-trajectory deployment claim",
              "Revealed v3 data remain development evidence", "No arrays/checkpoints loaded or scientific source edited"],
              "pair_export_contract": {"schema_version": 1, "scope": "frozen_gate_common_diagnostic",
                  "arms": {"<reference|gate_no_rank>": {"raw_sha256": "original raw SHA", "cache_signature": "original cache signature",
                    "evaluation_code_hash": "original evaluator hash", "checkpoints": {"gate/<seed>": {"best_sha256": "original best SHA", "best_epoch": "original best epoch"}},
                    "rows": [{"seed": "int", "task": "gate", "split": "str", "horizon": "int", "condition": "str", "pair": "original pair ID",
                       **{m: "original float64 pair_calibration scalar" for m in METRICS}, "degenerate_target_contrast": "bool",
                       "recent_prediction_equal": "actual np.array_equal for recent_only, else null",
                       "recent_prediction_max_abs_difference": "actual max abs difference for recent_only, else null"}]}}}}
    try:
        shared = validate_shared(args.development, args.source, result["inputs"])
        stats = load_stats()
        export = read_json(pair_path) if pair_path.exists() else None
        if export is not None:
            result["inputs"]["pair_calibration"] = {"path": str(pair_path), "sha256": sha(pair_path)}
        for arm in ARMS:
            try:
                data = load_arm(args.development, arm, shared, result["inputs"])
                missing = "Per-pair calibration was not saved in the original compact results; marginal summaries cannot identify a paired bootstrap CI."
                pairs, issues = None, []
                if export is not None:
                    try:
                        pairs = load_pair_rows(export, arm, data, stats)
                    except (ValueError, KeyError, TypeError) as exc:
                        issues.append(f"Pair export verification failed: {exc}")
                else:
                    issues.append(missing)
                output = summarize_arm(data, pairs, stats)
                output.update({"issues": issues, "evidence_checks": data["checks"], "pair_export_verified": pairs is not None,
                               "direct_recent_prediction_equality_verified": pairs is not None})
                result["arms"][arm] = output
            except (ValueError, KeyError, TypeError, OSError) as exc:
                result["arms"][arm] = {"status": "INCOMPLETE", "issues": [str(exc)], "slices": []}
        # Reference is the predeclared primary diagnostic; no-rank is a separate
        # historical replication, never selected to rescue a failing reference.
        result["status"] = result["arms"]["reference"]["status"]
        result["primary_arm"] = "reference"
        result["secondary_arm"] = "gate_no_rank"
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result["issues"] = [str(exc)]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gate_common.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "gate_common.md").write_text(markdown(result))
    print(json.dumps({"status": result["status"], "arms": {k: {n: v.get(n) for n in ("status", "core_point_thresholds_met", "core_passed_count", "issues")} for k, v in result["arms"].items()}}, indent=2))
    return 2 if result["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
