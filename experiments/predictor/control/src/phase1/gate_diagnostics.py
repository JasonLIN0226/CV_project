"""Export compact, paired gate calibration from unchanged, frozen v3 models.

No source-run file is written. Sibling predictions are evaluation diagnostics,
not legal deployment inputs and not a Phase 1 adoption result.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .config import CONDITIONS, SPLITS
from .evaluation import pair_calibration, summarize_calibration, training_completion
from .inputs import FeatureStore, collate
from .training import load_predictor, paired_rows
from .utils import code_hash, file_hash, read_json, seed_all, write_json


def _verify_summary(config, actual, recorded):
    keys = ("seed", "task", "split", "condition", "horizon")
    recomputed = summarize_calibration(config, actual)["per_seed"]
    expected = {tuple(row[k] for k in keys): row for row in recorded["per_seed"]}
    observed = {tuple(row[k] for k in keys): row for row in recomputed}
    if set(expected) != set(observed):
        raise ValueError("Frozen calibration slice coverage differs from original evaluation")
    maximum = 0.0
    for key in expected:
        for metric, original in expected[key]["metrics"].items():
            current = observed[key]["metrics"][metric]
            if (current is None) != (original is None):
                raise ValueError("Frozen calibration degeneracy differs from original evaluation")
            if original is not None:
                difference = abs(current["mean"] - original["mean"])
                maximum = max(maximum, difference)
                if difference > 1e-7:
                    raise ValueError(f"Frozen calibration changed {key}/{metric}: {difference}")
    return maximum


def export_gate_pairs(development_output, output):
    source, destination = Path(development_output).resolve(), Path(output).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Diagnostic output must be outside the immutable v3 run")
    result = {"schema_version": 1, "scope": "frozen_gate_common_diagnostic", "arms": {},
              "independent_phase1_result": False,
              "diagnostic_code_hash": code_hash("gate_diagnostics.py"),
              "interpretation": "Paired offline diagnostics; no training or checkpoint reselection."}
    for arm in ("reference", "gate_no_rank"):
        root = source / "gate" / arm
        config = read_json(root / "resolved_config.json")
        config["output"], config["device"] = str(root), "cpu"
        if config.get("experiment", {}).get("arm") != arm or config["experiment"].get("tasks") != ["gate"]:
            raise ValueError("Wrong frozen gate arm")
        if config["mode"] != "smoke" and (config["data"]["seed"] != 6301 or
                config["training"]["seeds"] != [17, 29, 43] or config["training"]["epochs"] != 120):
            raise ValueError("Not the completed, fixed v3 development source")
        seed_all(config["training"]["seeds"][0], config["threads"])
        metrics = read_json(root / "evaluation" / "metrics.json")
        raw_sha = file_hash(root / "evaluation" / "raw_records.csv")
        if raw_sha != metrics["raw_sha256"]:
            raise ValueError("Original raw-record checksum mismatch")
        if metrics["evaluation_code_hash"] != code_hash("evaluation.py", "stats.py"):
            raise ValueError("Original evaluation implementation has changed")
        store = FeatureStore(config, max_cached=2 * max(config["data"]["pairs"].values()))
        completion = training_completion(config, store)
        if not completion["complete"] or completion != metrics["training_completion"]:
            raise ValueError("Current completed models differ from original evaluation proofs")
        if store.ix["signature"] != metrics["cache_signature"]:
            raise ValueError("Original feature cache signature mismatch")
        rows, checkpoints = [], {}
        for seed in config["training"]["seeds"]:
            model, checkpoint = load_predictor(config, seed, store, torch.device("cpu"), "gate")
            key = f"gate/{seed}"
            proof = {"best_sha256": file_hash(root / "predictor" / "gate" / f"seed_{seed}" / "best.pt"),
                     "best_epoch": checkpoint["epoch"]}
            if any(proof[k] != metrics["checkpoints"][key][k] for k in proof):
                raise ValueError("Frozen model differs from the originally evaluated checkpoint")
            checkpoints[key] = proof
            for split in SPLITS[1:]:
                records = [r for r in store.manifest["records"] if r["task"] == "gate" and r["split"] == split]
                pairs = paired_rows(records)
                if len(pairs) != config["data"]["pairs"][split]:
                    raise ValueError("Incomplete frozen diagnostic split")
                for condition in CONDITIONS + ["persistence"]:
                    # Use the same episode ordering and minibatch sizes as the original evaluator.
                    for start in range(0, len(pairs), config["training"]["batch_size"] // 2):
                        part = pairs[start:start + config["training"]["batch_size"] // 2]
                        ordered = [rec for pair in part for rec in pair]
                        samples = [store.sample(rec, condition if condition != "persistence" else "recent_only")
                                   for rec in ordered]
                        target = np.stack([sample[1] for sample in samples])
                        if condition == "persistence":
                            prediction = np.stack([np.tile(sample[0]["current"],
                                (len(config["data"]["horizons"]), 1)) for sample in samples])
                        else:
                            with torch.inference_mode():
                                prediction = model(collate([sample[0] for sample in samples], torch.device("cpu"))).numpy()
                        for pair_index, pair in enumerate(part):
                            order = sorted(range(2), key=lambda i: pair[i]["id"])
                            indices = [2 * pair_index + i for i in order]
                            for horizon_index, horizon in enumerate(config["data"]["horizons"]):
                                p, y = prediction[indices, horizon_index], target[indices, horizon_index]
                                equal = bool(np.array_equal(p[0], p[1])) if condition == "recent_only" else None
                                max_abs = float(np.max(np.abs(p[0] - p[1]))) if condition == "recent_only" else None
                                if condition == "recent_only" and not equal:
                                    raise ValueError("Recent sibling predictions are not exactly identical")
                                rows.append({"seed": seed, "task": "gate", "split": split,
                                    "condition": condition, "horizon": horizon, "pair": pair[0]["pair"],
                                    **pair_calibration(p, y), "recent_prediction_equal": equal,
                                    "recent_prediction_max_abs_difference": max_abs})
                print(f"frozen gate calibration: {arm} seed={seed} split={split}", flush=True)
        error = _verify_summary(config, rows, read_json(root / "evaluation" / "calibration_summary.json"))
        result["arms"][arm] = {"raw_sha256": raw_sha, "cache_signature": metrics["cache_signature"],
            "evaluation_code_hash": metrics["evaluation_code_hash"], "checkpoints": checkpoints,
            "training_completion": completion, "original_calibration_max_mean_difference": error,
            "row_count": len(rows), "rows": rows}
    write_json(destination / "gate_pair_calibration.json", result)
    print(f"compact paired calibration exported: {destination / 'gate_pair_calibration.json'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    export_gate_pairs(args.development_output, args.output)


if __name__ == "__main__":
    main()
