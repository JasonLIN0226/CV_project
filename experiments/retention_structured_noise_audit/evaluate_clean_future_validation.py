"""Evaluate the clean-future Gate experiment without opening any test data."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CONTROL_SOURCE = HERE.parent / "predictor" / "control" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from phase1.config import CONDITIONS  # noqa: E402
from phase1.evaluation import METRICS  # noqa: E402
from phase1.stats import clustered_interval, paired_improvement  # noqa: E402
from phase1.utils import file_hash, read_json, seed_all, write_json  # noqa: E402

from evaluate_gate import (  # noqa: E402
    evaluate_model,
    load_candidate,
    load_reference,
)


def _mean_rows(rows):
    return float(np.mean(rows))


def summarize_validation(c, raw):
    groups = defaultdict(list)
    for row in raw:
        groups[
            (
                row["seed"],
                row["task"],
                row["split"],
                row["condition"],
                row["horizon"],
            )
        ].append(row)
    per_seed = []
    for key, rows in groups.items():
        item = dict(
            zip(
                ("seed", "task", "split", "condition", "horizon"),
                key,
            )
        )
        item["metrics"] = {
            metric: clustered_interval(
                [row[metric] for row in rows],
                [row["pair"] for row in rows],
                c["evaluation"]["bootstrap"],
            )
            for metric in METRICS
        }
        item["target_pair_distance_quantiles"] = np.quantile(
            [row["target_pair_distance"] for row in rows],
            [0.0, 0.1, 0.5, 0.9, 1.0],
        ).tolist()
        item["almost_indistinguishable_fraction"] = float(
            np.mean(
                [
                    row["target_pair_distance"]
                    < c["decision"]["pair_distance_floor"]
                    for row in rows
                ]
            )
        )
        per_seed.append(item)

    pooled = defaultdict(list)
    for row in raw:
        pooled[
            (
                row["task"],
                row["split"],
                row["condition"],
                row["horizon"],
                row["episode"],
            )
        ].append(row)
    averaged = [
        {
            **rows[0],
            **{
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in METRICS
            },
        }
        for rows in pooled.values()
    ]
    comparisons = []
    for horizon in c["data"]["horizons"]:
        for seed in c["training"]["seeds"] + ["all_seeds_mean"]:
            data = averaged if seed == "all_seeds_mean" else [
                row for row in raw if row["seed"] == seed
            ]
            subset = [
                row
                for row in data
                if row["task"] == "gate"
                and row["split"] == "validation"
                and row["horizon"] == horizon
            ]
            by = {
                (row["condition"], row["episode"]): row
                for row in subset
            }
            episodes = sorted({row["episode"] for row in subset})
            for alternative in (
                "privileged_small_budget",
                "full_history",
                "random_small_budget",
            ):
                base = [by[("recent_only", episode)] for episode in episodes]
                better = [by[(alternative, episode)] for episode in episodes]
                improvements = {}
                for metric in (
                    "normalized_mse",
                    "latent_mse",
                    "state_position_mse",
                    "state_accuracy",
                    "pair_accuracy",
                ):
                    lower_better = metric.endswith("mse")
                    first = [
                        row[metric] if lower_better else 1.0 - row[metric]
                        for row in base
                    ]
                    second = [
                        row[metric] if lower_better else 1.0 - row[metric]
                        for row in better
                    ]
                    floor = (
                        c["decision"]["position_baseline_floor"]
                        if metric == "state_position_mse"
                        else 1e-8
                        if metric == "latent_mse"
                        else c["decision"]["baseline_floor"]
                    )
                    absolute_gain = (
                        c["decision"]["absolute_position_mse_gain"]
                        if metric == "state_position_mse"
                        else c["decision"]["absolute_latent_gain"]
                        if metric.endswith("mse")
                        else c["decision"]["absolute_state_gain"]
                    )
                    improvements[metric] = paired_improvement(
                        first,
                        second,
                        [row["pair"] for row in base],
                        c["evaluation"]["bootstrap"],
                        floor,
                        absolute_gain,
                    )
                comparisons.append(
                    {
                        "seed": seed,
                        "task": "gate",
                        "split": "validation",
                        "horizon": horizon,
                        "alternative": alternative,
                        "reference": "recent_only",
                        "improvement": improvements,
                    }
                )
            privileged = [by[("privileged_small_budget", episode)] for episode in episodes]
            wrong = [by[("wrong_evidence", episode)] for episode in episodes]
            shift = [
                first["pair_margin"] - second["pair_margin"]
                for first, second in zip(privileged, wrong)
            ]
            comparisons.append(
                {
                    "seed": seed,
                    "task": "gate",
                    "split": "validation",
                    "horizon": horizon,
                    "alternative": "wrong_evidence",
                    "reference": "privileged_small_budget",
                    "toward_wrong_future": clustered_interval(
                        shift,
                        [row["pair"] for row in privileged],
                        c["evaluation"]["bootstrap"],
                    ),
                }
            )
    return {
        "per_seed": per_seed,
        "comparisons": comparisons,
        "statistics": (
            "Validation-only 95% paired bootstrap over base pairs; "
            "all_seeds_mean is conditional on the fixed seeds."
        ),
    }


def validation_task_checks(c, audit, probes, metrics):
    rule = c["decision"]
    failures, uncertain, checks = [], [], []
    if not audit.get("passed"):
        failures.append("Clean-future dataset audit did not pass")
    if audit.get("scope") != "TRAIN_VALIDATION_ONLY":
        failures.append("Unexpected clean-future audit scope")

    probe_keys = {
        (row["task"], row["split"], row["variant"], row.get("horizon"))
        for row in probes["results"]
    }
    metric_keys = {
        (
            row["seed"],
            row["task"],
            row["split"],
            row["condition"],
            row["horizon"],
        )
        for row in metrics["per_seed"]
    }
    comparison_keys = {
        (
            row["seed"],
            row["task"],
            row["split"],
            row["alternative"],
            row["horizon"],
        )
        for row in metrics["comparisons"]
    }
    missing = []
    for variant in ("gate_cue", "recent_only"):
        if ("gate", "validation", variant, None) not in probe_keys:
            missing.append(f"probe gate/validation/{variant}")
    for horizon in c["data"]["horizons"]:
        if (
            "gate",
            "validation",
            "state_decoder_on_true_target",
            horizon,
        ) not in probe_keys:
            missing.append(f"readout gate/validation/h{horizon}")
        for seed in c["training"]["seeds"]:
            for condition in CONDITIONS + ["persistence"]:
                if (
                    seed,
                    "gate",
                    "validation",
                    condition,
                    horizon,
                ) not in metric_keys:
                    missing.append(
                        f"metric {seed}/gate/validation/{condition}/h{horizon}"
                    )
        for seed in c["training"]["seeds"] + ["all_seeds_mean"]:
            for alternative in (
                "privileged_small_budget",
                "full_history",
                "random_small_budget",
                "wrong_evidence",
            ):
                if (
                    seed,
                    "gate",
                    "validation",
                    alternative,
                    horizon,
                ) not in comparison_keys:
                    missing.append(
                        f"comparison {seed}/gate/validation/{alternative}/h{horizon}"
                    )
    if missing:
        uncertain.append(f"Missing required validation slices: {missing[0]}")

    for row in probes["results"]:
        if row["task"] != "gate" or row["split"] != "validation":
            continue
        if row["variant"] == "gate_cue":
            passed = (
                row["accuracy"]["mean"] >= rule["probe_accuracy"]
                and row["wilson_effective_pairs"][0] > 0.5
            )
            checks.append(
                {
                    "kind": "cue_probe",
                    "slice": "gate/validation",
                    "passed": bool(passed),
                    "accuracy": row["accuracy"],
                }
            )
            if not passed:
                failures.append("Validation cue probe is unreliable")
        if row["variant"] == "recent_only" and abs(
            row["accuracy"]["mean"] - 0.5
        ) > 1e-7:
            failures.append("Validation recent probe is not chance")
        if row["variant"] == "state_decoder_on_true_target":
            # The original Gate task requires the state-class readout only.
            # Position RMSE belongs to the Motion checks and remains reported
            # below as diagnostic evidence, not as a Gate condition.
            passed = row["accuracy"]["mean"] >= rule["readout_accuracy"]
            checks.append(
                {
                    "kind": "state_readout",
                    "slice": f"gate/validation/h{row['horizon']}",
                    "passed": bool(passed),
                    "accuracy": row["accuracy"],
                    "position_mse": row["position_mse"],
                }
            )
            if not passed:
                failures.append(
                    f"Validation state readout failed at h{row['horizon']}"
                )

    for comparison in metrics["comparisons"]:
        if comparison["split"] != "validation":
            continue
        name = (
            f"gate/validation/h{comparison['horizon']}/"
            f"seed{comparison['seed']}"
        )
        if comparison["alternative"] == "privileged_small_budget":
            latent = comparison["improvement"]["normalized_mse"]
            state = comparison["improvement"]["state_accuracy"]
            latent_threshold = (
                rule["latent_relative_improvement"]
                if latent["scale"].startswith("relative")
                else rule["absolute_latent_gain"]
            )
            state_threshold = (
                rule["state_relative_improvement"]
                if state["scale"].startswith("relative")
                else rule["absolute_state_gain"]
            )
            passed = (
                latent["mean"] >= latent_threshold
                and latent["low"] > 0
                and state["mean"] >= state_threshold
                and state["low"] > 0
            )
            checks.append(
                {
                    "kind": "privileged_benefit",
                    "slice": name,
                    "passed": bool(passed),
                    "latent": latent,
                    "state": state,
                }
            )
            if not passed:
                failures.append(f"{name} privileged benefit did not pass")
        elif comparison["alternative"] == "wrong_evidence":
            shift = comparison["toward_wrong_future"]
            passed = (
                shift["mean"] >= rule["wrong_evidence_gain"]
                and shift["low"] > 0
            )
            checks.append(
                {
                    "kind": "wrong_evidence",
                    "slice": name,
                    "passed": bool(passed),
                    "shift": shift,
                }
            )
            if not passed:
                failures.append(f"{name} wrong-evidence shift did not pass")
        elif comparison["alternative"] == "full_history":
            value = comparison["improvement"]["normalized_mse"]
            passed = value["mean"] >= -rule["full_history_tolerance"]
            checks.append(
                {
                    "kind": "full_history_tolerance",
                    "slice": name,
                    "passed": bool(passed),
                    "latent": value,
                }
            )
            if not passed:
                failures.append(f"{name} full history is materially worse")

    for row in metrics["per_seed"]:
        if row["split"] != "validation":
            continue
        name = f"gate/validation/h{row['horizon']}/seed{row['seed']}"
        if (
            row["condition"] == "recent_only"
            and abs(row["metrics"]["pair_accuracy"]["mean"] - 0.5) > 1e-7
        ):
            failures.append(f"{name} recent-only pair accuracy is not chance")
        if row["almost_indistinguishable_fraction"] > 0.1:
            failures.append(f"{name} has too many indistinguishable targets")

    if len(c["training"]["seeds"]) < rule["min_seeds"]:
        uncertain.append("Insufficient fixed seeds")
    if len(
        {
            row["pair"]
            for row in metrics["per_seed"]
            if row["split"] == "validation"
        }
    ) < rule["min_pairs"]:
        uncertain.append("Validation pair count is below the unchanged floor")
    passed = not failures and not uncertain and bool(checks)
    return {
        "passed": passed,
        "failures": failures,
        "uncertain": uncertain,
        "checks": checks,
    }


def fixed_reference_validation_checks(c, reference, candidate):
    keys = (
        "seed",
        "task",
        "split",
        "horizon",
        "condition",
        "pair",
        "episode",
    )
    ref = {tuple(row[key] for key in keys): row for row in reference}
    cand = {tuple(row[key] for key in keys): row for row in candidate}
    if set(ref) != set(cand):
        raise ValueError("Reference/candidate validation identities differ")
    rule = c["decision"]
    checks = []
    for horizon in c["data"]["horizons"]:
        for seed in c["training"]["seeds"] + ["all_seeds_mean"]:
            selected = [
                row
                for row in candidate
                if row["split"] == "validation"
                and row["horizon"] == horizon
                and row["condition"] == "privileged_small_budget"
                and (
                    seed == "all_seeds_mean"
                    or row["seed"] == seed
                )
            ]
            grouped = defaultdict(list)
            for row in selected:
                key = tuple(row[field] for field in keys)
                recent_key = list(key)
                recent_key[4] = "recent_only"
                grouped[(row["pair"], row["episode"])].append(
                    [
                        ref[key]["normalized_mse"],
                        row["normalized_mse"],
                        ref[tuple(recent_key)]["normalized_mse"],
                    ]
                )
            ordered = sorted(grouped)
            values = np.asarray(
                [np.mean(grouped[key], axis=0) for key in ordered],
                dtype=np.float64,
            )
            groups = [key[0] for key in ordered]
            name = f"gate/validation/h{horizon}/seed{seed}"
            cross = paired_improvement(
                values[:, 2],
                values[:, 1],
                groups,
                c["evaluation"]["bootstrap"],
                rule["baseline_floor"],
                rule["absolute_latent_gain"],
            )
            threshold = (
                rule["latent_relative_improvement"]
                if cross["scale"].startswith("relative")
                else rule["absolute_latent_gain"]
            )
            checks.append(
                {
                    "kind": "candidate_vs_reference_recent",
                    "slice": name,
                    "passed": bool(
                        cross["mean"] >= threshold
                        and cross["low"] > 0
                    ),
                    "improvement": cross,
                    "reference_recent_mse": float(values[:, 2].mean()),
                    "candidate_privileged_mse": float(values[:, 1].mean()),
                }
            )
            absolute = clustered_interval(
                values[:, 0] - values[:, 1],
                groups,
                c["evaluation"]["bootstrap"],
            )
            checks.append(
                {
                    "kind": "candidate_absolute_vs_reference_privileged",
                    "slice": name,
                    "passed": bool(
                        absolute["mean"] > 0 and absolute["low"] > 0
                    ),
                    "absolute_mse_reduction": absolute,
                    "reference_privileged_mse": float(values[:, 0].mean()),
                    "candidate_privileged_mse": float(values[:, 1].mean()),
                }
            )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or "A100" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("Course A100 required")
    seed_all(17, 4)
    reference_config = json.loads(args.reference_config.read_text())
    candidate_config = json.loads(args.candidate_config.read_text())
    manifest = read_json(Path(reference_config["output"]) / "dataset/manifest.json")
    if {row["split"] for row in manifest["records"]} != {
        "train",
        "validation",
    }:
        raise ValueError("Evaluation source contains a non-train/validation split")
    reference_completion = read_json(
        Path(reference_config["output"]) / "reference_completion.json"
    )
    if not reference_completion.get("complete"):
        raise ValueError("Reference training is incomplete")
    completion = read_json(
        Path(args.candidate_root) / "all_models_completion.json"
    )
    if not completion.get("complete"):
        raise ValueError("Candidate training is incomplete")
    if set(completion["models"]) != {
        f"gate/{seed}" for seed in candidate_config["training"]["seeds"]
    }:
        raise ValueError("Candidate completion model set differs")
    for seed in candidate_config["training"]["seeds"]:
        path = (
            Path(args.candidate_root)
            / "predictor/gate"
            / f"seed_{seed}/best.pt"
        )
        proof = completion["models"][f"gate/{seed}"]
        if file_hash(path) != proof["best_sha256"]:
            raise ValueError(f"Candidate checkpoint changed after completion: {seed}")

    reference_raw, _, _ = evaluate_model(
        reference_config,
        model_loader=load_reference,
        device=device,
        output=args.output / "reference",
    )
    candidate_raw, _, _ = evaluate_model(
        candidate_config,
        model_loader=load_candidate,
        device=device,
        output=args.output / "candidate",
    )
    if {row["split"] for row in reference_raw + candidate_raw} != {
        "validation"
    }:
        raise ValueError("Evaluation accessed a non-validation split")

    metrics = summarize_validation(candidate_config, candidate_raw)
    metrics["training_completion"] = completion
    probes = read_json(Path(candidate_config["output"]) / "probes/metrics.json")
    audit = read_json(
        Path(candidate_config["output"]) / "dataset/split_audit.json"
    )
    task_checks = validation_task_checks(
        candidate_config,
        audit,
        probes,
        metrics,
    )
    fixed_checks = fixed_reference_validation_checks(
        candidate_config,
        reference_raw,
        candidate_raw,
    )
    fixed_failures = [
        f"{row['slice']} {row['kind']}"
        for row in fixed_checks
        if not row["passed"]
    ]
    passed = task_checks["passed"] and not fixed_failures
    result = {
        "status": (
            "CLEAN_FUTURE_VALIDATION_PASS"
            if passed
            else "CLEAN_FUTURE_VALIDATION_REJECTED"
        ),
        "scope": "TRAIN_FIT_VALIDATION_ONLY",
        "task_checks": task_checks,
        "fixed_reference_checks": fixed_checks,
        "fixed_reference_failures": fixed_failures,
        "checks_passed": sum(
            bool(row["passed"]) for row in task_checks["checks"]
        )
        + sum(bool(row["passed"]) for row in fixed_checks),
        "checks_total": len(task_checks["checks"]) + len(fixed_checks),
        "reference_raw_sha256": file_hash(
            args.output / "reference/raw_records.csv"
        ),
        "candidate_raw_sha256": file_hash(
            args.output / "candidate/raw_records.csv"
        ),
        "reference_completion": reference_completion,
        "candidate_completion": completion,
        "dataset_signature": manifest["signature"],
        "thresholds_changed": False,
        "test_access": False,
        "fresh_9137_opened": False,
        "phase1_go": False,
        "full_phase1_pass": False,
        "note": (
            "This is the preregistered train/validation clean-future check. "
            "It cannot open or authorize the locked test split automatically."
        ),
    }
    write_json(args.output / "clean_future_validation_outcome.json", result)
    write_json(args.output / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks_passed": result["checks_passed"],
                "checks_total": result["checks_total"],
                "failures": task_checks["failures"],
                "fixed_reference_failures": fixed_failures,
                "test_access": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
