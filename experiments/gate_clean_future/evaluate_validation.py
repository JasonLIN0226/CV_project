"""Evaluate a clean-future Gate model using validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTROL = REPO / "experiments/predictor/control/src"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from phase1.config import CONDITIONS  # noqa: E402
from phase1.data import load_episode  # noqa: E402
from phase1.evaluation import (  # noqa: E402
    pair_calibration,
    pair_diagnostic,
    summarize_calibration,
    training_completion,
)
from phase1.inputs import FeatureStore, collate  # noqa: E402
from phase1.model import fit_basis  # noqa: E402
from phase1.probes import load_decoder  # noqa: E402
from phase1.stats import (  # noqa: E402
    clustered_interval,
    paired_improvement,
)
from phase1.training import load_predictor  # noqa: E402
from phase1.utils import file_hash, seed_all, write_csv, write_json  # noqa: E402


METRICS = (
    "latent_mse",
    "normalized_mse",
    "pair_accuracy",
    "pair_margin",
    "target_pair_distance",
    "state_accuracy",
    "state_position_mse",
    "history_count",
    "history_feature_bytes",
    "recent_feature_bytes",
    "projected_history_bytes",
    "timestamp_type_mask_bytes",
    "past_action_bytes",
    "evidence_hit",
)


def _summary(c, raw):
    groups = {}
    for row in raw:
        key = (
            row["seed"],
            row["task"],
            row["split"],
            row["condition"],
            row["horizon"],
        )
        groups.setdefault(key, []).append(row)
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

    pooled = {}
    for row in raw:
        key = (
            row["task"],
            row["split"],
            row["condition"],
            row["horizon"],
            row["episode"],
        )
        pooled.setdefault(key, []).append(row)
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
        for seed in [*c["training"]["seeds"], "all_seeds_mean"]:
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
            by_condition = {
                (row["condition"], row["episode"]): row for row in subset
            }
            episodes = sorted({row["episode"] for row in subset})
            for alternative in (
                "privileged_small_budget",
                "full_history",
                "random_small_budget",
            ):
                base = [by_condition[("recent_only", episode)] for episode in episodes]
                better = [by_condition[(alternative, episode)] for episode in episodes]
                improvements = {}
                for metric in (
                    "normalized_mse",
                    "latent_mse",
                    "state_position_mse",
                    "state_accuracy",
                    "pair_accuracy",
                ):
                    lower_is_better = metric.endswith("mse")
                    reference_values = [
                        row[metric] if lower_is_better else 1 - row[metric]
                        for row in base
                    ]
                    alternative_values = [
                        row[metric] if lower_is_better else 1 - row[metric]
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
                        reference_values,
                        alternative_values,
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
            privileged = [
                by_condition[("privileged_small_budget", episode)]
                for episode in episodes
            ]
            wrong = [
                by_condition[("wrong_evidence", episode)] for episode in episodes
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
                        [
                            privileged_row["pair_margin"]
                            - wrong_row["pair_margin"]
                            for privileged_row, wrong_row in zip(privileged, wrong)
                        ],
                        [row["pair"] for row in privileged],
                        c["evaluation"]["bootstrap"],
                    ),
                }
            )
    return {
        "per_seed": per_seed,
        "comparisons": comparisons,
        "statistics": (
            "95% percentile paired bootstrap over validation base pairs; "
            "all_seeds_mean is conditional on the three fixed seeds"
        ),
    }


def _qualification(c, completion, probe_metrics, summary, raw):
    rule = c["decision"]
    checks = []
    failures = []

    def record(kind, scope, passed, detail):
        checks.append(
            {"kind": kind, "scope": scope, "passed": bool(passed), "detail": detail}
        )
        if not passed:
            failures.append(f"{kind}:{scope}")

    record(
        "training_completion",
        "all_models",
        completion.get("complete") is True,
        completion.get("issues", []),
    )
    cue_probe = probe_metrics["results"]["gate_cue"]["validation_accuracy"]
    recent_probe = probe_metrics["results"]["recent_only"]["validation_accuracy"]
    record(
        "cue_probe",
        "validation",
        cue_probe >= rule["probe_accuracy"],
        cue_probe,
    )
    record(
        "recent_pair_chance",
        "probe",
        abs(recent_probe - 0.5) <= 1e-12,
        recent_probe,
    )
    for horizon, values in probe_metrics["readout"].items():
        record(
            "state_readout_accuracy",
            f"validation/h{horizon}",
            values["validation_accuracy"] >= rule["readout_accuracy"],
            values["validation_accuracy"],
        )
        record(
            "state_readout_position",
            f"validation/h{horizon}",
            values["validation_position_rmse"] <= rule["readout_position_rmse"],
            values["validation_position_rmse"],
        )

    per_seed = {
        (row["seed"], row["condition"], row["horizon"]): row
        for row in summary["per_seed"]
    }
    for seed in c["training"]["seeds"]:
        for horizon in c["data"]["horizons"]:
            recent = per_seed[(seed, "recent_only", horizon)]
            record(
                "recent_pair_chance",
                f"validation/h{horizon}/seed{seed}",
                abs(
                    recent["metrics"]["pair_accuracy"]["mean"] - 0.5
                )
                <= 1e-12,
                recent["metrics"]["pair_accuracy"]["mean"],
            )
            record(
                "targets_distinguishable",
                f"validation/h{horizon}/seed{seed}",
                recent["almost_indistinguishable_fraction"] <= 0.1,
                recent["almost_indistinguishable_fraction"],
            )

    comparisons = {
        (
            row["seed"],
            row["horizon"],
            row["alternative"],
        ): row
        for row in summary["comparisons"]
    }
    for seed in c["training"]["seeds"]:
        for horizon in c["data"]["horizons"]:
            benefit = comparisons[(seed, horizon, "privileged_small_budget")][
                "improvement"
            ]
            for metric, threshold in (
                ("normalized_mse", rule["latent_relative_improvement"]),
                ("state_accuracy", rule["state_relative_improvement"]),
            ):
                value = benefit[metric]
                threshold_value = (
                    rule["absolute_latent_gain"]
                    if metric == "normalized_mse"
                    and value["scale"].startswith("absolute")
                    else rule["absolute_state_gain"]
                    if metric == "state_accuracy"
                    and value["scale"].startswith("absolute")
                    else threshold
                )
                record(
                    f"privileged_{metric}",
                    f"validation/h{horizon}/seed{seed}",
                    value["mean"] >= threshold_value and value["low"] > 0,
                    value,
                )
            wrong = comparisons[
                (seed, horizon, "wrong_evidence")
            ]["toward_wrong_future"]
            record(
                "wrong_evidence",
                f"validation/h{horizon}/seed{seed}",
                wrong["mean"] >= rule["wrong_evidence_gain"]
                and wrong["low"] > 0,
                wrong,
            )
            full = comparisons[(seed, horizon, "full_history")]["improvement"][
                "normalized_mse"
            ]
            record(
                "full_history_not_worse",
                f"validation/h{horizon}/seed{seed}",
                full["mean"] >= -rule["full_history_tolerance"],
                full,
            )

    return {
        "passed": not failures,
        "status": (
            "VALIDATION_PASS_FOR_LOCKED_TEST"
            if not failures
            else "VALIDATION_REJECTED"
        ),
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "failures": failures,
        "checks": checks,
        "scope": "train_and_validation_only",
        "thresholds_unchanged": True,
        "test_accessed": False,
        "final_phase1_gate": "NOT_EVALUATED",
    }


def evaluate(c, output, device):
    store = FeatureStore(c)
    manifest_splits = {row["split"] for row in store.manifest["records"]}
    if manifest_splits != {"train", "validation"}:
        raise ValueError("Validation evaluator refuses missing or unexpected splits")
    completion = training_completion(c, store)
    if not completion["complete"]:
        raise ValueError("Training is incomplete: " + "; ".join(completion["issues"]))
    decoder = load_decoder(c, "gate")
    raw = []
    calibration = []
    checkpoints = {}
    validation_rows = [
        row
        for row in store.manifest["records"]
        if row["task"] == "gate" and row["split"] == "validation"
    ]
    for seed in c["training"]["seeds"]:
        model, checkpoint = load_predictor(c, seed, store, device, "gate")
        checkpoints[f"gate/{seed}"] = {
            "seed": seed,
            "best_epoch": int(checkpoint["epoch"]),
            "best_sha256": file_hash(
                Path(c["output"]) / f"predictor/gate/seed_{seed}/best.pt"
            ),
        }
        for condition in [*CONDITIONS, "persistence"]:
            prediction_rows = []
            for row in validation_rows:
                sample_condition = (
                    "recent_only" if condition == "persistence" else condition
                )
                sample, target, info = store.sample(row, sample_condition)
                if condition == "persistence":
                    prediction = np.tile(
                        sample["current"],
                        (len(c["data"]["horizons"]), 1),
                    )
                else:
                    with torch.inference_mode():
                        prediction = model(collate([sample], device))[0].cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise RuntimeError("Non-finite validation prediction")
                prediction_rows.append(
                    {
                        "row": row,
                        "prediction": prediction,
                        "target": target,
                        "info": info,
                    }
                )
            by_pair = {}
            for item in prediction_rows:
                by_pair.setdefault(item["row"]["pair"], []).append(item)
            for pair_id, pair_items in by_pair.items():
                if len(pair_items) != 2:
                    raise ValueError("Incomplete validation sibling pair")
                for horizon_index, horizon in enumerate(c["data"]["horizons"]):
                    vectors = [
                        (
                            item["prediction"][horizon_index],
                            item["target"][horizon_index],
                        )
                        for item in pair_items
                    ]
                    calibration.append(
                        {
                            "seed": seed,
                            "task": "gate",
                            "split": "validation",
                            "condition": condition,
                            "horizon": horizon,
                            "pair": pair_id,
                            **pair_calibration(
                                [vector[0] for vector in vectors],
                                [vector[1] for vector in vectors],
                            ),
                        }
                    )
                    pair_accuracy, pair_margin, separation = pair_diagnostic(
                        pair_items[0]["prediction"][horizon_index],
                        pair_items[0]["target"][horizon_index],
                        pair_items[1]["target"][horizon_index],
                    )
                    for item in pair_items:
                        row = item["row"]
                        target_index = int(row["query"]) + int(horizon)
                        meta = load_episode(c, row, True)
                        state = decoder.predict(
                            item["prediction"][horizon_index][None, :]
                        )[0]
                        predicted_raw = (
                            item["prediction"][horizon_index] * store.norm["std"]
                            + store.norm["mean"]
                        )
                        target_raw = (
                            item["target"][horizon_index] * store.norm["std"]
                            + store.norm["mean"]
                        )
                        retained = int(item["info"]["history_count"])
                        dimension = item["prediction"].shape[-1]
                        raw.append(
                            {
                                "seed": seed,
                                "task": "gate",
                                "split": "validation",
                                "condition": condition,
                                "horizon": horizon,
                                "episode": row["id"],
                                "pair": pair_id,
                                "gap": row["gap"],
                                "query": row["query"],
                                "latent_mse": float(
                                    np.mean((predicted_raw - target_raw) ** 2)
                                ),
                                "normalized_mse": float(
                                    np.mean(
                                        (
                                            item["prediction"][horizon_index]
                                            - item["target"][horizon_index]
                                        )
                                        ** 2
                                    )
                                ),
                                "pair_accuracy": pair_accuracy,
                                "pair_margin": pair_margin,
                                "target_pair_distance": separation,
                                "state_accuracy": float(
                                    (state[0] > 0)
                                    == (meta["classes"][target_index] > 0)
                                ),
                                "state_position_mse": float(
                                    np.mean(
                                        (
                                            state[1:]
                                            - meta["states"][target_index]
                                        )
                                        ** 2
                                    )
                                ),
                                "history_count": retained,
                                "history_feature_bytes": retained
                                * dimension
                                * 4,
                                "recent_feature_bytes": int(c["data"]["recent"])
                                * dimension
                                * 4,
                                "projected_history_bytes": retained
                                * model.projection.shape[1]
                                * 4,
                                "timestamp_type_mask_bytes": (
                                    retained + int(c["data"]["recent"])
                                )
                                * 13,
                                "past_action_bytes": (
                                    retained + int(c["data"]["recent"])
                                )
                                * 8,
                                "evidence_hit": float(item["info"]["evidence_hit"]),
                                "state_pred_x": float(state[1]),
                                "state_pred_y": float(state[2]),
                                "state_true_x": float(
                                    meta["states"][target_index, 0]
                                ),
                                "state_true_y": float(
                                    meta["states"][target_index, 1]
                                ),
                            }
                        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summary(c, raw)
    probe_metrics = json.loads(
        (Path(c["output"]) / "probes/metrics.json").read_text()
    )
    qualification = _qualification(
        c,
        completion,
        probe_metrics,
        summary,
        raw,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "raw_records.csv", raw)
    write_json(
        output / "validation_summary.json",
        {
            **summary,
            "training_completion": completion,
            "checkpoints": checkpoints,
            "probe_metrics": probe_metrics,
            "scope": "validation_only",
            "test_accessed": False,
        },
    )
    write_json(
        output / "calibration.json",
        {"rows": calibration, **summarize_calibration(c, calibration)},
    )
    write_json(output / "validation_qualification.json", qualification)
    return qualification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if c["data"]["clean_future_protocol"]["test_accessed"]:
        raise ValueError("Test access must remain false")
    device = torch.device(args.device)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or "A100" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("Course A100 required")
    seed_all(17, 4)
    result = evaluate(c, args.output.expanduser().resolve(), device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
