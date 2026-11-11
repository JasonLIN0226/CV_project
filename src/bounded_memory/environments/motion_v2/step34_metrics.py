"""Pair-clustered summaries for the registered Motion step 3/4 experiment.

Importing this module performs no I/O. Callers supply already-scored arrays;
this module never loads datasets, predictors, students, or hidden-state files.
All uncertainty is over scene pairs conditional on the fitted seeds.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np


SEEDS = (17, 29, 43)
HORIZONS = (3, 6)
PAIRS = 96
BOOTSTRAP_DRAWS = 2000
COST_LABELS = ("local_feature", "local_position", "use_time_feature", "use_time_position")
LABELS = COST_LABELS + ("future_attention",) + tuple("shuffle_" + name for name in COST_LABELS)
BASELINES = ("fifo", "reservoir", "temporal_buckets", "diversity", "surprise", "attention", "keep_first")
REFERENCES = ("posthoc_best", "last_visible", "b")


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _array(value, shape, name, *, nonnegative=True):
    array = np.asarray(value)
    _require(array.dtype.kind in "fiu" and array.dtype.kind != "b", name + " must be numeric, not boolean")
    array = np.asarray(array, dtype=np.float64)
    _require(array.shape == shape, name + " has incorrect axes/shape")
    _require(np.isfinite(array).all(), name + " contains nonfinite values")
    if nonnegative:
        _require((array >= 0).all(), name + " contains negative loss")
    return array


def _sha_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _indices(seed):
    _require(type(seed) is int and seed == 901, "The registered bootstrap seed is 901")
    return np.random.default_rng(seed).integers(0, PAIRS, size=(BOOTSTRAP_DRAWS, PAIRS))


def _interval(pair_values, indices):
    _require(pair_values.shape == (PAIRS,) and np.isfinite(pair_values).all(), "Expected 96 finite pair values")
    draws = pair_values[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975), method="linear")
    return {"mean": float(pair_values.mean()), "ci95": [float(low), float(high)], "pairs": PAIRS}


def _positive(interval):
    return interval["ci95"][0] > 0.0


def _includes_zero(interval):
    return interval["ci95"][0] <= 0.0 <= interval["ci95"][1]


def _pair_ids(pair_ids):
    _require(pair_ids is not None, "Pair identities in array order are required")
    ids = list(pair_ids)
    _require(len(ids) == PAIRS and all(isinstance(item, str) and item for item in ids)
             and len(set(ids)) == PAIRS, "Require 96 unique nonempty pair ids in array order")
    return ids


def _bootstrap_metadata(indices, aggregation):
    return {"repeats": BOOTSTRAP_DRAWS, "seed": 901, "unit": "scene pair", "pairs": PAIRS,
            "percentile_ci": [0.025, 0.975], "quantile_method": "linear", "aggregation": aggregation,
            "shared_indices_across_methods_metrics_horizons_and_seed_tables": True,
            "indices_sha256": _sha_array(indices),
            "interpretation": "Scene-pair uncertainty conditional on the fitted seeds; not a population interval over training seeds",
            "multiplicity": "No multiplicity correction; all prespecified comparisons are reported"}


def _save(result, output, indices):
    # Other stage evidence may already exist here; own receipts are exclusive.
    result = dict(result)
    index_bytes = io.BytesIO()
    np.savez_compressed(index_bytes, indices=indices)
    index_blob = index_bytes.getvalue()
    index_sha = hashlib.sha256(index_blob).hexdigest()
    result["bootstrap_indices_artifact"] = {"path": "bootstrap_indices.npz", "sha256": index_sha,
                                             "array_sha256": _sha_array(indices),
                                             "shape": list(indices.shape), "dtype": indices.dtype.str,
                                             "written": output is not None}
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        destination = Path(output)
        _require(not destination.is_symlink() and (not destination.exists() or destination.is_dir()),
                 "Output must be a directory, not a symlink")
        for name in ("summary.json", "status.json", "SHA256SUMS", "bootstrap_indices.npz"):
            owned = destination / name
            _require(not owned.exists() and not owned.is_symlink(), "Summary receipt already exists: " + name)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "bootstrap_indices.npz").open("xb") as handle:
            handle.write(index_blob)
        receipt = destination / "summary.json"
        with receipt.open("x", encoding="utf-8") as handle:
            handle.write(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        status = {"status": "completed", "summary_sha256": digest, "bootstrap_indices_sha256": index_sha,
                  "scientific_status": result["status"],
                  "test_or_confirmation_used": False, "new_model_training_performed_by_summary": False}
        status_text = json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with (destination / "status.json").open("x", encoding="utf-8") as handle:
            handle.write(status_text)
        with (destination / "SHA256SUMS").open("x", encoding="utf-8") as handle:
            handle.write(digest + "  summary.json\n")
            handle.write(hashlib.sha256(status_text.encode()).hexdigest() + "  status.json\n")
            handle.write(index_sha + "  bootstrap_indices.npz\n")
    return result


def _diagnostic_tables(values, bound, decoder, indices):
    table, contrasts = [], []
    for method, metrics in values.items():
        for hi, horizon in enumerate(HORIZONS):
            table.append({"method": method, "horizon": horizon,
                          "feature_mse": _interval(metrics["feature"][:, hi], indices),
                          "position_mse": _interval(metrics["position"][:, hi], indices)})
    for hi, horizon in enumerate(HORIZONS):
        for name, difference, key in (
            ("B_minus_b", bound[:, hi] - values["b"]["position"][:, hi], "position_mse_improvement"),
            ("d_minus_B", values["d"]["position"][:, hi] - bound[:, hi], "position_mse_excess"),
        ):
            contrasts.append({"comparison": name, "horizon": horizon, key: _interval(difference, indices)})
        contrasts.append({"comparison": "d_minus_b", "horizon": horizon,
                          "position_mse_improvement": _interval(values["d"]["position"][:, hi] - values["b"]["position"][:, hi], indices),
                          "feature_mse_improvement": _interval(values["d"]["feature"][:, hi] - values["b"]["feature"][:, hi], indices)})
        if "b_random" in values:
            contrasts.append({"comparison": "B_minus_b_random", "horizon": horizon,
                              "positive_favors": "b_random", "report_only": True, "required_gate": False,
                              "position_mse_improvement": _interval(bound[:, hi] - values["b_random"]["position"][:, hi], indices)})
            for name, higher, lower in (("d_minus_b_random", "d", "b_random"),
                                        ("b_random_minus_b", "b_random", "b")):
                contrasts.append({"comparison": name, "horizon": horizon,
                                  "positive_favors": lower, "report_only": True, "required_gate": False,
                                  "position_mse_improvement": _interval(values[higher]["position"][:, hi] - values[lower]["position"][:, hi], indices),
                                  "feature_mse_improvement": _interval(values[higher]["feature"][:, hi] - values[lower]["feature"][:, hi], indices)})
    references = [{"horizon": horizon, "B": _interval(bound[:, hi], indices),
                   "decoder_true_target_position_mse": _interval(decoder[:, hi], indices)}
                  for hi, horizon in enumerate(HORIZONS)]
    return table, contrasts, references


def summarize_diagnostics(losses, positions, decoder_true_losses, output, bootstrap_seed=901,
                          *, split="ordinary", pair_ids=None):
    """Summarize method losses [reader=3,pair=96,sibling=2,horizon=2].

    Methods must include b and d; b_random and other named diagnostics are
    reported as supplied. Only long-gap B-b at both horizons limits claim_scope;
    d-b, d-B, features and decoder errors are diagnostic, not extra gates.
    """
    _require(split in ("ordinary", "long_gap"), "Only ordinary/long_gap validation diagnostics are allowed")
    ids = _pair_ids(pair_ids)
    _require(isinstance(losses, dict) and {"b", "d"} <= set(losses), "Diagnostics require b and d")
    arrays = {}
    for method, metrics in losses.items():
        _require(isinstance(method, str) and method and isinstance(metrics, dict)
                 and set(metrics) == {"feature", "position"}, "Invalid diagnostic method or metric fields")
        arrays[method] = {name: _array(value, (3, PAIRS, 2, 2), method + "/" + name)
                          for name, value in metrics.items()}
    true_positions = _array(positions, (PAIRS, 2, 2, 2), "positions", nonnegative=False)
    decoder = _array(decoder_true_losses, (PAIRS, 2, 2), "decoder_true_losses")
    midpoint = true_positions.mean(axis=1, keepdims=True)
    bound = np.square(true_positions - midpoint).mean(axis=(1, 3))
    _require(np.isfinite(bound).all(), "Nonfinite direction bound")
    indices = _indices(bootstrap_seed)
    averaged = {method: {name: value.mean(axis=2).mean(axis=0) for name, value in metrics.items()}
                for method, metrics in arrays.items()}
    table, contrasts, references = _diagnostic_tables(averaged, bound, decoder.mean(axis=1), indices)
    direction = {str(row["horizon"]): {"lower95": row["position_mse_improvement"]["ci95"][0],
                                        "passed": _positive(row["position_mse_improvement"])}
                 for row in contrasts if row["comparison"] == "B_minus_b"}
    is_long_gap = split == "long_gap"
    passes = all(item["passed"] for item in direction.values())
    per_reader = []
    for si, seed in enumerate(SEEDS):
        values = {method: {name: value[si].mean(axis=1) for name, value in metrics.items()}
                  for method, metrics in arrays.items()}
        rows, comparisons, reference = _diagnostic_tables(values, bound, decoder.mean(axis=1), indices)
        per_reader.append({"reader_seed": seed, "table": rows, "contrasts": comparisons,
                           "position_reference_table": reference, "additional_gates": False})
    return _save({"schema_version": 1, "kind": "reader_diagnostics", "status": "completed_diagnostics",
                  "split": split, "pair_ids": ids, "reader_seeds": list(SEEDS),
                  "pair_identity_validation": "96_unique_in_input_order",
                  "loss_shape": [3, PAIRS, 2, 2], "table": table, "contrasts": contrasts,
                  "position_reference_table": references, "per_reader": per_reader,
                  "direction_bound": {"formula": "mean_sibling(mean_xy((true_xy-mean_sibling(true_xy))^2))",
                                      "sha256": _sha_array(bound), "no_seed_axis": True},
                  "long_gap_direction_check": {"applies": is_long_gap, "horizons": direction,
                                               "passed": passes if is_long_gap else None,
                                               "strict_lower95_required": 0.0},
                  "claim_scope": ("long_gap_reader_qualified" if passes else "long_gap_reader_not_qualified")
                                  if is_long_gap else "validation_only_no_long_gap_qualification",
                  "other_diagnostics_are_not_gates": True,
                  "bootstrap": _bootstrap_metadata(indices, "average siblings, then three fixed reader seeds before pair resampling"),
                  "input_sha256": {method: {name: _sha_array(value) for name, value in metrics.items()}
                                   for method, metrics in arrays.items()},
                  "true_positions_sha256": _sha_array(true_positions),
                  "decoder_true_losses_sha256": _sha_array(decoder),
                  "test_or_confirmation_used": False}, output, indices)


def _training_metadata(methods, trained_steps):
    _require(isinstance(trained_steps, dict), "trained_steps must map every student method to three selected steps")
    supplied = trained_steps
    _require(set(supplied) == {m for m in methods if m.startswith("student/")}, "Missing or unexpected trained_steps method")
    result = {}
    for method in methods:
        if not method.startswith("student/"):
            continue
        values = supplied.get(method)
        _require(isinstance(values, (list, tuple)) and len(values) == 3
                 and all(type(value) is int and value >= 0 for value in values), "Selected steps must be three nonnegative integers")
        result[method] = {"selected_steps": list(values), "includes_initialization": any(value == 0 for value in values),
                          "fully_trained_claim_allowed": all(value > 0 for value in values),
                          "status": "includes_initialization" if any(value == 0 for value in values) else "positive_step_checkpoints"}
    return result


def _online_table(values, indices):
    table = []
    for method, metrics in values.items():
        success = _interval(metrics["success"], indices)
        for hi, horizon in enumerate(HORIZONS):
            table.append({"method": method, "horizon": horizon, "success": success,
                          "feature_mse": _interval(metrics["feature"][:, hi], indices),
                          "position_mse": _interval(metrics["position"][:, hi], indices)})
    return table


def _comparison(values, candidate, comparator, indices):
    success = _interval(values[candidate]["success"] - values[comparator]["success"], indices)
    horizons = []
    for hi, horizon in enumerate(HORIZONS):
        position = _interval(values[comparator]["position"][:, hi] - values[candidate]["position"][:, hi], indices)
        feature = _interval(values[comparator]["feature"][:, hi] - values[candidate]["feature"][:, hi], indices)
        horizons.append({"horizon": horizon, "position_mse_improvement": position,
                         "feature_mse_improvement": feature,
                         "success_and_position_lower95_positive": _positive(success) and _positive(position)})
    return {"candidate": candidate, "comparator": comparator, "positive_favors": candidate,
            "success_difference": success, "horizons": horizons, "feature_is_report_only": True}


def _online_comparisons(values, indices):
    comparisons, conditioning = [], []
    for label in COST_LABELS:
        candidate = "student/set/" + label
        for comparator in ["baseline/" + name for name in BASELINES] + ["student/set/future_attention"]:
            comparisons.append({"label": label, **_comparison(values, candidate, comparator, indices)})
        row = {"label": label, **_comparison(values, candidate, "student/independent/" + label, indices)}
        row["success_cannot_distinguish_set_conditioning"] = _includes_zero(row["success_difference"])
        for horizon in row["horizons"]:
            horizon["cannot_distinguish_set_conditioning"] = _includes_zero(horizon["position_mse_improvement"])
        row["cannot_distinguish_set_conditioning"] = row["success_cannot_distinguish_set_conditioning"] or any(
            horizon["cannot_distinguish_set_conditioning"] for horizon in row["horizons"])
        conditioning.append(row)
    return comparisons, conditioning


def summarize_online(results, output, split="ordinary", bootstrap_seed=901, *, trained_steps=None,
                     replicated_methods=(), pair_ids=None):
    """Summarize losses [run=3,reader=3,pair=96,sibling=2,horizon=2].

    Success must be bool [run=3,pair=96,sibling=2]: both final memory frames
    visible in the new segment. Fixed baselines may repeat one trajectory over
    three runs, but callers must name those methods in replicated_methods.
    Missing training-step metadata is an error, never an implicit trained arm.
    """
    _require(split in ("ordinary", "long_gap"), "Only ordinary/long_gap validation summaries are permitted")
    ids = _pair_ids(pair_ids)
    required = {"student/" + architecture + "/" + label for architecture in ("set", "independent") for label in LABELS}
    required |= {"baseline/" + name for name in BASELINES}
    allowed = required | {"reference/" + name for name in REFERENCES}
    _require(isinstance(results, dict) and required <= set(results) <= allowed,
             "Online results require all 18 student arms and seven baselines; only named references are optional")
    replicated = list(replicated_methods)
    _require(len(set(replicated)) == len(replicated) and set(replicated) <= set(results)
             and all(not method.startswith("student/") for method in replicated), "Only listed baseline/reference methods may have copied runs")
    arrays = {}
    for method, metrics in results.items():
        _require(isinstance(metrics, dict) and set(metrics) == {"feature", "position", "success"}, "Unexpected online metric fields")
        values = {name: _array(metrics[name], (3, 3, PAIRS, 2, 2), method + "/" + name)
                  for name in ("feature", "position")}
        success = np.asarray(metrics["success"])
        _require(success.dtype == np.bool_ and success.shape == (3, PAIRS, 2), "Success must be boolean [3 runs,96 pairs,2 siblings]")
        values["success"] = success
        if method in replicated:
            _require(all(np.array_equal(value[0], value[index]) for value in values.values() for index in (1, 2)),
                     "Declared fixed baseline/reference has unequal copied runs")
        arrays[method] = values
    training = _training_metadata(results, trained_steps)
    indices = _indices(bootstrap_seed)
    averaged = {method: {"feature": value["feature"].mean(axis=3).mean(axis=(0, 1)),
                         "position": value["position"].mean(axis=3).mean(axis=(0, 1)),
                         "success": value["success"].mean(axis=2).mean(axis=0)}
                for method, value in arrays.items()}
    comparisons, conditioning = _online_comparisons(averaged, indices)
    claims = {}
    for label in COST_LABELS:
        rows = [row for row in comparisons if row["label"] == label]
        per_horizon = {str(horizon): all(row["horizons"][hi]["success_and_position_lower95_positive"] for row in rows)
                       for hi, horizon in enumerate(HORIZONS)}
        candidate = "student/set/" + label
        fully_trained = training[candidate]["fully_trained_claim_allowed"] and training["student/set/future_attention"]["fully_trained_claim_allowed"]
        claims[label] = {"role": "prespecified_primary" if label == "use_time_position" else "prespecified_secondary",
                         "comparators": 8, "per_horizon_all_comparisons_passed": per_horizon,
                         "all_numeric_comparisons_passed": all(per_horizon.values()),
                         "fully_trained_comparison_allowed": fully_trained,
                         "validation_criterion_passed": all(per_horizon.values()) and fully_trained,
                         "confirmation_claim_established": False,
                         "selection_after_results_allowed": False}
    per_seed = []
    for si, seed in enumerate(SEEDS):
        values = {method: {"feature": value["feature"][si].mean(axis=2).mean(axis=0),
                           "position": value["position"][si].mean(axis=2).mean(axis=0),
                           "success": value["success"][si].mean(axis=1)}
                  for method, value in arrays.items()}
        rows, architecture = _online_comparisons(values, indices)
        per_seed.append({"student_or_rng_seed": seed, "table": _online_table(values, indices),
                         "comparisons": rows, "set_conditioning": architecture, "additional_gates": False})
    return _save({"schema_version": 1, "kind": "online_comparison", "status": "completed_validation_comparison",
                  "split": split, "pair_ids": ids,
                  "pair_identity_validation": "96_unique_in_input_order",
                  "student_or_rng_seeds": list(SEEDS), "reader_seeds": list(SEEDS), "horizons": list(HORIZONS),
                  "raw_loss_shape": [3, 3, PAIRS, 2, 2], "raw_success_shape": [3, PAIRS, 2],
                  "success_definition": "Both retained memory frames are visible in the new segment",
                  "table": _online_table(averaged, indices), "comparisons": comparisons,
                  "set_conditioning": conditioning, "label_claims": claims, "per_student_seed": per_seed,
                  "training_checkpoint_status": training,
                  "replicated_fixed_methods": replicated,
                  "copied_runs_are_not_independent_samples": True,
                  "reference_methods_are_not_opponents": True,
                  "prespecified_primary": "use_time_position", "secondary_labels": list(COST_LABELS[:-1]),
                  "claim_scope": "validation_only_no_confirmation_claim",
                  "bootstrap": _bootstrap_metadata(indices, "average siblings, then three fixed student/RNG runs and three reader seeds before resampling scene pairs"),
                  "input_sha256": {method: {name: _sha_array(value) for name, value in values.items()}
                                   for method, values in arrays.items()},
                  "test_or_confirmation_used": False}, output, indices)
