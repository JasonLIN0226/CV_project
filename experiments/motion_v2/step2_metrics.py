"""Train-only position readout and preregistered paired validation statistics.

This module never generates data or opens a test split. Decoder fitting imports
the frozen Phase 1 LinearReadout only inside fit_train_decoder. Metrics operate on
the runner's already computed losses; they never refit or select a predictor.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


SEEDS = (17, 29, 43)
HORIZONS = (3, 6)
CONDITIONS = ("a", "b", "c", "d")
HISTORIES = {"a": "recent-only", "b": "two post-change visible frames",
             "c": "two stale visible frames", "d": "end frame plus one stale visible frame"}
RIDGES = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 901


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(_json_bytes({"shape": list(value.shape), "dtype": value.dtype.str}))
    h.update(value.tobytes())
    return h.hexdigest()


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def _new_output_dir(path):
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(), "Refusing to overwrite a decoder/metrics output directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _load_readout_class():
    # Explicit neighboring-file import also works under python -I.
    source = Path(__file__).with_name("reader_v2.py")
    spec = importlib.util.spec_from_file_location("motion_v2_decoder_reader", source)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    return reader.load_reference("probes").LinearReadout


def fit_train_decoder(store, config, output_dir, check_budget):
    """Fit a frozen [normalized 5120-feature] -> [x,y] readout using train only.

Four ridge values are compared by overall out-of-fold coordinate MSE across
five GroupKFold folds. One pair's two siblings and both horizons always share a
fold. LinearReadout.fit computes its feature mean/std on that fold's training
rows. Exact score ties select the smaller ridge. Validation is never loaded.

check_budget(stage: str) must raise on an exhausted allocation. Calls bracket
every fit and other expensive phases; errors propagate without an implicit retry.
"""
    from sklearn.model_selection import GroupKFold

    check_budget("decoder:before_inputs")
    _require(tuple(config["data"]["horizons"]) == HORIZONS, "Decoder horizons must be 3 and 6")
    _require(tuple(float(v) for v in config["probe"]["ridge"]) == RIDGES, "Decoder ridge grid differs from preregistration")
    all_records = store.manifest["records"]
    _require(all(r.get("split") in ("train", "validation") for r in all_records), "Unexpected/test split in feature store")
    records = [r for r in all_records if r["split"] == "train" and r.get("task") == "motion"]
    _require(len(records) == 1024 and len({r["id"] for r in records}) == 1024, "Decoder requires exactly 1024 distinct train episodes")
    counts = Counter(r["pair"] for r in records)
    _require(len(counts) == 512 and set(counts.values()) == {2}, "Decoder requires 512 complete train sibling pairs")
    records.sort(key=lambda r: (r["scene_index"], r["id"]))

    feature_rows, position_rows, row_ids = [], [], []
    for record in records:
        check_budget("decoder:collect_train_targets")
        features = np.asarray(store.features(record))
        _require(features.ndim == 2 and features.shape[1] == 5120, "Decoder input must be normalized 5120-dimensional features")
        meta = store.read_evaluation(record, keys=("states", "visible"))
        states, visible = np.asarray(meta["states"]), np.asarray(meta["visible"])
        for horizon in HORIZONS:
            target = int(record["query"]) + horizon
            _require(0 <= target < len(features) and target < len(states) and target < len(visible), "Decoder target out of bounds")
            _require(bool(visible[target]), "Decoder train target is hidden")
            _require(states.shape[1:] == (2,), "Decoder position target must contain x and y only")
            feature_rows.append(np.asarray(features[target], dtype=np.float32).copy())
            position_rows.append(np.asarray(states[target], dtype=np.float64).copy())
            row_ids.append({"row": len(row_ids), "split": "train", "pair": record["pair"], "episode_id": record["id"],
                            "horizon": horizon, "target_frame": target})
    X = np.stack(feature_rows)
    y = np.stack(position_rows)
    _require(X.shape == (2048, 5120) and y.shape == (2048, 2), "Unexpected decoder training matrix size")
    _require(np.isfinite(X).all() and np.isfinite(y).all(), "Non-finite decoder training inputs")
    groups = np.asarray([row["pair"] for row in row_ids])
    _require(set(Counter(groups).values()) == {4}, "Every decoder pair must provide two siblings times two horizons")
    check_budget("decoder:after_inputs")

    out = _new_output_dir(output_dir)
    ids_path = out / "decoder_fit_ids.json"
    _write_json(ids_path, {"source_split": "train", "rows": row_ids})
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
    membership = []
    for fold, (train, held_out) in enumerate(folds):
        train_pairs, held_out_pairs = sorted(set(groups[train])), sorted(set(groups[held_out]))
        _require(set(train_pairs).isdisjoint(held_out_pairs), "Decoder fold splits a sibling pair")
        membership.append({"fold": fold, "fit_pairs": train_pairs, "held_out_pairs": held_out_pairs,
                           "fit_rows": int(len(train)), "held_out_rows": int(len(held_out))})
    _require(Counter(p for fold in membership for p in fold["held_out_pairs"]) == Counter({p: 1 for p in counts}),
             "Each train pair must be held out in exactly one decoder fold")
    Readout = _load_readout_class()
    cv_rows = []
    best_ridge, best_score = None, float("inf")
    for ridge in RIDGES:
        predictions = np.full(y.shape, np.nan, dtype=np.float64)
        seen = np.zeros(len(y), dtype=np.int64)
        fold_results = []
        for fold, (train, held_out) in enumerate(folds):
            check_budget(f"decoder:before_cv_fit:ridge={ridge}:fold={fold}")
            model = Readout().fit(X[train], y[train], ridge)
            check_budget(f"decoder:after_cv_fit:ridge={ridge}:fold={fold}")
            _require(np.isfinite(model.mean).all() and np.isfinite(model.std).all() and np.isfinite(model.weight).all(),
                     "Non-finite decoder parameters")
            estimate = np.asarray(model.predict(X[held_out]), dtype=np.float64)
            _require(estimate.shape == y[held_out].shape and np.isfinite(estimate).all(), "Invalid decoder OOF prediction")
            predictions[held_out] = estimate
            seen[held_out] += 1
            fold_results.append({"fold": fold, "rows": int(len(held_out)),
                                 "position_mse": float(np.mean((estimate - y[held_out]) ** 2))})
            check_budget(f"decoder:after_cv_prediction:ridge={ridge}:fold={fold}")
        _require((seen == 1).all() and np.isfinite(predictions).all(), "Missing/duplicate decoder OOF rows")
        score = float(np.mean((predictions - y) ** 2))
        cv_rows.append({"ridge": ridge, "oof_position_mse": score, "folds": fold_results,
                        "oof_prediction_sha256": _array_sha(predictions)})
        if score < best_score:
            best_ridge, best_score = ridge, score
    _require(best_ridge in RIDGES and np.isfinite(best_score), "No valid decoder ridge selection")
    check_budget("decoder:before_final_train_fit")
    decoder = Readout().fit(X, y, best_ridge)
    check_budget("decoder:after_final_train_fit")
    for name in ("mean", "std", "weight", "bias"):
        array = np.asarray(getattr(decoder, name))
        _require(np.isfinite(array).all(), f"Non-finite final decoder {name}")
        array.setflags(write=False)
    decoder_path = out / "train_position_decoder.npz"
    # Preserve the original readout array schema without its atomic-write helper,
    # which may remove temporary files. This task forbids file deletion.
    with decoder_path.open("xb") as destination:
        np.savez_compressed(destination, mean=decoder.mean, std=decoder.std, weight=decoder.weight,
                            bias=decoder.bias, ridge=np.float64(decoder.ridge))
    cv_path = out / "decoder_cv.json"
    _write_json(cv_path, {"selection_split": "train", "group_unit": "pair", "folds": membership,
                          "ridge_grid": list(RIDGES), "scores": cv_rows, "selected_ridge": best_ridge,
                          "selection_metric": "overall out-of-fold mean squared error across rows and x/y coordinates",
                          "selected_oof_position_mse": best_score, "tie_rule": "smaller ridge on exact score tie",
                          "feature_scaling": "LinearReadout fits mean/std separately on each fold's fitting rows; final scaling fits all train rows"})
    provenance = {"schema_version": 1, "fit_split": "train", "hyperparameter_selection_split": "train",
                  "frozen_after_fit": True, "target": ["x", "y"], "train_pairs": 512, "episodes": 1024,
                  "rows": 2048, "features": 5120, "horizons": list(HORIZONS), "ridge": best_ridge,
                  "normalized_feature_array_sha256": _array_sha(X), "target_position_array_sha256": _array_sha(y),
                  "manifest_sha256": hashlib.sha256(_json_bytes(store.manifest)).hexdigest(),
                  "files_sha256": {p.name: _file_sha(p) for p in (ids_path, cv_path, decoder_path)},
                  "validation_used": False, "test_used": False}
    _write_json(out / "decoder_provenance.json", provenance)
    decoder.provenance = provenance
    check_budget("decoder:complete")
    return decoder


def _interval(values, indices):
    _require(values.shape == (96,) and np.isfinite(values).all(), "Incomplete/non-finite pair values")
    resampled = values[indices].mean(axis=1)
    bounds = np.quantile(resampled, [0.025, 0.975], method="linear")
    return {"mean": float(values.mean()), "ci95": [float(bounds[0]), float(bounds[1])], "pairs": 96}


def direction_lower_bound(true_positions):
    """Return the minimum pair position MSE for a shared sibling prediction.

    Input axes are [pair, sibling (2), horizon (3/6), xy (2)]. This CPU-only
    algebra does not import a model, fit anything, or access a dataset. It also
    accepts synthetic arrays with any positive pair count for software tests.
    """
    _require(true_positions is not None, "True target positions are required for the direction lower bound")
    positions = np.asarray(true_positions, dtype=np.float64)
    _require(positions.ndim == 4 and positions.shape[0] > 0 and positions.shape[1:] == (2, 2, 2),
             "True positions must have shape [pairs, 2 siblings, 2 horizons, 2 xy]")
    _require(np.isfinite(positions).all(), "Non-finite true target positions")
    midpoint = positions.mean(axis=1, keepdims=True)
    bound = np.square(midpoint - positions).mean(axis=(1, 3))
    _require(bound.shape == (len(positions), 2) and np.isfinite(bound).all() and (bound >= 0).all(),
             "Invalid/non-finite direction lower bound")
    return bound


def _position_reference_table(bound, decoder_pair_losses, indices):
    return [{"horizon": horizon,
             "B": _interval(bound[:, hi], indices),
             "decoder_true_target_position_mse": _interval(decoder_pair_losses[:, hi], indices)}
            for hi, horizon in enumerate(HORIZONS)]


def _tables(averages, indices, bound, decoder_pair_losses):
    table, contrasts = [], []
    for condition_index, condition in enumerate(CONDITIONS):
        for horizon_index, horizon in enumerate(HORIZONS):
            row = {"condition": condition, "history": HISTORIES[condition], "horizon": horizon}
            for metric, key in (("feature", "feature_mse"), ("position", "position_mse")):
                row[key] = _interval(averages[metric][:, condition_index, horizon_index], indices)
            table.append(row)
    for comparison, condition_index in (("d_minus_b", 3), ("c_minus_b", 2), ("a_minus_b", 0)):
        for horizon_index, horizon in enumerate(HORIZONS):
            row = {"comparison": comparison, "horizon": horizon, "positive_favors": "b"}
            for metric, key in (("feature", "feature_mse_improvement"), ("position", "position_mse_improvement")):
                differences = averages[metric][:, condition_index, horizon_index] - averages[metric][:, 1, horizon_index]
                row[key] = _interval(differences, indices)
            contrasts.append(row)
    for hi, horizon in enumerate(HORIZONS):
        contrasts.append({"comparison": "B_minus_b", "horizon": horizon, "positive_favors": "b",
                          "position_mse_improvement": _interval(bound[:, hi] - averages["position"][:, 1, hi], indices),
                          "required_gate": True})
        contrasts.append({"comparison": "d_minus_B", "horizon": horizon,
                          "positive_means": "d exceeds the identical-sibling-prediction lower bound",
                          "position_mse_excess": _interval(averages["position"][:, 3, hi] - bound[:, hi], indices),
                          "required_gate": False})
    return table, contrasts, _position_reference_table(bound, decoder_pair_losses, indices)


def evaluate_metrics(losses, output_dir=None, seed_values=SEEDS, pair_ids=None, *,
                     true_positions=None, decoder_true_target_losses=None):
    """Summarize fixed-seed losses and pair-bootstrap intervals without refitting.

losses must map 'feature' and 'position' to arrays shaped
[3 seeds, 96 pairs, 2 siblings, 4 conditions (a,b,c,d), 2 horizons (3,6)].
Position values must be mean squared error over x/y coordinates; feature values
must be mean squared error over the normalized 5120 target features. Both seed
and sibling axes are averaged before bootstrapping the 96 scene pairs. The same
2000 index draws are used for all absolute losses, contrasts and per-seed rows.

true_positions is mandatory [96 pairs, 2 siblings, 2 horizons, 2 xy]. The
direction lower bound B is computed once per pair/horizon, with no seed axis.
decoder_true_target_losses is mandatory [96 pairs, 2 siblings, 2 horizons]; it
contains the already-frozen decoder's MSE on true validation target features.
This readout diagnostic is not used for fitting, selection, or any gate.
"""
    _require(tuple(seed_values) == SEEDS, "Exactly seeds 17, 29 and 43 in registered order are required")
    _require(pair_ids is not None, "Validation pair identities are required")
    pair_ids = list(pair_ids)
    _require(len(pair_ids) == 96 and len(set(pair_ids)) == 96 and all(isinstance(p, str) and p for p in pair_ids),
             "Exactly 96 distinct validation pair ids are required")
    _require(set(losses) == {"feature", "position"}, "Loss dictionary must contain feature and position only")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in losses.items()}
    for name, value in arrays.items():
        _require(value.shape == (3, 96, 2, 4, 2), f"Missing/extra seed, pair, sibling, condition or horizon in {name} losses")
        _require(np.isfinite(value).all() and (value >= 0).all(), f"Non-finite or negative {name} MSE")
    _require(true_positions is not None, "True target positions are required; B cannot be omitted")
    positions = np.asarray(true_positions, dtype=np.float64)
    _require(positions.shape == (96, 2, 2, 2), "Validation true positions must have shape [96, 2, 2, 2]")
    bound = direction_lower_bound(positions)
    _require(decoder_true_target_losses is not None, "Frozen decoder true-target losses are required")
    decoder_losses = np.asarray(decoder_true_target_losses, dtype=np.float64)
    _require(decoder_losses.shape == (96, 2, 2), "Decoder true-target losses must have shape [96, 2, 2]")
    _require(np.isfinite(decoder_losses).all() and (decoder_losses >= 0).all(),
             "Non-finite or negative frozen-decoder true-target loss")
    decoder_pair_losses = decoder_losses.mean(axis=1)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 96, size=(BOOTSTRAP_REPEATS, 96))
    averaged = {name: value.mean(axis=2).mean(axis=0) for name, value in arrays.items()}
    table, contrasts, references = _tables(averaged, indices, bound, decoder_pair_losses)
    per_seed = []
    for seed_index, seed in enumerate(seed_values):
        seed_averaged = {name: value[seed_index].mean(axis=1) for name, value in arrays.items()}
        seed_table, seed_contrasts, seed_references = _tables(seed_averaged, indices, bound, decoder_pair_losses)
        per_seed.append({"seed": int(seed), "table": seed_table, "contrasts": seed_contrasts,
                         "position_reference_table": seed_references})
    primary_rows = [row for row in contrasts if row["comparison"] == "d_minus_b"]
    secondary_rows = [row for row in contrasts if row["comparison"] == "c_minus_b"]
    primary = {str(row["horizon"]): {"lower95": row["position_mse_improvement"]["ci95"][0],
                                      "passed": row["position_mse_improvement"]["ci95"][0] > 0.0} for row in primary_rows}
    secondary = {str(row["horizon"]): {"lower95": row["position_mse_improvement"]["ci95"][0],
                                        "passed": row["position_mse_improvement"]["ci95"][0] > 0.0} for row in secondary_rows}
    direction_rows = [row for row in contrasts if row["comparison"] == "B_minus_b"]
    direction = {str(row["horizon"]): {"lower95": row["position_mse_improvement"]["ci95"][0],
                                      "passed": row["position_mse_improvement"]["ci95"][0] > 0.0} for row in direction_rows}
    passed = all(item["passed"] for item in (*primary.values(), *secondary.values(), *direction.values()))
    result = {"schema_version": 2, "status": "passed" if passed else "stopped_predictor_gate_failed",
              "seed_values": list(seed_values), "condition_order": list(CONDITIONS), "horizons": list(HORIZONS),
              "validation_pairs": pair_ids, "raw_loss_shape": [3, 96, 2, 4, 2],
              "loss_definitions": {"feature_mse": "mean squared error across 5120 normalized target feature dimensions",
                                   "position_mse": "mean squared error across decoded x/y coordinates"},
              "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED, "unit": "scene pair",
                            "percentile_ci": [0.025, 0.975], "quantile_method": "linear",
                            "aggregation": "average two siblings and three fixed predictor seeds before resampling 96 pairs",
                            "shared_resamples_across_conditions_metrics_horizons_and_seed_tables": True,
                            "shared_resamples_include_B_and_frozen_decoder_true_target_diagnostic": True,
                            "B_has_no_seed_axis_and_is_not_an_extra_sample": True,
                            "interpretation": "Scene-pair uncertainty conditional on the three fitted seeds; not a population CI over model seeds",
                            "indices_sha256": _array_sha(indices)},
              "input_loss_sha256": {name: _array_sha(value) for name, value in arrays.items()},
              "input_true_positions_sha256": _array_sha(positions),
              "input_decoder_true_target_losses_sha256": _array_sha(decoder_losses),
              "direction_lower_bound": {"formula": "mean_sibling(mean_xy((mean_sibling(true_xy)-true_xy)^2))",
                                        "common_prediction_identity": "pair_loss = B + mean_xy((prediction-midpoint)^2)",
                                        "shape": [96, 2], "sha256": _array_sha(bound),
                                        "fit_required": False, "source_split": "validation"},
              "table": table, "contrasts": contrasts, "position_reference_table": references, "per_seed": per_seed,
              "gate": {"passed": passed, "primary_d_minus_b": primary, "secondary_c_minus_b": secondary,
                       "direction_B_minus_b": direction,
                       "rule": "All six decoded position MSE improvement lower 95% bounds must be strictly >0: d-b, c-b, and B-b, each at h=3 and h=6",
                       "a_minus_b_is_report_only": True, "feature_mse_is_report_only": True,
                       "d_minus_B_is_report_only": True, "decoder_true_target_loss_is_report_only": True,
                       "stop_after_pass_or_failure": True},
              "test_used": False, "retention_training_started": False}
    # Serializability check also rejects NaN before any results are persisted.
    _json_bytes(result)
    if output_dir is not None:
        out = _new_output_dir(output_dir)
        pair_path = out / "pair_averaged_losses.npz"
        with pair_path.open("xb") as handle:
            np.savez_compressed(handle, feature=averaged["feature"], position=averaged["position"],
                                B=bound, decoder_true_target_position_mse=decoder_pair_losses,
                                pair_ids=np.asarray(pair_ids), conditions=np.asarray(CONDITIONS), horizons=np.asarray(HORIZONS))
        indices_path = out / "bootstrap_indices.npz"
        with indices_path.open("xb") as handle:
            np.savez_compressed(handle, indices=indices)
        result["artifacts_sha256"] = {p.name: _file_sha(p) for p in (pair_path, indices_path)}
        _write_json(out / "metrics.json", result)
    return result
