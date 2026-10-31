"""Software-only checks of registered statistics; no episode or model is used.

These artificial loss arrays are not experimental train/validation/test data.
No decoder fitting, feature encoding, or model import occurs in this module.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def metrics():
    source = Path(__file__).with_name("step2_metrics.py")
    spec = importlib.util.spec_from_file_location("motion_v2_statistics_software_check", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_ids():
    return [f"software-only-pair-{i:03d}" for i in range(96)]


def constant_losses():
    values = np.empty((3, 96, 2, 4, 2), dtype=np.float64)
    for condition, value in enumerate((4.0, 1.0, 3.0, 2.0)):
        values[:, :, :, condition, :] = value
    return {"feature": values.copy(), "position": values.copy()}


def true_positions():
    # Siblings (0,0) and (4,0) have midpoint (2,0) and xy-averaged B=2.
    positions = np.zeros((96, 2, 2, 2), dtype=np.float64)
    positions[:, 1, :, 0] = 4.0
    return positions


def decoder_losses():
    return np.full((96, 2, 2), 0.25, dtype=np.float64)


def evaluate(metrics, losses, **kwargs):
    kwargs.setdefault("true_positions", true_positions())
    kwargs.setdefault("decoder_true_target_losses", decoder_losses())
    return metrics.evaluate_metrics(losses, **kwargs)


def row_for(result, comparison, horizon):
    return next(r for r in result["contrasts"] if r["comparison"] == comparison and r["horizon"] == horizon)


def test_positive_improvement_sign_and_fixed_loss_table(metrics):
    result = evaluate(metrics, constant_losses(), pair_ids=pair_ids())
    assert result["gate"]["passed"]
    assert len(result["table"]) == 8
    for horizon in (3, 6):
        delta = row_for(result, "d_minus_b", horizon)["position_mse_improvement"]
        assert delta == {"mean": 1.0, "ci95": [1.0, 1.0], "pairs": 96}
        assert row_for(result, "c_minus_b", horizon)["position_mse_improvement"]["mean"] == 2.0
        assert row_for(result, "a_minus_b", horizon)["position_mse_improvement"]["mean"] == 3.0
        assert row_for(result, "B_minus_b", horizon)["position_mse_improvement"] == {
            "mean": 1.0, "ci95": [1.0, 1.0], "pairs": 96}
        assert row_for(result, "d_minus_B", horizon)["position_mse_excess"]["mean"] == 0.0
        reference = next(row for row in result["position_reference_table"] if row["horizon"] == horizon)
        assert reference["B"] == {"mean": 2.0, "ci95": [2.0, 2.0], "pairs": 96}
        assert reference["decoder_true_target_position_mse"]["mean"] == 0.25


@pytest.mark.parametrize("horizon_index", [0, 1])
def test_both_primary_horizons_need_strictly_positive_lower_bound(metrics, horizon_index):
    losses = constant_losses()
    losses["position"][:, :, :, 3, horizon_index] = 1.0
    result = evaluate(metrics, losses, pair_ids=pair_ids())
    failed = str((3, 6)[horizon_index])
    other = str((3, 6)[1 - horizon_index])
    assert result["gate"]["primary_d_minus_b"][failed] == {"lower95": 0.0, "passed": False}
    assert result["gate"]["primary_d_minus_b"][other]["passed"]
    assert not result["gate"]["passed"]


def test_c_comparison_required_but_a_and_feature_not_gate(metrics):
    losses = constant_losses()
    losses["position"][:, :, :, 0, :] = 0.0  # a is better than b; report only.
    losses["feature"][:, :, :, 3, :] = 0.0  # feature d-b is negative; report only.
    assert evaluate(metrics, losses, pair_ids=pair_ids())["gate"]["passed"]
    losses["position"][:, :, :, 2, 1] = 0.5
    result = evaluate(metrics, losses, pair_ids=pair_ids())
    assert all(value["passed"] for value in result["gate"]["primary_d_minus_b"].values())
    assert result["gate"]["secondary_c_minus_b"]["3"]["passed"]
    assert not result["gate"]["secondary_c_minus_b"]["6"]["passed"]
    assert not result["gate"]["passed"]


def test_average_siblings_and_seeds_before_pair_bootstrap(metrics):
    losses = constant_losses()
    pair_delta = np.linspace(-2.0, 2.0, 96)
    seed_offset = np.asarray([-1.0, 0.0, 1.0])
    sibling_offset = np.asarray([-0.25, 0.25])
    losses["position"][:] = 10.0
    for seed in range(3):
        for sibling in range(2):
            for horizon in range(2):
                losses["position"][seed, :, sibling, 3, horizon] = 10.0 + pair_delta + seed_offset[seed] + sibling_offset[sibling]
    losses["feature"] = losses["position"] + 10.0
    # Do not use pytest's tmp_path cleanup: this task forbids deleting files.
    destination = os.environ.get("MOTION_V2_SOFTWARE_TEST_OUTPUT")
    assert destination, "Set MOTION_V2_SOFTWARE_TEST_OUTPUT to a new retained result directory"
    out = Path(destination)
    assert not out.exists(), "Software-test artifacts must be retained without overwrite"
    bound_by_pair = np.linspace(1.0, 3.0, 96)
    positions = true_positions()
    positions[:, 1, :, 0] = (2.0 * np.sqrt(2.0 * bound_by_pair))[:, None]
    diagnostic = decoder_losses()
    diagnostic[:, 0, :] = np.linspace(0.1, 0.4, 96)[:, None]
    diagnostic[:, 1, :] = diagnostic[:, 0, :] + 0.2
    result = evaluate(metrics, losses, output_dir=out, pair_ids=pair_ids(),
                      true_positions=positions, decoder_true_target_losses=diagnostic)
    with np.load(out / "bootstrap_indices.npz", allow_pickle=False) as saved:
        indices = saved["indices"]
    assert indices.shape == (2000, 96)
    assert np.array_equal(indices, np.random.default_rng(901).integers(0, 96, size=(2000, 96)))
    expected_interval = np.quantile(pair_delta[indices].mean(axis=1), [0.025, 0.975], method="linear")
    actual = row_for(result, "d_minus_b", 3)["position_mse_improvement"]
    assert actual["pairs"] == 96  # Never pseudo-replicate 3 seeds x 2 siblings as 576 pairs.
    assert actual["mean"] == pytest.approx(0.0, abs=1e-14)
    assert actual["ci95"] == pytest.approx(expected_interval.tolist(), abs=1e-14)
    assert actual["ci95"][1] - actual["ci95"][0] > 0.35
    for row in result["table"]:
        assert row["feature_mse"]["mean"] == pytest.approx(row["position_mse"]["mean"] + 10.0)
        assert row["feature_mse"]["ci95"] == pytest.approx(np.asarray(row["position_mse"]["ci95"]) + 10.0)
    for index, seed_result in enumerate(result["per_seed"]):
        delta = row_for(seed_result, "d_minus_b", 3)["position_mse_improvement"]
        assert delta["ci95"] == pytest.approx(expected_interval + seed_offset[index], abs=1e-14)
        b_delta = row_for(seed_result, "B_minus_b", 3)["position_mse_improvement"]
        assert b_delta["ci95"] == pytest.approx(
            np.quantile((bound_by_pair - 10.0)[indices].mean(axis=1), [0.025, 0.975], method="linear"), abs=1e-14)
        assert seed_result["position_reference_table"] == result["position_reference_table"]
    for reference in result["position_reference_table"]:
        expected_B = np.quantile(bound_by_pair[indices].mean(axis=1), [0.025, 0.975], method="linear")
        expected_decoder = np.quantile(diagnostic.mean(axis=1)[:, 0][indices].mean(axis=1),
                                       [0.025, 0.975], method="linear")
        assert reference["B"]["ci95"] == pytest.approx(expected_B, abs=1e-14)
        assert reference["decoder_true_target_position_mse"]["ci95"] == pytest.approx(expected_decoder, abs=1e-14)
        d_excess = row_for(result, "d_minus_B", reference["horizon"])["position_mse_excess"]
        expected_d_excess = np.quantile((10.0 + pair_delta - bound_by_pair)[indices].mean(axis=1),
                                        [0.025, 0.975], method="linear")
        assert d_excess["ci95"] == pytest.approx(expected_d_excess, abs=1e-14)
    with np.load(out / "pair_averaged_losses.npz", allow_pickle=False) as saved:
        assert saved["position"].shape == (96, 4, 2)
        assert saved["position"][:, 3, 0] - saved["position"][:, 1, 0] == pytest.approx(pair_delta, abs=1e-14)
        assert saved["B"].shape == (96, 2)
        assert saved["B"][:, 0] == pytest.approx(bound_by_pair, abs=1e-14)
        assert saved["decoder_true_target_position_mse"] == pytest.approx(diagnostic.mean(axis=1), abs=1e-14)
    with pytest.raises(ValueError, match="overwrite"):
        evaluate(metrics, losses, output_dir=out, pair_ids=pair_ids())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -0.1])
def test_nonfinite_or_negative_losses_stop(metrics, bad_value):
    losses = constant_losses()
    losses["position"][0, 0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="Non-finite or negative"):
        evaluate(metrics, losses, pair_ids=pair_ids())


def test_missing_seed_or_pair_stops(metrics):
    with pytest.raises(ValueError, match="seeds 17, 29 and 43"):
        evaluate(metrics, constant_losses(), seed_values=(17, 29), pair_ids=pair_ids())
    losses = constant_losses()
    losses["position"] = losses["position"][:, :-1]
    with pytest.raises(ValueError, match="Missing/extra"):
        evaluate(metrics, losses, pair_ids=pair_ids())
    duplicate_ids = pair_ids()
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="96 distinct"):
        evaluate(metrics, constant_losses(), pair_ids=duplicate_ids)
    with pytest.raises(ValueError, match="identities are required"):
        evaluate(metrics, constant_losses())


def test_shared_prediction_lower_bound_identity_and_midpoint_equality(metrics):
    # Deliberately asymmetric artificial coordinates exercise both axes/horizons.
    positions = np.arange(5 * 2 * 2 * 2, dtype=np.float64).reshape(5, 2, 2, 2) / 10.0
    midpoint = positions.mean(axis=1)
    bound = metrics.direction_lower_bound(positions)
    for offset in (np.zeros((5, 2, 2)), np.full((5, 2, 2), 0.3), -midpoint):
        common_prediction = midpoint + offset
        pair_loss = np.square(common_prediction[:, None] - positions).mean(axis=(1, 3))
        assert pair_loss == pytest.approx(bound + np.square(offset).mean(axis=-1), abs=1e-14)
        assert np.all(pair_loss >= bound - 1e-14)
    midpoint_loss = np.square(midpoint[:, None] - positions).mean(axis=(1, 3))
    assert np.array_equal(midpoint_loss, bound)


def test_bound_uses_float64_without_changing_input(metrics):
    positions = true_positions().astype(np.float32)
    original = positions.copy()
    bound = metrics.direction_lower_bound(positions)
    assert bound.dtype == np.float64
    assert bound.shape == (96, 2)
    assert np.array_equal(positions, original)


def test_B_minus_b_aggregates_siblings_and_fixed_seeds_but_keeps_per_seed_rows(metrics):
    losses = constant_losses()
    seed_values = (0.5, 1.0, 1.5)
    for seed_index, value in enumerate(seed_values):
        losses["position"][seed_index, :, 0, 1, :] = value - 0.125
        losses["position"][seed_index, :, 1, 1, :] = value + 0.125
    result = evaluate(metrics, losses, pair_ids=pair_ids())
    for horizon in (3, 6):
        assert row_for(result, "B_minus_b", horizon)["position_mse_improvement"] == {
            "mean": 1.0, "ci95": [1.0, 1.0], "pairs": 96}
        for seed_index, seed_result in enumerate(result["per_seed"]):
            expected = 2.0 - seed_values[seed_index]
            assert row_for(seed_result, "B_minus_b", horizon)["position_mse_improvement"] == {
                "mean": expected, "ci95": [expected, expected], "pairs": 96}


@pytest.mark.parametrize("horizon_index", [0, 1])
def test_direction_bound_is_mandatory_and_each_horizon_is_independent(metrics, horizon_index):
    losses = constant_losses()
    losses["position"][:, :, :, 2, :] = 4.0
    losses["position"][:, :, :, 3, :] = 3.0
    losses["position"][:, :, :, 1, horizon_index] = 2.0  # Equals B; strict lower must fail.
    result = evaluate(metrics, losses, pair_ids=pair_ids())
    failed, other = str((3, 6)[horizon_index]), str((3, 6)[1 - horizon_index])
    assert result["gate"]["direction_B_minus_b"][failed] == {"lower95": 0.0, "passed": False}
    assert result["gate"]["direction_B_minus_b"][other]["passed"]
    assert all(row["passed"] for row in result["gate"]["primary_d_minus_b"].values())
    assert all(row["passed"] for row in result["gate"]["secondary_c_minus_b"].values())
    assert not result["gate"]["passed"]
    for horizon in (3, 6):
        excess = row_for(result, "d_minus_B", horizon)
        assert excess["position_mse_excess"] == {"mean": 1.0, "ci95": [1.0, 1.0], "pairs": 96}
        assert not excess["required_gate"]


def test_bound_and_decoder_diagnostic_cannot_be_omitted(metrics):
    with pytest.raises(ValueError, match="True target positions are required"):
        metrics.evaluate_metrics(constant_losses(), pair_ids=pair_ids(), decoder_true_target_losses=decoder_losses())
    with pytest.raises(ValueError, match="Frozen decoder true-target losses are required"):
        metrics.evaluate_metrics(constant_losses(), pair_ids=pair_ids(), true_positions=true_positions())
    with pytest.raises(ValueError, match="feature and position only"):
        evaluate(metrics, {"position": constant_losses()["position"]}, pair_ids=pair_ids())


@pytest.mark.parametrize("shape", [(96, 2, 2), (96, 1, 2, 2), (96, 2, 1, 2), (96, 2, 2, 3), (0, 2, 2, 2)])
def test_invalid_position_shapes_rejected(metrics, shape):
    with pytest.raises(ValueError, match="shape"):
        metrics.direction_lower_bound(np.zeros(shape))


def test_metric_position_pair_count_and_decoder_shape_rejected(metrics):
    with pytest.raises(ValueError, match="Validation true positions.*shape"):
        evaluate(metrics, constant_losses(), pair_ids=pair_ids(), true_positions=true_positions()[:-1])
    with pytest.raises(ValueError, match="Decoder true-target losses.*shape"):
        evaluate(metrics, constant_losses(), pair_ids=pair_ids(), decoder_true_target_losses=np.zeros((96, 2)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_positions_rejected(metrics, bad_value):
    positions = true_positions()
    positions[0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="Non-finite true target positions"):
        evaluate(metrics, constant_losses(), pair_ids=pair_ids(), true_positions=positions)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -0.01])
def test_invalid_decoder_losses_rejected(metrics, bad_value):
    diagnostic = decoder_losses()
    diagnostic[0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="Non-finite or negative frozen-decoder"):
        evaluate(metrics, constant_losses(), pair_ids=pair_ids(), decoder_true_target_losses=diagnostic)


def test_decoder_diagnostic_does_not_change_gates(metrics):
    result = evaluate(metrics, constant_losses(), pair_ids=pair_ids(),
                      decoder_true_target_losses=np.full((96, 2, 2), 100.0))
    assert result["gate"]["passed"]
    assert result["gate"]["decoder_true_target_loss_is_report_only"]
