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


def row_for(result, comparison, horizon):
    return next(r for r in result["contrasts"] if r["comparison"] == comparison and r["horizon"] == horizon)


def test_positive_improvement_sign_and_fixed_loss_table(metrics):
    result = metrics.evaluate_metrics(constant_losses(), pair_ids=pair_ids())
    assert result["gate"]["passed"]
    assert len(result["table"]) == 8
    for horizon in (3, 6):
        delta = row_for(result, "d_minus_b", horizon)["position_mse_improvement"]
        assert delta == {"mean": 1.0, "ci95": [1.0, 1.0], "pairs": 96}
        assert row_for(result, "c_minus_b", horizon)["position_mse_improvement"]["mean"] == 2.0
        assert row_for(result, "a_minus_b", horizon)["position_mse_improvement"]["mean"] == 3.0


@pytest.mark.parametrize("horizon_index", [0, 1])
def test_both_primary_horizons_need_strictly_positive_lower_bound(metrics, horizon_index):
    losses = constant_losses()
    losses["position"][:, :, :, 3, horizon_index] = 1.0
    result = metrics.evaluate_metrics(losses, pair_ids=pair_ids())
    failed = str((3, 6)[horizon_index])
    other = str((3, 6)[1 - horizon_index])
    assert result["gate"]["primary_d_minus_b"][failed] == {"lower95": 0.0, "passed": False}
    assert result["gate"]["primary_d_minus_b"][other]["passed"]
    assert not result["gate"]["passed"]


def test_c_comparison_required_but_a_and_feature_not_gate(metrics):
    losses = constant_losses()
    losses["position"][:, :, :, 0, :] = 0.0  # a is better than b; report only.
    losses["feature"][:, :, :, 3, :] = 0.0  # feature d-b is negative; report only.
    assert metrics.evaluate_metrics(losses, pair_ids=pair_ids())["gate"]["passed"]
    losses["position"][:, :, :, 2, 1] = 0.5
    result = metrics.evaluate_metrics(losses, pair_ids=pair_ids())
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
    result = metrics.evaluate_metrics(losses, output_dir=out, pair_ids=pair_ids())
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
    with np.load(out / "pair_averaged_losses.npz", allow_pickle=False) as saved:
        assert saved["position"].shape == (96, 4, 2)
        assert saved["position"][:, 3, 0] - saved["position"][:, 1, 0] == pytest.approx(pair_delta, abs=1e-14)
    with pytest.raises(ValueError, match="overwrite"):
        metrics.evaluate_metrics(losses, output_dir=out, pair_ids=pair_ids())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -0.1])
def test_nonfinite_or_negative_losses_stop(metrics, bad_value):
    losses = constant_losses()
    losses["position"][0, 0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="Non-finite or negative"):
        metrics.evaluate_metrics(losses, pair_ids=pair_ids())


def test_missing_seed_or_pair_stops(metrics):
    with pytest.raises(ValueError, match="seeds 17, 29 and 43"):
        metrics.evaluate_metrics(constant_losses(), seed_values=(17, 29), pair_ids=pair_ids())
    losses = constant_losses()
    losses["position"] = losses["position"][:, :-1]
    with pytest.raises(ValueError, match="Missing/extra"):
        metrics.evaluate_metrics(losses, pair_ids=pair_ids())
    duplicate_ids = pair_ids()
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="96 distinct"):
        metrics.evaluate_metrics(constant_losses(), pair_ids=duplicate_ids)
    with pytest.raises(ValueError, match="identities are required"):
        metrics.evaluate_metrics(constant_losses())
