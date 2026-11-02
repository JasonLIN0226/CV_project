"""Synthetic-only preregistered tests; importing never opens study artifacts."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("motion_step34_metrics_under_test", Path(__file__).with_name("step34_metrics.py"))
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
IDS = ["synthetic_pair_" + str(index) for index in range(96)]


def diagnostic_inputs():
    positions = np.zeros((96, 2, 2, 2), dtype=np.float64)
    positions[:, 0, :, 0] = -1.0
    positions[:, 1, :, 0] = 1.0
    losses = {method: {"feature": np.full((3, 96, 2, 2), feature),
                       "position": np.full((3, 96, 2, 2), position)}
              for method, feature, position in (("b", .2, .1), ("d", .9, .8), ("b_random", .6, .4))}
    return losses, positions, np.full((96, 2, 2), .01)


def summarize_diagnostic(losses=None, positions=None, decoder=None, **kwargs):
    defaults = diagnostic_inputs()
    return METRICS.summarize_diagnostics(defaults[0] if losses is None else losses,
                                         defaults[1] if positions is None else positions,
                                         defaults[2] if decoder is None else decoder,
                                         kwargs.pop("output", None), pair_ids=kwargs.pop("pair_ids", IDS), **kwargs)


def online_inputs():
    results, steps = {}, {}
    pair_trend = np.arange(96, dtype=np.float64)[None, None, :, None, None] * .001
    reader_trend = np.arange(3, dtype=np.float64)[None, :, None, None, None] * .02
    sibling_trend = np.array([0.0, .03])[None, None, None, :, None]
    run_trend = np.arange(3, dtype=np.float64)[:, None, None, None, None] * .1
    for architecture in ("set", "independent"):
        for label in METRICS.LABELS:
            name = "student/" + architecture + "/" + label
            base = .2 if architecture == "set" and label in METRICS.COST_LABELS else .7
            values = np.broadcast_to(base + pair_trend + reader_trend + sibling_trend + run_trend,
                                     (3, 3, 96, 2, 2)).copy()
            results[name] = {"feature": values + .3, "position": values,
                             "success": np.full((3, 96, 2), base == .2, dtype=np.bool_)}
            steps[name] = [100, 200, 300]
    for baseline in METRICS.BASELINES:
        name = "baseline/" + baseline
        values = np.broadcast_to(1.0 + pair_trend + reader_trend + sibling_trend, (3, 3, 96, 2, 2)).copy()
        results[name] = {"feature": values + .3, "position": values,
                         "success": np.zeros((3, 96, 2), dtype=np.bool_)}
    return results, steps


def summarize_online(results=None, steps=None, **kwargs):
    defaults = online_inputs()
    return METRICS.summarize_online(defaults[0] if results is None else results, kwargs.pop("output", None),
                                    trained_steps=defaults[1] if steps is None else steps,
                                    pair_ids=kwargs.pop("pair_ids", IDS), **kwargs)


@pytest.fixture(scope="module")
def online_summary():
    return summarize_online()


def test_diagnostic_bound_and_shared_prediction_identity():
    result = summarize_diagnostic(split="long_gap")
    assert result["claim_scope"] == "long_gap_reader_qualified"
    assert result["long_gap_direction_check"]["passed"] is True
    for row in result["position_reference_table"]:
        assert row["B"] == {"mean": .5, "ci95": [.5, .5], "pairs": 96}
        assert row["decoder_true_target_position_mse"]["mean"] == pytest.approx(.01)
    row = next(row for row in result["contrasts"] if row["comparison"] == "d_minus_B")
    assert row["position_mse_excess"]["mean"] == pytest.approx(.3)
    assert len(result["table"]) == 6
    assert len(result["per_reader"]) == 3


def test_ordinary_diagnostic_never_qualifies_long_gap():
    result = summarize_diagnostic()
    assert result["long_gap_direction_check"]["passed"] is None
    assert result["claim_scope"] == "validation_only_no_long_gap_qualification"


def test_random_correct_frame_comparisons_are_report_only():
    result = summarize_diagnostic(split="long_gap")
    expected = {"B_minus_b_random": (.1, None), "d_minus_b_random": (.4, .3),
                "b_random_minus_b": (.3, .4)}
    rows = [row for row in result["contrasts"] if row["comparison"] in expected]
    assert len(rows) == 6
    for row in rows:
        position, feature = expected[row["comparison"]]
        assert row["position_mse_improvement"]["mean"] == pytest.approx(position)
        assert row["report_only"] is True and row["required_gate"] is False
        if feature is None:
            assert "feature_mse_improvement" not in row
        else:
            assert row["feature_mse_improvement"]["mean"] == pytest.approx(feature)
    assert len(result["per_reader"][0]["contrasts"]) == len(result["contrasts"])
    losses, positions, decoder = diagnostic_inputs()
    losses["b_random"]["position"][:] = 1000
    result = summarize_diagnostic(losses, positions, decoder, split="long_gap")
    assert result["long_gap_direction_check"]["passed"] is True


def test_zero_direction_lower_bound_fails_only_long_gap_scope():
    losses, positions, decoder = diagnostic_inputs()
    losses["b"]["position"][..., 1] = .5
    result = summarize_diagnostic(losses, positions, decoder, split="long_gap")
    assert result["long_gap_direction_check"]["horizons"]["3"]["passed"] is True
    assert result["long_gap_direction_check"]["horizons"]["6"]["passed"] is False
    assert result["claim_scope"] == "long_gap_reader_not_qualified"


def test_paired_difference_preserves_common_scene_variation():
    losses, positions, decoder = diagnostic_inputs()
    trend = np.arange(96, dtype=np.float64)[None, :, None, None] / 96
    losses["b"]["position"] += trend
    losses["d"]["position"] += trend
    result = summarize_diagnostic(losses, positions, decoder)
    for row in result["contrasts"]:
        if row["comparison"] == "d_minus_b":
            assert row["position_mse_improvement"]["ci95"] == pytest.approx([.7, .7])


@pytest.mark.parametrize("bad", [None, IDS[:-1], ["duplicate"] * 96, list(range(96))])
def test_missing_or_invalid_pair_identity_rejected(bad):
    with pytest.raises(ValueError):
        summarize_diagnostic(pair_ids=bad)


@pytest.mark.parametrize("split", ["test", "confirmation", "validation", "train", "test_long_gap"])
def test_other_splits_rejected(split):
    with pytest.raises(ValueError):
        summarize_diagnostic(split=split)
    with pytest.raises(ValueError):
        summarize_online(split=split)


@pytest.mark.parametrize("seed", [17, True, 901.0, "901"])
def test_bootstrap_seed_cannot_drift(seed):
    with pytest.raises(ValueError):
        summarize_diagnostic(bootstrap_seed=seed)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.01])
def test_nonfinite_or_negative_diagnostic_losses_rejected(bad):
    losses, positions, decoder = diagnostic_inputs()
    losses["b"]["position"][0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        summarize_diagnostic(losses, positions, decoder)


def test_wrong_diagnostic_axes_and_boolean_loss_rejected():
    losses, positions, decoder = diagnostic_inputs()
    losses["b"]["feature"] = losses["b"]["feature"].transpose(1, 0, 2, 3)
    with pytest.raises(ValueError):
        summarize_diagnostic(losses, positions, decoder)
    losses["b"]["feature"] = np.zeros((3, 96, 2, 2), dtype=bool)
    with pytest.raises(ValueError):
        summarize_diagnostic(losses, positions, decoder)


def test_online_seed_and_sibling_axes_average_before_pairs(online_summary):
    row = next(row for row in online_summary["table"] if row["method"] == "student/set/use_time_position" and row["horizon"] == 3)
    assert row["position_mse"]["mean"] == pytest.approx(.2 + .1 + .02 + .0475 + .015)
    seed_row = next(row for row in online_summary["per_student_seed"][0]["table"]
                    if row["method"] == "student/set/use_time_position" and row["horizon"] == 3)
    assert seed_row["position_mse"]["mean"] == pytest.approx(.2 + .02 + .0475 + .015)
    assert row["position_mse"]["pairs"] == 96
    assert online_summary["bootstrap"]["unit"] == "scene pair"


def test_all_prespecified_labels_and_comparators_reported(online_summary):
    assert len(online_summary["table"]) == 50
    assert len(online_summary["comparisons"]) == 32
    assert len(online_summary["per_student_seed"]) == 3
    assert set(online_summary["label_claims"]) == set(METRICS.COST_LABELS)
    for label, claim in online_summary["label_claims"].items():
        assert claim["validation_criterion_passed"] is True
        assert claim["confirmation_claim_established"] is False
        assert claim["selection_after_results_allowed"] is False
        assert claim["role"] == ("prespecified_primary" if label == "use_time_position" else "prespecified_secondary")
    assert all(row["feature_is_report_only"] for row in online_summary["comparisons"])


def test_references_never_become_comparators_and_copies_declared():
    results, steps = online_inputs()
    for reference in METRICS.REFERENCES:
        results["reference/" + reference] = copy.deepcopy(results["baseline/fifo"])
    fixed = [method for method in results if not method.startswith("student/")]
    result = summarize_online(results, steps, replicated_methods=fixed)
    assert len(result["table"]) == 56
    assert set(result["replicated_fixed_methods"]) == set(fixed)
    assert all(not row["comparator"].startswith("reference/") for row in result["comparisons"])


def test_step0_and_null_set_conditioning_reported():
    results, steps = online_inputs()
    method = "student/set/use_time_position"
    steps[method] = [100, 0, 300]
    results["student/independent/use_time_position"] = copy.deepcopy(results[method])
    # A feature-only failure must not change the success/position gate.
    results[method]["feature"][:] = 1000
    result = summarize_online(results, steps)
    claim = result["label_claims"]["use_time_position"]
    assert claim["all_numeric_comparisons_passed"] is True
    assert claim["fully_trained_comparison_allowed"] is False
    assert claim["validation_criterion_passed"] is False
    assert result["training_checkpoint_status"][method]["includes_initialization"] is True
    row = next(row for row in result["set_conditioning"] if row["label"] == "use_time_position")
    assert row["cannot_distinguish_set_conditioning"] is True
    assert row["success_cannot_distinguish_set_conditioning"] is True
    assert all(horizon["cannot_distinguish_set_conditioning"] for horizon in row["horizons"])


@pytest.mark.parametrize("steps", [None, {}, {"student/set/use_time_position": [1, 2, 3]}])
def test_incomplete_training_metadata_rejected(steps):
    results, _ = online_inputs()
    with pytest.raises(ValueError):
        METRICS.summarize_online(results, None, trained_steps=steps, pair_ids=IDS)


@pytest.mark.parametrize("bad", [True, 100.0, "100", -1, None])
def test_invalid_selected_step_rejected(bad):
    results, steps = online_inputs()
    steps["student/set/use_time_position"][0] = bad
    with pytest.raises(ValueError):
        summarize_online(results, steps)


def test_required_student_or_baseline_cannot_be_omitted():
    results, steps = online_inputs()
    del results["baseline/keep_first"]
    with pytest.raises(ValueError):
        summarize_online(results, steps)


@pytest.mark.parametrize("bad", [np.zeros((3, 96, 2), dtype=np.int8), np.zeros((3, 96), dtype=bool)])
def test_success_requires_boolean_full_axes(bad):
    results, steps = online_inputs()
    results["baseline/fifo"]["success"] = bad
    with pytest.raises(ValueError):
        summarize_online(results, steps)


def test_declared_copied_runs_must_be_exact():
    results, steps = online_inputs()
    results["baseline/fifo"]["position"][1, 0, 0, 0, 0] += .1
    with pytest.raises(ValueError):
        summarize_online(results, steps, replicated_methods=["baseline/fifo"])


def test_zero_success_difference_is_not_strict_improvement():
    results, steps = online_inputs()
    results["baseline/fifo"]["success"][:] = True
    result = summarize_online(results, steps)
    assert all(not value["all_numeric_comparisons_passed"] for value in result["label_claims"].values())


def test_shared_bootstrap_and_inputs_unchanged(online_summary):
    losses, positions, decoder = diagnostic_inputs()
    original = copy.deepcopy(losses)
    true_copy, decoder_copy = positions.copy(), decoder.copy()
    result = summarize_diagnostic(losses, positions, decoder)
    assert result["bootstrap"]["indices_sha256"] == online_summary["bootstrap"]["indices_sha256"]
    assert np.array_equal(positions, true_copy) and np.array_equal(decoder, decoder_copy)
    for method in losses:
        for metric in losses[method]:
            assert np.array_equal(losses[method][metric], original[method][metric])


def test_exclusive_outputs_and_sha256_receipts(tmp_path):
    output = tmp_path / "new_summary"
    result = summarize_diagnostic(output=output)
    assert json.loads((output / "summary.json").read_text()) == result
    status = json.loads((output / "status.json").read_text())
    assert status["summary_sha256"] == hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()
    index_sha = hashlib.sha256((output / "bootstrap_indices.npz").read_bytes()).hexdigest()
    assert status["bootstrap_indices_sha256"] == index_sha
    assert result["bootstrap_indices_artifact"]["sha256"] == index_sha
    with np.load(output / "bootstrap_indices.npz", allow_pickle=False) as archive:
        assert archive.files == ["indices"]
        indices = archive["indices"]
    assert indices.shape == (2000, 96)
    assert METRICS._sha_array(indices) == result["bootstrap"]["indices_sha256"]
    assert np.array_equal(indices, np.random.default_rng(901).integers(0, 96, size=(2000, 96)))
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ")
        assert expected == hashlib.sha256((output / name).read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        summarize_diagnostic(output=output)


def test_dangling_output_symlink_rejected(tmp_path):
    output = tmp_path / "dangling"
    output.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        summarize_diagnostic(output=output)


def test_choices_and_traces_can_preexist_without_being_changed(tmp_path):
    output = tmp_path / "stage_evidence"
    (output / "traces").mkdir(parents=True)
    (output / "frame_choices.json").write_text('{"already": "recorded"}\n')
    (output / "traces" / "first.json").write_text("preserved trace\n")
    summarize_diagnostic(output=output)
    assert (output / "frame_choices.json").read_text() == '{"already": "recorded"}\n'
    assert (output / "traces" / "first.json").read_text() == "preserved trace\n"
    assert (output / "summary.json").is_file()


@pytest.mark.parametrize("owned", ["summary.json", "status.json", "SHA256SUMS", "bootstrap_indices.npz"])
def test_existing_own_receipt_prevents_any_new_summary_output(tmp_path, owned):
    output = tmp_path / "existing"
    output.mkdir()
    (output / owned).write_text("keep existing receipt\n")
    with pytest.raises(ValueError):
        summarize_diagnostic(output=output)
    assert [path.name for path in output.iterdir()] == [owned]
    assert (output / owned).read_text() == "keep existing receipt\n"
