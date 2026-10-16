import copy
from pathlib import Path

import numpy as np
import pytest

from phase1.config import DEFAULT, CONDITIONS, SPLITS, scientific_hash
from phase1.evaluation import METRICS, pair_calibration, summarize, summarize_calibration
from phase1.reporting import decision, development_comparison
from phase1.utils import code_hash, file_hash, read_json, write_csv, write_json


def _arm(root, task, arm, privileged=.4, failed_seed=None, recent=1.0):
    c = copy.deepcopy(DEFAULT)
    c.update(mode="confirm", output=str(root))
    c["encoder"]["backend"] = "ijepa"
    c["data"]["pairs"] = {s: 32 for s in SPLITS}
    c["training"].update(epochs=120, seeds=[17, 29, 43])
    c["evaluation"]["bootstrap"] = 100
    c["experiment"] = {"arm": arm, "stage": "development", "tasks": [task]}
    write_json(root/"resolved_config.json", c)
    audit = {"passed": True}
    write_json(root/"dataset/split_audit.json", audit)
    probes = {"results": []}
    for split in SPLITS[2:]:
        for variant, accuracy in [("gate_cue" if task == "gate" else "motion_two_frame", 1.), ("recent_only", .5)]:
            probes["results"].append({"task": task, "split": split, "variant": variant,
                                     "accuracy": {"mean": accuracy}, "wilson_effective_pairs": [.8, 1.]})
        for h in c["data"]["horizons"]:
            probes["results"].append({"task": task, "split": split, "variant": "state_decoder_on_true_target", "horizon": h,
                                     "accuracy": {"mean": 1.}, "position_mse": {"mean": .0001}})
    write_json(root/"probes/metrics.json", probes)
    raw = []
    for seed in c["training"]["seeds"]:
        for split in SPLITS[1:]:
            for h in c["data"]["horizons"]:
                for cond in CONDITIONS+["persistence"]:
                    for pair in range(32):
                        for sibling in [0, 1]:
                            mse = {"recent_only": recent, "full_history": .4, "privileged_small_budget": privileged,
                                   "random_small_budget": .8, "wrong_evidence": 1.2, "persistence": 1.}[cond]
                            if failed_seed == seed and cond == "privileged_small_budget" and split == "test_distractor_shift":
                                mse = 1.2
                            state_accuracy = .5 if cond in {"recent_only", "persistence"} else .9
                            row = {"seed": seed, "task": task, "split": split, "condition": cond, "horizon": h,
                                   "episode": f"{split}-{pair}-{sibling}", "pair": f"{split}-{pair}", "query": 10, "gap": 5}
                            row.update({metric: 0. for metric in METRICS})
                            row.update(normalized_mse=mse, latent_mse=mse, state_accuracy=state_accuracy,
                                       state_position_mse=.01 if cond in {"recent_only", "persistence"} else .004,
                                       pair_accuracy=.5 if cond == "recent_only" else .9,
                                       pair_margin=-1. if cond == "wrong_evidence" else 1., target_pair_distance=.2,
                                       history_count=0)
                            raw.append(row)
    write_csv(root/"evaluation/raw_records.csv", raw)
    models, checkpoints = {}, {}
    for seed in c["training"]["seeds"]:
        signature = {"scientific_hash": scientific_hash(c), "cache_signature": "same-cache", "normalization_sha256": "same-normalization",
                     "basis_sha256": "same-basis", "code_hash": "same-training-code"}
        logpath = root/"predictor"/task/f"seed_{seed}"/"training.json"
        logs = [{"epoch": i, "validation_mse": 1/i} for i in range(1, 121)]
        write_json(logpath, {"logs": logs, "signature": signature})
        key = f"{task}/{seed}"
        models[key] = {"complete": True, "task": task, "seed": seed, "completed_epochs": 120, "final_step": 7680,
                       "selection_metric": "validation_mse", "best_epoch": 120, "best_sha256": f"best-{arm}-{seed}",
                       "training_json_sha256": file_hash(logpath), "signature": signature}
        checkpoints[key] = {"task": task, "seed": seed, "best_epoch": 120, "best_sha256": models[key]["best_sha256"]}
    metrics = {"mode": "confirm", "cache_signature": "same-cache", "raw_sha256": file_hash(root/"evaluation/raw_records.csv"),
               "evaluation_code_hash": code_hash("evaluation.py", "stats.py"), "checkpoints": checkpoints,
               "training_completion": {"complete": True, "required_epochs": 120, "models": models}, **summarize(c, raw)}
    write_json(root/"evaluation/metrics.json", metrics)
    return c, audit, probes, metrics


@pytest.fixture
def gate_arms(tmp_path):
    reference = tmp_path/"reference"
    candidate = tmp_path/"candidate"
    _arm(reference, "gate", "reference", privileged=.6)
    _arm(candidate, "gate", "gate_no_rank")
    return reference, candidate


def test_development_cannot_claim_phase1_go(gate_arms):
    _, root = gate_arms
    c = read_json(root/"resolved_config.json")
    audit, probes, metrics = [read_json(root/name) for name in ["dataset/split_audit.json", "probes/metrics.json", "evaluation/metrics.json"]]
    status, _, _ = decision(c, audit, probes, metrics)
    assert status == "DEVELOPMENT_PASS"
    result = development_comparison(*gate_arms, "gate")
    assert result["status"] == "SUPPORTED"
    assert all(r["passed"] for r in result["checks"])


def test_per_seed_failure_not_hidden_by_pooled_mean(gate_arms):
    reference, candidate = gate_arms
    _arm(candidate, "gate", "gate_no_rank", failed_seed=43)
    result = development_comparison(reference, candidate, "gate")
    assert result["status"] == "REJECTED"
    assert any("seed43" in reason for reason in result["reasons"])


def test_comparison_rejects_gain_from_weakened_recent_baseline(gate_arms):
    reference, candidate = gate_arms
    _arm(reference, "gate", "reference", privileged=.8, recent=.4)
    result = development_comparison(reference, candidate, "gate")
    assert result["status"] == "REJECTED"
    cross = [r for r in result["checks"] if r["kind"] == "candidate_vs_reference_recent"]
    assert cross and not any(r["passed"] for r in cross)


def test_comparison_rejects_incomplete_or_unmatched_records(gate_arms):
    reference, candidate = gate_arms
    metrics = read_json(candidate/"evaluation/metrics.json")
    metrics["training_completion"]["models"]["gate/43"]["completed_epochs"] = 119
    write_json(candidate/"evaluation/metrics.json", metrics)
    result = development_comparison(reference, candidate, "gate")
    assert result["status"] == "INCOMPLETE"
    assert "incomplete" in result["reasons"][0]


def test_single_change_and_checkpoint_selection_are_enforced(gate_arms):
    reference, candidate = gate_arms
    c = read_json(candidate/"resolved_config.json")
    c["model"]["hidden"] *= 2
    write_json(candidate/"resolved_config.json", c)
    result = development_comparison(reference, candidate, "gate")
    assert result["status"] == "INCOMPLETE"
    assert any("provenance" in r for r in result["reasons"])


def test_task_filter_excludes_unrelated_task_failure(gate_arms):
    _, root = gate_arms
    c = read_json(root/"resolved_config.json")
    audit, probes, metrics = [read_json(root/name) for name in ["dataset/split_audit.json", "probes/metrics.json", "evaluation/metrics.json"]]
    probes["results"].append({"task": "motion", "split": "test_id", "variant": "motion_two_frame",
                              "accuracy": {"mean": 0.}, "wilson_effective_pairs": [0., .1]})
    assert decision(c, audit, probes, metrics)[0] == "DEVELOPMENT_PASS"


def test_single_task_development_cannot_be_relabelled_confirmation(gate_arms):
    _, root = gate_arms
    c = read_json(root/"resolved_config.json")
    audit, probes, metrics = [read_json(root/name) for name in ["dataset/split_audit.json", "probes/metrics.json", "evaluation/metrics.json"]]
    c["experiment"]["stage"] = "confirmation"
    status, reasons, _ = decision(c, audit, probes, metrics)
    assert status == "INCONCLUSIVE"
    assert any("两个预定任务" in reason for reason in reasons)
    assert any("provenance" in reason for reason in reasons)


def test_motion_requires_exact_validation_precision_provenance(tmp_path):
    reference, candidate = tmp_path/"reference", tmp_path/"candidate"
    _arm(reference, "motion", "reference", privileged=.6)
    _arm(candidate, "motion", "motion_wide_span")
    assert development_comparison(reference, candidate, "motion")["status"] == "INCOMPLETE"
    diagnostic = {"passed": True, "scope": "validation_only", "cache_signature": "same-cache",
                  "input_basis_sha256": "same-basis", "normalization_sha256": "same-normalization", "comparisons": [
                      {"comparison": name, "split": "validation", "passed": True, "absolute_mse_reduction": {"low": .1}}
                      for name in ["wide_two_vs_current_two", "wide_span_two_vs_single_start", "wide_span_two_vs_single_end"]]}
    result = development_comparison(reference, candidate, "motion", diagnostic)
    assert result["status"] == "SUPPORTED", result["reasons"]
    diagnostic["comparisons"][-1]["absolute_mse_reduction"]["low"] = -.1
    assert development_comparison(reference, candidate, "motion", diagnostic)["status"] == "REJECTED"
    diagnostic["scope"] = "declared_development_splits"
    assert development_comparison(reference, candidate, "motion", diagnostic)["status"] == "INCOMPLETE"


def test_actual_motion_precision_emits_adoption_comparison_names(tmp_path):
    from phase1.cache import encode, fit_normalization
    from phase1.data import generate, validate_data
    from phase1.probes import motion_precision
    from phase1.utils import seed_all

    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path/"precision_schema")
    c["data"]["pairs"] = {s: 4 if s == "train" else 2 for s in SPLITS}
    c["evaluation"]["bootstrap"] = 100
    c["probe"]["ridge"] = [.1, 1.]
    seed_all(17)
    generate(c)
    validate_data(c)
    encode(c)
    fit_normalization(c)
    diagnostic = motion_precision(c, splits=["validation"])
    actual = {r["comparison"] for r in diagnostic["comparisons"]}
    required = {"wide_two_vs_current_two", "wide_span_two_vs_single_start", "wide_span_two_vs_single_end"}
    assert required.issubset(actual)
    assert diagnostic["scope"] == "validation_only"


def test_calibration_distinguishes_common_contrast_and_amplification():
    target = np.array([[1., 0.], [-1., 0.]])
    amplified = np.array([[2., 1.], [-2., 1.]])
    result = pair_calibration(amplified, target)
    assert result["alpha"] == 2.
    assert result["common_mse"] == .5
    assert result["contrast_mse"] == .5
    assert result["orthogonal_contrast_mse"] == 0.
    assert result["total_mse"] == 1.
    assert abs(result["decomposition_residual"]) < 1e-12
    assert pair_calibration(amplified[::-1], target[::-1]) == result


def test_calibration_handles_nonseparated_targets_without_nan():
    result = pair_calibration(np.ones((2, 3)), np.zeros((2, 3)))
    assert result["alpha"] is None and result["degenerate_target_contrast"]
    c = copy.deepcopy(DEFAULT)
    summary = summarize_calibration(c, [{"seed": 17, "task": "gate", "split": "validation", "condition": "recent_only",
                                         "horizon": 3, "pair": "p", **result}])
    assert summary["per_seed"][0]["metrics"]["alpha"] is None
    assert summary["per_seed"][0]["degenerate_target_pairs"] == 1
    with pytest.raises(ValueError, match="complete pair"):
        pair_calibration(np.ones((1, 3)), np.zeros((1, 3)))
