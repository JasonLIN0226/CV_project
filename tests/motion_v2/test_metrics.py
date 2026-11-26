"""Motion v2 table metrics, the variant choice and the rule that guards the validation look."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bounded_memory.evaluation.metrics import motion_v2 as metrics

ROOT = Path(__file__).resolve().parents[2]


def _script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_pair_losses_and_paired_difference():
    episodes = [{"rec": {"id": f"e{i}", "pair": f"p{i // 2}"}} for i in range(4)]
    tables = {f"e{i}": np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4) + i for i in range(4)}
    got = metrics.final_pair_losses(tables, episodes, [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert got.shape == (4, 2) and got[1, 0] == tables["e1"][0, 1, 2] and got[3, 1] == tables["e3"][1, 0, 3]
    baseline = np.ones((4, 2))
    policy = baseline.copy()
    policy[:2] -= 0.5                                  # lower on scene pair p0 only
    p = metrics.paired(policy, baseline, episodes, resamples=500)
    assert p["pairs"] == 2 and np.isclose(p["difference"], -0.25) and p["share_of_pairs_lower"] == 0.5
    assert p["ci95"][0] <= -0.25 <= p["ci95"][1]
    rounding = metrics.paired(baseline - 1e-19, baseline, episodes, resamples=50)   # the same choices, float noise
    assert rounding["share_of_pairs_lower"] == 0.0


def test_variant_choice_is_the_lowest_mean_dev_over_three_seeds(tmp_path):
    choose = _script("scripts/aggregate_results/motion_v2_scorer_variants.py")
    for name, dev in (("min", [0.010, 0.010, 0.010]), ("min_dagger", [0.009, 0.0095, 0.0105])):
        for seed, value in zip((17, 29, 43), dev):
            history = [{"selected_step": 0, "final_dev_mean": 0.012}]
            (tmp_path / f"{name}_seed{seed}.json").write_text(json.dumps(
                {"best_dev_mean": value, "best_round": 0, "history": history, "dev_attention": {"mean": 0.0104}}))
    summary = choose.summarize(tmp_path)
    assert summary["chosen"] == "min_dagger" and set(summary["variants"]) == {"min", "min_dagger"}
    (tmp_path / "soft_seed17.json").write_text((tmp_path / "min_seed17.json").read_text())
    with pytest.raises(SystemExit, match="lacks seeds"):
        choose.summarize(tmp_path)


def test_validation_is_read_only_after_a_holdout_gain():
    evaluate = _script("scripts/evaluate/motion_v2_deletion_cost_v2.py")
    holdout = {"variants": {"a": {"paired_with_attention": {"ci95": [-0.002, -0.0001]}},
                            "b": {"paired_with_attention": {"ci95": [-0.002, 0.0003]}}}}
    assert evaluate.validation_allowed(holdout, "a")
    assert not evaluate.validation_allowed(holdout, "b") and not evaluate.validation_allowed(holdout, "c")
