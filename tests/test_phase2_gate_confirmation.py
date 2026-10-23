"""Frozen comparison rule, split boundary, causal replay and support-mask tests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase2.diagnose_oracle_learnability import RidgeProbe, ProbeRetention
from experiments.phase2.gate_confirmation_data import validate_confirmation_records, SCOPE
from experiments.phase2.run_gate_confirmation import (
    PRIMARY_ARM, PRIMARY_CONTROLS, primary_decision, evidence_summary, load_probes, replay,
)


def test_primary_claim_requires_all_three_frozen_controls():
    rows = {"minus_"+name: {"mean": -.1, "low": -.2, "high": -.01} for name in PRIMARY_CONTROLS}
    stats = {"online": {PRIMARY_ARM: {"all": rows}}}
    assert primary_decision(stats)["all_three_upper_95pct_bounds_below_zero"]
    rows["minus_attention"]["high"] = 0
    assert not primary_decision(stats)["all_three_upper_95pct_bounds_below_zero"]


def test_mask_only_uses_existing_train_scale_and_changes_no_original(tmp_path):
    scale = np.r_[np.full(710, 1e-6), np.ones(193)]
    raw = {name: {"scale": scale.tolist(), "weights": np.ones(903).tolist(), "label_scale": 1.}
           for name in ("cost_ridge", "attention_ridge", "shuffled_cost_ridge")}
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps(raw))
    probes = load_probes(path)
    for name in raw:
        np.testing.assert_array_equal(probes["original_"+name].weights, np.ones(903))
        np.testing.assert_array_equal(probes["masked_"+name].weights, np.r_[np.zeros(710), np.ones(193)])
        assert probes["original_"+name].bytes == probes["masked_"+name].bytes == 14456


def test_online_replay_cannot_see_future_observations_or_actions():
    rng = np.random.default_rng(31)
    z = rng.normal(size=(15, 3)).astype(np.float32)
    actions = rng.normal(size=(14, 2)).astype(np.float32)
    scores = SimpleNamespace(_basis={"input_mean": np.zeros(3, np.float32), "projection": np.eye(3, dtype=np.float32)},
                             attention=lambda *args: (_ for _ in ()).throw(AssertionError("not the probe")))
    probe = RidgeProbe(np.ones(28), rng.normal(size=28), 1., "test")
    original = replay(ProbeRetention("episode", scores, probe), "episode", z, actions, 8)
    future_z, future_actions = z.copy(), actions.copy()
    future_z[9:] += 1000; future_actions[8:] -= 1000
    changed = replay(ProbeRetention("episode", scores, probe), "episode", future_z, future_actions, 8)
    assert original[2] == changed[2]


def test_first_information_loss_waits_until_all_equivalent_copies_gone():
    events = [{"time": 3, "retained_timestamps": [2], "recent_timestamps": [3]},
              {"time": 4, "retained_timestamps": [], "recent_timestamps": [3, 4]}]
    result = evidence_summary(events, [], [[1, 2]])
    assert result["first_evidence_information_loss"] == [4]
    assert result["cue_survival"] == 0.


def test_confirmation_rejects_old_split_before_any_arrays():
    with pytest.raises(ValueError, match="literature_confirmation"):
        validate_confirmation_records([{"split": "fresh_development"}], [])
    assert SCOPE == "literature_confirmation"


def test_confirmation_rejects_any_prior_scene_seed_or_identity():
    records = []
    for condition in ("id", "long_gap", "distractor_shift"):
        for index in range(32):
            group = len(records)//2
            ids = [f"{len(records)+j:024x}" for j in (0, 1)]
            for label in (0, 1):
                records.append({"id": ids[label], "sibling": ids[1-label], "pair": f"{group:024x}",
                    "split": SCOPE, "task": "gate", "condition": condition,
                    "generation_condition": "test_"+condition, "seed": 1000+group,
                    "scene_index": index, "query": 9, "frames": 16, "rgb_indices": [0]*16,
                    "feature_path": f"features/{ids[label]}.npz", "eval_path": f"evaluation_only/{ids[label]}.npz"})
    assert len(validate_confirmation_records(records, [])) == 192
    prior = [{"id": "other", "pair": "other", "seed": records[0]["seed"]}]
    with pytest.raises(ValueError, match="overlaps prior scenes: seed"):
        validate_confirmation_records(records, prior)
