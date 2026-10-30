"""Streaming/harness fixtures, not real JEPA or trained-policy results."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase2 import run_retention_baseline_pilot as pilot
from experiments.phase2.training_samples import EpisodeIdentity


class FixtureScores:
    def __init__(self, evaluator=None):
        self.contexts = []  # Test instrumentation; not used by the online object.
        self.delayed = []

    def attention(self, candidates, context):
        return np.array([float(e.representation @ context.recent[-1].representation) for e in candidates], np.float32)

    def prepare_context(self, recent, history, past_actions, current_time):
        assert set(past_actions) == {e.entry_id for e in recent + history}
        bias = np.mean([e.representation for e in recent + history], axis=0)
        value = SimpleNamespace(origin_time=current_time, numeric_bytes=bias.nbytes + 8, bias=bias.copy())
        self.contexts.append(value)
        return value

    def predict_context(self, prepared, current_representation, actions):
        assert actions.shape == (3, 2)
        result = (current_representation + prepared.bias + np.float32(actions.sum())).astype(np.float32)
        self.delayed.append((prepared.origin_time, actions.copy(), result.copy()))
        return result

    def predict(self, recent, history, past_actions, action_prefix, horizon):
        assert len(history) == 2 and action_prefix.shape == (horizon, 2)
        return (recent[-1].representation + np.mean([e.representation for e in history], axis=0)
                + np.float32(action_prefix.sum())).astype(np.float32)

    def audit(self):
        return {"fixture_only": True}


def arrays(length=22):
    rng = np.random.default_rng(29)
    return rng.normal(size=(length, 8)).astype(np.float32), rng.uniform(-.1, .1, size=(length - 1, 2)).astype(np.float32)


def run_prefix(z, a, policy):
    scores = FixtureScores()
    online = pilot.OnlineRetention("ep", policy, scores, seed=17)
    events = []
    for t, feature in enumerate(z):
        _, event = online.observe(feature, executed_action=None if t == 0 else a[t - 1])
        events.append(event)
    return online, events, scores


@pytest.mark.parametrize("policy", pilot.POLICIES)
def test_six_policies_stream_full_episode_and_share_same_fixed_prediction_contract(policy):
    z, a = arrays()
    identity = EpisodeIdentity("ep", "pair", "validation", "gate", 1)
    trace = []
    row, prediction = pilot.replay_episode(identity, z, a, 12, policy, FixtureScores(), base_seed=17, trace=trace.append)
    assert row["frames_replayed"] == len(z) and len(trace) == len(z)
    assert len(row["retained_timestamps"]) == 2 and prediction.shape == (2, 8)
    assert np.isfinite(row["feature_mse"]).all()
    assert row["logical_memory_maxima"]["resident_observations"] == 6
    assert row["logical_memory_maxima"]["birth_prefix_entries"] == 6
    assert row["logical_memory_maxima"]["transient_birth_prefix_entries"] == 7
    assert row["logical_memory_maxima"]["pending_contexts"] == (3 if policy == "prediction_surprise" else 0)


def test_surprise_uses_realized_three_actions_and_current_observation_with_bounded_pending_state():
    z, a = arrays()
    online, events, scores = run_prefix(z, a, "prediction_surprise")
    assert all(e["current_surprise"] is None for e in events[:6])
    for origin, actions, prediction in scores.delayed:
        assert np.array_equal(actions, a[origin:origin + 3])
        expected = np.mean((prediction.astype(np.float64) - z[origin + 3].astype(np.float64)) ** 2)
        assert events[origin + 3]["current_surprise"] == expected
    assert online.surprise_predictions == len(z) - 6 and len(online.pending) == 3
    assert all(p.context.origin_time >= len(z) - 3 for p in online.pending)
    assert set(online.births) == {e.entry_id for e in online.stream.snapshot.recent + online.stream.snapshot.retained}
    assert online.maxima["pending_context_numeric_bytes"] == 3 * 8 * 4
    assert online.maxima["pending_origin_time_bytes"] == 3 * 8
    assert online.maxima["transient_action_prefix_numeric_bytes"] == 8 * 8


@pytest.mark.parametrize("policy", ("attention", "prediction_surprise"))
def test_future_intervention_cannot_change_retention_prefix(policy):
    z, a = arrays()
    _, before, _ = run_prefix(z, a, policy)
    changed_z, changed_a = z.copy(), a.copy()
    changed_z[13:] += 100
    changed_a[12:] += 100
    _, after, _ = run_prefix(changed_z, changed_a, policy)
    assert before[:13] == after[:13]


def test_new_episode_does_not_inherit_pending_forecasts_or_memory():
    z, a = arrays()
    first, _, _ = run_prefix(z, a, "prediction_surprise")
    second = pilot.OnlineRetention("second", "prediction_surprise", FixtureScores(), seed=17)
    snapshot, event = second.observe(z[0])
    assert len(first.pending) == 3 and not second.pending and not snapshot.retained
    assert event["current_surprise"] is None and set(second.births) == {("second", 0)}


def test_aggregation_uses_independent_pairs_not_forks_or_sibling_count():
    rows = [{"task": "gate", "policy": "fifo", "group_id": str(pair), "feature_mse": loss,
             "online_seconds": 1., "logical_memory_maxima": {"total_logical_bytes": 100}}
            for pair, loss in [(0, [1, 3]), (0, [3, 5]), (1, [6, 8]), (1, [8, 10])]]
    cell = pilot.aggregate(rows)[0]
    assert cell["independent_pairs"] == 2 and cell["episodes"] == 4
    assert cell["mean_pair_feature_mse"] == [4.5, 6.5]


def test_worker_reads_only_validation_and_preserves_negative_training_context(tmp_path, monkeypatch):
    import torch
    from experiments.phase2 import prepare_controlled_data, train_predictor_pilot, frozen_predictor, causal_model_scores
    from phase1 import cache as cache_module
    data = tmp_path / "fixture_pixels_NOT_JEPA"
    prepare_controlled_data.run(data, seed=20261101, train_pairs=2, eval_pairs=2)
    config = json.loads((data / "resolved_config.json").read_text())
    config.update(mode="smoke", threads=1)
    config["encoder"].update(backend="smoke_pixels", smoke_size=2)
    cache = cache_module.encode(config)
    cache_module.fit_normalization(config)
    index = json.loads((data / "observation_index.json").read_text())
    roles = json.loads((data / "split_roles.json").read_text())["roles"]
    monkeypatch.setattr(train_predictor_pilot, "verify_inputs", lambda *a: (config, index, roles, cache))
    original = cache_module.load_feature
    loaded = []
    def guarded(c, rec, ix):
        assert rec["split"] == "validation"
        loaded.append(rec["id"])
        return original(c, rec, ix)
    monkeypatch.setattr(cache_module, "load_feature", guarded)
    class FixtureEvaluator:
        provenance = SimpleNamespace(frozen_predictor_sha256="a" * 64)
        _model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(frozen_predictor, "FrozenPredictorEvaluator", lambda *a: FixtureEvaluator())
    monkeypatch.setattr(causal_model_scores, "CausalModelScores", FixtureScores)
    training = tmp_path / "fixture_training"
    training.mkdir()
    (training / "training_result.json").write_text(json.dumps({"tasks": [{"task": t, "seed": 17, "best_validation_mse": .21} for t in ("gate", "motion")]}))
    for task in ("gate", "motion"):
        path = training / "predictor" / task / "seed_17"
        path.mkdir(parents=True)
        (path / "training.json").write_text(json.dumps({"initial_validation_by_mode": {"fixture": .2}}))
    output = tmp_path / "baseline_fixture"
    output.mkdir()
    pilot.worker(SimpleNamespace(data_root=data, encoded_report=tmp_path, predictor_root=training,
                                output=output, seed=17, threads=1))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["episode_policy_rows"] == 48
    assert len(loaded) == len(set(loaded)) == 8 and len(summary["aggregate"]) == 12
    assert all(r["relative_change_percent"] > 0 for r in summary["training_quality_context"])
    assert not summary["equal_total_memory_bytes"] and not summary["untouched_test_claim"]
    assert not summary["scientific_phase2_completed"] and summary["test_feature_arrays_loaded"] == 0
