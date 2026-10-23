"""Bounded parent and writer tests; vectors/callback are not JEPA/model results."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase2 import generate_deletion_pilot as pilot
from experiments.phase2.deletion import EvaluatorProvenance
from experiments.phase2.memory import MemoryEntry, StreamingMemory
from experiments.phase2.training_samples import EpisodeIdentity, SplitRegistry


def arrays(length=20):
    rng = np.random.default_rng(17)
    return rng.normal(size=(length, 8)).astype(np.float32), rng.normal(size=(length - 1, 2)).astype(np.float32)


@pytest.mark.parametrize("policy", pilot.PARENT_POLICIES)
def test_parent_is_real_persistent_overflow_and_actions_match_original_prefix(policy):
    z, a = arrays()
    got = pilot.capture_query_parent("ep", z, a, 12, policy=policy, seed=17)
    stream = StreamingMemory(4, 2, policy, seed=17)
    for t in range(12):
        stream.push(MemoryEntry("ep", t, z[t]))
    before = stream.snapshot
    assert tuple(e.entry_id for e in got.parent.candidates) == tuple(e.entry_id for e in before.retained + (before.recent[0],))
    assert [e.timestamp for e in got.parent.context.recent] == [9, 10, 11, 12]
    prefix = np.r_[np.zeros((1, 2), np.float32), np.cumsum(a[:12], axis=0)]
    for entry in got.parent.candidates + got.parent.context.recent:
        assert np.array_equal(got.past_actions[entry.entry_id], prefix[12] - prefix[entry.timestamp])
        assert not got.past_actions[entry.entry_id].flags.writeable
    assert got.accounting["persistent_action_prefix_entries_max"] == 6
    assert got.accounting["transient_action_prefix_entries_max"] == 7


def test_parent_cannot_observe_future_features_or_action_suffix():
    z, a = arrays()
    first = pilot.capture_query_parent("ep", z, a, 12, policy="reservoir", seed=51)
    z[13:] = np.nan
    a[12:] = np.nan
    second = pilot.capture_query_parent("ep", z, a, 12, policy="reservoir", seed=51)
    for x, y in zip(first.parent.candidates, second.parent.candidates):
        assert x.entry_id == y.entry_id and np.array_equal(x.representation, y.representation)
        assert np.array_equal(first.past_actions[x.entry_id], second.past_actions[y.entry_id])


def test_first_full_overflow_and_warmup_rejection():
    z, a = arrays()
    capture = pilot.capture_query_parent("ep", z, a, 6, policy="fifo", seed=1)
    assert [e.timestamp for e in capture.parent.candidates] == [0, 1, 2]
    assert [e.timestamp for e in capture.parent.context.recent] == [3, 4, 5, 6]
    with pytest.raises(ValueError, match="full R4/K2 overflow"):
        pilot.capture_query_parent("ep", z, a, 5, policy="fifo", seed=1)


class FixtureEvaluator:
    provenance = EvaluatorProvenance("a" * 64, "fixture_NOT_JEPA_NOT_TRAINED")

    def __init__(self):
        self.calls = []

    def bind(self, parent, past_action_summaries):
        allowed = {e.entry_id for e in parent.candidates + parent.context.recent}
        assert set(past_action_summaries) == allowed
        def predict(kept, context, horizon, action_prefix):
            assert len(kept) == 2 and action_prefix.shape == (horizon, 2)
            self.calls.append((tuple(e.entry_id for e in kept), horizon, action_prefix.copy()))
            return (context.recent[-1].representation + np.mean([e.representation for e in kept], axis=0)
                    + np.float32(action_prefix.sum())).astype(np.float32)
        return predict

    def audit(self):
        return {"fixture_only": True}


def test_partial_horizon_keeps_mask_and_enumerates_exact_equal_k():
    z, a = arrays(length=17)
    capture = pilot.capture_query_parent("ep", z, a, 12, policy="fifo", seed=17)
    evaluator = FixtureEvaluator()
    result = pilot.label_parent(capture, z, a, evaluator)
    assert result.labels.horizon_mask.tolist() == [True, False]
    assert np.isfinite(result.labels.feature_mse[:, 0]).all()
    assert np.isnan(result.labels.feature_mse[:, 1]).all()
    assert len(evaluator.calls) == 3 and all(c[1] == 3 for c in evaluator.calls)
    assert len(set(tuple(ids) for ids in result.labels.kept_ids)) == 3


def test_future_target_intervention_changes_only_labels():
    z, a = arrays()
    capture = pilot.capture_query_parent("ep", z, a, 12, policy="visual_diversity", seed=17)
    first = pilot.label_parent(capture, z, a, FixtureEvaluator())
    z[15] += 100
    second = pilot.label_parent(capture, z, a, FixtureEvaluator())
    assert np.array_equal(first.scorer_inputs.candidate_representations, second.scorer_inputs.candidate_representations)
    assert np.array_equal(first.scorer_inputs.recent_representations, second.scorer_inputs.recent_representations)
    assert not np.array_equal(first.labels.feature_mse[:, 0], second.labels.feature_mse[:, 0])
    assert np.array_equal(first.labels.feature_mse[:, 1], second.labels.feature_mse[:, 1])


def test_train_validation_shards_exclude_tests_and_separate_future_labels(tmp_path):
    records = []
    for number, split in enumerate(("train", "validation", "test_id")):
        for sibling in range(2):
            records.append({"id": f"{split}-{sibling}", "pair": split, "split": split,
                            "task": "gate", "seed": number, "query": 12})
    registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in records])
    loaded = []
    def load(rec):
        assert not rec["split"].startswith("test")
        loaded.append(rec["id"])
        return arrays()
    result = pilot.generate_task(records, registry, load, FixtureEvaluator(), "gate", tmp_path, base_seed=17)
    assert result["parent_policy_records"] == 16
    assert 4 <= result["unique_parents"] <= 16 and result["recomputed_parents"] >= 2
    assert len(loaded) == 4 and len(result["shards"]) == 2
    all_groups = {}
    for shard in result["shards"]:
        split = shard["split"]
        with np.load(tmp_path / "scorer_inputs" / f"gate_{split}.npz", allow_pickle=False) as inputs:
            assert "feature_mse" not in inputs and "targets" not in inputs and "actions" not in inputs
            assert inputs["candidate_representations"].shape[1:] == (3, 8)
            assert inputs["recent_representations"].shape[1:] == (4, 8)
            ids = inputs["parent_id"].copy()
        with np.load(tmp_path / "labels" / f"gate_{split}.npz", allow_pickle=False) as labels:
            assert np.array_equal(ids, labels["parent_id"])
            assert labels["feature_mse"].shape[1:] == (3, 2)
            assert labels["kept_timestamps"].shape[1:] == (3, 2)
        metadata = json.loads((tmp_path / "metadata" / f"gate_{split}.json").read_text())
        assert sum(len(row["parent_origins"]) for row in metadata) == 8
        all_groups[split] = {row["group_id"] for row in metadata}
    assert all_groups["train"].isdisjoint(all_groups["validation"])
    with pytest.raises(FileExistsError):
        pilot.generate_task(records, registry, load, FixtureEvaluator(), "gate", tmp_path, base_seed=17)


def test_parent_rng_shared_across_siblings_not_query_or_future():
    a = EpisodeIdentity("a", "pair", "train", "gate", 1)
    b = EpisodeIdentity("b", "pair", "train", "gate", 1)
    assert pilot.policy_seed(17, a, "reservoir") == pilot.policy_seed(17, b, "reservoir")
    assert pilot.policy_seed(17, a, "fifo") != pilot.policy_seed(17, a, "reservoir")


def test_worker_reads_real_fixture_cache_and_writes_complete_separate_shards(tmp_path, monkeypatch):
    # Actual generated data/cache integration; explicit pixel and callback
    # substitutions are software fixtures, not official encoder/model results.
    from experiments.phase2 import prepare_controlled_data, train_predictor_pilot, frozen_predictor
    from phase1.cache import encode, fit_normalization
    data = tmp_path / "pixels_NOT_JEPA"
    prepare_controlled_data.run(data, seed=20261031, train_pairs=2, eval_pairs=2)
    config = json.loads((data / "resolved_config.json").read_text())
    config.update(mode="smoke", threads=1)
    config["encoder"].update(backend="smoke_pixels", smoke_size=2)
    cache = encode(config)
    fit_normalization(config)
    index = json.loads((data / "observation_index.json").read_text())
    roles = json.loads((data / "split_roles.json").read_text())["roles"]
    report = tmp_path / "fixture-report"
    report.mkdir()
    (report / "manifest.json").write_text('{"fixture":"NOT_JEPA"}')
    monkeypatch.setattr(train_predictor_pilot, "verify_inputs", lambda *a: (config, index, roles, cache))
    monkeypatch.setattr(frozen_predictor, "FrozenPredictorEvaluator", lambda **kw: FixtureEvaluator())
    output = tmp_path / "labels-fixture"
    output.mkdir()
    pilot.worker(SimpleNamespace(data_root=data, encoded_report=report, predictor_root=tmp_path,
                                output=output, threads=1, seed=17))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "new_development_equal_k_deletion_pilot_completed"
    assert summary["parent_policy_records"] == 64
    assert summary["test_feature_arrays_loaded"] == 0
    assert len(summary["loaded_episode_ids"]) == 16
    assert sum(len(t["shards"]) for t in summary["tasks"]) == 4
    assert not summary["six_retention_baseline_results"] and not summary["scientific_phase2_completed"]
