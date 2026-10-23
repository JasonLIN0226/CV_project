"""Causal sampler software fixtures; no encoder, real training or model results."""
from dataclasses import replace

import numpy as np
import pytest

from experiments.phase2.memory import MemoryEntry, StreamingMemory
from experiments.phase2.training_samples import (
    DEFAULT_MODES, EpisodeIdentity, FeatureEpisode, SamplingConfig, SplitRegistry,
    SuppliedHistory, phase1_arrays, sample_predictor,
)


def identity(episode="e", group="pair", split="train", seed=10, task="gate", dynamics=None):
    return EpisodeIdentity(episode, group, split, task, seed, dynamics)


def episode(ident=None, *, semantics="additive_controls", length=24, features=None, actions=None):
    return FeatureEpisode(ident or identity(),
                          np.arange(length * 8, dtype=np.float32).reshape(length, 8) / 10 if features is None else features,
                          np.arange((length - 1) * 2, dtype=np.float32).reshape(length - 1, 2) / 100 if actions is None else actions,
                          semantics, "software-fixture-no-encoder")


def sample(ep=None, config=None, **kwargs):
    ep = ep or episode()
    return sample_predictor(ep, SplitRegistry([ep.identity]), config or SamplingConfig(),
                            query=kwargs.pop("query", 12), base_seed=kwargs.pop("base_seed", 17), **kwargs)


def test_original_record_adapter_reads_identity_only():
    record = dict(id="a", pair="p", split="train", task="gate", seed=10,
                  states=object(), evidence=object(), eval=object(), gap=object())
    assert EpisodeIdentity.from_record(record) == identity("a", "p")


@pytest.mark.parametrize("records,message", [
    ([identity(), identity()], "duplicate episode"),
    ([identity(), identity("b", "pair", "validation", 11)], "group"),
    ([identity(), identity("b", "other", "validation", 10)], "environment seed"),
])
def test_whole_episode_group_and_environment_seed_cannot_cross_splits(records, message):
    with pytest.raises(ValueError, match=message):
        SplitRegistry(records)


def test_registry_preserves_siblings_and_declared_dynamics_holdout_roles():
    records = [identity("a"), identity("b"), identity("c", "other", "test_long_gap", 20, dynamics="long")]
    registry = SplitRegistry(records, dynamics_split={"long": "test_long_gap"})
    assert registry.roles["test_long_gap"] == "development_evaluation"
    assert registry.roles["train"] == "training"
    with pytest.raises(ValueError, match="dynamics"):
        SplitRegistry([identity(dynamics="long")], dynamics_split={"long": "test_long_gap"})
    with pytest.raises(TypeError):
        registry.records["new"] = records[0]


def test_unknown_identity_and_wrong_split_refused_before_sampling():
    ep = episode(identity(split="validation"))
    registry = SplitRegistry([ep.identity])
    with pytest.raises(ValueError, match="requires split"):
        sample_predictor(ep, registry, SamplingConfig(), query=12, base_seed=17)
    unknown = episode(identity("unknown"))
    with pytest.raises(ValueError, match="missing from"):
        sample_predictor(unknown, registry, SamplingConfig(), query=12, base_seed=17)


@pytest.mark.parametrize("mode", DEFAULT_MODES + ("recent_only",))
def test_each_available_mode_is_causal_bounded_and_needs_no_model(mode):
    result = sample(config=SamplingConfig(memory_dropout=0), history_mode=mode)
    assert [e.timestamp for e in result.causal.recent] == [9, 10, 11, 12]
    assert len(result.causal.history) <= 2
    assert all(e.timestamp < 9 for e in result.causal.history)
    assert len({e.entry_id for e in result.causal.history}) == len(result.causal.history)
    assert result.causal.executed_actions.shape == (12, 2)
    assert result.future.actions.shape == (6, 2)
    assert result.metadata["mode"] == mode
    assert result.metadata["history_source"] != "supplied_streaming_snapshot_requires_external_policy_audit"


def test_random_subset_varies_cardinality_and_fixed_subset_varies_membership():
    cardinalities, subsets, modes = set(), set(), set()
    config = SamplingConfig(memory_dropout=0)
    for epoch in range(100):
        cardinalities.add(len(sample(config=config, history_mode="random_subset", epoch=epoch).causal.history))
        subsets.add(sample(config=config, history_mode="random_fixed", epoch=epoch).metadata["history_before_dropout"])
        modes.add(sample(config=config, epoch=epoch).metadata["mode"])
    assert cardinalities == {0, 1, 2}
    assert len(subsets) > 10
    assert modes == set(DEFAULT_MODES)


def test_dropout_is_explicit_independent_and_preserves_context_targets():
    preserved = sample(config=SamplingConfig(memory_dropout=0), history_mode="random_fixed")
    dropped = sample(config=SamplingConfig(memory_dropout=1), history_mode="random_fixed")
    assert len(preserved.causal.history) == 2 and dropped.causal.history == ()
    assert dropped.metadata["dropout_event"] and dropped.metadata["dropout_removed_entries"] == 2
    assert preserved.metadata["history_before_dropout"] == dropped.metadata["history_before_dropout"]
    np.testing.assert_array_equal(preserved.future.targets, dropped.future.targets)
    np.testing.assert_array_equal(preserved.future.actions, dropped.future.actions)
    np.testing.assert_array_equal(preserved.causal.executed_actions, dropped.causal.executed_actions)
    for left, right in zip(preserved.causal.recent, dropped.causal.recent):
        np.testing.assert_array_equal(left.representation, right.representation)
    outcomes = [sample(config=SamplingConfig(memory_dropout=.25), history_mode="random_fixed", draw=i).metadata["dropout_event"] for i in range(1200)]
    assert abs(np.mean(outcomes) - .25) < .04


def test_group_rng_reproducibility_and_sibling_coupling():
    first = sample(history_mode="random_fixed", epoch=2, draw=3)
    repeat = sample(history_mode="random_fixed", epoch=2, draw=3)
    sibling = sample(episode(identity("sibling")), history_mode="random_fixed", epoch=2, draw=3)
    for other in (repeat, sibling):
        assert first.metadata["history_before_dropout"] == other.metadata["history_before_dropout"]
        assert first.metadata["dropout_event"] == other.metadata["dropout_event"]


def test_future_intervention_does_not_change_selection_or_causal_inputs():
    original = episode()
    changed_features, changed_actions = original.features.copy(), original.actions.copy()
    changed_features[13:] = -123.
    changed_actions[12:] = 42.
    changed = episode(features=changed_features, actions=changed_actions)
    for mode in DEFAULT_MODES:
        a = sample(original, config=SamplingConfig(memory_dropout=0), history_mode=mode)
        b = sample(changed, config=SamplingConfig(memory_dropout=0), history_mode=mode)
        assert a.metadata["history_after_dropout"] == b.metadata["history_after_dropout"]
        for left, right in zip(a.causal.recent + a.causal.history, b.causal.recent + b.causal.history):
            np.testing.assert_array_equal(left.representation, right.representation)
        np.testing.assert_array_equal(a.causal.executed_actions, b.causal.executed_actions)
        assert not np.array_equal(a.future.targets, b.future.targets)
        assert not np.array_equal(a.future.actions, b.future.actions)
        assert not hasattr(a.causal, "targets") and not hasattr(a.causal, "future_actions")


def test_causal_action_storage_cannot_reach_future_suffix_and_inputs_are_immutable():
    features = episode().features.copy()
    actions = episode().actions.copy()
    ep = episode(features=features, actions=actions)
    features[:] = -1
    actions[:] = -1
    result = sample(ep, history_mode="random_fixed")
    assert result.causal.executed_actions.base.nbytes == result.causal.executed_actions.nbytes
    assert result.future.actions.base.nbytes == result.future.actions.nbytes
    for array in (ep.features, ep.actions, result.causal.executed_actions, result.future.targets):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.metadata["split"] = "test"


def test_validation_is_fixed_and_dropout_disabled():
    ep = episode(identity(split="validation"))
    result = sample(ep, config=SamplingConfig(memory_dropout=1), expected_split="validation", history_mode="random_fixed")
    assert len(result.causal.history) == 2
    assert result.metadata["dropout_probability"] == 0 and not result.metadata["dropout_event"]
    with pytest.raises(ValueError, match="fixed epoch"):
        sample(ep, expected_split="validation", epoch=1)


def test_warmup_and_incomplete_horizons_are_explicit():
    early = sample(query=3, history_mode="random_fixed")
    assert early.causal.history == ()
    near_end = sample(query=19, history_mode="fifo")
    np.testing.assert_array_equal(near_end.future.available, [True, False])
    assert np.isnan(near_end.future.targets[1]).all()
    with pytest.raises(ValueError, match="missing target"):
        phase1_arrays(near_end)
    with pytest.raises(ValueError, match="no available"):
        sample(query=23, history_mode="fifo")
    with pytest.raises(ValueError, match="complete recent"):
        sample(query=2)


def test_model_dependent_modes_require_supplied_snapshot_but_other_modes_work():
    for mode in ("attention", "prediction_surprise"):
        with pytest.raises(ValueError, match="supplied audited"):
            sample(history_mode=mode)
    ep = episode()
    # Explicit fixture callback only, not actual predictor attention.
    stream = StreamingMemory(4, 2, "attention", score_callback=lambda candidates, context: [e.timestamp for e in candidates])
    for t in range(13):
        snapshot = stream.push(MemoryEntry(ep.identity.episode_id, t, ep.features[t]))
    supplied = SuppliedHistory("attention", snapshot)
    result = sample(ep, config=SamplingConfig(memory_dropout=0), history_mode="attention", supplied_history=supplied)
    assert [e.timestamp for e in result.causal.history] == [7, 8]
    assert result.metadata["history_source"].endswith("requires_external_policy_audit")
    with pytest.raises(ValueError, match="policy must match"):
        sample(ep, history_mode="fifo", supplied_history=supplied)
    forged = replace(snapshot, retained=(MemoryEntry("e", 7, np.zeros(8, np.float32)), snapshot.retained[-1]))
    with pytest.raises(ValueError, match="differs from"):
        sample(ep, history_mode="attention", supplied_history=SuppliedHistory("attention", forged))


def test_legacy_bridge_matches_additive_alignment_and_separates_targets():
    ep = episode()
    result = sample(ep, config=SamplingConfig(memory_dropout=0), history_mode="fifo")
    arrays, target = phase1_arrays(result)
    assert set(arrays) == {"z", "age", "type", "past_actions", "actions", "current"}
    times = np.array([9, 10, 11, 12, 7, 8])
    np.testing.assert_array_equal(arrays["z"], ep.features[times])
    np.testing.assert_array_equal(arrays["age"], 12 - times)
    np.testing.assert_array_equal(arrays["type"], [0, 0, 0, 0, 1, 1])
    for i, t in enumerate(times):
        np.testing.assert_allclose(arrays["past_actions"][i], ep.actions[t:12].sum(0), atol=1e-6)
    np.testing.assert_array_equal(arrays["actions"], ep.actions[12:18])
    np.testing.assert_array_equal(target, ep.features[[15, 18]])
    with pytest.raises(ValueError, match="additive_controls"):
        phase1_arrays(sample(episode(semantics="absolute_targets"), history_mode="fifo"))


def test_legacy_collate_and_predictor_shape_with_tiny_untrained_cpu_fixture():
    torch = pytest.importorskip("torch")
    from phase1.inputs import collate
    from phase1.model import DynamicsPredictor
    from phase1.centered_transport import CenteredTransportPredictor
    selected = [sample(config=SamplingConfig(memory_dropout=p), history_mode="random_fixed") for p in (0., 1.)]
    dictionaries, targets = zip(*(phase1_arrays(s) for s in selected))
    batch = collate(dictionaries, "cpu")
    assert batch["mask"].sum(1).tolist() == [6, 4]
    basis = {"input_mean": np.zeros(8, np.float32), "projection": np.eye(8, dtype=np.float32)[:, :4],
             "target_mean": np.zeros(8, np.float32), "output_basis": np.eye(8, dtype=np.float32)[:, :4]}
    torch.manual_seed(0)
    reference = DynamicsPredictor(8, {"hidden": 16, "dropout": 0.}, (3, 6), basis)
    predictor = CenteredTransportPredictor(reference, basis, hidden=16).eval()
    with torch.inference_mode():
        prediction = predictor(batch)
    assert prediction.shape == (2, 2, 8) and torch.isfinite(prediction).all()
    assert np.stack(targets).shape == (2, 2, 8)


@pytest.mark.parametrize("change", [{"memory_dropout": -0.1}, {"memory_dropout": np.nan},
                                    {"history_modes": ("fifo", "fifo")}, {"capacity": 0},
                                    {"horizons": (6, 3)}])
def test_invalid_sampling_config_is_rejected(change):
    with pytest.raises(ValueError):
        SamplingConfig(**change)


def test_episode_alignment_and_nonfinite_arrays_rejected():
    with pytest.raises(ValueError, match="T-1"):
        episode(actions=np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="finite"):
        episode(features=np.full((24, 8), np.nan, np.float32))
