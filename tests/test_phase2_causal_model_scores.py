"""Numerical causal-score checks on the synthetic NOT_JEPA frozen fixture."""
from dataclasses import fields

import numpy as np
import pytest
import torch

from test_phase2_frozen_predictor import setup as frozen_setup
from experiments.phase2.causal_model_scores import CausalModelScores, PreparedPredictionContext
from experiments.phase2.deletion import OfflineFuture
from experiments.phase2.memory import PolicyContext
from experiments.phase2.training_samples import PredictorContext, PredictorSample, phase1_arrays
from phase1.inputs import collate
from phase1.model import project


def original_inputs(parent, history, actions, observations, basis):
    future = OfflineFuture((3, 6), observations[[15, 18]], actions[12:18], np.ones(2, bool))
    sample = PredictorSample(PredictorContext(12, parent.context.recent, tuple(history), actions[:12], "additive_controls"), future, {})
    arrays, _ = phase1_arrays(sample)
    arrays["projected_z"] = project(arrays["z"], basis)
    arrays["projected_current"] = project(arrays["current"], basis)
    return collate([arrays], "cpu")


@pytest.mark.parametrize("history_count", [0, 1, 2])
def test_prepared_statistics_reconstruct_exact_original_h3_with_nonzero_actions(frozen_setup, history_count):
    evaluator, parent, sums, actions, observations, basis, *_ = frozen_setup
    scores = CausalModelScores(evaluator)
    history = parent.candidates[:history_count]
    residents = parent.context.recent + history
    past = {e.entry_id: sums[e.entry_id].copy() for e in residents}
    before = evaluator.audit()["state_sha256"]
    original_batch = original_inputs(parent, history, actions, observations, basis)
    with torch.inference_mode():
        expected = evaluator._model(original_batch)[0].numpy()
    prepared = scores.prepare_context(parent.context.recent, history, past, 12)
    assert [f.name for f in fields(prepared)] == ["origin_time", "values"]
    assert prepared.values.dtype == np.float32 and prepared.values.shape == (5 * 4 + 3,)
    assert prepared.numeric_bytes == (5 * 4 + 3) * 4 + 8
    assert prepared.origin_time == 12
    assert np.array_equal(scores.predict_context(prepared, observations[12], actions[12:15]), expected[0])
    for h, index in ((3, 0), (6, 1)):
        assert np.array_equal(scores.predict(parent.context.recent, history, past, actions[12:12+h], h), expected[index])
    # Deferred reconstruction cannot consult mutated caller-side action summaries.
    for value in past.values():
        value[:] = 10000
    assert np.array_equal(scores.predict_context(prepared, observations[12], actions[12:15]), expected[0])
    with pytest.raises(ValueError):
        prepared.values.setflags(write=True)
    assert evaluator.audit()["state_sha256"] == before


def test_attention_is_original_model_content_weights_and_preparation_uses_no_future_actions(frozen_setup):
    evaluator, parent, sums, actions, observations, basis, *_ = frozen_setup
    scores = CausalModelScores(evaluator)
    batch = original_inputs(parent, parent.candidates, actions, observations, basis)
    captured = []
    hook = evaluator._model.score.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))
    with torch.inference_mode():
        evaluator._model(batch)
    hook.remove()
    p, current = batch["projected_z"], batch["projected_current"]
    history = batch["mask"] & (batch["type"] == 1)
    score = captured[0].squeeze(-1) + torch.log1p((p - current[:, None]).square().mean(-1))
    expected = torch.softmax(score.masked_fill(~history, -1e4), 1) * history
    expected = expected / expected.sum(1, keepdim=True).clamp_min(1e-8)
    def forbidden_action_call(*unused):
        raise AssertionError("Action-independent preparation called the future-action module")
    guard = evaluator._model.action.register_forward_hook(forbidden_action_call)
    try:
        actual = scores.attention(parent.candidates, PolicyContext(12, parent.context.recent))
        assert np.array_equal(actual, expected[0, 4:].numpy())
        history_entries = parent.candidates[:2]
        residents = parent.context.recent + history_entries
        scores.prepare_context(parent.context.recent, history_entries, {e.entry_id: sums[e.entry_id] for e in residents}, 12)
    finally:
        guard.remove()
    assert scores.audit()["frozen_predictor"]["passed"]


def test_context_refuses_wrong_origin_extra_archive_and_unrealized_action_suffix(frozen_setup):
    evaluator, parent, sums, actions, observations, *_ = frozen_setup
    scores = CausalModelScores(evaluator)
    history = parent.candidates[:2]
    residents = parent.context.recent + history
    past = {e.entry_id: sums[e.entry_id] for e in residents}
    with pytest.raises(ValueError, match="exactly current resident"):
        scores.prepare_context(parent.context.recent, history, sums, 12)
    context = scores.prepare_context(parent.context.recent, history, past, 12)
    with pytest.raises(ValueError, match="prepared origin"):
        scores.predict_context(context, observations[13], actions[12:15])
    with pytest.raises(ValueError, match="exactly three"):
        scores.predict_context(context, observations[12], actions[12:18])
    with pytest.raises(ValueError, match="horizon-length"):
        scores.predict(parent.context.recent, history, past, actions[12:18], 3)
    with pytest.raises(ValueError, match=r"5\*d\+3"):
        PreparedPredictionContext(12, np.zeros(24, np.float32))


def test_score_paths_refuse_changed_frozen_model(frozen_setup):
    evaluator, parent, *_ = frozen_setup
    scores = CausalModelScores(evaluator)
    evaluator._model.train()
    with pytest.raises(ValueError, match="changed"):
        scores.attention(parent.candidates, PolicyContext(12, parent.context.recent))
