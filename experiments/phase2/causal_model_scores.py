"""Causal attention and bounded delayed-surprise statistics from the real model.

Delayed surprise reconstructs the model's q->q+3 prediction after the three
actions have actually occurred. It is an action-conditioned error available at
q+3, not a forecast that knew those controls at q. Queue/state accounting belongs
to the streaming runner; this module retains no observation/action archive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

from .frozen_predictor import FrozenPredictorEvaluator
from .memory import MemoryEntry, PolicyContext
from phase1.inputs import collate
from phase1.model import project


def _immutable(array):
    a = np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class PreparedPredictionContext:
    """Only 5*d+3 float32 statistics plus an int64 origin timestamp.

    values = [projected_current, memory, velocity, transported_at_h3, past_mean,
              mean_age, has_history, 3/8]. No raw feature, action sequence, model,
    future information, episode/entry ID, or full historical context is stored.
    Consumers must use the same verified model instance that prepared it.
    """

    origin_time: int
    values: np.ndarray = field(repr=False)

    def __post_init__(self):
        if isinstance(self.origin_time, (bool, np.bool_)) or not isinstance(self.origin_time, (int, np.integer)) or self.origin_time < 0:
            raise ValueError("origin_time must be a nonnegative integer")
        a = np.asarray(self.values)
        if a.ndim != 1 or a.dtype != np.float32 or len(a) < 8 or (len(a) - 3) % 5 or not np.isfinite(a).all():
            raise ValueError("Prepared context must contain finite float32 values of length 5*d+3")
        object.__setattr__(self, "origin_time", int(self.origin_time))
        object.__setattr__(self, "values", _immutable(a))

    @property
    def numeric_bytes(self):
        """Logical numeric storage: packed float32 statistics plus int64 time."""
        return self.values.nbytes + 8


class CausalModelScores:
    def __init__(self, evaluator: FrozenPredictorEvaluator):
        if not isinstance(evaluator, FrozenPredictorEvaluator):
            raise ValueError("A verified FrozenPredictorEvaluator is required")
        evaluator.audit()
        self._evaluator = evaluator
        self._model = evaluator._model
        self._basis = evaluator._basis
        self.dimension = evaluator.feature_dimension
        self.projection_dimension = int(self._model.projection.shape[1])
        self._counts = {"attention_calls": 0, "contexts_prepared": 0,
                        "delayed_h3_predictions": 0, "direct_predictions": 0}

    def _entries(self, recent, history, current_time, max_history=2):
        recent, history = tuple(recent), tuple(history)
        if (isinstance(current_time, (bool, np.bool_)) or not isinstance(current_time, (int, np.integer))
                or len(recent) != 4 or len(history) > max_history):
            raise ValueError("Require four recent observations and the allowed bounded history")
        tokens = recent + history
        if any(not isinstance(e, MemoryEntry) for e in tokens):
            raise ValueError("Only MemoryEntry observations are accepted")
        if (recent[-1].timestamp != current_time
                or [e.timestamp for e in recent] != sorted(set(e.timestamp for e in recent))
                or any(e.timestamp >= recent[0].timestamp for e in history)
                or len({e.entry_id for e in tokens}) != len(tokens)
                or len({e.episode_id for e in tokens}) != 1):
            raise ValueError("Observation identities/timestamps cross the causal recent/history boundary")
        if any(e.representation.dtype != np.float32 or e.representation.shape != (self.dimension,) for e in tokens):
            raise ValueError("Require normalized float32 observations in the verified feature space")
        return recent, history

    def _batch(self, recent, history, past_actions, current_time, actions=None):
        recent, history = self._entries(recent, history, current_time)
        tokens = recent + history
        if set(past_actions) != {e.entry_id for e in tokens}:
            raise ValueError("Past-action map must contain exactly current resident entries")
        summaries = []
        for entry in tokens:
            value = np.asarray(past_actions[entry.entry_id])
            if value.shape != (2,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Require finite float32 executed-action summaries of shape (2,)")
            summaries.append(value.copy())
        if not np.array_equal(summaries[3], np.zeros(2, np.float32)):
            raise ValueError("Current observation must have zero executed-action suffix")
        z = np.stack([e.representation for e in tokens])
        sample = {"z": z, "age": np.array([current_time - e.timestamp for e in tokens], np.float32),
                  "type": np.array([0] * 4 + [1] * len(history), np.int64),
                  "past_actions": np.stack(summaries), "actions": np.zeros((6, 2), np.float32) if actions is None else actions,
                  "current": recent[-1].representation.copy(), "projected_z": project(z, self._basis),
                  "projected_current": project(recent[-1].representation, self._basis)}
        return collate([sample], "cpu")

    def _weights(self, p, current, history):
        # Same parameters, operation order, novelty term, masking and
        # normalization as the locked DynamicsPredictor.forward implementation.
        score = self._model.score(torch.cat([p, p - current[:, None]], -1)).squeeze(-1)
        score = score + torch.log1p((p - current[:, None]).square().mean(-1))
        w = torch.softmax(score.masked_fill(~history, -1e4), 1) * history
        return w / w.sum(1, keepdim=True).clamp_min(1e-8)

    def attention(self, candidates, context: PolicyContext):
        self._evaluator._ensure_frozen()
        if not isinstance(context, PolicyContext):
            raise ValueError("An explicit causal PolicyContext is required")
        recent, candidates = self._entries(context.recent, candidates, context.current_time, max_history=3)
        if not candidates:
            raise ValueError("Attention requires at least one historical candidate")
        tokens = recent + candidates
        p = torch.from_numpy(project(np.stack([e.representation for e in tokens]), self._basis))[None]
        current = torch.from_numpy(project(recent[-1].representation, self._basis))[None]
        mask = torch.tensor([[False] * 4 + [True] * len(candidates)])
        with torch.inference_mode():
            scores = self._weights(p, current, mask)[0, 4:].numpy().copy()
        self._evaluator._ensure_frozen()
        self._counts["attention_calls"] += 1
        return scores

    def prepare_context(self, recent, history, past_actions: Mapping, current_time: int):
        """Snapshot only action-independent q-time sufficient statistics."""
        self._evaluator._ensure_frozen()
        batch = self._batch(recent, history, past_actions, current_time)
        with torch.inference_mode():
            valid = batch["mask"]
            p = batch["projected_z"].masked_fill(~valid[..., None], 0)
            current = batch["projected_current"]
            hist = valid & (batch["type"] == 1)
            w = self._weights(p, current, hist)
            past = self._model.past_action(batch["past_actions"].masked_fill(~valid[..., None], 0))
            aligned = p + past
            age = batch["age"].masked_fill(~valid, 0) / 8
            mean_age = (w * age).sum(1)
            center = age - mean_age[:, None]
            memory = (w[..., None] * aligned).sum(1)
            past_mean = (w[..., None] * past).sum(1)
            variance = (w * center.square()).sum(1)
            velocity = -(w[..., None] * center[..., None] * (aligned - memory[:, None])).sum(1) / variance.clamp_min(.01)[:, None]
            transported = memory + velocity * (mean_age + 3 / 8)[:, None]
            stats = torch.stack([mean_age, hist.any(1).float(), torch.full_like(mean_age, 3 / 8)], -1)
            packed = torch.cat([current, memory, velocity, transported, past_mean, stats], -1)[0].numpy().copy()
        self._evaluator._ensure_frozen()
        self._counts["contexts_prepared"] += 1
        return PreparedPredictionContext(current_time, packed)

    def predict_context(self, context: PreparedPredictionContext, current_representation, actions):
        """Resolve q->q+3 once actual controls and the target become available.

        current_representation is the raw normalized observation at q, still
        resident in a four-frame recent buffer at q+3. No target is an input.
        The runner is responsible for matching context.origin_time to that entry.
        """
        self._evaluator._ensure_frozen()
        if not isinstance(context, PreparedPredictionContext) or len(context.values) != 5 * self.projection_dimension + 3:
            raise ValueError("Prepared statistics do not match this frozen predictor")
        original = np.asarray(current_representation)
        actions = np.asarray(actions)
        if original.dtype != np.float32 or original.shape != (self.dimension,) or not np.isfinite(original).all():
            raise ValueError("Require the origin observation's normalized float32 feature vector")
        if actions.dtype != np.float32 or actions.shape != (3, 2) or not np.isfinite(actions).all():
            raise ValueError("Delayed h3 prediction uses exactly three realized float32 actions")
        d = self.projection_dimension
        if not np.array_equal(project(original, self._basis), context.values[:d]):
            raise ValueError("Supplied current representation does not match the prepared origin")
        with torch.inference_mode():
            current, memory, velocity, transported, past_mean = torch.from_numpy(context.values[:5 * d].copy()).reshape(5, 1, d)
            stats = torch.from_numpy(context.values[5 * d:].copy())[None]
            controls = torch.from_numpy(actions.copy())[None]
            action = self._model.action(torch.cat([controls[:, 0], controls.sum(1), controls[:, -1]], -1))
            features = torch.cat([current, memory, velocity, transported, action, memory * action,
                                  past_mean, velocity * action, stats], -1)
            coeff = self._model.output(features)
            output = torch.from_numpy(original.copy())[None] + self._model.target_mean + coeff @ self._model.output_basis.T
            value = output[0].numpy().copy()
        self._evaluator._ensure_frozen()
        if value.dtype != np.float32 or value.shape != (self.dimension,) or not np.isfinite(value).all():
            raise ValueError("Invalid delayed frozen-model prediction")
        self._counts["delayed_h3_predictions"] += 1
        return value

    def predict(self, recent, history, past_actions: Mapping, action_prefix, horizon):
        """Ordinary query prediction for zero to two retained entries, h3/h6."""
        self._evaluator._ensure_frozen()
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon not in (3, 6):
            raise ValueError("Only horizons 3 and 6 were trained")
        prefix = np.asarray(action_prefix)
        if prefix.dtype != np.float32 or prefix.shape != (horizon, 2) or not np.isfinite(prefix).all():
            raise ValueError("Require exactly the horizon-length float32 action prefix")
        recent = tuple(recent)
        if not recent:
            raise ValueError("Recent context is required")
        actions = np.zeros((6, 2), np.float32)
        actions[:horizon] = prefix
        batch = self._batch(recent, history, past_actions, recent[-1].timestamp, actions)
        with torch.inference_mode():
            value = self._model(batch)[0, 0 if horizon == 3 else 1].numpy().copy()
        self._evaluator._ensure_frozen()
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite frozen-model prediction")
        self._evaluator._calls["count"] += 1
        self._counts["direct_predictions"] += 1
        return value

    def audit(self):
        return {"frozen_predictor": self._evaluator.audit(), "score_operations": dict(self._counts),
                "prepared_float32_values": 5 * self.projection_dimension + 3,
                "prepared_numeric_bytes_including_timestamp": (5 * self.projection_dimension + 3) * 4 + 8,
                "delayed_surprise_semantics": "q-time memory/context with controls realized by q+3; targets are never model inputs"}
