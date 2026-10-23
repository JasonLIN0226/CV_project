"""Offline equal-capacity deletion labels; no predictor or real dataset supplied.

Every parent contains exactly K+1 timestamped historical observations. Each fork
deletes one entry, retains exactly K, and uses identical recent context, target
features and recorded action continuation. The result keeps past-only scorer
inputs separate from offline labels. Labels are raw feature MSE per horizon;
missing horizons are explicit masks and NaNs, never silently averaged as zeros.

The prediction callback receives retained entries, recent context, a horizon and
only that horizon's recorded action prefix. It never receives target features.
The offline wrapper compares its prediction with the separate future target.
The caller must verify/freeze the actual model and audit the callback's closure
for hidden state, evidence-role labels, future targets and archive access. A
checkpoint hash records claimed provenance; this model-agnostic module cannot
inspect a model or establish that its parameters are frozen. A callable can also
be stateful: callers must arrange deterministic evaluation and identical model
state across forks. This contract alone is not a completed scorer training set.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import numpy as np

from .memory import MemoryEntry


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _immutable(array):
    """Copy through bytes so setflags(write=True) cannot undo read-only storage."""
    a = np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _float_array(value, name, ndim, *, finite=True):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"{name} must be a {ndim}-D float32/float64 array")
    if finite and not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return _immutable(a)


def _entries(value, name):
    entries = tuple(value)
    if not entries or any(not isinstance(e, MemoryEntry) for e in entries):
        raise ValueError(f"{name} must be a nonempty sequence of MemoryEntry")
    # Copy rather than expose input storage; surprise is not part of this label
    # contract or scorer input. Its separate causal computation is not assumed.
    entries = tuple(MemoryEntry(e.episode_id, e.timestamp, e.representation) for e in entries)
    times = [e.timestamp for e in entries]
    if times != sorted(set(times)):
        raise ValueError(f"{name} timestamps must be distinct and ordered")
    first = entries[0]
    if any(e.episode_id != first.episode_id for e in entries):
        raise ValueError(f"{name} must belong to one episode")
    if any(e.representation.shape != first.representation.shape or e.representation.dtype != first.representation.dtype for e in entries):
        raise ValueError(f"{name} feature shape/dtype must be consistent")
    return entries


@dataclass(frozen=True, eq=False)
class CausalContext:
    """Recent observations ending at the current time; no future or state fields."""

    current_time: int
    recent: tuple[MemoryEntry, ...]

    def __post_init__(self):
        now = _integer(self.current_time, "current_time")
        recent = _entries(self.recent, "recent")
        if recent[-1].timestamp != now:
            raise ValueError("recent must end at current_time")
        object.__setattr__(self, "current_time", now)
        object.__setattr__(self, "recent", recent)


@dataclass(frozen=True, eq=False)
class CausalParent:
    """One immutable overflow state; no archive or oracle construction metadata."""

    capacity: int
    candidates: tuple[MemoryEntry, ...]
    context: CausalContext

    def __post_init__(self):
        capacity = _integer(self.capacity, "capacity", 1)
        candidates = _entries(self.candidates, "candidates")
        if len(candidates) != capacity + 1:
            raise ValueError("parent must contain exactly K+1 candidates")
        if not isinstance(self.context, CausalContext):
            raise ValueError("context must be a CausalContext")
        context = CausalContext(self.context.current_time, self.context.recent)
        if candidates[-1].timestamp >= context.recent[0].timestamp:
            raise ValueError("all candidates must precede the recent buffer")
        reference = context.recent[0]
        if candidates[0].episode_id != reference.episode_id:
            raise ValueError("parent and recent must belong to the same episode")
        if candidates[0].representation.shape != reference.representation.shape or candidates[0].representation.dtype != reference.representation.dtype:
            raise ValueError("parent and recent feature shape/dtype must match")
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "context", context)


@dataclass(frozen=True, eq=False)
class OfflineFuture:
    """Label-only future data, separate from the causal parent.

    targets[h] is the feature vector at current_time + horizons[h]. Actions are
    ordered a_t, a_(t+1), ... and must cover every available target horizon.
    Unavailable target rows may contain NaNs; available rows must be finite.
    The trajectory/split manifest must establish this alignment in the caller.
    """

    horizons: tuple[int, ...]
    targets: np.ndarray
    actions: np.ndarray
    available: np.ndarray

    def __post_init__(self):
        horizons = tuple(_integer(h, "horizon", 1) for h in self.horizons)
        if not horizons or list(horizons) != sorted(set(horizons)):
            raise ValueError("horizons must be nonempty, distinct and ordered")
        targets = _float_array(self.targets, "targets", 2, finite=False)
        actions = _float_array(self.actions, "actions", 2)
        mask = np.asarray(self.available)
        if mask.dtype != np.dtype("bool") or mask.shape != (len(horizons),):
            raise ValueError("available must be a boolean vector with one element per horizon")
        if targets.shape[0] != len(horizons) or targets.shape[1] == 0:
            raise ValueError("targets must have shape (horizons, nonempty feature dimension)")
        if not np.isfinite(targets[mask]).all():
            raise ValueError("available targets must be finite")
        required_actions = max((h for h, valid in zip(horizons, mask) if valid), default=0)
        if actions.shape[1] == 0 or len(actions) < required_actions:
            raise ValueError("actions must cover every available horizon with a nonempty action dimension")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "available", _immutable(mask))


@dataclass(frozen=True)
class EvaluatorProvenance:
    """Claimed frozen checkpoint identity, not an automatic freeze verification."""

    frozen_predictor_sha256: str
    evaluator_id: str

    def __post_init__(self):
        if not isinstance(self.frozen_predictor_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.frozen_predictor_sha256):
            raise ValueError("frozen_predictor_sha256 must be a lowercase SHA-256 hex digest")
        if not isinstance(self.evaluator_id, str) or not self.evaluator_id.strip():
            raise ValueError("evaluator_id must be a nonempty identifier")


@dataclass(frozen=True, eq=False)
class ScorerInputs:
    """Past/current-only features. IDs are bookkeeping, not semantic targets."""

    capacity: int
    current_time: int
    candidate_ids: tuple[tuple[str, int], ...]
    candidate_timestamps: np.ndarray
    candidate_representations: np.ndarray
    recent_ids: tuple[tuple[str, int], ...]
    recent_timestamps: np.ndarray
    recent_representations: np.ndarray


@dataclass(frozen=True, eq=False)
class DeletionLabels:
    """Rows follow candidate deletion order; columns follow horizons.

    This object is an offline training target, never an input to the scorer.
    A row's raw loss is L(M minus candidate_i); any later centering, ranking or
    horizon weighting must be explicit and cannot change the candidate capacity.
    """

    deleted_ids: tuple[tuple[str, int], ...]
    kept_ids: tuple[tuple[tuple[str, int], ...], ...]
    horizons: tuple[int, ...]
    horizon_mask: np.ndarray
    feature_mse: np.ndarray


@dataclass(frozen=True, eq=False)
class DeletionSupervision:
    scorer_inputs: ScorerInputs
    labels: DeletionLabels
    provenance: EvaluatorProvenance


PredictionEvaluator = Callable[[tuple[MemoryEntry, ...], CausalContext, int, np.ndarray], np.ndarray]


def generate_deletion_supervision(parent: CausalParent, future: OfflineFuture,
                                  evaluator: PredictionEvaluator, *,
                                  provenance: EvaluatorProvenance) -> DeletionSupervision:
    """Enumerate all K+1 deletions and calculate raw per-horizon feature MSE.

    ``evaluator(kept, context, horizon, recorded_action_prefix)`` returns exactly
    one finite feature vector of the target dimension. It must use the same
    independently verified frozen predictor for every invocation. Future target
    vectors are deliberately unavailable through the callback arguments.

    An all-missing target returns all-masked/NaN labels without model calls; a
    downstream dataset writer must exclude it from loss-bearing training rows.
    No original result threshold, fresh confirmation gate or learned scorer is
    implemented here.
    """
    if not isinstance(parent, CausalParent) or not isinstance(future, OfflineFuture):
        raise ValueError("parent and future must use the explicit causal/offline contracts")
    if not isinstance(provenance, EvaluatorProvenance):
        raise ValueError("explicit evaluator provenance is required")
    if not callable(evaluator):
        raise ValueError("evaluator must be callable")
    dimension = parent.candidates[0].representation.size
    if future.targets.shape[1] != dimension:
        raise ValueError("target feature dimension must match memory representations")
    entries, recent = parent.candidates, parent.context.recent
    scorer = ScorerInputs(
        parent.capacity, parent.context.current_time,
        tuple(e.entry_id for e in entries),
        _immutable(np.array([e.timestamp for e in entries], dtype=np.int64)),
        _immutable(np.stack([e.representation for e in entries])),
        tuple(e.entry_id for e in recent),
        _immutable(np.array([e.timestamp for e in recent], dtype=np.int64)),
        _immutable(np.stack([e.representation for e in recent])),
    )
    loss = np.full((len(entries), len(future.horizons)), np.nan, dtype=np.float64)
    kept_ids = []
    for deleted in range(len(entries)):
        kept = tuple(e for i, e in enumerate(entries) if i != deleted)
        if len(kept) != parent.capacity:
            raise ValueError("internal deletion capacity mismatch")
        kept_ids.append(tuple(e.entry_id for e in kept))
        for column, horizon in enumerate(future.horizons):
            if not future.available[column]:
                continue
            # An independent immutable copy prevents mutation or .base access to
            # a later action suffix, including for a short-horizon prediction.
            prefix = _immutable(future.actions[:horizon])
            prediction = _float_array(evaluator(kept, parent.context, horizon, prefix), "prediction", 1)
            if prediction.shape != (dimension,):
                raise ValueError("prediction feature dimension must match the target")
            with np.errstate(over="ignore", invalid="ignore"):
                residual = prediction.astype(np.float64) - future.targets[column].astype(np.float64)
                value = np.mean(residual * residual)
            if not np.isfinite(value):
                raise ValueError("raw feature MSE must be finite")
            loss[deleted, column] = value
    labels = DeletionLabels(tuple(e.entry_id for e in entries), tuple(kept_ids),
                            future.horizons, _immutable(future.available), _immutable(loss))
    return DeletionSupervision(scorer, labels, provenance)
