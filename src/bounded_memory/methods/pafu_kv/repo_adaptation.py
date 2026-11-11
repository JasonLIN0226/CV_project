"""The repo's earlier PaFu-KV adaptation, kept as a second PaFu row on Gate.

It differs from gate.py in one rule: the three resident candidates are re-scored together at
every eviction, with inputs that include the other candidates and the current frame. gate.py
follows the paper more literally (one score per frame, written once and never updated).

The method itself (`make_student_inputs`, `balanced_attention_targets`, `SalienceStudent`,
`SalienceTrainingRecord`, `train_salience_student` and their two helpers) is extracted unchanged
from experiments/phase2/literature_baselines.py, the 22 Oct adaptation: past/same/future
attention target, set-conditioned student inputs, two-layer SiLU student, full-batch SmoothL1 fit
(100 epochs, hidden 32, lr 1e-3, final epoch). The harness below it is moved from
experiments/memory_baselines/pafu_repo.py: training states are the evictions visited by the four
behaviour policies of the scorer study, capped at `max_states` per seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...environments.gate.protocol import B, R
from ...evaluation.heuristics.gate import update as behavior_rule
from ..deletion_cost.retention import BEHAVIORS, behavior_update, deterministic_seed, rollout


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _immutable(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def make_student_inputs(candidate_projected, recent_projected, candidate_times,
                        recent_times, current_time):
    """C x (5D+1): candidate,current,delta,candidate mean,recent mean,log1p age.

    These are set-equivariant causal features. The caller may apply a frozen,
    train-only projection upstream; this function fits no normalization and
    receives no teacher target, future action, episode ID, or task evidence.
    """
    current_time = _integer(current_time, "current_time")
    candidate, recent = np.asarray(candidate_projected), np.asarray(recent_projected)
    if (candidate.ndim != 2 or recent.ndim != 2 or min(candidate.shape) < 1 or min(recent.shape) < 1
            or candidate.shape[1] != recent.shape[1] or candidate.dtype != np.float32 or recent.dtype != np.float32
            or not np.isfinite(candidate).all() or not np.isfinite(recent).all()):
        raise ValueError("candidate/recent features must be finite nonempty float32 matrices of equal width")
    ct, rt = np.asarray(candidate_times), np.asarray(recent_times)
    if (ct.shape != (len(candidate),) or rt.shape != (len(recent),)
            or not np.issubdtype(ct.dtype, np.integer) or not np.issubdtype(rt.dtype, np.integer)
            or np.any(ct < 0) or np.any(rt < 0) or len(np.unique(ct)) != len(ct)
            or np.any(np.diff(rt) <= 0) or rt[-1] != current_time or np.any(ct >= rt[0])):
        raise ValueError("timestamps must be unique; candidates precede ordered recent context ending now")
    current = np.broadcast_to(recent[-1], candidate.shape)
    candidate_mean = np.broadcast_to(candidate.mean(axis=0), candidate.shape)
    recent_mean = np.broadcast_to(recent.mean(axis=0), candidate.shape)
    ages = np.log1p(current_time - ct).astype(np.float32)[:, None]
    values = np.concatenate([candidate, current, candidate - current, candidate_mean, recent_mean, ages], axis=1)
    if not np.isfinite(values).all():
        raise ValueError("student features overflowed")
    return _immutable(values.astype(np.float32, copy=False))


def balanced_attention_targets(attention, query_times, candidate_times, *, decision_time=None):
    """Offline PaFu-inspired temporal attention target, never an online scorer.

    ``attention[query,key]`` must be actual normalized reader weights. For EACH
    key j, partition query timestamps into past/same/future relative to key j;
    take the maximum in each nonempty block and average available block maxima.
    Finally normalize across candidate keys. Missing blocks are omitted, never
    padded with fictitious zeros. At least one genuinely later query is required
    per key; otherwise this is not a future-attention target. The runner should
    also pass ``decision_time``: this requires a query strictly after the ONLINE
    decision, not merely after an old key timestamp. The runner must separately
    fix the maximum future horizon before label generation.

    An earlier query reading a later key is a bidirectional teacher probe, not a
    causal rollout of a predictor trained only with past keys. The caller must
    report that distribution shift. The function neither invents attention nor
    accepts losses in its place; trace/provenance audits remain runner duties.
    """
    a = np.asarray(attention)
    qt, kt = np.asarray(query_times), np.asarray(candidate_times)
    if (a.ndim != 2 or min(a.shape) < 1 or a.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(a).all() or np.any(a < 0)
            or not np.allclose(a.sum(axis=1), 1., rtol=0., atol=1e-5)):
        raise ValueError("attention must contain finite nonnegative normalized reader-weight rows")
    if (qt.shape != (a.shape[0],) or kt.shape != (a.shape[1],)
            or not np.issubdtype(qt.dtype, np.integer) or not np.issubdtype(kt.dtype, np.integer)
            or np.any(qt < 0) or np.any(kt < 0) or len(np.unique(qt)) != len(qt)
            or len(np.unique(kt)) != len(kt)):
        raise ValueError("query/key times must be nonnegative unique integer vectors of matching size")
    if decision_time is not None:
        decision_time = _integer(decision_time, "decision_time")
        if np.any(kt > decision_time) or not np.any(qt > decision_time):
            raise ValueError("teacher needs past/current keys and an actual query after the online decision")
    salience = np.empty(len(kt), np.float64)
    for j, time in enumerate(kt):
        groups = (qt < time, qt == time, qt > time)
        if not groups[-1].any():
            raise ValueError("every key needs an actual future attention query")
        salience[j] = np.mean([a[mask, j].max() for mask in groups if mask.any()])
    if not np.isfinite(salience).all() or salience.sum() <= 0:
        raise ValueError("attention salience cannot be normalized")
    return _immutable((salience / salience.sum()).astype(np.float32))


class SalienceStudent(nn.Module):
    """Shared two-layer scalar MLP; cross-candidate context is explicit in input."""

    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.input_dim = _integer(input_dim, "input_dim", 1)
        self.hidden = _integer(hidden, "hidden", 1)
        self.network = nn.Sequential(nn.Linear(self.input_dim, self.hidden), nn.SiLU(), nn.Linear(self.hidden, 1))

    def forward(self, inputs):
        if (inputs.ndim != 2 or inputs.shape[1] != self.input_dim or inputs.shape[0] < 1
                or inputs.dtype != torch.float32 or not torch.isfinite(inputs).all()):
            raise ValueError("student inputs must be a finite float32 C x input_dim tensor")
        return self.network(inputs).squeeze(-1)


@dataclass(frozen=True, eq=False)
class SalienceTrainingRecord:
    inputs: np.ndarray
    teacher_probability: np.ndarray
    group_id: str
    split: str = "train"
    target_kind: str = "future_attention"

    def __post_init__(self):
        inputs, target = np.asarray(self.inputs), np.asarray(self.teacher_probability)
        if inputs.ndim != 2 or min(inputs.shape) < 1 or inputs.dtype != np.float32 or not np.isfinite(inputs).all():
            raise ValueError("inputs must be a finite nonempty float32 matrix")
        if (target.shape != (len(inputs),) or target.dtype != np.float32 or not np.isfinite(target).all()
                or np.any(target < 0) or not np.isclose(target.sum(dtype=np.float64), 1., rtol=0., atol=1e-5)):
            raise ValueError("teacher_probability must be finite nonnegative float32 probabilities summing to one")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("a nonempty independent group_id is required")
        if self.split not in ("train", "validation", "test") or self.target_kind != "future_attention":
            raise ValueError("only explicit split names and future_attention targets are accepted; never deletion losses")
        object.__setattr__(self, "inputs", _immutable(inputs))
        object.__setattr__(self, "teacher_probability", _immutable(target))


def train_salience_student(records: Sequence[SalienceTrainingRecord], *, seed,
                           epochs=100, hidden=32, lr=1e-3, train_group_ids=None):
    """Train-only full-batch SmoothL1 on probabilities; return FINAL model, curve.

    Each state contributes equally regardless of candidate count. Curve[0] is
    the initial training loss and curve[e] is loss after e updates. No validation
    checkpoint selection, deletion loss input, future feature input, or hidden
    normalization is performed. The returned CPU student is frozen and in eval
    mode. ``train_group_ids`` optionally enforces an externally fixed split.
    """
    seed, epochs = _integer(seed, "seed"), _integer(epochs, "epochs", 1)
    hidden = _integer(hidden, "hidden", 1)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    records = tuple(records)
    if not records or any(not isinstance(record, SalienceTrainingRecord) for record in records):
        raise ValueError("nonempty SalienceTrainingRecord sequence required")
    if any(record.split != "train" or record.target_kind != "future_attention" for record in records):
        raise ValueError("only train-split future_attention records may train the student")
    if train_group_ids is not None:
        permitted = set(train_group_ids)
        if any(record.group_id not in permitted for record in records):
            raise ValueError("record group lies outside the fixed train group IDs")
    width = records[0].inputs.shape[1]
    if any(record.inputs.shape[1] != width for record in records):
        raise ValueError("all training states must share an input width")
    tensors = torch.from_numpy(np.concatenate([record.inputs for record in records]).copy())
    sizes = [len(record.inputs) for record in records]
    targets = [torch.from_numpy(record.teacher_probability.copy()) for record in records]

    # CPU RNG state is restored, so fitting a baseline does not perturb another
    # method's independently fixed initialization or episode generation seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SalienceStudent(width, hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        def objective():
            logits = model(tensors).split(sizes)
            return torch.stack([F.smooth_l1_loss(torch.softmax(logit, dim=0), target)
                                for logit, target in zip(logits, targets)]).mean()

        with torch.no_grad():
            curve = [float(objective())]
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite salience training loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                curve.append(float(objective()))
    model.eval()
    model.requires_grad_(False)
    return model, curve


@torch.inference_mode()
def _attention_logits(model, e, device):
    """Pooling logits of every eligible key for a query placed at every frame: [frames, q-R+1]."""
    keys = torch.from_numpy(e.p[:e.candidates]).to(device)
    queries = torch.from_numpy(e.p).to(device)
    delta = keys[None] - queries[:, None]
    logits = model.score(torch.cat([keys[None].expand(len(queries), -1, -1), delta], -1)).squeeze(-1)
    return (logits + torch.log1p(delta.square().mean(-1))).cpu().numpy().astype(np.float64)


def _teacher(logits, candidates, t):
    rows = logits[:t + 7][:, candidates]
    rows = np.exp(rows - rows.max(1, keepdims=True))
    return balanced_attention_targets(rows / rows.sum(1, keepdims=True), np.arange(t + 7), np.asarray(candidates),
                                      decision_time=t)


def _inputs(e, candidates, t):
    recent = np.arange(t - R + 1, t + 1)
    return make_student_inputs(e.p[list(candidates)], e.p[recent], np.asarray(candidates), recent, int(t))


def train(model, train_episodes, seed, device, max_states=20000):
    update = behavior_rule
    states = []
    for e in train_episodes:
        seen = set()
        for policy in BEHAVIORS:
            rng = np.random.default_rng(deterministic_seed(seed, e.id, policy))
            _, trace = rollout(e.p[:e.q + 1].copy(), e.actions[:e.q].copy(), e.q, behavior_update(policy, rng, update))
            for row in trace:
                candidates = tuple(row["previous"] + [row["arrival"]])
                if len(candidates) == B + 1 and (row["clock"], tuple(sorted(candidates))) not in seen:
                    seen.add((row["clock"], tuple(sorted(candidates))))
                    states.append((e, candidates, row["clock"]))
    total = len(states)
    if total > max_states:
        keep = np.sort(np.random.default_rng(seed).choice(total, size=max_states, replace=False))
        states = [states[i] for i in keep]
    logits, records = {}, []
    for e, candidates, t in states:
        if e.id not in logits:
            logits[e.id] = _attention_logits(model, e, device)
        records.append(SalienceTrainingRecord(_inputs(e, candidates, t), _teacher(logits[e.id], list(candidates), t),
                                              e.pair, split="train", target_kind="future_attention"))
    student, curve = train_salience_student(records, seed=seed, epochs=100, hidden=32, lr=1e-3)
    return student, {"decision_states_visited": total, "decision_states_used": len(records),
                     "loss_first": curve[0], "loss_last": curve[-1]}


@torch.inference_mode()
def select(student, episodes):
    out = []
    for e in episodes:
        kept = []
        for t in range(R, e.q + 1):
            pool = kept + [t - R]
            if len(pool) > B:
                keep_score = torch.softmax(student(torch.from_numpy(np.array(_inputs(e, pool, t)))), -1).numpy()
                pool.pop(min(range(len(pool)), key=lambda j: (keep_score[j], pool[j])))
            kept = pool
        out.append(sorted(kept))
    return out
