"""The repo's earlier PaFu-KV adaptation, re-run on the current Gate setup as a second PaFu row.

Everything method-specific is imported unchanged from experiments/phase2/literature_baselines.py
(the 22 Oct adaptation that reached about 81% on the 23 Oct fresh-seed Gate test): the
past/same/future attention target, the set-conditioned student inputs, the two-layer SiLU
student and its full-batch SmoothL1 fit (100 epochs, hidden 32, lr 1e-3, final epoch).

It differs from pafu.py in one rule. Here the three resident candidates are re-scored together
at every eviction, with inputs that include the other candidates and the current frame. pafu.py
follows the paper more literally: one score per frame, written once and never updated.

Only the harness around it is new: training states are the evictions visited by the four
behaviour policies of the current scorer study, capped at `max_states` per seed.
"""
from __future__ import annotations

import numpy as np
import torch

from .gate_data import B, R


@torch.inference_mode()
def _attention_logits(model, e, device):
    """Pooling logits of every eligible key for a query placed at every frame: [frames, q-R+1]."""
    keys = torch.from_numpy(e.p[:e.candidates]).to(device)
    queries = torch.from_numpy(e.p).to(device)
    delta = keys[None] - queries[:, None]
    logits = model.score(torch.cat([keys[None].expand(len(queries), -1, -1), delta], -1)).squeeze(-1)
    return (logits + torch.log1p(delta.square().mean(-1))).cpu().numpy().astype(np.float64)


def _teacher(logits, candidates, t):
    from experiments.phase2.literature_baselines import balanced_attention_targets

    rows = logits[:t + 7][:, candidates]
    rows = np.exp(rows - rows.max(1, keepdims=True))
    return balanced_attention_targets(rows / rows.sum(1, keepdims=True), np.arange(t + 7), np.asarray(candidates),
                                      decision_time=t)


def _inputs(e, candidates, t):
    from experiments.phase2.literature_baselines import make_student_inputs

    recent = np.arange(t - R + 1, t + 1)
    return make_student_inputs(e.p[list(candidates)], e.p[recent], np.asarray(candidates), recent, int(t))


def train(model, train_episodes, seed, device, max_states=20000):
    from experiments.gate_supervision_comparison_design_v2.core import BEHAVIORS, behavior_update, rollout
    from experiments.gate_supervision_comparison_design_v2.states import deterministic_seed, load_frozen_behavior
    from experiments.phase2.literature_baselines import SalienceTrainingRecord, train_salience_student
    from .gate_policies import ONLINE

    update = load_frozen_behavior(ONLINE)
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
