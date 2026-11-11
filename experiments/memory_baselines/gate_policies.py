"""Reference rungs and the repo's own policies on Gate, rebuilt from the frozen project code.

Nothing new is defined here: the heuristics come from gate_retention_comparison_v1/online.py,
and the deletion-cost scorer ("ours") is trained with the same teacher, initialisation,
minibatch order and optimiser as gate_supervision_comparison_design_v2.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import torch

from .gate_data import B, R, ROOT

ONLINE = ROOT / "experiments/gate_retention_comparison_v1/online.py"
HEURISTICS = ("fifo", "reservoir", "temporal_buckets", "visual_diversity", "prediction_surprise", "attention")


def _online_module():
    spec = importlib.util.spec_from_file_location("memory_baselines_online_rules", ONLINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recent_only(episodes):
    return [[] for _ in episodes]


def oracle_keyframe(episodes):
    """Privileged reference: the cue timestamps, padded with the newest eligible frames."""
    out = []
    for e in episodes:
        kept = [int(t) for t in e.evidence if int(t) <= e.q - R][:B]
        for t in range(e.q - R, -1, -1):
            if len(kept) == B:
                break
            if t not in kept:
                kept.append(t)
        out.append(sorted(kept))
    return out


def heuristics(model, episodes, seed, device):
    """The six frozen online rules, replayed exactly as in the 29 Oct validation."""
    from phase1.inputs import collate
    from phase1.utils import stable_seed

    online = _online_module()
    out = {name: [] for name in HEURISTICS}
    for e in episodes:
        q = e.q
        kept, _, _, _ = online.replay(e.z[:q + 1].copy(), e.p[:q + 1].copy(), e.actions[:q].copy(), q, model,
                                      collate, device, stable_seed(2026102317, seed, e.id, "reservoir"))
        for name in HEURISTICS:
            out[name].append(sorted(kept[name]))
    return out


def scorer_states(episodes, seed):
    """Query-time three-candidate states reached by the four behaviour policies (ours' training set)."""
    from experiments.gate_supervision_comparison_design_v2.core import BEHAVIORS, behavior_update, rollout
    from experiments.gate_supervision_comparison_design_v2.states import deterministic_seed, load_frozen_behavior

    update = load_frozen_behavior(ONLINE)
    for e in episodes:
        for policy in BEHAVIORS:
            rng = np.random.default_rng(deterministic_seed(seed, e.id, policy))
            state, _ = rollout(e.p[:e.q + 1].copy(), e.actions[:e.q].copy(), e.q,
                               behavior_update(policy, rng, update), capture_final=True)
            yield e, policy, state


def train_scorers(model, train_episodes, seed, device):
    """Deletion-cost scorer and its matched feature-attention-proxy twin (same init, same batches)."""
    from phase1.inputs import collate
    from experiments.gate_supervision_comparison_design_v2.core import EvictionScorer, paired_initialization
    from experiments.gate_supervision_comparison_design_v2.fit import fit_pair, minibatch_indices
    from experiments.gate_supervision_comparison_design_v2.teachers import OfflineTargets, teacher_pair

    x, ya, yd = [], [], []
    for e, _, state in scorer_states(train_episodes, seed):
        q = e.q
        targets = OfflineTargets("train", e.z[[q + 3, q + 6]].copy(), e.actions[q:q + 6].copy())
        labels = teacher_pair(state, targets, e.z[:q + 1].copy(), model, collate, device)
        x.append(state.features()); ya.append(labels["attention_target"]); yd.append(labels["deletion_target"])
    x, ya, yd = (np.stack(v).astype(np.float32) for v in (x, ya, yd))
    if len(x) == 4096:
        fit = fit_pair(x, ya, yd, seed=seed, split="train", device=device)  # the project's exact fit
        states = {"attention_proxy_student": fit["attention"], "deletion_cost_student": fit["deletion"]}
    else:
        # Same update rule for other dataset sizes (local smoke runs only).
        xt = torch.as_tensor(x, device=device)
        a, d = paired_initialization(seed)
        models = [a.to(device), d.to(device)]
        opts = [torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001) for m in models]
        labels = [torch.as_tensor(y, device=device) for y in (ya, yd)]
        for batch in minibatch_indices(seed, len(x)):
            idx = torch.as_tensor(batch, device=device)
            for m, opt, y in zip(models, opts, labels):
                opt.zero_grad(set_to_none=True)
                loss = -(y[idx] * torch.log_softmax(m(xt[idx]), dim=-1)).sum(-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        states = {"attention_proxy_student": models[0].state_dict(), "deletion_cost_student": models[1].state_dict()}
    out = {}
    for name, state in states.items():
        scorer = EvictionScorer().to(device)
        scorer.load_state_dict(state)
        out[name] = scorer.eval()
    return out, {"training_states": int(len(x))}


def frozen_scorers(model_root, seed, device):
    """The team's published scorers for this seed (same checks as the bundle's tools/load_student.py)."""
    import hashlib
    from pathlib import Path
    from experiments.gate_supervision_comparison_design_v2.core import EvictionScorer

    path = Path(model_root) / f"student_{seed}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    models = checkpoint.get("models")
    if checkpoint.get("seed") != seed or checkpoint.get("step") != 1000 or not isinstance(models, (list, tuple)) or len(models) != 2:
        raise ValueError("not the completed matched scorer checkpoint for this seed")
    out = {}
    for name, state in zip(("attention_proxy_student", "deletion_cost_student"), models):
        scorer = EvictionScorer().to(device)
        scorer.load_state_dict(state, strict=True)
        if not all(torch.isfinite(p).all() for p in scorer.parameters()):
            raise ValueError("scorer contains nonfinite parameters")
        out[name] = scorer.eval()
    return out, {"source": "team frozen checkpoint, not retrained", "file": path.name,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def scorer_selections(scorer, episodes):
    from experiments.gate_supervision_comparison_design_v2.core import rollout, score_update

    out = []
    for e in episodes:
        kept, _ = rollout(e.p[:e.q + 1].copy(), e.actions[:e.q].copy(), e.q, lambda s: score_update(scorer, s))
        out.append(sorted(kept))
    return out
