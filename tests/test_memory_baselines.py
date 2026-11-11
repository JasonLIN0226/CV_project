"""Software checks for experiments/memory_baselines. Synthetic inputs only; no scientific result."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memory_baselines import fm, pacc, pafu  # noqa: E402
from experiments.memory_baselines.framemorrow_ref import ranking_loss, teacher_scores  # noqa: E402
from experiments.memory_baselines.gate_data import B, R, Episode, item_batch, with_memory_tokens  # noqa: E402

D = 16


def episode(seed=0, q=12, dim=D):
    rng = np.random.default_rng(seed)
    frames = q + 7
    actions = rng.normal(size=(frames - 1, 2)).astype(np.float32)
    return Episode(id=f"e{seed}", pair=f"p{seed // 2}", sibling="", split="train", q=q, cue=(2, 3),
                   evidence=np.array([2]), z=rng.normal(size=(frames, 24)).astype(np.float32),
                   p=rng.normal(size=(frames, dim)).astype(np.float32), actions=actions,
                   classes=np.zeros(frames, np.int64),
                   prefix=np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, 0)]).astype(np.float32))


class TinyReader(torch.nn.Module):
    """Stands in for the frozen predictor: same score head and the same batch contract."""

    def __init__(self, d=D):
        super().__init__()
        self.score = torch.nn.Linear(2 * d, 1)
        self.out = torch.nn.Linear(d, 24)

    def forward(self, batch):
        history = batch["mask"] & (batch["type"] == 1)
        pooled = (batch["projected_z"] * history[..., None]).sum(1) / history.sum(1, keepdim=True).clamp_min(1)
        y = self.out(pooled + batch["projected_current"]) + batch["age"].sum(1, keepdim=True)
        return torch.stack([y, y + batch["actions"].sum((1, 2))[:, None]], 1)


def test_item_batch_rejects_frames_inside_the_recent_window():
    e = episode()
    batch = item_batch([e], [e.q], [[0, 3]], "cpu")
    assert batch["projected_z"].shape == (1, R + 2, D) and batch["type"][0].tolist() == [0] * R + [1, 1]
    assert batch["age"][0].tolist() == [3, 2, 1, 0, e.q, e.q - 3]
    with pytest.raises(ValueError):
        item_batch([e], [e.q], [[e.q - R + 1]], "cpu")


def test_pafu_teacher_groups_queries_before_at_and_after_each_frame():
    e, model = episode(), TinyReader()
    salience = pafu.teacher_salience(model, e, "cpu")
    assert salience.shape == (e.q + 1,) and np.isfinite(salience).all() and (salience > 0).all()
    # Frame 0 has no earlier query, so its score averages two groups; a middle frame averages three.
    from experiments.memory_baselines.gate_data import pooling_attention
    keys = torch.from_numpy(e.p[:e.q + 1])
    queries = torch.from_numpy(e.p[:e.q + 7])
    with torch.no_grad():
        a = pooling_attention(model, keys[None].expand(len(queries), -1, -1), queries).numpy()
    assert math.isclose(salience[0], (a[0, 0] + a[1:, 0].max()) / 2, rel_tol=1e-5)
    assert math.isclose(salience[5], (a[:5, 5].max() + a[5, 5] + a[6:, 5].max()) / 3, rel_tol=1e-5)


def test_pafu_policy_scores_each_frame_once_and_keeps_the_budget():
    e = episode(q=14)
    calls = []

    class Head(torch.nn.Module):
        def forward(self, x):
            calls.append(len(x))
            return x[:, 0]                       # score = first feature of the frame

    kept = pafu.select(Head(), [e], "cpu")[0]
    assert len(calls) == e.candidates and len(kept) == B and kept == sorted(kept)
    assert set(kept) == set(np.argsort(-e.p[:e.candidates, 0], kind="stable")[:B].tolist())


def test_compressor_history_never_reads_the_memory_queries():
    torch.manual_seed(0)
    c = pacc.Compressor(D, width=32, layers=2, heads=4).eval()
    p, past, age = torch.randn(2, 5, D), torch.randn(2, 5, 2), torch.rand(2, 5) * 8
    valid = torch.ones(2, 5, dtype=torch.bool)
    base = c.compress(p, past, age, valid)
    assert base.shape == (2, B, 32)
    # Padding must not change the memory, and each item decodes to (feature, action suffix, age >= 0).
    padded = c.compress(torch.cat([p, torch.randn(2, 3, D)], 1), torch.cat([past, torch.zeros(2, 3, 2)], 1),
                        torch.cat([age, torch.zeros(2, 3)], 1),
                        torch.cat([valid, torch.zeros(2, 3, dtype=torch.bool)], 1))
    assert torch.allclose(base, padded, atol=1e-5)
    item_p, item_past, item_age = c.items(base)
    assert item_p.shape == (2, B, D) and item_past.shape == (2, B, 2) and (item_age >= 0).all()


@pytest.mark.parametrize("variant", ["full", "recurrent"])
def test_compressor_trains_only_itself_and_gradients_pass_through_the_frozen_reader(variant):
    model = TinyReader().eval().requires_grad_(False)
    before = [p.clone() for p in model.parameters()]
    episodes = [episode(i) for i in range(6)]
    compressor, info = pacc.train(model, episodes, 17, "cpu", variant=variant, steps=8, batch=4, width=32)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    prediction, gap = pacc.predict(model, compressor, episodes, "cpu", variant=variant)
    assert prediction.shape == (6, 2, 24) and np.isfinite(prediction).all() and gap >= 0 and info["steps"] == 8


def test_memory_tokens_replace_only_the_history_slots():
    e = episode()
    recent = item_batch([e], [e.q], [[]], "cpu")
    out = with_memory_tokens(recent, torch.zeros(1, B, D), torch.zeros(1, B, 2), torch.ones(1, B))
    assert torch.equal(out["projected_z"][:, :R], recent["projected_z"]) and out["type"][0].tolist() == [0] * R + [1] * B


def test_framemorrow_teacher_and_loss_are_the_released_ones():
    history = torch.nn.functional.normalize(torch.randn(1, 4, 8), dim=-1)
    future = history[:, 2:3] * 3.0                                 # the continuation repeats history frame 2
    scores = teacher_scores(history, future)
    assert int(scores.argmax()) == 2
    assert float(ranking_loss(scores * 10, scores)["loss"]) < float(ranking_loss(-scores * 10, scores)["loss"])


def test_framemorrow_selection_respects_the_budget_and_the_causal_condition():
    episodes = [episode(i, q=10 + i) for i in range(4)]
    visual = {e.id: e.z for e in episodes}
    e = episodes[0]
    assert np.array_equal(fm._condition(e, e.q, "published"), e.actions[e.q:e.q + 6])
    assert np.array_equal(fm._condition(e, e.q, "online"), e.actions[e.q - 6:e.q])
    assert not fm._condition(e, 3, "online")[:3].any()             # nothing executed yet is left as zeros
    for variant, select in (("published", fm.select_published), ("online", fm.select_online)):
        selector, info = fm.train(episodes, visual, 0, "cpu", variant=variant, epochs=1, batch=8, hidden=32,
                                  layers=1, heads=4)
        kept = select(selector, episodes, visual, "cpu")
        assert all(len(k) == B and k == sorted(k) and max(k) <= ep.q - R for k, ep in zip(kept, episodes))
        assert info["held_out_pairs"] >= 1
    teacher = fm.select_teacher(episodes, visual, "cpu")
    assert all(len(k) == B for k in teacher)
