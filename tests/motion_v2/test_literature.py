"""Software checks for environments/motion_v2/literature.py on tiny synthetic Motion-like episodes.

A real Phase 1 DynamicsPredictor (small) stands in for each frozen reader, and the unchanged
FrozenAssets.sample / .losses are called on a stand-in object, so the scoring path is the one the
burst run uses. Nothing here measures a method: it checks shapes, legality and the batch equivalence.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bounded_memory.environments.motion_v2 import literature, step34_core
from phase1.inputs import collate
from phase1.model import DynamicsPredictor

D, P, T = 5120, 4, 30   # FrozenAssets.losses is written for 5120-d I-JEPA features


class FakeAssets:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        basis = {"input_mean": np.zeros(D, np.float32), "projection": rng.normal(size=(D, P)).astype(np.float32),
                 "target_mean": np.zeros(D, np.float32),
                 "output_basis": rng.normal(size=(D, 8)).astype(np.float32) / 30}
        self.models = []
        for s in (17, 29, 43):
            torch.manual_seed(s)
            self.models.append(DynamicsPredictor(D, {"hidden": 8, "dropout": 0.0}, [3, 6], basis).eval())
        self.basis, self.device = basis, torch.device("cpu")
        self.decoder = SimpleNamespace(predict=lambda x: x[:, :2])
        self.collate = collate
        self.budget = lambda *a, **k: None

    def sample(self, ep, clock, kept, held_action=False):
        return step34_core.FrozenAssets.sample(self, ep, clock, kept, held_action)

    def losses(self, requests):
        return step34_core.FrozenAssets.losses(self, requests)


def _raw(i, assets, rng):
    q = int(rng.integers(12, T - 7))
    z = (rng.normal(size=(T, D)) / 30).astype(np.float32)
    actions = rng.normal(size=(T - 1, 2)).astype(np.float32)
    rec = {"id": f"e{i:03d}", "pair": f"p{i // 2:03d}", "query": q, "split": "validation"}
    meta = {"states": rng.normal(size=(T, 2)).astype(np.float32), "sufficient": rng.random(T) < 0.5}
    return {"rec": rec, "z": z, "p": z @ assets.basis["projection"], "actions": actions,
            "prefix": np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, 0)]), "meta": meta}


@pytest.fixture(scope="module")
def world():
    assets, rng = FakeAssets(), np.random.default_rng(1)
    raw = [_raw(i, assets, rng) for i in range(12)]          # 6 scene pairs
    return assets, raw, [literature.adapt(r) for r in raw]


def test_adapter_copies_no_evaluation_fields(world):
    _, raw, eps = world
    assert not hasattr(eps[0], "meta") and eps[0].q == raw[0]["rec"]["query"] and eps[0].candidates == eps[0].q - 3


def test_gate_and_motion_batches_give_the_same_predictions(world):
    assets, raw, eps = world
    kept = [[0, e.q - 4] for e in eps]
    assert literature.batch_equivalence(assets, raw, eps, kept) < 1e-5


def test_score_frames_shapes_and_legality(world):
    assets, raw, eps = world
    scores = literature.score_frames(assets, raw, [[e.q - 5, e.q - 4] for e in eps])
    assert scores["position"].shape == (3, 6, 2, 2) and scores["success"].shape == (6, 2)
    with pytest.raises(ValueError):
        literature.score_frames(assets, raw, [[e.q - 4, e.q - 3] for e in eps])   # q-3 is still recent


@pytest.mark.parametrize("variant", ["full", "recurrent"])
def test_compress_to_remember_trains_and_scores(world, variant):
    from bounded_memory.methods.pacc import gate as pacc
    assets, raw, eps = world
    compressor, meta = pacc.train(assets.models[0], eps, 17, "cpu", variant=variant, steps=3, batch=4, width=16)
    scores = literature.score_tokens(assets, compressor, raw, eps, variant)
    assert scores["position"].shape == (3, 6, 2, 2) and np.isnan(scores["success"]).all()
    assert literature.summarize({17: scores}, [17])["success_rate"] is None


def test_pafu_and_framemorrow_select_legal_pairs(world):
    from bounded_memory.methods.framemorrow import gate as fm
    from bounded_memory.methods.pafu_kv import gate as pafu
    assets, raw, eps = world
    head, _ = pafu.train(assets.models[0], eps, 17, "cpu", steps=5, batch=8)
    visual = {e.id: e.z for e in eps}
    selector, _ = fm.train(eps, visual, 17, "cpu", variant="online", epochs=1, hidden=32, layers=1)
    for kept in (pafu.select(head, eps, "cpu"), pafu.teacher_select(assets.models[0], eps, "cpu"),
                 fm.select_online(selector, eps, visual, "cpu"), fm.select_published(selector, eps, visual, "cpu")):
        scores = literature.score_frames(assets, raw, kept)
        summary = literature.summarize({17: scores, 29: scores}, [17, 29])
        assert 0 <= summary["success_rate"] <= 1 and len(summary["position_mse_+3_per_seed"]) == 2
