"""Synthetic stand-ins for the frozen Motion v2 assets: real (small) Phase 1 readers, the unchanged
FrozenAssets methods called on a stand-in object, and random Motion-like episodes."""
from types import SimpleNamespace

import numpy as np
import torch

from bounded_memory.environments.motion_v2 import step34_core
from phase1.inputs import collate
from phase1.model import DynamicsPredictor

D, P, T = 5120, 128, 30  # FrozenAssets.losses wants 5120-d features; causal_features wants a 128-d projection


class FakeAssets:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        basis = {"input_mean": np.zeros(D, np.float32), "projection": (rng.normal(size=(D, P)) / 70).astype(np.float32),
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

    def salience(self, candidates, anchors):
        return step34_core.FrozenAssets.salience(self, candidates, anchors)

    def attention_tables(self, ep):
        return step34_core.FrozenAssets.attention_tables(self, ep)


def raw_episode(i, assets, rng, split="validation"):
    q = int(rng.integers(12, T - 7))
    z = (rng.normal(size=(T, D)) / 30).astype(np.float32)
    actions = rng.normal(size=(T - 1, 2)).astype(np.float32)
    rec = {"id": f"e{i:03d}", "pair": f"p{i // 2:03d}", "query": q, "split": split}
    meta = {"states": rng.normal(size=(T, 2)).astype(np.float32), "sufficient": rng.random(T) < 0.5}
    return {"rec": rec, "z": z, "p": z @ assets.basis["projection"], "actions": actions,
            "prefix": np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, 0)]), "meta": meta}
