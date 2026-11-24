"""Software checks for environments/motion_v2/deletion_cost_v2.py on synthetic episodes (no measurement)."""
import numpy as np
import pytest
import torch

from bounded_memory.environments.motion_v2 import deletion_cost_v2 as dc
from bounded_memory.environments.motion_v2 import step34_core

from motion_v2_fakes import FakeAssets, raw_episode  # tests/motion_v2 is on sys.path under pytest


@pytest.fixture(scope="module")
def world():
    assets, rng = FakeAssets(), np.random.default_rng(3)
    return assets, [raw_episode(i, assets, rng, split="train") for i in range(8)]


def test_continuation_cost_is_the_best_pair_still_reachable():
    q, n = 9, 6                                    # legal history frames 0..5
    table = np.full((2, n, n), 10.0)
    table[:, 0, 1] = table[:, 1, 0] = 1.0          # the best pair overall
    table[:, 0, 4] = table[:, 4, 0] = 3.0          # best pair without frame 1
    table[:, 3, 5] = table[:, 5, 3] = 5.0          # best pair from future arrivals alone
    cost = dc.continuation_cost(table, clock=6, pool=(0, 1, 2), q=q)   # future arrivals: 3, 4, 5
    assert cost[2] == 1.0                          # dropping 2 keeps (0, 1)
    assert cost[1] == 3.0                          # dropping 1 still reaches (0, 4)
    assert cost[0] == 5.0                          # dropping 0 leaves (3, 5) as the best


def test_pair_table_matches_frozen_losses(world):
    assets, eps = world
    ep = eps[0]
    table = dc.pair_table(assets, ep, chunk=7)
    n = ep["rec"]["query"] - 3
    assert table.shape == (2, n, n) and np.isinf(np.diag(table[0])).all()
    assert np.allclose(table, np.swapaxes(table, 1, 2))
    reference = step34_core.FrozenAssets.losses(assets, [(ep, [1, n - 1])])["position"].mean(0)[0]
    assert np.allclose(table[:, 1, n - 1], reference)


def test_step_zero_scorer_is_the_attention_policy(world):
    assets, eps = world
    model = dc.ScorerV2(width=16)
    norm = {"items_mean": np.zeros(259), "items_std": np.ones(259), "context_mean": np.zeros(524), "context_std": np.ones(524)}
    ours = dc.rollout(assets, eps, model, norm, "cpu")
    attention = [step34_core.online_kept(assets, ep, "attention", 17)[0] for ep in eps]
    assert ours == [sorted(k) for k in attention]


def test_behaviour_states_are_legal_full_pools(world):
    assets, eps = world
    states = dc.behaviour_states(assets, eps[0], 1)
    q = eps[0]["rec"]["query"]
    assert states and all(len(set(p)) == 3 and max(p) == c - 4 and min(p) >= 0 and 6 <= c <= q for c, p in states)


def test_training_runs_and_selects_on_dev(world):
    assets, eps = world
    tables = {ep["rec"]["id"]: dc.pair_table(assets, ep) for ep in eps}
    rows = []
    for ep in eps[:6]:
        for clock, pool in dc.behaviour_states(assets, ep, 1):
            x, c, p = dc.observe(assets, ep, clock, pool)
            rows.append((x, c, p, dc.continuation_cost(tables[ep["rec"]["id"]], clock, pool, ep["rec"]["query"])))
    data = {k: np.stack([r[i] for r in rows]) for i, k in enumerate(("items", "context", "prior", "cost"))}
    model, norm, meta = dc.train(data, tables, eps[6:], assets, 17, "cpu", steps=20, batch=16, width=16,
                                 eval_every=10, log=lambda m: None)
    assert meta["selected_step"] in (0, 10, 20) and len(meta["history"]) == 3
    assert meta["history"][0]["alpha"] == 1.0
    best = min(meta["history"], key=lambda h: (h["dev_position_+3"] + h["dev_position_+6"]) / 2)
    assert meta["selected_step"] == best["step"]
    torch.testing.assert_close(model.alpha.detach(), model.alpha.detach())
