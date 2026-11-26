"""Software checks for methods/deletion_cost/motion_v2.py on synthetic episodes (no measurement)."""
import numpy as np
import pytest
import torch

from bounded_memory.methods.deletion_cost import motion_v2 as dc
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
    model, norm, meta, final = dc.train(data, tables, eps[6:], assets, 17, "cpu", steps=20, batch=16, width=16,
                                 eval_every=10, log=lambda m: None)
    assert meta["selected_step"] in (0, 10, 20) and len(meta["history"]) == 3
    assert meta["history"][0]["alpha"] == 1.0
    best = min(meta["history"], key=lambda h: (h["dev_position_+3"] + h["dev_position_+6"]) / 2)
    assert meta["selected_step"] == best["step"]
    torch.testing.assert_close(model.alpha.detach(), model.alpha.detach())


def test_soft_continuation_cost_runs_from_the_minimum_to_the_mean():
    q, n = 9, 6
    rng = np.random.default_rng(0)
    table = rng.random((2, n, n))
    table = (table + np.swapaxes(table, 1, 2)) / 2
    table[:, np.arange(n), np.arange(n)] = np.inf
    hard = dc.continuation_cost(table, 6, (0, 1, 2), q)
    assert np.allclose(dc.continuation_cost(table, 6, (0, 1, 2), q, temperature=1e-6), hard, rtol=0, atol=1e-5)  # + T log(#pairs)
    mean = table.mean(0)
    for j, reach in enumerate(([1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 3, 4, 5])):
        sub = mean[np.ix_(reach, reach)][np.triu_indices(5, 1)]
        assert np.isclose(dc.continuation_cost(table, 6, (0, 1, 2), q, temperature=1e6)[j], sub.mean(), rtol=1e-4)
    soft = dc.continuation_cost(table, 6, (0, 1, 2), q, temperature=0.05)
    assert np.all(soft >= hard - 1e-12)


def test_rollout_records_its_own_full_pool_states(world):
    assets, eps = world
    kept, visited = dc.rollout(assets, eps, dc.ScorerV2(width=16), dc.identity_norm(), "cpu", record=True)
    assert kept == dc.rollout(assets, eps, dc.ScorerV2(width=16), dc.identity_norm(), "cpu")
    for ep in eps:
        states = visited[ep["rec"]["id"]]
        q = ep["rec"]["query"]
        assert len(states) == q - 5 and all(len(set(p)) == 3 and max(p) == c - 4 for c, p in states)


def test_state_metrics_and_variant_loop(world):
    assets, eps = world
    tables = {ep["rec"]["id"]: dc.pair_table(assets, ep) for ep in eps}
    fit, dev = eps[:6], eps[6:]
    behaviour = {ep["rec"]["id"]: dc.behaviour_states(assets, ep, 1) for ep in eps}
    base, dev_rows = (dc.labelled(assets, part, tables, behaviour, 0.01) for part in (fit, dev))
    m = dc.state_metrics(dc.ScorerV2(width=16), dc.identity_norm(), dev_rows, "cpu")
    assert m["states"] == len(dev_rows["cost"]) and 0 <= m["top1"] <= 1 and m["regret"] >= 0
    history, best = dc.fit_variant(assets, fit, dev, tables, base, dev_rows, behaviour, 0.01, 1, 17, "cpu", 20,
                                   log=lambda *_: None)
    assert [h["round"] for h in history] == [0, 1] and history[1]["states"] >= history[0]["states"]
    assert best.dev_mean == min(h["selected_dev_mean"] for h in history)
    assert best.round == [h["selected_dev_mean"] for h in history].index(best.dev_mean) and best.width == 256


def test_load_tables_names_what_is_missing(tmp_path, world):
    _, eps = world
    np.save(tmp_path / f"{eps[0]['rec']['id']}.npy", np.zeros((2, 3, 3)))
    assert set(dc.load_tables(tmp_path, eps[:1])) == {eps[0]["rec"]["id"]}
    with pytest.raises(FileNotFoundError, match="1 loss tables missing"):
        dc.load_tables(tmp_path, eps[:2])


def test_saved_scorer_loads_back_with_its_hash(tmp_path):
    model = dc.ScorerV2(width=16)
    best = dc.Best(0.01, 1, {k: v.clone() for k, v in model.state_dict().items()}, dc.identity_norm(), 16)
    digest = dc.save_scorer(tmp_path / "s.pt", best, variant="min")
    loaded, norm, again = dc.load_scorer(tmp_path / "s.pt", "cpu")
    assert digest == again and not list(tmp_path.glob("*.tmp"))
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))


def test_validation_is_read_only_after_a_holdout_gain_on_the_same_checkpoints():
    hashes = {17: "a", 29: "b", 43: "c"}
    holdout = {"variants": {
        "gain": {"paired_with_attention": {"ci95": [-0.002, -0.0001]}, "checkpoint_sha256": {str(k): v for k, v in hashes.items()}},
        "tie": {"paired_with_attention": {"ci95": [-0.002, 0.0003]}, "checkpoint_sha256": hashes}}}
    assert dc.validation_allowed(holdout, "gain", hashes)
    assert not dc.validation_allowed(holdout, "gain", {**hashes, 43: "retrained"})
    assert not dc.validation_allowed(holdout, "tie", hashes) and not dc.validation_allowed(holdout, "absent", hashes)
    assert not dc.validation_allowed({}, "gain", hashes)
