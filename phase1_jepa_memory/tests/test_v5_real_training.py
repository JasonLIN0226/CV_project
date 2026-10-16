import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from phase1.v5_real_training import (HORIZONS, POLICIES, QUERY_STEPS, balanced_rows,
                                    full_batch_loss, make_batch, prepared_episodes,
                                    query_rows, training_fit, validate_cache_report)
from phase1.v5_real_reader import CausalReader


def episodes():
    torch.manual_seed(4)
    return [{'recording': s, 'participant': s, 'episode': 0, 'split': s,
             'features': torch.randn(256, 8)} for s in ('train', 'validation')]


def test_exact_query_grid_all_horizons_and_balanced_epoch_coverage():
    es = episodes()
    rows = query_rows(es, 'train')
    assert len(rows) == 38 and len(QUERY_STEPS) == 38
    assert all(t + max(HORIZONS) < 256 for _, t in rows)
    toy = tuple((0, i) for i in range(32))
    batches = list(balanced_rows(list(reversed(range(32))), toy))
    assert len(batches) == 2 and all(len(b) == 64 for b in batches)
    for p in POLICIES:
        assert sorted((ei, t) for b in batches for ei, t, policy in b if policy == p) == list(toy)
    with pytest.raises(ValueError):
        list(balanced_rows([0] * 32, toy))


def test_validation_poisoning_cannot_affect_train_fit_or_fit_ids():
    es = episodes()
    train = query_rows(es, 'train')
    a, pa = training_fit(es, train, rank=4)
    changed = copy.deepcopy(es)
    changed[1]['features'][:] = 1e8
    b, pb = training_fit(changed, train, rank=4)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert pa == pb
    assert pa['normalization_and_input_rows'] == 256
    assert pa['future_residual_rows'] == 38 * 4
    with pytest.raises(ValueError):
        training_fit(es, query_rows(es, 'validation'), rank=4)


def test_dropout_preserves_recent_current_clock_and_ignores_targets():
    es = episodes()
    fit, _ = training_fit(es, query_rows(es, 'train'), rank=4)
    prepared = prepared_episodes(es, fit)
    rows = ((0, 40, 'full'), (0, 40, 'recent'))
    a = make_batch(prepared, rows, True, np.random.default_rng(9))
    # 8 recent samples survive history dropout; current has zero age and role1.
    assert (~a[2][1]).sum() == 8
    for i in range(2):
        last = (~a[2][i]).sum().item() - 1
        torch.testing.assert_close(a[1][i, last], torch.tensor([0., 1.]))
    prepared[0]['normalized'][41:] += 10000
    b = make_batch(prepared, rows, True, np.random.default_rng(9))
    for x, y in zip(a[:4], b[:4]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert not torch.equal(a[4], b[4])


def test_literal_full_feature_four_horizon_loss_and_gradients():
    es = episodes()
    fit, _ = training_fit(es, query_rows(es, 'train'), rank=4)
    prepared = prepared_episodes(es, fit)
    model = CausalReader(rank=4, hidden=16, layers=2, heads=4, ffn=32, dropout=0)
    rows = tuple((0, 40, p) for p in POLICIES)
    batch = make_batch(prepared, rows, False, None)
    loss, slices = full_batch_loss(model, batch, fit)
    assert slices.shape == (4, 4)
    torch.testing.assert_close(loss, slices.mean())
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_partial_scientific_cache_cannot_become_training_input():
    with pytest.raises(ValueError, match='incomplete'):
        validate_cache_report({'status': 'RUNNING_FROZEN_SINGLE_FRAME_ENCODING'}, {}, 6980)
    with pytest.raises(ValueError, match='incomplete'):
        validate_cache_report({'status': 'ALL78_FROZEN_CAUSAL_FEATURES_COMPLETE_NOT_PHASE1_GO',
                               'job_id': '6980', 'frames_encoded': 512}, {}, 6980)
