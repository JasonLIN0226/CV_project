"""Focused input, symmetry and supervision contracts; no study training."""
import numpy as np
import pytest
import torch
import hashlib
import json
from pathlib import Path

from experiments.phase2 import fit_terminal_students as students

from experiments.phase2.fit_terminal_students import (
    TerminalScorer, architecture_plan, fit_normalization, normalize_inputs,
    soft_targets, stable_parent_permutations, training_label_scales,
    validate_inputs, _validation,
)


def sample_inputs():
    rng = np.random.default_rng(17)
    return {'items': rng.normal(size=(5, 3, 7)).astype(np.float32),
            'context': rng.normal(size=(5, 4)).astype(np.float32)}


@pytest.mark.parametrize('extra', ['future_actions', 'roles', 'scene_id', 'query_target'])
def test_input_whitelist_rejects_metadata_and_future_fields(extra):
    x = sample_inputs()
    x[extra] = np.zeros((5, 1), np.float32)
    with pytest.raises(ValueError, match='exactly items and context'):
        validate_inputs(x)


def test_independent_score_cannot_access_other_candidate_items():
    torch.manual_seed(17)
    model = TerminalScorer(7, 4, 'independent', 12)
    x = sample_inputs()
    a = torch.from_numpy(x['items'])
    context = torch.from_numpy(x['context'])
    before = model(a, context)
    changed = a.clone()
    changed[:, 1:] += 10.
    after = model(changed, context)
    assert torch.equal(before[:, 0], after[:, 0])
    assert not torch.equal(before[:, 1:], after[:, 1:])


@pytest.mark.parametrize('architecture', ['independent', 'set'])
def test_scores_are_candidate_permutation_equivariant(architecture):
    torch.manual_seed(29)
    model = TerminalScorer(7, 4, architecture, 12)
    x = sample_inputs()
    items, context = torch.from_numpy(x['items']), torch.from_numpy(x['context'])
    order = [2, 0, 1]
    assert torch.allclose(model(items[:, order], context), model(items, context)[:, order], atol=1e-6, rtol=1e-6)


def test_normalization_fits_only_train_and_does_not_adapt_to_new_inputs():
    train = sample_inputs()
    normalization = fit_normalization(train)
    saved = {k: v.copy() for k, v in normalization.items()}
    future = {k: v+1000 for k, v in train.items()}
    transformed = normalize_inputs(future, normalization)
    assert all(np.array_equal(saved[k], normalization[k]) for k in saved)
    assert np.max(np.abs(transformed['items'])) > 100
    assert np.allclose(normalization['items_mean'], train['items'].mean((0, 1)), atol=1e-7)


def test_actual_parameters_match_and_all_are_in_forward_graph():
    plan = architecture_plan(259, 524)
    assert plan['independent']['parameters'] == 54401
    assert plan['set']['parameters'] == 54361
    assert plan['relative_parameter_gap'] < .05
    for arch in ('independent', 'set'):
        model = TerminalScorer(259, 524, arch, plan[arch]['width'])
        assert sum(p.numel() for p in model.parameters()) == plan[arch]['parameters']
        model(torch.randn(2, 3, 259), torch.randn(2, 524)).square().sum().backward()
        assert all(p.grad is not None for p in model.parameters())


def test_label_direction_and_train_scales():
    cost = np.array([[.1, .4, .8], [.7, .2, .5]])
    attention = np.array([[.6, .3, .1], [.2, .3, .5]])
    scales = training_label_scales(cost, attention)
    targets, _ = soft_targets(cost, attention, ['a', 'b'], scales, 20261108)
    assert np.array_equal(targets['cost'].argmax(1), cost.argmin(1))
    assert np.array_equal(targets['future_attention'].argmax(1), attention.argmin(1))
    assert scales['shuffle'] == scales['cost']
    assert all(np.allclose(x.sum(1), 1.) for x in targets.values())
    # Constant labels remain uniform, rather than fabricated hard winners.
    zero = np.ones((2, 3))
    uniform, _ = soft_targets(zero, zero, ['a', 'b'], training_label_scales(zero, zero), 20261108)
    assert all(np.allclose(x, 1/3) for x in uniform.values())


def test_shuffled_labels_are_fixed_per_parent_not_epoch_or_order():
    ids = ['parent-1', 'parent-2', 'parent-3', 'parent-4']
    first = stable_parent_permutations(ids, 20261108)
    second = stable_parent_permutations(ids, 20261108)
    order = [2, 0, 3, 1]
    reordered = stable_parent_permutations([ids[i] for i in order], 20261108)
    assert np.array_equal(first, second)
    assert np.array_equal(reordered, first[order])
    assert all(sorted(row.tolist()) == [0, 1, 2] for row in first)
    costs = np.arange(12).reshape(4, 3)
    scales = training_label_scales(costs, costs)
    targets, permutation = soft_targets(costs, costs, ids, scales, 20261108)
    assert np.allclose(targets['shuffle'], np.take_along_axis(targets['cost'], permutation, axis=1))


def test_validation_selection_metric_does_not_depend_on_true_cost_report():
    torch.manual_seed(43)
    model = TerminalScorer(7, 4, 'independent', 12)
    x = {k: torch.from_numpy(v) for k, v in sample_inputs().items()}
    target = torch.softmax(torch.randn(5, 3), dim=-1)
    first = _validation(model, x, target, np.zeros((5, 3)), 3)
    opposite_cost = np.arange(15).reshape(5, 3)*1000.
    second = _validation(model, x, target, opposite_cost, 3)
    assert first['own_label_ce'] == second['own_label_ce']
    assert first['report_only_true_cost'] != second['report_only_true_cost']


def test_frozen_prediction_roundtrip_needs_no_labels_or_ids(tmp_path):
    """Random untrained weights exercise checkpoint IO, not experiment fitting."""
    x = sample_inputs()
    norm = fit_normalization(x)
    np.savez(tmp_path/'normalization.npz', **norm)
    torch.manual_seed(17)
    model = TerminalScorer(7, 4, 'independent', 12)
    spec = {'item_dimension': 7, 'context_dimension': 4, 'architecture': 'independent', 'width': 12}
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    torch.save({'model': model.state_dict(), 'spec': spec, 'normalization_file': 'normalization.npz',
                'normalization_sha256': digest(tmp_path/'normalization.npz')}, tmp_path/'random.pt')
    prefit = {'source_sha256': digest(students.__file__), 'config': {'cpu_threads': 1}}
    (tmp_path/'prefit_manifest.json').write_text(json.dumps(prefit))
    manifest = {'prefit_manifest_sha256': digest(tmp_path/'prefit_manifest.json'),
                'arms': {'random': {'checkpoint': 'random.pt', 'checkpoint_sha256': digest(tmp_path/'random.pt'), 'architecture': 'independent'}}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    result = students.predict_students(tmp_path, x, batch_size=2, profile_parents=2)
    assert result['random']['logits'].shape == (5, 3)
    assert result['random']['drop'].shape == (5,)
    assert result['random']['timing']['total_forward_examples_including_profile'] == 12
    assert result['random']['timing']['normalization_numeric_bytes'] == 2*(7+4)*4
    assert not (tmp_path/'labels.npz').exists()
    assert not (tmp_path/'ids.json').exists()
