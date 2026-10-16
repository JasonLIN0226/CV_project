"""Exact causal-boundary adapter fixtures, never scientific model training."""
import copy
import pytest
import torch
from phase1.common_anchor import recent_only_teacher_batch as current
from phase1.common_anchor_legacy import recent_only_teacher_batch as original


def batch(n, projected=True):
    generator = torch.Generator().manual_seed(901 + n)
    z = torch.randn(n, 11, 8, generator=generator)
    age = torch.tensor([3., 2., 1., 0., 4., 6., 8., 10., 12., 14., 16.]).expand(n, -1).clone()
    kind = torch.tensor([0] * 4 + [1] * 7).expand(n, -1).clone()
    values = {'z': z, 'age': age, 'type': kind, 'past_actions': torch.randn(n, 11, 2, generator=generator),
              'actions': torch.randn(n, 6, 2, generator=generator), 'current': z[:, 3].clone(),
              'mask': torch.ones(n, 11, dtype=torch.bool)}
    if projected:
        values['projected_z'] = torch.randn(n, 11, 3, generator=generator)
        values['projected_current'] = values['projected_z'][:, 3].clone()
    # History and recent entries can be interleaved; recent chronological order
    # must be preserved. Padding may contain NaN and must stay physically absent.
    for i in range(n):
        recent_positions = torch.randperm(11, generator=generator)[:4].sort().values.tolist()
        history_positions = [j for j in range(11) if j not in recent_positions]
        order = [0] * 11
        for j, position in enumerate(recent_positions): order[position] = j
        for j, position in enumerate(history_positions): order[position] = 4 + j
        for key in ('z', 'age', 'type', 'past_actions', 'mask', 'projected_z'):
            if key in values: values[key][i] = values[key][i, order].clone()
        invalid = history_positions[-1]
        values['mask'][i, invalid] = False
        for key in ('z', 'age', 'past_actions', 'projected_z'):
            if key in values: values[key][i, invalid] = float('nan')
    return values


@pytest.mark.parametrize('n,projected', [(1, False), (1, True), (4, True), (64, True)])
def test_exact_outputs_input_immutability_and_detached_history(n, projected):
    values = batch(n, projected)
    versions = {k: v._version for k, v in values.items()}
    old, new = original(values), current(values)
    assert set(old) == set(new)
    assert all(torch.equal(old[k], new[k]) and old[k].dtype == new[k].dtype for k in old)
    assert new['z'].shape == (n, 4, 8)
    assert all(not v.requires_grad for v in new.values())
    assert versions == {k: v._version for k, v in values.items()}


@pytest.mark.parametrize('problem', ['unknown_target', 'future_visual', 'noncontiguous_recent',
                                   'wrong_current', 'wrong_projected_current', 'empty_recent'])
def test_original_rejections_remain_rejections(problem):
    values = batch(4)
    recent = values['mask'] & (values['type'] == 0)
    i = int(torch.where(recent[0])[0][0])
    if problem == 'unknown_target': values['target'] = torch.zeros(4, 2, 8)
    elif problem == 'future_visual': values['age'][0, i] = -1
    elif problem == 'noncontiguous_recent': values['age'][0, i] = 2
    elif problem == 'wrong_current': values['current'][0, 0] += 1
    elif problem == 'wrong_projected_current': values['projected_current'][0, 0] += 1
    elif problem == 'empty_recent': values['mask'][0, recent[0]] = False
    for implementation in (original, current):
        with pytest.raises(ValueError): implementation(values)
