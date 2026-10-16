import copy
import numpy as np
import torch
from phase1.v5_synthetic_reader import SyntheticCausalReader


def fixture():
    torch.manual_seed(17)
    basis = {'input_mean':np.zeros(8,np.float32),'projection':np.eye(8,4,dtype=np.float32),
             'target_mean':np.ones(8,np.float32)*.01,'output_basis':np.eye(8,4,dtype=np.float32)}
    batch = {'z':torch.randn(2,7,8),'age':torch.tensor([[3.,2.,1.,0.,10.,7.,0.]]*2),
             'type':torch.tensor([[0,0,0,0,1,1,0]]*2),'mask':torch.tensor([[1,1,1,1,1,1,0]]*2,dtype=torch.bool),
             'past_actions':torch.randn(2,7,2),'actions':torch.randn(2,6,2),'current':torch.randn(2,8)}
    return SyntheticCausalReader(basis).eval(), batch


def test_h3_cannot_see_h6_only_planned_controls():
    model, batch = fixture()
    original = model(batch)
    changed = copy.deepcopy(batch)
    changed['actions'][:,3:] = 1000
    prediction = model(changed)
    torch.testing.assert_close(original[:,0], prediction[:,0], rtol=0, atol=0)
    assert not torch.equal(original[:,1], prediction[:,1])


def test_padding_poison_and_reordering_preserve_actual_times():
    model, batch = fixture()
    original = model(batch)
    changed = copy.deepcopy(batch)
    for key in ('z','past_actions'):
        changed[key][:,-1] = 10000
    changed['age'][:,-1] = 10000
    changed['type'][:,-1] = 10000
    torch.testing.assert_close(original, model(changed), rtol=0, atol=0)
    order = torch.tensor([4,5,0,1,2,3,6])
    for key in ('z','past_actions','age','type','mask'):
        changed[key] = batch[key][:,order]
    torch.testing.assert_close(original, model(changed), rtol=0, atol=0)


def test_original_frozen_basis_and_full_space_output_are_preserved():
    model, batch = fixture()
    before = {k:model.state_dict()[k].clone() for k in ('input_mean','projection','target_mean','output_basis')}
    prediction = model(batch)
    assert prediction.shape == (2,2,8)
    prediction.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for key in before:
        torch.testing.assert_close(before[key],model.state_dict()[key],rtol=0,atol=0)
    # Dimensions outside the original output basis retain current+train mean.
    torch.testing.assert_close(prediction[:,:,4:],batch['current'][:,None,4:].expand(-1,2,-1)+model.target_mean[4:])
