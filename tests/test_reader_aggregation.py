"""Contract tests for the reader-only aggregation ablation, without experiments."""
import numpy as np
import pytest
import torch

from experiments.phase2.reader_aggregation import (
    VARIANTS, INTERVENTIONS, DynamicsPredictor, build_reader, describe_reader, reader_attention,
)


def basis():
    rng = np.random.default_rng(19)
    return {'input_mean': rng.normal(size=12).astype('float32'),
            'projection': rng.normal(size=(12,8)).astype('float32')/4,
            'target_mean': rng.normal(size=12).astype('float32'),
            'output_basis': rng.normal(size=(12,6)).astype('float32')/4}


def model(variant):
    torch.manual_seed(13)
    return build_reader(12, {'hidden':16,'dropout':0.}, (3,6), basis(), variant).eval()


def batch():
    g = torch.Generator().manual_seed(11)
    return {'z':torch.randn(3,7,12,generator=g), 'current':torch.randn(3,12,generator=g),
            'mask':torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,1]],dtype=torch.bool),
            'type':torch.tensor([[0,0,0,0,1,1,1]]*3),
            'age':torch.tensor([[3,2,1,0,6,5,4.]]*3),
            'past_actions':torch.randn(3,7,2,generator=g),
            'actions':torch.randn(3,9,2,generator=g)}


def copy_batch(b):
    return {k:v.clone() for k,v in b.items()}


def test_original_checkpoint_and_forward_are_exact_legacy_parity():
    c={'hidden':16,'dropout':0.}; torch.manual_seed(13)
    legacy=DynamicsPredictor(12,c,(3,6),basis()).eval()
    original=model('original')
    assert legacy.state_dict().keys()==original.state_dict().keys()
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in legacy.state_dict().items())
    original.load_state_dict(legacy.state_dict(),strict=True)
    assert torch.equal(legacy(batch()),original(batch()))


@pytest.mark.parametrize('variant',VARIANTS+INTERVENTIONS)
def test_slot_permutation_preserves_prediction_and_padding_is_inert(variant):
    m,b=model(variant),batch(); expected=m(b)
    perm=torch.tensor([6,1,4,0,5,2,3]); changed=copy_batch(b)
    for k in ['z','mask','type','age','past_actions']: changed[k]=changed[k][:,perm]
    assert torch.allclose(expected,m(changed),atol=2e-5,rtol=2e-5)
    # Even nonfinite padding must be erased before projections/attention.
    for k in ['z','past_actions']:
        changed[k]=b[k].masked_fill(~b['mask'][...,None],float('nan'))
    changed['mask']=b['mask'];changed['type']=b['type']
    changed['age']=b['age'].masked_fill(~b['mask'],float('nan'))
    assert torch.equal(expected,m(changed))


@pytest.mark.parametrize('variant',VARIANTS+INTERVENTIONS)
def test_no_history_and_all_zero_masks_have_finite_zero_history_gradients(variant):
    m,b=model(variant),batch()
    b['mask'].zero_()
    for k in ['z','past_actions','age']: b[k].requires_grad_()
    out=m(b)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    for k in ['z','past_actions','age']:
        assert torch.equal(b[k].grad,torch.zeros_like(b[k]))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize('variant',VARIANTS+INTERVENTIONS)
def test_only_explicit_observed_inputs_and_requested_plan_horizon_are_used(variant):
    m,b=model(variant),batch(); expected=m(b)
    changed=copy_batch(b)
    changed.update(target=torch.full((3,2,12),1e9), roles=torch.ones(3,7),
                   future_observations=torch.full((3,100,12),float('nan')))
    changed['actions'][:,6:]=float('nan')
    assert torch.equal(expected,m(changed))
    # Current context and proposed within-horizon controls remain legal inputs.
    changed=copy_batch(b);changed['current']+=1
    assert not torch.equal(expected,m(changed))
    changed=copy_batch(b);changed['actions'][:,:6]+=1
    assert not torch.equal(expected,m(changed))


@pytest.mark.parametrize('variant',VARIANTS+INTERVENTIONS)
def test_output_hook_still_exposes_one_coefficient_vector_per_horizon(variant):
    m,b=model(variant),batch(); captured=[]
    hook=m.output.register_forward_hook(lambda _,args,out:captured.append(out))
    try: prediction=m(b)
    finally: hook.remove()
    assert len(captured)==2
    reconstructed=b['current'][:,None]+m.target_mean+torch.stack(captured,1)@m.output_basis.T
    assert torch.allclose(prediction,reconstructed,atol=1e-6,rtol=1e-6)


def test_uniform_path_and_actual_multihead_weights():
    b=batch();m=model('attention_mean');w=m.attention_weights(b)
    history=b['mask'] & (b['type']==1)
    assert w.shape==(3,2,7)
    assert torch.equal(w[:,1],history.float()/history.sum(1,keepdim=True).clamp_min(1))
    multi=model('multihead'); w=multi.attention_weights(b)
    assert w.shape==(3,4,7)
    assert torch.equal(w[0],torch.zeros_like(w[0]))
    assert torch.allclose(w[1:].sum(-1),torch.ones(2,4))
    assert not torch.allclose(w[2,0],w[2,1])


@pytest.mark.parametrize('variant',VARIANTS+INTERVENTIONS)
def test_attention_helper_matches_native_pooling_and_masks(variant):
    m,b=model(variant),batch()
    # Row 2 has three history entries and no padding.
    p=(b['z'][2:]-m.input_mean)@m.projection
    q=(b['current'][2:]-m.input_mean)@m.projection
    got=reader_attention(m,p[:,4:],q,b['age'][2:,4:],b['past_actions'][2:,4:])
    if hasattr(m,'attention_weights'):
        expected=m.attention_weights({k:v[2:] for k,v in b.items()})[:,:,4:].mean(1)
    else:
        logits=m.score(torch.cat([p[:,4:],p[:,4:]-q[:,None]],-1)).squeeze(-1)
        expected=torch.softmax(logits+torch.log1p((p[:,4:]-q[:,None]).square().mean(-1)),-1)
    assert torch.allclose(got,expected,atol=1e-6,rtol=1e-6)
    if variant=='item_attention':
        with pytest.raises(ValueError,match='observed ages'):
            reader_attention(m,p[:,4:],q)


def test_capacity_control_matches_largest_arm_and_preserves_nonoutput_modules():
    c={'hidden':128,'dropout':0.};desc={}
    for v in VARIANTS:
        torch.manual_seed(17)
        m=build_reader(128,c,(3,6),variant=v)
        desc[v]=describe_reader(m)
        if v=='original':original=m.state_dict()
        if v=='wide_control':
            assert all(torch.equal(t,original[k]) for k,t in m.state_dict().items()
                       if not k.startswith('output.'))
    target=max(desc[v]['trainable_parameters'] for v in VARIANTS[1:4])
    assert desc['wide_control']['capacity_target_parameters']==target
    assert abs(desc['wide_control']['capacity_relative_difference'])<.01
    assert desc['wide_control']['output_hidden_width']>128


def test_projected_cache_and_raw_projection_agree():
    for variant in VARIANTS+INTERVENTIONS:
        m,b=model(variant),batch();cached=copy_batch(b)
        cached['projected_z']=(b['z']-m.input_mean)@m.projection
        cached['projected_current']=(b['current']-m.input_mean)@m.projection
        assert torch.allclose(m(b),m(cached),atol=1e-6,rtol=1e-6)


def test_uniform_intervention_loads_original_and_recomputes_all_temporal_stats():
    original=model('original');uniform=model('uniform_intervention');b=batch()
    uniform.load_state_dict(original.state_dict(),strict=True)
    assert uniform.state_dict().keys()==original.state_dict().keys()
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in uniform.state_dict().items())
    assert 'uniform_intervention' not in VARIANTS
    assert describe_reader(uniform)['requires_original_checkpoint']
    captured=[]
    hook=uniform.output.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
    try: uniform(b)
    finally: hook.remove()
    p=(b['z']-uniform.input_mean)@uniform.projection
    current=(b['current']-uniform.input_mean)@uniform.projection
    past=uniform.past_action(b['past_actions'])
    history=b['mask'] & (b['type']==1)
    w=history.float()/history.sum(1,keepdim=True).clamp_min(1)
    age=b['age']/8; mean_age=(w*age).sum(1);center=age-mean_age[:,None]
    aligned=p+past;memory=(w[...,None]*aligned).sum(1)
    past_mean=(w[...,None]*past).sum(1)
    variance=(w*center.square()).sum(1)
    velocity=-(w[...,None]*center[...,None]*(aligned-memory[:,None])).sum(1)/variance.clamp_min(.01)[:,None]
    ac=b['actions'][:,:3]
    action=uniform.action(torch.cat([ac[:,0],ac.sum(1),ac[:,-1]],-1))
    transported=memory+velocity*(mean_age+3/8)[:,None]
    stats=torch.stack([mean_age,history.any(1).float(),torch.full_like(mean_age,3/8)],-1)
    expected=torch.cat([current,memory,velocity,transported,action,memory*action,
                        past_mean,velocity*action,stats],-1)
    assert torch.allclose(captured[0],expected,atol=1e-6,rtol=1e-6)
