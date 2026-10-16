import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from phase1.v5_real_reader import (CausalReader,context_payload,pad_payloads,policy_contexts,
                                  feature_fit,normalize,full_feature_prediction)


def reader():
    torch.manual_seed(17)
    return CausalReader(rank=4,hidden=16,layers=2,heads=4,ffn=32,dropout=0).eval()


def test_future_features_cannot_change_past_query():
    torch.manual_seed(3);z=torch.randn(256,4);changed=z.clone();changed[41:]=10000
    contexts=policy_contexts(256,'Reservoir','fixture',0)
    a=pad_payloads([context_payload(z,40,contexts[40])]);b=pad_payloads([context_payload(changed,40,contexts[40])]);m=reader()
    torch.testing.assert_close(m(*a,torch.tensor([64.])),m(*b,torch.tensor([64.])),rtol=0,atol=0)
    with pytest.raises(ValueError):context_payload(z,40,((39,40),(41,)))


def test_padding_values_and_padding_length_do_not_change_prediction():
    z=torch.randn(256,4);c=policy_contexts(256,'recent','fixture',0)[40]
    payload=context_payload(z,40,c);a=pad_payloads([payload]);long=pad_payloads([payload,context_payload(z,40,policy_contexts(256,'full','fixture',0)[40])])
    long[0][0,len(payload[0]):]=10000;long[1][0,len(payload[0]):]=10000
    m=reader();p=m(*a,torch.tensor([16.]));q=m(*long,torch.tensor([16.,16.]))
    torch.testing.assert_close(p,q[:1],rtol=1e-5,atol=1e-5)


def test_deletion_keeps_actual_ages_and_future_ids_rejected():
    z=torch.randn(256,4)
    observed,metadata=context_payload(z,40,((33,34,35,36,37,38,39,40),(1,7,20)),128)
    assert observed.shape==(11,4)
    torch.testing.assert_close(metadata[:3,0],torch.tensor([39,33,20])/256)
    with pytest.raises(ValueError):context_payload(z,40,((33,34,40),(20,7)))


def test_fit_uses_supplied_training_rows_and_full_feature_loss():
    torch.manual_seed(1);x=torch.randn(30,8);r=torch.randn(40,8);fit=feature_fit(x,r,rank=4)
    torch.testing.assert_close(fit['mean'],x.mean(0))
    torch.testing.assert_close(fit['output_basis'].T@fit['output_basis'],torch.eye(4),rtol=1e-5,atol=1e-5)
    coeff=torch.randn(3,4,requires_grad=True);current=normalize(x[:3],fit)
    pred=full_feature_prediction(coeff,current,fit)
    assert pred.shape==(3,8)
    target=torch.randn(3,8);loss=(pred-target).square().mean();loss.backward()
    expected=2*(pred.detach()-target)@fit['output_basis']/pred.numel()
    torch.testing.assert_close(coeff.grad,expected)


def test_role_and_horizon_legal_inputs_change_the_reader():
    z=torch.randn(256,4);payload=context_payload(z,40,policy_contexts(256,'recent','fixture',0)[40]);a=pad_payloads([payload]);m=reader()
    p=m(*a,torch.tensor([1.]));q=m(*a,torch.tensor([64.]));assert not torch.equal(p,q)
