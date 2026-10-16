"""Software boundary fixtures only; no scientific images, training or outcome."""
import copy
import numpy as np
import torch
from phase1.model import DynamicsPredictor
from phase1.relational_memory import RelationalMemoryPredictor
from phase1.common_anchor import recent_only_teacher_batch

torch.set_num_threads(2)

def fixture():
    torch.manual_seed(19)
    dim,rank=16,8
    basis={'input_mean':np.zeros(dim,np.float32),'projection':np.eye(dim,rank,dtype=np.float32),
           'target_mean':np.zeros(dim,np.float32),'output_basis':np.eye(dim,rank,dtype=np.float32)}
    reference=DynamicsPredictor(dim,{'hidden':32,'dropout':0.},(3,6),basis)
    model=RelationalMemoryPredictor(reference,basis,hidden=32)
    z=torch.randn(2,8,dim)
    b={'z':z,'age':torch.tensor([[3.,2.,1.,0.,12.,7.,5.,0.]]).repeat(2,1),
       'type':torch.tensor([[0,0,0,0,1,1,1,0]]).repeat(2,1),
       'mask':torch.tensor([[1,1,1,1,1,1,1,0]],dtype=torch.bool).repeat(2,1),
       'past_actions':torch.randn(2,8,2),'actions':torch.randn(2,6,2),'current':z[:,3].clone()}
    return model,b

def test_empty_exact_even_after_update():
    m,b=fixture();recent=recent_only_teacher_batch(b)
    assert torch.equal(m(recent),m.reference(recent))
    old=copy.deepcopy(m.reference.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.001)
    m(b).square().mean().backward();opt.step()
    assert torch.equal(m(recent),m.reference(recent))
    assert all(torch.equal(v,m.reference.state_dict()[k]) for k,v in old.items())

def test_first_step_internal_gradients():
    m,b=fixture();m(b).square().mean().backward()
    assert float(m.frame[0].weight.grad.abs().sum())>0
    assert float(m.relation[0].weight.grad.abs().sum())>0
    assert all(p.grad is None for p in m.reference.parameters())

def test_prefix_causality_and_no_mutation():
    m,b=fixture();before=copy.deepcopy(b);a=m(b);b['actions'][:,3:]+=100
    assert torch.equal(a[:,0],m(b)[:,0])
    assert all(torch.equal(b[k],before[k]) for k in b if k!='actions')

def test_padding_nan_and_permutation():
    m,b=fixture();base=m(b)
    for k in ['z','past_actions']:
        b[k][:,-1]=float('nan')
    assert torch.allclose(base,m(b),atol=1e-6,rtol=0)
    # The inherited input contract keeps the recent FIFO chronological;
    # retained history is a set whose physical serialization may vary.
    order=torch.tensor([0,1,2,3,6,4,7,5]);p={k:(v[:,order] if k in ['z','mask','age','type','past_actions'] else v) for k,v in b.items()}
    assert torch.allclose(base,m(p),atol=2e-6,rtol=0)

def test_reject_future_and_target_and_single_trajectory():
    m,b=fixture();assert m({k:v[:1] for k,v in b.items()}).shape==(1,2,16)
    for kind in ['future','target']:
        c=copy.deepcopy(b)
        if kind=='future':c['age'][:,4]=-1
        else:c['target']=torch.randn(2,2,16)
        try:m(c)
        except ValueError:pass
        else:raise AssertionError('Rejected boundary accepted')

def test_observed_long_gap_is_finite():
    m,b=fixture();b['age'][:,4:7]+=32
    assert torch.isfinite(m(b)).all()
