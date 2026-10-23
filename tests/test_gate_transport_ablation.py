"""Actual frozen forward on toy arrays: software checks, not scientific evidence."""
import sys,importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'experiments/predictor/control/src'))
from phase1.centered_transport import CenteredTransportPredictor
sp=importlib.util.spec_from_file_location('intervention',R/'experiments/gate_transport_ablation/intervention.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
class Reference(nn.Module):
 def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.3))
 def forward(self,b):return b['current'][:,None].expand(-1,2,-1)*self.scale

def fixture(empty=False):
 torch.manual_seed(2);dim=8;pr=3;rank=4
 basis=dict(input_mean=np.zeros(dim,np.float32),projection=np.eye(dim,pr,dtype=np.float32),output_basis=np.eye(dim,rank,dtype=np.float32))
 model=CenteredTransportPredictor(Reference(),basis).eval().requires_grad_(False)
 z=torch.randn(2,6 if not empty else 4,dim);n,L,_=z.shape
 b=dict(z=z,mask=torch.ones(n,L,dtype=torch.bool),age=torch.tensor([[3.,2,1,0,12,5]] if not empty else [[3.,2,1,0]]).repeat(n,1),type=torch.tensor([[0,0,0,0,1,1]] if not empty else [[0,0,0,0]]).repeat(n,1),past_actions=torch.zeros(n,L,2),actions=torch.zeros(n,6,2),current=z[:,3])
 return model,b

def test_only_transport_projection_removed_actual_architecture():
 model,b=fixture();d=m.state_digest(model);pp,res=m.both(model,b)
 assert res<3e-5 and np.max(np.abs(pp['before']-pp['transport_off']))>0
 assert m.state_digest(model)==d and all(p.grad is None for p in model.parameters())
 assert all(not layer._forward_hooks for layer in (model.frame_values,model.transport_values,model.reference))

def test_empty_history_exact_and_hooks_cleaned_on_exception():
 model,b=fixture(True);pp,_=m.both(model,b);assert np.array_equal(pp['before'],pp['transport_off'])
 bad=dict(b,actions=torch.zeros(2,1,2))
 with pytest.raises(ValueError):m.traced_forward(model,bad,True)
 assert all(not layer._forward_hooks for layer in (model.frame_values,model.transport_values,model.reference))
