"""Provisional visual latent-flow residual with action-conditioned decoding.

Borrowed principles: nonlinear dynamical coordinates (E2C, 1506.07365;
Lusch et al.,1712.09707) and featurewise conditioning (FiLM,1709.07871).
This is an adaptation, not a reproduction or claim of learned physical state.
"""
import torch
from torch import nn
from .common_anchor import recent_only_teacher_batch
from .centered_transport import ContentWeights


class LatentFlowPredictor(nn.Module):
    def __init__(self,reference,basis,hidden=128):
        super().__init__();self.reference=reference.eval().requires_grad_(False)
        for k in ('input_mean','projection','output_basis'):
            self.register_buffer(k,torch.as_tensor(basis[k]).float().clone())
        dim=self.projection.shape[1];rank=self.output_basis.shape[1];state=16
        self.frames=ContentWeights(dim+3,4,hidden)
        self.relations=ContentWeights(3*dim+4,2,hidden)
        self.coordinates=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,state))
        self.value_layer=nn.Linear(6*state,hidden)
        self.condition=nn.Linear(7,2*hidden)
        self.decoder=nn.Sequential(nn.GELU(),nn.Linear(hidden,rank))
        nn.init.normal_(self.decoder[-1].weight,std=.001)
        nn.init.zeros_(self.decoder[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.reference.eval();return self

    def forward(self,batch):
        with torch.no_grad():anchor=self.reference(recent_only_teacher_batch(batch))
        hist=batch['mask']&(batch['type']==1)
        if 'projected_z' in batch:
            delta=batch['projected_z']-batch['projected_current'][:,None]
        else:delta=(batch['z']-batch['current'][:,None])@self.projection
        delta=delta.masked_fill(~hist[...,None],0)
        age=batch['age'].masked_fill(~hist,0)
        nearest=age.masked_fill(~hist,1e6).amin(1)*hist.any(1)
        span=(age.amax(1)-nearest).clamp_min(1)
        position=((age-nearest[:,None])/span[:,None]).masked_fill(~hist,0)
        past=batch['past_actions'].masked_fill(~hist[...,None],0)
        frame_key=torch.cat([delta,position[...,None],past/(1+age[...,None])],-1)
        order=age.masked_fill(~hist,-1).argsort(dim=1,descending=True,stable=True)
        gather=lambda x:x.gather(1,order[...,None].expand_as(x))
        d=gather(delta);a=age.gather(1,order);vp=hist.gather(1,order);pa=gather(past)
        dt=a[:,:-1]-a[:,1:];rv=vp[:,:-1]&vp[:,1:]&(dt>0)
        slope=(d[:,1:]-d[:,:-1])/dt.clamp_min(1)[...,None]
        action_rate=(pa[:,:-1]-pa[:,1:])/dt.clamp_min(1)[...,None]
        rel_key=torch.cat([d[:,:-1],d[:,1:],slope,(dt.clamp_min(0)/(1+dt.clamp_min(0)))[...,None],position.gather(1,order)[:,1:,None],action_rate],-1)
        # Subtract the shared origin to retain an exact zero-evidence boundary.
        origin=self.coordinates(torch.zeros_like(delta))
        state=(self.coordinates(delta)-origin).masked_fill(~hist[...,None],0)
        ordered=gather(state)
        velocity=(ordered[:,1:]-ordered[:,:-1])/dt.clamp_min(1)[...,None]
        outputs=[]
        for j,h in enumerate((3,6)):
            u=batch['actions'][:,:h]
            if u.shape[1:]!=(h,2):raise ValueError('Complete legal future action prefix required')
            context=torch.cat([u[:,0],u.sum(1),u[:,-1],u.new_full((len(u),1),h/8.)],-1)
            frames=self.frames(frame_key,hist,context)@state
            transported=(ordered[:,1:]+velocity*(a[:,1:]+h)[...,None]).masked_fill(~rv[...,None],0)
            transport=self.relations(rel_key,rv,context)@transported
            value=torch.cat([frames.flatten(1),transport.flatten(1)],-1)
            scale,shift=self.condition(context).chunk(2,-1)
            scale=1+.5*scale.tanh()
            decode=lambda x:self.decoder(self.value_layer(x)*scale+shift)
            coeff=decode(value)-decode(torch.zeros_like(value))
            outputs.append(anchor[:,j]+(coeff@self.output_basis.T)*hist.any(1)[:,None])
        return torch.stack(outputs,1)
