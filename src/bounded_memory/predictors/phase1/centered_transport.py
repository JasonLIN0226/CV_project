"""Provisional translation-invariant correction before native diagnostic review.

Last-observation centering is adapted from NLinear, Zeng et al., 2205.13504.
Content attention follows learned query pooling, not the complete NLinear or
Set Transformer architecture. All supplied history is read; no TopK selector.
"""
import math
import torch
from torch import nn
from .common_anchor import recent_only_teacher_batch

class ContentWeights(nn.Module):
    def __init__(self,inputs,queries,hidden=128,heads=4):
        super().__init__();self.heads=heads;self.width=hidden//heads
        self.keys=nn.Sequential(nn.Linear(inputs,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,hidden))
        self.queries=nn.Parameter(torch.empty(queries,hidden));nn.init.xavier_uniform_(self.queries)
        self.action=nn.Sequential(nn.Linear(7,hidden),nn.GELU(),nn.Linear(hidden,hidden))
    def forward(self,x,valid,actions):
        n,length,_=x.shape
        q=self.queries[None]+self.action(actions)[:,None]
        q=q.reshape(n,-1,self.heads,self.width).transpose(1,2)
        k=self.keys(x).reshape(n,length,self.heads,self.width).transpose(1,2)
        logits=(q@k.transpose(-2,-1))/math.sqrt(self.width)
        w=logits.masked_fill(~valid[:,None,None],-1e4).softmax(-1)*valid[:,None,None]
        return (w/w.sum(-1,keepdim=True).clamp_min(1e-8)).mean(1)

class CenteredTransportPredictor(nn.Module):
    """Frozen recent prediction plus centered, content-weighted linear values.

    Learned keys never receive an absolute visual level or query-to-history
    gap. Actual elapsed time enters only the explicit linear transport value.
    This makes the history correction invariant to a common additive visual
    offset and avoids feeding extrapolated feature magnitudes into the keys.
    """
    def __init__(self,reference,basis,hidden=128):
        super().__init__();self.reference=reference.eval().requires_grad_(False)
        for k in ('input_mean','projection','output_basis'):
            self.register_buffer(k,torch.as_tensor(basis[k]).float().clone())
        dim=self.projection.shape[1];rank=self.output_basis.shape[1]
        self.frames=ContentWeights(dim+3,4,hidden)
        self.relations=ContentWeights(3*dim+4,2,hidden)
        self.frame_values=nn.Linear(4*dim,rank,bias=False)
        self.transport_values=nn.Linear(2*dim,rank,bias=False)
        for layer in (self.frame_values,self.transport_values):nn.init.normal_(layer.weight,std=.001)
    def train(self,mode=True):
        super().train(mode);self.reference.eval();return self
    def forward(self,batch):
        recent=recent_only_teacher_batch(batch)
        with torch.no_grad():anchor=self.reference(recent)
        valid=batch['mask'];hist=valid&(batch['type']==1)
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
        rposition=position.gather(1,order)[:,1:]
        rel_key=torch.cat([d[:,:-1],d[:,1:],slope,(dt.clamp_min(0)/(1+dt.clamp_min(0)))[...,None],rposition[...,None],action_rate],-1)
        outputs=[]
        for j,h in enumerate((3,6)):
            u=batch['actions'][:,:h]
            if u.shape[1:]!=(h,2):raise ValueError('Complete legal future action prefix required')
            context=torch.cat([u[:,0],u.sum(1),u[:,-1],u.new_full((len(u),1),h/8.)],-1)
            fw=self.frames(frame_key,hist,context)
            frames=fw@delta
            rw=self.relations(rel_key,rv,context)
            transported=(d[:,1:]+slope*(a[:,1:]+h)[...,None]).masked_fill(~rv[...,None],0)
            transport=rw@transported
            coeff=self.frame_values(frames.flatten(1))+self.transport_values(transport.flatten(1))
            outputs.append(anchor[:,j]+(coeff@self.output_basis.T)*hist.any(1)[:,None])
        return torch.stack(outputs,1)
