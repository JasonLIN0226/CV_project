"""Generic v5 real-video reader and train-only fixed feature bases."""
import hashlib
import numpy as np
import torch
from torch import nn


def canonical_columns(matrix):
    pivots = matrix.abs().argmax(dim=0)
    signs = matrix[pivots,torch.arange(matrix.shape[1],device=matrix.device)].sign()
    signs[signs==0]=1
    return matrix*signs


@torch.no_grad()
def feature_fit(train_observed, train_future_residuals, rank=128):
    """Caller supplies training rows only; validation/test never select a basis."""
    if train_observed.ndim!=2 or train_future_residuals.ndim!=2:
        raise ValueError('fit requires feature matrices')
    if train_observed.shape[1]!=train_future_residuals.shape[1] or rank>=min(train_observed.shape):
        raise ValueError('insufficient train-only dimensions')
    if not torch.isfinite(train_observed).all() or not torch.isfinite(train_future_residuals).all():
        raise ValueError('nonfinite training features')
    x=train_observed.double()
    mean=x.mean(0)
    scale=((x-mean).square().mean(0).sqrt()).clamp_min(1e-6)
    normalized=(x-mean)/scale
    vals,vecs=torch.linalg.eigh(normalized.T@normalized/len(x))
    vals,vecs=vals[-rank:].flip(0),canonical_columns(vecs[:,-rank:].flip(1))
    if vals.min()<=1e-8:raise ValueError('training input basis is rank deficient')
    input_basis=vecs/vals.sqrt()
    residual=train_future_residuals.double()/scale
    output_mean=residual.mean(0)
    centered=residual-output_mean
    out_vals,out_vecs=torch.linalg.eigh(centered.T@centered/len(centered))
    output_basis=canonical_columns(out_vecs[:,-rank:].flip(1))
    if out_vals[-rank:].min()<=1e-10:raise ValueError('training output basis is rank deficient')
    return {k:v.float() for k,v in {'mean':mean,'scale':scale,'input_basis':input_basis,
                'output_mean':output_mean,'output_basis':output_basis}.items()}


def normalize(features,fit):
    return (features-fit['mean'])/fit['scale']


def policy_contexts(length,policy,recording,episode,recent=8,capacity=16):
    from v5_retention import stream_prefix
    if policy not in ('recent','FIFO','Reservoir','full','Uniform'):
        raise ValueError('unknown policy')
    if length!=256 or recent!=8 or capacity not in (8,16,32):
        raise ValueError('real v5 stream specification changed')
    if policy=='recent':return tuple((tuple(range(max(0,t-recent+1),t+1)),()) for t in range(length))
    if policy=='full':
        return tuple((tuple(range(max(0,t-recent+1),t+1)),tuple(range(max(0,t-recent+1)))) for t in range(length))
    seed=int.from_bytes(hashlib.sha256(f'v5-real-retention-0|{recording}|{episode}|{policy}'.encode()).digest()[:8],'big')
    draws=np.random.default_rng(seed).random(length-recent)
    return tuple((r,h) for r,h,_ in stream_prefix(length,recent,capacity,policy,draws))


def context_payload(episode_features,current_step,context,episode_start_seconds=0.0):
    """Gather only legal retained/recent observations, carrying actual ages."""
    recent,history=context
    ids=tuple(history)+tuple(recent)
    if (tuple(sorted(set(ids)))!=ids or any(i<0 or i>current_step for i in ids)
            or set(recent)&set(history) or not recent or recent[-1]!=current_step):
        raise ValueError('future, duplicate, reindexed, or nonchronological evidence')
    observed=episode_features[list(ids)]
    times=episode_start_seconds+torch.tensor(ids,device=observed.device,dtype=observed.dtype)*0.5
    clock=episode_start_seconds+current_step*0.5
    age=(clock-times)/128.0
    role=torch.tensor([0]*len(history)+[1]*len(recent),device=observed.device,dtype=observed.dtype)
    return observed,torch.stack((age,role),dim=-1)


def pad_payloads(payloads):
    if not payloads:raise ValueError('empty observed batch')
    maximum=max(len(p[0]) for p in payloads)
    prototype=payloads[0][0]
    x=prototype.new_zeros((len(payloads),maximum,prototype.shape[-1]))
    temporal=prototype.new_zeros((len(payloads),maximum,2))
    mask=torch.ones((len(payloads),maximum),device=prototype.device,dtype=torch.bool)
    for i,(visual,time) in enumerate(payloads):
        x[i,:len(visual)]=visual;temporal[i,:len(visual)]=time;mask[i,:len(visual)]=False
    return x,temporal,mask


class CausalReader(nn.Module):
    def __init__(self,rank=128,hidden=256,layers=2,heads=4,ffn=1024,dropout=0.1):
        super().__init__()
        self.visual=nn.Linear(rank,hidden)
        self.time=nn.Sequential(nn.Linear(2,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.horizon=nn.Sequential(nn.Linear(1,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.query=nn.Parameter(torch.zeros(hidden))
        layer=nn.TransformerEncoderLayer(hidden,heads,ffn,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(hidden),enable_nested_tensor=False)
        self.output=nn.Linear(hidden,rank)

    def forward(self,observed,temporal,padding,horizons):
        if temporal.shape!=observed.shape[:2]+(2,) or padding.shape!=observed.shape[:2]:
            raise ValueError('observed/time mask dimensions disagree')
        if (temporal[~padding,0]<0).any():raise ValueError('future observation age')
        # Each horizon gets its own query and prefix. No future targets or labels
        # enter the reader; the real branch has no planned-action input.
        x=self.visual(observed)+self.time(temporal)
        query=self.query[None,None,:]+self.horizon(horizons[:,None,None]/64.0)
        x=torch.cat((x,query),dim=1)
        keys=torch.cat((padding,torch.zeros((len(x),1),device=x.device,dtype=torch.bool)),dim=1)
        causal=torch.triu(torch.ones((x.shape[1],x.shape[1]),device=x.device,dtype=torch.bool),diagonal=1)
        return self.output(self.encoder(x,mask=causal,src_key_padding_mask=keys)[:,-1])


def full_feature_prediction(coefficients,current_normalized,fit):
    return current_normalized+fit['output_mean']+coefficients@fit['output_basis'].T
