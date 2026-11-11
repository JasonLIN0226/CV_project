from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import scientific_hash
from .utils import code_hash, digest, file_hash, read_json, write_json, write_npz


def fit_basis(c, store, task):
    """Train-only visual bottleneck and future-residual basis; no simulator labels."""
    path = Path(c['output'])/'predictor'/task/'basis.npz'
    sig = digest([scientific_hash(c), store.ix['signature'], code_hash('model.py'),
                  file_hash(Path(c['output'])/'features/normalization.npz'), task])
    if path.exists():
        meta = read_json(path.with_suffix('.json'))
        if meta['signature'] != sig or meta['sha256'] != file_hash(path):
            raise ValueError('Predictor basis/config mismatch')
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    rows = [r for r in store.manifest['records'] if r['task'] == task and r['split'] == 'train']
    if any(a['sibling'] != b['id'] for a, b in zip(rows[::2], rows[1::2])):
        raise ValueError('Visual basis requires adjacent complete sibling pairs')
    visual, residual = [], []
    for rec in rows:
        z, q = store.features(rec), rec['query']
        visual.append(z[:q+1])
        residual.append(z[q+np.asarray(c['data']['horizons'])]-z[q])
    x = np.concatenate(visual).astype(np.float32)
    y = np.stack(residual).astype(np.float32)
    input_mean, target_mean = x.mean(0), y.mean((0, 1))
    # Paired future differences strengthen the task subspace without hidden-state supervision.
    out = np.concatenate([(y-target_mean).reshape(-1,y.shape[-1]),
                          (2*(y[::2]-y[1::2])).reshape(-1,y.shape[-1])])
    def pca(a, whiten=False):
        rank = min(c['model']['rank'], min(a.shape)-1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c['evaluation']['random_seed'])
            _, singular, basis = torch.pca_lowrank(torch.from_numpy(a), q=rank, center=False, niter=3)
        if whiten:
            basis = basis/(singular/np.sqrt(len(a))).clamp_min(.05)
        return basis.numpy()
    basis = {'input_mean': input_mean, 'projection': pca(x-input_mean, True),
             'target_mean': target_mean, 'output_basis': pca(out),
             'train_ids': np.array([r['id'] for r in rows])}
    write_npz(path, **basis)
    write_json(path.with_suffix('.json'), {'signature': sig, 'sha256': file_hash(path),
               'fit_split': 'train', 'task': task, 'episodes': len(rows),
               'input_rank': basis['projection'].shape[1], 'output_rank': basis['output_basis'].shape[1]})
    return basis


def project(z, basis):
    return ((z-basis['input_mean'])@basis['projection']).astype(np.float32)


class DynamicsPredictor(nn.Module):
    """Content-weighted temporal fit plus action interaction, shared across memory conditions."""
    def __init__(self, dim, config, horizons, basis=None):
        super().__init__()
        self.horizons = tuple(horizons)
        if basis is None:
            basis = {'input_mean': np.zeros(dim,np.float32), 'projection': np.eye(dim,dtype=np.float32),
                     'target_mean': np.zeros(dim,np.float32), 'output_basis': np.eye(dim,dtype=np.float32)}
        for key in ['input_mean','projection','target_mean','output_basis']:
            self.register_buffer(key, torch.from_numpy(basis[key].copy()))
        d, hidden = self.projection.shape[1], config['hidden']
        self.score = nn.Sequential(nn.Linear(2*d,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.action = nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.Linear(hidden,d))
        self.past_action = nn.Linear(2,d,bias=False)
        self.output = nn.Sequential(nn.Linear(8*d+3,hidden),nn.GELU(),nn.Dropout(config['dropout']),
                                    nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,self.output_basis.shape[1]))

    def forward(self, batch):
        valid = batch['mask']
        z = batch['z'].masked_fill(~valid[...,None],0)
        p = batch['projected_z'].masked_fill(~valid[...,None],0) if 'projected_z' in batch else (z-self.input_mean)@self.projection
        current = batch['projected_current'] if 'projected_current' in batch else (batch['current']-self.input_mean)@self.projection
        history = valid & (batch['type']==1)
        score = self.score(torch.cat([p,p-current[:,None]],-1)).squeeze(-1)
        # Content salience is observable, not an evidence-timestamp or hidden-state input.
        score = score+torch.log1p((p-current[:,None]).square().mean(-1))
        w = torch.softmax(score.masked_fill(~history,-1e4),1)*history
        w = w/w.sum(1,keepdim=True).clamp_min(1e-8)
        past = self.past_action(batch['past_actions'].masked_fill(~valid[...,None],0))
        aligned = p+past
        age = batch['age'].masked_fill(~valid,0)/8
        mean_age = (w*age).sum(1)
        center = age-mean_age[:,None]
        memory = (w[...,None]*aligned).sum(1)
        past_mean = (w[...,None]*past).sum(1)
        variance = (w*center.square()).sum(1)
        velocity = -(w[...,None]*center[...,None]*(aligned-memory[:,None])).sum(1)/variance.clamp_min(.01)[:,None]
        # Transport is linear in elapsed time; the learned readout maps it to future image features.
        outputs = []
        for h in self.horizons:
            actions = batch['actions'][:,:h]
            action = self.action(torch.cat([actions[:,0],actions.sum(1),actions[:,-1]],-1))
            time = mean_age+h/8
            transported = memory+velocity*time[:,None]
            stats = torch.stack([mean_age,history.any(1).float(),torch.full_like(mean_age,h/8)],-1)
            features = torch.cat([current,memory,velocity,transported,action,memory*action,past_mean,velocity*action,stats],-1)
            coeff = self.output(features)
            outputs.append(batch['current']+self.target_mean+coeff@self.output_basis.T)
        return torch.stack(outputs,1)
