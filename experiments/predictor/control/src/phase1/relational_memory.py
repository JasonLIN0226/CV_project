"""Provisional causal history reader with masked attention pooling and relations.

PMA follows the design of Lee et al., Set Transformer (ICML 2019), official
MIT-licensed implementation https://github.com/juho-lee/set_transformer.
The adjacent observed-frame relation is an adaptation motivated by Santoro
et al., Relation Networks (2017), not a reproduction of those experiments.
No target, sibling identity, hidden simulator state or future image is read.
"""
import math
import torch
from torch import nn
from .common_anchor import recent_only_teacher_batch


class MaskedPMA(nn.Module):
    """Learned seed queries attend to every retained valid entry, with no Top-K."""
    def __init__(self, hidden=128, seeds=4, heads=4):
        super().__init__()
        self.heads, self.dim = heads, hidden // heads
        self.seeds = nn.Parameter(torch.empty(1, seeds, hidden))
        nn.init.xavier_uniform_(self.seeds)
        self.q, self.k, self.v = (nn.Linear(hidden, hidden) for _ in range(3))
        self.norm0, self.norm1 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, hidden))

    def forward(self, tokens, valid, context):
        n, length, hidden = tokens.shape
        if length == 0:
            return tokens.new_zeros((n, self.seeds.shape[1], hidden)), tokens.new_zeros((n, 0))
        query = self.seeds.expand(n, -1, -1) + context[:, None]
        split = lambda x: x.reshape(n, -1, self.heads, self.dim).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(tokens)), split(self.v(tokens))
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        weights = logits.masked_fill(~valid[:, None, None], -1e4).softmax(-1)
        weights = weights * valid[:, None, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (weights @ v).transpose(1, 2).reshape_as(query)
        x = self.norm0(query + pooled)
        x = self.norm1(x + self.ff(x))
        present = valid.any(1)[:, None, None]
        return x * present, weights.mean((1, 2))


class RelationalMemoryPredictor(nn.Module):
    """Frozen recent reference plus one direct, empty-history-gated correction.

    Pooling reads all retained history. Adjacent relations use observed visual
    differences and actual timestamp differences. Learned time inputs are
    bounded relative quantities. Linear latent transport supplies an explicit
    elapsed-time path; there is no unbounded learned absolute-age embedding.
    """
    def __init__(self, reference, basis, hidden=128):
        super().__init__()
        self.reference = reference.eval().requires_grad_(False)
        for key in ['input_mean', 'projection', 'output_basis']:
            self.register_buffer(key, torch.as_tensor(basis[key]).float().clone())
        dim = self.projection.shape[1]
        rank = self.output_basis.shape[1]
        self.frame = nn.Sequential(nn.Linear(2*dim+3, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.relation = nn.Sequential(nn.Linear(4*dim+4, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                      nn.Linear(hidden, hidden), nn.GELU())
        self.action = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.query = nn.Sequential(nn.Linear(dim+hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.frames_pool = MaskedPMA(hidden, seeds=4)
        self.relations_pool = MaskedPMA(hidden, seeds=2)
        self.head = nn.Sequential(nn.Linear(6*hidden+4*dim+2, 256), nn.GELU(),
                                  nn.LayerNorm(256), nn.Linear(256, rank))
        self.transport_skip = nn.Linear(dim, rank, bias=False)
        # Small nonzero readout allows gradients into the reader on step one.
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.transport_skip.weight, std=.001)

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, batch):
        recent = recent_only_teacher_batch(batch)  # validates strict whitelist and causality
        with torch.no_grad():
            anchor = self.reference(recent)
        valid = batch['mask']
        z = batch['z'].masked_fill(~valid[..., None], 0)
        p = (batch['projected_z'].masked_fill(~valid[..., None], 0) if 'projected_z' in batch
             else (z-self.input_mean) @ self.projection)
        current = (batch['projected_current'] if 'projected_current' in batch
                   else (batch['current']-self.input_mean) @ self.projection)
        history = valid & (batch['type'] == 1)
        age = batch['age'].masked_fill(~history, 0)
        past = batch['past_actions'].masked_fill(~history[..., None], 0)
        span = age.amax(1).clamp_min(1)
        relative_age = age / span[:, None]
        frames = self.frame(torch.cat([p, p-current[:, None], relative_age[..., None], past], -1))
        count = history.sum(1).clamp_min(1)
        mean = (p*history[..., None]).sum(1) / count[:, None]
        maximum = p.masked_fill(~history[..., None], -1e4).amax(1)
        minimum = p.masked_fill(~history[..., None], 1e4).amin(1)
        maximum = maximum * history.any(1)[:, None]
        minimum = minimum * history.any(1)[:, None]
        order = age.masked_fill(~history, -1).argsort(dim=1, descending=True, stable=True)
        gather = lambda x: x.gather(1, order[..., None].expand(-1, -1, x.shape[-1]))
        ordered_p, ordered_past = gather(p), gather(past)
        ordered_age = age.gather(1, order)
        ordered_valid = history.gather(1, order)
        dt = ordered_age[:, :-1] - ordered_age[:, 1:]
        relations_valid = ordered_valid[:, :-1] & ordered_valid[:, 1:] & (dt > 0)
        elapsed = dt.clamp_min(1)
        velocity = (ordered_p[:, 1:] - ordered_p[:, :-1]) / elapsed[..., None]
        action_rate = (ordered_past[:, :-1] - ordered_past[:, 1:]) / elapsed[..., None]
        outputs=[]
        for j,h in enumerate((3,6)):
            actions = batch['actions'][:, :h]
            if actions.shape[1:] != (h,2):
                raise ValueError('Incomplete planned-action prefix')
            context_action = self.action(torch.cat([actions[:,0],actions.sum(1),actions[:,-1],
                                                   actions.new_full((len(actions),1),h/8.)],-1))
            context = self.query(torch.cat([current,context_action],-1))
            pooled,_ = self.frames_pool(frames,history,context)
            transport = ordered_p[:,1:] + velocity * (ordered_age[:,1:]+h)[...,None]
            rel = self.relation(torch.cat([ordered_p[:,:-1],ordered_p[:,1:],velocity,transport,
                                          (elapsed/(elapsed+1))[...,None],
                                          (ordered_age[:,1:]/span[:,None])[...,None],action_rate],-1))
            rel_pool,rel_weights = self.relations_pool(rel,relations_valid,context)
            transported = (transport * rel_weights[...,None]).sum(1)
            features=torch.cat([pooled.flatten(1),rel_pool.flatten(1),current,mean,maximum,minimum,
                                history.any(1)[:,None].float(),torch.log1p(count.float())[:,None]],-1)
            coefficients=self.head(features)+self.transport_skip(transported-current)*relations_valid.any(1)[:,None]
            correction=coefficients @ self.output_basis.T
            outputs.append(anchor[:,j]+correction*history.any(1)[:,None])
        return torch.stack(outputs,1)
