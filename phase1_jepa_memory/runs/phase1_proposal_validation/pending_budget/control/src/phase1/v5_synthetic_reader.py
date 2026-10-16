"""V5 temporal reader over the unchanged original synthetic input boundary.

No new features, basis fitting, evidence policy, or hand-written physics are
introduced. The original legal cumulative past controls accompany each visual
observation; each query sees exactly its own planned-control prefix.
"""
import torch
from torch import nn


class SyntheticCausalReader(nn.Module):
    def __init__(self, basis, horizons=(3, 6)):
        super().__init__()
        if tuple(horizons) != (3, 6):
            raise ValueError('original synthetic horizons changed')
        self.horizons = tuple(horizons)
        for key in ('input_mean', 'projection', 'target_mean', 'output_basis'):
            self.register_buffer(key, torch.as_tensor(basis[key]).float().clone())
        rank = self.projection.shape[1]
        out_rank = self.output_basis.shape[1]
        if rank != out_rank:
            raise ValueError('original matched input/output basis rank required')
        self.visual = nn.Linear(rank, 256)
        self.past_control = nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256))
        self.future_control = nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256))
        self.time = nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256))
        self.horizon = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Linear(256, 256))
        self.query = nn.Parameter(torch.zeros(256))
        layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(256), enable_nested_tensor=False)
        self.output = nn.Linear(256, out_rank)

    def forward(self, batch):
        valid = batch['mask']
        if not valid.any(1).all() or (batch['age'][valid] < 0).any():
            raise ValueError('future or missing visual observation')
        if ((batch['type'][valid] != 0) & (batch['type'][valid] != 1)).any():
            raise ValueError('unknown observed visual role')
        z = batch['z'].masked_fill(~valid[..., None], 0)
        p = (batch['projected_z'].masked_fill(~valid[..., None], 0)
             if 'projected_z' in batch else (z - self.input_mean) @ self.projection)
        age = batch['age'].masked_fill(~valid, 0)
        role = (1 - batch['type']).float().masked_fill(~valid, 0)
        past = batch['past_actions'].masked_fill(~valid[..., None], 0)
        # Original collate uses recent then history. Sort actual ages so the
        # causal reader receives chronological evidence, preserving real time.
        order = age.masked_fill(~valid, -1).argsort(dim=1, descending=True, stable=True)
        gather = lambda x: x.gather(1, order[..., None].expand(-1, -1, x.shape[-1]))
        p, past = gather(p), gather(past)
        temporal = gather(torch.stack((age / 8., role), dim=-1))
        visual_valid = valid.gather(1, order)
        observed = self.visual(p) + self.past_control(past) + self.time(temporal)
        outputs = []
        for h in self.horizons:
            # Tokens after h are neither read, embedded, nor summarized for
            # this query. Altering h6-only actions cannot change h3 output.
            controls = batch['actions'][:, :h]
            if controls.shape[1:] != (h, 2):
                raise ValueError('planned prefix incomplete')
            future_time = torch.stack((torch.arange(1, h+1, device=z.device, dtype=z.dtype) / 8.,
                                       z.new_full((h,), 2.)), dim=-1)
            future = self.future_control(controls) + self.time(future_time)[None]
            query = self.query[None, None] + self.horizon(z.new_full((len(z), 1, 1), h / 8.))
            tokens = torch.cat((observed, future, query), dim=1)
            padding = torch.cat((~visual_valid, torch.zeros((len(z), h+1), dtype=torch.bool, device=z.device)), dim=1)
            causal = torch.triu(torch.ones((tokens.shape[1], tokens.shape[1]), dtype=torch.bool, device=z.device), diagonal=1)
            coeff = self.output(self.encoder(tokens, mask=causal, src_key_padding_mask=padding)[:, -1])
            outputs.append(batch['current'] + self.target_mean + coeff @ self.output_basis.T)
        return torch.stack(outputs, dim=1)
