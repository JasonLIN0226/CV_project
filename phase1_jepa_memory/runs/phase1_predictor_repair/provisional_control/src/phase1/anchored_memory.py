"""Frozen recent predictor plus a zero-initialized, past-only memory side net.

Adaptation of additive side tuning and ControlNet's zero initialization;
not an implementation or reproduction of either paper. All legal retained
entries enter the temporal reader. There is no Top-K retrieval or selector.
"""
import torch
from torch import nn

from .common_anchor import recent_only_teacher_batch


class MemorySideNet(nn.Module):
    def __init__(self, basis, hidden=256, layers=2, heads=4):
        super().__init__()
        for key in ('input_mean', 'projection', 'output_basis'):
            self.register_buffer(key, torch.as_tensor(basis[key]).float().clone())
        self.visual = nn.Linear(self.projection.shape[1], hidden)
        self.past_control = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.future_control = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.time = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.query = nn.Parameter(torch.zeros(hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        self.output = nn.Linear(hidden, self.output_basis.shape[1])
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, batch):
        valid = batch['mask']
        if not valid.any(1).all() or (batch['age'][valid] < 0).any():
            raise ValueError('Invalid or future visual observation')
        if not ((batch['type'][valid] == 0) | (batch['type'][valid] == 1)).all():
            raise ValueError('Unknown visual role')
        z = batch['z'].masked_fill(~valid[..., None], 0)
        p = (batch['projected_z'].masked_fill(~valid[..., None], 0) if 'projected_z' in batch
             else (z - self.input_mean) @ self.projection)
        age = batch['age'].masked_fill(~valid, 0)
        role = (1 - batch['type']).float().masked_fill(~valid, 0)
        past = batch['past_actions'].masked_fill(~valid[..., None], 0)
        order = age.masked_fill(~valid, -1).argsort(dim=1, descending=True, stable=True)
        gather = lambda x: x.gather(1, order[..., None].expand(-1, -1, x.shape[-1]))
        observed = (self.visual(gather(p)) + self.past_control(gather(past))
                    + self.time(gather(torch.stack((age / 8., role), dim=-1))))
        outputs = []
        for h in (3, 6):
            controls = batch['actions'][:, :h]
            if controls.shape[1:] != (h, 2):
                raise ValueError('Incomplete legal planned-action prefix')
            times = torch.stack((torch.arange(1, h + 1, device=z.device, dtype=z.dtype) / 8.,
                                 z.new_full((h,), 2.)), dim=-1)
            future = self.future_control(controls) + self.time(times)[None]
            query = self.query[None, None] + self.horizon(z.new_full((len(z), 1, 1), h / 8.))
            tokens = torch.cat((observed, future, query), dim=1)
            padding = torch.cat((~valid.gather(1, order),
                                 torch.zeros((len(z), h + 1), dtype=torch.bool, device=z.device)), dim=1)
            causal = torch.triu(torch.ones((tokens.shape[1], tokens.shape[1]), dtype=torch.bool,
                                          device=z.device), diagonal=1)
            encoded = self.encoder(tokens, mask=causal, src_key_padding_mask=padding)
            outputs.append(self.output(encoded[:, -1]))
        return torch.stack(outputs, dim=1)


class AnchoredMemoryPredictor(nn.Module):
    """One shared predictor for every memory condition, single-trajectory inference.

    y = frozen_recent(x_recent, a) + (s(x_recent, M, a) - s(x_recent, empty, a)) U^T.
    The two s calls use the same side-net parameters. Empty-memory output is
    exactly the frozen reference throughout training, not just at startup.
    """
    def __init__(self, reference, basis, **side_options):
        super().__init__()
        self.reference = reference.eval()
        self.reference.requires_grad_(False)
        self.side = MemorySideNet(basis, **side_options)
        self.register_buffer('projection', torch.as_tensor(basis['projection']).float().clone())

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, batch):
        recent = recent_only_teacher_batch(batch)
        with torch.no_grad():
            anchor = self.reference(recent)
        correction = self.side(batch) - self.side(recent)
        return anchor + correction @ self.side.output_basis.T
