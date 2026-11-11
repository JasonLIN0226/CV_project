"""Compress to Remember / PACC (Wu et al., arXiv 2609.36364) for a small frozen predictor.

No code is public, so this follows the paper text.

  kept     Compressor: a transformer that embeds the history tokens, appends learnable
           memory queries, and keeps only the query outputs after a LayerNorm (Eq. 1).
           History tokens attend to each other; queries attend to history and to each
           other; history never attends to the queries.
  kept     Training: the same frozen model is the teacher when it reads the full history
           and the student when it reads the memory tokens. Only the compressor is updated,
           by matching the two predictions with a squared error (Eq. 2-4).
  changed  The frozen model is the project's one-shot predictor, so there is no denoising
           schedule and no multi-chunk rollout; "on-policy continuation" has no analogue.
  changed  Memory tokens are emitted in the predictor's item space (projected feature,
           executed-action suffix, age), which is where it accepts memory.
  note     The paper compresses each block once and keeps every compressed block in a
           growing archive ("bounded attention context does not imply bounded storage").
           `full` reproduces that: the compressor reads the whole eligible history at the
           query. `recurrent` is NOT in the paper: it folds each arriving frame into two
           persistent slots, which is what a bounded store with irrevocable loss requires.

This file holds the compressor network and the learning-rate schedule; gate.py and synergy.py
train and score it in each environment. Moved unchanged from experiments/memory_baselines/pacc.py.
"""
from __future__ import annotations

import math

import torch
from torch import nn

B = 2   # memory slots, the project's budget


class Compressor(nn.Module):
    def __init__(self, d, width=128, layers=2, heads=4, slots=B):
        super().__init__()
        self.d, self.slots, self.width = d, slots, width
        self.embed = nn.Linear(d + 3, width)            # feature, action suffix (2), age/8
        self.slot_in = nn.Linear(width, width)          # recurrent variant: previous slots as tokens
        self.step_in = nn.Linear(2, width)              # recurrent variant: last executed action
        self.queries = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.kind = nn.Embedding(3, width)              # history frame / previous slot / step token
        block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.to_item = nn.Linear(width, d + 3)

    def _run(self, tokens, valid):
        rows, n = tokens.shape[:2]
        x = torch.cat([tokens, self.queries[None].expand(rows, -1, -1)], 1)
        valid = torch.cat([valid, torch.ones(rows, self.slots, dtype=torch.bool, device=x.device)], 1)
        blocked = torch.zeros(n + self.slots, n + self.slots, dtype=torch.bool, device=x.device)
        blocked[:n, n:] = True  # history tokens never read the queries
        out = self.blocks(x, mask=blocked, src_key_padding_mask=~valid)
        return self.norm(out[:, n:])

    def items(self, memory):
        out = self.to_item(memory)
        return out[..., :self.d], out[..., self.d:self.d + 2], 8 * nn.functional.softplus(out[..., -1])

    def compress(self, p, past, age, valid):
        """Paper form: one pass over the whole history. p [rows, n, d], past [rows, n, 2], age [rows, n]."""
        tokens = self.embed(torch.cat([p, past, age[..., None] / 8], -1)) + self.kind.weight[0]
        return self._run(tokens, valid)

    def fold(self, memory, p_new, last_action):
        """Recurrent form: previous slots + one arriving frame + the last executed action -> new slots."""
        rows = p_new.shape[0]
        zeros = torch.zeros(rows, 3, device=p_new.device)
        frame = self.embed(torch.cat([p_new, zeros], -1))[:, None] + self.kind.weight[0]
        step = self.step_in(last_action)[:, None] + self.kind.weight[2]
        tokens = [frame, step]
        if memory is not None:
            tokens.insert(0, self.slot_in(memory) + self.kind.weight[1])
        tokens = torch.cat(tokens, 1)
        return self._run(tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device))


def _schedule(step, steps, warmup=100):
    if step < warmup:
        return (step + 1) / warmup
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
