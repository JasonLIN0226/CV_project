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
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .gate_data import B, HORIZONS, R, item_batch, with_memory_tokens


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


def _history(episodes, clocks, device):
    """Padded eligible history (frames 0 .. t-R) for each (episode, clock)."""
    n = max(t - R + 1 for t in clocks)
    d = episodes[0].p.shape[1]
    p = np.zeros((len(episodes), n, d), np.float32)
    past = np.zeros((len(episodes), n, 2), np.float32)
    age = np.zeros((len(episodes), n), np.float32)
    valid = np.zeros((len(episodes), n), bool)
    for i, (e, t) in enumerate(zip(episodes, clocks)):
        k = t - R + 1
        p[i, :k] = e.p[:k]
        past[i, :k] = e.prefix[t] - e.prefix[:k]
        age[i, :k] = t - np.arange(k)
        valid[i, :k] = True
    return tuple(torch.from_numpy(a).to(device) for a in (p, past, age, valid))


def _teacher(model, episodes, clocks, device):
    with torch.no_grad():
        return model(item_batch(episodes, clocks, [list(range(t - R + 1)) for t in clocks], device))


def _student(model, compressor, memory, episodes, clocks, device):
    recent = item_batch(episodes, clocks, [[] for _ in episodes], device)
    return model(with_memory_tokens(recent, *compressor.items(memory)))


def _schedule(step, steps, warmup=100):
    if step < warmup:
        return (step + 1) / warmup
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def _recurrent_memory(compressor, episodes, clocks, device):
    """Fold frames 0 .. t-R one at a time; returns the slots held at each requested clock."""
    final = max(clocks)
    memory, out = None, [None] * len(episodes)
    for t in range(R, final + 1):
        k = t - R
        live = [i for i, c in enumerate(clocks) if c >= t]
        p_new = torch.from_numpy(np.stack([episodes[i].p[k] for i in live])).to(device)
        act = torch.from_numpy(np.stack([episodes[i].actions[t - 1] for i in live])).to(device)
        prev = None if memory is None else memory[[j for j, i in enumerate(prev_live) if clocks[i] >= t]]
        memory = compressor.fold(prev, p_new, act)
        prev_live = live
        for j, i in enumerate(live):
            if clocks[i] == t:
                out[i] = memory[j]
    return torch.stack(out)


def train(model, train_episodes, seed, device, variant="full", steps=3000, batch=64, lr=3e-4, width=128, layers=2,
          checkpoint=None, checkpoint_every=100, tag=None):
    """Train the compressor against the frozen model.

    With `checkpoint` (a file path) the loop saves its whole state every `checkpoint_every` steps and
    continues from that file when it exists, so a run stopped by a time limit is resumed by starting it
    again. The state covers the compressor, optimiser, schedule and the batch sampler, so a resumed run
    takes exactly the steps an uninterrupted one would. A finished run leaves a final checkpoint and is
    not retrained. `tag` names the asset set; a checkpoint from a different run is refused.
    """
    if variant not in ("full", "recurrent"):
        raise ValueError(variant)
    d = train_episodes[0].p.shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        compressor = Compressor(d, width=width, layers=layers)
    compressor = compressor.to(device).train()
    opt = torch.optim.AdamW(compressor.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, steps))
    rng = np.random.default_rng(seed)
    curve, start = [], 0
    identity = json.dumps({"seed": seed, "variant": variant, "steps": steps, "batch": batch, "lr": lr, "width": width,
                           "layers": layers, "train_episodes": len(train_episodes),
                           "device": torch.device(device).type, "tag": tag}, sort_keys=True)
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError(f"{checkpoint} was written by a different run: {state['identity']} != {identity}")
        compressor.load_state_dict(state["compressor"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        rng.bit_generator.state = json.loads(state["sampler"])
        curve, start = list(state["curve"]), int(state["step"])
        print(f"[pacc {variant} seed {seed}] continuing from step {start}/{steps}", flush=True)
    for step in range(start, steps):
        idx = rng.integers(0, len(train_episodes), size=batch)
        eps = [train_episodes[i] for i in idx]
        # Half the examples sit at the query, the rest at an earlier clock with more than B history frames.
        clocks = [e.q if rng.random() < 0.5 else int(rng.integers(R + B, e.q + 1)) for e in eps]
        target = _teacher(model, eps, clocks, device)
        if variant == "full":
            memory = compressor.compress(*_history(eps, clocks, device))
        else:
            memory = _recurrent_memory(compressor, eps, clocks, device)
        loss = (_student(model, compressor, memory, eps, clocks, device) - target).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == steps - 1:
            curve.append(float(loss))
        done = step + 1
        if checkpoint is not None and (done % checkpoint_every == 0 or done == steps):
            partial = Path(f"{checkpoint}.tmp")
            torch.save({"identity": identity, "step": done, "curve": curve, "compressor": compressor.state_dict(),
                        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "sampler": json.dumps(rng.bit_generator.state)}, partial)
            os.replace(partial, checkpoint)
            print(f"[pacc {variant} seed {seed}] step {done}/{steps} loss {float(loss):.5f}", flush=True)
    return compressor.eval(), {"variant": variant, "steps": steps, "loss_first": curve[0], "loss_last": curve[-1],
                               "parameters": sum(p.numel() for p in compressor.parameters()),
                               "continued_from_step": start}


@torch.inference_mode()
def predict(model, compressor, episodes, device, variant="full", chunk=128):
    preds, gaps = [], []
    for i in range(0, len(episodes), chunk):
        eps = episodes[i:i + chunk]
        clocks = [e.q for e in eps]
        if variant == "full":
            memory = compressor.compress(*_history(eps, clocks, device))
        else:
            memory = _recurrent_memory(compressor, eps, clocks, device)
        pred = _student(model, compressor, memory, eps, clocks, device)
        gaps.append((pred - _teacher(model, eps, clocks, device)).square().mean((1, 2)).cpu().numpy())
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds), float(np.concatenate(gaps).mean())
