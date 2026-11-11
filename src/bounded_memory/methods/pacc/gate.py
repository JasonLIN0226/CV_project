"""Compress to Remember on Gate: training against the frozen predictor and scoring.

`full` is the published form (one pass over the whole eligible history at the query).
`recurrent` is not in the paper: each arriving frame is folded into two persistent slots.
Only the compressor is trained; the predictor stays frozen and gradients pass through it.
Training checkpoints every `checkpoint_every` steps, so a run stopped by a time limit continues
when it is started again. Moved unchanged from experiments/memory_baselines/pacc.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from ...environments.gate.protocol import B, HORIZONS, R, item_batch, with_memory_tokens
from .compressor import Compressor, _schedule


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
