"""Compress to Remember on the terminal (Synergy) tasks: three cards are blended into two
learned memory items; nothing is deleted. The compressor is trained to match the frozen reader's
output with all three cards. Moved unchanged from experiments/memory_baselines/synergy_run.py.
"""
from __future__ import annotations

import numpy as np
import torch

from ...environments.synergy.assets import qform, reader_batch, reader_coefficients
from .compressor import Compressor, _schedule


def pacc(model, train_store, train, conf_store, conf, seed, steps=3000, batch=64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        compressor = Compressor(128)
    opt = torch.optim.AdamW(compressor.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, steps))
    rng = np.random.default_rng(seed)
    caps = [train_store.capsule(r) for r in train]
    reference = train_store.banks["reference"]

    def memory(cs):
        p, age, past = (torch.from_numpy(np.stack([c[k][4:] for c in cs])) for k in ("pz", "age", "past"))
        return compressor.items(compressor.compress(p, past, age, torch.ones(p.shape[:2], dtype=torch.bool)))

    for _ in range(steps):
        cs = [caps[i] for i in rng.integers(0, len(caps), size=batch)]
        plans = reference[rng.integers(0, len(reference), size=batch)]
        with torch.no_grad():
            target = reader_coefficients(model, reader_batch(cs, plans))        # reader with all three frames
        loss = (reader_coefficients(model, reader_batch(cs, plans, memory(cs))) - target).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 5.0)
        opt.step(); sched.step()
    compressor.eval()
    evaluation = conf_store.banks["evaluation"]
    losses = np.empty(len(conf))
    with torch.inference_mode():
        for a in range(0, len(conf), 64):
            rr = conf[a:a + 64]
            cs = [conf_store.capsule(r) for r in rr]
            rep = [c for c in cs for _ in evaluation]
            plans = np.tile(evaluation, (len(cs), 1, 1))
            coeff = reader_coefficients(model, reader_batch(rep, plans, memory(rep))).numpy().astype(np.float64)
            resid = np.stack([conf_store.target_residual(r, "evaluation") for r in rr])
            losses[a:a + len(rr)] = qform(coeff.reshape(len(cs), len(evaluation), 2, -1) - resid, conf_store.G).mean(1)
    return losses
