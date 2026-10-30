"""Paired optimizer with atomic, step-boundary recovery; no work on import."""
from __future__ import annotations

import os
from pathlib import Path
import torch
from .core import paired_initialization
from .fit import minibatch_indices


def _save(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def fit_pair_checkpointed(features, target_a, target_d, *, seed, signature, checkpoint,
                          device, stop_requested=lambda: False, progress=lambda *a: None):
    """Run 1000 identical paired updates, restoring trusted local checkpoints.

    A snapshot is committed only after both arms complete the same update.
    On SIGTERM a caller's stop flag requests a safe snapshot at the next paired
    boundary. An abrupt kill can roll back to the last 100-step checkpoint.
    Input signature must bind the sealed state pack, teacher pack and source.
    """
    if seed not in (17, 29, 43) or not isinstance(signature, str) or len(signature) != 64:
        raise ValueError('unexpected seed or unbound input signature')
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = [torch.as_tensor(y, dtype=torch.float32, device=device) for y in (target_a, target_d)]
    if x.shape != (4096, 3, 517) or any(y.shape != (4096, 3) for y in labels):
        raise ValueError('expected complete shared training pack')
    if not torch.isfinite(x).all() or any(not torch.isfinite(y).all() for y in labels):
        raise ValueError('nonfinite training input')
    for y in labels:
        if (y < 0).any() or not torch.allclose(y.sum(dim=1), torch.ones(len(x), device=device), atol=1e-6):
            raise ValueError('invalid target distribution')
    a, d = paired_initialization(seed)
    models = [a.to(device), d.to(device)]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001) for m in models]
    path = Path(checkpoint)
    step = 0
    if path.exists():
        # Only previously-created checkpoints from the verified run directory.
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck['signature'] != signature or ck['seed'] != seed or not 0 <= ck['step'] <= 1000:
            raise ValueError('checkpoint signature, seed or step mismatch')
        for model, state in zip(models, ck['models']):
            model.load_state_dict(state)
        for opt, state in zip(optimizers, ck['optimizers']):
            opt.load_state_dict(state)
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if torch.device(device).type == 'cuda' and ck['rng_cuda'] is not None:
            torch.cuda.set_rng_state(ck['rng_cuda'].cpu(), device=device)
        step = int(ck['step'])
    batches = minibatch_indices(seed, len(x))
    for index in range(step, 1000):
        inds = torch.as_tensor(batches[index], device=device)
        for model, optimizer, target in zip(models, optimizers, labels):
            optimizer.zero_grad(set_to_none=True)
            loss = -(target[inds] * torch.log_softmax(model(x[inds]), dim=-1)).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        step = index + 1
        stop = bool(stop_requested())
        if step % 100 == 0 or step == 1000 or stop:
            _save(path, {'signature': signature, 'seed': seed, 'step': step,
                         'models': [m.state_dict() for m in models],
                         'optimizers': [o.state_dict() for o in optimizers],
                         'rng_cpu': torch.get_rng_state(),
                         'rng_cuda': torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None})
            progress(seed, step, path)
        if stop:
            raise InterruptedError('paired snapshot committed; inspect scheduler ledger before resume')
    return {'attention': models[0].state_dict(), 'deletion': models[1].state_dict(),
            'seed': seed, 'steps_per_arm': 1000}
