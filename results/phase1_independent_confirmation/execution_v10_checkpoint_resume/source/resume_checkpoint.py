"""Exact state restoration; never replay a completed research training step."""
import copy
from pathlib import Path
import numpy as np
import torch
from phase1.utils import file_hash, read_json, restore_rng, rng_state, write_json


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def restore_exact(model, optimizer, scheduler, checkpoint):
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng(checkpoint['rng'])
    # Read-back compares actual device state, including all CUDA RNG streams.
    restored = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=rng_state())
    for key, value in restored.items():
        if not equal(value, checkpoint[key]):
            raise ValueError('Restoration is not exact: ' + key)


def load_allowed(root, task, seed, role, signature, model, optimizer, scheduler):
    execution = Path(__file__).resolve().parent
    audit = read_json(execution/'interruption_native_audit.json')
    proof = audit['checkpoint']
    root = Path(root)
    if (task, seed, role) != ('gate', 17, 'reference'):
        raise ValueError('Only the declared interrupted model may already exist')
    if (file_hash(root/'latest.pt') != proof['latest_sha256'] or
            file_hash(root/'best.pt') != proof['best_sha256']):
        raise ValueError('Retained checkpoint changed since the native CPU audit')
    checkpoint = torch.load(root/'latest.pt', map_location='cpu', weights_only=False)
    if checkpoint['signature'] != signature or checkpoint['epoch'] != 80 or checkpoint['step'] != 5120:
        raise ValueError('Wrong retained scientific signature/epoch boundary')
    restore_exact(model, optimizer, scheduler, checkpoint)
    write_json(root/'execution_resume_provenance.json', dict(
        original_latest_sha256=proof['latest_sha256'], original_best_sha256=proof['best_sha256'],
        original_source_lock_sha256=audit['source_lock_sha256'],
        new_execution_source_lock_sha256=file_hash(execution/'source_lock.json'),
        signature_preserved=True, retained_epochs=80, retained_steps=5120,
        model_optimizer_scheduler_all_rng_exact=True, no_model_forward_for_restoration=True,
        next_epoch=81, no_original_checkpoint_or_epoch_rewritten=True))
    return copy.deepcopy(checkpoint)
