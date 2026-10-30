"""Functions for a future authorized label stage; import performs no work."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from .artifacts import read_state_pack, digest_file
from .pack import VerifiedManifest, label_one_frozen_state
from .states import iter_states


def make_teacher_arrays(state_root, store, load_episode, update_impl, model, collate, device):
    """Compute both teachers in frozen state order, only on train/validation.

    The caller must separately obtain user authorization and ensure the complete
    paired workflow fits the preset cap before invoking this function. It does
    not persist results, load any test data, train, or alter the predictor.
    Summed teacher timers cover projection/cosine and deletion-forward work;
    total stage time, I/O, startup and allocation time must be measured separately.
    """
    rows, features, seal = read_state_pack(state_root)
    verified = VerifiedManifest(rows, seal['manifest_sha256'])
    records = {r['id']: r for r in store.manifest['records']}
    generator = iter_states(store, load_episode, update_impl, seal['seed'])
    results = {'attention_energy': [], 'deletion_energy': [], 'targets_a': [], 'targets_d': []}
    elapsed = {'attention_seconds': 0.0, 'deletion_seconds': 0.0}
    for expected, expected_features in zip(rows, features):
        metadata, state = next(generator)
        if metadata != expected or not np.array_equal(state.features(), expected_features):
            raise ValueError('replayed state/features differ from frozen manifest')
        rec = records[metadata['episode']]
        result = label_one_frozen_state(store, rec, state, metadata, verified,
                                        load_episode=load_episode, model=model,
                                        collate=collate, device=device)
        results['attention_energy'].append(result['attention_energy'])
        results['deletion_energy'].append(result['deletion_energy'])
        results['targets_a'].append(result['attention_target'])
        results['targets_d'].append(result['deletion_target'])
        for key in elapsed:
            elapsed[key] += result[key]
    try:
        next(generator)
    except StopIteration:
        pass
    else:
        raise ValueError('extra replay states beyond frozen manifest')
    return {name: np.stack(values) for name, values in results.items()}, elapsed


def make_teacher_arrays_checkpointed(state_root, store, load_episode, update_impl, model,
                                     collate, device, *, checkpoint, stop_requested=lambda: False,
                                     progress=lambda *a: None):
    """Durably commit finished label rows; never reuse an unsealed state pack."""
    rows, features, seal = read_state_pack(state_root)
    verified = VerifiedManifest(rows, seal['manifest_sha256'])
    signature = digest_file(Path(state_root) / 'seal.json')
    n = len(rows)
    arrays = {k: np.zeros((n, 3), np.float64 if k.endswith('energy') else np.float32)
              for k in ('attention_energy', 'deletion_energy', 'targets_a', 'targets_d')}
    elapsed = {'attention_seconds': 0., 'deletion_seconds': 0.}
    offset = 0
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['state_seal_sha256']) != signature or int(saved['seed']) != seal['seed']:
                raise ValueError('teacher checkpoint belongs to other frozen states')
            offset = int(saved['completed'])
            if not 0 <= offset <= n:
                raise ValueError('invalid committed teacher offset')
            for k in arrays:
                if saved[k].shape != (n, 3) or not np.isfinite(saved[k]).all():
                    raise ValueError('invalid teacher checkpoint arrays')
                arrays[k][:] = saved[k]
            elapsed = {key: float(saved[key]) for key in elapsed}
    records = {r['id']: r for r in store.manifest['records']}
    generator = iter_states(store, load_episode, update_impl, seal['seed'])
    for index, (expected, expected_features) in enumerate(zip(rows, features)):
        metadata, state = next(generator)
        if metadata != expected or not np.array_equal(state.features(), expected_features):
            raise ValueError('replayed state/features differ from frozen manifest')
        if index < offset:
            continue
        result = label_one_frozen_state(store, records[metadata['episode']], state, metadata, verified,
                                        load_episode=load_episode, model=model, collate=collate, device=device)
        for name, key in [('attention_energy', 'attention_energy'), ('deletion_energy', 'deletion_energy'),
                          ('targets_a', 'attention_target'), ('targets_d', 'deletion_target')]:
            arrays[name][index] = result[key]
        for key in elapsed:
            elapsed[key] += result[key]
        completed = index + 1
        stop = bool(stop_requested())
        if completed % 64 == 0 or completed == n or stop:
            temporary = checkpoint.with_name(checkpoint.name + '.partial')
            with temporary.open('wb') as out:
                np.savez_compressed(out, **arrays, **elapsed, completed=np.array(completed),
                                    seed=np.array(seal['seed']), state_seal_sha256=np.array(signature))
                out.flush(); os.fsync(out.fileno())
            os.replace(temporary, checkpoint)
            progress(seal['seed'], completed, checkpoint)
        if stop:
            raise InterruptedError('teacher checkpoint committed; verify ledger before resume')
    try:
        next(generator)
    except StopIteration:
        pass
    else:
        raise ValueError('extra replay states beyond frozen manifest')
    return arrays, elapsed
