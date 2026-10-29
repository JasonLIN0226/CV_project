"""Exclusive, hash-bound artifacts for an eventual authorized train/val run.

Importing this module writes nothing. A sealed state pack must exist before
teacher targets are written. Failures leave an unsealed directory for inspection
instead of overwriting a previous run or silently accepting a partial pack.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np

from .pack import manifest_digest
from .core import BEHAVIORS
from .teachers import deletion_distribution

HEX256 = re.compile(r'^[0-9a-f]{64}$')
ASSET_LOCK = 'e827118c33ababc49d85c567ffb66fae364f1e7a30b544a80f0a849ae99135e3'


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _write_json_exclusive(path, data):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(data, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')


def _valid_rows(rows, seed):
    allowed = {'split', 'episode', 'pair', 'seed', 'behavior', 'rng_seed', 'clock', 'candidates'}
    if seed not in (17, 29, 43) or len(rows) != 4864:
        raise ValueError('unexpected seed or state count')
    counts = Counter()
    episodes = {}
    pairs_by_split = {'train': set(), 'validation': set()}
    episode_pairs = {}
    for row in rows:
        if set(row) != allowed or row['seed'] != seed or row['split'] not in {'train', 'validation'}:
            raise ValueError('unexpected state metadata or forbidden split')
        if row['behavior'] not in BEHAVIORS:
            raise ValueError('unexpected behavior')
        ids = row['candidates']
        if len(ids) != 3 or len(set(ids)) != 3 or any(not isinstance(t, int) or t < 0 or t > row['clock'] - 4 for t in ids):
            raise ValueError('illegal candidate timestamps')
        counts[row['split']] += 1
        episodes.setdefault((row['split'], row['episode']), []).append(row['behavior'])
        pairs_by_split[row['split']].add(row['pair'])
        episode_pairs.setdefault((row['split'], row['episode']), set()).add(row['pair'])
    if counts != {'train': 4096, 'validation': 768}:
        raise ValueError('unexpected train/validation state counts')
    if len(episodes) != 1216 or any(sorted(p) != sorted(BEHAVIORS) for p in episodes.values()):
        raise ValueError('every episode must have four preregistered behaviors')
    if pairs_by_split['train'] & pairs_by_split['validation'] or any(len(p) != 1 for p in episode_pairs.values()):
        raise ValueError('pair crosses splits or episode identities disagree')


def _validate_teachers(arrays, n):
    if any(a.shape != (n, 3) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('invalid teacher arrays')
    for energy_name, target_name in [('attention_energy', 'targets_a'), ('deletion_energy', 'targets_d')]:
        expected = np.stack([deletion_distribution(x) for x in arrays[energy_name]])
        if not np.allclose(arrays[target_name], expected, atol=1e-6, rtol=1e-6):
            raise ValueError('teacher target disagrees with preregistered energy transform')


def write_state_pack(root, pack, *, seed, asset_lock_sha256, protocol_sha256, implementation_sha256):
    """Write one complete state pack once; never overwrites existing files."""
    if asset_lock_sha256 != ASSET_LOCK:
        raise ValueError('original asset lock SHA does not match preregistered lock')
    if not all(HEX256.fullmatch(v) for v in (protocol_sha256, implementation_sha256)):
        raise ValueError('missing protocol/implementation SHA')
    rows = pack['metadata']
    _valid_rows(rows, seed)
    features = np.asarray(pack['features'])
    if features.shape != (4864, 3, 517) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError('invalid shared student features')
    if manifest_digest(rows) != pack['manifest_sha256']:
        raise ValueError('state manifest digest mismatch')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(root / 'states.json', rows)
    with (root / 'features.npz').open('xb') as out:
        np.savez_compressed(out, features=features)
    seal = {'status': 'SEALED_STATE_PACK_V2', 'seed': seed,
            'manifest_sha256': pack['manifest_sha256'], 'asset_lock_sha256': asset_lock_sha256,
            'protocol_sha256': protocol_sha256, 'implementation_sha256': implementation_sha256,
            'states_file_sha256': digest_file(root / 'states.json'),
            'features_file_sha256': digest_file(root / 'features.npz')}
    _write_json_exclusive(root / 'seal.json', seal)
    return seal


def read_state_pack(root):
    """Verify on-disk bytes before returning any data; never loads pickles."""
    root = Path(root)
    seal = json.loads((root / 'seal.json').read_text())
    if seal.get('status') != 'SEALED_STATE_PACK_V2' or seal.get('asset_lock_sha256') != ASSET_LOCK:
        raise ValueError('unexpected state seal')
    for key, filename in [('states_file_sha256', 'states.json'), ('features_file_sha256', 'features.npz')]:
        if seal[key] != digest_file(root / filename):
            raise ValueError('state pack file hash mismatch')
    rows = json.loads((root / 'states.json').read_text())
    _valid_rows(rows, seal['seed'])
    if manifest_digest(rows) != seal['manifest_sha256']:
        raise ValueError('state manifest hash mismatch')
    with np.load(root / 'features.npz', allow_pickle=False) as payload:
        if set(payload.files) != {'features'}:
            raise ValueError('unexpected feature pack content')
        features = payload['features']
    if features.dtype != np.float32 or features.shape != (4864, 3, 517) or not np.isfinite(features).all():
        raise ValueError('invalid state features')
    return rows, features, seal


def write_teacher_pack(root, state_root, attention_energy, deletion_energy, targets_a, targets_d, *, timing):
    """Persist same-order teachers after state seal; never starts computation."""
    rows, _, state_seal = read_state_pack(state_root)
    arrays = {name: np.asarray(value, dtype=(np.float64 if name.endswith('energy') else np.float32)) for name, value in
              [('attention_energy', attention_energy), ('deletion_energy', deletion_energy),
               ('targets_a', targets_a), ('targets_d', targets_d)]}
    _validate_teachers(arrays, len(rows))
    if set(timing) != {'attention_seconds', 'deletion_seconds'} or any(not np.isfinite(v) or v < 0 for v in timing.values()):
        raise ValueError('must disclose separate measured teacher times')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'teacher.npz').open('xb') as out:
        np.savez_compressed(out, **arrays)
    seal = {'status': 'SEALED_TEACHER_PACK_V2', 'state_seal_sha256': digest_file(Path(state_root) / 'seal.json'),
            'manifest_sha256': state_seal['manifest_sha256'], 'seed': state_seal['seed'],
            'teacher_file_sha256': digest_file(root / 'teacher.npz'), 'timing': timing,
            'label_costs_equal': False, 'future_used_offline_only': True}
    _write_json_exclusive(root / 'seal.json', seal)
    return seal


def read_teacher_pack(root, state_root):
    rows, features, state_seal = read_state_pack(state_root)
    root = Path(root)
    seal = json.loads((root / 'seal.json').read_text())
    if (seal.get('status') != 'SEALED_TEACHER_PACK_V2'
        or seal.get('state_seal_sha256') != digest_file(Path(state_root) / 'seal.json')
        or seal.get('manifest_sha256') != state_seal['manifest_sha256']
        or seal.get('seed') != state_seal['seed']
        or seal.get('teacher_file_sha256') != digest_file(root / 'teacher.npz')):
        raise ValueError('teacher pack is not bound to frozen states')
    with np.load(root / 'teacher.npz', allow_pickle=False) as payload:
        if set(payload.files) != {'attention_energy', 'deletion_energy', 'targets_a', 'targets_d'}:
            raise ValueError('unexpected teacher pack content')
        arrays = {k: payload[k] for k in payload.files}
    _validate_teachers(arrays, len(rows))
    return rows, features, arrays, seal
