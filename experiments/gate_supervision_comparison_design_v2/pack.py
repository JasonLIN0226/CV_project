"""Utilities for later frozen train/validation state and teacher packs.

No work is performed on import. In particular, no manifest or label pack is
created by defining these functions. Execution requires separate authorization,
hash verification, and the protocol's GPU budget preflight.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
import numpy as np

from .integration import labels_for_record
from .states import BEHAVIORS, iter_states, validate_manifest


def manifest_digest(metadata):
    """Bind label generation to the same ordered state identities for both arms."""
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def construct_states(store, load_episode, update_impl, seed):
    """Build features and metadata only; does not access teacher targets.

    The input FeatureStore itself may hold whole offline episodes in RAM. This
    function verifies logical state construction, not B=2 physical memory.
    """
    validate_manifest(store.manifest)
    rows, features = [], []
    for meta, state in iter_states(store, load_episode, update_impl, seed):
        rows.append(meta)
        features.append(state.features())
    counts = Counter((r['split'], r['behavior']) for r in rows)
    for policy in BEHAVIORS:
        if counts[('train', policy)] != 1024 or counts[('validation', policy)] != 192:
            raise ValueError('unexpected shared state counts')
    if len(rows) != 4864:
        raise ValueError('unexpected total number of state records')
    x = np.stack(features).astype(np.float32)
    if x.shape != (4864, 3, 517) or not np.isfinite(x).all():
        raise ValueError('invalid student feature pack')
    identity = [(r['episode'], tuple(sorted(r['candidates']))) for r in rows]
    return {'metadata': rows, 'features': x, 'manifest_sha256': manifest_digest(rows),
            'counts': dict(Counter(r['split'] for r in rows)),
            'duplicate_identity_records': len(identity) - len(set(identity))}


class VerifiedManifest:
    """Immutable membership index built once after verifying the frozen digest."""
    def __init__(self, metadata, frozen_sha256):
        encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'),
                             allow_nan=False)
        if hashlib.sha256(encoded.encode('utf-8')).hexdigest() != frozen_sha256:
            raise ValueError('state manifest differs from frozen digest')
        rows = json.loads(encoded)
        self._members = frozenset(json.dumps(r, sort_keys=True, separators=(',', ':')) for r in rows)
        self.sha256 = frozen_sha256

    def contains(self, metadata):
        return json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False) in self._members


def train_inputs(metadata, features, labels_a, labels_d, verified_manifest, seed):
    """Extract the identical ordered train states; exclude validation entirely.

    Both label arrays must correspond to the manifest order. A later persistent
    pack format must cryptographically bind labels and features to these row
    identities; this in-memory helper by itself cannot prove array provenance.
    """
    if seed not in (17, 29, 43) or not isinstance(verified_manifest, VerifiedManifest):
        raise ValueError('unexpected seed or missing verified manifest')
    if manifest_digest(metadata) != verified_manifest.sha256:
        raise ValueError('metadata order differs from frozen manifest')
    x = np.asarray(features)
    a, d = np.asarray(labels_a), np.asarray(labels_d)
    if x.shape != (len(metadata), 3, 517) or a.shape != (len(metadata), 3) or d.shape != a.shape:
        raise ValueError('pack shape mismatch')
    if any(r['split'] not in {'train', 'validation'} or r['seed'] != seed for r in metadata):
        raise ValueError('forbidden split or mismatched seed')
    train = [i for i, row in enumerate(metadata) if row['split'] == 'train']
    if len(train) != 4096:
        raise ValueError('expected exactly 4096 training records')
    by_episode = {}
    for i in train:
        row = metadata[i]
        by_episode.setdefault(row['episode'], []).append(row['behavior'])
    if len(by_episode) != 1024 or any(sorted(p) != sorted(BEHAVIORS) for p in by_episode.values()):
        raise ValueError('expected four pre-registered behaviors per training episode')
    selected = np.asarray(train, dtype=np.int64)
    return x[selected].copy(), a[selected].copy(), d[selected].copy()


def label_one_frozen_state(store, rec, state, metadata, verified_manifest, *, load_episode, model, collate, device):
    """One later offline label calculation, bound to a frozen state manifest.

    The manifest and metadata checks prevent accidentally labelling an unrelated
    state. Validation targets must never be supplied to the optimizer. No result
    is written by this utility.
    """
    if not isinstance(verified_manifest, VerifiedManifest):
        raise ValueError('verified frozen state manifest required')
    if not verified_manifest.contains(metadata):
        raise ValueError('state is absent from frozen manifest')
    if (metadata['episode'] != rec['id'] or metadata['split'] != rec['split']
            or metadata['clock'] != state.clock
            or tuple(metadata['candidates']) != state.candidates):
        raise ValueError('record or candidate identity mismatch')
    return labels_for_record(store, rec, state, load_episode=load_episode,
                             model=model, collate=collate, device=device)
