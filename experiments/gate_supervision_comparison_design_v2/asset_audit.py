"""Read-only source-lock and train/validation feature-cache audit.

No audit runs at import. The asset lock may be read for provenance; this module
never opens the independent test dataset or mutates original assets.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .artifacts import ASSET_LOCK, digest_file
from .states import validate_manifest


def verify_entries(entries, allowed_roots):
    """Verify locked bytes with realpath containment before opening a file."""
    roots = [Path(root).resolve(strict=True) for root in allowed_roots]
    verified = {}
    for raw_path, info in entries.items():
        path = Path(raw_path).resolve(strict=True)
        if not any(path == root or root in path.parents for root in roots):
            raise ValueError('locked asset resolves outside authorized roots')
        if not path.is_file() or path.stat().st_size != info['bytes'] or digest_file(path) != info['sha256']:
            raise ValueError(f'locked asset differs: {raw_path}')
        verified[raw_path] = info['sha256']
    return verified


def audit_frozen_assets(lock_path, project_root, validation_audit_path):
    """Verify the complete original lock and train/validation metadata.

    No checkpoint is unpickled. This checks the feature index, not each cached
    feature file's content; use audit_feature_files when explicitly needed.
    """
    lock_path = Path(lock_path)
    if digest_file(lock_path) != ASSET_LOCK:
        raise ValueError('original model/source lock has changed')
    lock = json.loads(lock_path.read_text())
    if lock.get('status') != 'FROZEN_CLEAN_FUTURE_GATE_INDEPENDENT_V1' or lock.get('test_opened_at_freeze') is not False:
        raise ValueError('unexpected original provenance lock')
    project = Path(project_root).resolve(strict=True)
    audit_path = Path(validation_audit_path).resolve(strict=True)
    entries = lock['files']
    if len(entries) != 141 or str(audit_path) not in entries:
        raise ValueError('unexpected source-lock inventory')
    audit = verify_entries(entries, [project, audit_path])
    root = project / 'results/gate_clean_future_20261022'
    manifest = json.loads((root / 'dataset/manifest.json').read_text())
    validate_manifest(manifest)
    index = json.loads((root / 'features/index.json').read_text())
    episode_ids = {r['id'] for r in manifest['records']}
    if set(index['episodes']) != episode_ids:
        raise ValueError('cache index and allowed manifest differ')
    return {'source_lock_sha256': ASSET_LOCK, 'locked_files_checked': len(audit),
            'episodes_by_split': dict(Counter(r['split'] for r in manifest['records'])),
            'indexed_episodes': len(index['episodes']), 'cache_content_checked': False}


def audit_feature_files(project_root):
    """Check every train/val cached feature file against its frozen index.

    The manifest is validated before opening feature files. No decoder, model,
    teacher, test split or target generation is used. Never writes artifacts.
    """
    root = Path(project_root).resolve(strict=True) / 'results/gate_clean_future_20261022'
    manifest = json.loads((root / 'dataset/manifest.json').read_text())
    validate_manifest(manifest)
    index = json.loads((root / 'features/index.json').read_text())
    ids = {r['id'] for r in manifest['records']}
    if set(index['episodes']) != ids:
        raise ValueError('cache index and allowed manifest differ')
    feature_root = (root / 'features').resolve(strict=True)
    for episode_id in sorted(ids):
        info = index['episodes'][episode_id]
        path = (feature_root / info['path']).resolve(strict=True)
        if feature_root not in path.parents or not path.is_file():
            raise ValueError('feature path escapes authorized cache')
        if digest_file(path) != info['sha256']:
            raise ValueError(f'feature cache SHA mismatch: {episode_id}')
    return {'cache_content_checked': True, 'feature_files': len(ids)}
