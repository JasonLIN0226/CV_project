"""Verify downloaded packages or extracted assets using only Python's stdlib."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(manifest, root, extracted=False):
    if manifest['status'] != 'NATIVE_EXPORT_COMPLETE_BEFORE_AFTER_SHA_MATCH':
        raise ValueError('Handoff is not a complete native export')
    entries = manifest['files'] if extracted else manifest['packages']
    for entry in entries:
        rel = entry['relative_path'] if extracted else entry['name']
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe manifest path')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Missing or mismatched file: '+rel)
    if extracted:
        for variant in ('development', 'independent_confirmation'):
            models = [x for x in manifest['files'] if x['variant']==variant
                      and x.get('role') in ('reference','candidate')
                      and x['relative_path'].endswith('/best.pt')]
            if {(x['role'],x['task'],x['seed']) for x in models} != {
                    (role, task, seed) for role in ('candidate','reference')
                    for task in ('gate','motion') for seed in (17,29,43)}:
                raise ValueError('Incomplete 12-checkpoint variant: '+variant)
    return dict(status='PASS', verified_files=len(entries), mode='extracted' if extracted else 'packages',
                model_deserialized=False, training=False)


def verify_supplement(manifest, root, extracted=False):
    if manifest['status'] != 'PUBLISHED_COPY_MATCHES_TEAMMATE_EXPECTED_SHA':
        raise ValueError('Unknown supplemental provenance')
    entries = manifest['files'] if extracted else manifest['packages']
    for entry in entries:
        rel = entry['relative_path'] if extracted else entry['name']
        path = (root / rel).resolve()
        if (not path.is_relative_to(root.resolve()) or not path.is_file()
                or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']):
            raise ValueError('Missing or mismatched supplemental file: '+rel)
    return len(entries)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--extracted', action='store_true')
    p.add_argument('--supplement', help='Separate retained-reference provenance manifest')
    args=p.parse_args()
    result = verify(json.loads(Path(args.manifest).read_text()),Path(args.root),args.extracted)
    if args.supplement:
        result['supplemental_files_verified'] = verify_supplement(
            json.loads(Path(args.supplement).read_text()), Path(args.root), args.extracted)
    print(json.dumps(result))
