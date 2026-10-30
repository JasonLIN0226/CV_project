#!/usr/bin/env python3
"""Reconstruct and verify the frozen clean-future predictor source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
PREFIX = 'experiments/predictor/control/src/phase1/'
LOCK = ROOT / 'experiments/retention_structured_noise_audit/clean_future_source_lock.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    if lock['status'] != 'FROZEN_CLEAN_FUTURE_TRAIN_VALIDATION_V1':
        raise ValueError('Unexpected clean-future source lock')
    entries = [(p, h) for p, h in lock['files'].items() if p.startswith(PREFIX)]
    if len(entries) != 44:
        raise ValueError('Unexpected predictor source count')
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    sources = []
    for name, expected in entries:
        source = (ROOT / 'experiments/gate_clean_future/frozen_control/probes.py'
                  if name == PREFIX + 'probes.py' else ROOT / name)
        if digest(source) != expected:
            raise ValueError(f'Source hash mismatch: {name}')
        sources.append((source, name[len(PREFIX):], expected))
    for source, relative, expected in sources:
        dest = args.output / 'src/phase1' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if digest(dest) != expected:
            raise ValueError(f'Copy hash mismatch: {relative}')
    print(f'Reconstructed and verified {len(sources)} predictor files in {args.output}')


if __name__ == '__main__':
    main()
