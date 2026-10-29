#!/usr/bin/env python3
"""Verify the original clean-future source lock against its preserved snapshot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'experiments/retention_structured_noise_audit/clean_future_source_lock.json'
ORIGINAL_PROBES = 'experiments/predictor/control/src/phase1/probes.py'
FROZEN_PROBES = ROOT / 'experiments/gate_clean_future/frozen_control/probes.py'


def verify():
    lock = json.loads(LOCK.read_text())
    if lock['status'] != 'FROZEN_CLEAN_FUTURE_TRAIN_VALIDATION_V1':
        raise ValueError('Unexpected clean-future source lock')
    for original, expected in lock['files'].items():
        path = FROZEN_PROBES if original == ORIGINAL_PROBES else ROOT / original
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen clean-future source mismatch: {original}')
    return len(lock['files'])


if __name__ == '__main__':
    print(f'Clean-future frozen source verified: {verify()} files')
