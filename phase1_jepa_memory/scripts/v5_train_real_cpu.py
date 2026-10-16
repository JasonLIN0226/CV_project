#!/usr/bin/env python3
"""CPU-only real-reader launcher; no fitting before actual cache attestation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    began = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--source-lock-sha', required=True)
    parser.add_argument('--encoding-root', type=Path, required=True)
    parser.add_argument('--encoding-control', type=Path, required=True)
    parser.add_argument('--encoding-job', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock_path = args.control/'source_lock.json'
    if (any(x.is_symlink() for x in (lock_path,*lock_path.parents))
            or hashlib.sha256(lock_path.read_bytes()).hexdigest()!=args.source_lock_sha):
        raise ValueError('CPU training source lock changed')
    for name, expected in json.loads(lock_path.read_text())['files'].items():
        relative = Path(name)
        path = args.control/relative
        if (relative.is_absolute() or '..' in relative.parts
                or any(x.is_symlink() for x in (path,*path.parents))
                or hashlib.sha256(path.read_bytes()).hexdigest()!=expected):
            raise ValueError('CPU training source/protocol file changed')
    p = json.loads((args.control/'protocol.json').read_text())
    if (p.get('status')!='PREREGISTERED_BOUND_CPU_REAL_READER_NOT_EXECUTED'
            or p.get('encoding_job_id')!=args.encoding_job
            or p.get('actual_required_observations')!=50688
            or p.get('seeds')!=[17,29,43] or p.get('epochs')!=30
            or p.get('data_scope')!='all58train20validation_no_sealed'
            or p.get('course')!=os.getenv('SLURM_JOB_ACCOUNT')
            or p.get('course')!='csci_ga_2271-2026fa'):
        raise ValueError('Actual CPU real-data scientific protocol not established')
    sys.path[:0] = [str(args.control/'src'),str(args.control/'scripts')]
    from phase1.v5_real_training_cpu import main as train
    return train(args.control,args.encoding_root,args.encoding_control,args.encoding_job,args.output,began)


if __name__=='__main__':
    raise SystemExit(main())
