#!/usr/bin/env python3
"""Validate every locked source/input byte before importing science code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--control', type=Path, required=True)
p.add_argument('--source-lock-sha', required=True)
p.add_argument('--encoding-root', type=Path, required=True)
p.add_argument('--encoding-control', type=Path, required=True)
p.add_argument('--encoding-job', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
lock_path = args.control / 'source_lock.json'
assert not lock_path.is_symlink()
assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == args.source_lock_sha
lock = json.loads(lock_path.read_text())
for name, expected in lock['files'].items():
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    actual = args.control / relative
    assert not any(x.is_symlink() for x in (actual, *actual.parents))
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected, name
protocol = json.loads((args.control / 'protocol.json').read_text())
assert protocol['encoding_job_id'] == args.encoding_job
assert protocol['actual_required_observations'] == 50688
assert protocol['seeds'] == [17, 29, 43] and protocol['epochs'] == 30
assert protocol['data_scope'] == 'all58train20validation_no_sealed'
assert protocol['course'] == os.getenv('SLURM_JOB_ACCOUNT') == 'csci_ga_2271-2026fa'
sys.path.insert(0, str(args.control / 'scripts'))
sys.path.insert(0, str(args.control / 'src'))
from phase1.v5_real_training_l4 import main
sys.exit(main(args.control, args.encoding_root, args.encoding_control, args.encoding_job, args.output))
