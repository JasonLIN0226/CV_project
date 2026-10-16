#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import hashlib
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control',required=True)
    p.add_argument('--source-lock-sha',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    control = Path(a.control)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    lock = control/'source_lock.json'
    if sha(lock) != a.source_lock_sha: raise ValueError('Source lock mismatch')
    for rel,want in json.loads(lock.read_text())['files'].items():
        if sha(control/rel) != want: raise ValueError('Source hash mismatch: '+rel)
    sys.path.insert(0,str(control/'src'))
    from phase1.v5_synthetic_training import run
    from phase1.utils import write_json
    result = run(control,control/'configs/phase1_v5_gate_training_protocol.json',a.output)
    write_json(Path(a.output)/'training_result.json',result)
    print(result,flush=True)
    return 0 if result['status']=='TRAINING_COMPLETE_EVALUATION_PENDING' else 143 if result['status']=='INTERRUPTED' else 3

if __name__ == '__main__':
    raise SystemExit(main())
