#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--control',required=True)
    p.add_argument('--source-lock-sha',required=True)
    a=p.parse_args();control=Path(a.control)
    sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
    if sha(control/'source_lock.json')!=a.source_lock_sha:
        raise ValueError('Evaluation source lock mismatch')
    for rel,want in json.loads((control/'source_lock.json').read_text())['files'].items():
        if sha(control/rel)!=want:raise ValueError('Evaluation source changed: '+rel)
    sys.path.insert(0,str(control/'src'))
    from phase1.v5_motion_evaluation import run_evaluation
    run_evaluation(control,control/'configs/phase1_v5_motion_evaluation_protocol.json')
    return 0

if __name__=='__main__':raise SystemExit(main())
