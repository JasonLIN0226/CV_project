#!/usr/bin/env python3
"""Native CPU cache attestation after the one declared A100 encoding job.

This is a root numerical/source audit, not a separately delegated review or a
reader experiment. No labels, feature fitting, model training or GPU inference.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def regular(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Missing or symlinked explicit cache/source file')


def selected_digest(episodes):
    h = hashlib.sha256()
    for ei, descriptor in enumerate(episodes):
        start = Fraction(*descriptor['start_seconds'])
        for i in range(256):
            stamp = start + Fraction(i,2)
            h.update(f'{ei}:{i}:{stamp.numerator}/{stamp.denominator}\n'.encode())
    return h.hexdigest()


def run(control, expected_lock):
    began = time.monotonic()
    control = Path(control)
    lock = control/'source_lock.json'
    regular(lock)
    if sha(lock) != expected_lock:
        raise ValueError('Cache audit source lock changed')
    for rel, want in json.loads(lock.read_text())['files'].items():
        path = control/rel
        regular(path)
        if sha(path) != want:
            raise ValueError('Cache audit source changed: '+rel)
    p = json.loads((control/'protocol.json').read_text())
    if (os.getenv('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa'
            or not os.getenv('SLURM_JOB_ID') or not socket.gethostname().startswith('b-')):
        raise ValueError('Actual allocated course CPU required for cache audit')
    sys.path[:0] = [str(control/'src'),str(control/'scripts')]
    import numpy as np
    import torch
    from phase1.v5_real_training import validate_cache_report
    from phase1.v5_encoding_l4 import checkpoint_files
    torch.set_num_threads(4)
    account = subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',str(p['encoding_job_id']),
        '--format=JobIDRaw,Account,Partition,State,ExitCode,AllocTRES','-n','-P'],text=True)
    rows = [r.split('|') for r in account.splitlines() if r.strip()]
    if len(rows)!=1 or rows[0][:5] != [str(p['encoding_job_id']),'csci_ga_2271-2026fa','c12m85-a100-1','COMPLETED','0:0']:
        raise ValueError('Actual encoding is not COMPLETED0:0 on the declared course/A100 path')
    tres = dict(v.split('=',1) for v in rows[0][5].split(',') if '=' in v)
    if tres.get('gres/gpu') != '1':
        raise ValueError('Actual allocation did not contain exactly one GPU')
    encoding_control = Path(p['encoding_control'])
    e_lock = encoding_control/'source_lock.json'
    regular(e_lock)
    if sha(e_lock) != p['encoding_source_lock_sha256']:
        raise ValueError('Actual encoding source lock changed')
    for rel,want in json.loads(e_lock.read_text())['files'].items():
        file = encoding_control/rel
        regular(file)
        if sha(file)!=want:
            raise ValueError('Actual encoding source/input changed: '+rel)
    root = Path(p['encoding_root'])
    report_path = root/'encoding_report.json';regular(report_path)
    report_before_sha = sha(report_path)
    report = json.loads(report_path.read_text())
    for key,filename in [('source_lock_sha256','source_lock.json'),('protocol_sha256','protocol.json'),
                         ('manifest_sha256','manifest.json'),('pts_report_sha256','pts_report.json')]:
        if report[key]!=sha(encoding_control/filename):
            raise ValueError('Actual cache proof binding differs: '+key)
    ep = json.loads((encoding_control/'protocol.json').read_text())
    if sha(encoding_control/'protocol.json')!=p['encoding_protocol_sha256'] or 'A100' not in report['device']:
        raise ValueError('Declared encoding protocol/device differs')
    if report['runtime']['cpu_threads']!=4 or report['task_labels_read'] or report['sealed_rgb_or_numeric_labels_read']:
        raise ValueError('Compute or information boundary differs')
    manifest = json.loads((encoding_control/'manifest.json').read_text())
    actual = validate_cache_report(report,manifest,p['encoding_job_id'])
    metadata_path = root/'encoder_metadata.json';regular(metadata_path)
    if sha(metadata_path)!=report['encoder_metadata_sha256']:
        raise ValueError('Actual encoder metadata differs')
    metadata=json.loads(metadata_path.read_text())
    if any(metadata[key]!=ep['encoder'][key] for key in ('pooling','dtype')) or metadata['resolved_revision']!=ep['encoder']['revision'] or metadata['name']!=ep['encoder']['repository']:
        raise ValueError('Frozen encoder/processor provenance differs')
    cfg=ep['encoder']
    if checkpoint_files(cfg['cache_dir'],cfg['repository'],cfg['revision'])!=report['actual_official_checkpoint_files']:
        raise ValueError('Actual cached official weight/config/processor bytes changed')
    pts=json.loads((encoding_control/'pts_report.json').read_text())
    by_path={v['path']:v for v in pts['videos']}
    features=[]
    for got in actual:
        audited=by_path[got['rgb_path']]
        if got['episodes']!=audited['episodes'] or got['feature']['timestamp_sha256']!=selected_digest(got['episodes']) or got['feature']['timestamp_sha256']!=audited['selected_timestamps_sha256']:
            raise ValueError('Actual selected episode/PTS digest differs')
        f=got['feature'];path=root/'features'/f['path'];regular(path)
        before=sha(path)
        if path.stat().st_size!=f['bytes'] or before!=f['sha256']:
            raise ValueError('Actual feature file bytes differ')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if array.shape!=tuple(f['shape']) or array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError('Incomplete/nonfinite/invalid encoded feature array')
        if sha(path)!=before:
            raise ValueError('Feature file changed during audit')
        features.append({'path':str(path),'shape':list(array.shape),'sha256':before,'finite':True})
    if {x.name for x in (root/'features').iterdir()}!={x['feature']['path'] for x in actual}:
        raise ValueError('Unexpected, partial or extra payload in complete cache')
    if sha(report_path)!=report_before_sha or report_before_sha!=p.get('expected_actual_report_sha256',report_before_sha):
        raise ValueError('Completed encoding report changed during audit')
    output=Path(p['output'])
    if output.exists():
        raise ValueError('Refuse repeated cache audit overwrite')
    result={'status':'ACTUAL_FULL_CACHE_VERIFIED_NOT_READER_TRAINING_NOT_PHASE1_GO','complete':True,
        'audit_job_id':os.getenv('SLURM_JOB_ID'),'encoding_job_id':p['encoding_job_id'],
        'encoding_report_sha256':sha(report_path),'encoding_source_lock_sha256':sha(e_lock),
        'encoding_protocol_sha256':sha(encoding_control/'protocol.json'),'encoder_metadata_sha256':sha(metadata_path),
        'actual_feature_files':features,'frames':50688,'recordings':78,'episodes':198,
        'slurm_encoding_record':account,'source_lock_sha256':sha(lock),'protocol_sha256':sha(control/'protocol.json'),
        'scope':'Root source and numerical audit; no newly delegated independent reviewer',
        'reader_fit_or_training':False,'labels_read':False,'sealed_data_read':False,'phase1_go':False,
        'elapsed_seconds':time.monotonic()-began}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output),'sha256':sha(output),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--control',required=True)
    parser.add_argument('--source-lock-sha',required=True)
    args=parser.parse_args()
    run(args.control,args.source_lock_sha)
