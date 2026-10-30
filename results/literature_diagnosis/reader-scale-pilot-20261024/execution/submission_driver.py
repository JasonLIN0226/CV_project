"""Upload a reviewed bundle and this driver; submit one bounded A100 pilot.

The driver verifies the archive hash and refuses duplicate/ambiguous submission.
It never changes the old checkout, data, environment, or another user's job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import zipfile

BASE = Path('/scratch/xy3257/cv-phase2-20261020')
NAME = 'cv-reader-scale-20261024'
WORK = BASE / 'reader-scale-probe-20261024'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def batch_text(bundle):
    command = [str(BASE / 'venv/bin/python'), str(bundle / 'experiments/phase2/run_reader_scale_probe.py'),
               'run', '--inputs', str(bundle / 'inputs'), '--cache-dir', str(BASE / 'cache/ijepa'),
               '--output', str(WORK / 'run'), '--steps', '512', '--validate-every', '64', '--model-seed', '17']
    pack = '''
import hashlib, json, os, zipfile
from pathlib import Path
work = Path.cwd()
run = work / 'run'
completion = json.loads((run / 'completion.json').read_text())
assert completion['status'] == 'completed'
assert str(completion['hardware']['slurm_job_id']) == os.environ['SLURM_JOB_ID']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
packed, omitted = {}, {}
destination = work / 'reader-scale-results.zip'
with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for p in sorted(run.rglob('*')):
        if not p.is_file(): continue
        relative=str(p.relative_to(run))
        expected=completion['files'].get(relative)
        actual=sha(p)
        if expected is not None and expected != actual: raise RuntimeError('Output changed: '+relative)
        heavy=relative.startswith('data/features/') or relative in ('data/unique_rgb.npz','data/unique_features.npz')
        (omitted if heavy else packed)[relative]={'sha256':actual,'bytes':p.stat().st_size}
        if not heavy: z.write(p, 'run/'+relative)
    for name in ('submission_receipt.json','run.sbatch','submission_driver.py'):
        z.write(work/name, 'execution/'+name)
    z.writestr('archive_manifest.json',json.dumps({'packed':packed,'remote_cache_only':omitted},indent=2))
receipt={'status':'completed','job_id':os.environ['SLURM_JOB_ID'],'zip_sha256':sha(destination),
         'zip_bytes':destination.stat().st_size,'remote_cache_retained':True,
         'actual_gpu_uuid':completion['hardware']['actual_gpu_uuid']}
(work/'zip_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt),flush=True)
'''
    return '\n'.join(['#!/bin/bash', 'set -euo pipefail', 'umask 077',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1',
        'export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1',
        shlex.join(command), shlex.join([str(BASE / 'venv/bin/python'), '-']) + " <<'PACK_READER_PILOT'",
        pack, 'PACK_READER_PILOT', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if sha(args.archive) != args.sha256 or not re.fullmatch('[0-9a-f]{64}', args.sha256):
        raise ValueError('Reviewed input archive SHA256 mismatch')
    if not (BASE / 'venv/bin/python').is_file() or not (BASE / 'cache/ijepa').is_dir():
        raise ValueError('Expected existing HPC environment/cache missing')
    env = {k: os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','SHELL') if k in os.environ}
    active = subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xy3257','-n',NAME,'-o','%i %T'],
                                     text=True, env=env, timeout=30).strip()
    if active:
        raise RuntimeError('An identically named job already exists: '+active)
    WORK.mkdir(parents=True, exist_ok=True)
    receipt_path = WORK / 'submission_receipt.json'
    record = {'status':'preparing','created_utc':datetime.now(timezone.utc).isoformat(),
              'archive_sha256':args.sha256,'gpu_execution_verified':False}
    with receipt_path.open('x') as f:
        json.dump(record,f,indent=2)
    try:
        bundle = WORK / 'bundle'
        bundle.mkdir()
        with zipfile.ZipFile(args.archive) as archive:
            for member in archive.infolist():
                if not (bundle / member.filename).resolve().is_relative_to(bundle.resolve()):
                    raise ValueError('Unsafe archive member')
            archive.extractall(bundle)
        manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
        for name, expected in manifest['files'].items():
            p = (bundle / name).resolve()
            if not p.is_relative_to(bundle.resolve()) or sha(p) != expected:
                raise ValueError('Bundle file checksum mismatch: '+name)
        (WORK / 'run.sbatch').write_text(batch_text(bundle))
        (WORK / 'submission_driver.py').write_bytes(Path(__file__).read_bytes())
        command = ['/opt/slurm/bin/sbatch','--parsable','--export=ALL','--no-requeue',
                   '--job-name='+NAME,'--account=csci_ga_2271-2026fa','--partition=c12m85-a100-1',
                   '--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=00:45:00',
                   '--chdir='+str(WORK),'--output='+str(WORK/'slurm-%j.log'),str(WORK/'run.sbatch')]
        record.update(command=command, status='submitting', bundle_manifest_sha256=sha(bundle/'bundle_manifest.json'))
        receipt_path.write_text(json.dumps(record,indent=2)+'\n')
        result = subprocess.run(command,text=True,capture_output=True,env=env,timeout=45)
        record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode != 0 or not re.fullmatch(r'\d+(;[^\s]+)?\s*',result.stdout):
            raise RuntimeError('Submission not confirmed; inspect receipt and scheduler, do not blindly retry')
        record.update(status='submitted',job_id=result.stdout.strip().split(';')[0])
    except Exception as exc:
        record.update(status='failed_or_unknown',error=str(exc))
        raise
    finally:
        receipt_path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
