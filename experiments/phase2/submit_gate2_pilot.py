"""Submit one reviewed Gate2 reader/geometry pilot; no student training."""
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
WORK = BASE / 'gate2-terminal-pilot-20261025'
NAME = 'cv-gate2-terminal-20261025'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def batch_text(bundle):
    command = [str(BASE/'venv/bin/python'), str(bundle/'experiments/phase2/run_gate2_pilot.py'),
               '--output', str(WORK/'run'), '--cache-dir', str(BASE/'cache/ijepa')]
    pack = r'''
import hashlib, json, os, zipfile
from pathlib import Path
work = Path.cwd()
run = work / 'run'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
packed={}
destination=work/'gate2-pilot-results.zip'
with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for root,prefix in ((run,'run'),(work/'bundle','source')):
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts: continue
            name=prefix+'/'+str(p.relative_to(root))
            packed[name]={'sha256':sha(p),'bytes':p.stat().st_size}
            z.write(p,name)
    for name in ('submission_receipt.json','run.sbatch','submission_driver.py'):
        p=work/name
        packed['execution/'+name]={'sha256':sha(p),'bytes':p.stat().st_size}
        z.write(p,'execution/'+name)
    z.writestr('package_manifest.json',json.dumps({'files':packed,'job_id':os.environ['SLURM_JOB_ID']},indent=2))
with zipfile.ZipFile(destination) as z:
    if z.testzip() is not None: raise RuntimeError('ZIP CRC mismatch')
receipt={'status':'packaged_after_runner_exit_zero','job_id':os.environ['SLURM_JOB_ID'],
         'zip_sha256':sha(destination),'zip_bytes':destination.stat().st_size,'files':len(packed),
         'remote_originals_preserved':True,'scientific_status':'read run results; packaging is not a method success claim'}
(work/'zip_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
'''
    return '\n'.join(['#!/bin/bash','set -euo pipefail','umask 077',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4',
        'export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1',
        shlex.join(command), shlex.join([str(BASE/'venv/bin/python'),'-'])+" <<'PACK_GATE2'",
        pack, 'PACK_GATE2', ''])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}',args.sha256) or sha(args.archive)!=args.sha256:
        raise ValueError('Reviewed bundle hash mismatch')
    if not (BASE/'venv/bin/python').is_file() or not (BASE/'cache/ijepa').is_dir():
        raise ValueError('Expected existing Python environment and frozen encoder cache')
    env={k:os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','SHELL') if k in os.environ}
    active=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xy3257','-n',NAME,'-o','%i %T'],text=True,env=env,timeout=30).strip()
    if active: raise RuntimeError('Identically named job already exists: '+active)
    WORK.mkdir(parents=True,exist_ok=True)
    receipt=WORK/'submission_receipt.json'
    record={'status':'preparing','created_utc':datetime.now(timezone.utc).isoformat(),
            'archive_sha256':args.sha256,'gpu_execution_verified':False,'student_training':False}
    with receipt.open('x') as f: json.dump(record,f,indent=2)
    try:
        bundle=WORK/'bundle';bundle.mkdir()
        with zipfile.ZipFile(args.archive) as z:
            for member in z.infolist():
                if not (bundle/member.filename).resolve().is_relative_to(bundle.resolve()):
                    raise ValueError('Unsafe archive member')
            z.extractall(bundle)
        manifest=json.loads((bundle/'bundle_manifest.json').read_text())
        for name,expected in manifest['files'].items():
            p=(bundle/name).resolve()
            if not p.is_relative_to(bundle.resolve()) or sha(p)!=expected:
                raise ValueError('Bundle member mismatch: '+name)
        (WORK/'run.sbatch').write_text(batch_text(bundle))
        (WORK/'submission_driver.py').write_bytes(Path(__file__).read_bytes())
        command=['/opt/slurm/bin/sbatch','--parsable','--export=ALL','--no-requeue',
                 '--job-name='+NAME,'--account=csci_ga_2271-2026fa','--partition=c12m85-a100-1',
                 '--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=00:20:00',
                 '--chdir='+str(WORK),'--output='+str(WORK/'slurm-%j.log'),str(WORK/'run.sbatch')]
        record.update(status='submitting',command=command,bundle_manifest_sha256=sha(bundle/'bundle_manifest.json'))
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        result=subprocess.run(command,text=True,capture_output=True,env=env,timeout=45)
        record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode!=0 or not re.fullmatch(r'\d+(;[^\s]+)?\s*',result.stdout):
            raise RuntimeError('Submission not confirmed; inspect scheduler before any retry')
        record.update(status='submitted',job_id=result.stdout.strip().split(';')[0])
    except Exception as exc:
        record.update(status='failed_or_unknown',error=str(exc));raise
    finally:
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
