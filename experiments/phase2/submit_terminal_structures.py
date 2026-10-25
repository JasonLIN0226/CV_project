"""Submit/execute the frozen terminal R/D/S study in the existing NYU environment."""
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
import sys
import time
import zipfile

BASE = Path('/scratch/xy3257/cv-phase2-20261020')
WORK = BASE/'terminal-structures-20261025'
NAME = 'cv-terminal-structures-20261025'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute(bundle, run):
    """One automatic gated sequence; no tuning/retry after development outcomes."""
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Allocated CUDA GPU required')
    hardware = {'utc': datetime.now(timezone.utc).isoformat(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0),
                'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'], text=True)}
    if 'A100' not in hardware['cuda_device']: raise RuntimeError('Expected requested A100')
    run.parent.mkdir(parents=True, exist_ok=True)
    write(run.parent/'hardware.json', hardware)
    config = bundle/'experiments/phase2/configs/terminal_structures_20261025.json'
    runner = bundle/'experiments/phase2/run_terminal_structures.py'
    students = bundle/'experiments/phase2/fit_terminal_students.py'
    summary = bundle/'experiments/phase2/summarize_terminal_students.py'
    receipt = {'status': 'running', 'hardware': hardware, 'stages': [], 'confirmation_opened': False}
    def command(args):
        started = time.time()
        p = subprocess.run([sys.executable, *map(str,args)], cwd=bundle)
        receipt['stages'].append({'arguments': list(map(str,args)), 'returncode': p.returncode,
                                  'seconds': time.time()-started})
        write(run.parent/'execution.json', receipt)
        if p.returncode: raise RuntimeError('Stage failed: '+str(args[1:3]))
    def stage(name, *args): command([runner, name, '--output', run, *args])
    try:
        stage('freeze', '--config', config)
        stage('prepare')
        stage('encode', '--device', 'cuda', '--cache-dir', BASE/'cache/ijepa')
        stage('train', '--device', 'cuda')
        stage('evaluate', '--split', 'development', '--device', 'cuda')
        gate = json.loads((run/'development/reader_qualification.json').read_text())
        receipt['development_gates'] = gate
        if not gate['all_task_k2_qualified']:
            command([summary, '--run', run, '--split', 'development', '--audit', '--baseline-only'])
            receipt['status'] = 'completed_reader_gate_failed_no_students_no_confirmation'
        else:
            for split in ('train', 'validation'):
                stage('labels', '--split', split, '--device', 'cuda')
            command([students, 'fit', '--train', run/'exports/train', '--validation', run/'exports/validation',
                     '--output', run/'students', '--config', config])
            command([students, 'predict', '--models', run/'students', '--inputs', run/'exports/development',
                     '--output', run/'student_predictions/development.npz'])
            command([summary, '--run', run, '--split', 'development', '--audit'])
            # Frozen models and input boundaries are checked before generating CONF.
            stage('confirm-open', '--students-manifest', run/'students/manifest.json')
            receipt['confirmation_opened'] = True
            stage('encode', '--split', 'confirmation', '--device', 'cuda', '--cache-dir', BASE/'cache/ijepa')
            stage('evaluate', '--split', 'confirmation', '--device', 'cuda')
            command([students, 'predict', '--models', run/'students', '--inputs', run/'exports/confirmation',
                     '--output', run/'student_predictions/confirmation.npz'])
            command([summary, '--run', run, '--split', 'confirmation', '--audit'])
            receipt['status'] = 'completed_frozen_confirmation'
    except Exception as exc:
        receipt['status'] = 'execution_failed'; receipt['error'] = repr(exc)
        raise
    finally:
        write(run.parent/'execution.json', receipt)
        # Derived RGB/features remain intact on scratch; deterministic sources and
        # their hashes permit reconstruction. No user data or credentials bundled.
        package(run.parent, bundle, run)


def package(work, bundle, run):
    destination = work/'terminal-structures-results.zip'
    files, omitted = {}, {}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for root, prefix in ((run,'run'), (bundle,'source')):
            if not root.exists(): continue
            for p in sorted(root.rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts or '.git' in p.parts: continue
                name = prefix+'/'+str(p.relative_to(root))
                entry = {'sha256':sha(p), 'bytes':p.stat().st_size}
                if p.name in ('unique_rgb.bin','features.npy','features_raw.npy'):
                    omitted[name] = entry
                else:
                    files[name] = entry; z.write(p,name)
        for p in sorted(work.glob('*.json')):
            name = 'execution/'+p.name; files[name]={'sha256':sha(p),'bytes':p.stat().st_size};z.write(p,name)
        z.writestr('package_manifest.json',json.dumps({'files':files,'derived_data_preserved_on_scratch':omitted,
                   'scientific_status':'read execution and per-task reader gates; packaging is not a success claim'},indent=2))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP CRC failure')
    receipt={'path':str(destination),'sha256':sha(destination),'bytes':destination.stat().st_size,
             'included_files':len(files),'omitted_derived_files':len(omitted),'remote_originals_preserved':True}
    write(work/'package_receipt.json',receipt);print(json.dumps(receipt),flush=True)


def submit(archive, expected):
    if not re.fullmatch('[0-9a-f]{64}',expected) or sha(archive)!=expected: raise ValueError('Bundle hash mismatch')
    if not (BASE/'venv/bin/python').is_file() or not (BASE/'cache/ijepa').is_dir(): raise ValueError('Missing pinned environment/cache')
    env={k:os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','SHELL') if k in os.environ}
    active=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xy3257','-n',NAME,'-o','%i %T'],text=True,env=env,timeout=30).strip()
    if active: raise RuntimeError('Identically named active job exists: '+active)
    WORK.mkdir(parents=True,exist_ok=True)
    receipt=WORK/'submission_receipt.json'
    record={'status':'preparing','archive_sha256':expected,'created_utc':datetime.now(timezone.utc).isoformat(),
            'gpu_execution_verified':False,'confirmation_requires_k2_gate_and_frozen_students':True}
    with receipt.open('x') as f: json.dump(record,f,indent=2)
    try:
        bundle=WORK/'bundle';bundle.mkdir()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (bundle/member.filename).resolve().is_relative_to(bundle.resolve()):raise ValueError('Unsafe archive member')
            z.extractall(bundle)
        manifest=json.loads((bundle/'bundle_manifest.json').read_text())
        for name, expected_sha in manifest['files'].items():
            p=(bundle/name).resolve()
            if not p.is_relative_to(bundle.resolve()) or sha(p)!=expected_sha:raise ValueError('Source member mismatch: '+name)
        command=[str(BASE/'venv/bin/python'),str(bundle/'experiments/phase2/submit_terminal_structures.py'),
                 '--execute','--bundle',str(bundle),'--run',str(WORK/'run')]
        script='\n'.join(['#!/bin/bash','set -euo pipefail','umask 077',
            'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4',
            'export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1',
            shlex.join(command),''])
        (WORK/'run.sbatch').write_text(script)
        cmd=['/opt/slurm/bin/sbatch','--parsable','--export=ALL','--no-requeue','--job-name='+NAME,
             '--account=csci_ga_2271-2026fa','--partition=c12m85-a100-1','--gres=gpu:1',
             '--cpus-per-task=4','--mem=32G','--time=01:00:00','--chdir='+str(WORK),
             '--output='+str(WORK/'slurm-%j.log'),str(WORK/'run.sbatch')]
        record.update(status='submitting',command=cmd,bundle_manifest_sha256=sha(bundle/'bundle_manifest.json'))
        write(receipt,record)
        p=subprocess.run(cmd,text=True,capture_output=True,env=env,timeout=45)
        record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        if p.returncode or not re.fullmatch(r'\d+(;[^\s]+)?\s*',p.stdout):raise RuntimeError('Submission unknown; inspect scheduler before retry')
        record.update(status='submitted',job_id=p.stdout.strip().split(';')[0])
    except Exception as exc:
        record.update(status='failed_or_unknown',error=str(exc));raise
    finally:write(receipt,record);print(json.dumps(record,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path);p.add_argument('--sha256');p.add_argument('--execute',action='store_true')
    p.add_argument('--bundle',type=Path);p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.execute:
        if not a.bundle or not a.run:p.error('--bundle and --run required')
        execute(a.bundle.resolve(),a.run.resolve())
    else:
        if not a.archive or not a.sha256:p.error('--archive and --sha256 required')
        submit(a.archive,a.sha256)


if __name__=='__main__':main()
