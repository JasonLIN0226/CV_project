#!/usr/bin/env python3
"""One source/metadata handoff after audit7188; no model import or fitting.

The submitted GPU reader must independently repeat the cache and Slurm guards.
An attempted submission marker prevents a retry after an ambiguous outcome.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ENCODER=Path('/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame')
CACHE=Path('/scratch/xl4358/phase1_v5_real_encoding_frame/job_7187')
AUDITOR=Path('/scratch/xl4358/phase1_v5_a100_cache_audit_7187_frame')
TRAINING=Path('/scratch/xl4358/phase1_jepa_memory_v5_real_frame_training')
OUTPUT=Path('/scratch/xl4358/phase1_v5_real_frame_training')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def regular(path):
    path=Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Missing or symlinked explicit source/metadata payload')


def verify_lock(control,expected):
    path=control/'source_lock.json';regular(path)
    if sha(path)!=expected:raise ValueError('Frozen source lock changed')
    for rel,want in json.loads(path.read_text())['files'].items():
        name=Path(rel)
        if name.is_absolute() or '..' in name.parts:raise ValueError('Invalid source entry')
        file=control/name;regular(file)
        if sha(file)!=want:raise ValueError('Frozen source/input bytes changed: '+rel)


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run(control,expected):
    if (os.getenv('SLURM_JOB_ACCOUNT')!='csci_ga_2271-2026fa'
            or os.getenv('SLURM_JOB_PARTITION')!='n2c48m24'
            or not os.getenv('SLURM_JOB_ID') or not socket.gethostname().startswith('b-')):
        raise ValueError('Allocated course metadata handoff required')
    verify_lock(control,expected)
    if TRAINING.exists() or (control/'handoff_state.json').exists():
        raise ValueError('Existing handoff/reader root: inspect actual state; no duplicate submit')
    verify_lock(ENCODER,'cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e')
    verify_lock(AUDITOR,'fc02564863f8af84a79eed2bf0861b210252667c2f70411518a32a9eeb219d00')
    protocol=json.loads((control/'training_template.json').read_text())
    protocol.update(status='ACTUAL_CACHE_AUDIT_BOUND_FROZEN_GPU_READER_PROTOCOL',encoding_job_id=7187,
        encoding_source_lock_sha256=sha(ENCODER/'source_lock.json'),
        encoding_protocol_sha256=sha(ENCODER/'protocol.json'),actual_cache_audit_job_id=7188,
        actual_cache_audit_sha256=sha(CACHE/'full_cache_audit.json'),
        actual_encoding_report_sha256=sha(CACHE/'encoding_report.json'),
        actual_encoder_metadata_sha256=sha(CACHE/'encoder_metadata.json'),
        actual_cache_audit_source_lock_sha256=sha(AUDITOR/'source_lock.json'),
        actual_cache_audit_protocol_sha256=sha(AUDITOR/'protocol.json'))
    # Execute only the three tested, stdlib-only provenance guard functions.
    # No training module imports, Torch, features, basis fitting, or model calls.
    tree=ast.parse((control/'software/src/phase1/v5_real_training.py').read_text())
    names={'sha_file','check_regular','verify_external_cache_audit'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in selected}!=names:raise ValueError('Guard source incomplete')
    namespace={'Path':Path,'hashlib':hashlib,'json':json,'subprocess':subprocess}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<locked-cache-guards>','exec'),namespace)
    proof=namespace['verify_external_cache_audit'](protocol,CACHE,7187)
    account=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u',os.getenv('USER','xl4358'),
        '-o','%i|%T|%a|%b'],text=True)
    if any('csci_ga_2271-2026fa' in line and ('gpu:' in line or 'gpu=' in line)
           for line in account.splitlines()):
        raise ValueError('Another course GPU allocation is active; no parallel science')
    TRAINING.mkdir(mode=0o700);(TRAINING/'logs').mkdir()
    software=json.loads((control/'software_manifest.json').read_text())['files']
    for rel,want in software.items():
        file=control/'software'/rel
        if sha(file)!=want:raise ValueError('Reader software changed before binding')
        dest=TRAINING/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    atomic(TRAINING/'protocol.json',protocol)
    for name in ('full_cache_audit.json','encoding_report.json','encoder_metadata.json'):
        shutil.copyfile(CACHE/name,TRAINING/('actual_'+name))
    locked=[*software,'protocol.json','actual_full_cache_audit.json',
            'actual_encoding_report.json','actual_encoder_metadata.json']
    atomic(TRAINING/'source_lock.json',{'schema_version':1,
        'files':{name:sha(TRAINING/name) for name in locked},
        'scope':'One GPU reader after actual7187/7188 complete; original scientific math unchanged'})
    lock_sha=sha(TRAINING/'source_lock.json')
    batch=TRAINING/'submit_v5_real_frame_training.sh'
    batch.write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-real-reader
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output={TRAINING}/logs/train-%j.out
#SBATCH --error={TRAINING}/logs/train-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CONTROL={TRAINING}
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_train_real.py --control "$TASK_CONTROL" --source-lock-sha {lock_sha} --encoding-root {CACHE} --encoding-control {ENCODER} --encoding-job 7187 --output {OUTPUT}/job_${{SLURM_JOB_ID}}
''')
    batch.chmod(0o700)
    verify_lock(TRAINING,lock_sha)
    state={'status':'SOURCE_PROTOCOL_VERIFIED_SUBMISSION_ATTEMPT_STARTED',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'handoff_job_id':os.getenv('SLURM_JOB_ID'),'actual_cache_audit_sha256':sha(CACHE/'full_cache_audit.json'),
        'source_directory':str(TRAINING),'source_lock_sha256':lock_sha,
        'protocol_sha256':sha(TRAINING/'protocol.json'),'batch_sha256':sha(batch),
        'source_files':len(locked),'audit_all78_complete':proof['complete'],
        'no_CPU_model_fallback':True,'no_model_import_fit_or_training_in_handoff':True,
        'science_configuration_unchanged':True,'no_duplicate_submit':True}
    atomic(control/'handoff_state.json',state)
    result=subprocess.check_output(['/opt/slurm/bin/sbatch','--parsable','--no-requeue',str(batch)],text=True).strip()
    job=result.split(';')[0]
    if not job.isdigit():raise ValueError('Unknown submission outcome; inspect actual Slurm before any action')
    state.update(status='ACTUAL_GPU_READER_SUBMITTED',training_job_id=int(job),
                 output_directory=str(OUTPUT/('job_'+job)))
    atomic(control/'handoff_state.json',state)
    atomic(TRAINING/'submission.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--source-lock-sha',required=True);args=parser.parse_args()
    run(args.control,args.source_lock_sha)
