#!/usr/bin/env python3
"""After actual7251 completion, prove all six files, bind, and submit once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

TRAINING=Path('/scratch/xl4358/phase1_jepa_memory_v6_real_wd020')
RESULT=Path('/scratch/xl4358/phase1_v6_real_wd020/job_7251')
TARGET=Path('/scratch/xl4358/phase1_v6_real_diagnostics_7251')
OUTPUT=Path('/scratch/xl4358/phase1_v6_real_diagnostic_results')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(control,expected):
    path=control/'source_lock.json'
    if any(p.is_symlink() for p in (path,*path.parents)) or sha(path)!=expected:
        raise ValueError('Locked metadata/software source differs')
    for name,want in json.loads(path.read_text())['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid source entry')
        file=control/rel
        if any(p.is_symlink() for p in (file,*file.parents)) or sha(file)!=want:
            raise ValueError('Locked source changed: '+name)


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run(control,expected):
    if (os.getenv('SLURM_JOB_ACCOUNT')!='csci_ga_2271-2026fa' or not os.getenv('SLURM_JOB_ID')
            or os.getenv('SLURM_JOB_PARTITION')!='n2c48m24' or not socket.gethostname().startswith('b-')):
        raise ValueError('Actual allocated course model proof/metadata required')
    check(control,expected)
    if TARGET.exists() or (control/'submission_attempt.json').exists():
        raise ValueError('Existing formal source or submit marker; inspect state, no duplicate')
    record=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7251','-n','-P',
        '--format=JobIDRaw,State,ExitCode,Account,Partition,AllocTRES'],text=True).strip().split('|')
    if record[:5]!=['7251','COMPLETED','0:0','csci_ga_2271-2026fa','c12m85-a100-1'] or 'gres/gpu=1' not in record[5].split(','):
        raise ValueError('Actual GPU reader parent is not complete on declared resources')
    check(TRAINING,'24116c56a15ead3f47d5109be0a389eab179f01fbc63ba96fb69e9813bf3b5e5')
    sys.path[:0]=[str(control/'software/src'),str(control/'software/scripts')]
    import torch
    from phase1.v5_real_diagnostics import completion_proof
    torch.set_num_threads(4)
    # Read only current checkpoint bytes/tensors and saved training CSV. No fit,
    # model construction/forward, features, or scientific diagnostic here.
    files={f'seed_{seed}/{name}':sha(RESULT/f'seed_{seed}'/name)
        for seed in (17,29,43) for name in ('best.pt','latest.pt','training.csv','checkpoint_proof.json')}
    proof=completion_proof(RESULT,files,7251,.20)
    atomic(control/'actual_all6_model_proof.json',proof)
    protocol=json.loads((control/'diagnostic_template.json').read_text())
    protocol.update(status='ACTUAL_ALL6_MODEL_HASH_BOUND_FROZEN_REAL_DIAGNOSTICS',
        actual_checkpoint_files=files,actual_training_report_sha256=sha(RESULT/'training_report.json'),
        actual_fit_proof_sha256=sha(RESULT/'fit_proof.json'),
        actual_model_proof_sha256=sha(control/'actual_all6_model_proof.json'))
    protocol.pop('no_new_science_submitted',None)
    TARGET.mkdir(mode=0o700);(TARGET/'logs').mkdir()
    manifest=json.loads((control/'software_manifest.json').read_text())['files']
    for name,want in manifest.items():
        source=control/'software'/name
        if sha(source)!=want:raise ValueError('Frozen diagnostic software changed')
        destination=TARGET/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    atomic(TARGET/'protocol.json',protocol)
    shutil.copyfile(control/'actual_all6_model_proof.json',TARGET/'actual_all6_model_proof.json')
    for name in ('training_report.json','fit_proof.json'):
        shutil.copyfile(RESULT/name,TARGET/('actual_'+name))
    bound=[*manifest,'protocol.json','actual_all6_model_proof.json','actual_training_report.json','actual_fit_proof.json']
    atomic(TARGET/'source_lock.json',{'schema_version':1,'files':{name:sha(TARGET/name) for name in bound},
        'scope':'Frozen GPU-reader-derived diagnostics, no new fitting/training/labels/sealed'})
    locked_sha=sha(TARGET/'source_lock.json')
    batch=TARGET/'submit_frozen_real_diagnostics.sh'
    batch.write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=cv-v6-real-diagnostics
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output={TARGET}/logs/diagnostic-%j.out
#SBATCH --error={TARGET}/logs/diagnostic-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TASK_CONTROL={TARGET}
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_run_frozen_real_diagnostics.py --control "$TASK_CONTROL" --source-lock-sha {locked_sha} --output {OUTPUT}/job_${{SLURM_JOB_ID}}
''');batch.chmod(0o700);check(TARGET,locked_sha)
    queue=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u',os.getenv('USER','xl4358'),'-o','%i|%T|%a|%j'],text=True)
    if any('csci_ga_2271-2026fa' in line and ('cv-v6-real-wd020' in line or 'cv-v6-real-diagnostics' in line)
           for line in queue.splitlines()):
        raise ValueError('Existing reader/diagnostic science active; no parallel or duplicate submit')
    submission={'status':'DIAGNOSTIC_SUBMIT_ATTEMPT_STARTED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'handoff_job_id':os.getenv('SLURM_JOB_ID'),'actual_parent':7251,'source_directory':str(TARGET),
        'source_lock_sha256':locked_sha,'protocol_sha256':sha(TARGET/'protocol.json'),'batch_sha256':sha(batch),
        'actual_model_proof_sha256':sha(TARGET/'actual_all6_model_proof.json'),
        'all6_verified_before_scientific_inference':True,'no_new_fit_or_training':True,
        'diagnostic_scope':'Original fixed frozen health/deletion/replay CPU diagnostics; not CPU model encoding/training fallback',
        'labels_unavailable_not_guessed':True,'no_duplicate_submit':True}
    atomic(control/'submission_attempt.json',submission)
    result=subprocess.check_output(['/opt/slurm/bin/sbatch','--parsable','--no-requeue',str(batch)],text=True).strip().split(';')[0]
    if not result.isdigit():raise ValueError('Unknown submission outcome; do not retry')
    submission.update(status='ACTUAL_FROZEN_DIAGNOSTIC_SUBMITTED',diagnostic_job_id=int(result),
                      output_directory=str(OUTPUT/('job_'+result)))
    atomic(control/'submission_attempt.json',submission);atomic(TARGET/'submission.json',submission)
    print(json.dumps(submission),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--source-lock-sha',required=True)
    args=p.parse_args();run(args.control,args.source_lock_sha)
