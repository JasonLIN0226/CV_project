"""Bind the actual dependency-only recovery, retaining original evaluation source."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

OWN=Path(__file__).resolve().parent
ROOT=Path('/scratch/xl4358/phase1_far_reference')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_parent(rows,parent):
    row=rows.strip().split('|')
    if row[:4]!=[str(parent),'COMPLETED','0:0','csci_ga_2271-2026fa']:raise RuntimeError('dependency-only recovery not actually complete')


def validate_report(d):
    if d['status']!='PUBLIC_RELEASE_FILES_AND_DEPENDENCIES_PREPARED' or d['test_clips_declared']!=3840 or not d['no_redownload_or_reextract'] or d['model_inference_or_training']:
        raise RuntimeError('actual recovered release/environment incomplete')


def main():
    if os.environ.get('SLURM_JOB_ACCOUNT')!='csci_ga_2271-2026fa':raise RuntimeError('course required')
    for lock,base in [(OWN/'recovery_source_lock.json',OWN),(ROOT/'evaluation_source_lock.json',ROOT)]:
        for n,h in json.loads(lock.read_text())['files'].items():
            if sha(base/n)!=h:raise RuntimeError('frozen controller/evaluation source differs')
    if sha(ROOT/'evaluation_source_lock.json')!='a941ab106917c3efdb7516806b1f1a21faa870dc47d5e229e2fec10d512df0f9':raise RuntimeError('original scientific evaluation source differs')
    receipt=json.loads((OWN/'preparation_submission.json').read_text());parent=receipt['job_id']
    rows=subprocess.check_output(['/opt/slurm/bin/sacct','-j',str(parent),'-X','-n','-P','-o','JobID,State,ExitCode,Account'],text=True)
    validate_parent(rows,parent)
    p=OWN/'output'/('job_'+str(parent))/'preparation_report.json'; d=json.loads(p.read_text());validate_report(d)
    if d['recovery_source_sha256']!=sha(OWN/'recovery_source_lock.json'):raise RuntimeError('execution/recovery source differs')
    q=subprocess.check_output(['/opt/slurm/bin/squeue','-u','xl4358','-h','-o','%i %j %b'],text=True)
    if any(line.split()[0]!=os.environ['SLURM_JOB_ID'] for line in q.splitlines()):raise RuntimeError('another allocation exists')
    for n in ['evaluation_binding.json','evaluation_submission_attempt.json','evaluation_submission.json']:
        if (ROOT/n).exists():raise RuntimeError('unknown/existing evaluation marker; refuse retry')
    binding={'preparation_job':parent,'original_failed_preparation_job':7385,'original_cancelled_metadata_job':7386,'metadata_job':int(os.environ['SLURM_JOB_ID']),'preparation_report_path':str(p),'preparation_report_sha256':sha(p),'protocol_sha256':sha(ROOT/'reference_protocol.json'),'evaluation_source_sha256':sha(ROOT/'evaluation_source_lock.json'),'recovery_source_sha256':sha(OWN/'recovery_source_lock.json')}
    (ROOT/'evaluation_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    attempt=ROOT/'evaluation_submission_attempt.json'
    fd=os.open(attempt,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:json.dump({'status':'ATTEMPT_STARTED_UNKNOWN_OUTCOME',**binding},f)
    x=subprocess.run(['/opt/slurm/bin/sbatch','--parsable',str(ROOT/'evaluate_reference.sbatch')],capture_output=True,text=True)
    value=x.stdout.strip().split(';')[0]
    if x.returncode==0 and not value.isdigit():raise RuntimeError('unknown successful receipt; refuse retry')
    r={'status':'SUBMITTED_ACTUAL_RECEIPT' if x.returncode==0 else 'SUBMISSION_REJECTED_NO_JOB_ID','job_id':int(value) if x.returncode==0 else None,'returncode':x.returncode,'stdout':x.stdout.strip(),'stderr':x.stderr.strip(),**binding}
    (ROOT/'evaluation_submission.json').write_text(json.dumps(r,indent=2)+'\n');attempt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    if x.returncode:raise RuntimeError('actual GPU submission rejected')


if __name__=='__main__':main()
