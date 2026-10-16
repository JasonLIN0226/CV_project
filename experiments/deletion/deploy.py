"""One deployment and one receipt; no GPU, no training, no implicit retry."""
import datetime,hashlib,json,os,re,socket,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path('/scratch/xl4358/phase1_deletion_assay')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    job=int(sys.argv[1]);expected_archive=sys.argv[2]
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    f=dict(x.split('=',1) for x in native.split() if '=' in x)
    assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa' and f['NodeList']==socket.gethostname().split('.')[0]
    assert 'gres/gpu' not in f['AllocTRES'] and not ROOT.exists()
    active=subprocess.check_output(['/opt/slurm/bin/squeue','-u','xl4358','-h','-o','%i %j %b'],text=True)
    assert not any('gpu' in x.lower() or 'delete-assay' in x.lower() for x in active.splitlines())
    archive=Path('/scratch/xl4358/phase1_deletion_assay.zip');assert sha(archive)==expected_archive
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(n.startswith('phase1_deletion_assay/') and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(ROOT.parent)
    lock=json.loads((ROOT/'source_lock.json').read_text());assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    (ROOT/'logs').mkdir()
    attempt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),deployment_job=job,allocation=native,
        archive_sha256=expected_archive,source_lock_sha256=sha(ROOT/'source_lock.json'),GPU=0,training=False,
        command=['/opt/slurm/bin/sbatch','--parsable',str(ROOT/'assay.sbatch')])
    with (ROOT/'submission_attempt.json').open('x') as h:json.dump(attempt,h,indent=2)
    result=subprocess.run(attempt['command'],text=True,capture_output=True)
    receipt={**attempt,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    if result.returncode==0 and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_-]+)?\n?',result.stdout):
        receipt.update(job_id=int(result.stdout.strip().split(';')[0]),status='ACTUAL_SUBMISSION_RECEIPT')
    else:receipt['status']='UNKNOWN_DO_NOT_RETRY'
    (ROOT/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    assert receipt['status']=='ACTUAL_SUBMISSION_RECEIPT'
if __name__=='__main__':main()
