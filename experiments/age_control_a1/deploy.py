"""One native CPU deployment/receipt. This file never runs a model or retries."""
import datetime,hashlib,json,re,socket,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path('/scratch/xl4358/phase1_age_control_a1')
ARCHIVE=Path('/scratch/xl4358/phase1_age_control_a1.zip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(job,expected_sha):
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    f=dict(t.split('=',1) for t in native.split() if '=' in t)
    assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa'
    assert f['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in f['AllocTRES']
    assert not ROOT.exists(), 'Existing execution root: inspect receipt, never resubmit blindly'
    queued=subprocess.check_output(['/opt/slurm/bin/squeue','-u','xl4358','-h','-o','%i %j %T'],text=True)
    assert not any('cv-age-control-a1' in line for line in queued.splitlines())
    assert sha(ARCHIVE)==expected_sha
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert all(n.startswith('phase1_age_control_a1/') and '..' not in Path(n).parts and not Path(n).is_absolute() for n in z.namelist())
        z.extractall(ROOT.parent)
    lock=json.loads((ROOT/'source_lock.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    (ROOT/'logs').mkdir()
    command=['/opt/slurm/bin/sbatch','--parsable',str(ROOT/'run.sbatch')]
    attempt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),deployment_job=job,allocation=native,
        command=command,source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),archive_sha256=expected_sha,GPU=0,training=False)
    with (ROOT/'submission_attempt.json').open('x') as f:json.dump(attempt,f,indent=2)
    result=subprocess.run(command,capture_output=True,text=True)
    receipt={**attempt,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'status':'UNKNOWN_DO_NOT_RETRY'}
    if result.returncode==0 and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_-]+)?\n?',result.stdout):
        receipt.update(status='ACTUAL_SUBMISSION_RECEIPT',job_id=int(result.stdout.strip().split(';')[0]))
    (ROOT/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    assert receipt['status']=='ACTUAL_SUBMISSION_RECEIPT'

if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
