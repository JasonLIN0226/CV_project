"""Once on a verified course CPU allocation. Existing raw results stay read-only."""
import datetime,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(job):
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    n=dict(x.split('=',1) for x in native.split() if '=' in x)
    assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24'
    assert n['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in n['AllocTRES']
    lock=json.loads((ROOT/'source_lock.json').read_text());assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    receipt=dict(job=job,allocation=native,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='UNKNOWN_DO_NOT_RETRY',GPU=0,model_calls=0,source_lock_sha256=sha(ROOT/'source_lock.json'))
    with (ROOT/'execution_attempt.json').open('x') as f:json.dump(receipt,f,indent=2)
    env=dict(os.environ,SLURM_JOB_ID=str(job),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    out=ROOT/'output'/f'job_{job}';out.parent.mkdir(exist_ok=True)
    with (ROOT/'execution.out').open('x') as stdout,(ROOT/'execution.err').open('x') as stderr:
        for script in ('analyze.py','audit.py'):
            r=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-u',str(ROOT/script),str(out)],env=env,stdout=stdout,stderr=stderr)
            receipt[script+'_returncode']=r.returncode
            if r.returncode:break
        else:receipt['status']='COMPLETE_WITH_INDEPENDENT_AUDIT'
    receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    assert receipt['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
if __name__=='__main__':main(int(sys.argv[1]))
