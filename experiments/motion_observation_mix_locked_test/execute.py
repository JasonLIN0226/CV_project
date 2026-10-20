"""Exactly once on one normal authenticated course CPU allocation."""
import datetime,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(job):
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(t.split('=',1) for t in native.split() if '=' in t)
    assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24' and n['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in n['AllocTRES']
    assert all(sha(ROOT/k)==v for k,v in json.loads((ROOT/'source_lock.json').read_text())['files'].items())
    receipt=dict(job=job,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),allocation=native,status='UNKNOWN_DO_NOT_RETRY',GPU=0,training=False,source_lock_sha256=sha(ROOT/'source_lock.json'))
    with (ROOT/'execution_attempt.json').open('x') as f:json.dump(receipt,f,indent=2)
    env=dict(os.environ,SLURM_JOB_ID=str(job),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    out=ROOT/'output'/f'job_{job}';out.parent.mkdir(exist_ok=True)
    with (ROOT/'execution.out').open('x') as stdout,(ROOT/'execution.err').open('x') as stderr:
        for script in ('evaluate.py','audit.py'):
            r=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-u',str(ROOT/script),str(out) if script=='audit.py' else str(job)],env=env,stdout=stdout,stderr=stderr);receipt[script+'_returncode']=r.returncode
            if r.returncode:break
        else:receipt['status']='COMPLETE_WITH_INDEPENDENT_AUDIT'
    receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(ROOT/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    assert receipt['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
if __name__=='__main__':main(int(sys.argv[1]))
