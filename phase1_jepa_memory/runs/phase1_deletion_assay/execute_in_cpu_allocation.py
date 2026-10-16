"""Use the replacement course CPU allocation itself; never submit a GPU/job."""
import datetime,hashlib,json,os,socket,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path('/scratch/xl4358/phase1_deletion_assay')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    job=int(sys.argv[1]);archive_sha=sys.argv[2]
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    f=dict(x.split('=',1) for x in native.split() if '=' in x)
    assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa' and f['NodeList']==socket.gethostname().split('.')[0]
    assert int(f['NumCPUs'])>=4 and 'gres/gpu' not in f['AllocTRES'] and not ROOT.exists()
    mem=f['MinMemoryNode'];assert (mem.endswith('G') and float(mem[:-1])>=16) or (mem.endswith('M') and float(mem[:-1])>=16000)
    deadline=datetime.datetime.fromisoformat(f['EndTime']).replace(tzinfo=datetime.timezone.utc)
    def remaining():
        seconds=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-45
        if seconds<=0:raise TimeoutError('Actual CPU allocation deadline exhausted')
        return seconds
    assert remaining()>120
    archive=Path('/scratch/xl4358/phase1_deletion_assay.zip');assert sha(archive)==archive_sha
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(n.startswith('phase1_deletion_assay/') and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(ROOT.parent)
    lock=json.loads((ROOT/'source_lock.json').read_text());assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job,status='ACTUAL_EXISTING_CPU_ALLOCATION_EXECUTION',
        native_allocation=native,archive_sha256=archive_sha,source_lock_sha256=sha(ROOT/'source_lock.json'),
        execution_script_sha256=sha(__file__),training=False,GPU=0,new_Slurm_submissions=0)
    with (ROOT/'execution_attempt.json').open('x') as h:json.dump(receipt,h,indent=2)
    env={**os.environ,'SLURM_JOB_ID':str(job),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    with (ROOT/'execution.out').open('x') as output,(ROOT/'execution.err').open('x') as error:
        result=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-u',str(ROOT/'run_assay.py')],cwd=ROOT,env=env,stdout=output,stderr=error,timeout=remaining())
        receipt['assay_returncode']=result.returncode
        if result.returncode==0:
            env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            audit=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-u',str(ROOT/'independent_audit.py'),str(ROOT/'output'/f'job_{job}')],cwd=ROOT,env=env,stdout=output,stderr=error,timeout=remaining())
            receipt['independent_returncode']=audit.returncode
            receipt['status']='COMPLETE_WITH_INDEPENDENT_AUDIT' if audit.returncode==0 else 'AUDIT_FAILED_NO_AUTOMATIC_RETRY'
        else:receipt['status']='INFERENCE_FAILED_NO_AUTOMATIC_RETRY'
    (ROOT/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
