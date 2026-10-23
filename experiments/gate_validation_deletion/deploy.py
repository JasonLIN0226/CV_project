"""One explicit execution on a current course CPU allocation; never submits jobs."""
import datetime,hashlib,json,re,socket,subprocess,sys,zipfile,os
from pathlib import Path
ROOT=Path('/scratch/xl4358/phase1_gate_validation_deletion')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(job,archive_sha):
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
 fields=dict(t.split('=',1) for t in native.split() if '=' in t)
 assert fields['JobState']=='RUNNING' and fields['Partition']=='n2c48m24' and fields['Account']=='csci_ga_2271-2026fa'
 assert fields['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in fields['AllocTRES']
 archive=ROOT.with_suffix('.zip');assert sha(archive)==archive_sha and not ROOT.exists()
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and all(n.startswith(ROOT.name+'/') and '..' not in Path(n).parts and not Path(n).is_absolute() for n in z.namelist());z.extractall(ROOT.parent)
 lock=json.loads((ROOT/'source_lock.json').read_text());assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
 receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=native,archive_sha256=archive_sha,source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),GPU=0,training=False,status='UNKNOWN_DO_NOT_RETRY')
 with (ROOT/'execution_attempt.json').open('x') as f:json.dump(receipt,f,indent=2)
 env=dict(os.environ,SLURM_JOB_ID=str(job),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',HF_HOME='/scratch/xl4358/phase1_hf_cache')
 with (ROOT/'execution.out').open('x') as stdout,(ROOT/'execution.err').open('x') as stderr:
  r=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-u',str(ROOT/'evaluate.py'),str(job)],env=env,stdout=stdout,stderr=stderr)
  receipt['diagnostic_returncode']=r.returncode
  if r.returncode==0:
   env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
   a=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-u',str(ROOT/'audit.py'),str(ROOT/'output'/f'job_{job}')],env=env,stdout=stdout,stderr=stderr);receipt['independent_returncode']=a.returncode
   if a.returncode==0:receipt['status']='COMPLETE_WITH_INDEPENDENT_AUDIT'
 receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(ROOT/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
 assert receipt['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
