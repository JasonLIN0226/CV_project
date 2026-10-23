"""Independent CPU guardian B; no models, data, GPU allocation or retries."""
import json,hashlib,os,sys,time,signal,socket
from pathlib import Path
from heartbeat import fresh as heartbeat_fresh
from watchdog import ACCOUNT, native_job,owned_receipts,discover_submitting_jobs,cancel,write
root=Path(sys.argv[1]);sha=hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()
request=json.loads((root/'watchdog_request.json').read_text())
assert sha==request['source_lock_sha256']
me=native_job(int(os.environ['SLURM_JOB_ID']))
assert me['JobState']=='RUNNING' and me['Account']==ACCOUNT and 'gres/gpu=' not in me.get('AllocTRES','')
assert me.get('Restarts')=='0' and me['NodeList']==socket.gethostname().split('.')[0]
print('JOB',os.environ['SLURM_JOB_ID'],'HOST',socket.gethostname(),'IP',socket.gethostbyname(socket.gethostname()),flush=True)
print(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text(),flush=True)
stop={'requested':False}
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.update(requested=True))
known={};reason=None
try:
 while True:
  now=time.time()
  known.update(owned_receipts(root,sha,request['stage_allocation_limits']))
  known.update(discover_submitting_jobs(request))
  receipt=json.loads((root/'watchdog_submission.json').read_text())
  guardian=native_job(receipt['job'])
  report=json.loads((root/'watchdog_status.json').read_text())
  if (guardian['JobState']!='RUNNING' or guardian.get('Restarts')!='0'
      or guardian['Account']!=ACCOUNT or 'gres/gpu=' in guardian.get('AllocTRES','')
      or report.get('source_lock_sha256')!=sha
      or not heartbeat_fresh(report)):
   reason='Guardian A ended/restarted or heartbeat expired';break
  write(root/'guardian_peer_status.json',dict(job=int(os.environ['SLURM_JOB_ID']),source_lock_sha256=sha,
    updated_epoch=time.time(),guardian=receipt['job'],CPU_only=True))
  if stop['requested'] or (root/'watchdog_terminal.json').exists():
   reason='Guardian shutdown or primary termination';break
  if (root/'watchdog_done.json').exists():
   reason='Sequence complete';break
  time.sleep(5)
finally:
 # cancel() rechecks user/course/partition and never touches an unrelated job.
 for job in known:cancel(job)
 write(root/'guardian_peer_terminal.json',dict(reason=reason,jobs=known,
   native_terminal_verification_still_required=True,CPU_only=True))
