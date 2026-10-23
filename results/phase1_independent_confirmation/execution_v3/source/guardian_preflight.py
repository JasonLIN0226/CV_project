import json,hashlib,subprocess,sys
from pathlib import Path
from startup_client import require_live
from watchdog import native_job,write

def verify(root):
    root=Path(root);sha=hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()
    report=require_live(root,sha)
    native=native_job(report['job'])
    assert native.get('Requeue')=='0' and native.get('Restarts')=='0'
    cleanup=json.loads((root/'watchdog_cleanup_submission.json').read_text())
    c=native_job(cleanup['job'])
    assert c['Account']=='csci_ga_2271-2026fa' and c['JobState']=='PENDING'
    assert 'afterany:'+str(report['job']) in c['Dependency']
    assert c.get('Requeue')=='0' and 'gres/gpu=' not in c.get('AllocTRES','')
    # Standard-library metadata fixtures in the actual CPU environment.
    from watchdog import inspect_allocation,epoch
    s=epoch('2026-10-19T05:00:00')
    n=dict(Account='csci_ga_2271-2026fa',Partition='c12m85-a100-1',JobState='RUNNING',
      AllocTRES='cpu=12,mem=60G,gres/gpu=1',StartTime='2026-10-19T05:00:00',EndTime='Unknown',Restarts='0',Requeue='0')
    _,r=inspect_allocation({},n,s+181,startup_limit=180,allocation_limit=900,batch_started=False)
    assert r.startswith('Cloud startup')
    _,r=inspect_allocation({},n|{'Requeue':'1'},s+1,startup_limit=180,allocation_limit=900,batch_started=True)
    assert r.startswith('Scheduler requeue')
    _,r=inspect_allocation({},n|{'Restarts':'1'},s+1,startup_limit=180,allocation_limit=900,batch_started=True)
    assert r.startswith('Scheduler restarted')
    write(root/'guardian_native_preflight.json',dict(status='PASS_NATIVE_CPU_GUARDIAN_AND_CLEANUP',
      guardian=report['job'],cleanup=cleanup['job'],source_lock_sha256=sha,no_GPU=True,
      actual_GPU_startup_cancellation_not_exercised=True))
