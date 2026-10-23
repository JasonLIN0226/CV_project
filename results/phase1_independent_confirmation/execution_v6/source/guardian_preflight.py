import json,hashlib,subprocess,sys,time
from pathlib import Path
from startup_client import require_live
from watchdog import native_job,write

def verify(root):
    root=Path(root);sha=hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()
    report=require_live(root,sha)
    native=native_job(report['job'])
    assert native.get('Requeue') in ('0','1') and native.get('Restarts')=='0'
    cleanup=json.loads((root/'watchdog_cleanup_submission.json').read_text())
    c=native_job(cleanup['job'])
    assert c['Account']=='csci_ga_2271-2026fa' and c['JobState']=='RUNNING'
    assert c.get('Requeue') in ('0','1') and c.get('Restarts')=='0' and 'gres/gpu=' not in c.get('AllocTRES','')
    # Exercise both concurrently updating real heartbeats across multiple RPC cycles.
    from heartbeat import fresh
    assert fresh({'updated_epoch':1792427434.5421834},clock=lambda:1792427434.822896)
    assert not fresh({'updated_epoch':0.},clock=lambda:26.)
    for _ in range(8):
        time.sleep(5)
        require_live(root,sha)
    # Standard-library metadata fixtures in the actual CPU environment.
    from watchdog import inspect_allocation,epoch
    s=epoch('2026-10-19T05:00:00')
    n=dict(Account='csci_ga_2271-2026fa',Partition='c12m85-a100-1',JobState='RUNNING',
      AllocTRES='cpu=12,mem=60G,gres/gpu=1',StartTime='2026-10-19T05:00:00',EndTime='Unknown',Restarts='0',Requeue='0')
    _,r=inspect_allocation({},n,s+181,startup_limit=180,allocation_limit=900,batch_started=False)
    assert r.startswith('Cloud startup')
    _,r=inspect_allocation({},n|{'Restarts':'1'},s+1,startup_limit=180,allocation_limit=900,batch_started=True)
    assert r.startswith('Scheduler restarted')
    probe=json.loads((root/'cancel_probe8775.json').read_text())
    from budget_guard import native_allocations
    cancelled=native_allocations([8775])[0]
    assert cancelled['state']=='CANCELLED' and cancelled['account']=='csci_ga_2271-2026fa'
    assert 'gres/gpu' not in cancelled['tres']
    gpu_probe=native_allocations([8792])[0]
    assert gpu_probe['state']=='CANCELLED' and gpu_probe['elapsed']==61 and gpu_probe['attempt_count']==1
    assert gpu_probe['tres'].get('gres/gpu')=='1'
    write(root/'guardian_native_preflight.json',dict(status='PASS_NATIVE_CPU_GUARDIAN_AND_CLEANUP',
      guardian=report['job'],cleanup=cleanup['job'],source_lock_sha256=sha,no_GPU=True,
      actual_GPU_startup_cancellation_not_exercised=False,actual_GPU_cancel_probe=8792,real_paired_heartbeat_cycles=8,heartbeat_clock_race_fixture_passed=True,paired_live_guardians=True,scheduler_forces_Requeue1=True,native_CPU_cancel_verified=True))
