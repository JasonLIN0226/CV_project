"""CPU guardian handshake; no allocations or data access on import."""
from heartbeat import fresh as heartbeat_fresh
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from watchdog import ACCOUNT, native_job, write


def require_live(root, source_sha, *, clock=time.time, require_peer=True):
    root = Path(root)
    receipt = json.loads((root/'watchdog_submission.json').read_text())
    report = json.loads((root/'watchdog_status.json').read_text())
    if (receipt['source_lock_sha256'] != source_sha or report['source_lock_sha256'] != source_sha
            or receipt['job'] != report['job'] or not heartbeat_fresh(report,clock=clock)):
        raise ValueError('Current frozen CPU guardian heartbeat is required before any GPU work')
    native = native_job(receipt['job'])
    if native['JobState'] != 'RUNNING' or native['Account'] != ACCOUNT or native.get('Restarts')!='0' or 'gres/gpu=' in native.get('AllocTRES', ''):
        raise ValueError('Guardian actual course CPU allocation is not RUNNING')
    if (root/'watchdog_terminal.json').exists() or report.get('reason'):
        raise ValueError('Guardian stopped this sequence; no automatic restart')
    if require_peer:
        peer_receipt=json.loads((root/'watchdog_cleanup_submission.json').read_text())
        peer_status=json.loads((root/'guardian_peer_status.json').read_text())
        peer=native_job(peer_receipt['job'])
        if (peer['JobState']!='RUNNING' or peer['Account']!=ACCOUNT or peer.get('Restarts')!='0'
                or 'gres/gpu=' in peer.get('AllocTRES','')
                or peer_status.get('job')!=peer_receipt['job']
                or peer_status.get('source_lock_sha256')!=source_sha
                or not heartbeat_fresh(peer_status,clock=clock)
                or (root/'guardian_peer_terminal.json').exists()):
            raise ValueError('Second independently RUNNING CPU guardian heartbeat required')
    return report


def ensure(root, protocol, stage_limits):
    root = Path(root)
    source_sha = hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()
    receipt = root/'watchdog_submission.json'
    if not receipt.exists():
        # Preserve unknown outcomes exactly like the scientific chain.
        with (root/'watchdog_submission_attempt.json').open('x') as f:
            json.dump(dict(source_lock_sha256=source_sha,unknown_do_not_retry=True),f)
        request = dict(source_lock_sha256=source_sha, GPU_cap_seconds=protocol['GPU_cap_seconds'],
            prior_GPU_charge_seconds=protocol['prior_settled_seconds'],
            startup_limit_seconds=300,stage_allocation_limits=stage_limits,
            GPU_job_names={stage:'p1-'+stage+'-'+source_sha[:8] for stage in stage_limits})
        write(root/'watchdog_request.json',request)
        args=['/opt/slurm/bin/sbatch','--parsable','--account='+ACCOUNT,
            '--partition=n2c48m24','--cpus-per-task=1','--mem=1500M',
            '--time=12:00:00','--signal=B:TERM@60','--no-requeue','--export=NIL',
            '--job-name=p1-budget-watch-'+source_sha[:8],
            '--output='+str(root/'watchdog-%j.out'),'--error='+str(root/'watchdog-%j.err'),
            str(root/'watchdog.sh')]
        raw=subprocess.run(args,check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        job=int(raw.split(';')[0])
        write(receipt,dict(job=job,source_lock_sha256=source_sha,stdout=raw,CPU_only=True))
    deadline=time.monotonic()+900
    while not (root/'watchdog_status.json').exists():
        if time.monotonic()>=deadline:
            raise TimeoutError('CPU watchdog did not become ready; no GPU was submitted')
        time.sleep(5)
    report = require_live(root,source_sha,require_peer=False)
    cleanup = root/'watchdog_cleanup_submission.json'
    if not cleanup.exists():
        with (root/'watchdog_cleanup_submission_attempt.json').open('x') as f:
            json.dump(dict(source_lock_sha256=source_sha,unknown_do_not_retry=True),f)
        guardian = json.loads(receipt.read_text())['job']
        args=['/opt/slurm/bin/sbatch','--parsable','--account='+ACCOUNT,
            '--partition=n2c48m24','--cpus-per-task=1','--mem=1500M','--time=12:00:00',
            '--no-requeue','--export=NIL',
            '--job-name=p1-cleanup-'+source_sha[:8],
            '--output='+str(root/'cleanup-%j.out'),'--error='+str(root/'cleanup-%j.err'),
            str(root/'cleanup.sh')]
        raw=subprocess.run(args,check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        write(cleanup,dict(job=int(raw.split(';')[0]),source_lock_sha256=source_sha,CPU_only=True))
    deadline=time.monotonic()+900
    while not (root/'guardian_peer_status.json').exists():
        if time.monotonic()>=deadline:
            raise TimeoutError('Independent peer not ready; no GPU submitted')
        require_live(root,source_sha,require_peer=False)
        time.sleep(5)
    return require_live(root,source_sha)


def guarded_allocation_rows(root, rows):
    """Charge startup wall captured before the node's batch process existed."""
    report=json.loads((Path(root)/'watchdog_status.json').read_text())
    result=[]
    for row in rows:
        entry=report['GPU_jobs'].get(str(row['job']),{})
        result.append({**row,'guarded_elapsed':max(row['elapsed'],entry.get('charged_seconds',0))})
    return result
