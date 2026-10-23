"""Preparation only: CPU-side protection for one frozen confirmation sequence.

No model/data imports, allocation or Slurm calls occur on import. A deployment
must bind its actual prior GPU ledger and frozen receipt directory first.
"""
from heartbeat import fresh as heartbeat_fresh
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ACCOUNT = 'csci_ga_2271-2026fa'
GPU_PARTITION = 'c12m85-a100-1'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'OUT_OF_MEMORY'}


def epoch(value):
    if value in ('Unknown', 'None', '', None):
        return None
    return datetime.datetime.fromisoformat(value.removesuffix('Z')).replace(tzinfo=datetime.timezone.utc).timestamp()


def inspect_allocation(previous, native, now, *, startup_limit, allocation_limit,
                       batch_started=None):
    """Retain the earliest allocated start even if Slurm resets it on restart.

    The retained wall is an explicit conservative charge, never evidence of
    useful GPU computation. A restart stops this sequence instead of retrying.
    """
    if native['Account'] != ACCOUNT or native['Partition'] != GPU_PARTITION:
        raise ValueError('Watchdog refuses an unrelated account or partition')
    state = native['JobState'].split('+')[0]
    allocated = 'gres/gpu=1' in native.get('AllocTRES', '').split(',')
    if 'gres/gpu=' in native.get('AllocTRES', '') and not allocated:
        raise ValueError('Only the authorized single GPU can be watched')
    current = dict(previous)
    start = epoch(native.get('StartTime'))
    if allocated:
        if start is None:
            raise ValueError('Allocated GPU has no native start timestamp')
        current['first_allocated_start'] = min(start, current.get('first_allocated_start', start))
    first = current.get('first_allocated_start')
    end = epoch(native.get('EndTime')) if state in TERMINAL else None
    if first is not None:
        # PENDING after a restart remains conservatively charged until native
        # terminal reconciliation. It must not disappear from the budget.
        current['charged_seconds'] = max(current.get('charged_seconds', 0), math.ceil(max(0., (end or now)-first)))
    else:
        current['charged_seconds'] = 0
    current.update(native=native, observed_UTC=datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat())
    reason = None
    if state not in TERMINAL:
        if int(native.get('Restarts', '0')):
            reason = 'Scheduler restarted an allocation declared no-requeue'
        elif first is not None and now-first >= startup_limit and (
                state == 'CONFIGURING' or batch_started is False):
            reason = 'Cloud startup exceeded the frozen allocation allowance'
        elif first is not None and current['charged_seconds'] >= allocation_limit-30:
            reason = 'Whole allocation deadline, including startup, reached'
    return current, reason


def write(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def native_job(job):
    raw = subprocess.run(['/opt/slurm/bin/scontrol', 'show', 'job', '-o', str(job)],
                         check=True, capture_output=True, text=True, timeout=15).stdout
    fields = dict(x.split('=', 1) for x in raw.split() if '=' in x)
    if fields.get('JobId') != str(job):
        raise ValueError('Native job identity mismatch')
    return fields


def cancel(job):
    import pwd
    native = native_job(job)
    user = pwd.getpwuid(os.getuid()).pw_name
    if (native.get('Account') != ACCOUNT or native.get('Partition') != GPU_PARTITION
            or native.get('UserId', '').split('(')[0] != user):
        raise ValueError('Refuse cancellation outside the current user/course GPU receipts')
    if native['JobState'].split('+')[0] in TERMINAL:
        return
    subprocess.run(['/opt/slurm/bin/scancel', str(job)], check=True, timeout=15)


def owned_receipts(root, source_sha, stages):
    result = {}
    for stage in stages:
        path = root/'receipts'/(stage+'_submission.json')
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        if receipt['stage'] != stage or receipt['source_lock_sha256'] != source_sha:
            raise ValueError('Receipt is not from this frozen execution')
        job = receipt['job']
        if not isinstance(job, int) or job in result:
            raise ValueError('Distinct actual GPU receipts required')
        result[job] = stage
    return result


def discover_submitting_jobs(request):
    """Observe an sbatch whose acknowledgement may have been interrupted.

    Exact, frozen, execution-specific names are used only to discover existing
    jobs. This never retries a submission or creates a new receipt.
    """
    import pwd
    user = pwd.getpwuid(os.getuid()).pw_name
    result = {}
    for stage, name in request['GPU_job_names'].items():
        raw = subprocess.run(['/opt/slurm/bin/squeue', '-h', '-u', user,
            '--name='+name, '-o', '%i'], capture_output=True, text=True, check=True,
            timeout=15).stdout
        jobs = raw.split()
        if len(jobs) > 1 or any(not j.isdigit() for j in jobs):
            raise ValueError('Duplicate or unrecognized submission identity; no implicit recovery')
        for job in jobs:
            native = native_job(int(job))
            if native.get('JobName') != name or native.get('Account') != ACCOUNT:
                raise ValueError('Discovered allocation name/account mismatch')
            result[int(job)] = stage
    return result


def main(root):
    import hashlib
    root = Path(root).resolve()
    request = json.loads((root/'watchdog_request.json').read_text())
    source_sha = hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()
    if request['source_lock_sha256'] != source_sha or request['GPU_cap_seconds'] != json.loads((root/'protocol.json').read_text())['GPU_cap_seconds']:
        raise ValueError('Frozen watchdog request/authorization mismatch')
    from cpu_incarnation import claim as claim_cpu_incarnation, require as require_cpu_incarnation
    me = native_job(int(os.environ['SLURM_JOB_ID']))
    if me['JobState'] != 'RUNNING' or me['Account'] != ACCOUNT or 'gres/gpu=' in me.get('AllocTRES', ''):
        raise ValueError('Watchdog must run in an actual course CPU allocation')
    import socket
    if me['NodeList'] != socket.gethostname().split('.')[0]:
        raise ValueError('Guardian public identity is not its allocated node')
    claim_cpu_incarnation(root,'primary',me,source_sha)
    print('JOB', os.environ['SLURM_JOB_ID'], 'HOST', socket.gethostname(),
          'IP', socket.gethostbyname(socket.gethostname()), flush=True)
    print(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text(), flush=True)
    stop = {'requested': False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.update(requested=True))
    history, known = {}, {}
    reason = None
    try:
        while True:
            now = time.time()
            require_cpu_incarnation(root,'primary',int(os.environ['SLURM_JOB_ID']),native_job(int(os.environ['SLURM_JOB_ID'])),source_sha)
            known.update(owned_receipts(root, source_sha, request['stage_allocation_limits']))
            known.update(discover_submitting_jobs(request))
            for job, stage in known.items():
                previous = history.get(str(job), {})
                if previous.get('terminal'):
                    continue
                native = native_job(job)
                attempt_path = root/'stage_attempts'/(stage+'.json')
                batch_started = False
                if attempt_path.exists():
                    attempt = json.loads(attempt_path.read_text())
                    if attempt.get('job') != job or attempt.get('stage') != stage:
                        raise ValueError('Stage entrypoint proof does not match its actual receipt')
                    batch_started = True
                entry, why = inspect_allocation(previous, native, now,
                    startup_limit=request['startup_limit_seconds'],
                    allocation_limit=request['stage_allocation_limits'][stage],
                    batch_started=batch_started)
                entry['stage'] = stage
                entry['terminal'] = native['JobState'].split('+')[0] in TERMINAL
                history[str(job)] = entry
                if why:
                    reason = why
            peer_path=root/'watchdog_cleanup_submission.json'
            if peer_path.exists():
                peer_receipt=json.loads(peer_path.read_text())
                peer=native_job(peer_receipt['job'])
                peer_status_path=root/'guardian_peer_status.json'
                peer_status=json.loads(peer_status_path.read_text()) if peer_status_path.exists() else {}
                if known:
                    require_cpu_incarnation(root,'peer',peer_receipt['job'],peer,source_sha)
                if known and (peer['JobState']!='RUNNING'
                        or peer_status.get('source_lock_sha256')!=source_sha
                        or not heartbeat_fresh(peer_status)):
                    reason='Independent CPU peer lost/restarted or heartbeat expired'
            total = request['prior_GPU_charge_seconds'] + sum(x['charged_seconds'] for x in history.values())
            report = dict(job=int(os.environ['SLURM_JOB_ID']), source_lock_sha256=source_sha,
                updated_epoch=time.time(), native_CPU=me, GPU_jobs=history, charged_GPU_seconds=total,
                cap_seconds=request['GPU_cap_seconds'], reason=reason, no_model_or_data_access=True)
            write(root/'watchdog_status.json', report)
            if total >= request['GPU_cap_seconds']-30:
                reason = 'Cumulative GPU cap approached; preserve evidence and stop'
            if stop['requested'] or (root/'stage_failures').exists() and any((root/'stage_failures').glob('*.json')):
                reason = reason or 'Watchdog termination or sequence failure'
            if reason:
                break
            if (root/'watchdog_done.json').exists():
                if any(not x.get('terminal') for x in history.values()):
                    raise ValueError('Cannot release watchdog with an unfinished GPU receipt')
                write(root/'watchdog_terminal.json', {**report, 'status': 'ALL_REGISTERED_GPU_ALLOCATIONS_ENDED'})
                return
            time.sleep(5)
    except BaseException:
        # Fail closed: cancellation is limited to this frozen sequence's
        # independently identity-checked receipts, never arbitrary user jobs.
        from cleanup_receipts import cleanup_all
        from budget_guard import native_allocations
        cleanup_all(known, accounting=native_allocations, strict_cancel=cancel,
                    persist=lambda rows: write(root/'watchdog_cleanup_attempts.json', rows))
        raise
    for job in known:
        if not history.get(str(job), {}).get('terminal'):
            cancel(job)
    write(root/'watchdog_terminal.json', dict(status='SEQUENCE_STOP_REQUESTED', reason=reason,
        GPU_jobs=history, cancellation_native_verification_still_required=True))


if __name__ == '__main__':
    main(sys.argv[1])
