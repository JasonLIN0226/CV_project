"""Shared actual-allocation budget, with a forecast for the WHOLE remaining trial.

Importing this module never queries Slurm, allocates a node, or opens fresh data.
The settled pre-confirmation ledger is evidence, not another per-stage allowance.
"""
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time

CAP = 54051
PRIOR = 39651
TASKS = ('gate', 'motion')
SEEDS = (17, 29, 43)
RATES = {
    ('gate', 'reference'): 2.572098377,
    ('motion', 'reference'): 2.622685901,
    ('gate', 'candidate'): 5.667439204999994,
    ('motion', 'candidate'): 3.5664218639999774,
}
TOTAL_FRAMES = 198656
ENCODER_RATE = 40.451236
FIXED_RESERVE = 968.295233886
MAP_FIT_SECONDS = 27.
DEV_EQUIVALENCE_SECONDS = 300.


class BudgetStop(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def start_epoch(text):
    if text in ('Unknown', 'None', ''):
        raise ValueError('Actual allocation start is missing')
    return datetime.datetime.fromisoformat(text.removesuffix('Z')).replace(
        tzinfo=datetime.timezone.utc).timestamp()


def native_allocations(job_ids):
    if not job_ids:
        return []
    if len(set(job_ids)) != len(job_ids) or any(not str(j).isdigit() for j in job_ids):
        raise ValueError('Unique actual receipt IDs required')
    import pwd
    user = pwd.getpwuid(os.getuid()).pw_name
    out = subprocess.run(['/opt/slurm/bin/sacct', '-D', '-X', '-n', '-P', '-u', user,
        '-A', 'csci_ga_2271-2026fa', '-S', '2026-10-01', '-E', '2026-11-01', '-j',
        ','.join(map(str, job_ids)), '--format=JobID,State,ExitCode,ElapsedRaw,Start,End,Account,AllocTRES'],
        check=True, capture_output=True, text=True, timeout=20,
        env={**os.environ, 'TZ':'UTC'}).stdout
    grouped = {}
    for line in out.splitlines():
        fields = line.split('|')
        if len(fields) == 9 and fields[-1] == '':
            fields.pop()
        if len(fields) != 8:
            raise ValueError('Unexpected native allocation row')
        job, state, exitcode, elapsed, start, end, account, tres = fields
        if account != 'csci_ga_2271-2026fa' or int(job) not in job_ids:
            raise ValueError('Native allocation account or receipt mismatch')
        if start not in ('Unknown','None','') and not start.startswith('2026-10-'):
            raise ValueError('Reused historical JobID rejected')
        row = dict(job=int(job), state=state.split()[0], exitcode=exitcode,
            elapsed=int(elapsed), start=start, end=end, account=account,
            tres=dict(x.split('=',1) for x in tres.split(',') if '=' in x), native=line)
        grouped.setdefault(int(job), []).append(row)
    if set(grouped) != set(job_ids):
        raise ValueError('Native receipts incomplete')
    result = []
    for job in job_ids:
        attempts = sorted(grouped[job], key=lambda r: r['start'] if r['start'] not in ('Unknown','None','') else '9999')
        previous_end = None
        for row in attempts:
            if row['start'] in ('Unknown','None',''):
                if row['elapsed']:
                    raise ValueError('Allocated duration has no start')
                continue
            begin = start_epoch(row['start'])
            if previous_end is not None and begin < previous_end:
                raise ValueError('Overlapping native allocation intervals')
            if row['end'] not in ('Unknown','None',''):
                previous_end = start_epoch(row['end'])
                if abs(previous_end-begin-row['elapsed']) > 1:
                    raise ValueError('Native elapsed disagrees with allocation interval')
        last = attempts[-1]
        prior = sum(r['elapsed'] for r in attempts[:-1])
        result.append({**last, 'elapsed':prior+last['elapsed'],
            'current_attempt_elapsed':last['elapsed'], 'prior_attempt_elapsed':prior,
            'attempt_count':len(attempts), 'attempts':attempts})
    return result


def charged_seconds(rows, now):
    """Count every allocated GPU second, even failed or scientifically unused."""
    charged, active = 0., []
    for row in rows:
        if row['account'] != 'csci_ga_2271-2026fa':
            raise ValueError('Wrong course account')
        gpu = row['tres'].get('gres/gpu')
        if gpu is None:
            never_allocated = (row['elapsed'] == 0 and row['start'] in ('Unknown','None','')
                               and row['state'] in ('CANCELLED','FAILED'))
            if row['state'] not in ('PENDING', 'CONFIGURING') and not never_allocated:
                raise ValueError('GPU receipt does not have actual GPU allocation')
            charged += row.get('prior_attempt_elapsed',0)
            continue
        if gpu != '1':
            raise ValueError('Only a single allocated GPU is authorized')
        if row['state'] in ('RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
            if row['state'] == 'SUSPENDED':
                raise BudgetStop('Unexpected suspended GPU allocation; no optimistic accounting')
            wall = row.get('prior_attempt_elapsed',0) + max(row.get('current_attempt_elapsed',row['elapsed']), now - start_epoch(row['start']))
            active.append(row['job'])
            charged += math.ceil(wall)
        elif row['state'] == 'PENDING':
            charged += row.get('prior_attempt_elapsed',0)
            continue
        else:
            if row['end'] in ('Unknown', 'None', ''):
                raise ValueError('Ended GPU allocation lacks native terminal evidence')
            charged += row['elapsed']
    if len(active) > 1:
        raise BudgetStop('Parallel GPU allocations exceed the single-GPU contract')
    return charged, active


class SequenceGuard:
    status = 'NATIVE_RECONCILED_COMPLETE_CONFIRMATION_GUARD'
    cap_seconds = CAP

    def __init__(self, root, *, query=native_allocations, clock=time.time):
        self.root, self.query, self.clock = Path(root), query, clock
        self.path = self.root / 'sequence_budget.json'
        self.state = read(self.path)
        if (self.state['prior_settled_seconds'] != PRIOR
                or self.state['cap_seconds'] != CAP
                or self.state['total_frames'] != TOTAL_FRAMES):
            raise ValueError('Original shared budget/full-scope contract changed')
        if self.state.get('native_prior_verified') is not True:
            raise ValueError('Native prior settlement must be reconciled before fresh execution')

    def save(self):
        write(self.path, self.state)

    def remaining_raw(self):
        training = 0.
        for role in ('reference', 'candidate'):
            for task in TASKS:
                rate = max(RATES[task, role], self.state.get('epoch_rate', {}).get(task+'/'+role, 0.))
                for seed in SEEDS:
                    key = f'{role}/{task}/{seed}'
                    epochs = self.state.get('epochs', {}).get(key, 0)
                    if not isinstance(epochs, int) or not 0 <= epochs <= 120:
                        raise ValueError('Invalid retained epoch boundary')
                    training += (120 - epochs) * rate
        encoded = self.state.get('encoded_frames', 0)
        if not 0 <= encoded <= TOTAL_FRAMES:
            raise ValueError('Encoded-frame accounting exceeds complete scope')
        rate = min(ENCODER_RATE, self.state.get('encoder_frames_per_second', ENCODER_RATE))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Invalid actual encoder rate')
        encoding = 0. if self.state.get('full_encoding_audited',False) else (TOTAL_FRAMES - encoded) / rate
        maps = 0. if self.state.get('maps_locked', False) else MAP_FIT_SECONDS
        preflight = 0. if self.state.get('native_GPU_equivalence_done',False) else DEV_EQUIVALENCE_SECONDS
        return dict(training=training, encoding=encoding, maps=maps, native_preflight=preflight)

    def check(self, allocation_start=None, task=None, seed=None, role=None, retained_epochs=None):
        receipts = self.state.get('GPU_job_ids', [])
        rows = self.query(receipts)
        charge, active = charged_seconds(rows, self.clock())
        if self.query is native_allocations:
            from startup_client import require_live, guarded_allocation_rows
            execution = self.root.parent
            import hashlib
            source_sha = hashlib.sha256((execution/'source_lock.json').read_bytes()).hexdigest()
            require_live(execution,source_sha)
            guarded = guarded_allocation_rows(execution,rows)
            charge = sum(r['guarded_elapsed'] for r in guarded)
        if allocation_start is not None:
            current = os.environ.get('SLURM_JOB_ID')
            if current is None or int(current) not in active:
                raise ValueError('Current science must be on its actual RUNNING GPU receipt')
            row = next(r for r in rows if r['job'] == int(current))
            if start_epoch(allocation_start) != start_epoch(row['start']):
                raise ValueError('Allocation start differs from native scheduler')
        raw = self.remaining_raw()
        reserve = 0. if not active and self.state.get('full_encoding_audited',False) and sum(raw.values())==0 else FIXED_RESERVE
        projected = PRIOR + charge + 1.15 * sum(raw.values()) + reserve
        report = dict(status='COMPLETE_SEQUENCE_COST_FORECAST', actual_GPU_seconds=PRIOR+charge,
            remaining_cap_seconds=CAP-PRIOR-charge, projected_cumulative_seconds=projected,
            remaining_raw=raw, fixed_reserve=reserve, safety=1.15,
            fits=projected <= CAP, native=rows, cap_seconds=CAP,
            scientific_pass=False, UTC=datetime.datetime.fromtimestamp(self.clock(), datetime.timezone.utc).isoformat())
        write(self.root/'last_budget_forecast.json', report)
        if PRIOR+charge >= CAP or projected > CAP:
            write(self.root/'budget_stop.json', {**report, 'status':'TRUE_COMPLETE_SCOPE_BUDGET_STOP'})
            raise BudgetStop('Full unchanged confirmation no longer fits the authorized cumulative cap')
        return report

    def record_epoch(self, task, seed, role, epoch, seconds):
        key = f'{role}/{task}/{seed}'
        previous = self.state.setdefault('epochs', {}).get(key, 0)
        if task not in TASKS or seed not in SEEDS or role not in ('reference', 'candidate') or epoch != previous+1:
            raise ValueError('No skipped/repeated epoch accounting')
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Finite positive actual epoch duration required')
        self.state['epochs'][key] = epoch
        rates = self.state.setdefault('epoch_rate', {})
        rates[task+'/'+role] = max(rates.get(task+'/'+role, 0), seconds)
        self.save()
        write(self.root/'training_progress.json',dict(role=role,task=task,seed=seed,epoch=epoch,
            required_epochs=120,completed_models=sum(v==120 for v in self.state['epochs'].values()),
            total_models=12,phase1_go=False))

    def require_completed_reference(self, path, signature):
        p = Path(path)
        if signature.get('role') != 'reference' or self.state.get('epochs', {}).get(
                f"reference/{signature['task']}/{signature['seed']}") != 120:
            raise ValueError('Candidate requires its freshly completed same-seed reference')
        from phase1.proposal_trial import verify_completed_model
        proof = verify_completed_model(p.parent, signature)
        from phase1.utils import file_hash
        if proof['best_sha256'] != file_hash(p):
            raise ValueError('Fresh reference completion/current best hash mismatch')
        return proof
