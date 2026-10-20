"""Read-only V9 settlement on an authenticated, actual course CPU allocation.

No submission, cancellation, model forward, test access, or checkpoint writes.
Output is compact evidence; this script does not authorize a GPU restart.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import sys

ROOT = Path('/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation')
SOURCE_SHA = '76fd0dd1070fc614bbbf246eee7bc4c3ff7d308388dd67e26b663ee8a9a58a30'
ACCOUNT = 'csci_ga_2271-2026fa'


def run(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=30,
                          env={**os.environ, 'TZ': 'UTC'})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    # Refuse login/jump hosts, GPU allocations, and unrelated course jobs.
    job = int(os.environ['SLURM_JOB_ID'])
    current = run(['/opt/slurm/bin/scontrol', 'show', 'job', '-o', str(job)])
    current.check_returncode()
    fields = dict(v.split('=', 1) for v in current.stdout.split() if '=' in v)
    assert fields['JobState'] == 'RUNNING' and fields['Account'] == ACCOUNT
    assert fields['Partition'] in ('interactive', 'n2c48m24') and 'gres/gpu=' not in fields['AllocTRES']
    assert fields['NodeList'] == socket.gethostname().split('.')[0]
    assert sha(ROOT/'source_lock.json') == SOURCE_SHA
    lock = json.loads((ROOT/'source_lock.json').read_text())
    source_bad = [name for name, digest in lock['files'].items() if sha(ROOT/name) != digest]
    assert not source_bad, source_bad
    sys.path.insert(0, str(ROOT))
    from budget_guard import native_allocations
    ids = [8946, 8949, 8950, 8951, 9029, 9030]
    rows = native_allocations(ids)
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'OUT_OF_MEMORY'}
    assert all(r['state'].split('+')[0] in terminal for r in rows)
    detail = run(['/opt/slurm/bin/sacct', '-D', '-X', '-n', '-P', '-u', pwd.getpwuid(os.getuid()).pw_name,
                  '-A', ACCOUNT, '-S', '2026-10-19', '-E', '2026-10-21', '-j', ','.join(map(str, ids)),
                  '--format=JobID,JobName,State,Reason,ExitCode,DerivedExitCode,Start,End,ElapsedRaw,Timelimit,NodeList,Comment,AdminComment,SystemComment'])
    # Preserve any unavailable attribution field as an explicit error; do not invent a cause.
    detail_proof = dict(returncode=detail.returncode, stdout=detail.stdout, stderr=detail.stderr)
    for row in rows:
        assert row['account'] == ACCOUNT
    old_guard = json.loads((ROOT/'watchdog_status.json').read_text())
    gpu_charge = {}
    for row in rows:
        if row['job'] not in (8951, 9030):
            continue
        assert row['tres'].get('gres/gpu') == '1'
        observed = old_guard['GPU_jobs'].get(str(row['job']), {})
        charge = max(row['elapsed'], observed.get('charged_seconds', 0))
        # Include known configuration wall preceding batch Start.
        first = observed.get('first_allocated_start')
        if first is not None:
            end = datetime.datetime.fromisoformat(row['end']).replace(tzinfo=datetime.timezone.utc).timestamp()
            charge = max(charge, int(end - first + .999999))
        gpu_charge[str(row['job'])] = charge
    # 8951 was already conservatively settled at1265 seconds including configuration.
    gpu_charge['8951'] = max(1265, gpu_charge['8951'])
    ledger = dict(prior=40499, cap=69299, charge_by_job=gpu_charge,
                  settled=40499 + sum(gpu_charge.values()))
    ledger['remaining'] = ledger['cap'] - ledger['settled']
    # CPU tensor inspection only. No network, forward, or held-out data access.
    import torch
    torch.set_num_threads(1)
    def finite(v):
        if torch.is_tensor(v):
            return bool(torch.isfinite(v).all())
        if isinstance(v, dict):
            return all(finite(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return all(finite(x) for x in v)
        return not isinstance(v, float) or __import__('math').isfinite(v)
    model_root = ROOT/'fresh/models/reference/gate/seed_17'
    latest = torch.load(model_root/'latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(model_root/'best.pt', map_location='cpu', weights_only=False)
    training = json.loads((model_root/'training.json').read_text())
    logs = training['logs']
    assert latest['epoch'] == len(logs) == 80 and latest['step'] == 5120
    assert [r['epoch'] for r in logs] == list(range(1, 81))
    assert [r['step'] for r in logs] == list(range(64, 5121, 64))
    assert finite(latest['model']) and finite(latest['optimizer']) and finite(logs)
    assert latest['signature'] == best['signature'] == training['signature']
    sig = latest['signature']
    assert (sig['task'], sig['role'], sig['seed']) == ('gate', 'reference', 17)
    assert sig['permit_sha256'] == sha(ROOT/'permit.json')
    assert sig['normalization_sha256'] == sha(ROOT/'fresh/features/normalization.npz')
    assert sig['basis_sha256'] == sha(ROOT/'fresh/predictor/gate/basis.npz')
    selected = min(logs, key=lambda r: r['validation_mse'])
    assert best['epoch'] == selected['epoch'] and best['best'] == selected['validation_mse']
    assert latest['scheduler']['last_epoch'] == 80
    assert latest['optimizer']['state'] and len(latest['optimizer']['param_groups']) == 1
    assert latest['rng']['cuda'] and all(torch.is_tensor(x) for x in latest['rng']['cuda'])
    assert set(latest['rng']) == {'python', 'numpy', 'torch', 'cuda'}
    outputs = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), audit_job=job,
                   native_CPU=current.stdout, source_lock_sha256=SOURCE_SHA, source_files=len(lock['files']),
                   terminal_rows=rows, cancellation_attribution=detail_proof,
                   watchdog_status=old_guard, conservative_ledger=ledger,
                   checkpoint=dict(epoch=80, step=5120, latest_sha256=sha(model_root/'latest.pt'),
                                   best_sha256=sha(model_root/'best.pt'), signature=sig,
                                   best_epoch=best['epoch'], optimizer_groups=latest['optimizer']['param_groups'],
                                   scheduler=latest['scheduler'], finite=True, cuda_rng_present=True,
                                   no_model_forward=True, exact_GPU_resume_equivalence_not_yet_proven=True),
                   no_restart_authorized=True, scientific_result='INCOMPLETE_EXECUTION')
    print(json.dumps(outputs, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
