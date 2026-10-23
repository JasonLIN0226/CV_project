"""Resume after the fresh lock was written by the raced lock allocation."""
import datetime
import json
import pathlib
import subprocess
import sys

R = pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume')
OUT = R / 'execution_repair' / 'final_continuation'
sys.path[:0] = [
    str(R),
    str(R / 'original_src'),
    str(R / 'execution_repair'),
    str(OUT),
]

import runner

w = runner.w
w.source_check()
runner.self_live()

data = pathlib.Path(w.config()['output'])
assert w.terminal(9242)['exitcode'] == '0:0'
training = json.loads((R / 'stage_completed/train_gpu.json').read_text())
assert int(training['job']) == 9242 and training['execution_completed']
budget = json.loads((data / 'sequence_budget.json').read_text())
assert len(budget['epochs']) == 12 and all(v == 120 for v in budget['epochs'].values())
assert budget['maps_locked']

permit = R / 'permit.json'
c = w.config()
lock_path = data / 'all_models_lock.json'
assert lock_path.is_file()
lock = json.loads(lock_path.read_text())
runner.locks.verify_all_lock(c, permit, lock)
assert not (data / 'confirmation_test_generation_attempt.json').exists()

archive = OUT / 'before_lock_reuse'
archive.mkdir(exist_ok=True)


def move_unique(source):
    destination = archive / source.name
    counter = 1
    while destination.exists():
        destination = archive / (source.stem + f'.r{counter}' + source.suffix)
        counter += 1
    source.rename(destination)


for name in (
    'submitted.json',
    'stage_failures/lock_cpu.json',
    'stage_attempts/lock_cpu.json',
):
    source = R / name if name.startswith('stage_') else OUT / name
    if source.exists():
        move_unique(source)
for stage in runner.STAGES[1:]:
    for suffix in ('submission', 'submission_attempt'):
        source = R / 'receipts' / f'{stage}_{suffix}.json'
        if source.exists():
            move_unique(source)

report = w.SequenceGuard(data).check()
assert report['remaining_cap_seconds'] >= 7320

receipts = []


def submit(stage, parent):
    gpu = stage == 'encode_test'
    seconds = (
        7200
        if gpu
        else 10800
        if stage == 'evaluate_cpu'
        else 900
        if stage == 'evidence_cpu'
        else 3600
    )
    args = [
        '/opt/slurm/bin/sbatch',
        '--parsable',
        '--account=csci_ga_2271-2026fa',
        '--partition=' + ('c12m85-a100-1' if gpu else 'n2c48m24'),
        '--cpus-per-task=' + ('1' if stage in ['audit_cpu', 'evidence_cpu'] else '4'),
        '--mem=' + ('60G' if gpu else '1500M' if stage == 'evidence_cpu' else '16G'),
        '--time=' + str(seconds // 60),
        '--no-requeue',
        '--job-name=p1-final-' + stage,
        '--output=' + str(R / (stage + '-final-%j.out')),
        '--error=' + str(R / (stage + '-final-%j.err')),
        '--export=NIL',
    ]
    if parent is not None:
        args.append('--dependency=afterok:' + str(parent))
    if gpu:
        args.append('--gres=gpu:1')
    args.append(
        '--wrap=/scratch/xl4358/phase1_env/bin/python -u '
        + str(OUT / 'runner.py')
        + ' '
        + stage
    )
    raw = subprocess.check_output(
        args, text=True, stderr=subprocess.STDOUT, timeout=30
    ).strip()
    job = int(raw.split(';')[0])
    receipt = {
        'stage': stage,
        'job': job,
        'parent': parent,
        'source_lock_sha256': w.file_hash(R / 'source_lock.json'),
        'final_continuation': True,
        'actual_submission_UTC': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }
    (R / 'receipts' / f'{stage}_submission.json').write_text(
        json.dumps(receipt, indent=2) + '\n'
    )
    (R / 'receipts' / f'{stage}_submission_attempt.json').write_text(
        json.dumps(
            {
                'stage': stage,
                'parent': parent,
                'unknown_do_not_retry': True,
                'final_continuation': True,
            },
            indent=2,
        )
        + '\n'
    )
    if gpu:
        guard = w.SequenceGuard(data)
        guard.state['GPU_job_ids'].append(job)
        guard.save()
    receipts.append(receipt)
    return job


lock_job = submit('lock_cpu', None)
parent = lock_job
for stage in runner.STAGES[2:]:
    parent = submit(stage, parent)

manifest = {
    'receipts': receipts,
    'source_lock_sha256': w.file_hash(R / 'source_lock.json'),
    'budget_before': {
        key: report[key]
        for key in ['actual_GPU_seconds', 'remaining_cap_seconds']
    },
    'runner_sha256': w.file_hash(OUT / 'runner.py'),
    'reused_verified_lock_sha256': w.file_hash(lock_path),
    'reason': 'Existing fresh lock contains all 12 models and 2 maps; no test attempt exists.',
}
(OUT / 'submitted.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
