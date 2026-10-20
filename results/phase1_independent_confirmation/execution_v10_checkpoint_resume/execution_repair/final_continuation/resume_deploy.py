"""Execution-only retry after the completed parent makes afterok invalid."""
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

D = pathlib.Path(w.config()['output'])
assert w.terminal(9242)['exitcode'] == '0:0'
proof = json.loads((R / 'stage_completed/train_gpu.json').read_text())
assert int(proof['job']) == 9242 and proof['execution_completed']
state = json.loads((D / 'sequence_budget.json').read_text())
assert len(state['epochs']) == 12 and all(e == 120 for e in state['epochs'].values())
assert state['maps_locked']

marker = OUT / 'submitted.json'
assert not marker.exists()

# Preserve the failed lock attempt. The directory may already exist because
# the previous deployment stopped after archiving it and before submitting.
archive = OUT / 'failed9245'
archive.mkdir(exist_ok=True)


def archive_unique(source):
    destination = archive / source.name
    if destination.exists():
        destination = archive / (source.stem + '.resume_failure' + source.suffix)
    counter = 1
    while destination.exists():
        destination = archive / (
            source.stem + f'.resume_failure_{counter}' + source.suffix
        )
        counter += 1
    source.rename(destination)


for name in ['stage_failures/lock_cpu.json', 'stage_attempts/lock_cpu.json']:
    source = R / name
    if source.exists():
        archive_unique(source)

for stage in runner.STAGES[1:]:
    for name in [f'receipts/{stage}_submission.json', f'receipts/{stage}_submission_attempt.json']:
        source = R / name
        if source.exists():
            record = json.loads(source.read_text())
            job = record.get('job')
            if job:
                native = w.native_allocations([int(job)])[0]
                assert native['state'] in ['FAILED', 'CANCELLED', 'COMPLETED', 'TIMEOUT']
            archive_unique(source)

report = w.SequenceGuard(D).check()
assert report['remaining_cap_seconds'] >= 7320

receipts = []
parent = 9242
for stage in runner.STAGES[1:]:
    attempt = R / 'receipts' / f'{stage}_submission_attempt.json'
    with attempt.open('x') as handle:
        json.dump(
            dict(
                stage=stage,
                parent=parent,
                unknown_do_not_retry=True,
                final_continuation=True,
            ),
            handle,
        )

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
    # 9242 is already terminal; Slurm rejects afterok for this completed parent.
    # Submit the first restored stage directly, then preserve the remaining chain.
    if parent != 9242:
        args.append('--dependency=afterok:' + str(parent))
    if gpu:
        args.append('--gres=gpu:1')
    args.append(
        '--wrap=/scratch/xl4358/phase1_env/bin/python -u '
        + str(OUT / 'runner.py')
        + ' '
        + stage
    )
    raw = subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    job = int(raw.split(';')[0])
    receipt = dict(
        stage=stage,
        job=job,
        parent=parent,
        source_lock_sha256=w.file_hash(R / 'source_lock.json'),
        final_continuation=True,
    )
    (R / 'receipts' / f'{stage}_submission.json').write_text(json.dumps(receipt, indent=2))
    if gpu:
        guard = w.SequenceGuard(D)
        guard.state['GPU_job_ids'].append(job)
        guard.save()
    receipts.append(receipt)
    parent = job

marker.write_text(
    json.dumps(
        dict(
            receipts=receipts,
            source_lock_sha256=w.file_hash(R / 'source_lock.json'),
            budget_before={
                key: report[key]
                for key in ['actual_GPU_seconds', 'remaining_cap_seconds']
            },
            runner_sha256=w.file_hash(OUT / 'runner.py'),
        ),
        indent=2,
    )
)
print(marker.read_text())
