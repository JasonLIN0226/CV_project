"""Register one remaining-budget GPU job after successful CPU preparation."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert os.environ['SLURM_JOB_ACCOUNT'] == 'csci_ga_2271-2026fa'
    parent = int(sys.argv[1])
    state = subprocess.check_output(['/opt/slurm/bin/sacct', '-X', '-j', str(parent), '-n', '-P', '-o', 'JobID,State,ExitCode'], text=True).strip()
    assert state == f'{parent}|COMPLETED|0:0', state
    lock = json.loads((HERE / 'source_lock.json').read_text())
    assert all(sha(HERE / n) == h for n, h in lock['files'].items())
    certificate = HERE / 'output' / f'job_{parent}' / 'certificate.json'
    c = json.loads(certificate.read_text())
    assert c['status'] == 'CURRENT_IDENTITIES_AND_OFFICIAL_ELIGIBILITY_COMPLETE'
    assert c['source_lock_sha256'] == sha(HERE / 'source_lock.json')
    assert c['test_records'] == 3840 and c['prior_GPU_seconds'] == 1580
    assert c['remaining_soft_GPU_seconds'] == 12220
    assert sha(certificate.parent / 'eligible_clips.json') == c['eligible_clips_sha256']
    assert c['eligibility_only_shape_tensor_optimization'] is True
    assert c['current_SHA_audit_source_job'] == 7412
    assert c['stat_boundary_all_data_precedes_SHA_audit_job_start'] is True
    assert sha(certificate.parent / 'paired_original_shape_equivalence.json') == c['paired_equivalence_sha256']
    old = Path('/scratch/xl4358/phase1_far_launch_recovery')
    failed_previous = Path('/scratch/xl4358/phase1_far_metadata_eligibility')
    boundaries = ['evaluation_binding.json', 'evaluation_submission_attempt.json', 'evaluation_submission.json', 'evaluation_unknown_or_failed_submission.json']
    assert not any((old / n).exists() for n in boundaries), 'Old GPU outcome exists or unknown'
    assert not any((failed_previous / n).exists() for n in boundaries), 'Previous preparation GPU outcome exists or unknown'
    retirement = HERE / 'superseded_preparation_retirement.json'
    attempt = HERE / 'superseded_retirement_attempt.json'
    fd = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'new_parent': parent, 'old_jobs': [7413, 7412]}, f)
    ended = {}
    for jid in [7413, 7412]:
        def state():
            return subprocess.check_output(['/opt/slurm/bin/sacct', '-X', '-j', str(jid), '-n', '-P', '-o', 'JobID,State,ExitCode,Start,End,Account'], text=True).strip()
        before = state()
        assert 'csci_ga_2271-2026fa' in before
        active = ['PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING']
        cancel = None
        if before.split('|')[1] in active:
            p = subprocess.run(['/opt/slurm/bin/scancel', str(jid)], text=True, capture_output=True)
            cancel = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
            assert p.returncode == 0, cancel
        for _ in range(8):
            after = state()
            if after.split('|')[1] not in active:
                break
            time.sleep(2)
        assert after.split('|')[1] not in active, 'Old preparation continuation is not terminal'
        ended[str(jid)] = {'before': before, 'cancel': cancel, 'after': after}
    retirement.write_text(json.dumps(ended, indent=2) + '\n')
    assert not any((old / n).exists() for n in boundaries), 'Old GPU marker appeared during retirement; refuse new submission'
    assert not any((failed_previous / n).exists() for n in boundaries), 'Previous preparation marker appeared; refuse new submission'
    jobs = subprocess.check_output(['/opt/slurm/bin/squeue', '-h', '-u', 'xl4358', '-o', '%i|%b|%j'], text=True)
    assert not any('gpu:' in line for line in jobs.splitlines()), jobs
    for n in ['evaluation_binding.json', 'evaluation_submission_attempt.json', 'evaluation_submission.json']:
        assert not (HERE / n).exists(), 'Existing/unknown submission boundary: ' + n
    binding = {'certificate_path': str(certificate), 'certificate_sha256': sha(certificate),
               'preparation_job': parent, 'source_lock_sha256': sha(HERE / 'source_lock.json'),
               'prior_GPU_seconds': 1580, 'remaining_soft_GPU_seconds': 12220,
               'remaining_wall_seconds': 12820}
    (HERE / 'evaluation_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    attempt = HERE / 'evaluation_submission_attempt.json'
    fd = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump({'status': 'ATTEMPT_STARTED_UNKNOWN_UNTIL_RECEIPT', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **binding}, f, indent=2)
    p = subprocess.run(['/opt/slurm/bin/sbatch', '--parsable', str(HERE / 'evaluate_recovered.sbatch')], capture_output=True, text=True)
    text = p.stdout.strip()
    result = {'returncode': p.returncode, 'stdout': text, 'stderr': p.stderr, **binding}
    if p.returncode != 0 or re.fullmatch(r'\d+', text) is None:
        (HERE / 'evaluation_unknown_or_failed_submission.json').write_text(json.dumps(result, indent=2) + '\n')
        raise RuntimeError('Submission not confirmed; preserve attempt and do not retry')
    result.update(status='SUBMITTED_ACTUAL_RECEIPT', job_id=int(text), metadata_job=int(os.environ['SLURM_JOB_ID']))
    (HERE / 'evaluation_submission.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
