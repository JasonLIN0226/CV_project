"""Register one remaining-budget GPU job after successful CPU preparation."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

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
