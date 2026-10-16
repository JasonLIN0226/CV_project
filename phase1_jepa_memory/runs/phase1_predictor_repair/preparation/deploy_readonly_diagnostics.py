"""Run only on the freshly verified course deployment node, never on Torch."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import zipfile

ROOT = Path('/scratch/xl4358/phase1_predictor_repair')
ARCHIVE = Path('/scratch/xl4358/phase1_predictor_repair_readonly_diagnostics.zip')
ARCHIVE_SHA = '4d3a5269e347a1bda4f8ca1b5edc1bec0f04ba103e5d6597032127a5a7b70d65'
LOCK_SHA = '53cb9e66ea62b0cd2d4a591d2bb703ff2ec9fb4b5e80712b20b91dd28fba57fe'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    job = sys.argv[1]
    if not re.fullmatch(r'[0-9]+', job):
        raise ValueError('Actual deployment job ID required')
    native = subprocess.run(['/opt/slurm/bin/scontrol', 'show', 'job', '-o', job],
                            check=True, capture_output=True, text=True).stdout
    fields = dict(x.split('=', 1) for x in native.split() if '=' in x)
    if (fields.get('JobState') != 'RUNNING' or fields.get('Account') != 'csci_ga_2271-2026fa'
            or fields.get('NodeList') != socket.gethostname().split('.')[0]
            or fields.get('NumNodes') != '1' or 'cpu=' not in fields.get('AllocTRES', '')
            or 'gres/gpu' in fields.get('AllocTRES', '')):
        raise ValueError('Current real course zero-GPU deployment allocation required')
    active = subprocess.run(['/opt/slurm/bin/squeue', '-u', 'xl4358', '-h', '-n', 'cv-phase1-diagnose'],
                            check=True, capture_output=True, text=True).stdout
    if active.strip() or ROOT.exists():
        raise ValueError('Existing active diagnostic or root; refuse overwrite / implicit resubmission')
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Exact source archive SHA mismatch')
    with zipfile.ZipFile(ARCHIVE) as archive:
        expected = {'phase1_predictor_repair/' + name for name in
                    ['diagnostic_protocol.json', 'diagnose_native.py', 'diagnose.sbatch', 'diagnostic_source_lock.json']}
        if set(archive.namelist()) != expected or archive.testzip() is not None:
            raise ValueError('Unexpected diagnostic archive members / CRC')
        archive.extractall(ROOT.parent)
    if sha(ROOT / 'diagnostic_source_lock.json') != LOCK_SHA:
        raise ValueError('Diagnostic source lock SHA mismatch')
    lock = json.loads((ROOT / 'diagnostic_source_lock.json').read_text())
    for rel, expected in lock['files'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('Diagnostic native source mismatch: ' + rel)
    (ROOT / 'logs').mkdir()
    marker = ROOT / 'diagnostic_submission_attempt.json'
    attempt = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'deployment_job': int(job), 'deployment_native': native,
               'archive_sha256': ARCHIVE_SHA, 'diagnostic_source_lock_sha256': LOCK_SHA,
               'GPU': 0, 'command': ['/opt/slurm/bin/sbatch', '--parsable', str(ROOT / 'diagnose.sbatch')]}
    with marker.open('x') as handle:
        json.dump(attempt, handle, indent=2); handle.write('\n')
    result = subprocess.run(attempt['command'], capture_output=True, text=True)
    receipt = {**attempt, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    if result.returncode == 0 and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_-]+)?\n?', result.stdout):
        receipt.update(job_id=int(result.stdout.strip().split(';')[0]), status='SUBMITTED_ACTUAL_RECEIPT')
    else:
        receipt.update(status='SUBMISSION_RESULT_REQUIRES_INSPECTION_NO_AUTOMATIC_RETRY')
    (ROOT / 'diagnostic_submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt['status'] != 'SUBMITTED_ACTUAL_RECEIPT':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
