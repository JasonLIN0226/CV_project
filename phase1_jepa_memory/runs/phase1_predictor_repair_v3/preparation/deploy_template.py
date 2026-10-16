"""Deploy the single frozen repair source and submit its guarded CPU preflight only."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import zipfile

ROOT = Path('/scratch/xl4358/phase1_predictor_repair_v3/execution')
ARCHIVE = Path('/scratch/xl4358/phase1_predictor_repair_v3_execution.zip')
ARCHIVE_SHA = '47ae722c60ae5f3653eee92514cb3ba76b6b509d781d2cb50025633cd6dc6a91'
LOCK_SHA = '0118e9b61f13d6ae0b271f3faa77bf276d6e7af1f6c97992ae0974902e6ba2d0'


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
    active = subprocess.run(['/opt/slurm/bin/squeue', '-u', 'xl4358', '-h', '-o', '%i %j %b'],
                            check=True, capture_output=True, text=True).stdout
    if any('gpu' in line.lower() or 'cv-phase1-repair' in line for line in active.splitlines()) or ROOT.exists():
        raise ValueError('Existing GPU/repair receipt or root; inspect, never implicitly resubmit')
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Exact repair source archive SHA mismatch')
    with zipfile.ZipFile(ARCHIVE) as archive:
        prefix = 'execution/'
        if (len(archive.namelist()) != 62 or archive.testzip() is not None
                or any(not name.startswith(prefix) or '..' in Path(name).parts for name in archive.namelist())):
            raise ValueError('Unexpected repair source archive path/members/CRC')
        archive.extractall(ROOT.parent)
    control = ROOT / 'control'
    if sha(control / 'source_lock.json') != LOCK_SHA:
        raise ValueError('Frozen repair source lock SHA mismatch')
    if sha(control / 'protocol.json') != '4fc74cc1ed12eb7c1f39d62aaf992567f58f94fc8e3c0d308a8844f8e1bd5f57':
        raise ValueError('Fixed full repair protocol SHA mismatch')
    lock = json.loads((control / 'source_lock.json').read_text())
    for rel, expected in lock['files'].items():
        if sha(control / rel) != expected:
            raise ValueError('Repair native source mismatch: ' + rel)
    independent = ROOT / 'independent_audit'
    if sha(independent / 'source_lock.json') != '1e784f6dec75dd6ba820829d8e447e854a2426c706e7020ff0ce9c0685c4e3cc':
        raise ValueError('Independent original scalar audit source lock differs')
    for rel, expected in json.loads((independent / 'source_lock.json').read_text())['files'].items():
        if sha(independent / rel) != expected:
            raise ValueError('Independent scalar audit source changed')
    if sha(ROOT.parent/'diagnostic_results.json') != 'c64b1367ed7b3dc9ce3e1e6e389537b1d3ece93e4fd5dd92a822d0c8db8cfcd6':
        raise ValueError('Actual read-only completed-model diagnosis changed')
    (ROOT / 'logs').mkdir()
    marker = ROOT / 'preflight_submission_attempt.json'
    attempt = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'deployment_job': int(job), 'deployment_native': native,
               'archive_sha256': ARCHIVE_SHA, 'source_lock_sha256': LOCK_SHA,
               'GPU': 0, 'command': ['/opt/slurm/bin/sbatch', '--parsable', str(control / 'preflight.sbatch')]}
    with marker.open('x') as handle:
        json.dump(attempt, handle, indent=2); handle.write('\n')
    result = subprocess.run(attempt['command'], capture_output=True, text=True)
    receipt = {**attempt, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    if result.returncode == 0 and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_-]+)?\n?', result.stdout):
        receipt.update(job_id=int(result.stdout.strip().split(';')[0]), status='SUBMITTED_ACTUAL_RECEIPT')
    else:
        receipt.update(status='SUBMISSION_RESULT_REQUIRES_INSPECTION_NO_AUTOMATIC_RETRY')
    (ROOT / 'preflight_submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt['status'] != 'SUBMITTED_ACTUAL_RECEIPT':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
