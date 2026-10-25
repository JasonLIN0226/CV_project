"""Preserve an interrupted CPU pilot, then submit one independent Slurm retry."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shlex
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, required=True)
    args = ap.parse_args()
    work = args.work.resolve()
    receipt = work / 'scaled_recovery_submission.json'
    if receipt.exists():
        raise ValueError('Submission receipt exists; inspect it, never resubmit automatically')
    entry = work / 'bundle/experiments/phase2/run_scaled_use_time_probe.py'
    packer = work / 'pack_scaled_probe.py'
    expected = {
        entry: '94442ad37315dc1832459fffcefc09adf7a27a3b374a54e4d8a087cab90d2f3f',
        packer: 'be7fe094c474790ad7d71a5b2e2e621101913bf921fca807d50421a96a5bdbb4',
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError('Reviewed source hash mismatch: ' + str(path))
    accounting = subprocess.run(['/opt/slurm/bin/sacct', '-j', '11023', '--parsable2',
        '--format=JobID,State,ExitCode,ElapsedRaw,Timelimit,Reason,AllocTRES'],
        text=True, capture_output=True, timeout=30, check=True)
    if 'CANCELLED' not in accounting.stdout:
        raise ValueError('Expected original CPU session to be cancelled; inspect current state')
    (work / 'sacct-interrupted-11023.txt').write_text(accounting.stdout)
    partial = work / 'scaled-use-time-probe'
    archived = work / 'scaled-use-time-probe-interrupted-11023'
    if archived.exists() or not partial.is_dir() or (partial / 'summary.json').exists():
        raise ValueError('Require one incomplete, unarchived attempt')
    oldlog = work / 'scaled-probe-actual.log'
    archived_log = work / 'scaled-probe-session11023.log'
    if archived_log.exists() or not oldlog.is_file():
        raise ValueError('Require the original unarchived run log')
    partial.rename(archived)
    oldlog.rename(archived_log)
    python = work.parent / 'venv/bin/python'
    script = work / 'scaled-recovery.sbatch'
    command = shlex.join([str(python), '-u', str(entry), '--work', str(work)])
    package = shlex.join([str(python), str(packer), '--work', str(work)])
    script.write_text('#!/bin/bash\nset -euo pipefail\n'
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
        + command + '\n' + package + '\n')
    request = ['/opt/slurm/bin/sbatch', '--parsable', '--account=csci_ga_2271-2026fa',
        '--partition=n2c48m24', '--cpus-per-task=4', '--mem=16G', '--time=00:30:00',
        '--no-requeue', '--job-name=gate-scaled-use-time',
        '--output=' + str(work / 'scaled-probe-actual.log'), str(script)]
    # Write an attempt guard before submission; unknown outcomes require inspection.
    attempt = {'status':'submitting', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'command':request, 'script_sha256':sha(script), 'submitter_sha256':sha(__file__),
        'reviewed_source_sha256':{str(p):d for p,d in expected.items()},
        'archived_partial':str(archived), 'archived_log':str(archived_log),
        'parent_control_job':os.environ.get('SLURM_JOB_ID'), 'new_gpu_count':0}
    receipt.write_text(json.dumps(attempt, indent=2)+'\n')
    submitted = subprocess.run(request, text=True, capture_output=True, timeout=60)
    attempt.update(status='submitted' if submitted.returncode == 0 else 'submission_failed',
        returncode=submitted.returncode, stdout=submitted.stdout, stderr=submitted.stderr)
    receipt.write_text(json.dumps(attempt, indent=2)+'\n')
    print(json.dumps(attempt), flush=True)
    if submitted.returncode != 0:
        raise RuntimeError('Submission failed; inspect saved receipt')


if __name__ == '__main__':
    main()
