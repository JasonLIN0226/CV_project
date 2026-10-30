"""Submit the locked reader aggregation ablation to the existing NYU A100 pool.

Source archive and every source member are verified before submission. This
wrapper never regenerates or copies the earlier TRAIN features, modifies the
encoder cache, or opens a confirmation split. The runner owns scientific gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import time
import zipfile


BASE = Path('/scratch/xy3257/cv-phase2-20261020')
WORK = BASE/'reader-aggregation-20261025'
SOURCE = BASE/'terminal-structures-20261025/run'
NAME = 'cv-reader-aggregation-20261025'
CONFIG = 'experiments/phase2/configs/reader_aggregation_20261025.json'
RUNNER = 'experiments/phase2/run_reader_aggregation.py'
WRAPPER = 'experiments/phase2/submit_reader_aggregation.py'
DERIVED_NAMES = {'unique_rgb.bin', 'features.npy', 'features_raw.npy'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def safe_member(name):
    """Accept only canonical relative POSIX member names, not OS-dependent paths."""
    p = PurePosixPath(name)
    if (not name or '\\' in name or '\x00' in name or p.is_absolute()
            or any(x in ('', '.', '..') for x in name.rstrip('/').split('/'))
            or ':' in name or str(p) != name.rstrip('/')):
        raise ValueError('Unsafe archive member: '+repr(name))
    return p


def extract_verified(archive, bundle):
    """Inspect all entries before extraction; then require exact manifest coverage."""
    bundle = Path(bundle)
    if bundle.exists():
        raise ValueError('Bundle directory must be new')
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for info in z.infolist():
            p = safe_member(info.filename)
            if info.filename in seen:
                raise ValueError('Duplicate archive member: '+info.filename)
            seen.add(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('Nonregular archive member: '+info.filename)
            if info.flag_bits & 1:
                raise ValueError('Encrypted source archives are unsupported')
            if info.is_dir() and not stat.S_ISDIR(mode) and stat.S_IFMT(mode) != 0:
                raise ValueError('Inconsistent directory entry: '+str(p))
        bundle.mkdir()
        z.extractall(bundle)
    return validate_bundle(bundle)


def validate_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest_path = bundle/'bundle_manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Source bundle manifest missing')
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty or malformed source manifest')
    actual = set()
    for p in bundle.rglob('*'):
        if p.is_symlink():
            raise ValueError('Source bundle contains symlink: '+str(p))
        if p.is_file() and p != manifest_path:
            actual.add(p.relative_to(bundle).as_posix())
    if actual != set(files):
        raise ValueError('Source manifest must cover every file exactly')
    for name, expected in files.items():
        safe_member(name)
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Malformed source hash: '+name)
        path = bundle/name
        if not path.resolve().is_relative_to(bundle) or sha(path) != expected:
            raise ValueError('Source member mismatch: '+name)
    return manifest


def runner_command(bundle, run):
    return [sys.executable, str(bundle/RUNNER), '--config', str(bundle/CONFIG),
            '--source', str(SOURCE), '--output', str(run), '--device', 'cuda',
            '--cache-dir', str(BASE/'cache/ijepa')]


def package(work, bundle, run):
    """Export new results/source only; omitted derived data remain on scratch."""
    work, bundle, run = map(Path, (work, bundle, run))
    destination = work/'reader-aggregation-results.zip'
    temporary = work/'reader-aggregation-results.zip.partial'
    if destination.exists() or temporary.exists():
        raise ValueError('Results package already exists; preserve it and inspect before retry')
    files, omitted, logs = {}, {}, []
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for root, prefix in ((run, 'run'), (bundle, 'source')):
            if not root.exists():
                continue
            for p in sorted(root.rglob('*')):
                if p.is_symlink():
                    raise ValueError('Result/source symlink is not packaged: '+str(p))
                if not p.is_file() or '__pycache__' in p.parts or '.git' in p.parts:
                    continue
                name = prefix+'/'+p.relative_to(root).as_posix()
                entry = {'sha256':sha(p), 'bytes':p.stat().st_size}
                if p.name in DERIVED_NAMES:
                    omitted[name] = entry
                else:
                    files[name] = entry
                    z.write(p, name)
        for p in sorted(work.glob('*.json')):
            if p.is_symlink():
                raise ValueError('Execution receipt symlink: '+str(p))
            name = 'execution/'+p.name
            files[name] = {'sha256':sha(p), 'bytes':p.stat().st_size}
            z.write(p, name)
        # Slurm is still writing while this wrapper packages. Snapshot bytes once
        # and explicitly label them nonfinal, so a manifest cannot imply a tail.
        for p in sorted(work.glob('slurm-*.log')):
            if p.is_symlink():
                raise ValueError('Slurm log symlink: '+str(p))
            content = p.read_bytes()
            name = 'execution/'+p.name+'.snapshot'
            files[name] = {'sha256':hashlib.sha256(content).hexdigest(), 'bytes':len(content)}
            logs.append({'archive_member':name, 'remote_path':str(p), 'snapshot_utc':utc(),
                         'final_log':False, 'truncated_or_incomplete':True,
                         'reason':'Wrapper and Slurm may append after this snapshot; retrieve final tail separately.'})
            z.writestr(name, content)
        z.writestr('package_manifest.json', json.dumps({
            'files':files, 'derived_data_preserved_on_scratch':omitted,
            'previous_run_reference_only':str(SOURCE), 'previous_training_data_copied':False,
            'slurm_log_snapshots':logs,
            'scientific_status':'Read execution and run/study_status.json; package creation is not a scientific success claim.',
        }, indent=2))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise RuntimeError('Results ZIP CRC failure')
    temporary.rename(destination)
    receipt = {'path':str(destination), 'sha256':sha(destination), 'bytes':destination.stat().st_size,
               'included_files':len(files), 'omitted_derived_files':len(omitted),
               'remote_originals_preserved':True, 'slurm_logs_final':False,
               'created_utc':utc()}
    write(work/'package_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def execute(bundle, run):
    """Exactly one monolithic runner invocation, with an execution/package receipt."""
    bundle, run = Path(bundle).resolve(), Path(run).resolve()
    work = run.parent
    if bundle != work/'bundle' or run.name != 'run':
        raise ValueError('Expected sibling WORK/bundle and WORK/run paths')
    receipt = {'status':'running', 'started_utc':utc(), 'commands':[],
               'source_run':str(SOURCE), 'confirmation_opened':False,
               'confirmation_policy':'No confirmation or student stage in this ablation wrapper'}
    # A second direct --execute must not overwrite the first execution receipt.
    with (work/'execution.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    error = None
    try:
        if run.exists():
            raise ValueError('Output run already exists; do not repeat an ablation')
        validate_bundle(bundle)
        for name in (CONFIG, RUNNER):
            if not (bundle/name).is_file():
                raise ValueError('Missing runner/config source: '+name)
        import torch
        hardware = {'utc':utc(), 'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                    'torch_version':torch.__version__, 'cuda_available':torch.cuda.is_available()}
        write(work/'hardware.json', hardware)
        if not hardware['cuda_available']:
            raise RuntimeError('Allocated CUDA GPU required')
        hardware.update(cuda_device=torch.cuda.get_device_name(0),
                        nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                                           '--format=csv,noheader'], text=True, timeout=30))
        write(work/'hardware.json', hardware)
        receipt['hardware'] = hardware
        if 'A100' not in hardware['cuda_device']:
            raise RuntimeError('Expected allocated A100 GPU')
        receipt['gpu_execution_verified'] = True
        command = runner_command(bundle, run)
        started = time.monotonic()
        item = {'arguments':command, 'started_utc':utc(), 'returncode':None}
        receipt['commands'].append(item)
        write(work/'execution.json', receipt)
        process = subprocess.run(command, cwd=bundle)
        item.update(returncode=process.returncode, seconds=time.monotonic()-started)
        receipt['runner_returncode'] = process.returncode
        status_path = run/'study_status.json'
        if status_path.exists():
            receipt['study_status'] = json.loads(status_path.read_text())
            receipt['study_status_sha256'] = sha(status_path)
            receipt['confirmation_opened'] = bool(receipt['study_status'].get('confirmation_opened', False))
        if process.returncode:
            raise RuntimeError('Reader aggregation runner failed; see saved return code and study status')
        if not status_path.is_file():
            raise RuntimeError('Runner returned zero but study_status.json is absent')
        if receipt['confirmation_opened']:
            raise RuntimeError('Runner unexpectedly reports opening confirmation; preserve this protocol violation')
        receipt['status'] = 'runner_completed_read_study_status'
    except BaseException as exc:
        error = exc
        receipt.update(status='execution_failed', error=repr(exc))
    finally:
        receipt['finished_utc'] = utc()
        # Preserve runner status even when its process raised/interrupted.
        status_path = run/'study_status.json'
        if status_path.is_file() and 'study_status' not in receipt:
            try:
                receipt.update(study_status=json.loads(status_path.read_text()), study_status_sha256=sha(status_path))
                receipt['confirmation_opened'] = bool(receipt['study_status'].get('confirmation_opened', False))
            except Exception as exc:
                receipt['study_status_read_error'] = repr(exc)
        write(work/'execution.json', receipt)
        try:
            package(work, bundle, run)
        except BaseException as exc:
            receipt['package_error'] = repr(exc)
            write(work/'execution.json', receipt)
            if error is None:
                error = exc
    if error is not None:
        raise error
    return receipt


def submit(archive, expected):
    archive = Path(archive)
    if not re.fullmatch('[0-9a-f]{64}', expected) or sha(archive) != expected:
        raise ValueError('Source archive hash mismatch')
    if not (BASE/'venv/bin/python').is_file() or not (BASE/'cache/ijepa').is_dir():
        raise ValueError('Missing pinned environment or encoder cache')
    if not SOURCE.is_dir():
        raise ValueError('Earlier source run is absent; do not regenerate training features')
    env = {k:os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','SHELL') if k in os.environ}
    active = subprocess.check_output(['/opt/slurm/bin/squeue', '-h', '-u', 'xy3257', '-n', NAME,
                                      '-o', '%i %T'], text=True, env=env, timeout=30).strip()
    if active:
        raise RuntimeError('Identically named active job exists: '+active)
    WORK.mkdir(parents=True, exist_ok=False)
    receipt_path = WORK/'submission_receipt.json'
    record = {'status':'preparing', 'archive_sha256':expected, 'created_utc':utc(),
              'gpu_execution_verified':False, 'source_run':str(SOURCE), 'confirmation_opened':False}
    with receipt_path.open('x') as f:
        json.dump(record, f, indent=2)
    try:
        bundle = WORK/'bundle'
        extract_verified(archive, bundle)
        for name in (CONFIG, RUNNER, WRAPPER):
            if not (bundle/name).is_file():
                raise ValueError('Bundle missing required source: '+name)
        command = [str(BASE/'venv/bin/python'), str(bundle/WRAPPER), '--execute',
                   '--bundle', str(bundle), '--run', str(WORK/'run')]
        script = '\n'.join(['#!/bin/bash', 'set -euo pipefail', 'umask 077',
            'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4',
            'export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1',
            shlex.join(command), ''])
        (WORK/'run.sbatch').write_text(script)
        cmd = ['/opt/slurm/bin/sbatch', '--parsable', '--export=ALL', '--no-requeue', '--job-name='+NAME,
               '--account=csci_ga_2271-2026fa', '--partition=c12m85-a100-1', '--gres=gpu:1',
               '--cpus-per-task=4', '--mem=32G', '--time=02:00:00', '--chdir='+str(WORK),
               '--output='+str(WORK/'slurm-%j.log'), str(WORK/'run.sbatch')]
        record.update(status='submitting', command=cmd, bundle_manifest_sha256=sha(bundle/'bundle_manifest.json'))
        write(receipt_path, record)
        process = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=45)
        record.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
        if process.returncode or not re.fullmatch(r'\d+(;[^\s]+)?\s*', process.stdout):
            raise RuntimeError('Submission unknown; inspect scheduler before retry')
        record.update(status='submitted', job_id=process.stdout.strip().split(';')[0])
    except Exception as exc:
        record.update(status='failed_or_unknown', error=repr(exc))
        raise
    finally:
        write(receipt_path, record)
        print(json.dumps(record, indent=2), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path); p.add_argument('--sha256')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--bundle', type=Path); p.add_argument('--run', type=Path)
    args = p.parse_args()
    if args.execute:
        if not args.bundle or not args.run:
            p.error('--execute requires --bundle and --run')
        execute(args.bundle, args.run)
    else:
        if not args.archive or not args.sha256:
            p.error('Submission requires --archive and --sha256')
        submit(args.archive, args.sha256)


if __name__ == '__main__':
    main()
