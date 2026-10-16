#!/usr/bin/env python3
"""Compare decoder scheduling on the actual stopped first512, without a model.

This is an allocated course data/implementation audit. It never encodes model
features, fits a basis, trains a reader, or claims complete cache feasibility.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def regular(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Missing or symlinked explicit audit input')


def run(control, expected_lock):
    control = Path(control)
    lock = control / 'source_lock.json'
    regular(lock)
    if sha(lock) != expected_lock:
        raise ValueError('Diagnostic source lock changed')
    for rel, want in json.loads(lock.read_text())['files'].items():
        file = control / rel
        regular(file)
        if sha(file) != want:
            raise ValueError('Diagnostic source bytes changed: ' + rel)
    p = json.loads((control / 'protocol.json').read_text())
    if (os.getenv('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa'
            or os.getenv('SLURM_JOB_PARTITION') != 'n2c48m24'
            or not os.getenv('SLURM_JOB_ID')
            or not socket.gethostname().startswith('b-')):
        raise ValueError('Actual allocated course CPU data audit required')
    if int(os.getenv('SLURM_CPUS_PER_TASK', '0')) != 4:
        raise ValueError('Declared four CPU audit changed')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 4:
        raise ValueError('Insufficient allocated CPUs')
    os.sched_setaffinity(0, affinity[:4])
    econtrol = Path(p['encoding_control'])
    elock = econtrol / 'source_lock.json'
    regular(elock)
    if sha(elock) != p['encoding_source_lock_sha256']:
        raise ValueError('Stopped encoding source lock changed')
    for rel, want in json.loads(elock.read_text())['files'].items():
        file = econtrol / rel
        regular(file)
        if sha(file) != want:
            raise ValueError('Stopped encoding source changed: ' + rel)
    report_file = Path(p['encoding_root']) / 'encoding_report.json'
    regular(report_file)
    if sha(report_file) != p['encoding_report_sha256']:
        raise ValueError('Actual failure report changed')
    report = json.loads(report_file.read_text())
    if (report['status'] != 'STOPPED_FIXED_FIRST512_PROJECTION_EXCEEDS_BUDGET'
            or report['job_id'] != '7179' or report['frames_encoded'] != 512
            or len(report['completed_recordings']) != 1):
        raise ValueError('Actual first512 failure boundary differs')
    sacct = subprocess.check_output(['/opt/slurm/bin/sacct', '-X', '-j', '7179',
        '-n', '-P', '--format=JobIDRaw,Account,Partition,State,ExitCode'], text=True)
    if sacct.strip().split('|')[:5] != ['7179', 'csci_ga_2271-2026fa',
                                         'c12m85-a100-1', 'FAILED', '2:0']:
        raise ValueError('Actual stopped parent must be FAILED2:0')
    row = report['completed_recordings'][0]
    path = Path(p['rgb_root']) / row['rgb_path']
    regular(path)
    pts = json.loads((econtrol / 'pts_report.json').read_text())
    audited = {v['path']: v for v in pts['videos']}[row['rgb_path']]
    if row['episodes'] != audited['episodes'] or row['feature']['shape'] != [512, 5120]:
        raise ValueError('Prespecified first recording/episode count differs')
    sys.path.insert(0, str(econtrol / 'scripts'))
    from v5_encode_real import verify
    _, vendor = verify(econtrol, p['encoding_source_lock_sha256'])
    sys.path.insert(0, str(control))
    from v5_causal_frame_stream import causal_frames
    sys.path.insert(0, str(vendor))
    import av
    import numpy as np
    if av.__version__ != '18.0.0':
        raise ValueError('Pinned actual decoder runtime changed')
    partial = Path(p['encoding_root']) / 'features' / row['feature']['path']
    regular(partial)
    partial_sha = sha(partial)
    if partial_sha != row['feature']['sha256'] or partial.stat().st_size != row['feature']['bytes']:
        raise ValueError('Actual partial feature bytes changed')
    partial_array = np.load(partial, mmap_mode='r', allow_pickle=False)
    if partial_array.shape != (512, 5120) or partial_array.dtype != np.float32 or not np.isfinite(partial_array).all():
        raise ValueError('Actual partial encoding is invalid/nonfinite')
    if sha(partial) != partial_sha:
        raise ValueError('Partial cache changed during numerical audit')
    clocks = [Fraction(i, 2) for i in range(512)]
    results = []
    started = time.monotonic()
    for mode in ['SLICE', 'FRAME']:
        before = time.monotonic()
        if sha(path) != row['rgb_sha256']:
            raise ValueError('Actual RGB bytes changed before decode')
        hash_before_seconds = time.monotonic() - before
        decode_start = time.monotonic()
        rows = []
        pts_digest = hashlib.sha256()
        with av.open(str(path), mode='r') as container:
            if len(container.streams.video) != 1:
                raise ValueError('Observed camera stream differs')
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 4
            stream.codec_context.thread_type = mode
            for clock, stamp, frame in causal_frames(container.decode(stream), clocks):
                if time.monotonic() - started > 1500:
                    raise RuntimeError('Declared diagnostic soft-stop reached')
                if clock != stamp:
                    raise ValueError('Actual chosen PTS differs from frozen exact clock')
                index = len(rows)
                pts_digest.update(f'{index//256}:{index%256}:{stamp.numerator}/{stamp.denominator}\n'.encode())
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.ndim != 3 or rgb.shape[-1] != 3 or str(rgb.dtype) != 'uint8':
                    raise ValueError('RGB representation changed')
                rows.append({'clock': [clock.numerator, clock.denominator],
                             'pts': [stamp.numerator, stamp.denominator],
                             'shape': list(rgb.shape),
                             'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
        decode_seconds = time.monotonic() - decode_start
        if len(rows) != 512:
            raise ValueError('Missing actual first512 observations')
        if pts_digest.hexdigest() != row['feature']['timestamp_sha256']:
            raise ValueError('Actual first512 PTS digest differs from completed prefix')
        after = time.monotonic()
        if sha(path) != row['rgb_sha256']:
            raise ValueError('Actual RGB changed during decode')
        item = {'thread_type': mode, 'threads': 4, 'frames': rows,
                'hash_before_seconds': hash_before_seconds,
                'decode_rgb_digest_seconds': decode_seconds,
                'hash_after_seconds': time.monotonic() - after,
                'total_seconds': time.monotonic() - before}
        results.append(item)
        print(json.dumps({k: v for k, v in item.items() if k != 'frames'}), flush=True)
    equal = results[0]['frames'] == results[1]['frames']
    output = Path(p['output'])
    if output.exists():
        raise ValueError('Refuse overwrite of previous actual diagnostic')
    result = {'status': 'EXACT_FIRST512_PTS_AND_RGB_EQUAL' if equal else 'DECODER_OUTPUT_DIFFERS_DO_NOT_DEPLOY',
              'job_id': os.getenv('SLURM_JOB_ID'), 'host': socket.gethostname(),
              'source_lock_sha256': sha(lock), 'encoding_report_sha256': sha(report_file),
              'actual_parent_sacct': sacct, 'rgb_path': row['rgb_path'],
              'rgb_sha256': row['rgb_sha256'], 'modes': results,
              'original_wheel_and_all_vendor_bytes_verified': True,
              'actual_partial_feature_finite_shape_dtype_sha_stable': True,
              'actual_partial_feature_sha256': partial_sha,
              'all512_exact_clock_pts_shape_rgb_equal': equal,
              'warm_io_order_not_controlled': True, 'cpu_threads': 4,
              'model_imported': False, 'model_inference': False, 'features_written': False,
              'basis_fit': False, 'reader_training': False, 'sealed_data_read': False,
              'full78_budget_feasibility_proven': False, 'elapsed_seconds': time.monotonic()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(output), 'sha256': sha(output)}), flush=True)
    if not equal:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True)
    parser.add_argument('--source-lock-sha', required=True)
    args = parser.parse_args()
    run(args.control, args.source_lock_sha)
