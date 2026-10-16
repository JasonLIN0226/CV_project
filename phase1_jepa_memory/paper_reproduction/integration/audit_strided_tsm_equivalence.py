"""Course CPU-only data comparison: no model, fit, inference, or label use."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time

from strided_tsm_reader import read_recording
import lmdb
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    state = subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o','7305'], text=True)
    if 'JobState=RUNNING' not in state or 'Account=csci_ga_2271-2026fa' not in state:
        raise RuntimeError('actual course data allocation required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original_root = Path('/scratch/xl4358/phase1_space_official_data/latents_train_val')
    fixed = sorted(original_root.glob('*.pt'))[:3]
    if len(fixed) != 3:
        raise RuntimeError('first three upstream completed recordings required')
    env = lmdb.open('/scratch/xl4358/phase1_space_official_data/lmdb/C10095_rgb',
                    readonly=True, lock=False, readahead=False, max_readers=32)
    rows=[]
    started=time.monotonic()
    for path in fixed:
        before=sha(path)
        old=torch.load(path, map_location='cpu', weights_only=True)
        t=time.monotonic()
        features, frames=read_recording(env, old['video_id'], 15)
        elapsed=time.monotonic()-t
        if not torch.equal(features, old['latents']) or not torch.equal(frames, old['clip_indices']):
            raise RuntimeError('not byte-equivalent to upstream features/indices')
        if not torch.equal(frames.float()/30., old['timestamps_sec']):
            raise RuntimeError('timestamp differs')
        if sha(path)!=before:
            raise RuntimeError('reference changed during read')
        rows.append({'video_id':old['video_id'],'frames':len(features),
                     'upstream_payload_sha256':before,'features_bitwise_equal':True,
                     'frame_indices_equal':True,'timestamps_equal':True,
                     'optimized_read_seconds':elapsed})
    env.close()
    report={'status':'ALL_THREE_FIXED_UPSTREAM_RECORDINGS_BITWISE_IDENTICAL',
            'rows':rows,'elapsed_seconds':time.monotonic()-started,
            'source_sha256':sha(Path(__file__).with_name('strided_tsm_reader.py')),
            'upstream_original_source_modified':False,'model_fit_training':False,
            'test_payload_read':False,'timing_not_a_full_job_guarantee':True}
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
