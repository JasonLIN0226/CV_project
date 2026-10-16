"""Read-only preparation of original FAR inputs and original eligibility cache."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/scratch/xl4358/phase1_far_reference')
UP = ROOT / 'upstream'
OUT = HERE / 'output' / ('job_' + os.environ['SLURM_JOB_ID'])
START = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(n, x):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / n
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(x, indent=2) + '\n')
    q.replace(p)


def main():
    assert os.environ['SLURM_JOB_ACCOUNT'] == 'csci_ga_2271-2026fa'
    assert not os.environ.get('CUDA_VISIBLE_DEVICES', '')
    lock = json.loads((HERE / 'source_lock.json').read_text())
    assert all(sha(HERE / n) == h for n, h in lock['files'].items())
    for name, base in [('evaluation_source_lock.json', ROOT), ('upstream_source_lock.json', UP)]:
        x = json.loads((ROOT / name).read_text())
        assert all(sha(base / n) == h for n, h in x['files'].items())
    terminal = subprocess.check_output(['/opt/slurm/bin/sacct', '-X', '-j', '7403', '-n', '-P', '-o', 'JobID,State,ElapsedRaw'], text=True).strip().split('|')
    assert terminal[:2] == ['7403', 'CANCELLED by 0'] and int(terminal[2]) == 1580
    assert not list((ROOT / 'evaluation/job_7403').glob('scalars/**/*.npz'))
    for n, h in [('executed_data_proof.json', '4c8853c81766d4765fa5292be088df57ab67d0a216567de4ffa88ac035ae4541'), ('executed_model_proof.json', 'c0b51f927d27a93f8cc4d6758243ccce8784b01604c9f15ef881c695dfdb014a')]:
        assert sha(ROOT / 'evaluation/job_7403' / n) == h
    software = json.loads((HERE / 'dreamsim_source_identity.json').read_text())
    assert all(sha(ROOT / 'vendor' / n) == h for n,h in software.items())
    proof = json.loads((ROOT / 'evaluation/job_7403/executed_data_proof.json').read_text())
    ids = json.loads((UP / 'results/indices/loopnav/loopnav_latent_test.json').read_text())['sample_ids']
    assert [x['path'] for x in proof['files']] == ids and len(ids) == 3840
    files = []
    for i, entry in enumerate(proof['files']):
        for name, expected in [(entry['path'], entry['sha256']), (str(Path(entry['path']).with_suffix('.json')), entry['frame_log_sha256'])]:
            p = UP / name
            before = p.stat()
            assert sha(p) == expected, name
            after = p.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            files.append({'path': name, 'sha256': expected, 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns})
        if (i + 1) % 128 == 0:
            save('progress.json', {'stage': 'CURRENT_CONTENT_IDENTITIES', 'records': i + 1, 'total': 3840, 'elapsed_seconds': time.monotonic() - START})
    import torch
    torch.set_num_threads(1)
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    os.chdir(UP)
    spec = importlib.util.spec_from_file_location('official_data_preparation', UP / 'scripts/world/eval_rollout.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    with initialize_config_dir(config_dir=str(UP / 'configs'), version_base=None):
        cfg = compose(config_name='eval_rollout_loopnav')
    data_cfg = OmegaConf.load(UP / str(list(cfg.arms.values())[-1].run_dir) / 'config.yaml')
    dn = str(cfg.dataset)
    if dn not in data_cfg:
        data_cfg[dn] = OmegaConf.load(UP / 'configs/dataset' / (dn + '.yaml'))
    data_cfg[dn].test.base_dataset.sampler = data_cfg[dn].sampler.context_traj
    data_cfg[dn].test.base_dataset.sampler.context_pool_size = int(cfg.context_pool_size)
    data_cfg.len_traj_pred = 1024
    dataset = instantiate(data_cfg[dn].test.base_dataset)
    ds_spec = instantiate(data_cfg[dn].spec)
    import numpy as np
    params = {k: (None if cfg[k] is None else int(cfg[k]) if k != 'sampler' else cfg[k]) for k in ('num_clips', 'start_offset', 'rollout_stride', 'min_steps', 'max_steps', 'context_pool_size', 'seed')}
    params.update(dataset=dn, min_coverage=None if cfg.min_coverage is None else float(cfg.min_coverage))
    params['min_steps'] = max(1, params['min_steps'] or 0)
    assert cfg.min_coverage is None
    eligible = []
    for scanned, idx in enumerate(np.random.default_rng(int(cfg.seed)).permutation(len(dataset)), 1):
        idx = int(idx)
        random.seed(int(cfg.seed) + idx)
        item = dataset[idx]
        ep = official.build_episode(item, cfg, ds_spec, clip_seed=int(cfg.seed) + idx)
        if ep is not None:
            eligible.append([idx, len(ep['targets']), ep['video_id']])
        del item, ep
        if scanned % 128 == 0:
            save('progress.json', {'stage': 'ORIGINAL_ELIGIBILITY_SCAN', 'records': scanned, 'total': len(dataset), 'elapsed_seconds': time.monotonic() - START})
    assert len(dataset) == scanned == 3840 and len(eligible) == 3796
    save('eligible_clips.json', {'params': params, 'clips': eligible, 'scanned': scanned})
    models = json.loads((ROOT / 'bundles.json').read_text())['bundles']
    models = [e for e in models if e.get('corpus') == 'loopnav' or e['path'] == 'oasis_500m_vit_vae.pth']
    for e in models:
        assert sha(UP / 'models' / e['path']) == e['sha256']
    metrics = []
    cache = UP / 'models/cache/hub'
    for n in ['clip_vitb16_pretrain.pth.tar', 'open_clip_vitb16_pretrain.pth.tar', 'dino_vitb16_pretrain.pth', 'ensemble_lora/adapter_model.safetensors', 'ensemble_lora/adapter_config.json', 'checkpoints/alexnet-owt-7be5be79.pth', 'checkpoints/dino_vitbase16_pretrain.pth']:
        assert (cache / n).is_file(), n
    for p in sorted(cache.rglob('*')):
        if p.is_file() and p.suffix not in ['.pyc']:
            metrics.append({'path': str(p.relative_to(UP)), 'sha256': sha(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns})
    for e in files:
        s = (UP / e['path']).stat()
        assert (s.st_size, s.st_mtime_ns) == (e['bytes'], e['mtime_ns'])
    save('certificate.json', {'status': 'CURRENT_IDENTITIES_AND_OFFICIAL_ELIGIBILITY_COMPLETE', 'job_id': int(os.environ['SLURM_JOB_ID']), 'test_records': 3840, 'frames': proof['frames'], 'files': files, 'models': models, 'metrics': metrics, 'eligible_clips_sha256': sha(OUT / 'eligible_clips.json'), 'original_numeric_proof_sha256': sha(ROOT / 'evaluation/job_7403/executed_data_proof.json'), 'original_model_proof_sha256': sha(ROOT / 'evaluation/job_7403/executed_model_proof.json'), 'original_numeric_proof_reused_after_exact_current_SHA': True, 'new_numeric_tensor_traversal_claimed': False, 'original_scientific_source_lock_sha256': sha(ROOT / 'upstream_source_lock.json'), 'source_lock_sha256': sha(HERE / 'source_lock.json'), 'prior_GPU_seconds': 1580, 'remaining_soft_GPU_seconds': 12220, 'elapsed_seconds': time.monotonic() - START, 'model_forward_or_training': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save('failure.json', {'error': str(e), 'type': type(e).__name__, 'elapsed_seconds': time.monotonic() - START})
        raise
