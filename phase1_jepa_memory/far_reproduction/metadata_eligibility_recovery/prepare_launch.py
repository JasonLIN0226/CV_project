"""Prepare original eligibility without materializing unused visual payloads."""
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
OLD = Path('/scratch/xl4358/phase1_far_launch_recovery')
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
    for base, p in [(HERE, HERE / 'source_lock.json'),
                    (OLD, OLD / 'source_lock.json'),
                    (UP, ROOT / 'upstream_source_lock.json'),
                    (ROOT, ROOT / 'evaluation_source_lock.json')]:
        lock = json.loads(p.read_text())
        assert all(sha(base / n) == h for n, h in lock['files'].items())
    assert sha(OLD / 'source_lock.json') == '4c12b5da80c6bcb50fa61bc720f1ab16dcfd0d06d64b568aee1b6f0d9ee60d03'
    monitor = OLD / 'monitor_20261014T2123.json'
    assert sha(monitor) == '3ae1ef33e57a5828010d6857b734d5748d12e402a2fd9f9bff9ba34520da4220'
    m = json.loads(monitor.read_text())
    assert m['progress.json']['stage'] == 'ORIGINAL_ELIGIBILITY_SCAN'
    assert m['progress.json']['records'] == 1280
    assert m['logs/prepare-7412.err']['bytes'] == 0
    start = subprocess.check_output(['/opt/slurm/bin/sacct', '-X', '-j', '7412', '-n', '-P', '-o', 'Start'], text=True).strip().strip('|')
    start_ns = int(datetime.datetime.fromisoformat(start).replace(tzinfo=datetime.timezone.utc).timestamp() * 1e9)
    original = ROOT / 'evaluation/job_7403/executed_data_proof.json'
    assert sha(original) == '4c8853c81766d4765fa5292be088df57ab67d0a216567de4ffa88ac035ae4541'
    proof = json.loads(original.read_text())
    ids = json.loads((UP / 'results/indices/loopnav/loopnav_latent_test.json').read_text())['sample_ids']
    assert [e['path'] for e in proof['files']] == ids and len(ids) == 3840
    files = []
    for e in proof['files']:
        for n, h in [(e['path'], e['sha256']), (str(Path(e['path']).with_suffix('.json')), e['frame_log_sha256'])]:
            st = (UP / n).stat()
            assert st.st_mtime_ns < start_ns, 'Changed since completed current SHA verification: ' + n
            files.append({'path': n, 'sha256': h, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns})
    software = json.loads((HERE / 'dreamsim_source_identity.json').read_text())
    assert all(sha(ROOT / 'vendor' / n) == h for n, h in software.items())
    save('progress.json', {'stage': 'EXISTING_CURRENT_SHA_PROOF_AND_STAT_BOUNDARY_ACCEPTED', 'records': 3840, 'elapsed_seconds': time.monotonic() - START})

    import torch
    import numpy as np
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    torch.set_num_threads(1)
    os.chdir(UP)
    mod = importlib.util.spec_from_file_location('official_eligibility', UP / 'scripts/world/eval_rollout.py')
    official = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(official)
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
    assert dataset.transform is None and not dataset.event_frac
    real_load = dataset.backend.load_clip

    def shape_load(sample):
        loaded = real_load(sample)
        assert len(loaded) == 4 and loaded[3] is None, 'Unexpected retrieval/audio payload; refuse optimization'
        return (loaded[0].to(device='meta'), *loaded[1:])

    order = np.random.default_rng(int(cfg.seed)).permutation(len(dataset))
    paired = []
    for pos in np.linspace(0, len(order) - 1, 16, dtype=int):
        idx = int(order[pos])
        dataset.backend.load_clip = real_load
        random.seed(int(cfg.seed) + idx)
        a = dataset[idx]
        ea = official.build_episode(a, cfg, ds_spec, int(cfg.seed) + idx)
        dataset.backend.load_clip = shape_load
        random.seed(int(cfg.seed) + idx)
        b = dataset[idx]
        eb = official.build_episode(b, cfg, ds_spec, int(cfg.seed) + idx)
        assert a['frames'].shape == b['frames'].shape and a['frames'].dtype == b['frames'].dtype
        for k in ['pose', 'context_indices', 'goal_indices', 'context_mask']:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), (idx, k)
        assert a['video_id'] == b['video_id']
        assert (ea is None) == (eb is None)
        if ea is not None:
            for k in ['ctx_pose_g', 'goal_pose_g', 'ctx_idx', 'goal_idx', 'ctx_mask', 'targets']:
                assert np.array_equal(ea[k], eb[k]), (idx, k)
            assert ea['eff_len'] == eb['eff_len'] and ea['seed'] == eb['seed']
        paired.append({'index': idx, 'eligible': ea is not None, 'steps': None if ea is None else len(ea['targets']), 'exact_metadata_axis_match': True})
        del a, b, ea, eb
    save('paired_original_shape_equivalence.json', {'paired_real_records': paired, 'visual_values_not_compared_or_used_for_eligibility': True})
    dataset.backend.load_clip = shape_load
    eligible = []
    for scanned, idx in enumerate(order, 1):
        idx = int(idx)
        random.seed(int(cfg.seed) + idx)
        item = dataset[idx]
        ep = official.build_episode(item, cfg, ds_spec, clip_seed=int(cfg.seed) + idx)
        if ep is not None:
            eligible.append([idx, len(ep['targets']), ep['video_id']])
        del item, ep
        if scanned % 128 == 0:
            save('progress.json', {'stage': 'ORIGINAL_FUNCTIONS_METADATA_ELIGIBILITY_SCAN', 'records': scanned, 'total': len(dataset), 'elapsed_seconds': time.monotonic() - START})
    assert len(dataset) == scanned == 3840 and len(eligible) == 3796
    params = {k: (None if cfg[k] is None else int(cfg[k]) if k != 'sampler' else cfg[k]) for k in ('num_clips', 'start_offset', 'rollout_stride', 'min_steps', 'max_steps', 'context_pool_size', 'seed')}
    params.update(dataset=dn, min_coverage=None if cfg.min_coverage is None else float(cfg.min_coverage))
    params['min_steps'] = max(1, params['min_steps'] or 0)
    assert cfg.min_coverage is None
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
        if p.is_file() and p.suffix != '.pyc':
            metrics.append({'path': str(p.relative_to(UP)), 'sha256': sha(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns})
    for e in files:
        st = (UP / e['path']).stat()
        assert (st.st_size, st.st_mtime_ns) == (e['bytes'], e['mtime_ns'])
    assert sha(ROOT / 'evaluation/job_7403/executed_model_proof.json') == 'c0b51f927d27a93f8cc4d6758243ccce8784b01604c9f15ef881c695dfdb014a'
    save('certificate.json', {'status': 'CURRENT_IDENTITIES_AND_OFFICIAL_ELIGIBILITY_COMPLETE', 'job_id': int(os.environ['SLURM_JOB_ID']), 'test_records': 3840, 'frames': proof['frames'], 'files': files, 'models': models, 'metrics': metrics, 'eligible_clips_sha256': sha(OUT / 'eligible_clips.json'), 'original_numeric_proof_sha256': sha(original), 'original_model_proof_sha256': sha(ROOT / 'evaluation/job_7403/executed_model_proof.json'), 'original_numeric_proof_reused_after_exact_current_SHA': True, 'current_SHA_audit_source_job': 7412, 'current_SHA_audit_evidence_sha256': sha(monitor), 'stat_boundary_all_data_precedes_SHA_audit_job_start': True, 'new_numeric_tensor_traversal_claimed': False, 'new_full_current_SHA_scan_claimed': False, 'eligibility_only_shape_tensor_optimization': True, 'paired_equivalence_sha256': sha(OUT / 'paired_original_shape_equivalence.json'), 'original_scientific_source_lock_sha256': sha(ROOT / 'upstream_source_lock.json'), 'source_lock_sha256': sha(HERE / 'source_lock.json'), 'prior_GPU_seconds': 1580, 'remaining_soft_GPU_seconds': 12220, 'elapsed_seconds': time.monotonic() - START, 'model_forward_or_training': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save('failure.json', {'error': str(e), 'type': type(e).__name__, 'elapsed_seconds': time.monotonic() - START})
        raise
