from pathlib import Path
import builtins
import datetime
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/scratch/xl4358/phase1_far_reference')
UP = ROOT / 'upstream'
START = time.monotonic()
OUT = HERE / 'output' / ('job_' + os.environ['SLURM_JOB_ID'])
OUT.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(ROOT / 'vendor'), str(UP / 'src'), str(HERE)]
from online_memory import OnlineMemory


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    p = OUT / name
    p.write_text(json.dumps(data, indent=2) + '\n')


def main():
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from far.data.transforms import global_to_local_pose, normalize_data
    from far.constants import hub_cache_dir
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'A100' in torch.cuda.get_device_name(0)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    proof = json.loads((HERE / 'input_proof.json').read_text())
    assert sha(HERE / 'protocol.json') == proof['protocol_sha256']
    locks = [HERE / 'source_lock.json', ROOT / 'upstream_source_lock.json']
    source_identity = {}
    for lock in locks:
        entries = json.loads(lock.read_text())['files']
        base = HERE if lock.parent == HERE else UP
        if isinstance(entries, list):
            entries = {x['path']: x['sha256'] for x in entries}
        for n, h in entries.items():
            assert sha(base / n) == h, n
        source_identity[str(lock)] = sha(lock)
    for e in proof['files']:
        assert sha(UP / e['path']) == e['sha256'], e['path']
    for e in proof['models']:
        assert sha(UP / 'models' / e['path']) == e['sha256'], e['path']
    spec_mod = importlib.util.spec_from_file_location('official_phase1_reference', UP / 'scripts/world/eval_rollout.py')
    official = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(official)
    os.chdir(UP)
    model, dc = official.build_model(str(UP / 'models/loopnav/far_multicue'), '0700000', 'cuda')
    numeric = {}
    for n, v in model.state_dict().items():
        if v.is_floating_point():
            assert torch.isfinite(v).all().item(), n
            numeric[n] = {'shape': list(v.shape), 'dtype': str(v.dtype)}
    save('executed_model_proof.json', {'all_actual_loaded_floating_tensors_finite': True,
         'tensors': numeric, 'input_proof_sha256': sha(HERE / 'input_proof.json'),
         'source_locks': source_identity, 'GPU': torch.cuda.get_device_name(0),
         'threads': torch.get_num_threads(), 'torch': torch.__version__, 'inference_started': False})
    with initialize_config_dir(config_dir=str(UP / 'configs'), version_base=None):
        cfg = compose(config_name='eval_rollout_loopnav')
    cfg.pnp_chunk_size = 1
    cfg.worldmem_pool_size = None
    dn = str(cfg.dataset)
    if dn not in dc:
        dc[dn] = OmegaConf.load(UP / 'configs/dataset' / (dn + '.yaml'))
    dc[dn].test.base_dataset.sampler = dc[dn].sampler.context_traj
    dc[dn].test.base_dataset.sampler.context_pool_size = 1000
    dc.len_traj_pred = 1024
    dataset = instantiate(dc[dn].test.base_dataset)
    spec = instantiate(dc[dn].spec)
    stats = {k: v.unsqueeze(0).numpy() for k, v in spec.local_pos_stats.items()}
    model.core.diffusion = instantiate(dc.model.diffusion, timestep_respacing='ddim20')
    model.core.num_memories = 11
    mem = model.core.memory
    mem.chunk_size = 1
    import lpips
    try:
        import peft.import_utils as pi
        import peft.tuners.lora.torchao as pt
        pi.is_torchao_available = lambda *a, **k: False
        pt.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
    from dreamsim import dreamsim
    lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    ds, _ = dreamsim(pretrained=True, device='cuda', cache_dir=hub_cache_dir())

    @torch.no_grad()
    def rank(candidates, current):
        entries = candidates + [current]
        poses = global_to_local_pose(np.stack([x['pose'] for x in entries]), current['pose'])
        poses[:, :3] /= spec.meters_per_waypoint
        times = torch.tensor([(x['frame'] - current['frame']) / spec.fps for x in entries], dtype=torch.float32, device='cuda')
        action0 = np.zeros(poses.shape[-1], dtype=np.float32)
        action0[:3] = normalize_data(action0[None, :3], stats)[0]
        old = mem.num_sets
        mem.num_sets = 1
        try:
            result = mem(torch.stack([x['latent'] for x in entries])[None].cuda(),
                         torch.as_tensor(poses, dtype=torch.float32, device='cuda')[None], times[None],
                         torch.as_tensor(action0, device='cuda')[None, None],
                         torch.zeros((1, 1), device='cuda'),
                         context_mask=torch.ones((1, len(entries)), dtype=torch.bool, device='cuda'),
                         num_memories=16)
        finally:
            mem.num_sets = old
        selected = result[2][0, 0, 0].detach().cpu().numpy()
        source_times = times[:-1].cpu().numpy()
        chosen = [int(np.argmin(np.abs(source_times - t))) for t in selected]
        assert all(abs(float(source_times[i]) - float(t)) < 1e-4 for i, t in zip(chosen, selected))
        return chosen

    @torch.no_grad()
    def rollout(ep, policy, seed):
        state = OnlineMemory(policy, seed, rank=rank)
        order = sorted({int(i): j for j, i in enumerate(ep['ctx_idx']) if ep['ctx_mask'][j]}.items())
        assert len(order) >= 17
        for idx, j in order:
            state.observe({'id': 'o:' + str(idx), 'frame': idx,
                           'pose': ep['ctx_pose_g'][j], 'latent': ep['ctx_lat'][j].clone()})
        preds = []
        origin = state.current['frame']
        for t in ep['targets']:
            entries = state.read()
            idx, pose = int(ep['goal_idx'][t]), ep['goal_pose_g'][t]
            local = global_to_local_pose(np.stack([x['pose'] for x in entries] + [pose]), state.current['pose'])
            local[:, :3] /= spec.meters_per_waypoint
            action = local[-1].copy()
            action[:3] = normalize_data(action[None, :3], stats)[0]
            pred = model.core.sample_latent(
                torch.stack([x['latent'] for x in entries])[None].cuda(),
                torch.as_tensor(action, dtype=torch.float32, device='cuda')[None],
                torch.tensor([(idx - origin) / spec.max_frame_offset], device='cuda'),
                num_goals=1, context_poses=torch.as_tensor(local[:-1], dtype=torch.float32, device='cuda')[None],
                context_times=torch.tensor([[(x['frame'] - origin) / spec.fps for x in entries]], device='cuda'),
                context_mask=torch.ones((1, len(entries)), dtype=torch.bool, device='cuda'), use_ddim=True)[0].float().cpu()
            preds.append(pred)
            state.observe({'id': 'g:' + str(idx), 'frame': idx, 'pose': pose.copy(), 'latent': pred})
            if time.monotonic() - START >= protocol['budget']['new_soft_seconds']:
                raise RuntimeError('PHASE1_SOFT_BUDGET_STOP')
        return torch.stack(preds), state.trace

    @torch.no_grad()
    def metrics(pred, goal):
        values = [[], [], []]
        for k in range(0, len(pred), 8):
            p = model.tokenizer.detokenize(pred[k:k+8].cuda()).clamp(-1, 1) * .5 + .5
            g = model.tokenizer.detokenize(goal[k:k+8].cuda()).clamp(-1, 1) * .5 + .5
            part = [-10 * torch.log10(((p-g)**2).flatten(1).mean(1)),
                    lp(p*2-1, g*2-1).flatten(), ds(g, p).flatten()]
            for dst, x in zip(values, part):
                assert torch.isfinite(x).all().item()
                dst.append(x.cpu().numpy())
        return {n: np.concatenate(v) for n, v in zip(['psnr', 'lpips', 'dreamsim'], values)}

    done = 0
    first_started = time.monotonic()
    for position, (idx, steps, video) in enumerate(proof['selected_clips']):
        for seed in protocol['seeds']:
            random.seed(seed + idx)
            item = dataset[idx]
            ep = official.build_episode(item, cfg, spec, seed + idx)
            assert len(ep['targets']) == steps and ep['video_id'] == video
            for policy in protocol['policies']:
                torch.manual_seed(seed + idx)
                pred, trace = rollout(ep, policy, seed + idx)
                scores = metrics(pred, ep['goal_lat'][ep['targets']])
                np.savez_compressed(OUT / f'{idx}_{seed}_{policy}.npz', **scores,
                    frame_idx=ep['goal_idx'][ep['targets']], video_id=np.array(video), seed=seed)
                save(f'trace_{idx}_{seed}_{policy}.json', trace)
                done += steps
                save('progress.json', {'completed_steps': done, 'records': (position*3+seed)*4+protocol['policies'].index(policy)+1,
                    'elapsed_seconds': time.monotonic()-START})
            del item, ep
        if position == 0:
            elapsed = time.monotonic()-START
            work = time.monotonic()-first_started
            projection = max(elapsed + work*7,
                elapsed + (proof['total_target_steps']-done)*work/done)
            save('first_clip_budget.json', {'projection_seconds': projection, 'soft_seconds': protocol['budget']['new_soft_seconds'],
                 'passed': projection <= protocol['budget']['new_soft_seconds'], 'complete_clip_all_policies_all_seeds': True})
            if projection > protocol['budget']['new_soft_seconds']:
                raise RuntimeError('PHASE1_FULL_FIXED_PILOT_BUDGET_EXCEEDED')
    save('completion.json', {'status': 'ALL_FIXED_PHASE1_PREDICTIONS_COMPLETE_AUDIT_AND_AGGREGATION_PENDING',
         'records': 96, 'completed_steps': done, 'elapsed_seconds': time.monotonic()-START})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save('failure.json', {'type': type(error).__name__, 'error': str(error),
            'elapsed_seconds': time.monotonic()-START, 'no_adaptive_resubmission': True})
        raise
