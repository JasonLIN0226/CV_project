"""Train one Synergy reader (multihead or item_attention) on encoded TRAIN/VALIDATION features.

Extracted unchanged from experiments/phase2/run_reader_aggregation.py (only the two import lines inside train_one changed).
Frozen recipe: seed 17, AdamW, 1,024 updates, pure feature MSE on complete factorial groups,
memory sizes 0/1/2/3 in rotation, the checkpoint with the lowest validation MSE is kept. `legacy`
is the dataset module, under the name the original code used for it.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from . import dataset as legacy
from ...predictors.readers import build_reader, describe_reader

read, write, sha, emit = legacy.read, legacy.write, legacy.sha, legacy.emit


def initialize_model(c, base_c, basis, variant, common_state):
    import torch
    torch.manual_seed(c['seed'])
    model = build_reader(len(basis['input_mean']), base_c['reader'], legacy.H,
                         basis, variant, heads=c['heads'])
    state = model.state_dict()
    shared = [key for key in state if key in common_state and state[key].shape == common_state[key].shape]
    state.update({key: common_state[key].clone() for key in shared})
    model.load_state_dict(state, strict=True)
    return model, shared


def train_one(c, base_c, source, out, variant, device, common_state):
    import torch
    from ...predictors.phase1.inputs import collate
    from ...predictors.phase1.utils import seed_all
    dest = out/'readers'/variant
    dest.mkdir(parents=True, exist_ok=False)
    basis = dict(np.load(source/'basis.npz', allow_pickle=False))
    data = legacy.Data(source/'data', basis)
    seed_all(c['seed'], 4)
    model, shared = initialize_model(c, base_c, basis, variant, common_state)
    architecture = describe_reader(model)
    architecture['shared_initial_state_keys'] = shared
    architecture['parameter_bytes_float32'] = architecture['trainable_parameters']*4
    architecture['adam_moment_bytes_float32'] = architecture['trainable_parameters']*8
    write(dest/'architecture.json', architecture)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, c['scheduler']['milestones'], gamma=c['scheduler']['gamma'])
    tg, vg = legacy.grouped(data.records, 'train'), legacy.grouped(data.records, 'validation')
    rng = np.random.default_rng(c['seed'])
    # Deterministic input-only cache: all arms see the same capsules and subset schedule.
    cached = {}
    for r in data.records:
        if r['split'] in ('train', 'validation'):
            cap = data.capsule(r)
            cached[r['id']] = (cap, (data.project(cap['z']), data.project(cap['current'])), data.target(r))
    schedule = hashlib.sha256()
    def make(group, size, random, record_schedule=False):
        keep = sorted(random.choice(3, size, replace=False).tolist())
        samples, targets = [], []
        for r in group:
            cap, pc, target = cached[r['id']]
            samples.append(legacy.reader_sample(cap, keep, data.actual_plan(r), basis, projection_cache=pc))
            targets.append(target)
        if record_schedule:
            schedule.update(json.dumps([[r['id'] for r in group], keep], separators=(',', ':')).encode())
        return collate(samples, device), torch.from_numpy(np.stack(targets)).to(device)
    def validation():
        model.eval(); cells = {}
        with torch.inference_mode():
            for ti, task in enumerate(legacy.TASKS):
                for size in range(4):
                    vrng = np.random.default_rng(17000+100*ti+size)
                    vals = []
                    for group in vg[task]:
                        batch, y = make(group, size, vrng)
                        vals.append(float((model(batch)-y).square().mean()))
                    cells[task+'/'+str(size)] = float(np.mean(vals))
        return {'equal_task_size_mse': float(np.mean(list(cells.values()))), 'cells': cells}
    best, logs, began = float('inf'), [], time.perf_counter()
    for step in range(c['steps']+1):
        if step % c['validate_every'] == 0 or step == c['steps']:
            v = validation(); logs.append({'step': step, **v})
            if v['equal_task_size_mse'] < best:
                best = v['equal_task_size_mse']
                torch.save({'model': model.state_dict(), 'variant': variant, 'heads': c['heads'],
                            'step': step, 'validation': v, 'basis_sha256': sha(source/'basis.npz'),
                            'model_config': base_c['reader']}, dest/'reader_selected.pt')
            write(dest/'learning_curve.json', logs)
            emit('aggregation_training', variant=variant, step=step, validation=v['equal_task_size_mse'])
        if step == c['steps']:
            break
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses = []
        for task in legacy.TASKS:
            gl = []
            for gi in rng.choice(len(tg[task]), c['groups_per_task'], replace=False):
                batch, y = make(tg[task][int(gi)], step % 4, rng, True)
                gl.append(legacy.group_loss(model(batch), y))
            losses.append(torch.stack(gl).mean())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite reader loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
    selected = torch.load(dest/'reader_selected.pt', map_location='cpu', weights_only=False)
    result = {'variant': variant, 'checkpoint': str((dest/'reader_selected.pt').relative_to(out)),
              'checkpoint_sha256': sha(dest/'reader_selected.pt'), 'basis_sha256': sha(source/'basis.npz'),
              'selected_step': selected['step'], 'validation': selected['validation'],
              'training_schedule_sha256': schedule.hexdigest(), 'seconds': time.perf_counter()-began,
              'architecture': architecture, 'selection_splits': ['validation'], 'seed': c['seed']}
    write(dest/'reader_locked.json', result)
    return result
