"""One preregistered GPU development trial; old research artifacts stay read-only."""
import copy
import ast
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .anchored_memory import AnchoredMemoryPredictor
from .common_anchor import common_anchor_loss, recent_only_teacher_batch
from .inputs import FeatureStore, collate
from .model import DynamicsPredictor, fit_basis
from .training import TRAIN_CONDITIONS, paired_rows, regression_loss, validation_loss
from .utils import (read_json, write_json, write_csv, file_hash, seed_all, rng_state,
                    save_torch, digest)

SELECTION = 'equal_four_condition_two_horizon_mse_earliest_minimum'


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or np.isfinite(value)


def verify_control(control):
    control = Path(control)
    p = read_json(control / 'protocol.json')
    lock = read_json(control / 'source_lock.json')
    for rel, sha in lock['files'].items():
        if file_hash(control / rel) != sha:
            raise ValueError('Frozen source differs: ' + rel)
    if (p['tasks'] != ['gate', 'motion'] or p['seeds'] != [17, 29, 43]
            or p['epochs'] != 120 or p['data_seed'] != 6301):
        raise ValueError('Original fixed development boundary changed')
    configs = {}
    for task, spec in p['immutable_inputs'].items():
        if file_hash(spec['data_source_config']) != spec['data_source_config_sha256']:
            raise ValueError('Original data configuration changed')
        c = read_json(spec['data_source_config'])
        for key, expected in spec['original_contract'].items():
            if c[key] != expected:
                raise ValueError('Original scientific boundary changed: ' + key)
        if (c['experiment'] != {'stage': 'development', 'arm': 'reference', 'tasks': [task]}
                or c['data']['seed'] != 6301):
            raise ValueError('Wrong task or previously sealed data')
        if file_hash(spec['frozen_v3_verification']) != spec['frozen_v3_verification_sha256']:
            raise ValueError('Original reference verification changed')
        verified = read_json(spec['frozen_v3_verification'])
        native_source = Path('/scratch/xl4358/phase1_jepa_memory/src/phase1')
        source_hash = digest({name: file_hash(native_source / name)
                              for name in ('training.py', 'model.py', 'inputs.py')})
        if source_hash != verified['source_code_hash']:
            raise ValueError('Actual original reference source no longer matches its proof')
        # Verify the executable model/input classes, allowing only unrelated
        # additions elsewhere in the reused dependency module.
        for name, definitions in {'model.py': ['DynamicsPredictor'],
                                  'inputs.py': ['FeatureStore', 'memory_indices', 'selected_evidence', 'collate']}.items():
            def syntax(path, symbol):
                tree = ast.parse(path.read_text())
                return ast.dump(next(n for n in tree.body if getattr(n, 'name', None) == symbol),
                                include_attributes=False)
            for symbol in definitions:
                if syntax(native_source / name, symbol) != syntax(Path(__file__).parent / name, symbol):
                    raise ValueError('Original model/input executable definition differs: ' + symbol)
        if file_hash(Path(c['output']) / 'predictor' / task / 'basis.npz') != spec['basis_sha256']:
            raise ValueError('Original basis changed; fitting a new basis is forbidden')
        configs[task] = c
    return p, lock, configs


def reference_model(c, p, task, seed, basis):
    verified = read_json(p['immutable_inputs'][task]['frozen_v3_verification'])
    expected = verified['verified_models'][task + '/reference'][f'{task}/{seed}']['best_sha256']
    path = Path(c['output']) / 'predictor' / task / f'seed_{seed}' / 'best.pt'
    if file_hash(path) != expected:
        raise ValueError('Actual frozen reference weights changed')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not finite_tree(ck['model']):
        raise ValueError('Nonfinite reference')
    reference = DynamicsPredictor(len(basis['input_mean']), c['model'], (3, 6), basis)
    reference.load_state_dict(ck['model'], strict=True)
    return reference.eval(), expected


def signature(control, p, c, store, task, seed, reference_sha):
    return {'reader': 'AnchoredMemoryPredictor', 'task': task, 'seed': seed,
            'protocol_sha256': file_hash(Path(control) / 'protocol.json'),
            'source_lock_sha256': file_hash(Path(control) / 'source_lock.json'),
            'data_config_sha256': p['immutable_inputs'][task]['data_source_config_sha256'],
            'cache_signature': store.ix['signature'],
            'normalization_sha256': file_hash(Path(c['output']) / 'features/normalization.npz'),
            'basis_sha256': p['immutable_inputs'][task]['basis_sha256'],
            'reference_best_sha256': reference_sha, 'selection': SELECTION}


def train_epoch(c, store, model, optimizer, pairs, seed, epoch, device, task, stop, deadline):
    model.train()
    rng = np.random.default_rng(seed + 100003 * epoch)
    order = rng.permutation(len(pairs))
    counts = {k: 0 for k in TRAIN_CONDITIONS}
    totals = {k: np.zeros(4, dtype=np.float64) for k in TRAIN_CONDITIONS}
    steps = 0
    for start in range(0, len(order), 32):
        selected = [pairs[i] for i in order[start:start + 32]]
        rows = [r for pair in selected for r in pair]
        for condition in rng.permutation(TRAIN_CONDITIONS):
            samples = []
            for pair in selected:
                sample_seed = int(rng.integers(2 ** 32))
                samples.extend(store.sample(r, str(condition), np.random.default_rng(sample_seed)) for r in pair)
            batch = collate([s[0] for s in samples], device)
            target = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss, mse, paired = regression_loss(prediction, target, 2., .5)
            anchor = prediction.new_zeros(())
            if task == 'gate' and condition != 'recent_only':
                with torch.no_grad():
                    teacher = model.reference(recent_only_teacher_batch(batch))
                anchor = common_anchor_loss(prediction, teacher, rows, condition=str(condition))
                loss = loss + anchor
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.side.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            n = len(samples)
            counts[str(condition)] += n
            totals[str(condition)] += np.asarray([float(x.detach()) for x in (loss, mse, paired, anchor)]) * n
            if stop['requested'] or time.perf_counter() >= deadline:
                return None
    if counts != {k: 1024 for k in TRAIN_CONDITIONS} or steps != 64:
        raise ValueError('Incomplete balanced epoch')
    return {'steps': steps, 'counts': counts,
            'means': {k: (totals[k] / counts[k]).tolist() for k in TRAIN_CONDITIONS}}


def verify_completed_model(root, sig, frozen_reference_state=None):
    latest = torch.load(root / 'latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
    logs = latest['logs']
    training = read_json(root / 'training.json')
    if training['signature'] != sig or training['logs'] != logs or training['seed'] != sig['seed']:
        raise ValueError('Independent actual training log differs from checkpoint')
    if (latest['signature'] != sig or best['signature'] != sig or latest['epoch'] != 120
            or latest['step'] != 7680 or len(logs) != 120):
        raise ValueError('Incomplete model or provenance')
    for i, row in enumerate(logs, 1):
        if (row['epoch'] != i or row['step'] != i * 64
                or row['mix_counts'] != {k: 1024 for k in TRAIN_CONDITIONS}
                or not finite_tree(row)):
            raise ValueError('Epoch/log/coverage mismatch')
        score = float(np.mean([row['val_' + k + '_mse'] for k in TRAIN_CONDITIONS]))
        if not np.isclose(score, row['validation_mse'], atol=1e-14, rtol=0):
            raise ValueError('Checkpoint selection metric changed')
    earliest = min(logs, key=lambda r: r['validation_mse'])['epoch']
    if best['epoch'] != earliest or best['logs'] != logs[:earliest]:
        raise ValueError('Checkpoint was not earliest validation minimum')
    for ck in (latest, best):
        if (not finite_tree(ck['model']) or not finite_tree(ck['optimizer'])
                or ck['scheduler']['last_epoch'] != ck['epoch']
                or set(ck['rng']) != {'python', 'numpy', 'torch', 'cuda'}
                or not ck['optimizer']['state']):
            raise ValueError('Finite/RNG/scheduler proof failed')
        if any(float(state['step']) != ck['step'] for state in ck['optimizer']['state'].values()):
            raise ValueError('Optimizer boundary mismatch')
        if frozen_reference_state is not None:
            for key, tensor in frozen_reference_state.items():
                if not torch.equal(ck['model']['reference.' + key], tensor.cpu()):
                    raise ValueError('Frozen reference parameters or buffers changed')
    return {'complete': True, 'task': sig['task'], 'seed': sig['seed'], 'signature': sig,
            'completed_epochs': 120, 'final_step': 7680, 'best_epoch': earliest,
            'best_sha256': file_hash(root / 'best.pt'), 'latest_sha256': file_hash(root / 'latest.pt'),
            'frozen_reference_verified': frozen_reference_state is not None,
            'selection_metric': SELECTION, 'finite': True, 'RNG_optimizer_boundary_verified': True}


def run_training(control, output):
    started = time.perf_counter()
    p, lock, configs = verify_control(control)
    root = Path(output)
    if root.exists():
        raise ValueError('Refuse overwrite or implicit scientific retry')
    if not torch.cuda.is_available():
        raise RuntimeError('This scientific training requires actual allocated CUDA; no CPU fallback')
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa'
            or torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name(0)):
        raise ValueError('Actual course allocation / single A100 required')
    seed_all(17, 4)
    root.mkdir(parents=True)
    write_json(root / 'run_contract.json', {'protocol': p, 'source_files': lock['files'], 'phase1_go': False})
    write_json(root / 'actual_gpu.json', {'job': os.environ['SLURM_JOB_ID'], 'account': os.environ['SLURM_JOB_ACCOUNT'],
               'GPU': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'threads': torch.get_num_threads()})
    stop = {'requested': False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.update(requested=True))
    deadline = started + 6600
    proofs = {}
    for task in p['tasks']:
        c = configs[task]
        store = FeatureStore(c, max_cached=1216)
        rows = [r for r in store.manifest['records'] if r['task'] == task]
        train = paired_rows([r for r in rows if r['split'] == 'train'])
        validation = paired_rows([r for r in rows if r['split'] == 'validation'])
        if (len(train), len(validation)) != (512, 96):
            raise ValueError('Original development sample counts changed')
        basis = fit_basis(c, store, task)
        if set(basis['train_ids'].tolist()) != {r['id'] for pair in train for r in pair}:
            raise ValueError('Basis training identities differ')
        store.bases = {task: basis}
        for seed in p['seeds']:
            seed_all(seed, 4)
            reference, reference_sha = reference_model(c, p, task, seed, basis)
            frozen = cpu_tree(reference.state_dict())
            model = AnchoredMemoryPredictor(reference, basis).cuda()
            sig = signature(control, p, c, store, task, seed, reference_sha)
            model_root = root / 'predictor' / task / f'seed_{seed}'
            model_root.mkdir(parents=True)
            opt = torch.optim.AdamW(model.side.parameters(), lr=.001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 40, .5)
            logs, best_score, step, seconds = [], float('inf'), 0, 0.
            def snapshot(epoch):
                return {'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(opt.state_dict()),
                        'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'epoch': epoch, 'step': step,
                        'best': best_score, 'logs': copy.deepcopy(logs), 'seconds': seconds,
                        'signature': sig, 'seed': seed, 'task': task,
                        'resume_granularity': 'epoch_boundary', 'checkpoint_selection': SELECTION}
            save_torch(model_root / 'latest.pt', snapshot(0))
            for epoch in range(120):
                boundary = snapshot(epoch)
                began = time.perf_counter()
                result = train_epoch(c, store, model, opt, train, seed, epoch, 'cuda', task, stop, deadline)
                if result is None:
                    save_torch(model_root / 'latest.pt', boundary)
                    status = {'status': 'INTERRUPTED_OR_SOFT_BUDGET_STOP', 'task': task, 'seed': seed,
                              'completed_epochs': epoch, 'phase1_go': False}
                    write_json(root / 'outcome.json', status)
                    return status
                val = validation_loss(c, store, model, validation, 'cuda', task)
                score = float(np.mean([val[k]['mse'] for k in TRAIN_CONDITIONS]))
                elapsed = time.perf_counter() - began
                scheduler.step()
                step += result['steps']
                seconds += elapsed
                row = {'epoch': epoch + 1, 'step': step, 'validation_mse': score, 'epoch_seconds': elapsed,
                       'seconds': seconds, 'mix_counts': result['counts'], 'lr': opt.param_groups[0]['lr'],
                       **{f'train_{k}_{name}': v[i] for k, v in result['means'].items()
                          for i, name in enumerate(('loss', 'mse', 'pair_mse', 'anchor_mse'))},
                       **{f'val_{k}_{name}': v for k, means in val.items() for name, v in means.items()}}
                if not finite_tree(row):
                    raise RuntimeError('Nonfinite actual epoch')
                logs.append(row)
                improved = score < best_score
                best_score = min(best_score, score)
                ck = snapshot(epoch + 1)
                if not finite_tree(ck['model']) or not finite_tree(ck['optimizer']):
                    raise RuntimeError('Nonfinite actual checkpoint')
                save_torch(model_root / 'latest.pt', ck)
                if improved:
                    save_torch(model_root / 'best.pt', ck)
                write_csv(model_root / 'training.csv', logs)
                write_json(model_root / 'training.json', {'signature': sig, 'logs': logs, 'seed': seed})
                write_json(root / 'progress.json', {'task': task, 'seed': seed, 'epoch': epoch + 1,
                           'step': step, 'completed_models': len(proofs), 'held_out_evaluated': False})
                print(f'EPOCH {task} seed={seed} {epoch+1}/120 val={score:.8f} seconds={elapsed:.2f}', flush=True)
                if task == 'gate' and seed == 17 and epoch == 0:
                    projection = time.perf_counter() - started + elapsed * 719
                    write_json(root / 'first_epoch_budget.json', {'projection_seconds': projection,
                               'soft_seconds': 6600, 'passed': projection <= 6600, 'method_pass': False})
                    if projection > 6600:
                        status = {'status': 'COMPUTE_BUDGET_STOP', 'phase1_go': False}
                        write_json(root / 'outcome.json', status)
                        return status
            proofs[f'{task}/{seed}'] = verify_completed_model(model_root, sig, frozen)
            del model
            torch.cuda.empty_cache()
    write_json(root / 'all_models_completion.json', {'complete': True, 'models': proofs,
               'phase1_go': False, 'held_out_evaluated': False})
    status = {'status': 'ALL_SIX_MODELS_COMPLETE_EVALUATION_PENDING', 'elapsed_seconds': time.perf_counter() - started,
              'phase1_go': False}
    write_json(root / 'outcome.json', status)
    return status
