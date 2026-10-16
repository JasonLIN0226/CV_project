"""Preparation: fixed relational predictor training; no implicit submission.

This module is not an executable scientific trial until its complete protocol,
source lock, native input proof, rate probe, and original-gate evaluator exist.
"""
"""One preregistered GPU development trial; old research artifacts stay read-only."""
import copy
import ast
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .relational_memory import RelationalMemoryPredictor
from .exact_runtime_cache import CachedFeatureStore, cached_validation_loss
from .common_anchor import common_anchor_loss, recent_only_teacher_batch
from .inputs import FeatureStore, collate
from .model import DynamicsPredictor, fit_basis
from .training import TRAIN_CONDITIONS, paired_rows, regression_loss, validation_loss
from .utils import (read_json, write_json, write_csv, file_hash, seed_all, rng_state,
                    save_torch, digest, restore_rng)

from .proposal_trial import (SELECTION, cpu_tree, finite_tree, verify_control, reference_model, verify_completed_model)
from .resource_guard import full_sequence_projection, elapsed_from_allocation
from .data import load_episode
from .encoders import build_encoder
from .common_anchor_legacy import recent_only_teacher_batch as original_recent_only_teacher_batch
import math
import subprocess
import datetime


def learning_rate_factor(epoch_index):
    """One fixed schedule for every task and seed; no validation-dependent tuning."""
    if epoch_index < 0:
        raise ValueError('Nonnegative epoch boundary required')
    if epoch_index < 5:
        return (epoch_index + 1) / 5.
    progress = min((epoch_index - 4) / 115., 1.)
    return .1 + .9 * (1 + math.cos(math.pi * progress)) / 2


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def signature(control, protocol, config, store, task, seed, reference_sha):
    return {'reader': 'RelationalMemoryPredictor', 'task': task, 'seed': seed,
            'protocol_sha256': file_hash(Path(control) / 'protocol.json'),
            'source_lock_sha256': file_hash(Path(control) / 'source_lock.json'),
            'repair_decision_sha256': protocol['repair_decision_sha256'],
            'data_config_sha256': protocol['immutable_inputs'][task]['data_source_config_sha256'],
            'cache_signature': store.ix['signature'],
            'normalization_sha256': file_hash(Path(config['output']) / 'features/normalization.npz'),
            'basis_sha256': protocol['immutable_inputs'][task]['basis_sha256'],
            'reference_best_sha256': reference_sha, 'selection': SELECTION,
            'optimizer': 'AdamW_lr0.0003_wd0.0001_clip1_warmup5_cosine120',
            **({'epoch_execution_provenance': {'epoch1': protocol['resume']['partials'][task+'/latest'],
                  'epoch1_original_job': 7524, 'epoch1_original_source_lock_sha256': protocol['resume']['old_source_lock_sha256'],
                  'epochs2_to120_source_lock_sha256': file_hash(Path(control)/'source_lock.json')}}
               if protocol.get('resume') and seed == 17 else {})}


def actual_allocation_start():
    job = os.environ.get('SLURM_JOB_ID')
    if (not job or os.environ.get('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or 'A100' not in torch.cuda.get_device_name(0)):
        raise ValueError('Actual single course A100 allocation required; no CPU training fallback')
    command = ['/opt/slurm/bin/sacct', '-j', job, '-X', '--noheader', '--parsable2',
               '--format=JobID,State,Start,Account,AllocTRES']
    native = subprocess.run(command, check=True, capture_output=True, text=True,
                            env={**os.environ, 'TZ': 'UTC'}).stdout
    rows = [line.split('|') for line in native.splitlines() if line]
    if len(rows) != 1:
        raise ValueError('One actual allocation record required')
    row = rows[0]
    tres = dict(v.split('=', 1) for v in row[4].split(','))
    if (row[0] != job or row[1] != 'RUNNING' or row[3] != 'csci_ga_2271-2026fa'
            or tres.get('gres/gpu') != '1' or row[2] in {'Unknown', 'None'}):
        raise ValueError('Actual allocation identity / GPU TRES / StartTime mismatch')
    start = row[2] + 'Z'
    elapsed_from_allocation(start)
    return start, native


def verify_repair_contract(control):
    protocol, lock, configs = verify_control(control)
    decision = read_json(Path(control) / 'repair_decision.json')
    if file_hash(Path(control) / 'repair_decision.json') != protocol['repair_decision_sha256']:
        raise ValueError('The single fixed post-diagnosis repair decision changed')
    if (decision['candidate'] != 'RelationalMemoryPredictor'
            or decision['training']['learning_rate'] != .0003
            or decision['training']['epochs'] != 120
            or decision['training']['steps_each_model'] != 7680):
        raise ValueError('Fixed repair choice / original full training count changed')
    native = read_json(Path(control).parent / 'native_preflight.json')
    if native['source_lock_sha256'] != file_hash(Path(control) / 'source_lock.json') or not native['all_exact']:
        raise ValueError('Complete actual new-source input/cache/RNG proof required')
    if native['prior_GPU_used_seconds'] != 6411 or native['remaining_shared_GPU_seconds'] != 22389:
        raise ValueError('Actual reconciled eight-hour shared GPU budget proof required')
    receipt = read_json(Path(control).parent / 'training_submission.json')
    if (receipt.get('status') != 'SUBMITTED_ACTUAL_RECEIPT'
            or str(receipt.get('job_id')) != os.environ.get('SLURM_JOB_ID')):
        raise ValueError('Only the single actual registered GPU receipt can train this candidate')
    parent = subprocess.run(['/opt/slurm/bin/sacct', '-j', str(native['job']), '-X', '--noheader',
                             '--parsable2', '--format=State,ExitCode'],
                            check=True, capture_output=True, text=True).stdout.strip()
    if parent != 'COMPLETED|0:0':
        raise ValueError('Actual completed CPU input preflight parent required')
    return protocol, lock, configs, decision, native


def measure_remaining_sequence_resources(contexts, root, stop, deadline):
    """Cost-only probes on existing development data, charged to this GPU job.

    Fresh 9137 is not generated or opened here. Reference probes are discarded
    throughput work, never claimed to be fresh confirmation reference models.
    """
    actual_allocation_start()
    measured = {}
    for task, (config, store, train, validation, basis) in contexts.items():
        seed_all(7301, 4)
        model = DynamicsPredictor(len(basis['input_mean']), config['model'], (3, 6), basis).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        torch.cuda.synchronize(); began = time.perf_counter()
        # Original baseline workload has no frozen-reference anchoring branch.
        result = train_epoch(config, store, model, optimizer, train, 7301, 0, 'cuda',
                             'resource_reference_profile_only', stop, deadline)
        if result is None:
            raise RuntimeError('Interrupted reference cost probe; no scientific completion claim')
        cached_validation_loss(config, store, model, validation, 'cuda', task)
        torch.cuda.synchronize(); measured[task] = time.perf_counter() - began
        del model, optimizer
        torch.cuda.empty_cache()
    first_config, store, train, _, _ = contexts['gate']
    config = copy.deepcopy(first_config)
    config['encoder']['local_files_only'] = True  # existing official files only
    if config['encoder']['backend'] != 'ijepa' or config['encoder']['batch_size'] != 2 or config['encoder']['dtype'] != 'float32':
        raise ValueError('Original official float32 batch2 encoder is required')
    frames = np.concatenate([load_episode(config, row, keys=('rgb',))['rgb']
                             for pair in train[:2] for row in pair])[:32]
    if len(frames) != 32:
        raise ValueError('Complete existing development RGB cost probe required')
    torch.cuda.synchronize(); encoder_began = time.perf_counter()
    encoder = build_encoder(config, torch.device('cuda'))
    if (encoder.metadata['resolved_revision'] != config['encoder']['revision']
            or encoder.metadata['pooling'] != 'grid2' or encoder.metadata['dtype'] != 'float32'
            or any(p.requires_grad for p in encoder.model.parameters())):
        raise ValueError('Actual frozen official encoder identity / precision differs')
    for value in encoder.model.state_dict().values():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite actual encoder tensor')
    a = encoder.encode(frames[:2]); reversed_a = encoder.encode(frames[:2][::-1].copy())
    duplicate_a = encoder.encode(np.repeat(frames[:1], 2, axis=0))
    exact = np.array_equal(a, reversed_a[::-1]) and np.array_equal(a[:1], duplicate_a[:1]) and np.array_equal(duplicate_a[:1], duplicate_a[1:])
    if not exact or not np.isfinite(a).all() or a.shape != (2, 5120):
        raise ValueError('Actual batch2 encoder finite/deterministic/permutation probe failed')
    torch.cuda.synchronize(); encoder_setup_seconds = time.perf_counter() - encoder_began
    rates = []
    for start in range(0, 32, 8):
        torch.cuda.synchronize(); began = time.perf_counter()
        for index in range(start, start + 8, 2):
            encoded = encoder.encode(frames[index:index + 2])
            if encoded.shape != (2, 5120) or not np.isfinite(encoded).all():
                raise ValueError('Actual encoder throughput probe produced invalid features')
        torch.cuda.synchronize(); rates.append(8 / (time.perf_counter() - began))
    result = {'reference_epoch_seconds': measured, 'encoder_frames_per_second': min(rates),
              'encoder_setup_and_actual_weight_proof_seconds': encoder_setup_seconds,
              'encoder_probe_rates': rates, 'actual_original_encoder_batch2_probe_exact': exact,
              'confirmation_full_frame_bound': 198656, 'uses_RGB_deduplication': False,
              'original_encoder_metadata': encoder.metadata,
              'only_existing_6301_training_RGB': True, 'fresh9137_opened': False,
              'reference_probe_is_confirmation_training': False, 'all_probe_wall_time_is_charged': True}
    write_json(root / 'native_resource_probe.json', result)
    del encoder
    torch.cuda.empty_cache()
    return result

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
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1., error_if_nonfinite=True)
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




def run_training(control, output):
    # One fixed post-diagnosis repair; old completed sources remain immutable.
    started = time.perf_counter()
    p, lock, configs, decision, fast_proof = verify_repair_contract(control)
    approved_budget = read_json(Path(control) / 'authorization.json')
    if (approved_budget.get('authorized') is not True or approved_budget.get('GPU_total_cap_seconds') != 28800
            or approved_budget.get('prior_actual_GPU_seconds') != 6411
            or approved_budget.get('remaining_GPU_seconds') != 22389):
        raise ValueError('Direct human cumulative28800s budget and complete actual prior ledger required')
    allocation_start, native_allocation = actual_allocation_start()
    root = Path(output)
    if root.exists():
        raise ValueError('Refuse overwrite or implicit scientific retry')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU scientific training')
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa'
            or torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name(0)):
        raise ValueError('Actual course allocation / single A100 required')
    seed_all(17, 4)
    root.mkdir(parents=True)
    write_json(root / 'run_contract.json', {'protocol': p, 'source_files': lock['files'], 'approved_budget': approved_budget, 'fast_input_proof': fast_proof, 'phase1_go': False})
    write_json(root / 'actual_gpu.json', {'job': os.environ['SLURM_JOB_ID'], 'account': os.environ['SLURM_JOB_ACCOUNT'],
               'GPU': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'threads': torch.get_num_threads()})
    stop = {'requested': False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.update(requested=True))
    deadline = time.perf_counter() + 22389 - elapsed_from_allocation(allocation_start) - 180
    contexts, warm_cost = {}, {}
    for task in p['tasks']:
        began = time.perf_counter()
        cfg = configs[task]
        store = CachedFeatureStore(cfg, max_cached=1216)
        rows = [x for x in store.manifest['records'] if x['task'] == task and x['split'] in ('train', 'validation')]
        train = paired_rows([x for x in rows if x['split'] == 'train'])
        validation = paired_rows([x for x in rows if x['split'] == 'validation'])
        if (len(train), len(validation)) != (512, 96):
            raise ValueError('Original development sample counts changed')
        basis = fit_basis(cfg, store, task)
        if set(basis['train_ids'].tolist()) != {x['id'] for pair in train for x in pair}:
            raise ValueError('Basis training identities differ')
        store.bases = {task: basis}
        # Exactly the original SHA/finite/normalization/projection code, just
        # executed once before timing steady epochs. No held-out row is read.
        for rec in rows:
            store.features(rec)
            store.projected_features(rec)
        if len(store.lru) != 1216 or len(store.projected) != 1216:
            raise ValueError('Incomplete legal train/validation preload')
        if store.metadata_fingerprint != fast_proof['tasks'][task]['metadata_fingerprint']:
            raise ValueError('Actual legal metadata differs from complete equivalence proof')
        store.prepare_validation(validation, 'cuda')
        boundary_batches = 0
        for entries in store.validation_batches.values():
            for batch, _, _, versions in entries:
                original, current = original_recent_only_teacher_batch(batch), recent_only_teacher_batch(batch)
                if (set(original) != set(current) or any(not torch.equal(original[k], current[k]) for k in original)
                        or any(batch[k]._version != version for k, version in versions.items())):
                    raise ValueError('Actual CUDA causal-boundary transform is not exactly equivalent')
                boundary_batches += 1
        write_json(root / ('actual_recent_boundary_' + task + '.json'),
                   {'exact_original_batches': boundary_batches, 'all_tensors_bitwise_equal': True,
                    'input_versions_unchanged': True, 'no_model_or_gradient_step': True})
        contexts[task] = (cfg, store, train, validation, basis)
        warm_cost[task] = time.perf_counter() - began
        write_json(root / 'preload.json', {'tasks': warm_cost, 'held_out_loaded': False,
                   'original_FeatureStore_unchanged': True, 'elapsed_seconds': time.perf_counter() - started})
    proofs, measured, resources = {}, {}, None
    completed_counts = {task: {seed: 0 for seed in p['seeds']} for task in p['tasks']}
    # First complete epoch of BOTH tasks, then restore each exact checkpoint.
    # These are scientific epochs1/120, not discarded duplicated benchmarks.
    schedule = [(task, 17, 1) for task in p['tasks']]
    schedule += [(task, seed, 120) for task in p['tasks'] for seed in p['seeds']]
    for task, seed, target_epoch in schedule:
        cfg, store, train, validation, basis = contexts[task]
        seed_all(seed, 4)
        reference, reference_sha = reference_model(cfg, p, task, seed, basis)
        frozen = cpu_tree(reference.state_dict())
        model = RelationalMemoryPredictor(reference, basis).cuda()
        sig = signature(control, p, cfg, store, task, seed, reference_sha)
        model_root = root / 'predictor' / task / f'seed_{seed}'
        opt = torch.optim.AdamW(trainable_parameters(model), lr=.0003, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, learning_rate_factor)
        logs, best_score, step, seconds, initial_epoch = [], float('inf'), 0, 0., 0
        resumed_from_original = False
        if model_root.exists():
            ck = torch.load(model_root / 'latest.pt', map_location='cpu', weights_only=False)
            if ck['signature'] != sig or ck['seed'] != seed or ck['task'] != task or ck['epoch'] != 1:
                raise ValueError('Only this job own exact first-epoch checkpoint can be restored')
            model.load_state_dict(ck['model'], strict=True)
            opt.load_state_dict(ck['optimizer'])
            scheduler.load_state_dict(ck['scheduler'])
            logs, best_score, step, seconds, initial_epoch = ck['logs'], ck['best'], ck['step'], ck['seconds'], ck['epoch']
            restore_rng(ck['rng'])
        else:
            model_root.mkdir(parents=True)
            if seed == 17:
                origin = p['resume']
                prior_path = Path(origin['old_training_root']) / 'predictor' / task / 'seed_17' / 'latest.pt'
                if file_hash(prior_path) != origin['partials'][task+'/latest']['sha256']:
                    raise ValueError('Original actual partial checkpoint changed')
                ck = torch.load(prior_path, map_location='cpu', weights_only=False)
                expected = signature(origin['old_control'], read_json(Path(origin['old_control'])/'protocol.json'),
                                     cfg, store, task, seed, reference_sha)
                if ck['signature'] != expected or ck['epoch'] != 1 or ck['step'] != 64:
                    raise ValueError('Original epoch1 provenance/boundary differs')
                if not finite_tree(ck['model']) or not finite_tree(ck['optimizer']):
                    raise ValueError('Original actual partial is nonfinite')
                model.load_state_dict(ck['model'], strict=True)
                opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
                logs, best_score, step, seconds, initial_epoch = ck['logs'], ck['best'], ck['step'], ck['seconds'], 1
                restore_rng(ck['rng']); resumed_from_original = True
        def snapshot(epoch):
            return {'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(opt.state_dict()),
                    'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'epoch': epoch, 'step': step,
                    'best': best_score, 'logs': copy.deepcopy(logs), 'seconds': seconds,
                    'signature': sig, 'seed': seed, 'task': task,
                    'resume_granularity': 'epoch_boundary', 'checkpoint_selection': SELECTION}
        if resumed_from_original:
            # New provenance container, same actual old-source tensors/state/logs.
            save_torch(model_root / 'latest.pt', snapshot(1))
            save_torch(model_root / 'best.pt', snapshot(1))
            write_csv(model_root / 'training.csv', logs)
            write_json(model_root / 'training.json', {'signature':sig,'logs':logs,'seed':seed})
        elif initial_epoch == 0:
            save_torch(model_root / 'latest.pt', snapshot(0))
        elapsed = logs[-1]['epoch_seconds'] if logs else 0.
        completed_counts[task][seed] = initial_epoch
        for epoch in range(initial_epoch, target_epoch):
            boundary = snapshot(epoch)
            began = time.perf_counter()
            result = train_epoch(cfg, store, model, opt, train, seed, epoch, 'cuda', task, stop, deadline)
            if result is None:
                save_torch(model_root / 'latest.pt', boundary)
                status = {'status': 'INTERRUPTED_OR_SOFT_BUDGET_STOP', 'task': task, 'seed': seed,
                          'completed_epochs': epoch, 'phase1_go': False}
                write_json(root / 'outcome.json', status)
                return status
            val = cached_validation_loss(cfg, store, model, validation, 'cuda', task)
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
            completed_counts[task][seed] = epoch + 1
            measured[task] = max(measured.get(task, 0.), elapsed)
            if resources is not None and (epoch + 1) % 20 == 0:
                rolling = full_sequence_projection(
                    new_gpu_seconds_charged=elapsed_from_allocation(allocation_start),
                    candidate_epoch_seconds=measured,
                    reference_epoch_seconds=resources['reference_epoch_seconds'],
                    development_epochs_done=completed_counts,
                    encoder_frames_per_second=resources['encoder_frames_per_second'],
                    confirmation_frame_bound=resources['confirmation_full_frame_bound'],
                    encoder_equivalence_passed=resources['actual_original_encoder_batch2_probe_exact'],
                    fixed_overhead_seconds=900 + 2 * resources['encoder_setup_and_actual_weight_proof_seconds'])
                write_json(root / 'rolling_complete_sequence_budget.json', rolling)
                if not rolling['fits_shared_cap']:
                    status = {'status': 'COMPLETE_SEQUENCE_BUDGET_STOP_WITH_TRUE_PARTIAL',
                              'phase1_go': False, 'task': task, 'seed': seed,
                              'completed_epochs': epoch + 1, 'no_scope_shortening': True}
                    write_json(root / 'outcome.json', status)
                    return status
            print(f'EPOCH {task} seed={seed} {epoch+1}/120 val={score:.8f} seconds={elapsed:.2f}', flush=True)
        if target_epoch == 1:
            measured[task] = elapsed
        else:
            proofs[f'{task}/{seed}'] = verify_completed_model(model_root, sig, frozen)
        store.verify_boundaries()
        del model, opt, reference
        torch.cuda.empty_cache()
        if target_epoch == 1 and len(measured) == 2:
            old_resource_path = Path(p['resume']['old_training_root']) / 'native_resource_probe.json'
            expected_resource_sha = p['resume']['resource_probe_sha256']
            if file_hash(old_resource_path) != expected_resource_sha:
                raise ValueError('Original actual complete-sequence cost proof changed')
            resources = read_json(old_resource_path)
            write_json(root / 'native_resource_probe.json', {**resources,
                       'inherited_actual_cost_measurement_from_job':7524,
                       'original_native_resource_probe_sha256':expected_resource_sha})
            forecast = full_sequence_projection(
                new_gpu_seconds_charged=elapsed_from_allocation(allocation_start),
                candidate_epoch_seconds=measured,
                reference_epoch_seconds=resources['reference_epoch_seconds'],
                development_epochs_done={t: {17: 1, 29: 0, 43: 0} for t in p['tasks']},
                encoder_frames_per_second=resources['encoder_frames_per_second'],
                confirmation_frame_bound=resources['confirmation_full_frame_bound'],
                encoder_equivalence_passed=resources['actual_original_encoder_batch2_probe_exact'],
                fixed_overhead_seconds=900 + 2 * resources['encoder_setup_and_actual_weight_proof_seconds'])
            write_json(root / 'first_epoch_budget.json', {**forecast,
                       'measured_full_epochs': measured, 'preload_seconds': warm_cost,
                       'allocation_start_utc': allocation_start, 'native_allocation': native_allocation,
                       'fresh9137_opened': False, 'method_pass': False})
            if not forecast['fits_shared_cap']:
                status = {'status': 'COMPUTE_BUDGET_STOP', 'phase1_go': False,
                          'no_more_execution_retry_under_this_budget': True}
                write_json(root / 'outcome.json', status)
                return status
    write_json(root / 'all_models_completion.json', {'complete': True, 'models': proofs,
               'phase1_go': False, 'held_out_evaluated': False})
    status = {'status': 'ALL_SIX_MODELS_COMPLETE_EVALUATION_PENDING', 'elapsed_seconds': time.perf_counter() - started,
              'phase1_go': False}
    write_json(root / 'outcome.json', status)
    return status
