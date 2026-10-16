"""Fixed v5 real-reader training on complete, provenance-checked causal caches.

This module never reads action annotations or sealed observations. Validation
selects earliest checkpoints; it never fits a visual basis or normalization.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import random
import signal
import socket
import subprocess
import time

import numpy as np
import torch

from .v5_real_reader import (CausalReader, context_payload, feature_fit,
                           full_feature_prediction, normalize, pad_payloads,
                           policy_contexts)

POLICIES = ('recent', 'FIFO', 'Reservoir', 'full')
HORIZONS = (1, 4, 16, 64)
SEEDS = (17, 29, 43)
QUERY_STEPS = tuple(range(40, 192, 4))
COMPLETE_ENCODING = 'ALL78_FROZEN_CAUSAL_FEATURES_COMPLETE_NOT_PHASE1_GO'


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def payload_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('missing or symlinked cache/source payload')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def validate_cache_report(report, manifest, expected_job_id):
    """Require actual complete encodings; a success label alone cannot pass."""
    if (report.get('status') != COMPLETE_ENCODING or str(report.get('job_id')) != str(expected_job_id)
            or report.get('frames_encoded') != 50688 or report.get('forwards') != 25344
            or report.get('planned_frames') != 50688 or report.get('preflight_passed') is not True
            or report.get('course') != 'csci_ga_2271-2026fa'):
        raise ValueError('incomplete or mismatched scientific encoding')
    if any(report.get(k) is not False for k in ('new_weights_trained', 'predictor_trained',
                                               'task_labels_read', 'sealed_rgb_or_numeric_labels_read')):
        raise ValueError('encoding information boundary not established')
    expected = manifest['development_rgb']
    actual = report['completed_recordings']
    if len(expected) != 78 or len(actual) != 78:
        raise ValueError('development recording count changed')
    paths, counts = set(), {'train': 0, 'validation': 0}
    people = {'train': set(), 'validation': set()}
    for index, (want, got) in enumerate(zip(expected, actual)):
        for key in ('recording', 'participant', 'split'):
            if want[key] != got[key]:
                raise ValueError('recording/split identity changed')
        if want['path'] != got['rgb_path'] or want['sha256'] != got['rgb_sha256']:
            raise ValueError('actual RGB binding changed')
        if got['split'] not in counts or not 1 <= len(got['episodes']) <= 4:
            raise ValueError('sealed or illegal episodes')
        n = len(got['episodes']) * 256
        feature = got['feature']
        if (feature['path'] != f'recording_{index:03d}.npy' or feature['path'] in paths
                or feature['shape'] != [n, 5120] or feature['dtype'] != 'float32'
                or feature['forwards'] != n // 2 or len(feature['sha256']) != 64
                or len(feature['timestamp_sha256']) != 64):
            raise ValueError('partial, duplicate, or altered feature payload')
        paths.add(feature['path'])
        counts[got['split']] += n
        people[got['split']].add(got['participant'])
    if counts != {'train': 34816, 'validation': 15872}:
        raise ValueError('actual train/validation counts differ')
    if len(people['train']) != 29 or len(people['validation']) != 10 or people['train'] & people['validation']:
        raise ValueError('participant-disjoint split changed')
    return actual


def load_verified_features(encoding_root, encoding_control, expected_job_id, device):
    encoding_root, encoding_control = Path(encoding_root), Path(encoding_control)
    if encoding_root.name != 'job_' + str(expected_job_id):
        raise ValueError('current actual cache job binding changed')
    lock = encoding_control / 'source_lock.json'
    check_regular(lock)
    if sha_file(lock) != 'cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e':
        raise ValueError('actual encoding source lock changed')
    for name, expected in json.loads(lock.read_text())['files'].items():
        path = encoding_control / name
        check_regular(path)
        if sha_file(path) != expected:
            raise ValueError('actual encoding source/input bytes changed')
    report_path = encoding_root / 'encoding_report.json'
    check_regular(report_path)
    report = json.loads(report_path.read_text())
    if (report['source_lock_sha256'] != sha_file(lock)
            or report['manifest_sha256'] != sha_file(encoding_control / 'manifest.json')
            or report['pts_report_sha256'] != sha_file(encoding_control / 'pts_report.json')
            or report['protocol_sha256'] != sha_file(encoding_control / 'protocol.json')):
        raise ValueError('cache/source/data/config proof mismatch')
    metadata_path = encoding_root / 'encoder_metadata.json'
    check_regular(metadata_path)
    if sha_file(metadata_path) != report['encoder_metadata_sha256']:
        raise ValueError('actual encoder metadata changed')
    actual = validate_cache_report(report, json.loads((encoding_control / 'manifest.json').read_text()), expected_job_id)
    episodes = []
    proof = {'encoding_job_id': str(expected_job_id), 'encoding_report_sha256': sha_file(report_path),
             'encoding_source_lock_sha256': sha_file(lock), 'actual_feature_files': []}
    for recording in actual:
        f = recording['feature']
        path = encoding_root / 'features' / f['path']
        check_regular(path)
        if path.stat().st_size != f['bytes'] or sha_file(path) != f['sha256']:
            raise ValueError('actual complete features do not match their proof')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != tuple(f['shape']) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('invalid complete frozen features')
        tensor = torch.from_numpy(np.array(array, copy=True)).to(device)
        for ei, descriptor in enumerate(recording['episodes']):
            episodes.append({'recording': recording['recording'], 'participant': recording['participant'],
                             'split': recording['split'], 'episode': ei,
                             'descriptor': descriptor, 'features': tensor[ei*256:(ei+1)*256]})
        proof['actual_feature_files'].append({'path': str(path), 'sha256': f['sha256'], 'rows': len(array)})
    if len(episodes) != 198:
        raise ValueError('actual eligible episode count changed')
    return episodes, proof


def query_rows(episodes, split):
    return tuple((ei, t) for ei, e in enumerate(episodes) if e['split'] == split for t in QUERY_STEPS)


def training_fit(episodes, train_rows, rank=128):
    if any(episodes[ei]['split'] != 'train' for ei, _ in train_rows):
        raise ValueError('nontraining target supplied to fit')
    observed = torch.cat([e['features'] for e in episodes if e['split'] == 'train'])
    residuals = torch.stack([episodes[ei]['features'][t+h] - episodes[ei]['features'][t]
                             for ei, t in train_rows for h in HORIZONS])
    fit = feature_fit(observed, residuals, rank=rank)
    ids = [(e['recording'], e['episode'], i) for e in episodes if e['split'] == 'train' for i in range(256)]
    targets = [(episodes[ei]['recording'], episodes[ei]['episode'], t, h) for ei, t in train_rows for h in HORIZONS]
    proof = {'fit_split': 'train', 'normalization_and_input_rows': len(observed),
             'future_residual_rows': len(residuals), 'input_ids_sha256': payload_digest(ids),
             'future_residual_ids_sha256': payload_digest(targets), 'basis_rank': rank,
             'validation_or_test_fit_rows': 0}
    return fit, proof


def prepared_episodes(episodes, fit):
    prepared = []
    for e in episodes:
        full = normalize(e['features'], fit)
        prepared.append({**e, 'normalized': full, 'compressed': full @ fit['input_basis'],
                         'contexts': {p: policy_contexts(256, p, e['recording'], e['episode']) for p in POLICIES}})
    return prepared


def balanced_rows(permutation, queries, base_batch=16):
    if len(permutation) != len(queries) or sorted(permutation) != list(range(len(queries))):
        raise ValueError('epoch does not cover each original query exactly once')
    if len(queries) % base_batch:
        raise ValueError('registered balanced real training batch no longer divides queries')
    for start in range(0, len(queries), base_batch):
        base = [queries[i] for i in permutation[start:start+base_batch]]
        yield tuple((ei, t, p) for p in POLICIES for ei, t in base)


def make_batch(episodes, rows, training, rng):
    payloads = []
    currents, targets = [], []
    for ei, t, policy in rows:
        e = episodes[ei]
        recent, history = e['contexts'][policy][t]
        if training:
            # Bernoulli dropout affects retained history only. Recent/current
            # timestamps remain available; no label/loss-driven selection.
            history = tuple(i for i in history if rng.random() >= .15)
        payloads.append(context_payload(e['compressed'], t, (recent, history)))
        currents.append(e['normalized'][t])
        targets.append(torch.stack([e['normalized'][t+h] for h in HORIZONS]))
    return (*pad_payloads(payloads), torch.stack(currents), torch.stack(targets))


def full_batch_loss(model, batch, fit):
    x, temporal, padding, currents, targets = batch
    per_horizon = []
    for hi, h in enumerate(HORIZONS):
        coeff = model(x, temporal, padding, x.new_full((len(x),), h))
        predicted = full_feature_prediction(coeff, currents, fit)
        per_horizon.append((predicted - targets[:, hi]).square().mean(dim=1))
    losses = torch.stack(per_horizon, dim=1)
    return losses.mean(), losses


@torch.no_grad()
def validation_scores(model, episodes, queries, fit, deadline):
    model.eval()
    values = {p: np.zeros(4, dtype=np.float64) for p in POLICIES}
    counts = {p: 0 for p in POLICIES}
    for policy in POLICIES:
        for start in range(0, len(queries), 64):
            deadline()
            rows = tuple((ei, t, policy) for ei, t in queries[start:start+64])
            _, losses = full_batch_loss(model, make_batch(episodes, rows, False, None), fit)
            if not torch.isfinite(losses).all():
                raise ValueError('nonfinite validation scores')
            values[policy] += losses.double().sum(0).cpu().numpy()
            counts[policy] += len(rows)
    if any(n != len(queries) for n in counts.values()):
        raise ValueError('incomplete equal-condition validation')
    means = {p: (v / counts[p]).tolist() for p, v in values.items()}
    return float(np.mean(list(means.values()))), means


def schedule_lr(step, total, base=3e-4, warmup_fraction=.05):
    warmup = max(1, math.ceil(total * warmup_fraction))
    if step < warmup:
        return base * (step + 1) / warmup
    return base * .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def atomic_checkpoint(path, state):
    temp = Path(path).with_suffix('.partial.pt')
    torch.save(state, temp)
    temp.replace(path)


def verify_external_cache_audit(protocol, encoding_root, expected_job_id):
    """Require actual completed7187/7188 and source-bound full-cache audit."""
    path=Path(encoding_root)/'full_cache_audit.json'
    check_regular(path)
    expected=protocol.get('actual_cache_audit_sha256')
    if not isinstance(expected,str) or len(expected)!=64 or sha_file(path)!=expected:
        raise ValueError('Actual complete cache audit must be hashed and bound before real training')
    for key, name in [('actual_encoding_report_sha256','encoding_report.json'),
                      ('actual_encoder_metadata_sha256','encoder_metadata.json')]:
        value=protocol.get(key)
        payload=Path(encoding_root)/name
        check_regular(payload)
        if not isinstance(value,str) or len(value)!=64 or sha_file(payload)!=value:
            raise ValueError('Actual cache report/metadata must match the formal binding')
    proof=json.loads(path.read_text())
    if (expected_job_id!=7187 or protocol.get('actual_cache_audit_job_id')!=7188
            or proof.get('audit_job_id')!='7188'
            or proof.get('status')!='ACTUAL_FULL_CACHE_VERIFIED_NOT_READER_TRAINING_NOT_PHASE1_GO'
            or proof.get('complete') is not True or proof.get('encoding_job_id')!=expected_job_id
            or proof.get('source_lock_sha256')!='fc02564863f8af84a79eed2bf0861b210252667c2f70411518a32a9eeb219d00'
            or proof.get('protocol_sha256')!='51c0b25cffb63a9f7ea2e8f3a059c4ab99cc020b976d1a400d6ac167ecf78b26'
            or proof.get('encoding_source_lock_sha256')!='cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e'
            or proof.get('encoding_protocol_sha256')!='c456f06f572df47e66e7863ca752e22214c44846ab9be9934955f31ffa79ac7c'
            or proof.get('encoding_report_sha256')!=protocol.get('actual_encoding_report_sha256')
            or proof.get('encoder_metadata_sha256')!=protocol.get('actual_encoder_metadata_sha256')
            or proof.get('reader_fit_or_training') is not False or proof.get('labels_read') is not False
            or proof.get('sealed_data_read') is not False or proof.get('frames')!=50688
            or proof.get('recordings')!=78 or proof.get('episodes')!=198
            or len(proof.get('actual_feature_files',[]))!=78):
        raise ValueError('Actual cache audit is incomplete or incompatible')
    account=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7187,7188',
        '-n','-P','--format=JobIDRaw,State,ExitCode,Account,Partition,AllocTRES'],text=True)
    rows={line.split('|')[0]:line.split('|') for line in account.splitlines() if line.strip()}
    if (set(rows)!={'7187','7188'}
            or rows['7187'][1:5]!=['COMPLETED','0:0','csci_ga_2271-2026fa','c12m85-a100-1']
            or rows['7188'][1:5]!=['COMPLETED','0:0','csci_ga_2271-2026fa','n2c48m24']):
        raise ValueError('Actual encoding/audit Slurm completion differs')
    tres={part.split('=',1)[0]:part.split('=',1)[1] for part in rows['7187'][5].split(',') if '=' in part}
    if tres.get('gres/gpu')!='1':
        raise ValueError('Actual parent did not contain exactly one GPU')
    audit_tres={part.split('=',1)[0]:part.split('=',1)[1] for part in rows['7188'][5].split(',') if '=' in part}
    if audit_tres.get('gres/gpu','0')!='0':
        raise ValueError('Actual cache audit did not use the declared zeroGPU allocation')
    return proof


def reuse_completed_fit(protocol, episodes, train_rows):
    """Reuse proven training-only coordinates; no new feature-fit trial."""
    import shutil
    import subprocess
    binding=protocol['reuse_fit']
    root=Path(binding['root'])
    old_control=root.parent.parent/'phase1_jepa_memory_v5_real_frame_training'
    for path,want in [(old_control/'source_lock.json',binding['source_lock_sha256']), (old_control/'protocol.json',binding['protocol_sha256'])]:
        check_regular(path)
        if sha_file(path)!=want:raise ValueError('Original fit source binding changed')
    for name,want in json.loads((old_control/'source_lock.json').read_text())['files'].items():
        path=old_control/name;check_regular(path)
        if sha_file(path)!=want:raise ValueError('Original fit source bytes changed')
    if str(root)!='/scratch/xl4358/phase1_v5_real_frame_training/job_7209':
        raise ValueError('Original fit parent differs')
    actual=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7209','-n','-P',
        '--format=JobIDRaw,State,ExitCode,Account,Partition'],text=True).strip().split('|')
    if actual[:5]!=['7209','COMPLETED','0:0','csci_ga_2271-2026fa','c12m85-a100-1']:
        raise ValueError('Actual frozen fit parent incomplete')
    paths={key:root/name for key,name in [('npz','train_only_fit.npz'),('proof','fit_proof.json')]}
    for key,path in paths.items():
        check_regular(path)
        if sha_file(path)!=binding[key+'_sha256']:raise ValueError('Original fit bytes changed')
    proof=json.loads(paths['proof'].read_text())
    ids=[(e['recording'],e['episode'],i) for e in episodes if e['split']=='train' for i in range(256)]
    targets=[(episodes[ei]['recording'],episodes[ei]['episode'],t,h) for ei,t in train_rows for h in HORIZONS]
    if (proof['fit_split']!='train' or proof['normalization_and_input_rows']!=34816
        or proof['future_residual_rows']!=20672 or proof['basis_rank']!=128
        or proof['validation_or_test_fit_rows']!=0 or proof['input_ids_sha256']!=payload_digest(ids)
        or proof['future_residual_ids_sha256']!=payload_digest(targets)):
        raise ValueError('Actual train-only fit row identity changed')
    array=np.load(paths['npz'],allow_pickle=False)
    dimensions={'mean':(5120,),'scale':(5120,),'input_basis':(5120,128),
                'output_mean':(5120,),'output_basis':(5120,128)}
    if set(array.files)!=set(dimensions):raise ValueError('Fit arrays differ')
    fit={}
    for key,shape in dimensions.items():
        value=array[key]
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid exact frozen fit')
        fit[key]=torch.from_numpy(value.copy()).cuda()
    if (fit['scale']<=0).any():raise ValueError('Invalid frozen scale')
    array.close()
    if sha_file(paths['npz'])!=binding['npz_sha256']:raise ValueError('Fit changed while reading')
    return fit,proof


def main(control, encoding_root, encoding_control, expected_job_id, output):
    if (os.getenv('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa'
            or not os.getenv('SLURM_JOB_ID') or not socket.gethostname().startswith('b-')
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or 'A100' not in torch.cuda.get_device_name(0)
            or os.getenv('SLURM_JOB_PARTITION') != 'c12m85-a100-1'):
        raise RuntimeError('real reader requires one allocated course Cloud A100')
    output, control = Path(output), Path(control)
    if any(p.is_symlink() for path in (output, control) for p in (path, *path.parents)):
        raise ValueError('training source/output symlink forbidden')
    if output.exists():
        raise ValueError('fresh training output required; no blind partial recovery')
    os.umask(0o077)
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    if hasattr(os, 'sched_getaffinity'):
        cores = sorted(os.sched_getaffinity(0))
        if len(cores) < 4:
            raise RuntimeError('insufficient allocated CPU cores')
        os.sched_setaffinity(0, cores[:4])
    started = time.monotonic()
    report_path = output / 'training_report.json'
    report = {'status': 'RUNNING_REAL_READER_NOT_PHASE1_GO', 'job_id': os.getenv('SLURM_JOB_ID'),
              'course': os.getenv('SLURM_JOB_ACCOUNT'), 'host': socket.gethostname(),
              'device': torch.cuda.get_device_name(0), 'seeds': list(SEEDS), 'epochs_required': 30,
              'completed_seeds': [], 'labels_or_sealed_observations_read': False,
              'phase1_passed': False, 'first_epoch_projection_passed': False,
              'source_lock_sha256': sha_file(control / 'source_lock.json'),
              'protocol_sha256': sha_file(control / 'protocol.json')}
    protocol = json.loads((control / 'protocol.json').read_text())
    if (protocol['encoding_job_id'] != expected_job_id or protocol['seeds'] != list(SEEDS)
            or protocol['epochs'] != 30 or protocol['horizons'] != list(HORIZONS)
            or protocol['reader'] != {'rank':128,'hidden':256,'layers':2,'heads':4,'ffn':1024,'dropout':.1}
            or protocol['batch_size'] != 64 or protocol['base_queries_per_batch'] != 16
            or protocol['history_dropout'] != .15
            or protocol['optimizer'] != {'name':'AdamW','lr':.0003,'weight_decay':.20,
                                         'warmup_fraction':.05,'schedule':'cosine_to_zero'}
            or protocol['budget'] != {'a100':1,'cpu_threads':4,'hours':4,'soft_stop_seconds':13800,
                                     'preflight':'first_complete_epoch_including_validation_times90_plus_setup'}):
        raise ValueError('fixed real-reader scientific protocol changed')

    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(report_path, report)

    def deadline():
        if time.monotonic() - started >= 13800:
            raise RuntimeError('fixed four-hour real-reader budget exhausted')

    def interrupted(signum, _frame):
        report.update(status='INTERRUPTED_NO_COMPLETE_MODEL_PROOF', signal=signum)
        persist()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    persist()
    try:
        verify_external_cache_audit(protocol, encoding_root, expected_job_id)
        report['external_cache_audit_sha256']=protocol['actual_cache_audit_sha256']
        episodes, cache_proof = load_verified_features(encoding_root, encoding_control, expected_job_id, 'cuda:0')
        train, validation = query_rows(episodes, 'train'), query_rows(episodes, 'validation')
        if len(train) != 5168 or len(validation) != 2356:
            raise ValueError('fixed actual query grid changed')
        fit, fit_proof = reuse_completed_fit(protocol, episodes, train)
        fit_path = output / 'train_only_fit.npz'
        import shutil
        shutil.copyfile(Path(protocol['reuse_fit']['root'])/'train_only_fit.npz',fit_path)
        if sha_file(fit_path)!=protocol['reuse_fit']['npz_sha256']:raise ValueError('Copied fit differs')
        if fit_proof['actual_npz_sha256']!=sha_file(fit_path) or fit_proof['actual_npz_bytes']!=fit_path.stat().st_size:raise ValueError('Exact fit proof mismatch')
        shutil.copyfile(Path(protocol['reuse_fit']['root'])/'fit_proof.json',output/'fit_proof.json')
        atomic_json(output / 'cache_proof.json', cache_proof)
        episodes = prepared_episodes(episodes, fit)
        report.update(train_queries=len(train), validation_queries=len(validation),
                      fit_proof_sha256=sha_file(output / 'fit_proof.json'),
                      cache_proof_sha256=sha_file(output / 'cache_proof.json'))
        signature = payload_digest({k: report[k] for k in ('source_lock_sha256', 'protocol_sha256',
                                                          'fit_proof_sha256', 'cache_proof_sha256')})
        report['training_signature'] = signature
        setup_seconds = time.monotonic() - started
        for seed in SEEDS:
            deadline()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            model = CausalReader().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.20)
            parameters = sum(p.numel() for p in model.parameters())
            report['parameters'] = parameters
            seed_root = output / f'seed_{seed}'
            seed_root.mkdir()
            steps_per_epoch = len(train) // 16
            best, best_epoch, global_step = float('inf'), None, 0
            seed_started = time.monotonic()
            with (seed_root / 'training.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_full_mse', 'validation_full_mse',
                                                           'steps', 'elapsed_seconds', 'validation_slices'])
                writer.writeheader()
                for epoch in range(1, 31):
                    deadline()
                    model.train()
                    train_sum, samples = 0., 0
                    permutation = rng.permutation(len(train)).tolist()
                    for rows in balanced_rows(permutation, train):
                        deadline()
                        optimizer.zero_grad(set_to_none=True)
                        lr = schedule_lr(global_step, steps_per_epoch * 30)
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        loss, losses = full_batch_loss(model, make_batch(episodes, rows, True, rng), fit)
                        if not torch.isfinite(loss):
                            raise ValueError('nonfinite training loss')
                        loss.backward()
                        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise ValueError('nonfinite training gradients')
                        optimizer.step()
                        train_sum += float(losses.detach().double().sum())
                        samples += len(rows)
                        global_step += 1
                    if samples != len(train) * 4 or global_step != epoch * steps_per_epoch:
                        raise ValueError('unbalanced or incomplete epoch')
                    valid, slices = validation_scores(model, episodes, validation, fit, deadline)
                    if valid < best:
                        best, best_epoch = valid, epoch
                        atomic_checkpoint(seed_root / 'best.pt', {'model': model.state_dict(), 'seed': seed,
                                          'epoch': epoch, 'validation': valid, 'training_signature': signature})
                    boundary = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'seed': seed,
                                'epoch': epoch, 'global_step': global_step, 'best_epoch': best_epoch,
                                'best_validation': best, 'training_signature': signature,
                                'python_rng': random.getstate(), 'numpy_global_rng': np.random.get_state(),
                                'retention_dropout_shuffle_rng': rng.bit_generator.state,
                                'torch_cpu_rng': torch.get_rng_state(), 'torch_cuda_rng': torch.cuda.get_rng_state_all()}
                    atomic_checkpoint(seed_root / 'latest.pt', boundary)
                    writer.writerow({'epoch': epoch, 'train_full_mse': train_sum / (samples * 4),
                                     'validation_full_mse': valid, 'steps': steps_per_epoch,
                                     'elapsed_seconds': time.monotonic() - seed_started,
                                     'validation_slices': json.dumps(slices, sort_keys=True)})
                    stream.flush()
                    report.update(current_seed=seed, current_epoch=epoch, global_step=global_step)
                    if seed == SEEDS[0] and epoch == 1:
                        projection = setup_seconds + (time.monotonic() - seed_started) * 90
                        report['first_epoch_projected_total_seconds'] = projection
                        if projection > 13800:
                            report['status'] = 'STOPPED_FIXED_FIRST_EPOCH_PROJECTION_EXCEEDS_BUDGET'
                            persist()
                            return 2
                        report['first_epoch_projection_passed'] = True
                    persist()
                    print(json.dumps({'seed': seed, 'epoch': epoch, 'validation_full_mse': valid}), flush=True)
            # Reload both actual files and their original boundary/RNG bindings
            # before marking any seed complete. No held-out evaluation here.
            best_state = torch.load(seed_root / 'best.pt', map_location='cpu', weights_only=False)
            last_state = torch.load(seed_root / 'latest.pt', map_location='cpu', weights_only=False)
            if (last_state['epoch'] != 30 or last_state['global_step'] != 30 * steps_per_epoch
                    or best_state['epoch'] != best_epoch or best_state['seed'] != seed
                    or any(c['training_signature'] != signature for c in (best_state, last_state))
                    or not all(torch.isfinite(t).all() for c in (best_state, last_state) for t in c['model'].values())
                    or not all(k in last_state for k in ('optimizer', 'python_rng', 'numpy_global_rng',
                                      'retention_dropout_shuffle_rng', 'torch_cpu_rng', 'torch_cuda_rng'))):
                raise ValueError('actual current checkpoint/boundary proof incomplete')
            proof = {'seed': seed, 'epochs': 30, 'steps': global_step, 'best_epoch': best_epoch,
                     'training_signature': signature, 'parameters': parameters,
                     'condition_samples_per_epoch': {p: len(train) for p in POLICIES},
                     'horizons_per_query': list(HORIZONS), 'actual_current_weights_finite': True,
                     'effective_train_query_horizon_examples':len(train)*4*4*30,
                     'validation_query_horizon_examples':len(validation)*4*4*30,
                     'actual_boundary_rng_optimizer_present': True,
                     'files': {n: {'sha256': sha_file(seed_root / n), 'bytes': (seed_root / n).stat().st_size}
                               for n in ('best.pt', 'latest.pt', 'training.csv')}}
            atomic_json(seed_root / 'checkpoint_proof.json', proof)
            report['completed_seeds'].append(proof)
            persist()
        for f in cache_proof['actual_feature_files']:
            if sha_file(f['path']) != f['sha256']:
                raise ValueError('encoded cache bytes changed during reader training')
        report['status'] = 'ALL3_REAL_READERS_30EPOCHS_CURRENT_WEIGHTS_PROVEN_NOT_PHASE1_GO'
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        persist()
        return 0
    except Exception as error:
        report.update(status='FAILED_NO_COMPLETE_MODEL_PROOF', error_type=type(error).__name__, error=str(error)[:300])
        persist()
        raise
