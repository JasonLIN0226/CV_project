"""One frozen, paired reader-aggregation diagnostic; never opens original CONF.

Reuse the parent TRAIN/VAL arrays and basis read-only. Fit probes on TRAIN/VAL,
train all five readers, bind their checkpoints, then score old DEV and a newly
generated diagnostic seed. No retention students or baseline winner selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/predictor/control/src')]
from experiments.phase2 import run_terminal_structures as legacy
from experiments.phase2.reader_aggregation import build_reader, describe_reader, VARIANTS

read, write, sha, emit = legacy.read, legacy.write, legacy.sha, legacy.emit


def source_hashes():
    names = list(legacy.source_files()) + [
        'experiments/phase2/run_reader_aggregation.py',
        'experiments/phase2/reader_aggregation.py',
        'experiments/phase2/probe_reader_features.py',
        'experiments/phase2/submit_reader_aggregation.py',
        'docs/READER_AGGREGATION_PREREGISTRATION_20261025.md']
    return {name: sha(ROOT/name) for name in names}


def verify_parent(c, source):
    if sha(source/'protocol.json') != c['source_run_protocol_sha256']:
        raise ValueError('Parent protocol binding failed')
    if legacy.source_files() != c['parent_source_files_sha256']:
        raise ValueError('Parent scientific source changed')
    if (source/'confirmation_opened.json').exists():
        raise ValueError('This pilot requires the original confirmation to remain closed')
    archive = source.parent/'terminal-structures-results.zip'
    expected = c['parent_archive_sha256']
    if sha(archive) != expected:
        raise ValueError('Parent released archive binding failed')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('package_manifest.json'))
    entries = manifest['files'] | manifest['derived_data_preserved_on_scratch']
    names = ['basis.npz', 'protocol.json', 'reader_selected.pt', 'data/records.json',
             'data/features.npy', 'data/features_raw.npy', 'data/normalization.npz',
             'data/actions.npz', 'data/action_banks.npz', 'data/rgb_index.json']
    checks = {}
    for name in names:
        expected = entries['run/'+name]
        p = source/name
        actual = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        if actual != expected:
            raise ValueError('Parent cache binding failed: '+name)
        checks[name] = actual
    return {'source': str(source), 'archive_sha256': c['parent_archive_sha256'],
            'verified_files': checks, 'readonly': True}


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
    from phase1.inputs import collate
    from phase1.utils import seed_all
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


class Predictor(legacy.Predictor):
    def __init__(self, c, base_c, source, out, variant, device):
        import torch
        self.torch, self.device, self.c = torch, device, base_c
        self.basis = dict(np.load(source/'basis.npz', allow_pickle=False))
        checkpoint_variant = 'original' if variant == 'uniform_intervention' else variant
        path = out/'readers'/checkpoint_variant/'reader_selected.pt'
        lock = read(out/'readers'/checkpoint_variant/'reader_locked.json')
        if sha(path) != lock['checkpoint_sha256'] or sha(source/'basis.npz') != lock['basis_sha256']:
            raise ValueError('Frozen reader binding changed')
        state = torch.load(path, map_location=device, weights_only=False)
        self.model = build_reader(len(self.basis['input_mean']), base_c['reader'], legacy.H,
                                  self.basis, variant, heads=c['heads']).to(device)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.eval().requires_grad_(False)
        self.calls, self.examples, self.seconds, self.reconstruction_max, self.attention_queries = 0, 0, 0., 0., 0


def verify_prefit_lock(c, source, out):
    lock = read(out/'all_models_locked.json')
    if lock['source_sha256'] != source_hashes():
        raise ValueError('Source changed after all-model lock')
    if sha(out/'protocol.json') != lock['protocol_sha256']:
        raise ValueError('Protocol changed after all-model lock')
    protocol = read(out/'protocol.json')
    runtime_config = {key: value for key, value in c.items() if key != 'runtime_device'}
    if runtime_config != protocol['config']:
        raise ValueError('Runtime config changed after protocol freeze')
    if (lock['config_file_sha256'] != protocol['config_file_sha256']
            or sha(Path(lock['config_file_path'])) != lock['config_file_sha256']):
        raise ValueError('Config file changed after protocol freeze')
    if sha(out/'parent_binding.json') != protocol['parent_binding_sha256']:
        raise ValueError('Parent binding changed after protocol freeze')
    if sha(out/'shared_initial_state.pt') != lock['shared_initial_state_sha256']:
        raise ValueError('Shared initial state changed after all-model lock')
    if set(lock['readers']) != set(c['variants']):
        raise ValueError('Missing preregistered reader')
    for name, entry in lock['readers'].items():
        if sha(out/entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Reader changed: '+name)
    for name, digest in lock['probe_files'].items():
        if sha(out/'probes'/name) != digest:
            raise ValueError('Probe changed: '+name)
    if sha(source/'basis.npz') != lock['basis_sha256']:
        raise ValueError('Basis changed after fitting')
    return lock


def validate_diagnostic_seed(c, base_c):
    collisions = [name for name, split in base_c['splits'].items()
                  if split['seed'] == c['diagnostic']['seed']]
    if collisions:
        raise ValueError('Fresh diagnostic seed overlaps parent splits: '+', '.join(collisions))


def measure_latency(predictor, data, records):
    """Fixed input-order timing; excludes encoding, targets and qualification work."""
    import torch
    report = {}
    for size in (2, 3):
        requests = [legacy.reader_sample(data.capsule(r), list(range(size)),
                    data.actual_plan(r), data.basis) for r in records[:16]]
        if not requests:
            raise ValueError('No records for reader latency measurement')
        predictor.predict(requests)
        elapsed = []
        for _ in range(5):
            if str(predictor.device).startswith('cuda'): torch.cuda.synchronize()
            start = time.perf_counter()
            predictor.predict(requests)
            if str(predictor.device).startswith('cuda'): torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-start)
        report['K'+str(size)] = {'batch_parents': len(requests), 'repeats': 5,
            'warmup_batches': 1, 'batch_seconds': elapsed,
            'median_seconds_per_parent': float(np.median(elapsed)/len(requests)),
            'forecast_horizons_per_parent': len(legacy.H),
            'scope': 'reader predict including collation and output reconstruction; excludes encoder',
            'sample_rule': 'first16_record_order_fixed_slot_indices_no_target_selection'}
    return report


def evaluate_readers(c, base_c, source, out, data_path, label, record_split):
    results = {}
    for variant in c['variants']+['uniform_intervention']:
        verify_prefit_lock(c, source, out)
        dest = out/'evaluation'/label/variant
        dest.mkdir(parents=True, exist_ok=False)
        predictor = Predictor(c, base_c, source, out, variant, c['runtime_device'])
        data = legacy.Data(data_path, predictor.basis)
        records = [r for r in data.records if r['split'] == record_split]
        started = time.perf_counter()
        result = legacy.qualify(base_c, data, predictor, records, dest)
        results[variant] = result
        write(dest/'execution.json', {'seconds': time.perf_counter()-started, 'reader_examples': predictor.examples,
              'reader_batches': predictor.calls, 'reader_inference_seconds': predictor.seconds,
              'coefficient_reconstruction_max_abs_difference': predictor.reconstruction_max,
              'checkpoint_variant': 'original' if variant == 'uniform_intervention' else variant,
              'no_training_intervention': variant == 'uniform_intervention',
              'scope': label, 'record_split': record_split, 'family_count': len({r['family'] for r in records})})
        write(dest/'latency.json', measure_latency(predictor, data, records))
        # Save causal per-item pooling weights for auditing, without claiming causal attribution.
        from experiments.phase2.reader_aggregation import reader_attention
        import torch
        weights = []
        for r in records:
            cap = data.capsule(r)
            with torch.inference_mode():
                w = reader_attention(predictor.model,
                    torch.from_numpy(data.project(cap['z'][4:])[None]).to(c['runtime_device']),
                    torch.from_numpy(data.project(cap['current'])[None]).to(c['runtime_device']),
                    torch.from_numpy(cap['age'][4:][None]).to(c['runtime_device']),
                    torch.from_numpy(cap['past_actions'][4:][None]).to(c['runtime_device'])).cpu().numpy()[0]
            weights.append({'id': r['id'], 'weights': w.tolist()})
        write(dest/'pooling_weights.json', {'meaning': 'mean over pooling paths; after item mixing this is not original-item causal attribution', 'rows': weights})
    return results


def summarize(c, out):
    result = {'scope': 'paired single-seed reader diagnostic, not method confirmation', 'training_seeds': [c['seed']],
              'retention_students_trained': False, 'original_confirmation_opened': False, 'splits': {}}
    for label in ('existing_development', 'fresh_diagnostic'):
        variants = {}; family_values = {}
        for variant in c['variants']+['uniform_intervention']:
            dest = out/'evaluation'/label/variant
            q = read(dest/'reader_qualification.json')
            variants[variant] = {k: q[k] for k in ('summary', 'by_condition', 'gates', 'all_task_k2_qualified', 'all_task_k3_qualified')}
            rows = read(dest/'reader_qualification_rows.json')
            groups = defaultdict(list)
            for r in rows:
                if r['arm'] == 'full': kind = 'K3'
                elif r['arm'].startswith('native_') and r['semantic_complete']: kind = 'native_K2'
                else: continue
                groups[(r['task_family'], r['condition'], kind, r['family'])].append(r)
            family_values[variant] = {'|'.join(key): {'accuracy': float(np.mean([r['strict_correct'] for r in rr])),
                                      'mse': float(np.mean([r['mse'] for r in rr]))} for key, rr in groups.items()}
        paired = {}
        for variant in c['variants'][1:]+['uniform_intervention']:
            groups = defaultdict(list)
            for key, value in family_values[variant].items():
                baseline = family_values['original'][key]
                groups[key.rsplit('|', 1)[0]].append([value['accuracy']-baseline['accuracy'], value['mse']-baseline['mse']])
            paired[variant] = {}
            for key, values in sorted(groups.items()):
                x = np.asarray(values)
                rng = np.random.default_rng(c['bootstrap']['seed'])
                boot = x[rng.integers(len(x), size=(c['bootstrap']['resamples'], len(x)))].mean(1)
                paired[variant][key] = {'families': len(x), 'accuracy_difference': float(x[:, 0].mean()),
                    'mse_difference': float(x[:, 1].mean()), 'accuracy_descriptive95': np.quantile(boot[:, 0], [.025, .975]).tolist(),
                    'mse_descriptive95': np.quantile(boot[:, 1], [.025, .975]).tolist()}
        result['splits'][label] = {'readers': variants, 'paired_minus_original': paired,
            'interval_scope': 'unadjusted descriptive family bootstrap; one training seed; no winner selection'}
        write(out/'evaluation'/label/'family_metrics.json', family_values)
    write(out/'summary.json', result)
    return result


def run(config, source, out, device, cache):
    import torch
    from experiments.phase2 import probe_reader_features as probes
    c = read(config)
    if c['status'] != 'ready-for-source-freeze-not-yet-executed' or c['must_resolve_before_freeze']:
        raise ValueError('Protocol is not ready for freezing')
    if tuple(c['variants']) != VARIANTS or c['students_enabled'] or c['original_confirm_open_enabled']:
        raise ValueError('Diagnostic scope guard')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Study output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    parent = verify_parent(c, source)
    base_c = read(source/'protocol.json')['config']
    validate_diagnostic_seed(c, base_c)
    for key in ('steps', 'validate_every', 'groups_per_task', 'learning_rate', 'weight_decay', 'reader', 'history_sizes', 'horizons'):
        if c[key] != base_c[key]:
            raise ValueError('Unapproved data/loss/training change: '+key)
    write(out/'parent_binding.json', parent)
    write(out/'protocol.json', {'config': c, 'config_file_sha256': sha(config), 'source_sha256': source_hashes(),
          'parent_binding_sha256': sha(out/'parent_binding.json'), 'frozen_unix': time.time(),
          'original_confirmation_opened': False, 'fresh_diagnostic_generated': False})
    c['runtime_device'] = device
    status = {'status': 'running', 'stages': [], 'original_confirmation_opened': False, 'retention_students_trained': False}
    began = time.perf_counter()
    def stage(name, fn):
        started = time.perf_counter(); emit('aggregation_stage', name=name)
        value = fn(); status['stages'].append({'stage': name, 'seconds': time.perf_counter()-started})
        write(out/'study_status.json', status)
        return value
    try:
        stage('probe_fit_train_validation', lambda: probes.fit(source, out/'probes', c))
        basis = dict(np.load(source/'basis.npz', allow_pickle=False))
        torch.manual_seed(c['seed'])
        common = build_reader(len(basis['input_mean']), base_c['reader'], legacy.H, basis, 'original', c['heads']).state_dict()
        torch.save(common, out/'shared_initial_state.pt')
        readers = {}
        for variant in c['variants']:
            readers[variant] = stage('train_'+variant, lambda v=variant: train_one(c, base_c, source, out, v, device, common))
        if len({x['training_schedule_sha256'] for x in readers.values()}) != 1:
            raise ValueError('Paired reader training schedules differ')
        probe_files = {str(p.relative_to(out/'probes')): sha(p) for p in sorted((out/'probes').rglob('*')) if p.is_file()}
        write(out/'all_models_locked.json', {'readers': readers, 'probe_files': probe_files,
              'basis_sha256': sha(source/'basis.npz'), 'shared_initial_state_sha256': sha(out/'shared_initial_state.pt'),
              'protocol_sha256': sha(out/'protocol.json'), 'config_file_path': str(config.resolve()),
              'config_file_sha256': sha(config),
              'source_sha256': source_hashes(), 'locked_unix': time.time(), 'fresh_diagnostic_generated': False,
              'original_confirmation_opened': False})
        stage('qualify_existing_development', lambda: evaluate_readers(c, base_c, source, out, source/'data', 'existing_development', 'development'))
        verify_prefit_lock(c, source, out)
        checked_parent = stage('reverify_parent_before_diagnostic', lambda: verify_parent(c, source))
        if checked_parent != parent:
            raise ValueError('Parent cache binding changed before fresh diagnostic')
        if (out/'diagnostic_opened.json').exists():
            raise ValueError('Fresh diagnostic already opened')
        diagnostic_c = copy.deepcopy(base_c)
        diagnostic_c['splits']['development'] = {'seed': c['diagnostic']['seed'], 'families': c['diagnostic']['families']}
        write(out/'diagnostic_opened.json', {'seed': c['diagnostic']['seed'], 'all_models_lock_sha256': sha(out/'all_models_locked.json'),
              'original_confirmation': False, 'generator_split': 'development', 'analysis_role': 'fresh_diagnostic', 'once_only': True})
        banks = dict(np.load(source/'data/action_banks.npz', allow_pickle=False))
        stage('generate_fresh_diagnostic', lambda: legacy.generate(diagnostic_c, out/'diagnostic_data', ['development'], existing_banks=banks))
        old_families = {r['family'] for r in read(source/'data/records.json')}
        new_families = {r['family'] for r in read(out/'diagnostic_data/records.json')}
        if old_families & new_families:
            raise ValueError('Family leakage into fresh diagnostic')
        stage('encode_fresh_diagnostic', lambda: legacy.encode(diagnostic_c, out/'diagnostic_data', cache, device, source/'data/normalization.npz'))
        stage('qualify_fresh_diagnostic', lambda: evaluate_readers(c, base_c, source, out, out/'diagnostic_data', 'fresh_diagnostic', 'development'))
        stage('probe_existing_development', lambda: probes.evaluate(out/'probes', source/'data', 'development', out/'probe_evaluation/existing_development'))
        stage('probe_fresh_diagnostic', lambda: probes.evaluate(out/'probes', out/'diagnostic_data', 'development', out/'probe_evaluation/fresh_diagnostic'))
        stage('summarize', lambda: summarize(c, out))
        verify_prefit_lock(c, source, out)
        status['status'] = 'completed_locked_reader_diagnostic_no_students_no_original_confirmation'
    except Exception as exc:
        status.update(status='failed', error=repr(exc)); raise
    finally:
        status['seconds'] = time.perf_counter()-began
        write(out/'study_status.json', status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--cache-dir', type=Path, required=True)
    a = p.parse_args()
    run(a.config.resolve(), a.source.resolve(), a.output.resolve(), a.device, a.cache_dir.resolve())


if __name__ == '__main__':
    main()
