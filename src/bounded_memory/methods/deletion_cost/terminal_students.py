"""Matched independent / DeepSets students for the terminal-memory experiment.

This module is a training component, not an experiment entry point by default.
Only `items[N,3,D]` and `context[N,C]` enter a scorer. Exports supply already
causal, TRAIN-projected inputs. Parent IDs are used solely to construct stable
shuffled supervision; labels, IDs and task metadata never enter forward().

Each architecture uses the same initialization seed and batch sequence across
cost, future-attention and shuffled-cost labels. Different architecture shapes
do not imply identical initial weights. Checkpoint selection uses each arm's
own VAL soft-label cross entropy, never actual deletion cost for another arm.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import torch
from torch import nn


INPUT_FIELDS = frozenset(('items', 'context'))
LABELS = ('cost', 'future_attention', 'shuffle')
ARCHITECTURES = ('independent', 'set')


@dataclass(frozen=True)
class StudentConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    steps: int = 1000
    batch_size: int = 128
    validate_every: int = 100
    validation_batch_size: int = 512
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    independent_width: int = 64
    shuffle_seed: int = 20261108
    batch_seed_offset: int = 300000
    standard_deviation_floor: float = 1e-4
    label_scale_floor: float = 1e-8
    cpu_threads: int = 4
    checkpoint_rule: str = 'own_validation_label_cross_entropy_strict_min_earliest_tie_including_step0'


@dataclass
class StudentExport:
    inputs: dict[str, np.ndarray]
    cost: np.ndarray
    attention: np.ndarray
    ids: list[str]
    appearance_inputs: dict[str, np.ndarray] | None = None
    sampling_weights: np.ndarray | None = None


def validate_inputs(inputs: Mapping[str, np.ndarray], dimensions=None):
    if set(inputs) != INPUT_FIELDS:
        raise ValueError('Scorer accepts exactly items and context; metadata/future fields are forbidden')
    x = {k: np.asarray(inputs[k], np.float32) for k in INPUT_FIELDS}
    if x['items'].ndim != 3 or x['items'].shape[1] != 3 or x['items'].shape[0] == 0:
        raise ValueError('Require nonempty items[N,3,D]')
    if x['context'].ndim != 2 or len(x['context']) != len(x['items']):
        raise ValueError('Require context[N,C] with matching row count')
    if x['items'].shape[2] == 0 or not all(np.isfinite(a).all() for a in x.values()):
        raise ValueError('Scorer inputs must be finite with nonempty item dimension')
    if dimensions is not None and (x['items'].shape[2], x['context'].shape[1]) != tuple(dimensions):
        raise ValueError('Scorer input dimensions differ from frozen training export')
    return {k: np.ascontiguousarray(a) for k, a in x.items()}


def load_export(directory):
    """Load tensor-only scorer inputs and separate evaluator supervision."""
    directory = Path(directory)
    with np.load(directory/'inputs.npz', allow_pickle=False) as f:
        inputs = validate_inputs({k: f[k] for k in f.files})
    with np.load(directory/'labels.npz', allow_pickle=False) as f:
        attention_key = 'attention' if 'attention' in f.files else 'future_attention'
        cost, attention = np.asarray(f['cost'], np.float64), np.asarray(f[attention_key], np.float64)
    ids = json.loads((directory/'ids.json').read_text())
    if isinstance(ids, dict):
        ids = ids['ids']
    ids = [x['id'] if isinstance(x, dict) else x for x in ids]
    appearance = None
    if (directory/'appearance.npz').exists():
        with np.load(directory/'appearance.npz', allow_pickle=False) as f:
            appearance = validate_inputs({k: f[k] for k in f.files})
    weights = np.load(directory/'sampling_weights.npy', allow_pickle=False) if (directory/'sampling_weights.npy').exists() else None
    result = StudentExport(inputs, cost, attention, ids, appearance, weights)
    validate_export(result)
    return result


def validate_export(export):
    inputs = validate_inputs(export.inputs)
    n = len(inputs['items'])
    for name in ('cost', 'attention'):
        value = np.asarray(getattr(export, name))
        if value.shape != (n, 3) or not np.isfinite(value).all():
            raise ValueError(f'{name} labels must be finite [N,3]')
    if len(export.ids) != n or len(set(export.ids)) != n or not all(isinstance(x, str) for x in export.ids):
        raise ValueError('Require one unique string parent ID per row, outside model inputs')
    if export.appearance_inputs is not None:
        app = validate_inputs(export.appearance_inputs)
        if len(app['items']) != n or app['context'].shape[1] != 0:
            raise ValueError('Appearance-only control requires aligned item statistics and no context')
    if export.sampling_weights is not None:
        w = np.asarray(export.sampling_weights, np.float64)
        if w.shape != (n,) or not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError('Sampling weights must be finite positive [N], not scorer features')
    return inputs


def fit_normalization(train_inputs, floor=1e-4):
    x = validate_inputs(train_inputs)
    result = {}
    for name, axis in (('items', (0, 1)), ('context', 0)):
        values = x[name].astype(np.float64)
        result[name+'_mean'] = values.mean(axis=axis).astype(np.float32)
        result[name+'_std'] = np.maximum(values.std(axis=axis), floor).astype(np.float32)
    return result


def normalize_inputs(inputs, normalization):
    dimensions = (len(normalization['items_mean']), len(normalization['context_mean']))
    x = validate_inputs(inputs, dimensions)
    return {k: ((x[k]-normalization[k+'_mean'])/normalization[k+'_std']).astype(np.float32) for k in INPUT_FIELDS}


def parameter_formula(architecture, input_dimension, width):
    return input_dimension*width+(1 if architecture == 'independent' else 2)*width*width+3*width+1


def architecture_plan(item_dimension, context_dimension, independent_width=64):
    """Resolve width by parameter arithmetic before looking at any labels."""
    dimension = item_dimension+context_dimension
    target = parameter_formula('independent', dimension, independent_width)
    candidates = range(1, independent_width+1)
    set_width = min(candidates, key=lambda w: (abs(parameter_formula('set', dimension, w)-target), w))
    set_count = parameter_formula('set', dimension, set_width)
    relative_gap = abs(set_count-target)/target
    if relative_gap > .05:
        raise ValueError('Cannot match actual participating parameter counts within5%')
    return {'item_dimension': item_dimension, 'context_dimension': context_dimension,
            'independent': {'width': independent_width, 'parameters': target},
            'set': {'width': set_width, 'parameters': set_count}, 'relative_parameter_gap': relative_gap}


class TerminalScorer(nn.Module):
    def __init__(self, item_dimension, context_dimension, architecture, width):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError('Unknown scorer architecture')
        self.item_dimension, self.context_dimension = item_dimension, context_dimension
        self.architecture, self.width = architecture, width
        dimension = item_dimension+context_dimension
        self.phi = nn.Sequential(nn.Linear(dimension, width), nn.ReLU())
        head_input = width if architecture == 'independent' else 2*width
        self.head = nn.Sequential(nn.Linear(head_input, width), nn.ReLU(), nn.Linear(width, 1))
        actual = sum(p.numel() for p in self.parameters())
        assert actual == parameter_formula(architecture, dimension, width)

    def forward(self, items, context):
        if items.ndim != 3 or items.shape[1:] != (3, self.item_dimension):
            raise ValueError('Item tensor shape mismatch')
        if context.shape != (items.shape[0], self.context_dimension):
            raise ValueError('Common context shape mismatch')
        local = torch.cat((items, context[:, None, :].expand(-1, 3, -1)), dim=-1)
        hidden = self.phi(local)
        if self.architecture == 'set':
            pooled = hidden.mean(1, keepdim=True).expand(-1, 3, -1)
            hidden = torch.cat((hidden, pooled), dim=-1)
        return self.head(hidden).squeeze(-1)


def training_label_scales(cost, attention, floor=1e-8):
    result = {}
    for name, values in (('cost', cost), ('future_attention', attention)):
        v = np.asarray(values, np.float64)
        centered = v-v.mean(1, keepdims=True)
        result[name] = max(float(np.sqrt(np.mean(centered**2))), floor)
    result['shuffle'] = result['cost']
    return result


def stable_parent_permutations(ids, seed):
    # IDs are used in this offline label construction only, never as features.
    rows = []
    for parent in ids:
        digest = hashlib.sha256(json.dumps([int(seed), parent], separators=(',', ':')).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        rows.append(rng.permutation(3))
    return np.stack(rows)


def soft_targets(cost, attention, ids, scales, shuffle_seed):
    permutations = stable_parent_permutations(ids, shuffle_seed)
    label = {'cost': np.asarray(cost, np.float64), 'future_attention': np.asarray(attention, np.float64)}
    label['shuffle'] = np.take_along_axis(label['cost'], permutations, axis=1)
    targets = {}
    for name, values in label.items():
        # Small cost and small salience both favor deleting the corresponding item.
        desirability = -(values-values.mean(1, keepdims=True))/scales[name]
        desirability -= desirability.max(1, keepdims=True)
        exp = np.exp(desirability)
        targets[name] = (exp/exp.sum(1, keepdims=True)).astype(np.float32)
    return targets, permutations


def soft_cross_entropy(logits, targets):
    return -(targets*torch.log_softmax(logits, dim=-1)).sum(-1)


def _tensor_inputs(inputs):
    return {k: torch.from_numpy(v) for k, v in inputs.items()}


def _validation(model, inputs, target, report_cost, batch_size, weights=None):
    """Own-label CE is returned separately from the report-only true cost."""
    model.eval()
    ce_sum, chosen_cost_sum, weight_sum = 0., 0., 0.
    with torch.inference_mode():
        for at in range(0, len(target), batch_size):
            end = min(at+batch_size, len(target))
            logits = model(inputs['items'][at:end], inputs['context'][at:end])
            w = np.ones(end-at, np.float64) if weights is None else np.asarray(weights[at:end], np.float64)
            ce_sum += float(np.dot(soft_cross_entropy(logits, target[at:end]).numpy(), w))
            selected = logits.argmax(1).numpy()
            chosen_cost_sum += float(np.dot(report_cost[np.arange(at, end), selected], w))
            weight_sum += float(w.sum())
    return {'own_label_ce': ce_sum/weight_sum, 'report_only_true_cost': chosen_cost_sum/weight_sum}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def _array_hash(value):
    x = np.ascontiguousarray(value)
    h = hashlib.sha256(str((x.dtype.str, x.shape)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def fit_students(train, validation, output_dir, config=StudentConfig()):
    """Train all predeclared arms; caller must freeze its experiment first."""
    train_inputs, val_inputs = validate_export(train), validate_export(validation)
    if set(train.ids) & set(validation.ids):
        raise ValueError('TRAIN and VAL parent IDs overlap')
    if config.steps <= 0 or config.steps % config.validate_every or config.batch_size <= 0:
        raise ValueError('Require positive fixed steps divisible by validation interval')
    if config.checkpoint_rule != StudentConfig.checkpoint_rule:
        raise ValueError('Checkpoint selection must use own-label VAL CE, earliest strict minimum')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Require a new empty student output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    dimensions = (train_inputs['items'].shape[2], train_inputs['context'].shape[1])
    validate_inputs(val_inputs, dimensions)
    plan = architecture_plan(*dimensions, config.independent_width)
    normalization = fit_normalization(train_inputs, config.standard_deviation_floor)
    np.savez(output/'normalization.npz', **normalization)
    tensors = {'train': _tensor_inputs(normalize_inputs(train_inputs, normalization)),
               'validation': _tensor_inputs(normalize_inputs(val_inputs, normalization))}
    scales = training_label_scales(train.cost, train.attention, config.label_scale_floor)
    train_targets, train_perm = soft_targets(train.cost, train.attention, train.ids, scales, config.shuffle_seed)
    val_targets, val_perm = soft_targets(validation.cost, validation.attention, validation.ids, scales, config.shuffle_seed)
    np.savez(output/'supervision_audit.npz', train_cost=train_targets['cost'], train_attention=train_targets['future_attention'],
             train_shuffle=train_targets['shuffle'], val_cost=val_targets['cost'], val_attention=val_targets['future_attention'],
             val_shuffle=val_targets['shuffle'], train_permutations=train_perm, val_permutations=val_perm)
    _write(output/'supervision_ids.json', {'train': train.ids, 'validation': validation.ids,
                                         'usage': 'offline stable label permutations only; not scorer features'})
    appearance = train.appearance_inputs is not None or validation.appearance_inputs is not None
    if appearance and (train.appearance_inputs is None or validation.appearance_inputs is None):
        raise ValueError('Appearance control must supply both TRAIN and VAL')
    appearance_plan = None
    if appearance:
        adim = train.appearance_inputs['items'].shape[2]
        validate_inputs(validation.appearance_inputs, (adim, 0))
        appearance_norm = fit_normalization(train.appearance_inputs, config.standard_deviation_floor)
        np.savez(output/'appearance_normalization.npz', **appearance_norm)
        tensors['appearance_train'] = _tensor_inputs(normalize_inputs(train.appearance_inputs, appearance_norm))
        tensors['appearance_validation'] = _tensor_inputs(normalize_inputs(validation.appearance_inputs, appearance_norm))
        appearance_plan = {'item_dimension': adim, 'context_dimension': 0,
                           'width': config.independent_width,
                           'parameters': parameter_formula('independent', adim, config.independent_width),
                           'scope': 'Input-restricted independent cost control, not parameter-matched to main arms'}
    prefit = {'status': 'locked_before_student_initialization_or_fit', 'config': asdict(config), 'architecture_plan': plan,
              'appearance_plan': appearance_plan, 'training_label_scales': scales,
              'initialization': 'same RNG seed across labels within an architecture; different architecture shapes have different weights',
              'checkpoint_selection': 'own VAL label CE only, strict improvement, earliest tie; includes step0',
              'model_fields': sorted(INPUT_FIELDS), 'metadata_in_forward': False,
              'normalization_fit_split': 'TRAIN only', 'label_scale_fit_split': 'TRAIN only',
              'source_sha256': _sha(__file__), 'inputs_hashes': {
                  'train': {k: _array_hash(v) for k, v in train_inputs.items()},
                  'validation': {k: _array_hash(v) for k, v in val_inputs.items()}},
              'labels_hashes': {split: {k: _array_hash(getattr(export, k)) for k in ('cost', 'attention')}
                                for split, export in (('train', train), ('validation', validation))},
              'normalization_sha256': _sha(output/'normalization.npz'),
              'appearance_inputs_hashes': None if not appearance else {
                  'train': {k: _array_hash(v) for k, v in train.appearance_inputs.items()},
                  'validation': {k: _array_hash(v) for k, v in validation.appearance_inputs.items()}},
              'supervision_audit_sha256': _sha(output/'supervision_audit.npz'),
              'sampling': 'identical per-seed generator reset for every arm; with replacement; weights affect sampler only',
              'sampling_weights_sha256': None if train.sampling_weights is None else _array_hash(train.sampling_weights),
              'validation_weights_sha256': None if validation.sampling_weights is None else _array_hash(validation.sampling_weights)}
    _write(output/'prefit_manifest.json', prefit)
    results, curves = {}, {}
    arms = [(arch, label) for arch in ARCHITECTURES for label in LABELS]
    if appearance:
        arms.append(('appearance', 'cost'))
    for arm_arch, label in arms:
        app = arm_arch == 'appearance'
        arch = 'independent' if app else arm_arch
        arm_plan = appearance_plan if app else {'item_dimension': plan['item_dimension'], 'context_dimension': plan['context_dimension'], **plan[arch]}
        fit_x = tensors['appearance_train' if app else 'train']
        valid_x = tensors['appearance_validation' if app else 'validation']
        for seed in config.seeds:
            torch.manual_seed(seed)
            model = TerminalScorer(arm_plan['item_dimension'], arm_plan['context_dimension'], arch, arm_plan['width'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            generator = torch.Generator().manual_seed(seed+config.batch_seed_offset)
            targets = torch.from_numpy(train_targets[label])
            valid_target = torch.from_numpy(val_targets[label])
            weights = None if train.sampling_weights is None else torch.as_tensor(train.sampling_weights, dtype=torch.float64)
            history, best_ce, best_state, selected_step = [], float('inf'), None, None
            started = time.perf_counter()
            for step in range(config.steps+1):
                if step % config.validate_every == 0:
                    metrics = _validation(model, valid_x, valid_target, validation.cost, config.validation_batch_size, validation.sampling_weights)
                    if not np.isfinite(metrics['own_label_ce']):
                        raise ValueError('Nonfinite validation CE')
                    history.append({'step': step, **metrics})
                    if metrics['own_label_ce'] < best_ce:
                        best_ce, selected_step = metrics['own_label_ce'], step
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if step == config.steps:
                    break
                model.train()
                indices = torch.randint(len(targets), (config.batch_size,), generator=generator) if weights is None else torch.multinomial(weights, config.batch_size, replacement=True, generator=generator)
                logits = model(fit_x['items'][indices], fit_x['context'][indices])
                loss = soft_cross_entropy(logits, targets[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training CE')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
                optimizer.step()
            name = f'{arm_arch}_{label}_seed{seed}'
            norm_file = 'appearance_normalization.npz' if app else 'normalization.npz'
            spec = {'architecture': arch, 'item_dimension': arm_plan['item_dimension'],
                    'context_dimension': arm_plan['context_dimension'], 'width': arm_plan['width']}
            checkpoint = {'model': best_state, 'spec': spec, 'normalization_file': norm_file,
                          'normalization_sha256': _sha(output/norm_file), 'label': label, 'seed': seed,
                          'selected_step': selected_step, 'target_scale': scales[label]}
            torch.save(checkpoint, output/(name+'.pt'))
            results[name] = {'checkpoint': name+'.pt', 'checkpoint_sha256': _sha(output/(name+'.pt')),
                             'selected_step': selected_step, 'selected_val_label_ce': best_ce,
                             'report_only_selected_val_true_cost': next(x['report_only_true_cost'] for x in history if x['step'] == selected_step),
                             'parameters': sum(p.numel() for p in model.parameters()),
                             'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
                             'training_seconds': time.perf_counter()-started, 'architecture': arm_arch, 'label': label, 'seed': seed}
            curves[name] = history
    manifest = {'status': 'completed_fixed_student_training', 'prefit_manifest_sha256': _sha(output/'prefit_manifest.json'),
                'arms': results, 'main_architecture_label_arms': 6, 'training_seeds': list(config.seeds),
                'appearance_control': 'completed' if appearance else 'not_supplied_not_run',
                'device': 'cpu', 'torch_version': torch.__version__, 'numpy_version': np.__version__,
                'step0_in_selection': True, 'no_DEV_or_confirmation_inputs': True}
    _write(output/'learning_curves.json', curves)
    _write(output/'manifest.json', manifest)
    return manifest


def predict_students(output_dir, inputs, appearance_inputs=None, batch_size=512, profile_parents=32):
    """Predict one deletion per parent; no outcome or metadata argument exists."""
    output = Path(output_dir)
    manifest = json.loads((output/'manifest.json').read_text())
    if _sha(output/'prefit_manifest.json') != manifest['prefit_manifest_sha256']:
        raise ValueError('Student prefit manifest changed')
    prefit = json.loads((output/'prefit_manifest.json').read_text())
    if _sha(__file__) != prefit['source_sha256']:
        raise ValueError('Student source changed since fitting')
    torch.set_num_threads(prefit['config']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    result = {}
    for name, record in manifest['arms'].items():
        path = output/record['checkpoint']
        if _sha(path) != record['checkpoint_sha256']:
            raise ValueError('Student checkpoint hash changed')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        norm_path = output/checkpoint['normalization_file']
        if _sha(norm_path) != checkpoint['normalization_sha256']:
            raise ValueError('Student normalization hash changed')
        raw = appearance_inputs if record['architecture'] == 'appearance' else inputs
        if raw is None:
            raise ValueError('Appearance checkpoint requires allowed appearance tensors')
        with np.load(norm_path, allow_pickle=False) as f:
            normalization = {k: f[k] for k in f.files}
            normalized = normalize_inputs(raw, normalization)
        tensors = _tensor_inputs(normalized)
        model = TerminalScorer(**checkpoint['spec'])
        model.load_state_dict(checkpoint['model'])
        model.eval().requires_grad_(False)
        scores, single_times = [], []
        with torch.inference_mode():
            for _ in range(5):
                model(tensors['items'][:1], tensors['context'][:1])
            for index in range(min(profile_parents, len(normalized['items']))):
                started = time.perf_counter()
                model(tensors['items'][index:index+1], tensors['context'][index:index+1])
                single_times.append(time.perf_counter()-started)
            started = time.perf_counter()
            for at in range(0, len(normalized['items']), batch_size):
                scores.append(model(tensors['items'][at:at+batch_size], tensors['context'][at:at+batch_size]).numpy())
            batch_seconds = time.perf_counter()-started
        logits = np.concatenate(scores)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite student prediction')
        result[name] = {'logits': logits, 'drop': logits.argmax(1), 'timing': {
            'device': 'cpu', 'batch1_profile_parents': len(single_times), 'warmup_forward_examples': 5,
            'batch1_median_seconds': float(np.median(single_times)) if single_times else None,
            'batch1_p95_seconds': float(np.quantile(single_times, .95)) if single_times else None,
            'main_prediction_batch_size': batch_size, 'main_prediction_seconds': batch_seconds,
            'main_prediction_forward_examples': len(logits),
            'total_forward_examples_including_profile': len(logits)+len(single_times)+5,
            'checkpoint_and_normalization_loading_included': False,
            'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'normalization_numeric_bytes': sum(v.nbytes for v in normalization.values()),
            'normalization_bytes_exclude_input_feature_workspace': True}}
    return result


def benchmark_synthetic(item_dimension=259, context_dimension=524, steps=20, threads=4):
    """Disposable synthetic optimizer timing, not a study fit or checkpoint."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    plan = architecture_plan(item_dimension, context_dimension)
    items = torch.randn(128, 3, item_dimension)
    context = torch.randn(128, context_dimension)
    targets = torch.softmax(torch.randn(128, 3), -1)
    rows = {}
    for architecture in ARCHITECTURES:
        model = TerminalScorer(item_dimension, context_dimension, architecture, plan[architecture]['width'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        durations = []
        for index in range(steps+5):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = soft_cross_entropy(model(items, context), targets).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if index >= 5:
                durations.append(time.perf_counter()-started)
        rows[architecture] = {'timed_steps': steps, 'warmup_steps': 5,
                              'median_step_seconds': float(np.median(durations)),
                              'mean_step_seconds': float(np.mean(durations)),
                              'projected1000_step_seconds_without_validation_or_IO': float(np.mean(durations)*1000)}
    return {'scope': 'Synthetic CPU software timing only; no study data, reader, or saved trained model',
            'shape': {'items': [128, 3, item_dimension], 'context': [128, context_dimension]},
            'cpu_threads': threads, 'architecture_plan': plan, 'timings': rows,
            'projected18_main_student_training_seconds_without_validation_or_IO':
                sum(r['mean_step_seconds'] for r in rows.values())*1000*3*3,
            'limitations': 'Linear arithmetic extrapolation on this machine, not a promised HPC wall time; excludes18 model initialization, VAL inference and IO'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--train', type=Path, required=True)
    fit.add_argument('--validation', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--config', type=Path)
    predict = sub.add_parser('predict')
    predict.add_argument('--models', type=Path, required=True)
    predict.add_argument('--inputs', type=Path, required=True)
    predict.add_argument('--output', type=Path, required=True)
    benchmark = sub.add_parser('benchmark')
    benchmark.add_argument('--item-dimension', type=int, default=259)
    benchmark.add_argument('--context-dimension', type=int, default=524)
    benchmark.add_argument('--steps', type=int, default=20)
    benchmark.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        config = {}
        if args.config:
            raw = json.loads(args.config.read_text())
            config = raw.get('students', raw)
        result = fit_students(load_export(args.train), load_export(args.validation), args.output, StudentConfig(**config))
        print(json.dumps({'status': result['status'], 'completed_arms_times_seeds': len(result['arms']),
                          'appearance_control': result['appearance_control']}))
    elif args.command == 'predict':
        models = args.models.parent if args.models.is_file() else args.models
        # Deployment path opens only allowed input tensors, never labels.npz or IDs.
        with np.load(args.inputs/'inputs.npz', allow_pickle=False) as f:
            inputs = validate_inputs({k: f[k] for k in f.files})
        appearance = None
        if (args.inputs/'appearance.npz').exists():
            with np.load(args.inputs/'appearance.npz', allow_pickle=False) as f:
                appearance = validate_inputs({k: f[k] for k in f.files})
        started = time.perf_counter()
        predictions = predict_students(models, inputs, appearance)
        duration = time.perf_counter()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValueError('Do not overwrite saved student predictions')
        arrays = {name+'__'+kind: row[kind] for name, row in predictions.items() for kind in ('logits', 'drop')}
        np.savez_compressed(args.output, **arrays)
        receipt = {'scope': 'frozen student prediction only, no outcome or metadata in forward',
                   'parents': len(inputs['items']), 'arms_times_seeds': len(predictions),
                   'seconds_including_checkpoint_loading_and_tensor_preparation': duration,
                   'student_manifest_sha256': _sha(models/'manifest.json'),
                   'inputs_npz_sha256': _sha(args.inputs/'inputs.npz'), 'predictions_sha256': _sha(args.output),
                   'labels_npz_opened': False, 'ids_json_opened': False,
                   'per_arm_timing_and_parameter_bytes': {name: row['timing'] for name, row in predictions.items()}}
        _write(args.output.with_suffix('.json'), receipt)
        print(json.dumps(receipt))
    else:
        result = benchmark_synthetic(args.item_dimension, args.context_dimension, args.steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write(args.output, result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
