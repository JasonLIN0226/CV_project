"""Controlled content/age/executed-action diagnosis on fixed Gate parents.

Artificial crossed inputs are evaluator-only interventions, not physical
counterfactuals, deployable histories, or repaired benchmark performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'experiments/predictor/control/src'):
    sys.path.insert(0, str(p))
import numpy as np
import torch
from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.diagnose_gate_oracle_metric import group_ci, paired_components, save, sha
from experiments.phase2.memory import MemoryEntry
from experiments.phase2.portable_snapshot import PortableSnapshot, state_hash
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator

FACTOR_NAMES = ('content', 'age', 'past_action')
MAIN_ARMS = tuple(product((0, 1), repeat=3))
WRONG_ARMS = tuple((2, age, action) for age, action in product((0, 1), repeat=2))


def factor_entries(record, z, sibling_z, actions, comparison, content, age, action):
    """Return independent inputs, preserving the original shared filler exactly.

    Content 0=noncue, 1=correct cue, 2=sibling cue. Age/action levels refer to
    their respective original noncue(0) or cue(1) timestamps independently.
    """
    if content not in (0, 1, 2) or age not in (0, 1) or action not in (0, 1):
        raise ValueError('Unknown controlled factor level')
    yes, no = set(comparison['with_cue']), set(comparison['without_cue'])
    shared = yes & no
    if len(yes) != 2 or len(no) != 2 or len(shared) != 1:
        raise ValueError('Require two original K2 sets with exactly one common filler')
    cue_time, noncue_time, filler = next(iter(yes-no)), next(iter(no-yes)), next(iter(shared))
    q = record['query']
    time_choices = (noncue_time, cue_time)
    slot_time = time_choices[age]
    feature = (z[noncue_time], z[cue_time], sibling_z[cue_time])[content]
    recent = tuple(MemoryEntry(record['id'], t, z[t]) for t in range(q-3, q+1))
    slot = MemoryEntry(record['id'], slot_time, feature)
    history = tuple(sorted((slot, MemoryEntry(record['id'], filler, z[filler])), key=lambda e: e.timestamp))
    past = {e.entry_id: actions[e.timestamp:q].sum(0, dtype=np.float32) for e in recent+history}
    past[slot.entry_id] = actions[time_choices[action]:q].sum(0, dtype=np.float32)
    return recent, history, past, {
        'cue_time': cue_time, 'noncue_time': noncue_time, 'shared_filler_time': filler,
        'slot_time': slot_time, 'slot_index': 4+next(i for i, e in enumerate(history) if e.entry_id == slot.entry_id),
        'action_summary_source_time': time_choices[action],
        'slot_past_action': past[slot.entry_id].tolist(),
    }


def factor_effects(values):
    """Balanced finite differences and exact endpoint Shapley allocation.

    Interaction effects are averages of ordinary differences of differences.
    Shapley allocations average all six orders of toggling the three factors;
    their sum exactly equals response111-response000, with no causal claim.
    """
    if set(values) != set(MAIN_ARMS) or not np.isfinite(list(values.values())).all():
        raise ValueError('Require all eight finite factorial responses')
    out = {}
    for size in (1, 2, 3):
        for selected in combinations(range(3), size):
            other = [i for i in range(3) if i not in selected]
            differences = []
            for levels in product((0, 1), repeat=len(other)):
                value = 0.
                for switches in product((0, 1), repeat=size):
                    point = [0]*3
                    for i, v in zip(other, levels): point[i] = v
                    for i, v in zip(selected, switches): point[i] = v
                    value += (-1)**(size-sum(switches))*values[tuple(point)]
                differences.append(value)
            prefix = 'main_' if size == 1 else 'interaction_'
            out[prefix+'_'.join(FACTOR_NAMES[i] for i in selected)] = float(np.mean(differences))
    for i, name in enumerate(FACTOR_NAMES):
        other = [j for j in range(3) if j != i]
        total = 0.
        for size in range(3):
            for chosen in combinations(other, size):
                low = [int(j in chosen) for j in range(3)]
                high = low.copy(); high[i] = 1
                weight = math.factorial(size)*math.factorial(2-size)/math.factorial(3)
                total += weight*(values[tuple(high)]-values[tuple(low)])
        out['endpoint_allocation_'+name] = float(total)
    out['endpoint_gap'] = float(values[1, 1, 1]-values[0, 0, 0])
    if not np.isclose(sum(out['endpoint_allocation_'+name] for name in FACTOR_NAMES), out['endpoint_gap'], rtol=0, atol=1e-12):
        raise AssertionError('Order-averaged endpoint allocation is not exact')
    return out


def read_paths(scores, batch, slot_index):
    """Inspect the unchanged reader paths; do not alter its predictions."""
    with torch.inference_mode():
        valid = batch['mask']; hist = valid & (batch['type'] == 1)
        p = batch['projected_z'].masked_fill(~valid[..., None], 0)
        weight = scores._weights(p, batch['projected_current'], hist)
        past = scores._model.past_action(batch['past_actions'].masked_fill(~valid[..., None], 0))
        aligned = p+past
        age = batch['age'].masked_fill(~valid, 0)/8
        mean_age = (weight*age).sum(1)
        centered = age-mean_age[:, None]
        memory = (weight[..., None]*aligned).sum(1)
        variance = (weight*centered.square()).sum(1)
        velocity = -(weight[..., None]*centered[..., None]*(aligned-memory[:, None])).sum(1)/variance.clamp_min(.01)[:, None]
        return {'slot_attention': float(weight[0, slot_index]),
                'mean_age': float(mean_age[0]), 'age_variance': float(variance[0]),
                'memory_norm': float(torch.linalg.vector_norm(memory)),
                'velocity_norm': float(torch.linalg.vector_norm(velocity)),
                'transport_h3_norm': float(torch.linalg.vector_norm(velocity*(mean_age+3/8)[:, None])),
                'transport_h6_norm': float(torch.linalg.vector_norm(velocity*(mean_age+6/8)[:, None]))}


def grouped(rows, key):
    return {condition: group_ci([r for r in rows if condition == 'all' or r['condition'] == condition], key)
            for condition in ('all', 'id', 'long_gap', 'distractor_shift')
            if any(condition == 'all' or r['condition'] == condition for r in rows)}


def run(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Require a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); started = time.monotonic()
    sources = [Path(__file__), ROOT/'experiments/phase2/diagnose_gate_oracle_metric.py',
               ROOT/'experiments/phase2/causal_model_scores.py', ROOT/'experiments/predictor/control/src/phase1/model.py']
    inputs = [args.local_run/name for name in ('parents.json', 'episode_rows.json', 'predictions.npz', 'protocol.json')]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protocol = {
        'version': 'gate-cue-content-age-action-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'fixed posthoc diagnosis on all72 comparable parents from already observed fresh192 development set',
        'main_factors': {'content': ['noncue', 'correct cue'], 'age': ['noncue time', 'cue time'],
                         'past_action': ['suffix from noncue time', 'suffix from cue time']},
        'arms': [list(v) for v in MAIN_ARMS+WRONG_ARMS],
        'wrong_cue': 'sibling feature at the same original cue timestamp, at all4 matched age/action settings',
        'fixed': ['shared filler content/time/action', 'four recent frames', 'raw current skip', 'query and future controls', 'predictor weights', 'h3/h6 same joint history'],
        'endpoints': '111 is original best-with-cue; 000 original best-without-cue; all144 exact saved-vector comparisons required',
        'effects': 'all balanced main/interaction contrasts plus exact order-averaged (Shapley) allocation of endpoint111-minus000 loss; evaluator input effects, not causal physics',
        'cohorts': 'all72 and fixed originally cue-harmed27; 5000 scene-pair bootstrap samples seed20261022',
        'representative_rule': 'max positive content allocation among originalbad; max age allocation among originalbad; most negative correct-minus-wrong loss at cue age/action among original cue-helped; deterministic episode-ID ties',
        'model_training': False, 'policy_training': False, 'test_arrays': False,
        'runtime': {'python': sys.executable, 'numpy': np.__version__, 'torch': torch.__version__,
                    'threads': 1, 'blas': getattr(np.__config__, 'CONFIG', {}).get('Build Dependencies', {}).get('blas', {})},
        'source_sha256': source_hashes, 'input_sha256': input_hashes,
        'limitations': ['Crossed content/time/action inputs can be OOD and physically inconsistent.',
                        'Wrong cue is an evaluator-only matched content intervention.',
                        'Chosen endpoints minimize the original future loss within their fixed cue classes; they are hindsight, not causal policies.',
                        'Factor interactions preclude one unconditional scalar cause; allocation convention shares interactions across factors.',
                        'No adapted policy performance or independent generalization claim.']}
    save(out/'protocol.json', protocol)
    for p in sources:
        dest = out/'source'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    old = PortableSnapshot(args.snapshot); evaluator = FollowupGateEvaluator(old, args.followup)
    scores = CausalModelScores(evaluator); frozen_hash = state_hash(scores._model)
    fresh = FreshGateDataset(args.fresh); fresh.verify_old_normalization(old.norm)
    parents = [r for r in json.loads((args.local_run/'parents.json').read_text()) if r['comparison_status'] == 'available']
    assert len(parents) == 72 and len(fresh.records) == 192
    prior_rows = json.loads((args.local_run/'episode_rows.json').read_text())
    with np.load(args.local_run/'predictions.npz', allow_pickle=False) as a:
        prior = {(r['episode_id'], r['keeps_cue']): (r, p.copy()) for r, p in zip(prior_rows, a['predictions']) if r['arm'] == 'original'}
    rows, predictions, effects, matched, cases = [], [], [], [], []
    checks = {'endpoint_vectors_exact': 0, 'factor_endpoints': 0, 'attention_age_action_invariance': 0}
    for index, parent in enumerate(parents):
        rec = fresh.by_id[parent['episode_id']]; z, actions = fresh.episode(rec)
        sibling_rec = fresh.by_id[rec['sibling']]; sibling_z, sibling_actions = fresh.episode(sibling_rec)
        q = rec['query']; times = q+np.array([3, 6]); target, sibling_target = z[times], sibling_z[times]
        if not np.array_equal(actions, sibling_actions) or sibling_rec['query'] != q:
            raise AssertionError('Sibling control/query mismatch')
        response = {}; original_bad = prior[rec['id'], True][0]['metrics']['mse']-prior[rec['id'], False][0]['metrics']['mse'] > 1e-8
        for content, age, action in MAIN_ARMS+WRONG_ARMS:
            recent, history, past, meta = factor_entries(rec, z, sibling_z, actions, parent, content, age, action)
            if np.array_equal(z[meta['cue_time']], sibling_z[meta['cue_time']]):
                raise AssertionError('Sibling cue content is not distinct')
            batch = scores._batch(recent, history, past, q, actions[q:q+6].copy())
            with torch.inference_mode(): pred = scores._model(batch)[0].numpy().copy()
            if (content, age, action) in ((0, 0, 0), (1, 1, 1)):
                saved_row, saved_pred = prior[rec['id'], content == 1]
                if not np.array_equal(pred, saved_pred): raise AssertionError('Original endpoint differs from saved frozen prediction')
                checks['endpoint_vectors_exact'] += 1
            measure = paired_components(pred, target, sibling_target)
            measure['contrast_amplitude_mean'] = float(np.mean(measure['contrast_amplitude']))
            measure['raw_feature_mse'] = float(np.mean(((pred.astype(np.float64)-target)*old.norm['std'])**2))
            reader = read_paths(scores, batch, meta['slot_index'])
            row = {'episode_id': rec['id'], 'sibling_id': rec['sibling'], 'pair': rec['pair'], 'condition': rec['condition'],
                   'content': content, 'age': age, 'past_action': action, 'original_bad': original_bad,
                   'input': meta, 'reader': reader, **measure}
            rows.append(row); predictions.append(pred); response[content, age, action] = row
        for content in (0, 1, 2):
            weights = [response[content, age, action]['reader']['slot_attention'] for age, action in product((0, 1), repeat=2)]
            if not all(w == weights[0] for w in weights): raise AssertionError('Age/action unexpectedly changed content-only attention')
            checks['attention_age_action_invariance'] += 1
        for metric in ('mse', 'contrast_mse', 'orthogonal_mse', 'raw_feature_mse', 'contrast_amplitude_mean', 'feature_branch_correct'):
            effect = factor_effects({arm: response[arm][metric] for arm in MAIN_ARMS})
            effects.append({'episode_id': rec['id'], 'pair': rec['pair'], 'condition': rec['condition'],
                            'original_bad': original_bad, 'metric': metric, **effect})
        for age, action in product((0, 1), repeat=2):
            correct, wrong = response[1, age, action], response[2, age, action]
            matched.append({'episode_id': rec['id'], 'pair': rec['pair'], 'condition': rec['condition'],
                            'original_bad': original_bad, 'age': age, 'past_action': action,
                            **{key: correct[key]-wrong[key] for key in ('mse', 'contrast_mse', 'orthogonal_mse', 'raw_feature_mse', 'contrast_amplitude_mean', 'feature_branch_correct')}})
        meta = response[1, 1, 1]['input']
        cases.append({'episode_id': rec['id'], 'sibling_id': rec['sibling'], 'pair': rec['pair'], 'condition': rec['condition'],
                      'parent': parent['parent'], 'cue_time': meta['cue_time'], 'noncue_time': meta['noncue_time'],
                      'filler_time': meta['shared_filler_time'], 'query': q, 'target_times': times.tolist(),
                      'original_bad': original_bad,
                      'age_difference_frames': meta['noncue_time']-meta['cue_time'],
                      'past_action_suffix_difference': (actions[meta['cue_time']:q].sum(0, dtype=np.float32)-actions[meta['noncue_time']:q].sum(0, dtype=np.float32)).tolist()})
        checks['factor_endpoints'] += 1
        if (index+1) % 24 == 0: print(json.dumps({'episodes': index+1, 'total': 72}), flush=True)
    save(out/'episode_rows.json', rows); save(out/'factor_effects.json', effects)
    save(out/'correct_minus_wrong.json', matched); save(out/'cases.json', cases)
    np.savez_compressed(out/'predictions.npz', predictions=np.stack(predictions),
                        case_names=np.array([f"{r['episode_id']}:{r['content']}{r['age']}{r['past_action']}" for r in rows]))
    effect_summary, matched_summary = {}, {}
    for cohort in ('all', 'original_bad'):
        part = [r for r in effects if cohort == 'all' or r['original_bad']]
        effect_summary[cohort] = {metric: {key: grouped([r for r in part if r['metric'] == metric], key)
            for key in part[0] if key.startswith(('main_', 'interaction_', 'endpoint_'))}
            for metric in sorted({r['metric'] for r in part})}
        matched_summary[cohort] = {f'age{age}_action{action}': {key: grouped([r for r in matched if
            r['age'] == age and r['past_action'] == action and (cohort == 'all' or r['original_bad'])], key)
            for key in ('mse', 'contrast_mse', 'raw_feature_mse', 'contrast_amplitude_mean', 'feature_branch_correct')}
            for age, action in product((0, 1), repeat=2)}
    mse_effects = {r['episode_id']: r for r in effects if r['metric'] == 'mse'}
    correct_wrong = {r['episode_id']: r for r in matched if r['age'] == 1 and r['past_action'] == 1}
    bad = [r for r in cases if r['original_bad']]
    helped = [r for r in cases if mse_effects[r['episode_id']]['endpoint_gap'] < -1e-8]
    representative = {}
    for name, pool, score in (
            ('content_harm', bad, lambda r: -mse_effects[r['episode_id']]['endpoint_allocation_content']),
            ('age_contribution', bad, lambda r: -mse_effects[r['episode_id']]['endpoint_allocation_age']),
            ('correct_cue_success', helped, lambda r: correct_wrong[r['episode_id']]['mse'])):
        chosen = min(pool, key=lambda r: (score(r), r['episode_id']))
        representative[name] = chosen | {'mse_effects': mse_effects[chosen['episode_id']],
                                         'correct_minus_wrong_at_original_cue_age_action': correct_wrong[chosen['episode_id']]}
    save(out/'representatives.json', representative)
    for p, expected in {**source_hashes, **input_hashes}.items():
        if sha(ROOT/p) != expected: raise AssertionError('Frozen source/input changed: '+p)
    if state_hash(scores._model) != frozen_hash: raise AssertionError('Model state changed')
    summary = {'status': 'completed', 'elapsed_seconds': time.monotonic()-started, 'episodes': len(parents),
               'scene_pairs': len({r['pair'] for r in parents}), 'prediction_rows': len(rows),
               'original_bad_episodes': len(bad), 'checks': checks, 'protocol_sha256': sha(out/'protocol.json'),
               'source_sha256': sha(__file__), 'factor_effects': effect_summary, 'correct_minus_wrong': matched_summary,
               'model_state_bitwise_unchanged': True, 'predictor_audit': scores.audit(),
               'old_input_audit': old.audit(), 'fresh_input_audit': fresh.audit(),
               'thread_environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}}
    save(out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'elapsed_seconds', 'episodes', 'prediction_rows')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT/'build/hpc-20261020/development-assets')
    parser.add_argument('--followup', type=Path, default=ROOT/'build/history-diagnosis-20261020/followup-blas1')
    parser.add_argument('--fresh', type=Path, default=ROOT/'build/literature-baselines-20261022/fresh-data')
    parser.add_argument('--local-run', type=Path, default=ROOT/'build/literature-diagnosis-20261022/reader-local-v2')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
