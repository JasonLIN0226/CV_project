"""Future train/validation comparison; no work happens on import.

Selectors receive only historical arrays. Cue/evidence, labels, futures, and
decoder state are read only after all policies have made their selections.
This module does not train, generate data, or change the original Gate verdict.
"""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter
import numpy as np
import torch

from .core import B, R, rollout, score_update
from .states import validate_manifest

NEW_POLICIES = ('attention_proxy_student', 'deletion_cost_student')
BASELINES = ('fifo', 'reservoir', 'temporal_buckets', 'visual_diversity',
             'prediction_surprise', 'attention')


def verify_trace(trace, query, selected):
    """Independent replay verifies exact arrival order and no resurrection."""
    memory = []
    if len(trace) != max(0, query - R + 1):
        raise ValueError('incomplete online trace')
    for clock, row in zip(range(R, query + 1), trace):
        arrival = clock - R
        actual = row['selected']
        if (row['clock'] != clock or row['previous'] != memory or row['arrival'] != arrival
                or len(actual) != min(B, len(memory) + 1) or len(set(actual)) != len(actual)
                or not set(actual) <= set(memory + [arrival])):
            raise ValueError('invalid eviction trace or resurrection')
        memory = actual
    if memory != list(selected):
        raise ValueError('final memory differs from trace')


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_episode(store, rec, predictor, students, decoder, *, seed, device,
                     collate, load_episode, selected_evidence, baseline_replay,
                     sample_with_indices, append_metric_rows, pair_diagnostic,
                     reservoir_seed):
    """Compare the frozen students and prior baselines on one validation episode.

    The callers must verify hashes and the state/teacher/student seals and budget
    first. This function is only invoked in a later authorized evaluation, and
    forbids test records. Timers report complete student replay and *combined*
    baseline replay; old per-update timings exclude surprise forecast work.
    """
    if rec.get('split') != 'validation' or rec.get('task') != 'gate':
        raise ValueError('only Gate validation records allowed')
    if seed not in (17, 29, 43) or set(students) != set(NEW_POLICIES):
        raise ValueError('unexpected model seed or student arms')
    if store.c['data']['recent'] != R or store.c['data']['budget'] != B or tuple(store.c['data']['horizons']) != (3, 6):
        raise ValueError('frozen B/R/horizon configuration mismatch')
    q = int(rec['query'])
    # Load the allowed prefix and pass only copies to the online code.
    z = store.features(rec)[:q + 1].copy()
    projected = store.projected_features(rec)[:q + 1].copy()
    observed_actions = load_episode(store.c, rec, keys=('actions',))['actions'][:q].copy()
    predictor.eval()
    _sync(device); start = perf_counter()
    selected, old_trace, old_update_times, aux_bytes = baseline_replay(
        z, projected, observed_actions, q, predictor, collate, device, reservoir_seed)
    _sync(device); baseline_seconds = perf_counter() - start
    if set(selected) != set(BASELINES):
        raise ValueError('baseline policy set differs from protocol')
    traces = {}
    timings = {'baseline_combined_seconds': baseline_seconds,
               'baseline_update_times_incomplete': old_update_times,
               'surprise_aux_payload_bytes': aux_bytes}
    for policy, model in students.items():
        model.eval()
        if sum(p.numel() for p in model.parameters()) != 45633:
            raise ValueError('student parameter count mismatch')
        _sync(device); start = perf_counter()
        chosen, trace = rollout(projected, observed_actions, q, lambda state: score_update(model, state))
        _sync(device); timings[policy + '_seconds'] = perf_counter() - start
        verify_trace(trace, q, chosen)
        selected[policy] = chosen
        traces[policy] = trace
    # Original baseline implementation returns interleaved policy traces.
    for policy in BASELINES:
        policy_trace = [{k: v for k, v in row.items() if k != 'policy'}
                        for row in old_trace if row['policy'] == policy]
        verify_trace(policy_trace, q, selected[policy])
        traces[policy] = policy_trace
    # This cue-aware oracle is an explicitly privileged reference, not an online selector.
    evidence = selected_evidence(store.c, rec, load_episode(store.c, rec, True, keys=('evidence',))['evidence'])
    manual = [int(t) for t in evidence if int(t) <= q - R][:B]
    for t in range(q - R, -1, -1):
        if len(manual) == B:
            break
        if t not in manual:
            manual.append(t)
    selected['manual_keyframe'] = sorted(manual)
    selected['recent_only'] = []
    names = list(selected)
    examples = [sample_with_indices(store, rec, selected[name], load_episode, selected_evidence) for name in names]
    prediction = predictor(collate([item[0] for item in examples], device)).cpu().numpy()
    if not np.isfinite(prediction).all() or len(prediction) != len(names):
        raise ValueError('nonfinite or incomplete predictor output')
    rows = []
    for name, pred, (_, target, info) in zip(names, prediction, examples):
        before = len(rows)
        append_metric_rows(rows, seed=seed, task='gate', split='validation', policy=name,
                           rec=rec, chosen=selected[name], target=target, prediction=pred,
                           sample_info=info, c=store.c, store=store, decoders=decoder,
                           load_episode=load_episode, pair_diagnostic=pair_diagnostic)
        for row in rows[before:]:
            row['online_irreversible'] = name in BASELINES or name in NEW_POLICIES
            row['manual_privileged'] = name == 'manual_keyframe'
            row['selected_indices'] = ';'.join(map(str, selected[name]))
            row['event_hit'] = int(bool(set(range(int(rec['cue_start']), int(rec['cue_end']) + 1)) & set(selected[name])))
    return {'rows': rows, 'selected': selected, 'traces': traces, 'timings': timings}


def validation_records(store):
    """Select whole-pair validation only after validating the manifest firewall."""
    rows = validate_manifest(store.manifest)
    return [rec for rec in rows if rec['split'] == 'validation']


def aggregate_validation(rows, validation_manifest):
    """Descriptive paired validation aggregation, with strict completeness.

    Validation is not an independent test and does not change the old Gate
    verdict. Scene-cluster inferential testing on later fresh test data is a
    separate implementation and is not implied by these descriptive means.
    """
    recs = {r['id']: r for r in validation_manifest}
    if len(recs) != 192 or any(r.get('split') != 'validation' or r.get('task') != 'gate' for r in recs.values()):
        raise ValueError('expected 192 unique Gate validation episodes')
    policies = set(BASELINES) | set(NEW_POLICIES) | {'recent_only', 'manual_keyframe'}
    seeds, horizons = {17, 29, 43}, {3, 6}
    expected = {(s, e, p, h) for s in seeds for e in recs for p in policies for h in horizons}
    seen = set()
    metrics = ('state_accuracy', 'normalized_mse', 'state_position_mse', 'pair_accuracy', 'event_hit', 'evidence_hit')
    grouped = defaultdict(list)
    for row in rows:
        key = (row['seed'], row['episode'], row['condition'], row['horizon'])
        if key not in expected or key in seen:
            raise ValueError('duplicate or unexpected validation result row')
        rec = recs[row['episode']]
        if row.get('split') != 'validation' or row.get('pair') != rec['pair']:
            raise ValueError('result identity or split mismatch')
        for name in metrics:
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError('nonfinite validation metric')
            if name in {'state_accuracy', 'pair_accuracy', 'event_hit', 'evidence_hit'} and not 0 <= value <= 1:
                raise ValueError('invalid accuracy or hit rate')
            if name in {'normalized_mse', 'state_position_mse'} and value < 0:
                raise ValueError('negative squared error')
        seen.add(key)
        grouped[(row['condition'], row['seed'], row['horizon'])].append(row)
    if seen != expected:
        raise ValueError('validation result rows are incomplete')
    by_group = []
    for (policy, seed, horizon), group in sorted(grouped.items()):
        entry = {'policy': policy, 'seed': seed, 'horizon': horizon, 'episodes': len(group)}
        entry.update({name: float(np.mean([r[name] for r in group])) for name in metrics})
        hit = [r for r in group if r['event_hit'] == 1]
        miss = [r for r in group if r['event_hit'] == 0]
        entry['route_accuracy_given_event_hit'] = float(np.mean([r['state_accuracy'] for r in hit])) if hit else None
        entry['route_accuracy_given_event_miss'] = float(np.mean([r['state_accuracy'] for r in miss])) if miss else None
        by_group.append(entry)
    pooled = {policy: float(np.mean([g['state_accuracy'] for g in by_group if g['policy'] == policy]))
              for policy in sorted(policies)}
    return {'split': 'validation', 'rows': len(rows), 'episodes': len(recs), 'group_metrics': by_group,
            'pooled_route_accuracy': pooled,
            'students_exceed_recent_only_on_pooled_validation': {
                name: pooled[name] > pooled['recent_only'] for name in NEW_POLICIES},
            'scope': 'descriptive validation only; not a fresh independent test or Phase1 pass'}


def evaluate_full_validation(contexts, *, dependency_factory, seed_factory):
    """Evaluate all seeds/validation episodes in a later authorized run.

    Each context has keys store, predictor, students, decoder, device. Shared
    functions from the frozen code are supplied by dependency_factory(seed).
    This function itself writes no artifacts and performs no training.
    """
    if set(contexts) != {17, 29, 43}:
        raise ValueError('all three frozen predictor/student seeds required')
    rows, traces, times = [], [], []
    canonical = None
    for seed in (17, 29, 43):
        context = contexts[seed]
        recs = validation_records(context['store'])
        ids = {(r['id'], r['pair'], r['sibling']) for r in recs}
        if canonical is None:
            canonical = ids
        elif ids != canonical:
            raise ValueError('seed contexts must use identical validation episodes')
        deps = dependency_factory(seed)
        for rec in recs:
            result = evaluate_episode(context['store'], rec, context['predictor'], context['students'],
                    context['decoder'], seed=seed, device=context['device'],
                    reservoir_seed=seed_factory(seed, rec), **deps)
            rows.extend(result['rows'])
            traces.append({'seed': seed, 'episode': rec['id'], 'policies': result['traces']})
            times.append({'seed': seed, 'episode': rec['id'], **result['timings']})
    return {'rows': rows, 'traces': traces, 'timings': times,
            'summary': aggregate_validation(rows, validation_records(contexts[17]['store']))}
