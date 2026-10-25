"""Posthoc, read-only audit of archived terminal reader attention and errors.

No inference or fitting. Actual-plan prototype classes and 32-plan expected-MSE
oracle choices are deliberately separate. Attention is the current-query K3
softmax; K2 weights are algebraic renormalizations, not new model executions.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(xs):
    x = np.asarray(xs, float)
    return {'n': len(x), 'mean': float(x.mean()), 'min': float(x.min()),
            'median': float(np.median(x)), 'max': float(x.max())}


def rate(xs):
    return {'n': len(xs), 'count': int(sum(xs)), 'rate': float(np.mean(xs))}


def audit(run):
    run = Path(run)
    paths = [run/'data/records.json', run/'development/baseline_decisions.json',
             run/'development/reader_qualification_rows.json', run/'protocol.json']
    records = {r['id']: r for r in read(paths[0])}
    decisions, qualification = read(paths[1]), read(paths[2])
    qmap = {(q['id'], q['arm']): q for q in qualification}
    assert len(qmap) == len(qualification)
    assert len({d['id'] for d in decisions}) == len(decisions)
    attention, k2, synergy, singles, accuracy, oracle = (defaultdict(list) for _ in range(6))
    oracle_rows = []
    for d in decisions:
        r = records[d['id']]
        assert r['split'] == 'development'
        w = np.asarray(d['attention'], float)
        assert w.shape == (3,) and np.isfinite(w).all() and np.min(w) >= 0
        assert abs(w.sum()-1) < 1e-6
        task, condition = r['task_family'], r['condition']
        for i in range(3):
            salient = i == r['salient_slot']
            attention[f'{task}/role/{r["roles"][i]}'].append(w[i])
            attention[f'{task}/{condition}/role/{r["roles"][i]}'].append(w[i])
            attention[f'{task}/{condition}/salient/{salient}'].append(w[i])
            attention[f'{task}/{condition}/axis/{r["semantic_axes"][i]}/salient/{salient}'].append(w[i])
            if task == 'S':
                attention[f'S/tag/{r["tags"][i]}'].append(w[i])
                attention['S/role/'+('paired' if r['roles'][i] != 'D' else 'singleton')].append(w[i])
        if task == 'D':
            salrole = 'none' if r['salient_slot'] is None else r['roles'][r['salient_slot']]
            attention[f'D/{condition}/salient_role/{salrole}/B'].append(w[r['roles'].index('B')])
        if task != 'S':
            continue
        good = r['semantic_correct_delete_set'][0]
        keep = [i for i in range(3) if i != good]
        tag = r['tags'][keep[0]]
        same = r['slot_payload'][keep[0]] == r['slot_payload'][keep[1]]
        complete = qmap[r['id'], 'native_'+str(good)]
        full = qmap[r['id'], 'full']
        weights = w[keep]/w[keep].sum()
        for key in ['all', f'majority_tag/{tag}', f'{condition}/majority_tag/{tag}/same_payload/{same}']:
            k2[key+'/weight'].extend(weights)
            k2[key+'/minimum_weight'].append(weights.min())
            k2[key+'/uniform'].append(bool(np.max(np.abs(weights-.5)) <= 1e-7))
            k2[key+'/complete_pair_K3_mass'].append(w[keep].sum())
        for key in [f'majority_tag/{tag}/same_payload/{same}', f'{condition}/majority_tag/{tag}/same_payload/{same}']:
            synergy[key+'/K3_correct'].append(full['strict_correct'])
            synergy[key+'/native_K2_complete_correct'].append(complete['strict_correct'])
            synergy[key+'/K2_correct_K3_wrong'].append(complete['strict_correct'] and not full['strict_correct'])
            synergy[key+'/K2_wrong_K3_correct'].append(not complete['strict_correct'] and full['strict_correct'])
        selected = int(np.argmin(d['expected_losses']))
        if selected != good:
            chosen = qmap[r['id'], 'native_'+str(selected)]
            row = {'id': r['id'], 'family': r['family'], 'condition': condition, 'majority_tag': tag,
                   'same_payload': same, 'oracle_delete': selected, 'semantic_delete': good,
                   'evaluation32_mse_advantage': d['expected_losses'][good]-d['expected_losses'][selected],
                   'actual_complete_wrong': not complete['strict_correct'],
                   'actual_missing_selected_correct': chosen['strict_correct'],
                   'actual_missing_selected_lower_mse': chosen['mse'] < complete['mse'],
                   'actual_missing_selected_closer_midpoint': chosen['midpoint_mse'] < complete['midpoint_mse'],
                   'actual_complete_signed_prototype_distance_gap': complete['distances'][1-r['outcome_class']]-complete['distances'][r['outcome_class']],
                   'actual_selected_signed_prototype_distance_gap': chosen['distances'][1-r['outcome_class']]-chosen['distances'][r['outcome_class']]}
            oracle_rows.append(row)
            for key, value in row.items():
                if key.startswith(('actual_', 'evaluation32_')):
                    oracle[key].append(value)
    for q in qualification:
        r = records[q['id']]
        task, condition, arm = r['task_family'], r['condition'], q['arm']
        if task in ('R', 'D') and arm.startswith('single_'):
            i = int(arm.split('_')[1])
            if task == 'D' and r['roles'][i] == 'D':
                continue
            known_bit = q['class']//2 if r['semantic_axes'][i] == 'a' else q['class'] % 2
            correct = known_bit == r['slot_payload'][i]
            salient = i == r['salient_slot']
            for key in [f'{task}/salient/{salient}', f'{task}/{condition}/salient/{salient}']:
                singles[key+'/known_axis_bit_correct'].append(correct)
                singles[key+'/full_outcome_correct'].append(q['strict_correct'])
        if task in ('R', 'D') and (arm == 'full' or arm.startswith(('native_', 'fixed_')) and q['semantic_complete']):
            salrole = 'none' if r['salient_slot'] is None else r['roles'][r['salient_slot']]
            key = f'{task}/{condition}/salient_role/{salrole}/{arm.split("_")[0]}'
            accuracy[key+'/joint_correct'].append(q['strict_correct'])
            accuracy[key+'/a_bit_correct'].append(q['class']//2 == r['bits'][0])
            accuracy[key+'/b_bit_correct'].append(q['class']%2 == r['bits'][1])
            accuracy[key+'/mean_bit_correct'].extend([q['class']//2 == r['bits'][0], q['class']%2 == r['bits'][1]])
    # Explicit counterexample to a general impossibility claim: a mean of four
    # local-type one-hot vectors preserves the entire three-item multiset.
    cases = 0
    for tags in itertools.product((0, 1), repeat=3):
        if len(set(tags)) == 1:
            continue
        pair_tag = 0 if tags.count(0) == 2 else 1
        for payload in itertools.product((0, 1), repeat=3):
            pooled = np.eye(4)[[2*t+b for t, b in zip(tags, payload)]].mean(0)
            counts = np.rint(3*pooled).astype(int)
            decoded_tag = int(counts[2:].sum() == 2)
            decoded_xor = int(counts[2*decoded_tag+1] % 2)
            expected = sum(b for t,b in zip(tags,payload) if t == pair_tag) % 2
            assert decoded_tag == pair_tag and decoded_xor == expected
            cases += 1
    return {
        'scope': 'Posthoc descriptive audit of archived development data; no new fitting, inference, confirmation, or attention intervention.',
        'provenance': {str(p.relative_to(run)): sha(p) for p in paths},
        'n_parents': len(decisions), 'n_families': len({d['family'] for d in decisions}),
        'definitions': {
            'attention': 'Current-query K3 content softmax, not future-attention training labels.',
            'K2_weights': 'Algebraic K3-weight renormalization over the retained pair; logits are itemwise and unchanged. Mean over both items is trivially 0.5 and is not evidence of uniform attention.',
            'single': 'One candidate plus current blank at fixed ages [6,4]; known-axis bit is extracted from the nearest full-outcome prototype. It is not joint future accuracy; D decoy is excluded.',
            'oracle': 'argmin expected MSE across 32 evaluation plans. Diagnostic classes below use the distinct actual episode plan.',
            'confidence': 'Prototype-distance margins are geometric scores, not calibrated confidence. Closeness to the prototype mean does not establish an uncertainty mechanism.',
            'K2': 'Accuracy tables distinguish native and fixed ages rather than pooling them.'},
        'attention': {k: stats(v) for k,v in sorted(attention.items())},
        'S_K2_renormalized': {k: rate(v) if k.endswith('/uniform') else stats(v) for k,v in sorted(k2.items())},
        'S_actual_plan_strata': {k: rate(v) for k,v in sorted(synergy.items())},
        'single_cue_plus_blank': {k: rate(v) for k,v in sorted(singles.items())},
        'R_D_actual_plan_complete_accuracy': {k: rate(v) for k,v in sorted(accuracy.items())},
        'S_oracle_semantic_disagreements': {
            'n': len(oracle_rows), 'denominator': sum(d['task_family'] == 'S' for d in decisions),
            'metrics': {k: rate(v) if isinstance(v[0], (bool,np.bool_)) else stats(v) for k,v in sorted(oracle.items())},
            'rows': oracle_rows},
        'reader_structure': {
            'individual_content_logits': True,
            'downstream_channels': ['weighted aligned memory', 'weighted temporal velocity', 'mean age', 'past action mean', 'future action embedding', 'memory/action and velocity/action products', 'nonlinear decoder'],
            'two_item_velocity_identity': '-w*(1-w)*(age1-age2)*(aligned1-aligned2) / max(w*(1-w)*(age1-age2)^2,0.01)',
            'general_mean_pooling_counterexample': {'local_embedding': 'one-hot four (tag,bit) types', 'pooled': 'mean of three embeddings', 'nonlinear_decoder': 'recover counts; find tag with count two; parity of its bit-one count', 'exact_correct': cases, 'total': 48, 'scope': 'Existence example only; not a claim that archived DINO/projection or learned decoder realizes it.'}},
        'interpretation_limits': [
            'Strong tag-dependent attention and K3 error associations support a reader bottleneck hypothesis, not proof that softmax alone causes all errors.',
            'Adding the third item changes several reader channels together. It is a saved subset intervention, not an isolated attention-weight intervention.',
            'The independent scalar eviction bound does not apply to a nonlinear decoder operating on an aggregated set representation.',
            'K2 equal weights hold for identical local observations; different payload and/or salient styling can yield unequal weights.',
            'Oracle semantic disagreement can coexist with correct nearest-prototype classification. Do not equate all 159 cases with confidently wrong complete predictions or midpoint hedging.'
        ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Posthoc output must be outside frozen run')
    result = audit(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    result['audit_script_sha256'] = sha(__file__)
    (args.output/'reader_attention_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'parents': result['n_parents'], 'families': result['n_families'],
                      'S_oracle_disagreements': result['S_oracle_semantic_disagreements']['n'],
                      'output': str(args.output/'reader_attention_audit.json')}))

if __name__ == '__main__':
    main()
