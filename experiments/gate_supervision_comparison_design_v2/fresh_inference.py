"""Future sealed independent comparison; no data are read or generated on import.

These pure functions operate on supplied records only. They do not authorize or
open a test, generate episodes, train a model, or change the original Gate result.
Caller must verify signed provenance, checkpoint/source freeze, non-exposure,
and test-opening permission before any real independent evaluation.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

SPLITS = ('test_id', 'test_long_gap')
POLICIES = ('attention_proxy_student', 'deletion_cost_student')
SEEDS = (17, 29, 43)
HORIZONS = (3, 6)


def verify_fresh_manifest(records, *, generation_seed, prior_ids, prior_pairs, prior_scene_seeds,
                          model_frozen_before_generation, previously_unopened):
    """Fail closed when test identity, freeze, or exposure evidence is missing.

    Boolean provenance assertions must be independently substantiated by the
    caller; this function cannot prove truthfulness of the supplied audit data.
    """
    if generation_seed != 2026102701 or model_frozen_before_generation is not True or previously_unopened is not True:
        raise ValueError('fresh seed, freeze and non-exposure must be verified')
    if prior_ids is None or prior_pairs is None or prior_scene_seeds is None:
        raise ValueError('complete prior identity sets are required')
    if len(records) != 256:
        raise ValueError('expected 64 sibling pairs per test condition')
    by_id, groups = {}, defaultdict(list)
    for r in records:
        if r.get('split') not in SPLITS or r.get('task') != 'gate':
            raise ValueError('unexpected independent-test split or task')
        if r['id'] in by_id or r['id'] in prior_ids or r['pair'] in prior_pairs or r['seed'] in prior_scene_seeds:
            raise ValueError('duplicate or previously exposed identity')
        by_id[r['id']] = r
        groups[(r['split'], r['pair'])].append(r)
    if any(sum(split == s for split, _ in groups) != 64 for s in SPLITS):
        raise ValueError('expected 64 scene pairs in each condition')
    if len({pair for _, pair in groups}) != 128:
        raise ValueError('scene pair identity crosses conditions')
    for group in groups.values():
        if (len(group) != 2 or group[0]['sibling'] != group[1]['id']
                or group[1]['sibling'] != group[0]['id'] or group[0]['seed'] != group[1]['seed']):
            raise ValueError('incomplete or mismatched sibling pair')
    if len({group[0]['seed'] for group in groups.values()}) != len(groups):
        raise ValueError('scene seed repeated across distinct pairs')
    return {'episodes': len(records), 'pairs_per_condition': 64, 'disjoint_id_pair_scene_seed': True}


def paired_route_inference(rows, records, *, bootstrap_seed=0, resamples=10000):
    """Paired D-minus-A scene bootstrap; 97.5% two-sided percentile CI.

    Exactly two siblings, three paired model seeds and two horizons are
    averaged per scene before resampling. Two intervals use Bonferroni family
    alpha 0.05. The only positive claim requires both lower bounds > 0.
    Call verify_fresh_manifest and the independent audit before invoking this
    on real data; output alone is not provenance evidence.
    """
    if bootstrap_seed != 0 or resamples != 10000:
        raise ValueError('bootstrap settings are fixed by protocol')
    recs = {r['id']: r for r in records}
    if len(recs) != 256 or any(r.get('split') not in SPLITS for r in recs.values()):
        raise ValueError('expected complete fresh two-condition manifest')
    expected = {(s, e, p, h) for s in SEEDS for e in recs for p in POLICIES for h in HORIZONS}
    seen = set()
    grouped = defaultdict(list)
    for row in rows:
        key = (row['seed'], row['episode'], row['condition'], row['horizon'])
        if key not in expected or key in seen:
            raise ValueError('duplicate or unexpected inference row')
        rec = recs[row['episode']]
        if row.get('split') != rec['split'] or row.get('pair') != rec['pair']:
            raise ValueError('inference row identity mismatch')
        value = float(row['state_accuracy'])
        if not np.isfinite(value) or value not in (0., 1.):
            raise ValueError('route accuracy must be a finite episode indicator')
        grouped[(rec['split'], rec['pair'], row['condition'])].append(value)
        seen.add(key)
    if seen != expected:
        raise ValueError('independent comparison rows incomplete')
    rng = np.random.default_rng(bootstrap_seed)
    out = {}
    for split in SPLITS:
        pairs = sorted({r['pair'] for r in recs.values() if r['split'] == split})
        if len(pairs) != 64:
            raise ValueError('independent condition requires 64 scenes')
        deltas = []
        for pair in pairs:
            left = grouped[(split, pair, POLICIES[0])]
            right = grouped[(split, pair, POLICIES[1])]
            if len(left) != 12 or len(right) != 12:
                raise ValueError('each scene needs 2 siblings x 3 seeds x 2 horizons')
            deltas.append(float(np.mean(right) - np.mean(left)))
        deltas = np.asarray(deltas)
        samples = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
        boot = deltas[samples].mean(axis=1)
        interval = np.quantile(boot, [0.0125, 0.9875])
        out[split] = {'scenes': len(pairs), 'mean_d_minus_a': float(deltas.mean()),
                      'ci_97_5_percent': [float(v) for v in interval]}
    return {'conditions': out,
            'positive_on_both_conditions': all(out[s]['ci_97_5_percent'][0] > 0 for s in SPLITS),
            'familywise_alpha': 0.05, 'resamples': resamples, 'seed': bootstrap_seed,
            'original_gate_passed': False}
