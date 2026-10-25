"""Exact finite toy preflight for a terminal K=2 / three-candidate study.

No data generation for the visual experiment, model inference, fitting, or GPU
use occurs here. Forecasts are analytically specified conditional means under
independent fair bits. All arithmetic used for selection is rational/exact.

Two independent-score bounds are checked: all 75 weak orders over four local
(tag,payload) types, and all 720 total orders of six conditional (slot,tag)
scores. The latter permits arbitrary local payload, age, fixed nuisance and
common context, provided those are held fixed while all six nonconstant tag
patterns are equally weighted and the context does not reveal the pattern.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path


HALF = Fraction(1, 2)
ORDERS = tuple(itertools.permutations(range(3)))


def distance(a, b):
    """Mean squared feature distance; common dimension within each example."""
    assert len(a) == len(b) and len(a) > 0
    return sum((Fraction(x)-Fraction(y))**2 for x, y in zip(a, b))/len(a)


def choices(predictions, full):
    pairwise = [[distance(a, b) for b in predictions] for a in predictions]
    medoid_scores = [sum(row) for row in pairwise]
    sensitivity_scores = [distance(p, full) for p in predictions]
    # Python min returns the earliest chronological slot in exact ties.
    return {
        'medoid_drop': min(range(3), key=medoid_scores.__getitem__),
        'sensitivity_drop': min(range(3), key=sensitivity_scores.__getitem__),
        'medoid_scores': medoid_scores,
        'sensitivity_scores': sensitivity_scores,
        'pairwise_mse': pairwise,
        'medoid_min_set': [i for i in range(3) if medoid_scores[i] == min(medoid_scores)],
        'sensitivity_min_set': [i for i in range(3) if sensitivity_scores[i] == min(sensitivity_scores)],
    }


def visible_role_reader(items, task):
    values = {item['role']: item['payload'] for item in items}
    if task == 'redundancy':
        return [values.get('A', values.get('Aprime', HALF)), values.get('B', HALF)]
    if task == 'additive_decoy':
        return [values.get('A', HALF), values.get('B', HALF)]
    assert task == 'xor_visible_roles'
    return [values['A'] ^ values['B']] if 'A' in values and 'B' in values else [HALF]


def ideal_tasks():
    summaries, all_rows = {}, []
    for task in ('redundancy', 'additive_decoy', 'xor_visible_roles'):
        rows = []
        bit_values = itertools.product(range(2), repeat=2 if task == 'redundancy' else 3)
        for payload in bit_values:
            a, b = payload[:2]
            if task == 'redundancy':
                items = [{'role': 'A', 'payload': a}, {'role': 'Aprime', 'payload': a}, {'role': 'B', 'payload': b}]
            else:
                items = [{'role': 'A', 'payload': a}, {'role': 'B', 'payload': b}, {'role': 'D', 'payload': payload[2]}]
            for order in ORDERS:
                ordered = [items[i] for i in order]
                full = visible_role_reader(ordered, task)
                predictions = [visible_role_reader([x for j, x in enumerate(ordered) if j != drop], task) for drop in range(3)]
                outcome = choices(predictions, full)
                correct = [i for i, x in enumerate(ordered) if x['role'] in (('A', 'Aprime') if task == 'redundancy' else ('D',))]
                losses = [distance(p, full) for p in predictions]
                row = {'task': task, 'payloads': payload, 'order': order, 'visible_items': ordered,
                       'target_and_qualified_full_prediction': full, 'deleted_subset_predictions': predictions,
                       'semantic_correct_delete_set': correct, 'losses': losses, **outcome}
                for method in ('medoid', 'sensitivity'):
                    drop = outcome[method+'_drop']
                    row[method+'_preserves_necessary_evidence'] = drop in correct
                    row[method+'_regret'] = losses[drop]-min(losses)
                rows.append(row)
        summaries[task] = {
            'equally_weighted_cases': len(rows),
            'target_dimension': len(rows[0]['target_and_qualified_full_prediction']),
            'unique_targets': len({tuple(r['target_and_qualified_full_prediction']) for r in rows}),
            'methods': {method: {
                'preserves_necessary_evidence_count': sum(r[method+'_preserves_necessary_evidence'] for r in rows),
                'mean_regret': sum(r[method+'_regret'] for r in rows)/len(rows),
                'exact_tie_count': sum(len(r[method+'_min_set']) > 1 for r in rows),
            } for method in ('medoid', 'sensitivity')},
        }
        if task == 'xor_visible_roles':
            # Counterexample to "XOR makes independent eviction impossible":
            # visible local roles suffice to assign D the lowest fixed score.
            successes = 0
            for row in rows:
                score = [0 if item['role'] == 'D' else 1 for item in row['visible_items']]
                drop = min(range(3), key=score.__getitem__)
                successes += drop in row['semantic_correct_delete_set']
            summaries[task]['independent_visible_role_scorer_preserves_count'] = successes
        all_rows.extend(rows)
    assert summaries['redundancy']['methods']['medoid']['preserves_necessary_evidence_count'] == 24
    assert summaries['additive_decoy']['methods']['medoid']['preserves_necessary_evidence_count'] == 48
    assert summaries['xor_visible_roles']['methods']['medoid']['preserves_necessary_evidence_count'] == 0
    assert summaries['xor_visible_roles']['independent_visible_role_scorer_preserves_count'] == 48
    assert all(s['methods']['sensitivity']['preserves_necessary_evidence_count'] == s['equally_weighted_cases'] for s in summaries.values())
    return summaries, all_rows


def relational_reader(observed):
    """Only visible tag/payload inputs; no hidden role or omitted item."""
    for tag in (0, 1):
        values = [x['payload'] for x in observed if x['tag'] == tag]
        if len(values) == 2:
            return [values[0] ^ values[1]]
    return [HALF]


def weak_orders(n):
    # Normalize every real-valued scalar score vector to ranks 0..k-1.
    # Enumerating every surjection to those ranks covers ALL weak orders.
    return [r for r in itertools.product(range(n), repeat=n) if set(r) == set(range(max(r)+1))]


def relational_xor():
    rows, marginals = [], Counter()
    for pair_tag in (0, 1):
        for a, b, d in itertools.product((0, 1), repeat=3):
            canonical = [{'tag': pair_tag, 'payload': a}, {'tag': pair_tag, 'payload': b}, {'tag': 1-pair_tag, 'payload': d}]
            for order in ORDERS:
                observed = [canonical[i] for i in order]
                singleton = order.index(2)
                for slot, item in enumerate(observed):
                    marginals[(item['tag'], item['payload'], slot, slot == singleton)] += 1
                full = relational_reader(observed)
                assert full == [a ^ b]
                predictions = [relational_reader([x for j, x in enumerate(observed) if j != drop]) for drop in range(3)]
                outcome = choices(predictions, full)
                assert predictions[singleton] == full
                assert all(predictions[i] == [HALF] for i in range(3) if i != singleton)
                rows.append({'pair_tag_evaluator_only': pair_tag, 'observed_items': observed,
                             'canonical_order_evaluator_only': order, 'singleton_slot_evaluator_only': singleton,
                             'target': a ^ b, 'full_prediction': full, 'deleted_subset_predictions': predictions, **outcome})
    orders = weak_orders(4)
    assert len(orders) == 75 and len(rows) == 96
    ranking_results = []
    for rank in orders:
        correct = 0
        for row in rows:
            scores = [rank[2*x['tag']+x['payload']] for x in row['observed_items']]
            drop = min(range(3), key=scores.__getitem__)
            correct += drop == row['singleton_slot_evaluator_only']
        ranking_results.append({'rank_by_type_00_01_10_11': rank, 'correct_deletion_count': correct,
                                'correct_deletion_rate': Fraction(correct, len(rows))})
    best = max(r['correct_deletion_count'] for r in ranking_results)
    worst = min(r['correct_deletion_count'] for r in ranking_results)
    assert best == 48 and worst == 32
    marginal_rows = []
    for tag, bit in itertools.product((0, 1), repeat=2):
        singleton = sum(marginals[(tag, bit, slot, True)] for slot in range(3))
        total = singleton + sum(marginals[(tag, bit, slot, False)] for slot in range(3))
        assert singleton == 24 and total == 72
        for slot in range(3):
            assert marginals[(tag, bit, slot, True)] == 8
            assert marginals[(tag, bit, slot, False)] == 16
        marginal_rows.append({'tag': tag, 'payload': bit, 'singleton_occurrences': singleton,
                              'all_occurrences': total, 'probability_singleton_given_type': Fraction(singleton, total)})
    summary = {
        'equally_weighted_cases': 96, 'type_order': [[0, 0], [0, 1], [1, 0], [1, 1]],
        'independent_weak_orders_exhaustively_enumerated': len(orders),
        'independent_exact_tie_rule': 'earliest chronological slot',
        'best_independent_correct_count': best, 'best_independent_correct_rate': Fraction(best, 96),
        'worst_independent_correct_count': worst,
        'best_weak_order_count': sum(r['correct_deletion_count'] == best for r in ranking_results),
        'all_equal_scores_correct_count': next(r['correct_deletion_count'] for r in ranking_results if len(set(r['rank_by_type_00_01_10_11'])) == 1),
        'bound_scope': 'Only fixed scalar g(tag,payload) over four discrete visible types, no age, arbitrary context, continuous features or additional state.',
        'no_claim': 'Marginal role balance does not imply every independent ranker is random; argmin compares scalar scores.',
        'medoid_correct_count': sum(r['medoid_drop'] == r['singleton_slot_evaluator_only'] for r in rows),
        'qualified_full_sensitivity_correct_count': sum(r['sensitivity_drop'] == r['singleton_slot_evaluator_only'] for r in rows),
        'past_only_set_rule_correct_count': sum(next(i for i,x in enumerate(r['observed_items']) if sum(y['tag']==x['tag'] for y in r['observed_items']) == 1) == r['singleton_slot_evaluator_only'] for r in rows),
        'local_role_marginals': marginal_rows,
        'single_observation_information_about_target_bits': 0,
        'complete_pair_information_about_target_bits': 1,
        'note_information_values': 'Fair independent payload bits and deterministic XOR; analytic distribution facts, not estimated PID of learned features.',
    }
    assert summary['medoid_correct_count'] == 0
    assert summary['qualified_full_sensitivity_correct_count'] == summary['past_only_set_rule_correct_count'] == 96
    return summary, rows, ranking_results


def algebra_checks():
    # General formula for additive decoys. a and b are errors relative to the
    # complete forecast, in the same Euclidean feature coordinates.
    # S_A-S_D = ||a||^2-2<a,b>; S_B-S_D = ||b||^2-2<a,b>.
    # Mean-squared distance divides every term by the same feature dimension.
    general_checked = 0
    vectors = tuple(itertools.product((-1, 0, 1), repeat=2))
    for ea, eb in itertools.product(vectors, repeat=2):
        a = sum(x*x for x in ea)
        b = sum(x*x for x in eb)
        cross = sum(x*y for x,y in zip(ea, eb))
        rule = choices([[0, 0], ea, eb], [0, 0])
        assert (0 in rule['medoid_min_set']) == (2*cross <= min(a, b))
        assert rule['medoid_scores'][1]-rule['medoid_scores'][0] == Fraction(a-2*cross, 2)
        assert rule['medoid_scores'][2]-rule['medoid_scores'][0] == Fraction(b-2*cross, 2)
        general_checked += 1
    orthogonal = choices([[0, 0], [1, 0], [0, 2]], [0, 0])
    aligned = choices([[0, 0], [1, 0], [2, 0]], [0, 0])
    assert orthogonal['medoid_drop'] == 0 and aligned['medoid_drop'] == 1
    assert aligned['sensitivity_drop'] == 0
    # A wrong full-set anchor can make the sensitivity policy preserve error.
    # This is an unqualified-reference counterexample, not a Bayes reader.
    wrong_anchor = choices([[0], [HALF], [HALF]], [HALF])
    assert wrong_anchor['sensitivity_drop'] in (1, 2)
    return {
        'general_decoy_condition': 'With eA=p_dropA-p_dropD, eB=p_dropB-p_dropD, dropD is a medoid minimizer iff 2<eA,eB> <= min(||eA||^2,||eB||^2). Strict inequalities give uniqueness.',
        'exact_two_dimensional_vector_pairs_checked': general_checked,
        'action_marginalization': 'For a fixed action bank, apply the condition to expectations of the squared norms and inner product, with one deletion after averaging.',
        'orthogonal_example': {'forecast_order': ['dropD', 'dropA', 'dropB'], 'predictions': [[0, 0], [1, 0], [0, 2]], **orthogonal},
        'aligned_error_counterexample': {'forecast_order': ['dropD', 'dropA', 'dropB'], 'predictions': [[0, 0], [1, 0], [2, 0]], **aligned,
            'scope': 'Algebraic forecast geometry; not a claim these are the conditional means of the fair two-axis toy.'},
        'wrong_full_reference_counterexample': {'forecast_order': ['dropD', 'dropA', 'dropB'], 'true_target': 0, 'incorrect_full_reference': HALF, 'predictions': [[0], [HALF], [HALF]], **wrong_anchor,
            'scope': 'Demonstrates need to qualify full K+1 reader; sensitivity can preserve a wrong reference.'},
    }


def conditional_position_bound():
    """Condition on slot payloads/ages/nuisance and nonleaking common context.

    Each independent scorer can now assign arbitrary scores to six variables:
    (slot0,tag0),(slot0,tag1),...,(slot2,tag1). For the six equiprobable
    nonconstant tag patterns, successful deletion means picking the singleton.
    Every score assignment induces one of the 720 total rankings after an
    earliest-slot tie refinement. Thus enumerating all rankings is exhaustive.
    """
    patterns = [x for x in itertools.product((0, 1), repeat=3) if len(set(x)) == 2]
    histogram, examples = Counter(), []
    for order in itertools.permutations(range(6)):
        rank = {variable: i for i, variable in enumerate(order)}
        successes = 0
        for tags in patterns:
            singleton = next(slot for slot, tag in enumerate(tags) if tags.count(tag) == 1)
            drop = min(range(3), key=lambda slot: rank[2*slot+tags[slot]])
            successes += drop == singleton
        histogram[successes] += 1
        if successes == 3 and len(examples) < 3:
            examples.append(order)
    assert sum(histogram.values()) == 720 and max(histogram) == 3
    assert histogram == {1: 96, 2: 528, 3: 96}
    # Verify conditioning is legitimate in the exact relational dataset:
    # each of the eight fixed chronological payload triples has six equally
    # likely tag patterns. Original physical identities give multiplicity two.
    conditioned = Counter()
    for pair_tag in (0, 1):
        for a, b, d in itertools.product((0, 1), repeat=3):
            items = [(pair_tag, a), (pair_tag, b), (1-pair_tag, d)]
            for order in ORDERS:
                current = [items[i] for i in order]
                conditioned[(tuple(x[1] for x in current), tuple(x[0] for x in current))] += 1
    assert len(conditioned) == 8*6 and set(conditioned.values()) == {2}
    return {
        'score_variables': [[slot, tag] for slot in range(3) for tag in (0, 1)],
        'equal_weight_tag_patterns': patterns,
        'total_score_orders_enumerated': 720,
        'histogram_successful_patterns_per_ranking': dict(sorted(histogram.items())),
        'maximum_correct_patterns': 3,
        'patterns_per_condition': 6,
        'maximum_correct_fraction': HALF,
        'example_maximizing_order_variable_indices': examples,
        'payload_conditions_checked': 8,
        'each_condition_pattern_multiplicity': 2,
        'ties': 'Any numerical tie refined by earliest chronological slot induces a total order included among720; values within one slot never co-occur.',
        'covered': 'Arbitrary independent scalar g(common_context, local_observation, age, executed-action summary), conditional on fixed slot payloads, nuisance, ages and common context; no other-candidate information.',
        'required_balance': 'All six nonconstant tag patterns remain equiprobable given that fixed context/payload/nuisance condition; common context and auxiliary state must not reveal the repeated tag or singleton.',
        'not_covered': 'Pool-conditioned scores, recurrent/controller state encoding other tags, correlated styling or scene metadata revealing roles, or an unbalanced conditional tag distribution.',
        'interpretation': 'Conditional upper bound3/6 applies to each payload/context condition, hence at most50% overall; it is not a measured neural-network score.',
    }


def serializable(obj):
    if isinstance(obj, Fraction):
        return {'numerator': obj.numerator, 'denominator': obj.denominator, 'value': float(obj)}
    if isinstance(obj, dict):
        return {k: serializable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [serializable(v) for v in obj]
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ideal, ideal_rows = ideal_tasks()
    relational, relational_rows, ranks = relational_xor()
    summary = {
        'scope': 'Exact analytic design preflight, not a visual experiment, trained-reader result or independent confirmation.',
        'computation': {'arithmetic': 'fractions.Fraction for distances and selection', 'trained_models': 0, 'reader_model_calls': 0, 'GPU_jobs': 0, 'data_seeds': None},
        'assumptions': ['Independent fair payload bits.', 'Analytically correct conditional-mean reader when qualified.', 'K=2 retained items, three observed terminal candidates, one irreversible deletion.', 'No action variation in these symbolic cases; action-bank result is an algebraic extension only.', 'Squared feature distance and earliest-slot exact-tie rule.'],
        'ideal_tasks': ideal, 'relational_xor': relational, 'conditional_position_aware_independent_bound': conditional_position_bound(), 'algebra': algebra_checks(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['The visual reader must separately qualify both K and K+1 histories and its uncertainty geometry.', 'XOR has two target classes despite four input-bit combinations; four-way future classification would be invalid.', 'Relational tags make roles past-observable but do not establish any neural scorer training result.', 'The conditional50% bound covers age/context-dependent independent scores only under its explicit conditional exchangeability and no-other-candidate-information assumptions; the visual generator must check those assumptions.', 'Do not pool MSE across symbolic tasks with different target dimension or claim a complete PID estimator.'],
    }
    outputs = {'summary.json': summary, 'ideal_cases.json': ideal_rows, 'relational_cases.json': relational_rows, 'independent_weak_orders.json': ranks}
    for name, value in outputs.items():
        (args.output/name).write_text(json.dumps(serializable(value), indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'exact_preflight_passed', 'ideal_case_counts': {k: v['equally_weighted_cases'] for k,v in ideal.items()},
                      'relational_cases': len(relational_rows), 'all_weak_orders': len(ranks), 'best_independent_retention': '48/96',
                      'relational_medoid_retention': '0/96', 'qualified_full_sensitivity_retention': '96/96', 'model_calls': 0}))


if __name__ == '__main__':
    main()
