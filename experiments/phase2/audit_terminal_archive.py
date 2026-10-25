"""Read-only post-run audit of the terminal three-structure result archive.

All outputs go to --output, outside the extracted scientific archive. This
reproduces source/manifest, saved qualification arithmetic, coefficient-distance
and baseline checks, plus explicitly posthoc mechanism counts. No model calls,
training, new selection rules, or modification of frozen artifacts occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def audit_manifest(base, output, repository=None):
    package, bundle, protocol = (read(base/x) for x in ('package_manifest.json', 'source/bundle_manifest.json', 'run/protocol.json'))
    checks = []
    for name, entry in package['files'].items():
        path = base/name
        checks.append({'kind': 'package_member', 'path': name,
                       'passed': path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']})
    for name, digest in bundle['files'].items():
        checks.append({'kind': 'source_bundle', 'path': name, 'passed': sha(base/'source'/name) == digest})
    for name, digest in protocol['source_sha256'].items():
        checks.append({'kind': 'frozen_source', 'path': name, 'passed': sha(base/'source'/name) == digest})
    if repository is not None:
        names = list(bundle['files'])
        requests = ''.join(bundle['git_commit']+':'+name+'\n' for name in names).encode()
        process = subprocess.run(['git', 'cat-file', '--batch'], input=requests, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd=repository, check=True)
        data, offset = process.stdout, 0
        for name in names:
            end = data.index(b'\n', offset)
            header = data[offset:end].split(); offset = end+1
            if len(header) != 3 or header[1] != b'blob':
                raise ValueError('Frozen commit file is not a Git blob: '+name)
            length = int(header[2]); content = data[offset:offset+length]; offset += length+1
            checks.append({'kind': 'git_source', 'path': name, 'passed': hashlib.sha256(content).hexdigest() == bundle['files'][name]})
    config = base/'source/experiments/phase2/configs/terminal_structures_20261025.json'
    checks.append({'kind': 'frozen_config', 'passed': sha(config) == protocol['config_file_sha256'] and read(config) == protocol['config']})
    locked = read(base/'run/reader_locked.json')
    checks.append({'kind': 'reader_checkpoint', 'passed': sha(base/'run/reader_selected.pt') == locked['selected_sha256']})
    checks.append({'kind': 'reader_basis', 'passed': sha(base/'run/basis.npz') == locked['basis_sha256']})
    result = {'status': 'passed' if all(x['passed'] for x in checks) else 'failed', 'checks': len(checks),
              'failures': [x for x in checks if not x['passed']],
              'counts': {kind: sum(x['kind'] == kind for x in checks) for kind in sorted({x['kind'] for x in checks})},
              'git_commit': bundle['git_commit'], 'git_comparison_run': repository is not None, 'checked_files': checks,
              'missing_derived_files_declared_by_package': package['derived_data_preserved_on_scratch'],
              'source_scope': 'Archive/source hashes and optional Git binding checked. Declared large derived feature/RGB caches are not included; full feature-level reader replay is not claimed.'}
    write(output/'manifest_source_audit.json', result)
    if result['failures']:
        raise ValueError('Manifest/source checks failed')
    return result


def audit_qualification(base, output):
    run = base/'run'
    rows = read(run/'development/reader_qualification_rows.json')
    saved = read(run/'development/reader_qualification.json')
    config = read(run/'protocol.json')['config']
    records = {r['id']: r for r in read(run/'data/records.json')}
    count, failures = 0, []
    def check(ok, label):
        nonlocal count
        count += 1
        if not ok:
            failures.append(label)
    def mean(selected, key):
        return float(np.mean([r[key] for r in selected])) if selected else None
    for row in rows:
        distances = np.asarray(row['distances'])
        minima = np.flatnonzero(distances <= distances.min()+config['tolerance']).tolist()
        label = row['id']+'/'+row['arm']
        check(len(distances) == (2 if row['task_family'] == 'S' else 4), label+'/classes')
        check(row['min_set'] == minima and row['class'] == int(distances.argmin()) and row['strict_correct'] == (minima == [row['target_class']]), label+'/classification')
        check(abs(row['mse']-distances[row['target_class']]) < 1e-12, label+'/mse')
        if row['arm'].startswith(('native_', 'fixed_')):
            check(row['semantic_complete'] == (row['drop'] in records[row['id']]['semantic_correct_delete_set']), label+'/semantic')
    gates, summary = {}, {}
    for task in ('R', 'D', 'S'):
        task_rows = [r for r in rows if r['task_family'] == task]
        k2 = [r for r in task_rows if r['arm'].startswith(('native_', 'fixed_')) and r['semantic_complete']]
        flips = [r for r in task_rows if r['arm'].startswith('flip_')]
        full = [r for r in task_rows if r['arm'] == 'full']
        missing = [r for r in task_rows if r['arm'].startswith('fixed_') and not r['semantic_complete']]
        single = [r for r in task_rows if r['arm'].startswith('single_')]
        complete = [r for r in k2 if r['arm'].startswith('fixed_')]
        s = {'K2_complete_accuracy': mean(k2, 'strict_correct'), 'flip_accuracy': mean(flips, 'strict_correct'),
             'K3_accuracy': mean(full, 'strict_correct'), 'fixed_complete_mse': mean(complete, 'mse'),
             'fixed_missing_mse': mean(missing, 'mse'), 'single_mse': mean(single, 'mse'),
             'missing_tie_credit_accuracy': mean(missing, 'tie_fraction_credit'),
             'missing_midpoint_mse': mean(missing, 'midpoint_mse'),
             'missing_equal_distance_fraction': float(np.mean([len(r['min_set']) > 1 for r in missing]))}
        summary[task] = s
        for key, value in s.items():
            check(np.isclose(value, saved['summary'][task][key], rtol=0, atol=1e-12), task+'/'+key)
        gates[task+'/K2_content'] = s['K2_complete_accuracy'] >= config['qualification_accuracy'] and s['flip_accuracy'] >= config['qualification_accuracy'] and s['fixed_complete_mse'] < s['fixed_missing_mse']-config['qualification_mse_margin']
        gates[task+'/K3_content'] = s['K3_accuracy'] >= config['qualification_accuracy']
        for condition in sorted({r['condition'] for r in task_rows}):
            cell = [r for r in task_rows if r['condition'] == condition]
            complete = [r for r in cell if r['arm'].startswith('fixed_') and r['semantic_complete']]
            missing = [r for r in cell if r['arm'].startswith('fixed_') and not r['semantic_complete']]
            flips = [r for r in cell if r['arm'].startswith('flip_')]
            native = [r for r in cell if r['arm'].startswith('native_') and r['semantic_complete']]
            full = [r for r in cell if r['arm'] == 'full']
            roles = sorted({r['drop_role'] for r in complete})
            cm = [mean([r for r in complete if r['drop_role'] == role], 'mse') for role in roles]
            ca = [mean([r for r in complete if r['drop_role'] == role], 'strict_correct') for role in roles]
            mm = [mean([r for r in missing if r['drop_role'] == role], 'mse') for role in sorted({r['drop_role'] for r in missing})]
            good = all(x >= config['qualification_accuracy'] for x in ca) and mean(native, 'strict_correct') >= config['qualification_accuracy'] and bool(flips) and mean(flips, 'strict_correct') >= config['qualification_accuracy'] and all(x < y-config['qualification_mse_margin'] for x in cm for y in mm)
            gates[task+'/'+condition+'/K2_content'] = bool(good)
            gates[task+'/'+condition+'/K3_content'] = mean(full, 'strict_correct') >= config['qualification_accuracy']
            gates[task+'/K2_content'] = gates[task+'/K2_content'] and good
            gates[task+'/K3_content'] = gates[task+'/K3_content'] and gates[task+'/'+condition+'/K3_content']
    for key, value in gates.items():
        check(bool(value) == saved['gates'][key], key)
    result = {'status': 'passed' if not failures else 'failed', 'checks': count, 'failures': failures,
              'qualification_rows': len(rows), 'summary': summary, 'recomputed_gates': {k: bool(v) for k, v in gates.items()},
              'scope': 'Saved-distance classification/tie arithmetic and prespecified aggregate+condition gates; no new learned-reader inference or omitted-feature-cache reconstruction.'}
    write(output/'qualification_arithmetic_audit.json', result)
    if failures:
        raise ValueError('Qualification arithmetic mismatch')
    return result


def mechanism_diagnosis(base, output):
    development = base/'run/development'
    rows = read(development/'baseline_decisions.json')
    q = {(r['id'], r['arm']): r for r in read(development/'reader_qualification_rows.json')}
    diagnosed = []
    for row in rows:
        semantic = row['semantic_correct_delete_set']
        bad = [i for i in range(3) if i not in semantic]
        loss = np.asarray(row['expected_losses'])
        oracle, medoid, sensitivity = int(loss.argmin()), row['policies']['medoid']['drop'], row['policies']['full_sensitivity']['drop']
        distances = np.asarray(row['selection_details']['medoid']['pairwise_mse'])
        native_complete = [q[(row['id'], 'native_'+str(i))] for i in semantic]
        full = q[(row['id'], 'full')]
        record = {k: row[k] for k in ('id', 'family', 'task_family', 'condition', 'semantic_correct_delete_set')}
        record.update({'medoid_drop': medoid, 'full_sensitivity_drop': sensitivity, 'oracle_drop': oracle,
            'oracle_semantic_wrong': oracle not in semantic, 'medoid_semantic_wrong': medoid not in semantic,
            'full_sensitivity_semantic_wrong': sensitivity not in semantic,
            'all_semantic_complete_native_predictions_correct_actual_plan': all(x['strict_correct'] for x in native_complete),
            'any_semantic_complete_native_prediction_correct_actual_plan': any(x['strict_correct'] for x in native_complete),
            'selected_medoid_native_prediction_correct_actual_plan': q[(row['id'], 'native_'+str(medoid))]['strict_correct'],
            'full_native_prediction_correct_actual_plan': full['strict_correct'],
            'medoid_regret_eval32': row['policies']['medoid']['regret'],
            'full_sensitivity_regret_eval32': row['policies']['full_sensitivity']['regret'],
            'oracle_advantage_over_best_semantic_delete_eval32': float(loss[semantic].min()-loss.min()),
            'medoid_minus_best_semantic_loss_eval32': float(loss[medoid]-loss[semantic].min()),
            'pairwise_prediction_mse_reference16': distances.tolist()})
        if len(semantic) == 1:
            good = semantic[0]; b1, b2 = bad
            bb, gb = distances[b1, b2], [distances[good, b1], distances[good, b2]]
            record.update({'incomplete_pair_distance': float(bb), 'complete_to_incomplete_distances': list(map(float, gb)),
                'incomplete_pair_strictly_closest': bool(bb < min(gb)),
                'incomplete_pair_closer_than_at_least_one_complete_edge': bool(bb < max(gb)),
                'complete_prediction_medoid_strictly_optimal': bool(bb > max(gb))})
        else:
            good1, good2 = semantic; wrong = bad[0]
            gg, gb = distances[good1, good2], [distances[wrong, good1], distances[wrong, good2]]
            record.update({'complete_pair_distance': float(gg), 'complete_to_incomplete_distances': list(map(float, gb)),
                'complete_pair_strictly_closest': bool(gg < min(gb)),
                'incomplete_prediction_medoid_strictly_optimal': bool(gg > max(gb))})
        diagnosed.append(record)
    groups = {}
    for task in ('R', 'D', 'S'):
        for condition in [None]+sorted({r['condition'] for r in diagnosed if r['task_family'] == task}):
            selected = [r for r in diagnosed if r['task_family'] == task and (condition is None or r['condition'] == condition)]
            qualified = [r for r in selected if r['all_semantic_complete_native_predictions_correct_actual_plan']]
            failures = [r for r in selected if r['medoid_semantic_wrong']]
            semantic_oracle = [r for r in selected if not r['oracle_semantic_wrong']]
            result = {'records': len(selected), 'families': len({r['family'] for r in selected}),
                'medoid_semantic_losses': len(failures), 'oracle_semantic_losses': sum(r['oracle_semantic_wrong'] for r in selected),
                'all_complete_native_correct_actual_plan_records': len(qualified),
                'medoid_losses_with_all_complete_native_correct_actual_plan': sum(r['medoid_semantic_wrong'] for r in qualified),
                'medoid_regret_in_actual_plan_qualified_subset_evaluated_on_other32plans': float(np.mean([r['medoid_regret_eval32'] for r in qualified])) if qualified else None,
                'medoid_losses_with_eval32_oracle_semantic_correct': sum(r['medoid_semantic_wrong'] for r in semantic_oracle),
                'eval32_oracle_semantic_correct_records': len(semantic_oracle),
                'medoid_failures_still_classified_correct_on_actual_plan': sum(r['selected_medoid_native_prediction_correct_actual_plan'] for r in failures),
                'full_K3_actual_plan_correct_count': sum(r['full_native_prediction_correct_actual_plan'] for r in selected),
                'full_K3_wrong_while_all_complete_K2_correct_actual_plan': sum(not r['full_native_prediction_correct_actual_plan'] for r in qualified),
                'mean_oracle_advantage_over_best_semantic_delete_eval32': float(np.mean([r['oracle_advantage_over_best_semantic_delete_eval32'] for r in selected]))}
            if task != 'R':
                closest = [r for r in selected if r['incomplete_pair_strictly_closest']]
                result.update({'incomplete_pair_strictly_closest_count': len(closest),
                    'medoid_failure_among_incomplete_pair_closest_count': sum(r['medoid_semantic_wrong'] for r in closest),
                    'incomplete_pair_closer_than_at_least_one_complete_edge_count': sum(r['incomplete_pair_closer_than_at_least_one_complete_edge'] for r in selected),
                    'complete_prediction_medoid_strictly_optimal_count': sum(r['complete_prediction_medoid_strictly_optimal'] for r in selected),
                    'incomplete_pair_closest_and_complete_native_correct_actual_plan': sum(r['all_semantic_complete_native_predictions_correct_actual_plan'] for r in closest),
                    'mean_incomplete_pair_distance': float(np.mean([r['incomplete_pair_distance'] for r in selected])),
                    'mean_complete_to_incomplete_distance': float(np.mean([r['complete_to_incomplete_distances'] for r in selected]))})
            else:
                result.update({'complete_pair_strictly_closest_count': sum(r['complete_pair_strictly_closest'] for r in selected),
                    'incomplete_prediction_medoid_strictly_optimal_count': sum(r['incomplete_prediction_medoid_strictly_optimal'] for r in selected)})
            groups[task+'/'+(condition or 'all')] = result
    result = {'scope': 'Posthoc descriptive diagnosis of saved predictions, no new inference or selection. Prespecified gates retain their original status.',
        'input_permissions_and_plan_distinction': {
            'geometry': '16 TRAIN reference plans; no evaluation targets in selector',
            'regret': '32 independent evaluation plans, joint h3/h6 expected loss',
            'native_correct_subset': 'Qualification on recorded actual episode plan, not proof of correct classification on all32 evaluation plans',
            'warning': 'Native-correct filtering is posthoc description, not a new prespecified confirmation or causal effect estimate.'},
        'groups': groups, 'rows': diagnosed}
    write(output/'mechanism_diagnosis.json', result)
    return result


def baseline_and_coefficient_audit(base, output, audit_coefficients=True):
    path = base/'source/experiments/phase2/summarize_terminal_students.py'
    spec = importlib.util.spec_from_file_location('frozen_terminal_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Redirect every derived write. All archived scientific files remain untouched.
    module.write = lambda path, value: write(output/Path(path).name, value)
    return module.build_report(base/'run', 'development', audit=audit_coefficients, baseline_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extracted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repository', type=Path, help='Optional local Git checkout containing the frozen source commit')
    parser.add_argument('--skip-coefficients', action='store_true', help='Skip the optional saved-coefficient audit; do not imply it was rerun')
    args = parser.parse_args()
    base, output = args.extracted.resolve(), args.output.resolve()
    if output == base or base in output.parents:
        raise ValueError('Audit output must be outside the extracted scientific archive')
    output.mkdir(parents=True, exist_ok=True)
    manifest = audit_manifest(base, output, args.repository)
    qualification = audit_qualification(base, output)
    baseline = baseline_and_coefficient_audit(base, output, not args.skip_coefficients)
    mechanism = mechanism_diagnosis(base, output)
    result = {'status': 'passed', 'source_sha256': sha(__file__), 'new_model_calls': 0,
              'manifest_checks': manifest['checks'], 'qualification_checks': qualification['checks'],
              'baseline_parents': baseline['parents'], 'base_families': baseline['base_families'],
              'coefficient_audit': baseline['coefficient_audit'],
              'S_mechanism_summary': mechanism['groups']['S/all'],
              'scientific_archive_modified': False,
              'files': {p.name: sha(p) for p in sorted(output.glob('*.json')) if p.name != 'audit_execution.json'}}
    write(output/'audit_execution.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'manifest_checks', 'qualification_checks', 'baseline_parents', 'base_families', 'coefficient_audit')}))


if __name__ == '__main__':
    main()
