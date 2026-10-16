#!/usr/bin/env python3
"""One predeclared Gate common-anchor trial; never launches jobs or confirmation."""
import argparse
import copy
from pathlib import Path

from phase1.cache import load_index
from phase1.cli import preflight
from phase1.config import prepare
from phase1.evaluation import evaluate
from phase1.experiment import prepare_shared, import_basis, _link, _copy_verified
from phase1.reporting import decision, figures, _development_artifacts
from phase1.training import train
from phase1.utils import read_json, write_json, file_hash, seed_all, environment
from phase1.v4_adoption import comparison
from phase1.v4_pipeline import (GateTrainingAuthorization, bind_gate_authorization,
    gate_all_models_completion, require_gate_evaluation_ready)


def arm_report(c):
    root = Path(c['output'])
    _, metrics, raw, audit, probes = _development_artifacts(root, 'gate')
    status, reasons, checks = decision(c, audit, probes, metrics)
    figures(c, probes, metrics, raw)
    write_json(root/'decision.json', {'status':status,'reasons':reasons,'checks':checks,
        'preregistered_rules':c['decision'],'experiment':c['experiment'],'phase1_go':False})
    lines=['# V4 Gate '+c['experiment']['role']+' development', '',
        '**'+status+'**; original arm rules only, not adoption or independent Phase 1 GO.', '',
        'Common-anchor weight '+str(c['experiment']['common_anchor_weight'])+
        '; original MSE + 2 paired MSE + .5 ranking. Teacher uses stopped current-parameter recent prediction.', '',
        '| Seed | Split | h | Recent MSE | Full-history MSE | Privileged MSE | Random MSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    conditions=['recent_only','full_history','privileged_small_budget','random_small_budget']
    for seed in c['training']['seeds']:
        for split in ['test_id','test_long_gap','test_distractor_shift']:
            for h in c['data']['horizons']:
                part=[r for r in metrics['per_seed'] if r['seed']==seed and r['task']=='gate' and r['split']==split and r['horizon']==h]
                values=[next(r['metrics']['normalized_mse']['mean'] for r in part if r['condition']==condition) for condition in conditions]
                lines.append(f'| {seed} | {split} | {h} | '+' | '.join(f'{v:.8f}' for v in values)+' |')
    lines += ['', 'Per-seed paired confidence intervals and task/wrong-content checks are retained in evaluation/metrics.json and decision.json.', '',
        'Training: all three 120-epoch models complete; earliest minimum of equally weighted four-condition validation MSE selects each checkpoint.',
        'Actual total and discarded teacher forwards are reported in each predictor/gate/seed_*/training.json. Matched adoption checks compare effective completed-epoch compute.']
    if reasons: lines += ['', 'Original-rule failures:']+['- '+r for r in reasons]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return status


def run(source_output, frozen_v3_root, frozen_source_directory, diagnostic_root, output, protocol):
    source_output, frozen_v3_root, diagnostic_root, root = map(Path,
        (source_output, frozen_v3_root, diagnostic_root, output))
    declaration = read_json(protocol)
    if (declaration['protocol_version'] != 4 or declaration['training_seeds'] != [17, 29, 43]
            or declaration['epochs'] != 120 or declaration['gate_conditional_candidate']['coefficient'] != 1
            or declaration['gate_conditional_candidate']['original_ranking_weight'] != .5
            or declaration['gate_conditional_candidate']['matched_teacher_forwards'] is not True):
        raise ValueError('Gate trial differs from the predeclared fixed single-factor experiment')
    if file_hash(diagnostic_root/'gate_common.json') != declaration['prerequisite_evidence']['gate_common_sha256']:
        raise ValueError('Gate prerequisite diagnostic changed after preregistration')
    authorization = GateTrainingAuthorization(str(diagnostic_root/'gate_common.json'), str(frozen_v3_root),
        str(source_output), str(frozen_source_directory), str(diagnostic_root/'gate_training_authorization.json'),
        str(diagnostic_root/'gate_pair_calibration.json'))
    base = prepare_shared(source_output, root/'shared')
    base['device'], base['threads'] = 'cpu', 4
    if not preflight(base)['passed']:
        raise RuntimeError('Allocated CPU preflight failed')
    configs = {}
    for role in ('reference', 'candidate'):
        c = copy.deepcopy(base)
        c['output'] = str((root/'gate'/role).resolve())
        c['experiment'] = {'protocol_version':4, 'role':role, 'stage':'development', 'tasks':['gate'],
            'variant':'reference' if role=='reference' else 'gate_common_anchor_v1',
            'common_anchor_weight':0 if role=='reference' else 1, 'matched_teacher_forwards':True,
            'protocol_sha256':file_hash(protocol)}
        c = bind_gate_authorization(c, authorization)
        prepare(c)
        for folder in ('dataset', 'features'):
            _link(Path(base['output'])/folder, Path(c['output'])/folder)
        import_basis(base, c, 'gate', load_index(base))
        # Gate input/target readouts are unchanged, already fitted on train only.
        # Reuse their actual bytes, including provenance, rather than refitting.
        probes = frozen_v3_root/'gate/reference/probes'
        for p in probes.iterdir():
            if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.npz'}:
                _copy_verified(p, Path(c['output'])/'probes'/p.name)
        for image in (Path(base['output'])/'figures').glob('pair_gate_*.png'):
            _copy_verified(image, Path(c['output'])/'figures'/image.name)
        write_json(Path(c['output'])/'probe_reuse.json', {'source':str(probes),
            'metrics_sha256':file_hash(probes/'metrics.json'), 'byte_identical':True,
            'scope':'Original train-only Gate evidence and target readouts; unchanged input and target contract'})
        configs[role] = c
    write_json(root/'environment.json', environment('cpu'))
    write_json(root/'preregistered_protocol.json', declaration)
    for role, c in configs.items():
        seed_all(c['training']['seeds'][0], c['threads'])
        result = train(c, v4_authorization=authorization)
        if result['interrupted']:
            return {'status':'INCOMPLETE','interrupted':True,'role':role,'phase1_go':False}
    # All six locked development models complete before any held-out arm evaluation.
    all_complete = gate_all_models_completion(configs['reference'], configs['candidate'])
    if not all_complete['complete']:
        raise ValueError('All six actual fixed-budget Gate models must complete before evaluation')
    write_json(root/'gate/all_models_completion.json', all_complete)
    for role, c in configs.items():
        write_json(Path(c['output'])/'training_completion.json', all_complete['roles'][role])
    for c in configs.values():
        require_gate_evaluation_ready(c, all_complete)
        evaluate(c)
        arm_report(c)
    result = comparison(root/'gate/reference', root/'gate/candidate', 'gate',
        gate_diagnostic=diagnostic_root/'gate_common.json', frozen_v3_root=frozen_v3_root,
        frozen_feature_root=source_output, frozen_source_directory=frozen_source_directory,
        pair_export=diagnostic_root/'gate_pair_calibration.json', output=root/'gate')
    lines=['# V4 Gate common-anchor development', '', '**'+result['status']+'**; development evidence, not Phase 1 GO.', '',
        'Only the common-anchor weight changes: reference 0 versus candidate 1. Both arms execute the same stopped current-parameter recent teacher forwards.',
        'Original ranking .5, seeds 17/29/43, 120 epochs, source data/features/bases, and equal validation-MSE checkpoint selection are preserved.', '',
        'All original task rules, fixed-reference recent checks, and critical absolute-MSE checks apply.']
    lines += ['', '- '+reason] if (reason := '; '.join(result.get('reasons',[]))) else []
    lines += ['', 'Motion remains an independent pending comparison; this report cannot authorize confirmation.']
    (root/'gate/development_comparison.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'decision.json', {'stage':'development','gate':result['status'],'motion':'INCOMPLETE',
        'confirmation_ready':False,'phase1_go':False})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-output',required=True)
    p.add_argument('--frozen-v3-root',required=True)
    p.add_argument('--frozen-source-directory',required=True)
    p.add_argument('--diagnostic-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--protocol',required=True)
    a=p.parse_args()
    result=run(a.source_output,a.frozen_v3_root,a.frozen_source_directory,a.diagnostic_root,a.output,a.protocol)
    print('V4 Gate development: '+str(result.get('status')),flush=True)
    return 143 if result.get('interrupted') else 0


if __name__=='__main__':
    raise SystemExit(main())
