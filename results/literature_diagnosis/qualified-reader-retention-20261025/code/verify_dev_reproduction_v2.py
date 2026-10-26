"""User-authorized v2 DEV gate. No fitting, CONF access, or threshold selection.

Rejudge the preserved multihead replay; infer item_attention first256 once;
then replay all frozen students on DEV. Read the preregistered README first.
"""
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe
from path_config import REPO, RUN, XFER, AGG, RECORDS, ACTIONS, APPEARANCE_OUTPUT
from experiments.phase2 import fit_terminal_students as fts

RES = Path(__file__).resolve().parents[1]
OLD = REPO/'build/qualified-reader-conf-20261026/dev-preflight'
OUT = REPO/'build/qualified-reader-conf-20261026/dev-preflight-v2'
REPORT = OUT/'dev_reproduction_v2.json'
README_SHA = '4c0c5898376e32801323a6ce3e3dad24ad0e67accc6828376ecc30a2867c68d8'
FIELDS = ('medoid_sum', 'sensitivity', 'attention', 'eval_cost')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    Path(p).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def array_sha(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str((x.dtype.str, x.shape)).encode()+x.tobytes()).hexdigest()


def check_shapes(actual, archived):
    if actual.shape != archived.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError('Require matching nonempty N by 3 arrays')
    if not len(actual) or not np.isfinite(actual).all() or not np.isfinite(archived).all():
        raise ValueError('Nonfinite or empty numerical arrays')


def summarize_comparison(actual, archived, tolerance, eligible, old_drop, new_drop):
    delta = abs(actual-archived)
    over = delta > tolerance[:, None]
    mismatch = old_drop != new_drop
    unexempted = mismatch & ~eligible
    ratios = np.divide(delta, tolerance[:, None], out=np.zeros_like(delta), where=tolerance[:, None] > 0)
    zero_violations = (tolerance[:, None] == 0) & (delta != 0)
    return {'parents': len(actual), 'candidate_values': int(actual.size),
            'max_absolute_error': float(delta.max()),
            'max_error_over_allowed_tol': None if zero_violations.any() else float(ratios.max()),
            'zero_tolerance_nonzero_error_cells': int(zero_violations.sum()),
            'numerical_exceeding_cells': int(over.sum()),
            'numerical_exceeding_parent_indices': np.flatnonzero(over.any(1)).tolist(),
            'eligible_tie_parents': int(eligible.sum()),
            'actual_choice_mismatches': int(mismatch.sum()),
            'actual_tie_exemptions': int((mismatch & eligible).sum()),
            'unexempted_mismatches': int(unexempted.sum()),
            'unexempted_mismatch_parent_indices': np.flatnonzero(unexempted).tolist(),
            'passed': bool(not over.any() and not unexempted.any())}


def compare_rule(actual, archived, attention=False):
    check_shapes(actual, archived)
    tolerance = 1e-4*archived.max(1)
    ordered = np.sort(archived, axis=1)
    if attention:
        if (archived <= 0).any() or (actual <= 0).any():
            raise ValueError('Nonpositive attention: preregistration requires stopping')
        eligible = np.log(ordered[:, 1]/ordered[:, 0]) <= 1e-4
    else:
        if (tolerance < 0).any():
            raise ValueError('Negative distance tolerance')
        eligible = ordered[:, 1]-ordered[:, 0] <= tolerance
    result = summarize_comparison(actual, archived, tolerance, eligible, archived.argmin(1), actual.argmin(1))
    result['choice'] = 'argmin'
    result['numerical_tolerance'] = '1e-4 * max(archived row)'
    result['tie_eligibility'] = 'log(archived second_smallest / smallest) <= 1e-4' if attention else 'archived second_smallest - smallest <= tolerance'
    return result


def compare_student(actual, archived, new_drop, old_drop):
    check_shapes(actual, archived)
    if not np.array_equal(new_drop, actual.argmax(1)) or not np.array_equal(old_drop, archived.argmax(1)):
        raise ValueError('Saved or inferred student drop differs from logits.argmax')
    tolerance = 1e-4*np.maximum(1., abs(archived).max(1))
    ordered = np.sort(archived, axis=1)
    eligible = ordered[:, -1]-ordered[:, -2] <= tolerance
    result = summarize_comparison(actual, archived, tolerance, eligible, old_drop, new_drop)
    result['choice'] = 'argmax; independently checked against fit_terminal_students.predict_students and both saved arrays'
    result['numerical_tolerance'] = '1e-4 * max(1, max(abs(archived logits)))'
    result['tie_eligibility'] = 'archived largest - second_largest <= tolerance'
    return result


def compare_stage(reader, path, recs):
    with np.load(path, allow_pickle=False) as replay, np.load(RES/f'stageA/{reader}_development.npz', allow_pickle=False) as old:
        if replay['ids'].tolist() != [r['id'] for r in recs[:256]] or old['ids'].tolist() != [r['id'] for r in recs]:
            raise ValueError('Stage A DEV IDs differ')
        checks = {k: compare_rule(replay[k], old[k][:256], attention=k == 'attention') for k in FIELDS}
    return {'replay_path': str(path), 'replay_sha256': sha(path), 'archived_sha256': sha(RES/f'stageA/{reader}_development.npz'),
            'arrays': checks, 'passed': all(c['passed'] for c in checks.values())}


def infer_item_stage(st, recs):
    model = pipe.load_reader('item_attention', st)
    parsed = ast.parse((Path(__file__).parent/'run_main.py').read_text())
    defs = [n for n in parsed.body if isinstance(n, ast.FunctionDef)]
    ns = {'pipe': pipe, 'np': np, 'model': model, 'G': st.G}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(Path(__file__).parent/'run_main.py'), 'exec'), ns)
    acc = {k: [] for k in FIELDS}; start = time.time()
    for at in range(0, 256, 64):
        rr = recs[at:at+64]; caps = [st.capsule(r) for r in rr]
        Cr = ns['bank_coeffs'](caps, st.banks['reference'])
        medoid, sensitivity = ns['selectors'](Cr)
        acc['medoid_sum'].append(medoid); acc['sensitivity'].append(sensitivity)
        acc['attention'].append(pipe.pooled_attention(model, caps))
        Ce = ns['bank_coeffs'](caps, st.banks['evaluation'])
        resid = np.stack([st.target_residual(r, 'evaluation') for r in rr])
        acc['eval_cost'].append(ns['bank_losses'](Ce, resid).mean(1))
        print(json.dumps({'stage': 'DEV item_attention v2', 'parents': at+len(rr), 'seconds': round(time.time()-start, 2)}), flush=True)
    dest = OUT/'item_attention_development_first256.npz'
    np.savez_compressed(dest, ids=np.array([r['id'] for r in recs[:256]]), **{k: np.concatenate(v) for k, v in acc.items()})
    return dest


def tie_reference(recs):
    rows = []; sources = {}
    for reader in ('multihead', 'item_attention'):
        p = RES/f'stageA/{reader}_development.npz'; sources[reader] = sha(p)
        with np.load(p, allow_pickle=False) as z:
            if z['ids'].tolist() != [r['id'] for r in recs]:
                raise ValueError('Full DEV stageA IDs differ in robustness report')
            for method, field in [('full_sensitivity', 'sensitivity'), ('medoid', 'medoid_sum'), ('oracle', 'eval_cost')]:
                values = z[field]; tol = 1e-4*values.max(1)
                ties = values-values.min(1, keepdims=True) <= tol[:, None]
                if not np.isfinite(values).all() or (ties.sum(1) < 1).any():
                    raise ValueError('Invalid robustness row')
                correct = np.array([[d in r['semantic_correct_delete_set'] for d in range(3)] for r in recs])
                average_lost = ((~correct)*ties).sum(1)/ties.sum(1)
                exact_lost = (~correct)[np.arange(len(recs)), values.argmin(1)].astype(float)
                for task in 'RDS':
                    idx = [i for i,r in enumerate(recs) if r['task_family'] == task]
                    groups = defaultdict(list)
                    for i in idx: groups[recs[i]['family']].append(i)
                    exact_family = [float(exact_lost[g].mean()) for g in groups.values()]
                    average_family = [float(average_lost[g].mean()) for g in groups.values()]
                    argmin_rate, average_rate = float(np.mean(exact_family)), float(np.mean(average_family))
                    rows.append({'reader': reader, 'task': task, 'method': method, 'parents': len(idx), 'families': len(groups),
                                 'eligible_tie_parents': int((ties[idx].sum(1)>1).sum()),
                                 'tie_set_sizes': {str(k): int((ties[idx].sum(1)==k).sum()) for k in (1,2,3)},
                                 'argmin_lost_count': float(exact_lost[idx].sum()), 'tie_average_lost_count': float(average_lost[idx].sum()),
                                 'argmin_family_mean_lost_rate': argmin_rate, 'tie_average_family_mean_lost_rate': average_rate,
                                 'difference_tie_minus_argmin': average_rate-argmin_rate})
    return {'scope': 'read-only full DEV stageA; descriptive robustness, not CONF or changed formal decisions',
            'tolerance': 'value-min(value) <= 1e-4*max(value) within each parent',
            'source_sha256': sources, 'rows': rows, 'all_18_rates_unchanged': all(x['difference_tie_minus_argmin']==0 for x in rows)}


def main(report):
    if OUT.exists(): raise ValueError('Refuse to overwrite v2 output directory')
    if sha(RES/'conf/README.md') != README_SHA: raise ValueError('Preregistered README SHA differs before replay')
    OUT.mkdir(parents=True)
    shutil.copy2(RES/'conf/README.md', OUT/'README.preregistered.md')
    receipt = {'saved_utc': now(), 'readme_sha256': sha(OUT/'README.preregistered.md'), 'expected_readme_sha256': README_SHA,
               'verifier_source_sha256': sha(__file__), 'before_any_v2_comparison_or_inference': True}
    write(OUT/'preregistration_receipt.json', receipt)
    report['preregistration'] = receipt
    report['v1_report_preserved_sha256'] = sha(OLD/'dev_reproduction.raw.json')
    report['old_multihead_replay_sha256'] = sha(OLD/'multihead_development_first256.npz')
    report['versions'] = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}
    report['execution_settings'] = {'requested_torch_threads': 2, 'deterministic_algorithms': True,
                                    'environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
    report['source_sha256'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    report['frozen_student_source_sha256'] = sha(fts.__file__)
    if RECORDS.resolve() != (RUN/'data/records.json').resolve() or ACTIONS.resolve() != (RUN/'data/actions.npz').resolve():
        raise ValueError('DEV preflight accepts original DEV paths only')
    if XFER.resolve() != (REPO/'build/_claude_transfer_20261025').resolve():
        raise ValueError('DEV preflight accepts original projected cache only')
    st = pipe.Store()
    if any(r['split']=='confirmation' for r in st.records): raise ValueError('CONF must not be present in DEV Store')
    recs = [r for r in st.records if r['split']=='development']
    if len(recs)!=1728 or len({r['family'] for r in recs})!=24: raise ValueError('Unexpected DEV size')
    report['records'] = 1728; report['families'] = 24
    report['stageA'] = {}
    report['stageA']['multihead'] = compare_stage('multihead', OLD/'multihead_development_first256.npz', recs)
    report['stageA']['multihead']['reuse_without_new_reader_inference'] = True
    if not report['stageA']['multihead']['passed']: raise ValueError('Preserved multihead failed v2 gate')
    print(json.dumps({'stage':'preserved_multihead_rejudged','passed':True}),flush=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    report['execution_settings']['actual_torch_threads'] = torch.get_num_threads()
    item_path = infer_item_stage(st,recs)
    report['stageA']['item_attention'] = compare_stage('item_attention', item_path, recs)
    if not report['stageA']['item_attention']['passed']: raise ValueError('item_attention failed v2 gate')
    appearance_receipt = json.loads((OLD/'appearance.log').read_text().splitlines()[-1])
    if appearance_receipt['hash_mismatches']!=0: raise ValueError('Appearance failed RGB verification')
    report['appearance'] = {'receipt': appearance_receipt, 'path': str(APPEARANCE_OUTPUT), 'sha256': sha(APPEARANCE_OUTPUT)}
    with np.load(APPEARANCE_OUTPUT,allow_pickle=False) as app: app_by_id = dict(zip(app['ids'].tolist(),app['items']))
    feats = [st.student_features(r) for r in recs]
    inputs = {'items':np.stack([f[0] for f in feats]),'context':np.stack([f[1] for f in feats])}
    appearance = {'items':np.stack([app_by_id[r['id']] for r in recs]),'context':np.empty((len(recs),0),np.float32)}
    report['student_input_hashes'] = {k:array_sha(v) for k,v in inputs.items()}
    report['students'] = {}
    for reader in ('multihead','item_attention'):
        for mode in ('standard','distilled'):
            group = reader+'_'+mode; models = OLD/'students'/group
            preds = fts.predict_students(models,inputs,appearance if mode=='standard' else None)
            report['execution_settings']['actual_threads_after_'+group] = torch.get_num_threads()
            with np.load(models/'dev_predictions.npz',allow_pickle=False) as old:
                if old['ids'].tolist()!=[r['id'] for r in recs]: raise ValueError('Student DEV IDs differ')
                tests = {name:compare_student(v['logits'],old[name+'__logits'],v['drop'],old[name+'__drop']) for name,v in preds.items()}
            dest = OUT/(group+'_dev_predictions.npz')
            np.savez_compressed(dest,ids=np.array([r['id'] for r in recs]),**{k+'__drop':v['drop'] for k,v in preds.items()},**{k+'__logits':v['logits'] for k,v in preds.items()})
            report['students'][group] = {'arms':tests,'prediction_sha256':sha(dest),'manifest_sha256':sha(models/'manifest.json'),
                                       'archived_prediction_sha256':sha(models/'dev_predictions.npz'),'passed':all(v['passed'] for v in tests.values())}
            if not report['students'][group]['passed']: raise ValueError('Student v2 gate failed: '+group)
            print(json.dumps({'stage':'student_replay','group':group,'passed':True,'arms':len(tests)}),flush=True)
    robustness = tie_reference(recs)
    write(OUT/'dev_tie_robustness.json',robustness)
    report['tie_robustness'] = {'path':str(OUT/'dev_tie_robustness.json'),'sha256':sha(OUT/'dev_tie_robustness.json'),
                              'all_18_rates_unchanged':robustness['all_18_rates_unchanged']}
    report['status'] = 'passed'; report['confirmation_accessed'] = False


if __name__=='__main__':
    report = {'status':'running','started_utc':now(),'scope':'v2 DEV only; frozen models; no fitting or CONF access'}
    try:
        main(report)
    except Exception as e:
        report['status']='failed'; report['error']=repr(e)
        raise
    finally:
        report['ended_utc']=now()
        if OUT.exists() and not REPORT.exists(): write(REPORT,report)
        print(json.dumps({'status':report['status'],'report':str(REPORT)}),flush=True)
