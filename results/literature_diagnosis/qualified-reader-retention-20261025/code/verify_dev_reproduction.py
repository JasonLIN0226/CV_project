"""Fail-closed DEV-only reproduction gate; no CONF generation/evaluation or fitting.

Run after appearance.py with default DEV paths. The run_main numerical functions
are compiled directly from its AST, without executing its top-level split loop.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe
from path_config import REPO, RUN, XFER, AGG, RECORDS, ACTIONS, APPEARANCE_OUTPUT
from experiments.phase2 import fit_terminal_students as fts

RES = Path(__file__).resolve().parents[1]
OUT = REPO / 'build/qualified-reader-conf-20261026/dev-preflight'
REPORT = OUT / 'dev_reproduction.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def numerical_functions(path):
    parsed = ast.parse(Path(path).read_text())
    return {n.name: ast.dump(n, include_attributes=False) for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def compare(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Array shape/finite mismatch')
    delta = abs(actual - expected)
    relative = np.divide(delta, abs(expected), out=np.full_like(delta, np.inf), where=expected != 0)
    relative[(expected == 0) & (actual == 0)] = 0
    old, new = expected.argmin(1), actual.argmin(1)
    mismatch = np.flatnonzero(old != new)
    non_tie = [int(i) for i in mismatch if not (
        expected[i, new[i]] - expected[i, old[i]] < 1e-6 and
        actual[i, old[i]] - actual[i, new[i]] < 1e-6)]
    return {'max_abs_error': float(delta.max()), 'max_relative_error': float(relative.max()),
            'relative_tolerance': 1e-4, 'argmin_mismatches': len(mismatch),
            'tie_exemptions': len(mismatch) - len(non_tie), 'non_tie_mismatch_indices': non_tie,
            'passed': bool(relative.max() <= 1e-4 and not non_tie)}


def main(report):
    if REPORT.exists():
        raise ValueError('Refuse to overwrite existing DEV gate report')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if RECORDS.resolve() != (RUN/'data/records.json').resolve() or ACTIONS.resolve() != (RUN/'data/actions.npz').resolve():
        raise ValueError('DEV preflight accepts original DEV paths only')
    if XFER.resolve() != (REPO/'build/_claude_transfer_20261025').resolve():
        raise ValueError('DEV preflight accepts original projected cache only')
    report['paths'] = {k: str(v) for k, v in dict(REPO=REPO,RUN=RUN,XFER=XFER,AGG=AGG,RECORDS=RECORDS,ACTIONS=ACTIONS).items()}
    report['versions'] = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}
    report['source_sha256'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    report['frozen_student_source_sha256'] = sha(fts.__file__)
    report['numerical_function_ast_matches'] = {}
    for name in ('pipe.py', 'run_main.py', 'evaluate_dev.py'):
        current = numerical_functions(Path(__file__).parent/name)
        original = numerical_functions(OUT/'original_code'/name)
        # Store changes only its external file paths; all data/reader math is unchanged.
        if name == 'pipe.py':
            current.pop('Store'); original.pop('Store')
        match = current == original
        report['numerical_function_ast_matches'][name] = match
        if not match:
            raise ValueError('Numerical function changed: '+name)
    st = pipe.Store()
    if any(r['split'] == 'confirmation' for r in st.records):
        raise ValueError('CONF must not be present in DEV Store')
    recs = [r for r in st.records if r['split'] == 'development']
    if len(recs) != 1728:
        raise ValueError('Unexpected DEV record count')
    app = np.load(APPEARANCE_OUTPUT, allow_pickle=False)
    app_by_id = dict(zip(app['ids'].tolist(), app['items']))
    feats = [st.student_features(r) for r in recs]
    inputs = {'items': np.stack([f[0] for f in feats]), 'context': np.stack([f[1] for f in feats])}
    appearance = {'items': np.stack([app_by_id[r['id']] for r in recs]), 'context': np.empty((len(recs), 0), np.float32)}
    report['records'] = len(recs); report['families'] = len({r['family'] for r in recs})
    report['readers'] = {}
    tree = ast.parse((Path(__file__).parent/'run_main.py').read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    compiled = compile(ast.Module(body=defs, type_ignores=[]), str(Path(__file__).parent/'run_main.py'), 'exec')
    for reader in ('multihead', 'item_attention'):
        start = time.time(); model = pipe.load_reader(reader, st)
        ns = {'pipe': pipe, 'np': np, 'model': model, 'G': st.G}
        exec(compiled, ns)
        acc = {k: [] for k in ('medoid_sum', 'sensitivity', 'attention', 'eval_cost')}
        for at in range(0, 256, 64):
            rr = recs[at:at+64]; caps = [st.capsule(r) for r in rr]
            Cr = ns['bank_coeffs'](caps, st.banks['reference'])
            medoid, sensitivity = ns['selectors'](Cr)
            acc['medoid_sum'].append(medoid); acc['sensitivity'].append(sensitivity)
            acc['attention'].append(pipe.pooled_attention(model, caps))
            Ce = ns['bank_coeffs'](caps, st.banks['evaluation'])
            resid = np.stack([st.target_residual(r, 'evaluation') for r in rr])
            acc['eval_cost'].append(ns['bank_losses'](Ce, resid).mean(1))
            print(json.dumps({'stage': 'DEV reproduction', 'reader': reader, 'parents': at+len(rr), 'seconds': round(time.time()-start, 1)}), flush=True)
        arrays = {k: np.concatenate(v) for k, v in acc.items()}
        path = OUT/f'{reader}_development_first256.npz'
        np.savez_compressed(path, ids=np.array([r['id'] for r in recs[:256]]), **arrays)
        archived = np.load(RES/f'stageA/{reader}_development.npz', allow_pickle=False)
        if archived['ids'].tolist() != [r['id'] for r in recs]:
            raise ValueError('DEV stageA IDs differ')
        row = {'stageA': {k: compare(v, archived[k][:256]) for k, v in arrays.items()},
               'recomputed_sha256': sha(path), 'archived_sha256': sha(RES/f'stageA/{reader}_development.npz'), 'students': {}}
        report['readers'][reader] = row
        if not all(x['passed'] for x in row['stageA'].values()):
            raise ValueError('Stage A numerical or argmin mismatch for '+reader)
        for mode in ('standard', 'distilled'):
            models = OUT/'students'/f'{reader}_{mode}'
            predictions = fts.predict_students(models, inputs, appearance if mode == 'standard' else None)
            old = np.load(models/'dev_predictions.npz', allow_pickle=False)
            if old['ids'].tolist() != [r['id'] for r in recs]:
                raise ValueError('Student DEV IDs differ')
            tests = {}
            for name, v in predictions.items():
                bad = np.flatnonzero(v['drop'] != old[name+'__drop'])
                tests[name] = {'drop_mismatches': len(bad), 'mismatch_indices': bad.tolist(),
                               'max_logit_abs_error': float(np.max(abs(v['logits']-old[name+'__logits'])))}
            row['students'][mode] = tests
            np.savez_compressed(OUT/f'{reader}_{mode}_dev_predictions.npz', ids=np.array([r['id'] for r in recs]),
                                **{k+'__drop': v['drop'] for k,v in predictions.items()},
                                **{k+'__logits': v['logits'] for k,v in predictions.items()})
            if any(v['drop_mismatches'] for v in tests.values()):
                raise ValueError('Student drop mismatch for '+reader+' '+mode)
            print(json.dumps({'reader': reader, 'mode': mode, 'all_dev_student_drops_match': True, 'arms': len(tests)}), flush=True)
        row['seconds'] = time.time()-start
    report['status'] = 'passed'


if __name__ == '__main__':
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'DEV only; no fitting or confirmation access'}
    try:
        main(report)
    except Exception as e:
        report['status'] = 'failed'; report['error'] = repr(e)
        raise
    finally:
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        if not REPORT.exists():
            REPORT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status': report['status'], 'report': str(REPORT)}), flush=True)
