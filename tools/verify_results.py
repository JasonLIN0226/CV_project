#!/usr/bin/env python3
"""Recompute published pair-level statistics on CPU without models or HPC access."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSAY=ROOT/'results/deletion'

def load(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_manifest():
    manifest=load(ROOT/'results/manifest.json')
    for item in manifest['files']:
        p=ROOT/item['path']
        assert p.is_file(),f'Missing published file: {item["path"]}'
        assert digest(p)==item['sha256'],f'Changed published source/result: {item["path"]}'
    for directory in ('experiments/predictor/control', 'experiments/predictor/independent_audit', 'experiments/deletion'):
        base=ROOT/directory
        for name,expected in load(base/'source_lock.json')['files'].items():
            assert digest(base/name)==expected,f'Frozen scientific source changed: {directory}/{name}'
    return len(manifest['files'])

def main(skip_manifest=False):
    files=None if skip_manifest else check_manifest()
    summary=load(ASSAY/'summary.json');audit=load(ASSAY/'independent_audit.json')
    assert audit['status']=='INDEPENDENT_SCALAR_AUDIT_PASS'
    assert digest(ASSAY/'summary.json')==audit['summary_sha256']
    assert summary['raw_rows']==audit['raw_rows']==55296
    assert not summary['original_phase1_go'] and not summary['unseen_confirmation']
    assert not summary['policy_learnability_proven'] and summary['GPU_seconds']==0
    rows=[]
    for p in sorted(ASSAY.glob('pair_contrasts_*.csv')):
        rows.extend(csv.DictReader(p.open()))
    assert len(rows)==11520
    identity=lambda r:(r['task'],r['seed'],r['split'],r['pair'],r['kind'],r['horizon'],r['role'])
    assert len({identity(r) for r in rows})==len(rows)
    for r in rows:
        assert np.isfinite([float(r[k]) for k in ('critical_feature','nuisance_feature','critical_task','nuisance_task')]).all()
    indices=np.random.default_rng(0).integers(0,128,size=(10000,128))
    def bounds(a):
        a=np.asarray(a,dtype=np.float64);assert a.shape==(128,)
        means=a[indices].mean(axis=1)
        return [float(a.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))]
    def select(spec,kind,horizon,role):
        out=[r for r in rows if r['task']==spec['task'] and int(r['seed'])==spec['seed'] and r['split']==spec['split'] and r['kind']==kind and int(r['horizon'])==horizon and r['role']==str(role)]
        out.sort(key=lambda r:r['pair']);assert len(out)==128
        assert len({r['pair'] for r in out})==128
        return out
    comparisons=0
    for spec in summary['contrasts']:
        chosen=select(spec,'critical_vs_nuisance',spec['horizon'],spec['role']);passed=[]
        for metric,col in [('feature_mse','feature'),('task_error','task')]:
            critical=np.array([float(r['critical_'+col]) for r in chosen]);nuisance=np.array([float(r['nuisance_'+col]) for r in chosen])
            observed=bounds(critical-nuisance);expected=spec['metrics'][metric]
            assert np.allclose(observed,[expected['difference'][k] for k in ('mean','low','high')],atol=2e-12,rtol=1e-9)
            assert np.isclose(critical.mean(),expected['critical_mean']) and np.isclose(nuisance.mean(),expected['nuisance_mean'])
            reduction=(critical.mean()-nuisance.mean())/max(critical.mean(),1e-8)
            flag=bool(critical.mean()>=1e-8 and reduction>=.10 and observed[1]>0)
            assert flag==expected['passed'] and np.isclose(reduction,expected['relative_reduction']['mean'])
            passed.append(flag);comparisons+=1
        assert all(passed)==spec['passed']
    for spec in summary['rankings']:
        chosen=select(spec,'cross_horizon',spec['test_h'],spec['select_h']);flags=[]
        # The critical_* columns hold pair-aggregated random-minus-chosen gains here.
        for metric,col in [('feature_mse','feature'),('task_error','task')]:
            observed=bounds([float(r['critical_'+col]) for r in chosen]);expected=spec['gains'][metric]
            assert np.allclose(observed,[expected[k] for k in ('mean','low','high')],atol=2e-12,rtol=1e-9)
            flags.append(observed[1]>0);comparisons+=1
        assert all(flags)==spec['passed']
    assert comparisons==180
    counts={}
    for task in ('gate','motion'):
        contrasts=[r for r in summary['contrasts'] if r['task']==task]
        ranks=[r for r in summary['rankings'] if r['task']==task]
        counts[task]={'critical_pass':sum(r['passed'] for r in contrasts),'critical_total':len(contrasts),'ranking_pass':sum(r['passed'] for r in ranks),'ranking_total':len(ranks)}
    assert counts['gate']==dict(critical_pass=6,critical_total=18,ranking_pass=18,ranking_total=18)
    assert counts['motion']==dict(critical_pass=36,critical_total=36,ranking_pass=18,ranking_total=18)
    outcome=load(ROOT/'results/predictor/evaluation/job_7575/outcome.json')
    assert outcome['status']=='DEVELOPMENT_REJECTED' and not outcome['phase1_go'] and not outcome['confirmation_opened']
    for task,expected in [('gate',(50,32)),('motion',(35,8))]:
        t=outcome['tasks'][task]
        assert len(t['original_checks'])==51 and sum(x['passed'] for x in t['original_checks'])==expected[0]
        assert len(t['fixed_reference_checks'])==32 and sum(x['passed'] for x in t['fixed_reference_checks'])==expected[1]
        counts[task]['original_checks_pass']=expected[0];counts[task]['fixed_checks_pass']=expected[1]
    result={'status':'PUBLISHED_PAIR_LEVEL_VERIFICATION_PASS','source_files_verified':files,'compact_pair_rows':len(rows),'independently_recomputed_statistics':comparisons,'counts':counts,'original_phase1_go':False,'joint_policy_stage_ready':False,'new_model_forward':False,'new_training':False,'new_GPU_seconds':0}
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skip-manifest',action='store_true');main(parser.parse_args().skip_manifest)
