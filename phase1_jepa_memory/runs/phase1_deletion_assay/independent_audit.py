"""Separate scalar-only implementation: raw forks -> clustered effects/rankings."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bounds(values):
    a=np.array(values,dtype=float);gen=np.random.default_rng(0)
    means=a[gen.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
    return float(a.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))
def main(folder):
    folder=Path(folder);saved=json.loads((folder/'summary.json').read_text())
    assert digest(folder/'raw_deletions.csv')==saved['raw_sha256']
    rows=list(csv.DictReader((folder/'raw_deletions.csv').open()))
    assert len(rows)==55296
    keys=[(r['task'],r['seed'],r['split'],r['pair'],r['episode'],r['pool'],r['deleted'],r['horizon']) for r in rows]
    assert len(set(keys))==len(rows)
    for r in rows:
        par=set(map(int,r['parent'].split(';')));kept=set(map(int,r['kept'].split(';')))
        assert len(par)==3 and len(kept)==2 and par-kept=={int(r['deleted'])}
        assert np.isfinite(float(r['feature_mse'])) and np.isfinite(float(r['task_error']))
    comparisons=0
    for spec in saved['contrasts']:
        block=[r for r in rows if r['task']==spec['task'] and int(r['seed'])==spec['seed'] and r['split']==spec['split'] and int(r['horizon'])==spec['horizon']]
        pairs=sorted({r['pair'] for r in block});assert len(pairs)==128
        for metric,expected in spec['metrics'].items():
            cv=[];nv=[]
            for pair in pairs:
                rr=[r for r in block if r['pair']==pair]
                c=[float(r[metric]) for r in rr if r['deleted_role']==spec['role']]
                n=[float(r[metric]) for r in rr if r['deleted_role']=='nuisance']
                assert len(c)==4 and len(n)==(8 if spec['task']=='gate' else 4)
                cv.append(np.mean(c));nv.append(np.mean(n))
            cv=np.array(cv);nv=np.array(nv)
            observed=bounds(cv-nv);exp=expected['difference']
            assert np.allclose(observed,[exp['mean'],exp['low'],exp['high']],atol=2e-12,rtol=1e-9)
            reduction=(cv.mean()-nv.mean())/max(cv.mean(),1e-8)
            passed=bool(cv.mean()>=1e-8 and observed[1]>0 and reduction>=.10)
            assert passed==expected['passed'] and np.isclose(reduction,expected['relative_reduction']['mean'])
            comparisons+=1
    for spec in saved['rankings']:
        block=[r for r in rows if r['task']==spec['task'] and int(r['seed'])==spec['seed'] and r['split']==spec['split']]
        gain={}
        for pair in sorted({r['pair'] for r in block}):
            rr=[r for r in block if r['pair']==pair];values=[]
            for episode in sorted({r['episode'] for r in rr}):
              for pool in ('arrival','early_nuisance'):
                cell=[r for r in rr if r['episode']==episode and r['pool']==pool]
                select=[r for r in cell if int(r['horizon'])==spec['select_h']]
                best=min(float(r['feature_mse']) for r in select)
                chosen={r['deleted'] for r in select if float(r['feature_mse'])<=best+1e-8}
                target=[r for r in cell if int(r['horizon'])==spec['test_h']]
                values.append([np.mean([float(r[m]) for r in target])-np.mean([float(r[m]) for r in target if r['deleted'] in chosen]) for m in ('feature_mse','task_error')])
            gain[pair]=np.mean(values,axis=0)
        for j,metric in enumerate(('feature_mse','task_error')):
            observed=bounds([gain[p][j] for p in sorted(gain)]);exp=spec['gains'][metric]
            assert np.allclose(observed,[exp['mean'],exp['low'],exp['high']],atol=2e-12,rtol=1e-9)
            comparisons+=1
        assert spec['passed']==all(v['low']>0 for v in spec['gains'].values())
    for task,t in saved['tasks'].items():
        for split,s in t['scope'].items():
            checks=[x['passed'] for x in saved['contrasts']+saved['rankings'] if x['task']==task and x['split']==split]
            assert s['supported']==all(checks)
        assert t['ID_pilot_ready']==t['scope']['test_id']['supported']
    expected='READY_FOR_ID_BOUNDED_POLICY_PILOT' if all(t['ID_pilot_ready'] for t in saved['tasks'].values()) else 'JOINT_POLICY_STAGE_NOT_READY'
    assert expected==saved['decision'] and not saved['original_phase1_go']
    assets=json.loads((folder/'asset_ledger.json').read_text());assert digest(folder/'asset_ledger.json')==saved['asset_ledger_sha256']
    assert all(digest(k)==v for k,v in assets.items())
    proof=dict(status='INDEPENDENT_SCALAR_AUDIT_PASS',raw_rows=len(rows),independent_scalar_comparisons=comparisons,
        all_equal_capacity_and_pair_axis=True,assets_stable=True,summary_sha256=digest(folder/'summary.json'),
        raw_sha256=digest(folder/'raw_deletions.csv'),decision=expected,no_model_forward_or_training=True)
    (folder/'independent_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)

if __name__=='__main__':main(sys.argv[1])
