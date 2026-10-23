"""Independent oracle choices, all paired intervals and cached vector identity."""
import collections,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PARENT=Path('/scratch/xl4358/phase1_motion_locked_test/output/job_8349')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main(out):
    out=Path(out);s=read(out/'summary.json');raw=rows(out/'raw_scoring.csv');paired=rows(out/'paired_scoring.csv')
    assert len(raw)==82944 and len(paired)==2304 and sha(out/'raw_scoring.csv')==s['raw_sha256'] and sha(out/'paired_scoring.csv')==s['paired_sha256']
    parentrows=rows(PARENT/'raw_deletions.csv');old={(r['seed'],r['split'],r['episode'],r['pool'],r['deleted'],r['horizon']):r for r in parentrows};groups=collections.defaultdict(list);scoremax=0.
    for r in raw:
        key=tuple(r[k] for k in ('seed','split','episode','pool','deleted','horizon'));original=old[key]
        assert all(r[k]==v for k,v in original.items())
        delta=[float(r['pred_'+a])-float(r['target_decoded_'+a]) for a in ('x','y')];score=sum(x*x for x in delta)/2
        scoremax=max(scoremax,abs(score-float(r['decoded_score'])));assert scoremax<1e-10
        task=sum((float(r['pred_'+a])-float(r['true_'+a]))**2 for a in ('x','y'))/2;assert abs(task-float(r['task_error']))<2e-6
        groups[r['split'],int(r['seed']),r['pair'],r['episode'],r['pool'],int(r['horizon'])].append(r)
    assert len(groups)==27648 and all(len(rr)==3 for rr in groups.values())
    collected=collections.defaultdict(list)
    for (sp,seed,pair,ep,pool,h),rr in groups.items():
        rr=sorted(rr,key=lambda r:int(r['deleted']));other=sorted(groups[sp,seed,pair,ep,pool,9-h],key=lambda r:int(r['deleted']))
        f=[float(r['feature_mse']) for r in rr];t=[float(r['task_error']) for r in rr];best=min(t);taskbest=[i for i in range(3) if t[i]-best<=1e-12];v={}
        for name,ix in [('uniform',[0,1,2]),('fifo',[0]),('reject',[2]),('task_oracle',taskbest)]:
            v[name+'_feature_mse']=sum(f[i] for i in ix)/len(ix);v[name+'_task_error']=sum(t[i] for i in ix)/len(ix)
        for score_kind in ('full','decoded'):
            for selection in ('same','transfer'):
                source=rr if selection=='same' else other;field='feature_mse' if score_kind=='full' else 'decoded_score';tol=1e-8 if score_kind=='full' else 1e-12
                cost=[float(r[field]) for r in source];ix=[i for i in range(3) if cost[i]-min(cost)<=tol];name=score_kind+'_'+selection
                v[name+'_feature_mse']=sum(f[i] for i in ix)/len(ix);v[name+'_task_error']=sum(t[i] for i in ix)/len(ix)
                v[name+'_task_optimal']=sum(i in taskbest for i in ix)/len(ix);v[name+'_task_regret']=v[name+'_task_error']-best
        collected[sp,seed,pair,h].append(v)
    maxpaired=0.
    for r in paired:
        vv=collected[r['split'],int(r['seed']),r['pair'],int(r['horizon'])];assert len(vv)==12
        maxpaired=max(maxpaired,max(abs(sum(v[k] for v in vv)/12-float(r[k])) for k in vv[0]))
    assert maxpaired<1e-12
    statistics=s['statistics'];maxstat=0.;n=0;ix=np.random.default_rng(0).integers(0,128,(10000,128));weights=np.stack([np.bincount(a,minlength=128) for a in ix])/128
    def vector(spec,column):
        seeds=(17,29,43) if spec['seed']=='mean' else (int(spec['seed']),);ids=sorted({k[2] for k in collected if k[0]==spec['split'] and k[3]==spec['horizon']});assert len(ids)==128
        return np.array([sum(sum(v[column] for v in collected[spec['split'],seed,p,spec['horizon']])/12 for seed in seeds)/len(seeds) for p in ids])
    def check(v,w):
        nonlocal n,maxstat
        lo,hi=np.quantile(weights@v,[.025,.975]);d=max(abs(v.mean()-w['mean']),abs(lo-w['low']),abs(hi-w['high']));assert d<2e-12;maxstat=max(maxstat,d);n+=1
    for x in statistics['comparisons']:
        for m,w in x['metrics'].items():check(vector(x,x['baseline']+'_'+m)-vector(x,x['arm']+'_'+m),w['paired_gain'])
        assert x['positive_both_metrics']==all(v['paired_gain']['low']>0 for v in x['metrics'].values())
    for x in statistics['ranking']:
        check(vector(x,x['arm']+'_task_optimal'),x['agreement']);check(vector(x,x['arm']+'_task_regret'),x['regret'])
    for x in statistics['direct']:
        for m,w in x['full_minus_decoded'].items():check(vector(x,'full_'+x['selection']+'_'+m)-vector(x,'decoded_'+x['selection']+'_'+m),w)
    assert n==864
    parent=read(PARENT/'summary.json');parents={(x['split'],x['horizon'],x['seed'],x['choice'],x['baseline']):x for x in parent['deletion']['oracle_baseline_comparisons']};parentdiff=0.
    for x in statistics['comparisons']:
        if not x['arm'].startswith('full'):continue
        orig=parents[x['split'],x['horizon'],x['seed'],'oracle_'+x['arm'].split('_')[1],x['baseline']]
        for metric,w in x['metrics'].items():
            parentdiff=max(parentdiff,max(abs(w['paired_gain'][k]-orig['metrics'][metric]['paired_gain'][k]) for k in ('mean','low','high')))
        assert x['positive_both_metrics']==orig['positive_both_metrics']
    assert parentdiff<2e-12
    # All prespecified complete-vector witnesses must equal the prior unchanged candidate.
    meta=rows(out/'witness_metadata.csv');oldmeta=rows(PARENT/'witness_metadata.csv')
    key=lambda r:(int(r['seed']),r['split'],r['episode'],int(r['horizon']),r['kept'])
    lookup={key(r):i for i,r in enumerate(oldmeta) if r['stage']=='deletion'}
    with np.load(out/'witness_vectors.npz',allow_pickle=False) as f:W={k:f[k] for k in f.files}
    with np.load(PARENT/'witness_vectors.npz',allow_pickle=False) as f:O={k:f[k] for k in f.files}
    vector_max=0.
    for i,r in enumerate(meta):
        j=lookup[key(r)]
        for nk,ok in [('baseline','baseline'),('prediction','prediction'),('target','target'),('truth','position')]:vector_max=max(vector_max,float(abs(W[nk][i]-O[ok][j]).max()))
    assert len(meta)==324 and vector_max<1e-7
    # Rebuild original acceptance from the unchanged parent, never substitute the new score.
    sys.path.insert(0,str(PARENT.parent.parent));from audit_original import audit_original
    original=audit_original(PARENT);assert not original['possible_go']
    assert all(sha(k)==v for k,v in read(out/'asset_ledger.json').items()) and all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
    result=dict(status='PASS_INDEPENDENT_LOCKED_SCORING_AUDIT',raw_rows=82944,paired_rows=2304,statistics=n,max_paired_difference=maxpaired,max_statistic_difference=maxstat,original_full_score_statistic_max_difference=parentdiff,score_identity_max_difference=scoremax,witness_vectors=324,witness_parent_max_difference=vector_max,original_acceptance=original,original_prediction_checks_met=40,original_fixed_checks_met=13,GPU=0,training=False,original_Phase1_GO=False)
    write=out/'independent_audit.json';write.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
