"""Keep both original outcome metrics while changing only oracle score selection."""
import collections
import numpy as np
ARMS=('full_same','full_transfer','decoded_same','decoded_transfer')
BASELINES=('uniform','fifo','reject')
def interval(v):
    a=np.asarray(v,float);assert a.shape==(128,) and np.isfinite(a).all()
    ix=np.random.default_rng(0).integers(0,128,(10000,128));lo,hi=np.quantile(a[ix].mean(1),[.025,.975])
    return dict(mean=float(a.mean()),low=float(lo),high=float(hi))
def aggregate(rows):
    groups=collections.defaultdict(dict);seen=set()
    for r in rows:
        key=(r['split'],int(r['seed']),r['pair'],r['episode'],r['pool']);h=int(r['horizon']);d=int(r['deleted']);parent=tuple(map(int,r['parent'].split(';')))
        assert key+(h,d) not in seen;seen.add(key+(h,d));assert len(parent)==3 and parent==tuple(sorted(set(parent))) and d in parent
        assert set(map(int,r['kept'].split(';')))==set(parent)-{d} and int(r['history_count'])==2 and max(parent)<int(r['query'])-3
        assert int(r['feature_dim'])==5120 and r['task']=='motion' and np.isfinite([float(r[m]) for m in ('feature_mse','task_error','decoded_score')]).all()
        groups[key][h,d]=r
    assert len(rows)==82944 and len(groups)==13824
    combined=collections.defaultdict(list)
    for (split,seed,pair,ep,pool),rr in sorted(groups.items()):
        parent=tuple(map(int,next(iter(rr.values()))['parent'].split(';')));assert set(rr)=={(h,d) for h in (3,6) for d in parent}
        losses=np.array([[[float(rr[h,d][m]) for m in ('feature_mse','task_error')] for d in parent] for h in (3,6)])
        scores=np.array([[float(rr[h,d]['decoded_score']) for d in parent] for h in (3,6)])
        for hi,h in enumerate((3,6)):
            v={};task=losses[hi,:,1];optimal=np.flatnonzero(task<=task.min()+1e-12)
            for name,idx in [('uniform',np.arange(3)),('fifo',np.array([0])),('reject',np.array([2])),('task_oracle',optimal)]:
                for mi,m in enumerate(('feature_mse','task_error')):v[name+'_'+m]=float(losses[hi,idx,mi].mean())
            for name in ARMS:
                cost=scores if name.startswith('decoded') else losses[:,:,0];j=hi if name.endswith('same') else 1-hi;tol=1e-12 if name.startswith('decoded') else 1e-8
                idx=np.flatnonzero(cost[j]<=cost[j].min()+tol)
                for mi,m in enumerate(('feature_mse','task_error')):v[name+'_'+m]=float(losses[hi,idx,mi].mean())
                v[name+'_task_optimal']=float(np.isin(idx,optimal).mean());v[name+'_task_regret']=float(task[idx].mean()-task.min())
            combined[split,seed,pair,h].append(v)
    result=[]
    for (split,seed,pair,h),rr in sorted(combined.items()):
        assert len(rr)==12
        result.append(dict(task='motion',split=split,seed=seed,pair=pair,horizon=h,**{k:float(np.mean([v[k] for v in rr])) for k in rr[0]}))
    assert len(result)==2304
    return result
def summarize(paired):
    comparisons=[];ranking=[];direct=[]
    for split in ('test_id','test_long_gap','test_distractor_shift'):
        for h in (3,6):
            rr=[r for r in paired if r['split']==split and int(r['horizon'])==h];ids=sorted({r['pair'] for r in rr});assert len(rr)==384 and len(ids)==128
            by={(int(r['seed']),r['pair']):r for r in rr};assert len(by)==384
            for seed in (17,29,43,'mean'):
                seeds=(17,29,43) if seed=='mean' else (seed,)
                def values(column):return np.array([np.mean([float(by[s,p][column]) for s in seeds]) for p in ids])
                meta=dict(split=split,horizon=h,seed=seed)
                for arm in ARMS:
                    ranking.append(dict(**meta,arm=arm,agreement=interval(values(arm+'_task_optimal')),regret=interval(values(arm+'_task_regret'))))
                    for base in BASELINES:
                        metrics={}
                        for m in ('feature_mse','task_error'):
                            b=values(base+'_'+m);a=values(arm+'_'+m);metrics[m]=dict(baseline_mean=float(b.mean()),chosen_mean=float(a.mean()),paired_gain=interval(b-a))
                        comparisons.append(dict(**meta,arm=arm,baseline=base,metrics=metrics,positive_both_metrics=all(x['paired_gain']['low']>0 for x in metrics.values())))
                for kind in ('same','transfer'):
                    metrics={m:interval(values('full_'+kind+'_'+m)-values('decoded_'+kind+'_'+m)) for m in ('feature_mse','task_error')}
                    direct.append(dict(**meta,selection=kind,full_minus_decoded=metrics))
    return dict(comparisons=comparisons,ranking=ranking,direct=direct,
                transfer_both_positive={arm:sum(x['positive_both_metrics'] for x in comparisons if x['arm']==arm) for arm in ('full_transfer','decoded_transfer')},transfer_comparisons_per_arm=72)
