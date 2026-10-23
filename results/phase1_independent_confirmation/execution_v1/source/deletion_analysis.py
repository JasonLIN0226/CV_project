"""Primary paired analysis of age-controlled, frozen-model deletion forks."""
from collections import defaultdict
import numpy as np

SEEDS=(17,29,43)
SPLITS=('test_id','test_long_gap','test_distractor_shift')
GROUPS={'gate':('cue_oldest','cue_middle','cue_newest'),'motion':('clock_all',)}
POLICIES=('uniform','fifo','reject','oracle_same','oracle_transfer')


def interval(values, repeats=10000):
    a=np.asarray(values,np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Invalid pair vector')
    ix=np.random.default_rng(0).integers(0,len(a),(repeats,len(a)))
    b=a[ix].mean(1)
    return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),pairs=len(a))


def aggregate(rows, pairs=128):
    cells=defaultdict(list);seen=set();axes={};ids=defaultdict(set)
    for r in rows:
        task=r['task'];seed=int(r['seed']);split=r['split'];h=int(r['horizon'])
        group=r['pool_group'];pool=r['pool'];pair=r['pair'];episode=r['episode']
        parent=tuple(map(int,r['parent'].split(';')));kept=set(map(int,r['kept'].split(';')));deleted=int(r['deleted'])
        expected_pools=GROUPS['gate'] if task=='gate' else tuple(f'clock_{i}' for i in range(6))
        if task not in GROUPS or seed not in SEEDS or split not in SPLITS or h not in (3,6):raise ValueError('Wrong axis')
        if group not in GROUPS[task] or pool not in expected_pools or (task=='gate' and group!=pool):raise ValueError('Wrong pool')
        if len(parent)!=3 or tuple(sorted(set(parent)))!=parent or kept!=set(parent)-{deleted} or len(kept)!=2:raise ValueError('Invalid deletion fork')
        if int(r['history_count'])!=2 or int(r['feature_dim'])!=5120 or max(parent)>=int(r['query'])-3 or min(parent)<0:raise ValueError('Causal/capacity boundary')
        if not np.isfinite([float(r['feature_mse']),float(r['task_error'])]).all():raise ValueError('Nonfinite metric')
        identity=(task,seed,split,pair,episode,pool,deleted,h)
        if identity in seen:raise ValueError('Duplicate raw row')
        seen.add(identity)
        meta=(parent,int(r['query']),r['evidence'])
        axis=(task,split,pair,pool)
        if axis in axes and axes[axis]!=meta:raise ValueError('Pools differ across model seeds')
        axes[axis]=meta;ids[(task,seed,split,group)].add(pair)
        cells[(task,seed,split,pair,episode,pool,group)].append(r)
    expected={(t,s,sp,g) for t,gs in GROUPS.items() for g in gs for s in SEEDS for sp in SPLITS}
    if set(ids)!=expected or any(len(v)!=pairs for v in ids.values()):raise ValueError('Incomplete pair coverage')
    memberships=defaultdict(set)
    for t,s,sp,pair,ep,pool,g in cells:memberships[(t,s,sp,pair,g)].add((ep,pool))
    for (t,s,sp,pair,g),members in memberships.items():
        episodes={ep for ep,_ in members};pools=(g,) if t=='gate' else tuple(f'clock_{i}' for i in range(6))
        if len(episodes)!=2 or members!={(ep,pool) for ep in episodes for pool in pools}:raise ValueError('Incomplete sibling/pool grid')
    for t in GROUPS:
        for sp in SPLITS:
            common=ids[(t,17,sp,GROUPS[t][0])]
            if any(ids[(t,s,sp,g)]!=common for s in SEEDS for g in GROUPS[t]):raise ValueError('Pair sets differ across groups')
    accumulated=defaultdict(list);critical=defaultdict(list);coverage=defaultdict(list)
    for (task,seed,split,pair,episode,pool,group),rr in cells.items():
        if len(rr)!=6:raise ValueError('Incomplete three deletions x two horizons')
        parent=tuple(map(int,rr[0]['parent'].split(';')))
        by={(int(r['horizon']),int(r['deleted'])):r for r in rr}
        if set(by)!={(h,d) for h in (3,6) for d in parent}:raise ValueError('Missing horizon/fork')
        losses=np.array([[[float(by[(h,d)][m]) for m in ('feature_mse','task_error')] for d in parent] for h in (3,6)])
        ev=tuple(map(int,rr[0]['evidence'].split(';')))
        if task=='gate':
            pos=parent.index(ev[0]);expected_pos=GROUPS['gate'].index(group)
            if pos!=expected_pos or len(ev)!=1:raise ValueError('Gate cue rank changed')
            if any(rr0['deleted_role']!=('cue' if int(rr0['deleted'])==ev[0] else 'other') for rr0 in rr):raise ValueError('Wrong Gate role')
        for hi,h in enumerate((3,6)):
            same=np.flatnonzero(losses[hi,:,0]<=losses[hi,:,0].min()+1e-8)
            other=np.flatnonzero(losses[1-hi,:,0]<=losses[1-hi,:,0].min()+1e-8)
            values=np.stack([losses[hi].mean(0),losses[hi,0],losses[hi,-1],losses[hi,same].mean(0),losses[hi,other].mean(0)])
            key=(task,seed,split,group,pair,h)
            accumulated[key].append(values)
            coverage[(task,seed,split,group,h)].append(dict(pair=pair,zero_spread=bool(np.ptp(losses[hi,:,0])<=1e-8),annotated_count=len(set(parent)&set(ev))))
            if task=='gate':critical[key].append(np.stack([losses[hi,pos],np.delete(losses[hi],pos,axis=0).mean(0)]))
    compact=[];crit={}
    for key,values in sorted(accumulated.items()):
        task,seed,split,group,pair,h=key;expected_n=2 if task=='gate' else 12
        if len(values)!=expected_n:raise ValueError('Sibling/pool count changed')
        a=np.mean(values,axis=0)
        row=dict(task=task,seed=seed,split=split,pool_group=group,pair=pair,horizon=h)
        for i,p in enumerate(POLICIES):
            for j,m in enumerate(('feature_mse','task_error')):row[p+'_'+m]=float(a[i,j])
        if task=='gate':
            ca=np.mean(critical[key],axis=0);crit[key]=ca
            for i,label in enumerate(('critical','other')):
                for j,m in enumerate(('feature_mse','task_error')):row[label+'_'+m]=float(ca[i,j])
        else:
            for label in ('critical','other'):
                for m in ('feature_mse','task_error'):row[label+'_'+m]=''
        compact.append(row)
    if len(compact)!=len(expected)*pairs*2:raise ValueError('Compact coverage')
    cov=[]
    for key,rr in sorted(coverage.items()):
        task,seed,split,group,h=key
        cov.append(dict(task=task,seed=seed,split=split,pool_group=group,horizon=h,
            underlying_episode_pools=len(rr),base_pairs=pairs,
            zero_feature_spread_fraction=float(np.mean([r['zero_spread'] for r in rr])),
            annotation_count_fractions={str(i):float(np.mean([r['annotated_count']==i for r in rr])) for i in (0,1,2)}))
    return compact,cov


def summarize(compact, repeats=10000, pairs=128):
    by=defaultdict(dict)
    for r in compact:
        key=(r['task'],r['split'],r['pool_group'],int(r['horizon']))
        by[key].setdefault(int(r['seed']),{})[r['pair']]=r
    contrasts=[];gains=[]
    for (task,split,group,h),seeds in sorted(by.items()):
        pair_ids=sorted(seeds[17])
        if set(seeds)!=set(SEEDS) or len(pair_ids)!=pairs or any(set(seeds[s])!=set(pair_ids) for s in SEEDS):raise ValueError('Unpaired seeds')
        def values(column,seed):
            all_values=np.array([[float(seeds[s][p][column]) for p in pair_ids] for s in SEEDS])
            return all_values.mean(0) if seed=='mean' else all_values[SEEDS.index(seed)]
        for seed in (*SEEDS,'mean'):
            meta=dict(task=task,seed=seed,split=split,pool_group=group,horizon=h)
            if task=='gate':
                metrics={}
                for m in ('feature_mse','task_error'):
                    c=values('critical_'+m,seed);n=values('other_'+m,seed);delta=interval(c-n,repeats)
                    relative=float((c.mean()-n.mean())/max(c.mean(),1e-8))
                    metrics[m]=dict(critical_mean=float(c.mean()),other_mean=float(n.mean()),difference=delta,
                        relative_reduction=relative,passed=bool(c.mean()>=1e-8 and relative>=.1 and delta['low']>0))
                contrasts.append({**meta,'metrics':metrics,'passed':all(v['passed'] for v in metrics.values())})
            for oracle in ('oracle_same','oracle_transfer'):
                for baseline in ('uniform','fifo','reject'):
                    metrics={}
                    for m in ('feature_mse','task_error'):
                        b=values(baseline+'_'+m,seed);o=values(oracle+'_'+m,seed)
                        metrics[m]=dict(baseline_mean=float(b.mean()),chosen_mean=float(o.mean()),paired_gain=interval(b-o,repeats),
                            relative_gain=None if b.mean()<1e-8 else float((b.mean()-o.mean())/b.mean()))
                    gains.append({**meta,'choice':oracle,'selection_horizon':h if oracle=='oracle_same' else 9-h,'baseline':baseline,
                        'metrics':metrics,'positive_both_metrics':all(v['paired_gain']['low']>0 for v in metrics.values()),
                        'interpretation':'descriptive same-time oracle' if oracle=='oracle_same' else 'cross-horizon future-informed diagnostic'})
    return dict(status='COMPLETE_STATIC_POOL_DIAGNOSIS',paired_rows=len(compact),gate_critical_contrasts=contrasts,
        oracle_baseline_comparisons=gains,independent_scalar_statistics=2*(len(contrasts)+len(gains)),
        original_phase1_go=False,joint_policy_stage_ready=False,policy_learnability_proven=False,fresh9137_opened=False,
        no_training_or_fitting=True,GPU_seconds=0,
        interpretation='Revealed6301 static-pool mechanism diagnosis. Privileged Gate pools and future-selected oracles are not deployable retention policies.')
