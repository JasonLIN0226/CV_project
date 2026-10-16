"""Pure timestamp forks and paired scalar analysis; no fitting or data access."""
import numpy as np

def pools(task, evidence, query, recent=4, evidence_end=None):
    ev=tuple(map(int,evidence)); stop=int(query)-recent+1
    if task not in ('gate','motion') or len(ev)!=(1 if task=='gate' else 2):
        raise ValueError('Wrong evidence roles')
    if len(set(ev))!=len(ev) or tuple(sorted(ev))!=ev or not all(0<=t<stop for t in ev):
        raise ValueError('Illegal evidence')
    end=ev[-1] if evidence_end is None else int(evidence_end)
    if not ev[-1]<=end<stop or (task=='motion' and end!=ev[-1]):raise ValueError('Illegal evidence window end')
    needed=2 if task=='gate' else 1
    filler=tuple(range(end+1,stop))
    if len(filler)<needed: raise ValueError('No legal nuisance timestamps')
    return [('arrival',tuple(sorted(ev+filler[-needed:]))),
            ('early_nuisance',tuple(sorted(ev+filler[:needed])))]

def forks(parent):
    if len(parent)!=3 or len(set(parent))!=3: raise ValueError('Exactly3 unique candidates required')
    return [(deleted,tuple(t for t in parent if t!=deleted)) for deleted in parent]

def interval(x,repeats=10000):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Invalid independent-pair array')
    rng=np.random.default_rng(0)
    boot=a[rng.integers(0,len(a),(repeats,len(a)))].mean(1)
    return dict(mean=float(a.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),pairs=len(a))

def effect(critical,nuisance):
    c=np.asarray(critical,np.float64);n=np.asarray(nuisance,np.float64)
    delta=interval(c-n);rng=np.random.default_rng(0)
    ix=rng.integers(0,len(c),(10000,len(c)))
    bc=c[ix].mean(1);bn=n[ix].mean(1)
    ratio=(bc-bn)/np.maximum(bc,1e-8)
    r=dict(mean=float((c.mean()-n.mean())/max(c.mean(),1e-8)),low=float(np.quantile(ratio,.025)),high=float(np.quantile(ratio,.975)))
    return dict(critical_mean=float(c.mean()),nuisance_mean=float(n.mean()),difference=delta,relative_reduction=r,
                passed=bool(c.mean()>=1e-8 and delta['low']>0 and r['mean']>=.10))

def summarize(rows):
    """Raw row identifies a sibling, pool, deleted index and actual paired losses."""
    expected=2*3*3*128*2*2*3*2
    if len(rows)!=expected:raise ValueError(f'Expected{expected} scalar fork/horizon rows, got{len(rows)}')
    by={}
    for r in rows:
        key=(r['task'],int(r['seed']),r['split'],r['pair'],r['episode'],r['pool'])
        by.setdefault(key,{})[(int(r['deleted']),int(r['horizon']))]=r
    if len(by)!=2*3*3*128*2*2:raise ValueError('Missing or duplicate episode/pool')
    for k,rs in by.items():
        if len(rs)!=6:raise ValueError('Missing deletion or horizon')
        for r in rs.values():
            if int(r['history_count'])!=2 or len(set(r['kept'].split(';')))!=2:raise ValueError('Unequal capacity')
            if not np.isfinite([float(r['feature_mse']),float(r['task_error'])]).all():raise ValueError('Nonfinite')
    contrasts=[];rankings=[];compact=[]
    for task in ('gate','motion'):
      roles=('cue',) if task=='gate' else ('start','end')
      for seed in (17,29,43):
       for split in ('test_id','test_long_gap','test_distractor_shift'):
        group={k:v for k,v in by.items() if k[:3]==(task,seed,split)}
        for h in (3,6):
         for role in roles:
          cpair={};npair={}
          for k,rs in group.items():
            rr=[r for (d,hh),r in rs.items() if hh==h]
            c=[r for r in rr if r['deleted_role']==role]
            n=[r for r in rr if r['deleted_role']=='nuisance']
            if len(c)!=1 or len(n)!=(2 if task=='gate' else 1):raise ValueError('Invalid evidence/nuisance roles')
            pair=k[3]
            cpair.setdefault(pair,[]).append([float(c[0][m]) for m in ('feature_mse','task_error')])
            npair.setdefault(pair,[]).append([np.mean([float(r[m]) for r in n]) for m in ('feature_mse','task_error')])
          pairs=sorted(cpair)
          if len(pairs)!=128 or any(len(cpair[p])!=4 for p in pairs):raise ValueError('Wrong pair clustering')
          ca=np.array([np.mean(cpair[p],0) for p in pairs]);na=np.array([np.mean(npair[p],0) for p in pairs])
          spec=dict(task=task,seed=seed,split=split,horizon=h,role=role)
          vals={m:effect(ca[:,j],na[:,j]) for j,m in enumerate(('feature_mse','task_error'))}
          contrasts.append({**spec,'metrics':vals,'passed':all(v['passed'] for v in vals.values())})
          for i,pair in enumerate(pairs):compact.append({**spec,'pair':pair,'kind':'critical_vs_nuisance',
              'critical_feature':float(ca[i,0]),'nuisance_feature':float(na[i,0]),'critical_task':float(ca[i,1]),'nuisance_task':float(na[i,1])})
        for select_h,test_h in ((3,6),(6,3)):
          pair_values={};agreement=[]
          for k,rs in group.items():
            ds=sorted({d for d,h in rs});a=np.array([float(rs[(d,select_h)]['feature_mse']) for d in ds])
            chosen=np.flatnonzero(a<=a.min()+1e-8)
            target=np.array([[float(rs[(d,test_h)][m]) for m in ('feature_mse','task_error')] for d in ds])
            gain=target.mean(0)-target[chosen].mean(0)
            pair_values.setdefault(k[3],[]).append(gain)
            target_best=np.flatnonzero(target[:,0]<=target[:,0].min()+1e-8)
            agreement.append(len(set(chosen)&set(target_best))/len(chosen))
          arr=np.array([np.mean(pair_values[p],0) for p in sorted(pair_values)])
          ci={m:interval(arr[:,j]) for j,m in enumerate(('feature_mse','task_error'))}
          rankings.append(dict(task=task,seed=seed,split=split,select_h=select_h,test_h=test_h,
              gains=ci,rank_agreement=float(np.mean(agreement)),passed=all(x['low']>0 for x in ci.values())))
          for i,pair in enumerate(sorted(pair_values)):compact.append(dict(task=task,seed=seed,split=split,horizon=test_h,role=str(select_h),pair=pair,
              kind='cross_horizon',critical_feature=float(arr[i,0]),nuisance_feature=0.,critical_task=float(arr[i,1]),nuisance_task=0.))
    task_results={}
    for task in ('gate','motion'):
        scope={}
        for split in ('test_id','test_long_gap','test_distractor_shift'):
            cc=[c for c in contrasts if c['task']==task and c['split']==split]
            rr=[r for r in rankings if r['task']==task and r['split']==split]
            scope[split]=dict(contrasts_passed=sum(c['passed'] for c in cc),contrasts_total=len(cc),
                ranking_passed=sum(r['passed'] for r in rr),ranking_total=len(rr),supported=all(c['passed'] for c in cc+rr))
        task_results[task]=dict(scope=scope,ID_pilot_ready=scope['test_id']['supported'])
    joint=all(t['ID_pilot_ready'] for t in task_results.values())
    return dict(raw_rows=len(rows),independent_base_pairs_per_task_split=128,contrasts=contrasts,rankings=rankings,tasks=task_results,
        decision='READY_FOR_ID_BOUNDED_POLICY_PILOT' if joint else 'JOINT_POLICY_STAGE_NOT_READY',
        original_phase1_go=False,unseen_confirmation=False,policy_learnability_proven=False),compact

def baseline_summary(rows):
    """Descriptive headroom only; same-horizon oracle min is selection-biased."""
    cells={}
    for r in rows:
        k=(r['task'],int(r['seed']),r['split'],r['pair'],r['episode'],r['pool'],int(r['horizon']))
        cells.setdefault(k,[]).append(r)
    grouped={}
    for k,rr in cells.items():
        assert len(rr)==3
        ordered=sorted(rr,key=lambda r:int(r['deleted']))
        losses=np.array([[float(r[m]) for m in ('feature_mse','task_error')] for r in ordered])
        best=np.flatnonzero(losses[:,0]<=losses[:,0].min()+1e-8)
        vals={'uniform_delete':losses.mean(0),'FIFO_delete_oldest':losses[0],
              'reject_newest':losses[-1],'future_feature_oracle':losses[best].mean(0)}
        for name,val in vals.items():
            group=(k[0],k[1],k[2],k[6],name)
            grouped.setdefault(group,{}).setdefault(k[3],[]).append(val)
    output=[]
    for k,pairs in grouped.items():
        a=np.array([np.mean(pairs[p],0) for p in sorted(pairs)])
        output.append(dict(task=k[0],seed=k[1],split=k[2],horizon=k[3],baseline=k[4],
            metrics={m:interval(a[:,j]) for j,m in enumerate(('feature_mse','task_error'))}))
    return dict(results=output,interpretation='Descriptive equal-capacity baselines. Future oracle is not an online policy. No advantage over reject-newest is assumed. Uniform deletion is not a full streaming Reservoir experiment.')
