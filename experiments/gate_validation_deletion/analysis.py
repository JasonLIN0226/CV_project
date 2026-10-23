"""Fixed descriptive validation analysis; original acceptance gates are not replaced."""
import collections
import numpy as np
SEEDS=(17,29,43);GROUPS=('cue_oldest','cue_middle','cue_newest');FIELDS=('feature_mse','task_error','common','contrast')
def interval(v,repeats=10000):
 a=np.asarray(v,np.float64)
 if not len(a) or not np.isfinite(a).all():raise ValueError('Finite pair vector required')
 ix=np.random.default_rng(0).integers(0,len(a),(repeats,len(a)));lo,hi=np.quantile(a[ix].mean(1),[.025,.975]);return dict(mean=float(a.mean()),low=float(lo),high=float(hi),pairs=len(a))
def summarize(rows,pairs=96,repeats=10000):
 ix={(int(r['seed']),r['pair'],r['variant'],int(r['horizon'])):r for r in rows}
 if len(ix)!=len(rows):raise ValueError('Duplicate pair/variant/horizon')
 ids=sorted({r['pair'] for r in rows});assert len(ids)==pairs
 variants=['recent_only','full_history','privileged_small_budget','random_small_budget']+[f'{g}/delete_{j}' for g in GROUPS for j in range(3)]
 assert set(ix)=={(s,p,v,h) for s in SEEDS for p in ids for v in variants for h in (3,6)}
 cells=[];comparisons=[];standards=[]
 def vec(seed,h,field,selector):
  ss=SEEDS if seed=='mean' else (seed,)
  return np.array([[selector(ix,s,p,h,field) for p in ids] for s in ss],np.float64).mean(0)
 for seed in (*SEEDS,'mean'):
  for h in (3,6):
   native=lambda ix,s,p,h,f:float(ix[(s,p,'privileged_small_budget',h)][f])
   recent=lambda ix,s,p,h,f:float(ix[(s,p,'recent_only',h)][f])
   for v in variants[:4]:
    standards.append(dict(seed=seed,horizon=h,condition=v,metrics={f:interval(vec(seed,h,f,lambda ix,s,p,h,f:float(ix[(s,p,v,h)][f])),repeats) for f in FIELDS}))
   for g in GROUPS:
    def selected(ix,s,p,h,f,kind):
     rs=[ix[(s,p,f'{g}/delete_{j}',h)] for j in range(3)]
     crit=[r for r in rs if not int(r['cue_present'])];other=[r for r in rs if int(r['cue_present'])];assert len(crit)==1 and len(other)==2
     if kind=='critical':return float(crit[0][f])
     if kind=='other':return np.mean([float(r[f]) for r in other])
     nonnative=[r for r in other if not int(r['native_privileged'])];assert len(nonnative)==(1 if g=='cue_oldest' else 2)
     return np.mean([float(r[f]) for r in nonnative])
    metrics={};penalty={}
    for f in FIELDS:
     critical=vec(seed,h,f,lambda ix,s,p,h,f:selected(ix,s,p,h,f,'critical'));other=vec(seed,h,f,lambda ix,s,p,h,f:selected(ix,s,p,h,f,'other'));delta=interval(critical-other,repeats);rel=float((critical.mean()-other.mean())/max(critical.mean(),1e-8))
     metrics[f]=dict(critical_mean=float(critical.mean()),other_mean=float(other.mean()),difference=delta,relative=rel,meets_original_numeric_conditions=bool(critical.mean()>=1e-8 and rel>=.1 and delta['low']>0))
     new=vec(seed,h,f,lambda ix,s,p,h,f:selected(ix,s,p,h,f,'nonnative'));anchor=vec(seed,h,f,native);penalty[f]=interval(new-anchor,repeats)
    cells.append(dict(seed=seed,horizon=h,pool_group=g,metrics=metrics,both_numeric_conditions=all(metrics[f]['meets_original_numeric_conditions'] for f in ('feature_mse','task_error'))))
    comparisons.append(dict(seed=seed,horizon=h,pool_group=g,nonnative_minus_native_penalty=penalty))
 summary={}
 for g in GROUPS:
  cc=[c for c in cells if c['seed']!='mean' and c['pool_group']==g];summary[g]=dict(cells=len(cc),feature=sum(c['metrics']['feature_mse']['meets_original_numeric_conditions'] for c in cc),task=sum(c['metrics']['task_error']['meets_original_numeric_conditions'] for c in cc),both=sum(c['both_numeric_conditions'] for c in cc))
 base={};by=collections.defaultdict(list)
 for r in rows:by[(r['pair'],r['variant'])].append(r)
 for v in variants:
  base[v]={f:interval([np.mean([float(r[f]) for r in by[(p,v)]]) for p in ids],repeats) for f in FIELDS}
 return dict(status='COMPLETE_FROZEN_VALIDATION_INPUT_DIAGNOSIS',validation_only=True,original_Phase1_GO=False,diagnostic_validation_cell_count=18,seedmean_cells=6,registered_test_acceptance_unchanged=True,by_pool_group=summary,original_contexts=standards,context_aggregates=base,cells=cells,comparisons=comparisons)
