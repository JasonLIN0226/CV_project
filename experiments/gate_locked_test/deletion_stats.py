import collections
import numpy as np
def compute(rr,repeats=10000):
 grouped=collections.defaultdict(list)
 for r in rr:grouped[(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']))].append(r)
 pairrows=[]
 for (seed,split,group,pair,h),rs in sorted(grouped.items()):
  assert len(rs)==6 and len({r['episode'] for r in rs})==2
  cue=int(rs[0]['cue']);cr=[r for r in rs if int(r['deleted'])==cue];ot=[r for r in rs if int(r['deleted'])!=cue];assert len(cr)==2 and len(ot)==4
  d=dict(seed=seed,split=split,pool_group=group,pair=pair,horizon=h)
  for metric in ('feature_mse','task_error'):
   d['critical_'+metric]=float(np.mean([float(r[metric]) for r in cr]));d['other_'+metric]=float(np.mean([float(r[metric]) for r in ot]))
  pairrows.append(d)
 by=collections.defaultdict(dict)
 for r in pairrows:by[(r['split'],r['pool_group'],r['horizon'])].setdefault(r['seed'],{})[r['pair']]=r
 cells=[]
 for (split,g,h),ss in sorted(by.items()):
  ids=sorted(ss[17]);assert len(ids)==128 and set(ss)=={17,29,43} and all(set(ss[s])==set(ids) for s in ss)
  ix=np.random.default_rng(0).integers(0,128,(repeats,128))
  for seed in (17,29,43,'mean'):
   cell=dict(seed=seed,split=split,pool_group=g,horizon=h,metrics={})
   for m in ('feature_mse','task_error'):
    a=np.array([[ss[s][i]['critical_'+m] for i in ids] for s in (17,29,43)]);b=np.array([[ss[s][i]['other_'+m] for i in ids] for s in (17,29,43)])
    a=a.mean(0) if seed=='mean' else a[(17,29,43).index(seed)];b=b.mean(0) if seed=='mean' else b[(17,29,43).index(seed)];delta=a-b;boot=delta[ix].mean(1);lo,hi=np.quantile(boot,[.025,.975]);rel=float(delta.mean()/max(a.mean(),1e-8))
    cell['metrics'][m]=dict(critical_mean=float(a.mean()),other_mean=float(b.mean()),delta=float(delta.mean()),low=float(lo),high=float(hi),relative=rel,passed=bool(a.mean()>=1e-8 and rel>=.1 and lo>0))
   cell['passed']=all(x['passed'] for x in cell['metrics'].values());cells.append(cell)
 return pairrows,cells

