import collections
import numpy as np
from phase1.stats import clustered_interval

def analyze(rows):
 by=collections.defaultdict(list)
 for r in rows:by[(r['seed'],r['split'],r['pool_group'],r['pair'],r['horizon'])].append(r)
 compact=[]
 for (seed,sp,g,pair,h),rr in sorted(by.items()):
  assert len(rr)==6 and len({r['episode'] for r in rr})==2
  critical=[r for r in rr if r['deleted']==r['cue']];other=[r for r in rr if r['deleted']!=r['cue']];assert len(critical)==2 and len(other)==4
  r=dict(seed=seed,split=sp,pool_group=g,pair=pair,horizon=h)
  for m in ('feature_mse','task_error'):
   r['critical_'+m]=float(np.mean([v[m] for v in critical]));r['other_'+m]=float(np.mean([v[m] for v in other]))
  compact.append(r)
 ix={(r['seed'],r['split'],r['pool_group'],r['pair'],r['horizon']):r for r in compact};cells=[]
 for sp in ('test_id','test_long_gap','test_distractor_shift'):
  ids=sorted({r['pair'] for r in compact if r['split']==sp});assert len(ids)==128
  for g in ('cue_oldest','cue_middle','cue_newest'):
   for h in (3,6):
    for seed in (17,29,43,'mean'):
     seeds=(17,29,43) if seed=='mean' else (seed,);cell=dict(seed=seed,split=sp,pool_group=g,horizon=h,metrics={})
     for m in ('feature_mse','task_error'):
      a=np.mean([[ix[(s,sp,g,p,h)]['critical_'+m] for p in ids] for s in seeds],0);b=np.mean([[ix[(s,sp,g,p,h)]['other_'+m] for p in ids] for s in seeds],0)
      diff=clustered_interval(a-b,ids,10000);relative=float((a.mean()-b.mean())/max(a.mean(),1e-8))
      cell['metrics'][m]=dict(critical_mean=float(a.mean()),other_mean=float(b.mean()),difference=diff,relative=relative,passed=bool(a.mean()>=1e-8 and relative>=.1 and diff['low']>0))
     cell['passed']=all(v['passed'] for v in cell['metrics'].values());cells.append(cell)
 return compact,cells
