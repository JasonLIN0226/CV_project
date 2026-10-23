"""Post-hoc descriptive decomposition; original gates are never recomputed here."""
import collections
import numpy as np
METRICS=('total','parallel','orthogonal','common','contrast','basis_inside','basis_outside','task_error','margin_loss','score_error','label_score_error','prediction_pair_signal','signal_gain')
def ci(a):
 a=np.asarray(a,np.float64);g=np.random.default_rng(0);b=a[g.integers(0,len(a),(10000,len(a)))].mean(1)
 return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),pairs=len(a))
def summarize(rows):
 cells=collections.defaultdict(dict)
 for r in rows:
  key=(int(r['seed']),r['split'],r['pool_group'],int(r['horizon']))
  assert r['pair'] not in cells[key];cells[key][r['pair']]=r
 assert len(cells)==54 and all(len(v)==128 for v in cells.values())
 result=[]
 for seed in (17,29,43,'mean'):
  for split in ('test_id','test_long_gap','test_distractor_shift'):
   for group in ('cue_oldest','cue_middle','cue_newest'):
    for h in (3,6):
     ids=sorted(cells[(17,split,group,h)]);seeds=(17,29,43) if seed=='mean' else (seed,)
     metrics={}
     for name in METRICS:
      c=np.array([[float(cells[(s,split,group,h)][pair]['critical_'+name]) for pair in ids] for s in seeds]).mean(0)
      o=np.array([[float(cells[(s,split,group,h)][pair]['other_'+name]) for pair in ids] for s in seeds]).mean(0)
      metrics[name]=dict(critical_mean=float(c.mean()),other_mean=float(o.mean()),difference=ci(c-o))
     result.append(dict(seed=seed,split=split,pool_group=group,horizon=h,metrics=metrics))
 per_seed=[r for r in result if r['seed']!='mean'];wrong=[r for r in per_seed if r['metrics']['total']['difference']['mean']<=0]
 def positive(name,rr):return sum(r['metrics'][name]['difference']['mean']>0 for r in rr)
 return dict(status='COMPLETE_POSTHOC_GATE_COST_DIAGNOSIS',paired_rows=len(rows),original_seed_cells=54,seed_mean_cells=18,
  wrong_total_direction_cells=len(wrong),wrong_total_but_positive_parallel=positive('parallel',wrong),wrong_total_but_positive_contrast=positive('contrast',wrong),wrong_total_but_positive_label_score_error=positive('label_score_error',wrong),
  pareto_tradeoff={k:sum(int(r[k]) for r in rows) for k in ('task_help_available','joint_help_available','all_task_help_increases_feature')},
  positive_parallel_cells=positive('parallel',per_seed),positive_contrast_cells=positive('contrast',per_seed),cells=result,
  new_acceptance_rule=False,original_Phase1_GO=False,policy_training=False,new_GPU_seconds=0)
