"""English report of one prelocked Motion test; no tuning or model calls."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 repo=Path(__file__).resolve().parents[1];root=repo/'results/motion_history_reference_locked_test';status=json.loads((root/'status.json').read_text());native=root/'native_completed';out=native/f'output/job_{status["job"]}'
 t=json.loads((native/'transfer_integrity.json').read_text());assert t['native_before_local_native_after_all_match'] and all(hashlib.sha256((native/k).read_bytes()).hexdigest()==v['sha256'] for k,v in t['files'].items())
 s=json.loads((out/'summary.json').read_text());a=json.loads((out/'independent_audit.json').read_text());assert a['status']=='PASS_INDEPENDENT_LOCKED_MOTION_AUDIT';p=s['prediction']
 with (out/'fixed_reference_pairs.csv').open() as f:rows=list(csv.DictReader(f))
 independent=[]
 from collections import defaultdict
 def interval(base,alt,pairs,ratio):
  g=defaultdict(list)
  for x,y,k in zip(base,alt,pairs):g[k].append([x,y])
  v=np.array([np.mean(x,0) for x in g.values()]);draw=np.random.default_rng(0).integers(0,len(v),(2000,len(v)))
  def effect(x):
   b=x[...,0].mean(-1);c=x[...,1].mean(-1);return (b-c)/np.maximum(b,.02) if ratio else b-c
  lo,hi=np.quantile(effect(v[draw]),[.025,.975]);return float(effect(v)),float(lo),float(hi)
 for check in p['fixed_reference_checks']:
  _,split,h,seed=check['slice'].split('/');h=int(h[1:]);seed=seed[4:];rr=[r for r in rows if r['split']==split and int(r['horizon'])==h and (seed=='all_seeds_mean' or int(r['seed'])==int(seed))];g=defaultdict(list)
  for r in rr:g[r['pair'],r['episode']].append([float(r['reference']),float(r['candidate']),float(r['reference_recent'])])
  order=sorted(g);v=np.array([np.mean(g[k],0) for k in order]);pairs=[k[0] for k in order]
  if check['kind']=='critical_absolute_mse_reduction':got=interval(v[:,0],v[:,1],pairs,False);want=check['absolute_mse_reduction'];flag=got[0]>0 and got[1]>0
  else:
   ratio=v[:,2].mean()>=.02;got=interval(v[:,2],v[:,1],pairs,ratio);want=check['improvement'];flag=got[0]>=(.1 if ratio else .01) and got[1]>0
  delta=max(abs(x-want[k]) for x,k in zip(got,['mean','low','high']));assert delta<1e-12 and flag==check['passed'];independent.append(delta)
 with (out/'paired_deletions.csv').open() as f:deletion_rows=list(csv.DictReader(f))
 deletion_differences=[]
 for spec in s['deletion']['oracle_baseline_comparisons']:
  rr=[r for r in deletion_rows if r['split']==spec['split'] and int(r['horizon'])==spec['horizon'] and (spec['seed']=='mean' or int(r['seed'])==int(spec['seed']))];ids=sorted({r['pair'] for r in rr});assert len(ids)==128
  for metric,want in spec['metrics'].items():
   delta=np.array([np.mean([float(r[spec['baseline']+'_'+metric])-float(r[spec['choice']+'_'+metric]) for r in rr if r['pair']==pair]) for pair in ids]);ix=np.random.default_rng(0).integers(0,128,(10000,128));lo,hi=np.quantile(delta[ix].mean(1),[.025,.975]);value=want['paired_gain'];diff=max(abs(delta.mean()-value['mean']),abs(lo-value['low']),abs(hi-value['high']));assert diff<2e-12;deletion_differences.append(diff)
 assert len(deletion_differences)==288
 review=dict(status='PASS',fixed_intervals=32,max_difference=max(independent),deletion_intervals=288,deletion_max_difference=max(deletion_differences),new_GPU=0,Phase1_GO=False);(root/'local_fixed_review.json').write_text(json.dumps(review,indent=2)+'\n')
 groups=[('Prediction',p['original_checks_passed'],51),('Fixed recent',sum(x['passed'] for x in p['fixed_reference_checks'] if x['kind']=='candidate_vs_reference_recent'),24),('Critical long gap',sum(x['passed'] for x in p['fixed_reference_checks'] if x['kind']=='critical_absolute_mse_reduction'),8),('Deletion transfer',s['deletion_positive'],s['deletion_total'])]
 fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');bars=ax.barh([x[0] for x in groups],[100*x[1]/x[2] for x in groups],color=['.7','.5','.3','.1'],edgecolor='black');ax.set(xlim=(0,118),xlabel='Registered comparisons met (%)',title='One locked historical-reference composite');ax.invert_yaxis()
 for bar,(_,n,d) in zip(bars,groups):ax.text(bar.get_width()+2,bar.get_y()+bar.get_height()/2,f'{n}/{d}',va='center')
 fig.savefig(root/'registered_checks.png',dpi=170);plt.close(fig)
 lines=['# Motion: historical-reference output plus current position','','## One locked candidate','','Use the existing frozen v3 predictor on the same two retained historical observations as the visual-feature base. Apply the minimum correction inside the fixed128-dimensional output basis that preserves the current guarded predictor position. Coefficient1; no mixing grid. For other memory sizes, retain the existing parent output. Both models receive the same legal observations/actions; future feature targets and simulator truth are scoring labels only. The existing RGB guard governs current position, including its fallback; it does not select another feature base.','','All96original validation pairs and3seeds met8/8original feature numeric checks,8/8position checks and8/8incremental feature comparisons before this exact composite was locked. Missing old-reference output vectors were filled by one frozen CPU inference; no model, decoder or transform was fitted. Validation and test source locks/weights are preserved separately.','','## Complete revealed-development recheck','','All original seeds17/29/43, ordinary/long-gap/distractor tests,128matched pairs per split and horizons3/6. The six original conditions yield34,560prediction rows.17,280unchanged recent/full-history/persistence outcomes are copied from the audited parent;17,280two-memory outcomes are recomputed. All82,944clock-only deletion outcomes preserve2past+4recent frames. No case or seed is removed. These are repeatedly revealed6301development data, not independent confirmation.','','![Coverage](../results/motion_history_reference_locked_test/registered_checks.png)','','| Check | Met / total |','|---|---:|']
 for label,n,d in groups:lines.append(f'|{label}|{n}/{d}|')
 lines+=['','Counts are registered comparisons sharing the same scenes, not independent experiments. Original numeric/CI rules, full-history safeguards and structural checks remain unchanged.','','## Every failed original prediction or fixed-reference check','','| Type | Slice |','|---|---|']
 failed=[x for x in p['original_checks']+p['fixed_reference_checks'] if not x['passed']]
 for x in failed:lines.append(f"|{x['kind']}|{x.get('slice',x.get('split',''))}|")
 if not failed:lines.append('|None|All original named and fixed-reference checks met|')
 lines+=['','### Remaining fixed-recent effect sizes','','| Slice | Feature reduction,95%CI | Original floor |','|---|---|---:|']
 for x in p['fixed_reference_checks']:
  if x['kind']=='candidate_vs_reference_recent' and not x['passed']:
   v=x['improvement'];lines.append(f"|{x['slice']}|{100*v['mean']:.2f}%[{100*v['low']:.2f},{100*v['high']:.2f}]|10%|")
 lines+=['','## Critical long-gap feature error, seed means','','| Horizon | Original reference | Composite | Reference minus composite,95%CI |','|---|---:|---:|---|']
 for x in p['fixed_reference_checks']:
  if x['kind']=='critical_absolute_mse_reduction' and x['slice'].endswith('seedall_seeds_mean'):
   v=x['absolute_mse_reduction'];lines.append(f"|{x['slice'].split('/')[2]}|{x['reference_privileged_mse']:.6f}|{x['candidate_privileged_mse']:.6f}|{v['mean']:.6f}[{v['low']:.6f},{v['high']:.6f}]|")
 lines+=['','## Every non-positive cross-horizon both-metric comparison','','| Split | Horizon | Seed | Baseline | Feature gain CI | Position gain CI |','|---|---:|---|---|---|---|']
 failed=[x for x in s['deletion']['oracle_baseline_comparisons'] if x['choice']=='oracle_transfer' and not x['positive_both_metrics']]
 for x in failed:
  f=x['metrics']['feature_mse']['paired_gain'];t=x['metrics']['task_error']['paired_gain'];lines.append(f"|{x['split']}|{x['horizon']}|{x['seed']}|{x['baseline']}|[{f['low']:.6g},{f['high']:.6g}]|[{t['low']:.6g},{t['high']:.6g}]|")
 if not failed:lines.append('|None|—|—|—|All positive|All positive|')
 lines+=['','The unchanged original feature-loss oracle chooses at one horizon and evaluates at the other, against uniform/FIFO/reject-newest in BOTH full-feature and position error. It remains future-informed offline supervision, not an online retention policy. The earlier decoded-scoring72/72result belongs to its own unchanged-parent experiment; it is not copied onto this new predictor.','','## Verification and limitations','',f"Independent audit reconstructed all51original flags,32fixed-reference comparisons,288deletion intervals and468whole-vector witnesses using independent QR/SVD and a separate dynamics replay. Scalar witness maximum difference {a['scalar_max_difference']:.3g}; vector maximum {a['vector_max_difference']:.3g}. Local32fixed and288deletion interval replays passed. All compact native-before/local/native-after hashes match; large raw arrays remainHPC.",'','No training, fitting, encoder run, policy training, fresh9137or newGPU. The fixed supervised decoder and renderer-specific RGB validity input remain limitations; equal frame counts do not establish feature-only byte-budget parity. This check neither proves real-video generalization nor completes the original fresh confirmation.','','## Decision','',f"**{s['status']}**. Original task status:{p['original_task_status']}. Independent audit success establishes reproducibility; it does not replace scientific gates.",'']
 if s['status']=='MOTION_DEVELOPMENT_REPAIR_SUPPORTED':lines+=['This exact composite has development support; retain its lock and separately freeze the complete original independent-confirmation protocol before any fresh data access. OverallPhase1_GO is not yet established.']
 else:lines+=['Retain every failure and stop this repair path. Do not tune another coefficient or base from these test results; do not open fresh9137or train a policy on a claim that originalPhase1passed.']
 lines+=['','CumulativeGPU remains12,723/28,800seconds, leaving16,077seconds. No GitHub update.']
 (repo/'docs/MOTION_HISTORY_REFERENCE_LOCKED_TEST.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status=s['status'],prediction=p['original_checks_passed'],fixed=p['fixed_checks_passed'],deletion=[s['deletion_positive'],s['deletion_total']],local_intervals=320)))
if __name__=='__main__':main()
