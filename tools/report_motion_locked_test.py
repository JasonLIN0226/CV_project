"""English report of one prelocked Motion test; no tuning or model calls."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 repo=Path(__file__).resolve().parents[1];root=repo/'results/motion_locked_test';status=json.loads((root/'status.json').read_text());native=root/'native_completed';out=native/f'output/job_{status["job"]}'
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
 groups=[('Original prediction',p['original_checks_passed'],51),('Fixed recent reference',sum(x['passed'] for x in p['fixed_reference_checks'] if x['kind']=='candidate_vs_reference_recent'),24),('Critical long-gap',sum(x['passed'] for x in p['fixed_reference_checks'] if x['kind']=='critical_absolute_mse_reduction'),8)]
 fig,ax=plt.subplots(figsize=(8.7,3.8),layout='constrained');bars=ax.barh([x[0] for x in groups],[x[1]/x[2]*100 for x in groups],color=['.7','.4','.1'],edgecolor='black');ax.set(xlim=(0,115),xlabel='Registered checks met (%)',title='Locked Motion recheck — original thresholds unchanged');ax.axvline(100,color='black',ls='--',lw=1)
 for bar,(_,n,d) in zip(bars,groups):ax.text(bar.get_width()+2,bar.get_y()+bar.get_height()/2,f'{n}/{d}',va='center')
 ax.invert_yaxis();fig.savefig(root/'registered_checks.png',dpi=180);plt.close(fig)
 text=['# Locked Motion development recheck','', '## Objective and fixed intervention','', 'Determine whether the original Motion requirements are met after the validation-selected input-validity guard. The three frozen predictors, caches, train-only transforms, capacity, samples, seeds, horizons and every original threshold remain unchanged. No training, fitting, encoder run or new GPU was used.','', 'At exactly two retained past frames, apply the existing geometric correction only when BOTH observed RGB arena crops differ from the current crop; otherwise preserve the baseline exactly. With other memory sizes the candidate is identity. The rule was locked before new test inference and was never tuned afterward. The wrong-evidence control uses the actual substituted RGB/features, not the original evidence.','', '**Limitation:** RGB validity is an added modality and the crop/equality rule exploits the fixed synthetic shutter. Frame counts are matched, but the auxiliary RGB validity input is not a demonstrated feature-only byte-budget match. This is revealed development data, not real-video robustness, direct V-JEPA validation, or fresh independent confirmation.','', '## Experimental coverage','', 'All 96 original validation pairs and 128 pairs in each ordinary, long-gap and distractor-shift test; three seeds (17, 29, 43) and horizons 3 and 6. Ordinary tests match the original generator, long-gap tests lengthen the occlusion, and distractor-shift tests change irrelevant outside-arena shapes/counts/palettes. Six original conditions yield 34,560 raw prediction rows. Exactly 17,280 unchanged recent/full-history/persistence outcomes reuse audited parent scalars; 17,280 small-capacity outcomes are recomputed with baseline replay. No failed seeds or hard cases are removed.','', 'All six clock-only triple pools per test pair, every deletion choice, the same 2-past + 4-recent frame capacity and 3 seeds yield 82,944 deletion outcomes and 2,304 compact paired rows. The original 10,000-draw paired bootstrap and all zero-signal pools remain.','', '## Original acceptance results','', '![Original registered checks](../results/motion_locked_test/registered_checks.png)','', 'Each denominator is a registered comparison count, not an independent dataset count. A positive majority does not replace the original conjunction.','', '| Check group | Met / total |','|---|---:|']
 for label,n,d in groups:text.append(f'|{label}|{n}/{d}|')
 text += ['','| Seed | Original prediction checks met |','|---|---:|']
 for seed in ['17','29','43','all_seeds_mean']:
  rr=[x for x in p['original_checks'] if x.get('slice','').endswith('/seed'+seed)];text.append(f'|{seed}|{sum(x["passed"] for x in rr)}/{len(rr)}|')
 text += ['', 'The three evidence-probe checks are shared; they are not repeated in each seed row. Each seed row contains six benefit and six content-intervention checks.','', f'Unnumbered full-history safeguard: {a["original"]["full_history_rule_met"]}; structural safeguards: {a["original"]["structural_rules_met"]}.','', '## Remaining failed checks','', '| Type | Slice |','|---|---|']
 failures=[x for x in p['original_checks']+p['fixed_reference_checks'] if not x['passed']]
 for x in failures:text.append(f'|{x["kind"]}|{x.get("slice",x.get("split",""))}|')
 if not failures:text.append('|None|All original named and fixed-reference checks met|')
 component=json.loads((out/'prediction_components.json').read_text())
 text += ['', '## Why Motion still falls short', '', '| Registered privileged-versus-recent component | Met / total |','|---|---:|',f'|Complete-feature benefit|{component["feature_checks_met"]}/24|',f'|Position benefit|{component["position_checks_met"]}/24|', '', 'Better position prediction does not guarantee accurate complete visual-feature prediction. The remaining failures concern the original feature requirements, especially long-gap and distractor shifts; the counts do not establish a unique background/representation cause. The fixed-reference long-gap check also requires a reliable absolute feature-error reduction against the original same-capacity reference.', '', '| Long-gap horizon, seed mean | Fixed reference MSE | Candidate MSE | Candidate error increase |','|---|---:|---:|---:|']
 for x in p['fixed_reference_checks']:
  if x['kind']=='critical_absolute_mse_reduction' and x['slice'].endswith('seedall_seeds_mean'):
   h=x['slice'].split('/')[2];ref=x['reference_privileged_mse'];cur=x['candidate_privileged_mse'];text.append(f'|{h}|{ref:.6f}|{cur:.6f}|{100*(cur-ref)/ref:.2f}%|')
 text += ['','## Same-capacity deletion consistency','',f'The future-feature oracle transfers a deletion choice to the other horizon and improves both feature and position errors in **{s["deletion_positive"]}/{s["deletion_total"]}** comparisons against uniform, FIFO and reject-newest. These include individual-seed and seed-mean estimates; they share data. Selection uses future targets for scoring only, so this is an upper-bound diagnostic, not a learned or deployable past-only retention policy.','', 'The single non-positive-confidence comparison is distractor-shift, seed43: deletion selected at horizon6, evaluated at horizon3 against reject-newest. Mean position-error reduction is positive (9.41%), but its paired gain interval crosses zero: [-0.0000955, 0.001198]. It is uncertainty in this comparison, not a demonstrated average position regression. All failures and uncertainty remain in the report.','', '## Verification','',f'Independent audit rebuilt 51 original flags, 32 fixed-reference comparisons and 288 deletion scalar intervals. Every observed guard decision was replayed ({a["observed_guard_decisions_replayed"]:,}); 468 prespecified complete-feature witnesses were independently projected using QR/SVD and a separate dynamics implementation. All 82,944 baseline deletion outcomes were checked against the parent, and all 17,280 unchanged original outcomes preserved. Maximum scalar witness difference {a["scalar_max_difference"]:.3g}. Local 32 fixed intervals and 288 deletion intervals reproduced; maximum differences {max(independent):.3g} and {max(deletion_differences):.3g}. Compact native-before/local/native-after hashes agree; large arrays and raw CSV remain on HPC.','', '## Conclusion and next action','',f'**{s["status"]}**. Original task decision: **{p["original_task_status"]}**. Software and audit success mean the measurements are reproducible, not that scientific acceptance is automatic.','']
 if s['status']=='MOTION_DEVELOPMENT_REPAIR_SUPPORTED':text += ['The locked Motion candidate has development support. Together with the separately supported Gate candidate this permits preparing the already-authorized genuine independent confirmation, subject to a complete frozen implementation and cumulative budget guard. This result alone is notPhase1_GO;fresh9137has not yet been opened by this run.']
 else:text += ['Motion still does not satisfy the complete original requirements. Keep this failure and stop without retuning on these test results. Do not open fresh9137 or train a retention policy on a claim that Phase1 passed. Gate remains its separately supported candidate; overall Phase1_GO is not established.']
 text += ['','GPU use is unchanged at 12,723/28,800 seconds; 16,077 seconds remain. No GitHub update.']
 (repo/'docs/MOTION_LOCKED_TEST.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(scientific_status=s['status'],original=p['original_checks_passed'],fixed=p['fixed_checks_passed'],deletion=f'{s["deletion_positive"]}/{s["deletion_total"]}',local_review=review)))
if __name__=='__main__':main()
