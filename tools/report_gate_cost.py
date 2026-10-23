"""Explain verified Gate diagnosis with paired aggregates, no model operations."""
from pathlib import Path
import collections,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gate_cost_diagnosis';N=OUT/'native_completed/output/job_8282'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ci(v):
 a=np.asarray(v,float);g=np.random.default_rng(0);b=a[g.integers(0,len(a),(10000,len(a)))].mean(1)
 return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)))
def main():
 s=json.loads((N/'summary.json').read_text());a=json.loads((N/'independent_audit.json').read_text());e=json.loads((N/'final_native_evidence.json').read_text());assert sha(N/'summary.json')==a['summary_sha256']==e['summary_sha256'];assert a['status']=='INDEPENDENT_GATE_GEOMETRY_AUDIT_PASS'
 with (N/'geometry_pairs.csv').open() as f:rows=list(csv.DictReader(f))
 C=OUT/'native_context/output/job_8282';context=json.loads((C/'context_scalar_diagnosis.json').read_text());assert context['source_sha256']==sha(OUT/'context_scalar_diagnosis.py')
 with (C/'context_pairs.csv').open() as f:cr=list(csv.DictReader(f))
 index={(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon'])):r for r in rows}
 checks=0
 for r in cr:
  rr=index[(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']))]
  assert abs(float(r['critical_feature'])-float(rr['critical_original_feature_mse']))<1e-12
  assert abs(float(r['mean_other_feature'])-float(rr['other_original_feature_mse']))<1e-12
  lo=float(r['mean_other_feature'])-float(r['other_feature_range'])/2;hi=float(r['mean_other_feature'])+float(r['other_feature_range'])/2;ct=float(r['critical_feature'])
  assert int(r['feature_context_sign_flip'])==int(lo<ct-1e-8 and hi>ct+1e-8)
  lt=float(r['mean_other_task'])-float(r['other_task_range'])/2;ht=float(r['mean_other_task'])+float(r['other_task_range'])/2;ct=float(r['critical_task'])
  assert int(r['task_context_sign_flip'])==int(lt<ct-1e-8 and ht>ct+1e-8)
  assert int(r['task_help_available'])==int(lt<ct-1e-8)
  assert int(r['helpful_vs_unhelpful_context'])==int(lt<ct-1e-8 and ht>=ct-1e-8);checks+=1
 assert checks==6912
 groups=collections.defaultdict(list)
 for r in rows:groups[(r['split'],r['pair'])].append(r)
 aggregates={};paired=[]
 for split in ('test_id','test_long_gap','test_distractor_shift'):
  pairs=sorted(pair for sp,pair in groups if sp==split);assert len(pairs)==128
  fields=[k for k in rows[0] if k.startswith(('critical_','other_'))]
  pr=[]
  for pair in pairs:
   rr=groups[(split,pair)];assert len(rr)==18
   pr.append(dict(split=split,pair=pair,**{k:float(np.mean([float(r[k]) for r in rr])) for k in fields}))
  paired.extend(pr);metrics={}
  for metric in ('total','parallel','orthogonal','common','contrast','basis_inside','basis_outside','task_error','margin_loss','score_error','label_score_error','signal_gain'):
   c=np.array([r['critical_'+metric] for r in pr]);o=np.array([r['other_'+metric] for r in pr]);metrics[metric]=dict(critical=float(c.mean()),other=float(o.mean()),delta=ci(c-o))
  m=-metrics['margin_loss']['other'];loss=metrics['label_score_error']['other']
  aggregates[split]=dict(metrics=metrics,mean_retained_signed_margin=m,retained_signed_margin_std=float(np.sqrt(max(0,loss-(m-1)**2))))
 output=dict(status='VERIFIED_POSTHOC_GATE_DIAGNOSIS',condition_aggregates=aggregates,context=context['overall'],local_context_rows_checked=checks,
  raw_reproduced=41472,max_old_error_difference=0,independent_intervals=936,original_Phase1_GO=False,GPU_seconds_added=0)
 (OUT/'readable_results.json').write_text(json.dumps(output,indent=2)+'\n')
 with (OUT/'pair_aggregates.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n');w.writeheader();w.writerows(paired)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axs=plt.subplots(1,3,figsize=(14,4.5),gridspec_kw={'width_ratios':[.95,1.2,1]})
 values=[11,40,3];labels=['Meets original rule','Mean points wrong way','Positive but <10%']
 axs[0].barh(np.arange(3),values,color=['.2','.55','.8']);axs[0].invert_yaxis();axs[0].set_yticks(range(3),labels);axs[0].set_xlim(0,47);axs[0].set_xlabel('Original feature checks (54 total)');axs[0].set_title('A. Why feature checks fail')
 for i,v in enumerate(values):axs[0].text(v+.6,i,str(v),va='center')
 splits=list(aggregates);x=np.arange(3)
 common=np.array([aggregates[sp]['metrics']['common']['delta']['mean'] for sp in splits]);contrast=np.array([aggregates[sp]['metrics']['contrast']['delta']['mean'] for sp in splits]);total=np.array([aggregates[sp]['metrics']['total']['delta']['mean'] for sp in splits])
 axs[1].axhline(0,color='.5',lw=.8);axs[1].bar(x-.16,common,width=.32,color='.65',label='Pair-common error');axs[1].bar(x+.16,contrast,width=.32,color='.2',label='Paired-future contrast error')
 low=np.array([aggregates[sp]['metrics']['total']['delta']['low'] for sp in splits]);high=np.array([aggregates[sp]['metrics']['total']['delta']['high'] for sp in splits]);axs[1].errorbar(x,total,yerr=[total-low,high-total],fmt='D',color='black',capsize=3,label='Total error (sum)');axs[1].set_ylabel('Cue deletion minus other deletions (MSE)');axs[1].legend(frameon=False,fontsize=8);axs[1].set_title('B. Shared and paired-contrast errors')
 critical=np.array([aggregates[sp]['metrics']['task_error']['critical'] for sp in splits])*100;other=np.array([aggregates[sp]['metrics']['task_error']['other'] for sp in splits])*100
 axs[2].bar(x-.17,critical,width=.34,color='.75',label='Delete cue');axs[2].bar(x+.17,other,width=.34,color='.15',label='Keep cue; mean two choices');axs[2].set_ylim(0,65);axs[2].set_ylabel('Original classification error (%)');axs[2].set_title('C. Better labels can coexist with worse MSE');axs[2].legend(frameon=False,fontsize=8)
 for i,v in enumerate(other):axs[2].text(i+.17,v+.7,f'{v:.1f}',ha='center',fontsize=9)
 for ax in axs[1:]:ax.set_xticks(x,['Ordinary','Long gap','Distractors'])
 fig.suptitle('Gate: diagnosing deletion cost without retraining',fontsize=14)
 fig.text(.01,-.04,'Panel B: positive means cue deletion hurts more; negative means keeping the cue increases that error component. Total = common + contrast.\nPanels B/C average seeds, cue positions and horizons within each base pair; the total-error CI resamples 128 pairs per condition. Original gates are unchanged.',fontsize=9)
 fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
 for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'gate_cost_diagnosis.{ext}',dpi=180,bbox_inches='tight')
 print(json.dumps(output,indent=2))
if __name__=='__main__':main()
