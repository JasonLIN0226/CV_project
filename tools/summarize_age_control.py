"""Summarize verified native outputs; no models, fitting or publication."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'results/age_control/native_completed/output/job_8275'
OUT=ROOT/'results/age_control'
def main():
 s=json.loads((NATIVE/'summary.json').read_text());a=json.loads((NATIVE/'independent_audit.json').read_text());p=json.loads((NATIVE/'final_native_evidence.json').read_text())
 digest=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 assert a['summary_sha256']==p['summary_sha256']==digest(NATIVE/'summary.json')
 assert a['status']=='INDEPENDENT_RAW_SCALAR_AUDIT_PASS' and p['status']=='FINAL_NATIVE_COMPLETION_VERIFIED'
 assert s['raw_rows']==124416 and s['paired_rows']==9216 and a['independent_scalar_statistics']==1296
 gates=[r for r in s['gate_critical_contrasts'] if r['seed']!='mean']
 gains=[r for r in s['oracle_baseline_comparisons'] if r['choice']=='oracle_transfer']
 result={'verified_raw_rows':124416,'verified_paired_rows':9216,'independently_recomputed_statistics':1296,'Gate_by_cue_position':{},'Motion_cross_horizon':{},'new_GPU_seconds':0,'original_Phase1_GO':False,'learned_policy_tested':False,'fresh_confirmation_tested':False}
 for group in ('cue_oldest','cue_middle','cue_newest'):
  rr=[r for r in gates if r['pool_group']==group]
  result['Gate_by_cue_position'][group]={'passed':sum(r['passed'] for r in rr),'total':len(rr),'feature_passed':sum(r['metrics']['feature_mse']['passed'] for r in rr),'task_passed':sum(r['metrics']['task_error']['passed'] for r in rr),'seed_mean_passed':sum(r['passed'] for r in s['gate_critical_contrasts'] if r['seed']=='mean' and r['pool_group']==group),'seed_mean_total':6}
 for baseline in ('uniform','fifo','reject'):
  rr=[r for r in gains if r['task']=='motion' and r['seed']!='mean' and r['baseline']==baseline]
  result['Motion_cross_horizon'][baseline]={'positive_both':sum(r['positive_both_metrics'] for r in rr),'total':len(rr)}
 (OUT/'readable_results.json').write_text(json.dumps(result,indent=2)+'\n')
 with (OUT/'comparison_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['task','seed','split','pool_group','horizon','baseline','metric','baseline_mean','chosen_mean','paired_gain','CI_low','CI_high','relative_gain']);w.writeheader()
  for r in gains:
   for m,v in r['metrics'].items():
    g=v['paired_gain'];w.writerow({**{k:r[k] for k in ('task','seed','split','pool_group','horizon','baseline')},'metric':m,'baseline_mean':v['baseline_mean'],'chosen_mean':v['chosen_mean'],'paired_gain':g['mean'],'CI_low':g['low'],'CI_high':g['high'],'relative_gain':v['relative_gain']})
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axs=plt.subplots(1,3,figsize=(13,4.3),gridspec_kw={'width_ratios':[.9,1.2,1.2]})
 groups=list(result['Gate_by_cue_position']);counts=[result['Gate_by_cue_position'][g]['passed'] for g in groups]
 axs[0].bar(np.arange(3),counts,color='0.2',width=.6);axs[0].set_xticks(range(3),['Oldest','Middle','Newest']);axs[0].set_ylim(0,21);axs[0].set_yticks([0,6,12,18]);axs[0].set_ylabel('Checks meeting both original effect rules');axs[0].set_xlabel('Position of the informative cue');axs[0].set_title('A. Gate: does cue deletion hurt?')
 for i,n in enumerate(counts):axs[0].text(i,n+.6,f'{n}/18',ha='center')
 splits=('test_id','test_long_gap','test_distractor_shift')
 for ax,metric,title in zip(axs[1:],('feature_mse','task_error'),('B. Motion: future-feature error','C. Motion: position error')):
  ax.axhline(0,color='.65',lw=1)
  for j,(baseline,label,marker,color) in enumerate((('uniform','Uniform deletion','o','black'),('fifo','Delete oldest','s','.4'),('reject','Delete newest','^','.7'))):
   values=[];lo=[];hi=[]
   for split in splits:
    r=next(r for r in gains if r['task']=='motion' and r['seed']=='mean' and r['horizon']==6 and r['split']==split and r['baseline']==baseline)
    v=r['metrics'][metric];b=v['baseline_mean'];g=v['paired_gain'];values.append(100*g['mean']/b);lo.append(100*(g['mean']-g['low'])/b);hi.append(100*(g['high']-g['mean'])/b)
   ax.errorbar(np.arange(3)+(j-1)*.15,values,yerr=[lo,hi],fmt=marker,color=color,capsize=3,ms=5,label=label)
  ax.set_xticks(range(3),['Ordinary','Long gap','Distractors']);ax.set_title(title);ax.set_ylabel('Improvement over baseline (%)');ax.set_xlabel('Choose deletion at h=3; score at h=6')
 axs[2].legend(frameon=False,fontsize=8)
 fig.suptitle('Same-capacity deletion controls with frozen predictors',fontsize=13,y=1.01)
 fig.text(.01,-.03,'Gate: 18 checks per cue position = 3 seeds x 3 conditions x 2 horizons. Motion: seed-mean paired gains; bars are 95% bootstrap CIs over 128 pairs.\nGains and CI bounds are rescaled by the observed baseline mean. Positive values favor the future-informed choice; this is not a learned retention policy.',fontsize=9)
 fig.tight_layout();folder=OUT/'figures';folder.mkdir(exist_ok=True)
 for ext in ('png','pdf'):fig.savefig(folder/f'age_control_summary.{ext}',dpi=180,bbox_inches='tight')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
