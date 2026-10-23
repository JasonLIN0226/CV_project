"""Readable original-gate recheck; preserves all revealed-development limitations."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];D=B/'results/gate_locked_test'
def main():
 status=json.loads((D/'status.json').read_text());O=D/'native_completed/output'/f"job_{status['job']}";s=json.loads((O/'summary.json').read_text());a=json.loads((O/'independent_audit.json').read_text());assert a['summary_sha256']==hashlib.sha256((O/'summary.json').read_bytes()).hexdigest() and a['status']=='PASS_INDEPENDENT_GATE_LOCKED_TEST_AUDIT'
 old=json.loads((B/'results/age_control/native_completed/output/job_8275/summary.json').read_text());oldpred=json.loads((B/'results/predictor/evaluation/job_7575/gate/development_comparison.json').read_text());oldmetrics=json.loads((B/'results/predictor/evaluation/job_7575/gate/evaluation/metrics.json').read_text());newmetrics=json.loads((O/'original/evaluation/metrics.json').read_text())
 splits=('test_id','test_long_gap','test_distractor_shift');labels=['Ordinary','Long gap','Distractor shift']
 cells=s['deletion']['cells'];oldcells=old['gate_critical_contrasts'];coverage=[]
 for sp in splits:
  before=[x for x in oldcells if x['split']==sp and x['seed']!='mean'];after=[x for x in cells if x['split']==sp and x['seed']!='mean'];assert len(before)==len(after)==18
  coverage.append(dict(split=sp,before=sum(x['passed'] for x in before),off=sum(x['passed'] for x in after),total=18))
 prediction=s['prediction'];originalbefore=sum(x['passed'] for x in oldpred['original_checks']);fixedbefore=sum(x['passed'] for x in oldpred['fixed_reference_checks'])
 def taskerror(metrics,sp,condition):return 1-np.mean([r['metrics']['state_accuracy']['mean'] for r in metrics['per_seed'] if r['split']==sp and r['condition']==condition])
 errors={sp:{v:{arm:float(taskerror(m,sp,v)) for arm,m in [('before',oldmetrics),('transport_off',newmetrics)]} for v in ('recent_only','full_history','privileged_small_budget')} for sp in splits}
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(1,3,figsize=(12,4.5));x=np.arange(3)
 for shift,arm,color,label in [(-.18,'before','.7','Original'),(.18,'off','.15','Transport off')]:
  values=[v[arm] for v in coverage];axs[0].bar(x+shift,values,.34,color=color,label=label)
  for i,y in enumerate(values):axs[0].text(i+shift,y+.4,f'{y}/18',ha='center',fontsize=8)
 axs[0].set(ylim=(0,21),yticks=[0,6,12,18],xticks=x,xticklabels=labels,ylabel='Joint deletion checks',title='A. All original test conditions')
 oldvalues=[originalbefore,fixedbefore];newvalues=[prediction['original_checks_passed'],prediction['fixed_checks_passed']];den=[51,32]
 for shift,values,color in [(-.18,oldvalues,'.7'),(.18,newvalues,'.15')]:
  heights=np.array(values)/den*100;axs[1].bar(np.arange(2)+shift,heights,.34,color=color)
  for i,y in enumerate(heights):axs[1].text(i+shift,y+1.2,f'{values[i]}/{den[i]}',ha='center',fontsize=8)
 axs[1].set(ylim=(0,115),yticks=[0,50,100],xticks=np.arange(2),xticklabels=['Original prediction','Fixed reference'],ylabel='Checks met (%)',title='B. Prediction requirements')
 for condition,marker,label in [('full_history','o','Full history'),('privileged_small_budget','s','Two-frame cue memory')]:
  for arm,color,ls in [('before','.6','--'),('transport_off','.1','-')]:axs[2].plot(x,[errors[sp][condition][arm]*100 for sp in splits],marker=marker,color=color,ls=ls,label=label+(' / original' if arm=='before' else ' / off'))
 axs[2].set(xticks=x,xticklabels=labels,ylabel='Task error (%) — lower is better',title='C. Forecast trade-off');axs[2].legend(fontsize=7,frameon=False)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=2,frameon=False)
 for ax in axs:ax.tick_params(axis='x',labelrotation=15)
 fig.tight_layout(rect=(0,0,1,.92));F=D/'figures';F.mkdir(exist_ok=True);fig.savefig(F/'locked_gate_test.png',dpi=190);fig.savefig(F/'locked_gate_test.pdf');plt.close(fig)
 result=dict(status=s['status'],prediction=prediction['original_checks_passed'],fixed=prediction['fixed_checks_passed'],deletion=s['deletion']['individual'],deletion_seedmean=s['deletion']['seedmean'],coverage=coverage,task_errors=errors,original_task_status=prediction['original_task_status'],full_history_rule_met=a['full_history_rule_met'],structural_rules_met=a['structural_rules_met'],new_GPU=0,original_Phase1_GO=False)
 (D/'readable_results.json').write_text(json.dumps(result,indent=2)+'\n')
 table='\n'.join(f"| {lab} | {v['before']}/18 | {v['off']}/18 |" for lab,v in zip(labels,coverage));failed=[x for x in prediction['original_checks'] if not x['passed']]+[x for x in prediction['fixed_reference_checks'] if not x['passed']]
 failtext='None among the named prediction checks.' if not failed else '\n'.join('- '+x['kind']+' — '+x.get('slice',x.get('split','')) for x in failed)
 conclusion='The locked candidate meets the complete original Gate development repair conjunction.' if s['status']=='GATE_DEVELOPMENT_REPAIR_SUPPORTED' else 'Gate is not fully repaired under the unchanged development requirements; retain every failed check.'
 text=f'''# Gate locked transport-off test

## Research question

Does the validation-selected, frozen transport-off version generalize to all original Gate development tests and satisfy the original prediction requirements? The practical objective is reliable deletion cost at a fixed two-frame memory capacity. Prediction accuracy is also checked so a favorable deletion ranking is not mistaken for a useful overall predictor.

## Locked design

The entire `transport_values` output is zero for every input context, with no capacity-dependent switch. The content branch, three checkpoints, recent reference, cached I-JEPA features, train-only normalization/basis/readout and all statistical thresholds remain unchanged. The intervention was locked after validation8327 and before this execution. No training, fitting, test-driven tuning, new encoding, policy optimization or fresh9137 data.

Ordinary means the original in-distribution test. Long gap increases the delay between useful evidence and the query. Distractor shift changes the distractor setting. All three original test splits contain128base pairs each; retain seeds17/29/43 and horizons3/6. In each cue-oldest/middle/newest parent pool, delete one of three historical frames and retain exactly two plus four recent frames. The correct expected ordering is that deleting the true cue increases both future-feature error and task error more than deleting a noncue frame.

Full deletion test:41472episode outcomes,6912paired rows,54individual-seed cells and18seed-mean cells. A cell meets the unchanged10%relative effect and paired95%lower>0 requirements in both errors. Bootstrap10000base-pair draws, seed0. A cell count is coverage across correlated checks, not54independent datasets.

Original prediction evaluation:34560raw outcomes across validation plus all three test splits, all six original conditions (recent/full/privileged two-frame/random two-frame/wrong evidence/persistence). The original scoring loop is AST-identical. Original bootstrap and rules are reused verbatim, including51named checks,32fixed-reference checks, full-history tolerance and structural checks. Unused timing probes were omitted. New GPU use:0seconds.

## Results

| Check | Original model | Locked transport off |
|---|---:|---:|
| Original prediction checks | {originalbefore}/51 | {prediction['original_checks_passed']}/51 |
| Fixed-reference checks | {fixedbefore}/32 | {prediction['fixed_checks_passed']}/32 |
| Joint deletion effects, individual seeds | 10/54 | {s['deletion']['individual']['both']}/54 |
| Joint deletion effects, seed mean | 1/18 | {s['deletion']['seedmean']['both']}/18 |

Original statistical decision: **{prediction['original_task_status']}**. Full-history tolerance satisfied: **{a['full_history_rule_met']}**. Structural safeguards satisfied: **{a['structural_rules_met']}**.

| Test condition | Original joint deletion cells | Transport-off joint cells |
|---|---:|---:|
{table}

![Locked Gate test](../results/gate_locked_test/figures/locked_gate_test.png)

**Figure guide.** A shows deletion reliability coverage separately for the three test conditions. B shows the fraction and exact count of named prediction requirements met. C shows the task-error trade-off: solid lines are transport off, dashed lines the original model; circles use full history and squares use the original two-frame cue memory. Lower error is better. A decline against the old model is not automatically a gate failure: the original thresholds, not a newly invented noninferiority rule, decide acceptance.

### Prediction checks not met

{failtext}

### Other original decision reasons

{chr(10).join('- '+x for x in prediction['reasons']) or 'None.'}

## Verification

One course CPU allocation{status['job']},4scientific threads,16GB,0GPU. Original weights and buffers match the validation-locked state digests throughout both evaluation stages. Content and recent-reference outputs are identical before/off on paired inputs, and the output difference matches the removed transport contribution. All41472old deletion scores replay within execution tolerance; failures are preserved. Independent reconstruction verifies34560prediction rows,51original checks,32fixed-reference comparisons and72deletion cells;324primitive prediction/target vectors are checked. Source, assets and compact native-before/local/native-afterSHA evidence are retained.

Source lock:`{status['source_lock_sha256']}`. Candidate lock:`{status['candidate_lock_sha256']}`. Native evidence:`results/gate_locked_test/native_completed/`. Weights/features/arrays remain on HPC; GitHub was not updated.

## Conclusion

**{conclusion}** This is revealed6301development evidence: previous tests were already seen during earlier work. It is not independent confirmation, and it does not alter Motion's prior failed prediction checks. Full Proposal Phase1_GO therefore remains unestablished. Fresh9137 remains closed and no retention policy has been trained. No follow-up switch or alternative candidate was selected from these test results.
'''
 (B/'docs/GATE_LOCKED_TEST.md').write_text(text);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
