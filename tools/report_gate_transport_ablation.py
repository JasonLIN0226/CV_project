"""Paired descriptive report for the one frozen transport-off intervention."""
import collections,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'results/gate_transport_ablation'
def ci(v):
 a=np.asarray(v,float);ix=np.random.default_rng(0).integers(0,len(a),(10000,len(a)));lo,hi=np.quantile(a[ix].mean(1),[.025,.975]);return dict(mean=float(a.mean()),low=float(lo),high=float(hi))
def main():
 status=json.loads((D/'status.json').read_text());N=D/'native_completed';O=N/'output'/f"job_{status['job']}"
 s=json.loads((O/'summary.json').read_text());audit=json.loads((O/'independent_audit.json').read_text());assert audit['status']=='PASS_INDEPENDENT_TWO_ARM_ABLATION_AUDIT' and audit['summary_sha256']==hashlib.sha256((O/'summary.json').read_bytes()).hexdigest()
 with (O/'paired_validation.csv').open() as f:rows=list(csv.DictReader(f))
 ids=sorted({r['pair'] for r in rows});assert len(ids)==96 and len(rows)==14976
 by=collections.defaultdict(list)
 for r in rows:by[(r['arm'],r['pair'],r['variant'])].append(r)
 arms=('before','transport_off');groups=('cue_oldest','cue_middle','cue_newest');fields=('feature_mse','task_error','common','contrast')
 counts={arm:{k:sum(v[k] for v in s['arms'][arm]['by_pool_group'].values()) for k in ('feature','task','both','cells')} for arm in arms}
 groupstats={};change={};context=s['arms']
 for arm in arms:
  groupstats[arm]={}
  for g in groups:
   vals=collections.defaultdict(list)
   for pair in ids:
    rr=[r for v in (f'{g}/delete_{j}' for j in range(3)) for r in by[(arm,pair,v)]]
    cr=[r for r in rr if not int(r['cue_present'])];ke=[r for r in rr if int(r['cue_present'])];non=[r for r in ke if not int(r['native_privileged'])]
    assert len(cr)==6 and len(ke)==12
    for f in fields:
     c=np.mean([float(r[f]) for r in cr]);k=np.mean([float(r[f]) for r in ke]);n=np.mean([float(r[f]) for r in non]);native=np.mean([float(r[f]) for r in by[(arm,pair,'privileged_small_budget')]])
     for name,val in [('delete',c),('keep',k),('delta',c-k),('penalty',n-native)]:vals[f'{name}_{f}'].append(val)
   groupstats[arm][g]={k:ci(v) for k,v in vals.items()}
 # Off minus before on exactly matched base pairs; average all seeds/horizons first.
 for variant in ('recent_only','full_history','privileged_small_budget','random_small_budget',*groups):
  values=collections.defaultdict(list)
  for pair in ids:
   vs=[variant] if variant not in groups else [f'{variant}/delete_{j}' for j in range(3)]
   for f in fields:
    mean=lambda ar:np.mean([float(r[f]) for v in vs for r in by[(ar,pair,v)]])
    values[f].append(mean('transport_off')-mean('before'))
  change[variant]={f:ci(v) for f,v in values.items()}
 result=dict(counts=counts,groupstats=groupstats,off_minus_before=change,validation_pairs=96,bootstrap=10000,unit='base pair; average seeds/horizons first',new_GPU=0,original_Phase1_GO=False)
 (D/'readable_results.json').write_text(json.dumps(result,indent=2)+'\n')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,3,figsize=(13,4.5));x=np.arange(3);labels=['Cue oldest','Cue middle','Cue newest']
 for ar,shift,color,lab in [('before',-.18,'.7','Original'),('transport_off',.18,'.15','Transport off')]:
  vals=[s['arms'][ar]['by_pool_group'][g]['both'] for g in groups];axs[0].bar(x+shift,vals,.34,color=color,label=lab)
  for i,v in enumerate(vals):axs[0].text(i+shift,v+.12,f'{v}/6',ha='center',fontsize=9)
  for ax,f in [(axs[1],'feature_mse'),(axs[2],'task_error')]:
   v=[groupstats[ar][g]['delta_'+f] for g in groups];means=np.array([z['mean'] for z in v]);err=np.array([[z['mean']-z['low'] for z in v],[z['high']-z['mean'] for z in v]])
   ax.errorbar(x+shift,means,yerr=err,fmt='o' if ar=='before' else 's',color=color,capsize=3,label=lab)
 axs[0].set(ylim=(0,7),yticks=[0,2,4,6],ylabel='Joint diagnostic cells',title='A. Same numeric criteria')
 for ax,title in [(axs[1],'B. Feature deletion effect'),(axs[2],'C. Task deletion effect')]:ax.axhline(0,color='.5',ls='--');ax.set(title=title,ylabel='Delete cue error − keep cue error')
 for ax in axs:ax.set_xticks(x,labels,rotation=12)
 fig.suptitle('Gate frozen validation: does removing temporal transport repair deletion utility?',fontsize=12)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.87));F=D/'figures';F.mkdir(exist_ok=True)
 fig.savefig(F/'transport_ablation.png',dpi=190);fig.savefig(F/'transport_ablation.pdf');plt.close(fig)
 table='\n'.join(f"| {g.replace('_',' ')} | {s['arms']['before']['by_pool_group'][g]['both']}/6 | {s['arms']['transport_off']['by_pool_group'][g]['both']}/6 |" for g in groups)
 contexts='\n'.join(f"| {v.replace('_',' ')} | {s['arms']['before']['context_aggregates'][v]['feature_mse']['mean']:.5f} | {s['arms']['transport_off']['context_aggregates'][v]['feature_mse']['mean']:.5f} | {s['arms']['before']['context_aggregates'][v]['task_error']['mean']:.1%} | {s['arms']['transport_off']['context_aggregates'][v]['task_error']['mean']:.1%} |" for v in ('recent_only','full_history','privileged_small_budget','random_small_budget'))
 both0=counts['before']['both'];both1=counts['transport_off']['both']
 text=f'''# Gate temporal-transport ablation — frozen validation only

## Objective and intervention

Test whether switching off the Gate predictor's temporal transport contribution, without training, restores useful deletion costs. The original model and transport-off model use identical weights, inputs, content branch and frozen recent predictor. A temporary hook zeros the complete transport projection output. This removes both its last-observation component and its slope-times-time component; it is not a slope-only test.

## Data and design

Reuse the original 96 validation pairs (192 episodes), frozen seeds17/29/43, horizons3/6 and cached normalized I-JEPA features. Four original contexts plus nine equal-capacity deletion forks yield29952 episode scores and14976 paired scores across two arms. Each fork starts with three historical frames and deletes one; all compared forks retain two past frames plus four recent frames. Cue-oldest/middle/newest refer to the cue's temporal position within that three-frame parent pool. No data encoding, fitting, optimizer, training or test-set inference. New GPU use:0seconds.

## Results

Joint feature-and-task deletion diagnostic cells: **{both0}/18 → {both1}/18**. Each cell is one seed×horizon×cue-position group. A cell retains the original relative-effect threshold10% and paired95%lower bound>0 for both feature MSE and task error. These are validation diagnostics; they do not replace the original held-out acceptance tests.

| Parent pool | Original | Transport off |
|---|---:|---:|
{table}

![Frozen transport ablation](../results/gate_transport_ablation/figures/transport_ablation.png)

**How to read the figure.** A counts cells meeting both unchanged numeric conditions. B and C show the error after deleting the cue minus the average error after deleting a noncue frame. Positive values mean removing the cue is worse, the expected importance ordering. Bars are paired95%bootstrap intervals across96 base pairs after averaging seeds/horizons; they are descriptive pooled effects, not individual-cell pass counts. Negative values mean the deletion cost ranks the cue in the wrong direction.

### Original-context prediction quality

| Context | Original feature MSE | Off feature MSE | Original task error | Off task error |
|---|---:|---:|---:|---:|
{contexts}

Lower error is better. Task error is the frozen readout's binary classification error on predicted features, not true-feature classification accuracy. The same recent-only baseline is used by both arms. Lower feature MSE without preserving useful cue responses is not sufficient to establish a repair.

## Verification and limits

All14976 original baseline episode scores reproduced the independently audited parent. Every paired forward had identical content outputs and reference outputs; complete weight/buffer state digests were unchanged for all three seeds. Maximum transport-subtraction residual:{s['max_transport_subtraction_residual']:.3g}. Empty-history outputs were exactly unchanged. Independent raw pairing, intervals,156 primitive prediction/target vector witnesses, source/assets and native transfer hashes were verified. No thresholds or model weights changed.

Only this one predeclared ablation was evaluated. The branches were trained jointly, so removing one branch may reveal co-adaptation. An unsuccessful removal does not prove temporal extrapolation is harmless; a successful validation removal would still need the unchanged original test evaluation before a broader claim. Original Phase1_GO and held-out deletion results are unchanged by this diagnostic. Fresh9137 remains closed; no retention policy was trained.

## Interpretation

**Switching off the complete transport branch repairs this validation deletion diagnostic:7/18 →18/18.** The native two-frame task error also improves31.16%→26.48%, while recent-only remains50%; the gain is not produced by erasing all cue sensitivity. This is direct intervention evidence that the transport contribution causes the observed diagnostic failures in the frozen model on these validation inputs.

**It is not a universal predictor repair.** With full history, task error rises3.13%→26.39% and feature MSE rises0.04625→0.07265. The paired task-error increase is23.26percentage points (95%CI17.10–29.17). Thus the same branch is useful with dense history and harmful for deletion utility under the tested two-frame subsets. This result localizes a composition-dependent branch problem; it does not isolate the elapsed-time slope from the rest of that branch or establish the unique cause of every previous failure.

A defensible next step is an explicitly frozen, validation-led distinction between the slope-times-time component and the rest of transport, or a separately declared sparse-memory correction. Neither has been executed here. Do not silently adopt a capacity-dependent switch, search more variants on these results, or claim Phase1 acceptance from these18cells. The unchanged held-out deletion checks, broader original task criteria and independent confirmation remain unverified for the ablated predictor.

Native provenance: CPU job{status['job']}; evaluator and independent audit return0; source lock `{status['source_lock_sha256']}`. Compact evidence is under `results/gate_transport_ablation/native_completed/`; weights and primitive arrays stay on HPC.
'''
 (R/'docs/GATE_TRANSPORT_ABLATION.md').write_text(text)
 print(json.dumps(dict(counts=counts,original_contexts={a:{v:{f:s['arms'][a]['context_aggregates'][v][f]['mean'] for f in fields[:2]} for v in ('recent_only','full_history','privileged_small_budget','random_small_budget')} for a in arms},off_minus_before=change),indent=2))
if __name__=='__main__':main()
