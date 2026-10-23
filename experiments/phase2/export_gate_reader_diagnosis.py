"""Publish completed reader diagnoses and recheck saved vectors without ML."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.phase2.export_literature_report import read,write,sha,scrub,source_info,style,save
import numpy as np
import matplotlib.pyplot as plt


def summarize(out):
    result={}
    for scope in ('global','local'):
        paired=read(out/scope/'paired_differences.json')
        original=[r for r in paired if r['arm']=='original']
        bad={r['episode_id'] for r in original if r['cue_minus_no_cue_mse']>1e-8}
        selected=[r for r in original if r['episode_id'] in bad]
        result[scope]={'comparable_episodes':len(original),'strict_no_cue_episodes':len(bad),
            'bad_episode_ids':sorted(bad),
            'selected_mean':{key:float(np.mean([r[key] for r in selected])) for key in selected[0] if key.startswith('cue_minus_')},
            'selected_deletion_improves_counts':{key:sum(r[key]>0 for r in selected) for key in
                ('cue_minus_no_cue_mse','cue_minus_no_cue_raw_feature_mse','cue_minus_no_cue_contrast_mse','cue_minus_no_cue_orthogonal_mse','cue_minus_no_cue_decoded_position_mse')},
            'reader_interventions':{arm:{'mean_cue_penalty_on_original_bad':float(np.mean([r['cue_minus_no_cue_mse'] for r in paired if r['arm']==arm and r['episode_id'] in bad])),
                'mean_cue_penalty_all':float(np.mean([r['cue_minus_no_cue_mse'] for r in paired if r['arm']==arm])),
                'strict_no_cue_cases_all':sum(r['cue_minus_no_cue_mse']>1e-8 for r in paired if r['arm']==arm)}
                for arm in ('original','no_transport','no_velocity','no_past_action','uniform_attention')}}
    parents=read(out/'local/parents.json');timing=read(out/'local/noncue_timing_audit.json')
    records=read(ROOT/'build/literature-baselines-20261022/fresh-data/records.json')
    byid={r['episode_id']:r for r in timing}
    result['timing']={'episodes':len(timing),'all_noncue_features_identical':sum(r['all_noncue_historical_features_identical'] for r in timing),
        'same_set_sibling_predictions_identical':sum(r['same_set_sibling_predictions_bitwise_identical'] for r in timing),
        'siblings_choose_different_best_nocue_times':sum(byid[r['id']]['best_times']!=byid[r['sibling']]['best_times'] for r in records),
        'mean_range':float(np.mean([r['loss_range_across_time_choices'] for r in timing])),
        'max_range':max(r['loss_range_across_time_choices'] for r in timing),
        'parents_without_complete_cue_deletion':sum(r['without_cue'] is None for r in parents)}
    return result


def render(out):
    (out/'figures').mkdir(parents=True, exist_ok=True)
    style();s=read(out/'diagnosis.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    x=np.arange(2);contrast=[s[k]['selected_mean']['cue_minus_no_cue_contrast_mse'] for k in ('global','local')]
    orth=[s[k]['selected_mean']['cue_minus_no_cue_orthogonal_mse'] for k in ('global','local')]
    axes[0].bar(x,contrast,label='Along sibling-future contrast',color='#237aa5')
    axes[0].bar(x,orth,bottom=contrast,label='Orthogonal remainder',color='#aeb5bc')
    axes[0].set(xticks=x,xticklabels=['Hindsight-selected\n46 / 192 episodes','Actual parent\n27 / 72 comparable'],ylabel='Cue minus no-cue feature MSE')
    axes[0].legend(fontsize=8)
    labels=['Raw features','Future contrast','Decoded position']
    for i,scope in enumerate(('global','local')):
        c=s[scope]['selected_deletion_improves_counts'];n=s[scope]['strict_no_cue_episodes']
        values=[c[k]/n for k in ('cue_minus_no_cue_raw_feature_mse','cue_minus_no_cue_contrast_mse','cue_minus_no_cue_decoded_position_mse')]
        axes[1].bar(np.arange(3)+(i-.5)*.34,values,.34,label=f'{scope} selected failures (n={n})',color=('#237aa5','#c55a35')[i])
    axes[1].set(xticks=np.arange(3),xticklabels=labels,ylim=(0,1.13),ylabel='Fraction also improved by cue deletion')
    axes[1].legend(fontsize=8)
    fig.suptitle('Cue-conditioned errors mostly track the wrong future branch')
    fig.text(.5,.01,'Selected failures, not population gains. Position readout is train-fitted and imperfect; contrast is evaluator-only.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94));save(fig,out/'figures','metric_decomposition')
    fig,ax=plt.subplots(figsize=(10,4.5))
    arms=['original','no_transport','no_velocity','no_past_action','uniform_attention']
    for scope,c in [('global','#237aa5'),('local','#c55a35')]:
        values=[s[scope]['reader_interventions'][a]['mean_cue_penalty_on_original_bad'] for a in arms]
        ax.plot(range(5),values,'o-',color=c,label=f'{scope}: original selected failures')
    ax.axhline(0,color='black',lw=.7);ax.set(xticks=range(5),xticklabels=['Original','No transport','No velocity','No past action','Uniform attention'],ylabel='Mean cue minus no-cue MSE',title='Frozen-reader interventions on the same fixed history sets')
    ax.legend(fontsize=9);fig.text(.5,.02,'Diagnostic path interventions can be out of distribution. No method was retrained or adopted.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));save(fig,out/'figures','reader_interventions')


def run(args):
    out=args.output.resolve()
    if args.render_only:
        render(out);return
    if out.exists() and any(out.iterdir()):raise ValueError('Require new output')
    out.mkdir(parents=True);raw={};verified=0;max_error=0.
    data=ROOT/'build/literature-baselines-20261022/fresh-data'
    records={r['id']:r for r in read(data/'records.json')}
    for scope,root in [('global',args.global_run),('local',args.local_run)]:
        dest=out/scope;dest.mkdir()
        for p in sorted(root.glob('*.json')):
            write(dest/p.name,scrub(read(p)));raw[f'{scope}/{p.name}']=source_info(p)
        raw[f'{scope}/predictions.npz']=source_info(root/'predictions.npz')
        rows=read(root/'episode_rows.json')
        with np.load(root/'predictions.npz',allow_pickle=False) as f:pred=f['predictions']
        assert len(pred)==len(rows)
        targetcache={}
        for row,p in zip(rows,pred):
            rid=row['episode_id'];rec=records[rid]
            if rid not in targetcache:
                with np.load(data/rec['feature_path'],allow_pickle=False) as a: y=a['z'][rec['query']+np.array([3,6])].astype(np.float64)
                sibling=records[rec['sibling']]
                with np.load(data/sibling['feature_path'],allow_pickle=False) as a:other=a['z'][rec['query']+np.array([3,6])].astype(np.float64)
                targetcache[rid]=(y,other)
            y,other=targetcache[rid];e=p.astype(np.float64)-y;direction=y-other
            coeff=np.sum(e*direction,-1)/np.sum(direction*direction,-1)
            along=np.mean((coeff[:,None]*direction)**2);orth=np.mean((e-coeff[:,None]*direction)**2)
            values={'mse':np.mean(e*e),'contrast_mse':along,'orthogonal_mse':orth}
            for k,v in values.items():
                error=abs(v-row['metrics'][k]);max_error=max(max_error,error)
                assert error<1e-12,(scope,rid,k)
            verified+=1
    write(out/'diagnosis.json',summarize(out))
    write(out/'audit.json',{'prediction_rows_independently_recomputed':verified,'mse_and_contrast_max_error':max_error,
        'model_calls':0,'scope':'Saved prediction arithmetic plus original full model replay audit in global/independent_review.json'})
    write(out/'failed_attempts.json',{
        'metric-reader-v1':'Stopped before fresh predictions: output_basis and target_mean were not in the narrow input projection dictionary. Fixed by using verified frozen-model buffers; complete run is v2.',
        'reader-local-v1':'Stopped because some parents retain two equivalent cue copies under every one-step deletion. v2 explicitly reports these 120 parents as ineligible for a cue-vs-no-cue contrast; none are silently counted as zero regret.',
        'original_attempt_protocols_retained_in_build':True,'scientific_results_from_complete_v2_only':True})
    (out/'README.md').write_text('''# Gate predictor / metric diagnosis

This is exploratory diagnosis of the existing 192 development episodes, with
the predictor, normalization and all online retention policies frozen.

- [Complete global comparisons](global/episode_rows.json) and [actual-parent comparisons](local/episode_rows.json).
- [Diagnosis counts](diagnosis.json), [arithmetic audit](audit.json), [independent frozen-model replay](global/independent_review.json).
- [Local candidate sets](local/parents.json), [identical noncue-feature timing audit](local/noncue_timing_audit.json).
- [Position-readout validation](global/readout_validation.json), [original unstable cue probes](local/cue_probe.json).
- [Feature-error decomposition](figures/metric_decomposition.png), [reader interventions](figures/reader_interventions.png).

All candidate-set choices are tied to the original loss. Reader interventions
do not reoptimize sets and are not a repaired method. The original cue probe has
severe train-support scaling instability; consult the input-support diagnosis
before interpreting it as evidence about representation information.

Prediction arrays, fitted evaluation readout and original inputs remain under
the hash-recorded build paths. Replot portable figures without them:

```sh
python experiments/phase2/export_gate_reader_diagnosis.py --render-only
```
''')
    render(out)
    write(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'raw_sources':raw,
        'source_files':{str(p.relative_to(ROOT)):source_info(p) for p in [Path(__file__),ROOT/'experiments/phase2/diagnose_gate_oracle_metric.py',ROOT/'experiments/phase2/diagnose_gate_reader_local.py']},
        'published_files':{str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({'files':len(list(out.rglob('*'))),'rows_checked':verified,'maximum_error':max_error}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render-only',action='store_true')
    p.add_argument('--global-run',type=Path,default=ROOT/'build/literature-diagnosis-20261022/metric-reader-v2')
    p.add_argument('--local-run',type=Path,default=ROOT/'build/literature-diagnosis-20261022/reader-local-v2')
    p.add_argument('--output',type=Path,default=ROOT/'results/literature_diagnosis/metric-reader-20261022')
    run(p.parse_args())
