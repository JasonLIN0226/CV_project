"""Export completed WR diagnostic bytes and plots; no model calls or fitting."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")


def redact(x,root):
    if isinstance(x,dict):return {str(k).replace(str(root)+"/","repository/"):redact(v,root) for k,v in x.items()}
    if isinstance(x,list):return [redact(v,root) for v in x]
    if isinstance(x,str):return x.replace(str(root)+"/","repository/")
    return x


def render(out):
    summary=json.loads((out/"summary.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    strata=[r for r in summary['wr_strata'] if r['split']=='fresh_development']
    x=np.arange(2);width=.32
    axes[0].bar(x-width/2,[r['cold_cue_survived']/r['episodes'] for r in strata],width,label='Original WR',color='#cf7656')
    axes[0].bar(x+width/2,[r['prefix_cue_survived']/r['episodes'] for r in strata],width,label='Shared initial prefix',color='#4d94ad')
    axes[0].set_xticks(x,['Early cue t=2/3\n62 episodes','Later cue t=4..7\n130 episodes'])
    axes[0].set_ylim(0,1.12);axes[0].set_ylabel('Cue retained at query');axes[0].legend(fontsize=9)
    axes[0].set_title('Startup intervention rescues all early cues')
    vals=[r for r in summary['by_condition_policy'] if r['policy']=='wr' and r['condition']!='old_train']
    y=np.arange(3)
    means=[r['prefix_minus_cold_mse']['mean'] for r in vals]
    lo=[m-r['prefix_minus_cold_mse']['ci95'][0] for m,r in zip(means,vals)]
    hi=[r['prefix_minus_cold_mse']['ci95'][1]-m for m,r in zip(means,vals)]
    axes[1].errorbar(means,y,xerr=[lo,hi],fmt='o',color='#4d94ad',capsize=4)
    axes[1].axvline(0,color='.6',ls='--');axes[1].set_yticks(y,['ID','Long gap','Distractor shift']);axes[1].invert_yaxis()
    axes[1].set_xlabel('Prefix minus original feature MSE\nMean and 95% scene-pair bootstrap CI')
    axes[1].set_title('Better retention does not guarantee lower error')
    fig.suptitle('WR startup isolation: fixed predictor, R=4, K=2, same prefix for every policy',fontsize=13)
    (out/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out/'figures'/f'wr_startup.{ext}',dpi=180)
    plt.close(fig)


def export(root,raw,out):
    out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((raw/'episode_results.json').read_text())
    summary=json.loads((raw/'summary.json').read_text())
    indexed={(r['episode_id'],r['policy'],r['prefix']):r for r in rows}
    old=json.loads((root/'results/literature_baselines/fresh-20261022/episode_results.json').read_text())
    oldidx={(r['episode_id'],r['policy']):r for r in old if r['seed']==17}
    cold=[r for r in rows if r['split']=='fresh_development' and r['prefix']==0]
    assert len(cold)==768
    for r in cold:
        expected=oldidx[(r['episode_id'],r['policy'])]
        assert r['mean_mse']==expected['mean_mse'] and r['kept_times']==expected['kept_times']
    with np.load(raw/'prediction_vectors.npz',allow_pickle=False) as data:
        losses=np.mean((data['predictions'].astype(np.float64)-data['targets'].astype(np.float64))**2,axis=-1)
    assert np.array_equal(losses,np.array([r['feature_mse'] for r in rows]))
    event_count=0
    with gzip.open(raw/'wr_trace.jsonl.gz','rt') as f:
        for line in f:
            e=json.loads(line);event_count+=1
            assert len(e['retained'])<=2 and len(e['recent'])<=4
            assert not set(e['retained']) & set(e['recent'])
    early=[r for r in cold if r['policy']=='wr' and r['cue_original_time']<=3]
    deltas=[indexed[(r['episode_id'],'wr',4)]['mean_mse']-r['mean_mse'] for r in early]
    comparison=[]
    for r in cold:
        warm=indexed[(r['episode_id'],r['policy'],4)]
        comparison.append({'episode_id':r['episode_id'],'pair':r['pair'],'condition':r['condition'],
                           'policy':r['policy'],'prefix_minus_cold_mse':warm['mean_mse']-r['mean_mse'],
                           'same_original_kept_set':warm['kept_original_times']==r['kept_original_times']})
    audit={'arithmetic_rows':len(rows),'horizons_per_row':2,'recomputed_mse_max_error':0,
           'original_fresh_rows_reproduced_exactly':len(cold),'wr_stream_events_checked':event_count,
           'frozen_model_clock_shift_checks':summary['frozen_model_shift_checks'],
           'shift_prediction_max_abs_error':summary['shift_prediction_max_abs_error'],
           'early_fresh_cues_rescued':sum(indexed[(r['episode_id'],'wr',4)]['cue_survived'] for r in early),
           'rescued_episodes_feature_mse_better':sum(d<-1e-12 for d in deltas),
           'rescued_episodes_feature_mse_worse':sum(d>1e-12 for d in deltas),
           'no_new_model_execution':True,'no_new_fitting':True}
    originals={}
    for name in ('protocol.json','summary.json','episode_results.json','shift_invariance.json','paired_examples.json','wr_trace.jsonl.gz'):
        source=raw/name; originals[name]={'sha256':sha(source),'bytes':source.stat().st_size}
        if source.suffix=='.json':write(out/name,redact(json.loads(source.read_text()),root))
        else:shutil.copyfile(source,out/name)
    write(out/'audit.json',audit);write(out/'paired_policy_comparisons.json',comparison)
    src=out/'source';src.mkdir(exist_ok=True)
    shutil.copyfile(root/'experiments/phase2/diagnose_wr_startup.py',src/'diagnose_wr_startup.py')
    render(out)
    write(out/'manifest.json',{'schema_version':1,'scope':'completed matched-prefix diagnostic; exact source/runtime preserved',
        'raw_directory_relative_to_repository':str(raw.relative_to(root)),
        'raw_sources':originals,
        'excluded_prediction_vectors':{'path':str((raw/'prediction_vectors.npz').relative_to(root)),
                                      'sha256':sha(raw/'prediction_vectors.npz'),'bytes':(raw/'prediction_vectors.npz').stat().st_size},
        'transformations':'Absolute checkout prefix changed to repository/ in published JSON keys/values; no numeric changes. Derived audit/statistics and plots contain no model calls.',
        'published_files':{str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    if args.render_only:render(args.output)
    else:export(args.root.resolve(),args.raw.resolve(),args.output.resolve())
