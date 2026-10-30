"""Fixed actual-parent reader diagnosis and privileged cue-information probes."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'experiments/predictor/control/src'): sys.path.insert(0,str(p))
import numpy as np
import torch
from experiments.phase2.diagnose_gate_oracle_metric import (ARMS,sha,save,fit_readout,decode,
    reader_intervention,metrics,group_ci)
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from phase1.model import project


def cue_partition(parent, equivalent, losses):
    alternatives=list(combinations(sorted(parent),2))
    yes=[(losses[k],k) for k in alternatives if equivalent.intersection(k)]
    no=[(losses[k],k) for k in alternatives if not equivalent.intersection(k)]
    return min(yes) if yes else None, min(no) if no else None


def run(args):
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Require empty output')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);began=time.monotonic()
    protocol={'version':'actual-attention-parent-reader-diagnosis-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'second targeted diagnostic after global oracle analysis; exploratory development',
        'source_sha256':sha(__file__),'helper_sha256':sha(Path(__file__).with_name('diagnose_gate_oracle_metric.py')),
        'parents':'all192 actual attention query K+1 states; compare best-with-cue to best-without-cue where both exist; report parents where duplicate cue copies make one-step complete cue deletion impossible',
        'reader_arms':list(ARMS),'set_selection':'original loss only, same sets for each reader intervention',
        'cue_probe':'fixedlambda1 linear ridge, old32train episodes only; privileged known cue vs cue-minus-current, binarylabel only in evaluation probe',
        'position_probe_sha256':sha(args.global_run/'position_readout.npz'),
        'not_claimed':['new trained world model','new retention strategy','untouched confirmation','hindsight choice is causal']}
    save(out/'protocol.json',protocol)
    old=PortableSnapshot(args.snapshot);ev=FollowupGateEvaluator(old,args.followup);scores=CausalModelScores(ev)
    fresh=FreshGateDataset(args.fresh);fresh.verify_old_normalization(old.norm);basis=scores._basis
    with np.load(args.global_run/'position_readout.npz',allow_pickle=False) as f:
        xy_readout={k:f[k].copy() for k in ('mean','std','target_mean','weight')}
    cue_train=defaultdict(list);cue_y=[];cue_rows=[]
    for rec in old.records:
        if rec['task']!='gate':continue
        z=old.features(rec);hidden=old.read_npz('data/dataset/'+rec['eval'],('evidence','label'))
        t=int(hidden['evidence'][0]);label=int(hidden['label']);q=rec['query']
        inputs={'cue':project(z[t],basis),'cue_minus_current':project(z[t],basis)-project(z[q],basis)}
        cue_rows.append({'id':rec['id'],'pair':rec['pair'],'split':rec['split'],'label':label,'inputs':inputs})
        if rec['split']=='train':
            for arm,x in inputs.items():cue_train[arm].append(x)
            cue_y.append([label])
    cue_models={arm:fit_readout(x,cue_y) for arm,x in cue_train.items()}
    parents={}
    with (args.evaluation/'online_trace.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['policy']=='attention' and r['time']==fresh.by_id[r['episode_id']]['query']:
                parents[r['episode_id']]=sorted(r['retained_timestamps']+[r['deleted_timestamp']])
    cache=defaultdict(dict)
    for r in json.loads((args.evaluation/'oracle_subsets.json').read_text()):
        cache[r['episode_id']][tuple(r['kept_times'])]=float(np.mean(r['feature_mse']))
    rows=[]; predictions=[]; selected=[];noncue_audit=[]
    for rec in fresh.records:
        z,actions=fresh.episode(rec);other=fresh.by_id[rec['sibling']];sz,_=fresh.episode(other)
        q=rec['query'];times=q+np.array([3,6]);target=z[times];sibling=sz[times]
        with np.load(fresh.path(rec['eval_path']),allow_pickle=False) as f:
            evidence=f['evidence'].copy();states=f['states'][times].copy();label=int(f['label'])
        with np.load(fresh.path(other['eval_path']),allow_pickle=False) as f:sibling_states=f['states'][times].copy()
        t=int(evidence[0]);cue_rows.append({'id':rec['id'],'pair':rec['pair'],'split':rec['condition'],'label':label,
            'inputs':{'cue':project(z[t],basis),'cue_minus_current':project(z[t],basis)-project(z[q],basis)}})
        eq={i for i in range(q-3) if any(np.array_equal(z[i],z[e]) for e in evidence)}
        with_cue,without=cue_partition(parents[rec['id']],eq,cache[rec['id']])
        if with_cue is None:raise AssertionError('Attention parent lost cue')
        selected.append({'episode_id':rec['id'],'pair':rec['pair'],'condition':rec['condition'],'parent':parents[rec['id']],
                         'with_cue':list(with_cue[1]),'without_cue':list(without[1]) if without else None,
                         'comparison_status':'available' if without else 'all_deletions_retain_a_cue_copy',
                         'shared_items':sorted(set(with_cue[1])&set(without[1])) if without else None})
        for arm in ARMS if without is not None else ():
            for keeps,(saved_loss,kept) in ((True,with_cue),(False,without)):
                pred,internal=reader_intervention(scores,rec,z,actions,kept,arm)
                m=metrics(pred,target,sibling,states,sibling_states,old.norm['std'],basis,xy_readout)
                if arm=='original' and abs(m['mse']-saved_loss)>1e-12:raise AssertionError('local original mismatch')
                rows.append({'episode_id':rec['id'],'pair':rec['pair'],'condition':rec['condition'],'arm':arm,
                             'keeps_cue':keeps,'kept_times':list(kept),'metrics':m,'reader':internal})
                predictions.append(pred)
        noncue=[i for i in range(q-3) if i not in eq]
        all_same=all(np.array_equal(z[i],z[noncue[0]]) for i in noncue)
        vals=[(loss,k) for k,loss in cache[rec['id']].items() if len(k)==2 and not eq.intersection(k)]
        best=min(vals);worst=max(vals)
        # Evaluate the SAME noncue set for siblings to certify no hidden identity input.
        p1,_=reader_intervention(scores,rec,z,actions,best[1],'original')
        p2,_=reader_intervention(scores,other,sz,actions,best[1],'original')
        if not np.array_equal(p1,p2):raise AssertionError('Identical noncue sibling inputs differ')
        noncue_audit.append({'episode_id':rec['id'],'pair':rec['pair'],'condition':rec['condition'],
            'all_noncue_historical_features_identical':all_same,'noncue_set_count':len(vals),
            'best_mse':best[0],'worst_mse':worst[0],'best_times':list(best[1]),'worst_times':list(worst[1]),
            'loss_range_across_time_choices':worst[0]-best[0], 'same_set_sibling_predictions_bitwise_identical':True})
    probe=[]
    for r in cue_rows:
        for arm,model in cue_models.items():
            value=float(decode(r['inputs'][arm],model)[0]);prediction=int(value>=.5)
            probe.append({k:v for k,v in r.items() if k!='inputs'}|{'arm':arm,'score':value,'correct':prediction==r['label']})
    paired=[]
    for a,b in zip(rows[::2],rows[1::2]):
        paired.append({k:a[k] for k in ('episode_id','pair','condition','arm')}|
                      {f'cue_minus_no_cue_{k}':a['metrics'][k]-b['metrics'][k] for k in
                       ('mse','raw_feature_mse','contrast_mse','orthogonal_mse','decoded_position_mse','feature_branch_correct','decoded_branch_correct')})
    summary={condition:{arm:{'mse_difference':group_ci([r for r in paired if r['condition']==condition and r['arm']==arm],'cue_minus_no_cue_mse')}
                       for arm in ARMS} for condition in ('id','long_gap','distractor_shift')}
    save(out/'episode_rows.json',rows);save(out/'paired_differences.json',paired);save(out/'parents.json',selected)
    save(out/'cue_probe.json',probe);save(out/'noncue_timing_audit.json',noncue_audit)
    np.savez_compressed(out/'predictions.npz',predictions=np.stack(predictions))
    if sha(__file__)!=protocol['source_sha256']:raise AssertionError('source changed')
    save(out/'summary.json',{'status':'completed','protocol_sha256':sha(out/'protocol.json'),'elapsed_seconds':time.monotonic()-began,
        'episodes':len(fresh.records),'rows':len(rows),'conditions':summary,
        'cue_probe':{arm:{split:group_ci([r for r in probe if r['arm']==arm and r['split']==split],'correct')
                         for split in ('train','validation','id','long_gap','distractor_shift')} for arm in cue_models},
        'frozen_audit':scores.audit(),'old_input_audit':old.audit(),'fresh_input_audit':fresh.audit()})
    print(json.dumps({'status':'completed','elapsed_seconds':time.monotonic()-began}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,default=ROOT/'build/hpc-20261020/development-assets')
    p.add_argument('--followup',type=Path,default=ROOT/'build/history-diagnosis-20261020/followup-blas1')
    p.add_argument('--fresh',type=Path,default=ROOT/'build/literature-baselines-20261022/fresh-data')
    p.add_argument('--evaluation',type=Path,default=ROOT/'build/literature-baselines-20261022/fresh-eval')
    p.add_argument('--global-run',type=Path,default=ROOT/'build/literature-diagnosis-20261022/metric-reader-v2')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
