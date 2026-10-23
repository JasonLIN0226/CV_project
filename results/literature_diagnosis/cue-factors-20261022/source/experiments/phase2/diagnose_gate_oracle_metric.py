"""Posthoc Gate oracle diagnosis: fixed reader interventions and task metrics.

All intervention predictions are diagnostic, not deployable baseline results.
Simulator states supervise only a train-fitted evaluation readout. No predictor,
encoder, retention policy, input projection or normalization is changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments/predictor/control/src"):
    sys.path.insert(0, str(directory))
import numpy as np
import torch
from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.memory import MemoryEntry
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.run_literature_baselines import QueryOracle
from phase1.model import project

ARMS = ("original", "no_transport", "no_velocity", "no_past_action", "uniform_attention")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def paired_components(pred, target, sibling):
    """Exact orthogonal split along the evaluator-only sibling contrast."""
    p, y, other = [np.asarray(v, np.float64) for v in (pred, target, sibling)]
    direction, midpoint = (y-other)/2, (y+other)/2
    energy = np.sum(direction**2, axis=-1)
    if np.any(energy <= 1e-20):
        raise ValueError("Sibling future contrast must be nonzero")
    amplitude = np.sum((p-midpoint)*direction, axis=-1)/energy
    along = (amplitude-1)**2*energy/y.shape[-1]
    orthogonal = np.mean((p-midpoint-amplitude[..., None]*direction)**2, axis=-1)
    mse = np.mean((p-y)**2, axis=-1)
    if not np.allclose(mse, along+orthogonal, rtol=0, atol=1e-12):
        raise AssertionError("Contrast decomposition failed")
    return {"mse": float(mse.mean()), "contrast_mse": float(along.mean()),
            "orthogonal_mse": float(orthogonal.mean()),
            "contrast_amplitude": amplitude.tolist(),
            "feature_branch_correct": float(np.mean(amplitude > 0))}


def fit_readout(x, y):
    """Fixed ridge lambda=1; train-only means/scales; labels never enter a policy."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
    ym = y.mean(0)
    a = (x-mean)/std
    weight = np.linalg.solve(a.T@a+np.eye(a.shape[1]), a.T@(y-ym))
    return {"mean": mean, "std": std, "target_mean": ym, "weight": weight}


def decode(x, readout):
    return (np.asarray(x, np.float64)-readout["mean"])/readout["std"]@readout["weight"]+readout["target_mean"]


def reader_intervention(scores, record, z, actions, kept, arm):
    """Reproduce the locked forward exactly, then modify one named path."""
    if arm not in ARMS:
        raise ValueError(arm)
    q, model = record["query"], scores._model
    recent = tuple(MemoryEntry(record["id"], t, z[t]) for t in range(q-3, q+1))
    history = tuple(MemoryEntry(record["id"], int(t), z[int(t)]) for t in kept)
    past_actions = {e.entry_id: actions[e.timestamp:q].sum(0, dtype=np.float32) for e in recent+history}
    batch = scores._batch(recent, history, past_actions, q, actions[q:q+6].copy())
    with torch.inference_mode():
        valid = batch['mask']
        p = batch['projected_z'].masked_fill(~valid[..., None], 0)
        current = batch['projected_current']
        hist = valid & (batch['type'] == 1)
        w = scores._weights(p, current, hist)
        if arm == "uniform_attention":
            w = hist.float()/hist.sum(1, keepdim=True).clamp_min(1)
        past = model.past_action(batch['past_actions'].masked_fill(~valid[..., None], 0))
        if arm == "no_past_action":
            past = torch.zeros_like(past)
        aligned = p+past
        age = batch['age'].masked_fill(~valid, 0)/8
        mean_age = (w*age).sum(1)
        center = age-mean_age[:, None]
        memory = (w[..., None]*aligned).sum(1)
        past_mean = (w[..., None]*past).sum(1)
        variance = (w*center.square()).sum(1)
        velocity = -(w[..., None]*center[..., None]*(aligned-memory[:, None])).sum(1)/variance.clamp_min(.01)[:, None]
        if arm == "no_velocity":
            velocity = torch.zeros_like(velocity)
        output = []
        transport_norm = []
        for h in (3, 6):
            a = batch['actions'][:, :h]
            action = model.action(torch.cat([a[:, 0], a.sum(1), a[:, -1]], -1))
            transported = memory+velocity*(mean_age+h/8)[:, None]
            if arm == "no_transport":
                transported = memory
            stats = torch.stack([mean_age, hist.any(1).float(), torch.full_like(mean_age, h/8)], -1)
            features = torch.cat([current, memory, velocity, transported, action,
                                  memory*action, past_mean, velocity*action, stats], -1)
            coeff = model.output(features)
            output.append((batch['current']+model.target_mean+coeff@model.output_basis.T)[0].numpy())
            transport_norm.append(float(torch.linalg.vector_norm(transported-memory)))
        prediction = np.stack(output)
        if arm == "original":
            expected = model(batch)[0].numpy()
            if not np.array_equal(prediction, expected):
                raise AssertionError("Diagnostic forward changed locked operation order")
        return prediction, {"attention": w[0, 4:].tolist(), "mean_age": float(mean_age[0]),
                            "velocity_norm": float(torch.linalg.vector_norm(velocity)),
                            "transport_displacement_norm": transport_norm}


def metrics(pred, target, sibling, states, sibling_states, norm_std, basis, readout):
    values = paired_components(pred, target, sibling)
    decoded = decode(project(pred, basis), readout)
    values.update(raw_feature_mse=float(np.mean(((pred.astype(np.float64)-target)*norm_std)**2)),
                  decoded_position_mse=float(np.mean((decoded-states)**2)),
                  decoded_position=decoded.tolist(),
                  decoded_branch_correct=float(np.mean(np.sum((decoded-states)**2, -1) <
                                                        np.sum((decoded-sibling_states)**2, -1))))
    return values


def group_ci(rows, key, seed=20261022):
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair']].append(float(row[key]))
    values = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (5000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "pairs": len(values), "episodes": len(rows)}


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    source_hash = sha(__file__)
    protocol = {"version": "gate-metric-reader-diagnosis-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "posthoc development diagnosis, no untouched confirmation",
                "data": "all192 fresh episodes; old32train episodes fit evaluation-only position readout",
                "source_sha256": source_hash, "readout": "fixed ridge lambda1 on frozen128D input projection, train-only standardization; all train frames",
                "arms": list(ARMS), "sets": "original feature-loss best-with-cue and best-without-cue over all historical K2; sets never reoptimized for interventions",
                "metric_split": "exact sibling-future contrast and orthogonal remainder; remainder is not proven task-irrelevant",
                "metrics": ["standardized feature MSE", "raw feature MSE", "paired contrast", "train-only decoded normalized XY"],
                "predictor_training": False, "retention_training": False, "test_arrays": False,
                "limitations": ["Reader interventions can be out of training distribution; not repaired model performance.",
                                "Future-pair contrast is evaluator-only; position probe is imperfect.",
                                "Output-basis floor is a representational bound, not deployable prediction."]}
    save(output/'protocol.json', protocol)
    old = PortableSnapshot(args.snapshot)
    evaluator = FollowupGateEvaluator(old, args.followup)
    scores = CausalModelScores(evaluator)
    fresh = FreshGateDataset(args.fresh)
    fresh.verify_old_normalization(old.norm)
    basis = scores._basis
    train_x, train_y, train_ids = [], [], []
    for rec in old.records:
        if rec['task']=='gate' and rec['split']=='train':
            z = old.features(rec)
            states = old.read_npz('data/dataset/'+rec['eval'], ('states',))['states']
            train_x.append(project(z, basis)); train_y.append(states); train_ids.append(rec['id'])
    readout = fit_readout(np.concatenate(train_x), np.concatenate(train_y))
    np.savez_compressed(output/'position_readout.npz', **readout, train_ids=np.array(train_ids))
    probe_rows = []
    for rec in old.records:
        if rec['task']=='gate':
            z=old.features(rec); state=old.read_npz('data/dataset/'+rec['eval'], ('states',))['states']
            times=rec['query']+np.array([3,6]); error=decode(project(z[times],basis),readout)-state[times]
            probe_rows.append({'id':rec['id'],'pair':rec['pair'],'split':rec['split'],
                               'position_mse':float(np.mean(error**2)), 'max_abs':float(np.max(np.abs(error)))})
    cache = defaultdict(list)
    for row in json.loads((args.evaluation/'oracle_subsets.json').read_text()):
        if len(row['kept_times']) == 2:
            cache[row['episode_id']].append((float(np.mean(row['feature_mse'])), tuple(row['kept_times'])))
    previous = json.loads((args.evaluation/'episode_results.json').read_text())
    attention_rows={r['episode_id']:r for r in previous if r['policy']=='attention'}
    rows, vectors, case_names = [], [], []
    output_basis = scores._model.output_basis.detach().cpu().numpy().astype(np.float64)
    target_mean = scores._model.target_mean.detach().cpu().numpy().astype(np.float64)
    singular = np.linalg.svd(output_basis, full_matrices=False)[0]
    for index, rec in enumerate(fresh.records):
        z, actions=fresh.episode(rec); other=fresh.by_id[rec['sibling']]; sz,_=fresh.episode(other)
        q=rec['query']; times=q+np.array([3,6]); target=z[times]; sibling=sz[times]
        state_path=fresh.path(rec['eval_path']); sibling_path=fresh.path(other['eval_path'])
        with np.load(state_path,allow_pickle=False) as a: states=a['states'][times].copy(); evidence=a['evidence'].copy()
        with np.load(sibling_path,allow_pickle=False) as a: sibling_states=a['states'][times].copy()
        equivalent={t for t in range(q-3) if any(np.array_equal(z[t],z[e]) for e in evidence)}
        with_cue=min((loss,k) for loss,k in cache[rec['id']] if equivalent.intersection(k))
        without_cue=min((loss,k) for loss,k in cache[rec['id']] if not equivalent.intersection(k))
        gt=decode(project(target,basis),readout)
        probe_rows.append({'id':rec['id'],'pair':rec['pair'],'split':rec['condition'],
                           'position_mse':float(np.mean((gt-states)**2)), 'max_abs':float(np.max(np.abs(gt-states))),
                           'branch_correct':float(np.mean(np.sum((gt-states)**2,-1)<np.sum((gt-sibling_states)**2,-1)))})
        delta=target.astype(np.float64)-z[q].astype(np.float64)-target_mean
        floor=np.mean((delta-(delta@singular)@singular.T)**2)
        recrows=[]
        for arm in ARMS:
            pair=[]
            for cue_flag,(old_loss,kept) in ((True,with_cue),(False,without_cue)):
                pred,internal=reader_intervention(scores,rec,z,actions,kept,arm)
                measure=metrics(pred,target,sibling,states,sibling_states,old.norm['std'],basis,readout)
                if arm=='original' and abs(measure['mse']-old_loss)>1e-12:
                    raise AssertionError('Saved oracle loss changed')
                row={'episode_id':rec['id'],'pair':rec['pair'],'condition':rec['condition'],'arm':arm,
                     'keeps_cue':cue_flag,'kept_times':list(kept),'metrics':measure,'reader':internal,
                     'output_subspace_floor':float(floor)}
                rows.append(row); recrows.append(row); pair.append(row)
                vectors.append(pred); case_names.append(f"{rec['id']}:{arm}:{int(cue_flag)}")
            gap=pair[0]['metrics']['mse']-pair[1]['metrics']['mse']
            for row in pair: row['cue_minus_no_cue_mse']=gap
        # A fixed baseline set is an independent check of the original forward.
        original_pred,_=reader_intervention(scores,rec,z,actions,attention_rows[rec['id']]['kept_times'],'original')
        if abs(np.mean((original_pred.astype(np.float64)-target)**2)-attention_rows[rec['id']]['mean_mse'])>1e-12:
            raise AssertionError('Baseline replay mismatch')
        if (index+1)%32==0: print(json.dumps({'episodes':index+1,'total':192}),flush=True)
    save(output/'episode_rows.json',rows); save(output/'readout_validation.json',probe_rows)
    np.savez_compressed(output/'predictions.npz',predictions=np.stack(vectors),case_names=np.array(case_names))
    paired=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        paired.append({'episode_id':a['episode_id'],'pair':a['pair'],'condition':a['condition'],'arm':a['arm'],
                       **{f'cue_minus_no_cue_{key}':a['metrics'][key]-b['metrics'][key]
                          for key in ('mse','raw_feature_mse','contrast_mse','orthogonal_mse','decoded_position_mse','feature_branch_correct','decoded_branch_correct')}})
    save(output/'paired_differences.json',paired)
    summary={}
    for condition in ('id','long_gap','distractor_shift'):
        summary[condition]={}
        for arm in ARMS:
            part=[r for r in paired if r['condition']==condition and r['arm']==arm]
            summary[condition][arm]={key:group_ci(part,key) for key in part[0] if key.startswith('cue_minus_')}
            summary[condition][arm]['strict_no_cue_fraction']=float(np.mean([r['cue_minus_no_cue_mse']>1e-8 for r in part]))
    if sha(__file__)!=source_hash: raise AssertionError('Source changed during run')
    result={'status':'completed','protocol_sha256':sha(output/'protocol.json'),'elapsed_seconds':time.monotonic()-started,
            'episodes':len(fresh.records),'prediction_rows':len(rows),'train_readout_episodes':len(train_ids),
            'train_readout_frames':len(np.concatenate(train_x)),'source_sha256':source_hash,'conditions':summary,
            'position_readout_validation':{split:group_ci([r for r in probe_rows if r['split']==split],'position_mse')
                 for split in ('train','validation','id','long_gap','distractor_shift')},
            'frozen_predictor_audit':scores.audit(),'old_input_audit':old.audit(),'fresh_input_audit':fresh.audit()}
    save(output/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('status','episodes','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,default=ROOT/'build/hpc-20261020/development-assets')
    p.add_argument('--followup',type=Path,default=ROOT/'build/history-diagnosis-20261020/followup-blas1')
    p.add_argument('--fresh',type=Path,default=ROOT/'build/literature-baselines-20261022/fresh-data')
    p.add_argument('--evaluation',type=Path,default=ROOT/'build/literature-baselines-20261022/fresh-eval')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
