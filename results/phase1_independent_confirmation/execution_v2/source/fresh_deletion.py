"""Original equal-capacity forks on fresh data; no learned retention policy.

Gate cue-rank pool construction is privileged diagnostic design. Motion pools
use only clocks and opaque pair-seeded RNG. Neither role annotations nor future
scores enter candidate inference. The cross-horizon oracle is offline only.
"""
from pathlib import Path
import numpy as np
import torch
from phase1.data import load_episode
from phase1.inputs import FeatureStore,collate
from phase1.probes import LinearReadout
from phase1.training import paired_rows
from phase1.utils import read_json,write_json,write_csv,file_hash,stable_seed
from staged_data import verify_permit
from locked_inference import MotionInference,gate_pair_with_transport
from fresh_evaluation import model_for,ExposedObservations
from counterbalanced_pools import gate_age_counterbalance,clock_only_static_pools
from pool_inputs import sample as fork_sample
from deletion_analysis import aggregate,summarize
from deletion_independent import audit_rows


def evaluate(c,permit):
    verify_permit(c,permit,models_locked=True)
    root=Path(c['output']);out=root/'deletion';out.mkdir(parents=True,exist_ok=False)
    store=FeatureStore(c,max_cached=16)
    rows,witnesses,metadata=[],[],[]
    exposed=ExposedObservations(c,store)
    for task in ('gate','motion'):
        with np.load(root/f'predictor/{task}/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
        store.bases={task:basis}
        decoder=LinearReadout.load(root/f'probes/{task}_state_decoder.npz')
        inference=None
        if task=='motion':
            heads={}
            for h in (3,6):
                with np.load(root/f'observation_maps/fitted/head_{h}.npz',allow_pickle=False) as f:heads[h]={k:f[k] for k in f.files}
            inference=MotionInference(basis,decoder,heads)
        for seed in (17,29,43):
            model,_=model_for(c,task,seed,'candidate',basis,permit)
            before={k:v.clone() for k,v in model.state_dict().items()}
            for split in ('test_id','test_long_gap','test_distractor_shift'):
                pairs=paired_rows([r for r in store.manifest['records'] if r['task']==task and r['split']==split])
                if len(pairs)!=256:raise ValueError('All256 fresh test pairs required')
                records=[r for pair in pairs for r in pair]
                for offset in range(0,len(records),8):
                    inputs,descriptions,contexts=[],[],[]
                    for rec in records[offset:offset+8]:
                        z=store.features(rec);projected=store.projected_features(rec);q=rec['query']
                        actions=exposed.observed(rec)['actions']
                        ev=tuple(map(int,load_episode(c,rec,True,keys=('evidence',))['evidence']))
                        end=q-int(rec['gap'])-4
                        pools=(list(gate_age_counterbalance(ev[0],end,q).items()) if task=='gate' else
                            [(f'clock_{i}',v) for i,v in enumerate(clock_only_static_pools(q,
                                stable_seed(20261017,'clock_only',rec['pair']),count=6))])
                        for name,parent in pools:
                            for deleted in parent:
                                kept=tuple(t for t in parent if t!=deleted)
                                s=fork_sample(z,projected,actions,q,kept)
                                inputs.append(s);descriptions.append((rec,name,parent,deleted,kept,ev))
                                contexts.append(exposed.context(rec,s,{'indices':kept},'deletion',inference)
                                    if inference is not None else None)
                    with torch.inference_mode():
                        batch=collate(inputs,'cpu')
                        if task=='gate':
                            base,predictions,transported=gate_pair_with_transport(model,batch);history=base
                        else:
                            base=model(batch).numpy();transported=np.zeros_like(base)
                            history=model.reference(batch).numpy();predictions=np.stack([
                                inference.correct(base[j],history[j],inputs[j],*contexts[j])
                                for j in range(len(inputs))])
                    if predictions.shape!=(len(inputs),2,5120) or not np.isfinite(predictions).all():
                        raise ValueError('Invalid full-feature deletion prediction')
                    for j,(rec,name,parent,deleted,kept,ev) in enumerate(descriptions):
                        # Scoring target/state are read after the complete model invocation.
                        q=rec['query'];z=store.features(rec)
                        truth=load_episode(c,rec,True,keys=('states','classes'))
                        decoded=decoder.predict(predictions[j])
                        deleted_role=('cue' if deleted==ev[0] else 'other') if task=='gate' else (
                            'start' if deleted==ev[0] else 'end' if deleted==ev[1] else 'other')
                        for hi,h in enumerate((3,6)):
                            target=z[q+h]
                            taskerror=(float((decoded[hi,0]>0)!=(truth['classes'][q+h]>0)) if task=='gate'
                                else float(np.mean((decoded[hi,1:]-truth['states'][q+h])**2)))
                            row=dict(task=task,seed=seed,split=split,pair=rec['pair'],episode=rec['id'],
                                pool=name,pool_group=name if task=='gate' else 'clock_all',query=q,
                                evidence_end=q-int(rec['gap'])-4,evidence=';'.join(map(str,ev)),
                                parent=';'.join(map(str,parent)),deleted=deleted,deleted_role=deleted_role,
                                kept=';'.join(map(str,kept)),history_count=2,horizon=h,feature_dim=5120,
                                feature_mse=float(np.mean((predictions[j,hi]-target)**2)),task_error=taskerror)
                            rows.append(row)
                            if offset==0 and j<18:
                                enabled,geo=(contexts[j] if contexts[j] is not None else (False,np.zeros((2,2))))
                                metadata.append(dict(task=task,seed=seed,split=split,condition='deletion',
                                    episode=rec['id'],horizon=h,kept=row['kept'],enabled=enabled,
                                    geometry_x=float(geo[hi,0]),geometry_y=float(geo[hi,1]),
                                    feature_mse=row['feature_mse'],task_error=taskerror))
                                witnesses.append((base[j,hi],predictions[j,hi],target,truth['states'][q+h],
                                    history[j,hi],truth['classes'][q+h],transported[j,hi]))
                    write_json(out/'progress.json',dict(task=task,seed=seed,split=split,rows=len(rows),expected=248832))
            if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):
                raise ValueError('Deletion evaluation changed a locked model')
            del model
    if len(rows)!=248832:raise ValueError('Incomplete fresh deletion forks')
    write_csv(out/'raw_deletions.csv',rows)
    compact,coverage=aggregate(rows,pairs=256)
    if len(compact)!=18432:raise ValueError('Incomplete fresh compact pair grid')
    stats=summarize(compact,repeats=10000,pairs=256)
    stats.update(fresh9137_opened=True,interpretation='Fresh9137 static-pool mechanism confirmation; future-informed oracles and privileged Gate pools are diagnostics, not deployable retention policies.')
    write_csv(out/'paired_policy_losses.csv',compact);write_json(out/'coverage.json',coverage)
    write_csv(out/'witness_metadata.csv',metadata)
    np.savez_compressed(out/'witness_vectors.npz',baseline=np.stack([v[0] for v in witnesses]),
        prediction=np.stack([v[1] for v in witnesses]),target=np.stack([v[2] for v in witnesses]),
        position=np.stack([v[3] for v in witnesses]),reference_history=np.stack([v[4] for v in witnesses]),
        target_class=np.array([v[5] for v in witnesses]),gate_transport=np.stack([v[6] for v in witnesses]))
    write_csv(out/'input_trace.csv',list(exposed.trace.values()))
    gate=stats['gate_critical_contrasts']
    motion=[v for v in stats['oracle_baseline_comparisons'] if v['task']=='motion' and v['choice']=='oracle_transfer']
    if len(gate)!=72 or len(motion)!=72:raise ValueError('Original Gate/Motion deletion acceptance axes changed')
    stats.update(raw_rows=len(rows),raw_sha256=file_hash(out/'raw_deletions.csv'),
        paired_sha256=file_hash(out/'paired_policy_losses.csv'),
        acceptance={'gate':{'passed':sum(v['passed'] for v in gate),'total':72},
                    'motion':{'passed':sum(v['positive_both_metrics'] for v in motion),'total':72}})
    write_json(out/'summary.json',stats)
    # Original independent raw reconstruction has no model/analyzer imports.
    audit=audit_rows(rows,compact,stats,pairs=256,repeats=10000)
    write_json(out/'independent_raw_audit.json',audit)
    return stats
