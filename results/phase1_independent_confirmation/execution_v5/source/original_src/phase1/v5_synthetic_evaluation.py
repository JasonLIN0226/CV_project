"""Explicit v5 checkpoint verification and original synthetic evaluation rules.

The immutable original configuration is the data/readout contract only. New
checkpoint signatures remain verbatim; they are never relabelled as an old
DynamicsPredictor or given a fabricated old trainer hash.
"""
import ast
import copy
from pathlib import Path

import numpy as np
import torch

from .config import CONDITIONS, SPLITS
from .data import load_episode
from .inputs import FeatureStore, collate
from .model import fit_basis
from .probes import load_decoder
from .evaluation import (_finite_tree, active_tasks, pair_calibration, pair_diagnostic, timed_forward,
                         summarize, summarize_calibration)
from .reporting import decision, _development_artifacts
from .stats import clustered_interval, paired_improvement
from .training import TRAIN_CONDITIONS
from .utils import read_json, write_json, write_csv, file_hash, code_hash, digest, seed_all
from .v5_synthetic_reader import SyntheticCausalReader
from .v5_synthetic_training import verify_contract

SELECTION = 'equal_four_condition_two_horizon_mse_earliest_minimum'


def validate_checkpoint_pair(seed, latest, best, training, signature, *, epochs=120, samples=1024, steps=64):
    """Pure verification of actual snapshots; missing proof fails closed."""
    for name, ck in (('latest', latest), ('best', best)):
        if (ck['signature'] != signature or ck['seed'] != seed or ck['task'] != signature['task']
                or ck['resume_granularity'] != 'epoch_boundary' or ck['checkpoint_selection'] != SELECTION):
            raise ValueError(name+' provenance/selection/boundary mismatch')
        if not _finite_tree(ck['model']) or not _finite_tree(ck['optimizer']):
            raise ValueError(name+' non-finite model/optimizer')
        if set(ck['rng']) != {'python','numpy','torch','cuda'}:
            raise ValueError(name+' incomplete RNG')
        if ck['step'] != ck['epoch']*steps or len(ck['logs']) != ck['epoch']:
            raise ValueError(name+' epoch/step/log mismatch')
        if ck['scheduler']['last_epoch'] != ck['epoch']:
            raise ValueError(name+' scheduler boundary mismatch')
        if ck['epoch'] > 0 and not ck['optimizer']['state']:
            raise ValueError(name+' missing actual optimizer states')
        if any(float(state['step']) != ck['step'] for state in ck['optimizer']['state'].values()):
            raise ValueError(name+' optimizer step mismatch')
    logs = latest['logs']
    if latest['epoch'] != epochs or training['signature'] != signature or training['logs'] != logs or training['seed'] != seed:
        raise ValueError('All prescribed training must complete with matching actual log')
    for epoch, row in enumerate(logs, 1):
        counts = ast.literal_eval(row['mix_counts']) if isinstance(row['mix_counts'],str) else row['mix_counts']
        if row['epoch'] != epoch or row['step'] != epoch*steps or counts != {k:samples for k in TRAIN_CONDITIONS}:
            raise ValueError('Unbalanced or noncontiguous epoch')
        if not _finite_tree(row):
            raise ValueError('Non-finite epoch log')
        actual_score = float(np.mean([row['val_'+k+'_mse'] for k in TRAIN_CONDITIONS]))
        if not np.isclose(actual_score,row['validation_mse'],rtol=0,atol=1e-14):
            raise ValueError('Validation selection differs from equal-condition MSE')
    earliest = min(logs,key=lambda row:row['validation_mse'])['epoch']
    if best['epoch'] != earliest or best['logs'] != logs[:earliest] or best['best'] != logs[earliest-1]['validation_mse']:
        raise ValueError('Best is not the actual earliest minimum matching prefix')
    return {'complete':True,'seed':seed,'task':signature['task'],'completed_epochs':epochs,
            'final_step':epochs*steps,'best_epoch':earliest,'selection_metric':SELECTION,
            'signature':copy.deepcopy(signature),'rng_optimizer_scheduler_boundary_verified':True,
            'expected_samples_per_condition_epoch':samples,'expected_steps_per_epoch':steps,'issues':[]}


def all_models_completion(control, training_root, store=None):
    control, training_root = Path(control), Path(training_root)
    c,p,lock = verify_contract(control,control/'configs/phase1_v5_gate_training_protocol.json')
    # Modules actually imported by this verifier must be the same bytes as the
    # trainer. Independent v5 evaluation code is bound separately in its lock.
    for name in ('v5_synthetic_reader.py','v5_synthetic_training.py','inputs.py','training.py','model.py',
                 'utils.py','evaluation.py','stats.py','probes.py','config.py','cache.py','data.py','envs.py'):
        if file_hash(Path(__file__).parent/name) != lock['files']['src/phase1/'+name]:
            raise ValueError('Loaded scientific module differs from actual training source: '+name)
    store = store or FeatureStore(c)
    expected = {'protocol_sha256':file_hash(control/'configs/phase1_v5_gate_training_protocol.json'),
                'source_lock_sha256':file_hash(control/'source_lock.json'),
                'data_config_sha256':p['data_source_config_sha256'],'cache_signature':store.ix['signature'],
                'basis_sha256':p['basis_sha256'],'normalization_sha256':file_hash(Path(c['output'])/'features/normalization.npz'),
                'reader':'SyntheticCausalReader','task':p['task'],'selection':SELECTION}
    contract = read_json(training_root/'run_contract.json')
    if contract['signature'] != expected or contract['protocol'] != p or contract['source_files'] != lock['files']:
        raise ValueError('Actual run contract differs from source/cache/basis/protocol')
    basis = fit_basis(c,store,p['task'])  # verify_contract requires the existing, SHA-checked basis.
    train_ids = {r['id'] for r in store.manifest['records'] if r['task']==p['task'] and r['split']=='train'}
    if set(basis['train_ids'].tolist()) != train_ids:
        raise ValueError('Original basis train IDs changed')
    models, issues = {}, []
    for seed in p['seeds']:
        key = f"{p['task']}/{seed}"
        sr = training_root/'predictor'/p['task']/f'seed_{seed}'
        try:
            latest = torch.load(sr/'latest.pt',map_location='cpu',weights_only=False)
            best = torch.load(sr/'best.pt',map_location='cpu',weights_only=False)
            proof = validate_checkpoint_pair(seed,latest,best,read_json(sr/'training.json'),expected)
            for ck in (latest,best):
                if ck['horizons'] != [3,6] or ck['feature_dim'] != 5120:
                    raise ValueError('Original horizons or target dimension changed')
                for name in ('input_mean','projection','target_mean','output_basis'):
                    if not torch.equal(ck['model'][name],torch.as_tensor(basis[name]).float()):
                        raise ValueError('Frozen original basis buffer changed: '+name)
                expected_lr = c['training']['lr']*c['training']['lr_gamma']**(ck['epoch']//c['training']['lr_step'])
                if any(not np.isclose(group['lr'],expected_lr,rtol=0,atol=1e-15) for group in ck['optimizer']['param_groups']):
                    raise ValueError('Original optimizer schedule differs')
            proof.update(best_sha256=file_hash(sr/'best.pt'),latest_sha256=file_hash(sr/'latest.pt'),
                         training_json_sha256=file_hash(sr/'training.json'))
            models[key] = proof
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,SyntaxError) as exc:
            issues.append(key+': '+str(exc));models[key]={'complete':False,'seed':seed,'task':p['task'],'issues':[str(exc)]}
    return {'complete':not issues,'required_epochs':120,'tasks':[p['task']],'seeds':p['seeds'],
            'models':models,'issues':issues,'signature':expected,'phase1_go':False}


def load_current_reader(control, training_root, seed, store, completion):
    if not completion.get('complete') or set(completion.get('models',{})) != {'gate/17','gate/29','gate/43'}:
        raise ValueError('All three current models must be proved complete before model evaluation')
    root = Path(training_root)/'predictor/gate'/f'seed_{seed}'
    proof = completion['models'][f'gate/{seed}']
    if file_hash(root/'best.pt') != proof['best_sha256']:
        raise ValueError('Current selected weights changed after all-model lock')
    ck = torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    if ck['signature'] != proof['signature']:
        raise ValueError('Current checkpoint signature changed')
    basis = fit_basis(store.c,store,'gate')
    store.bases={'gate':basis};store.projected.clear()
    model = SyntheticCausalReader(basis)
    model.load_state_dict(ck['model'],strict=True)
    return model.eval(),ck


def original_task_rules(data_config, audit, probes, metrics, completion):
    if not completion['complete']:
        raise ValueError('Original rules require all current model proofs')
    comparisons={(r['seed'],r['split'],r['horizon'],r['alternative']) for r in metrics['comparisons']}
    required={(seed,split,h,condition) for seed in [17,29,43,'all_seeds_mean'] for split in SPLITS[2:]
              for h in [3,6] for condition in ['privileged_small_budget','full_history','random_small_budget','wrong_evidence']}
    if not required.issubset(comparisons):
        raise ValueError('Missing original per-seed or pooled task comparisons')
    # This is a statistical rule profile, never a FeatureStore/configuration
    # for fitting data. The incompatible old provenance block is enforced by
    # all_models_completion instead, with the true v5 signature above.
    profile=copy.deepcopy(data_config)
    profile['data']['tasks']=['gate'];profile.pop('experiment',None)
    status,reasons,checks=decision(profile,audit,probes,metrics)
    mapped={'GO':'DEVELOPMENT_PASS','REVISE':'DEVELOPMENT_REVISE','INCONCLUSIVE':'DEVELOPMENT_INCOMPLETE'}
    return {'status':mapped[status],'reasons':reasons,'checks':checks,
            'proof_scope':'Original statistical rules unchanged; actual v5 source/checkpoints independently verified',
            'phase1_go':False}


def fixed_reference_checks(data_config, reference, candidate):
    keys=('seed','task','split','horizon','condition','pair','episode')
    ref={tuple(row[k] for k in keys):row for row in reference}
    cand={tuple(row[k] for k in keys):row for row in candidate}
    if len(ref)!=len(reference) or len(cand)!=len(candidate) or set(ref)!=set(cand):
        raise ValueError('Reference/candidate complete pair/sibling keys differ')
    checks=[];rule=data_config['decision'];repeats=data_config['evaluation']['bootstrap']
    for split in SPLITS[2:]:
        for h in (3,6):
            for seed in [17,29,43,'all_seeds_mean']:
                selected=[r for r in candidate if r['split']==split and r['horizon']==h and r['condition']=='privileged_small_budget'
                          and (seed=='all_seeds_mean' or r['seed']==seed)]
                grouped={}
                for row in selected:
                    key=tuple(row[k] for k in keys);base=list(key);base[4]='recent_only'
                    grouped.setdefault((row['pair'],row['episode']),[]).append(
                        [ref[key]['normalized_mse'],row['normalized_mse'],ref[tuple(base)]['normalized_mse']])
                ordered=sorted(grouped);values=np.array([np.mean(grouped[k],axis=0) for k in ordered]);groups=[k[0] for k in ordered]
                name=f'gate/{split}/h{h}/seed{seed}'
                cross=paired_improvement(values[:,2],values[:,1],groups,repeats,rule['baseline_floor'],rule['absolute_latent_gain'])
                threshold=rule['latent_relative_improvement'] if cross['scale'].startswith('relative') else rule['absolute_latent_gain']
                checks.append({'kind':'candidate_vs_reference_recent','slice':name,'passed':cross['mean']>=threshold and cross['low']>0,
                               'improvement':cross,'reference_recent_mse':float(values[:,2].mean()),'candidate_privileged_mse':float(values[:,1].mean())})
                if split=='test_distractor_shift':
                    absolute=clustered_interval(values[:,0]-values[:,1],groups,repeats)
                    checks.append({'kind':'critical_absolute_mse_reduction','slice':name,'passed':absolute['mean']>0 and absolute['low']>0,
                                   'absolute_mse_reduction':absolute,'reference_privileged_mse':float(values[:,0].mean()),
                                   'candidate_privileged_mse':float(values[:,1].mean())})
    if len([r for r in checks if r['kind']=='candidate_vs_reference_recent'])!=24 or len([r for r in checks if r['kind']=='critical_absolute_mse_reduction'])!=8:
        raise ValueError('Original fixed-reference and critical slice counts changed')
    return checks


# The loop below is a separate static copy of the original evaluator. Only
# loading/provenance/output paths differ; metrics, inputs and statistics are reused.
def evaluate_v5(c, control, training_root, output):
    device = torch.device("cpu")
    store = FeatureStore(c, max_cached=2*max(c["data"]["pairs"][s] for s in SPLITS[1:]))
    completion = all_models_completion(control, training_root, store)
    if completion is not None and not completion["complete"]:
        raise ValueError("Held-out evaluation requires all prescribed training to complete: "+"; ".join(completion["issues"]))
    decoders = {task: load_decoder(c, task) for task in active_tasks(c)}
    raw, latencies, checkpoints, calibration, pending = [], [], {}, [], {}
    for seed in c["training"]["seeds"]:
        for task in active_tasks(c):
            model, ck = load_current_reader(control, training_root, seed, store, completion)
            checkpoints[f"{task}/{seed}"] = {"task": task, "seed": seed, "best_epoch": ck["epoch"], "training_seconds": ck["seconds"],
                "best_sha256": file_hash(Path(training_root)/"predictor"/task/f"seed_{seed}"/"best.pt")}
            for split in SPLITS[1:]:
                rows = [r for r in store.manifest["records"] if r["task"] == task and r["split"] == split]
                for condition in CONDITIONS+["persistence"]:
                    if condition != "persistence":
                        sample = store.sample(rows[0], condition)[0]
                        latency = timed_forward(c, model, collate([sample], device), device)
                        latencies.append({"seed": seed, "task": task, "split": split, "condition": condition, **latency})
                    for start in range(0, len(rows), c["training"]["batch_size"]):
                        part = rows[start:start+c["training"]["batch_size"]]
                        samples = [store.sample(r, condition if condition != "persistence" else "recent_only") for r in part]
                        if condition == "persistence":
                            predictions = np.stack([np.tile(s[0]["current"], (len(c["data"]["horizons"]), 1)) for s in samples])
                        else:
                            with torch.inference_mode():
                                predictions = model(collate([s[0] for s in samples], device)).cpu().numpy()
                        if not np.isfinite(predictions).all():
                            raise ValueError("Non-finite actual v5 predictions")
                        for rec, sample, prediction in zip(part, samples, predictions):
                            _, target, info = sample
                            meta = load_episode(c, rec, True)
                            sibling = store.records[rec["sibling"]]
                            other = store.features(sibling)[rec["query"]+np.array(c["data"]["horizons"])]
                            state = decoders[task].predict(prediction)
                            D, R, K = prediction.shape[-1], c["data"]["recent"], info["history_count"]
                            for j, h in enumerate(c["data"]["horizons"]):
                                if task == "gate":
                                    key = (seed, task, split, condition, h, rec["pair"])
                                    pending.setdefault(key, {})[rec["id"]] = (prediction[j], target[j])
                                    if len(pending[key]) == 2:
                                        vectors = [pending[key][eid] for eid in sorted(pending[key])]
                                        calibration.append({"seed": seed, "task": task, "split": split, "condition": condition, "horizon": h,
                                                            "pair": rec["pair"], **pair_calibration([p for p, _ in vectors], [y for _, y in vectors])})
                                        del pending[key]
                                p, y = prediction[j]*store.norm["std"]+store.norm["mean"], target[j]*store.norm["std"]+store.norm["mean"]
                                cosine = 1-float(np.dot(p, y)/(max(np.linalg.norm(p)*np.linalg.norm(y), 1e-12)))
                                pairacc, margin, separation = pair_diagnostic(prediction[j], target[j], other[j])
                                ti = rec["query"]+h
                                raw.append({"seed": seed, "task": task, "split": split, "condition": condition, "horizon": h,
                                    "episode": rec["id"], "pair": rec["pair"], "gap": rec["gap"], "query": rec["query"],
                                    "latent_mse": float(np.mean((p-y)**2)), "normalized_mse": float(np.mean((prediction[j]-target[j])**2)),
                                    "cosine_distance": cosine, "pair_accuracy": pairacc, "pair_margin": margin,
                                    "target_pair_distance": separation, "state_accuracy": float((state[j, 0]>0) == (meta["classes"][ti]>0)),
                                    "state_position_mse": float(np.mean((state[j, 1:]-meta["states"][ti])**2)),
                                    "history_count": K, "history_feature_bytes": K*D*4, "recent_feature_bytes": R*D*4,
                                    "projected_history_bytes": K*model.projection.shape[1]*4,
                                    "timestamp_type_mask_bytes": (K+R)*13, "past_action_bytes": (K+R)*8,
                                    "future_action_bytes": h*2*4, "evidence_hit": float(info["evidence_hit"]),
                                    "state_pred_x": float(state[j, 1]), "state_pred_y": float(state[j, 2]),
                                    "state_true_x": float(meta["states"][ti, 0]), "state_true_y": float(meta["states"][ti, 1])})
    root = Path(output)/"evaluation"
    if pending:
        raise ValueError("Calibration encountered an incomplete gate sibling pair")
    write_csv(root/"raw_records.csv", raw)
    summary = summarize(c, raw)
    write_json(root/"metrics.json", {"mode": c["mode"], "cache_signature": store.ix["signature"],
        "raw_sha256": file_hash(root/"raw_records.csv"), "legacy_metric_helpers_code_hash": code_hash("evaluation.py", "stats.py"),
        "checkpoints": checkpoints, "training_completion": completion, "data_source_experiment": c["experiment"],
        "reader": "SyntheticCausalReader", "v5_evaluation_code_hash": code_hash("v5_synthetic_evaluation.py", "evaluation.py", "stats.py"),
        "phase1_go": False, **summary})
    write_json(root/"latency_memory.json", latencies)
    if calibration:
        write_json(root/"calibration_summary.json", {"cache_signature": store.ix["signature"], **summarize_calibration(c, calibration)})
    return read_json(root/"metrics.json"), raw, completion


def run_evaluation(evaluation_control, protocol_path):
    import time
    began=time.perf_counter()
    p=read_json(protocol_path)
    if p['stage']!='development' or p['task']!='gate' or p['seeds']!=[17,29,43] or p['epochs']!=120:
        raise ValueError('Only the fixed Gate development evaluation is authorized')
    if file_hash(p['training_control']+'/source_lock.json')!=p['training_source_lock_sha256']:
        raise ValueError('Actual training source lock differs from evaluation declaration')
    c,training_protocol,_=verify_contract(p['training_control'],p['training_control']+'/configs/phase1_v5_gate_training_protocol.json')
    if file_hash(p['training_control']+'/configs/phase1_v5_gate_training_protocol.json')!=p['training_protocol_sha256']:
        raise ValueError('Actual training protocol changed')
    root=Path(p['output'])
    if root.exists():
        raise ValueError('Refuse overwrite/re-evaluation of an existing science output')
    seed_all(17,4)
    # Strict current model completion occurs inside evaluate_v5 before any
    # held-out prediction/label decoding. Source contract is independent.
    metrics,raw,completion=evaluate_v5(c,p['training_control'],p['training_root'],root)
    write_json(root/'all_models_completion.json',completion)
    write_json(root/'evaluation_protocol.json',p)
    ref,rm,rr,audit,probes=_development_artifacts(c['output'],'gate')
    verification=read_json(training_protocol['frozen_v3_verification'])
    for rel in ('gate/reference/evaluation/raw_records.csv','gate/reference/evaluation/metrics.json','gate/reference/probes/metrics.json'):
        actual=Path(c['output'])/Path(rel).relative_to('gate/reference')
        if file_hash(actual)!=verification['actual_hpc_compact_sha256'][rel]:
            raise ValueError('Actual old reference artifact differs from frozen verification: '+rel)
    original_source=Path(p['frozen_reference_source'])/'src/phase1'
    old_hash=digest({name:file_hash(original_source/name) for name in ('training.py','model.py','inputs.py')})
    if old_hash!=verification['source_code_hash'] or any(model['signature']['code_hash']!=old_hash for model in rm['training_completion']['models'].values()):
        raise ValueError('Actual old reference trainer/source differs')
    for name in ('cache_signature','normalization_sha256','basis_sha256'):
        for seed in (17,29,43):
            if rm['training_completion']['models'][f'gate/{seed}']['signature'][name]!=completion['signature'][name]:
                raise ValueError('New reader and actual original reference differ in '+name)
    rules=original_task_rules(c,audit,probes,metrics,completion)
    checks=fixed_reference_checks(c,rr,raw)
    failures=[r['slice']+' '+r['kind'] for r in checks if not r['passed']]
    result={'status':'SUPPORTED' if rules['status']=='DEVELOPMENT_PASS' and not failures else 'REJECTED',
            'task':'gate','scope':'Revealed6301 development adoption only; not independent confirmation',
            'candidate_original_task_rules':rules,'fixed_reference_checks':checks,
            'reasons':rules['reasons']+failures,'reference_raw_sha256':rm['raw_sha256'],
            'candidate_raw_sha256':metrics['raw_sha256'],'current_models':completion,
            'actual_old_reference_source_code_hash':old_hash,'phase1_go':False,
            'elapsed_seconds':time.perf_counter()-began,'evaluation_source_lock_sha256':file_hash(Path(evaluation_control)/'source_lock.json')}
    write_json(root/'development_comparison.json',result)
    lines=['# V5 Gate 原规则开发评估','',f"**{result['status']}**；这是已曝光开发集上的候选采用判定，Phase1未通过。",'',
           '三个当前种子均完成120轮，独立核验best/latest、RNG、优化器、调度、原train-only基底与实际源码后才运行评估。',
           '新版因果Transformer加载器明确使用新类和真实新签名；未冒用旧DynamicsPredictor来源。',
           f"原任务规则：{rules['status']}。固定参考recent检查通过 {sum(x['passed'] for x in checks if x['kind']=='candidate_vs_reference_recent')}/24；关键绝对MSE检查通过 {sum(x['passed'] for x in checks if x['kind']=='critical_absolute_mse_reduction')}/8。",'',
           '| seed | best epoch | ID h3/h6 privileged MSE | distractor h3/h6 privileged MSE |','|---|---:|---|---|']
    for seed in (17,29,43):
        values=[]
        for split in ('test_id','test_distractor_shift'):
            values.append('/'.join(f"{next(r['metrics']['normalized_mse']['mean'] for r in metrics['per_seed'] if r['seed']==seed and r['split']==split and r['horizon']==h and r['condition']=='privileged_small_budget'):.8f}" for h in (3,6)))
        lines.append(f"| {seed} | {completion['models'][f'gate/{seed}']['best_epoch']} | {values[0]} | {values[1]} |")
    lines+=['','全部逐种子/pooled、OOD、错误证据方向、任务读出与配对区间保存在evaluation/metrics.json及development_comparison.json。',
            '训练误差或验证误差下降不替代采用门槛；Motion、真实删除可行性与独立确认仍需分别完成。']
    if result['reasons']:lines+=['','未通过项目：']+['- '+r for r in result['reasons']]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(f"V5_GATE_EVALUATION_COMPLETE status={result['status']} raw_rows={len(raw)} elapsed_seconds={result['elapsed_seconds']:.2f}",flush=True)
    return result
