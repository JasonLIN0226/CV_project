"""Repair candidate loader with unchanged original metric loop and rules."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from .config import CONDITIONS, SPLITS
from .data import load_episode
from .inputs import FeatureStore, collate
from .model import fit_basis
from .probes import load_decoder
from .evaluation import (active_tasks, pair_calibration, pair_diagnostic, timed_forward, summarize, summarize_calibration)
from .reporting import decision, _development_artifacts
from .utils import read_json, write_json, write_csv, file_hash, code_hash, seed_all
from .repair_trial import verify_control, reference_model, signature, verify_completed_model
from .centered_transport import CenteredTransportPredictor
from .v5_synthetic_evaluation import fixed_reference_checks as gate_reference_checks
from .v5_motion_evaluation import fixed_reference_checks as motion_reference_checks


def all_models_completion(control, training_root, store=None):
    p, lock, configs = verify_control(control)
    root = Path(training_root)
    declared = read_json(root / 'all_models_completion.json')
    contract = read_json(root / 'run_contract.json')
    if (not declared['complete'] or contract['protocol'] != p or contract['source_files'] != lock['files']
            or len(declared['models']) != 6):
        raise ValueError('All six actually completed models and source contract required')
    proofs = {}
    for task in p['tasks']:
        c = configs[task]
        current_store = store if store is not None and store.c['output'] == c['output'] else FeatureStore(c)
        basis = fit_basis(c, current_store, task)
        for seed in p['seeds']:
            reference, sha = reference_model(c, p, task, seed, basis)
            sig = signature(control, p, c, current_store, task, seed, sha)
            key = f'{task}/{seed}'
            proofs[key] = verify_completed_model(root / 'predictor' / task / f'seed_{seed}', sig,
                                                  reference.state_dict())
            if proofs[key] != declared['models'][key]:
                raise ValueError('Stored completion certificate differs from current actual weights')
    return {'complete': True, 'models': proofs, 'required_models': 6, 'phase1_go': False}


def load_current_reader(control, training_root, seed, store, completion):
    if not completion['complete'] or len(completion['models']) != 6:
        raise ValueError('All-six proof is required before evaluation')
    p, _, configs = verify_control(control)
    task = active_tasks(store.c)[0]
    basis = fit_basis(store.c, store, task)
    store.bases = {task: basis}
    store.projected.clear()
    reference, _ = reference_model(configs[task], p, task, seed, basis)
    model = CenteredTransportPredictor(reference, basis)
    path = Path(training_root) / 'predictor' / task / f'seed_{seed}' / 'best.pt'
    proof = completion['models'][f'{task}/{seed}']
    if file_hash(path) != proof['best_sha256']:
        raise ValueError('Actual selected weights changed after proof')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['signature'] != proof['signature']:
        raise ValueError('Actual selected signature differs')
    model.load_state_dict(ck['model'], strict=True)
    return model.eval(), ck


def run_evaluation(control, training_root, output):
    began = time.perf_counter()
    p, _, configs = verify_control(control)
    root = Path(output)
    if root.exists():
        raise ValueError('Refuse overwrite or implicit repeated evaluation')
    root.mkdir(parents=True)
    seed_all(17, 4)
    write_json(root / 'evaluation_attempt.json', {'protocol_sha256': file_hash(Path(control) / 'protocol.json'),
               'all6_current_proof_required': True, 'phase1_go': False})
    completion = all_models_completion(control, training_root)
    write_json(root / 'all_models_completion.json', completion)
    results = {}
    for task, c in configs.items():
        metrics, raw, _ = evaluate_candidate(c, control, training_root, root / task)
        _, old_metrics, reference_raw, audit, probes = _development_artifacts(c['output'], task)
        verification = read_json(p['immutable_inputs'][task]['frozen_v3_verification'])
        for rel in ('evaluation/raw_records.csv', 'evaluation/metrics.json', 'probes/metrics.json'):
            key = task + '/reference/' + rel
            if file_hash(Path(c['output']) / rel) != verification['actual_hpc_compact_sha256'][key]:
                raise ValueError('Actual reference metric/probe differs from previous native proof')
        profile = copy.deepcopy(c)
        profile['data']['tasks'] = [task]
        profile.pop('experiment', None)
        original, reasons, checks = decision(profile, audit, probes, metrics)
        fixed = (gate_reference_checks if task == 'gate' else motion_reference_checks)(c, reference_raw, raw)
        fixed_failed = [x['slice'] + ' ' + x['kind'] for x in fixed if not x['passed']]
        result = {'task': task, 'status': 'SUPPORTED' if original == 'GO' and not fixed_failed else 'REJECTED',
                  'scope': 'Revealed6301 development only; not independent Phase1 confirmation',
                  'original_task_status': original, 'original_checks': checks,
                  'fixed_reference_checks': fixed, 'reasons': reasons + fixed_failed,
                  'candidate_raw_sha256': metrics['raw_sha256'], 'reference_raw_sha256': old_metrics['raw_sha256'],
                  'raw_records': len(raw), 'phase1_go': False}
        if len(raw) != 34560:
            raise ValueError('Incomplete original validation/ID/long-gap/distractor six-condition raw output')
        write_json(root / task / 'development_comparison.json', result)
        results[task] = result
    status = {'status': 'DEVELOPMENT_SUPPORTED_CONFIRMATION_PENDING' if all(x['status'] == 'SUPPORTED' for x in results.values())
              else 'DEVELOPMENT_REJECTED', 'tasks': results, 'phase1_go': False,
              'confirmation_opened': False, 'elapsed_seconds': time.perf_counter() - began}
    write_json(root / 'outcome.json', status)
    return status

def evaluate_candidate(c, control, training_root, output):
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
                            raise ValueError("Non-finite actual candidate predictions")
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
        "reader": "CenteredTransportPredictor", "candidate_evaluation_code_hash": code_hash("repair_evaluation.py", "repair_trial.py", "relational_memory.py", "evaluation.py", "stats.py"),
        "phase1_go": False, **summary})
    write_json(root/"latency_memory.json", latencies)
    if calibration:
        write_json(root/"calibration_summary.json", {"cache_signature": store.ix["signature"], **summarize_calibration(c, calibration)})
    return read_json(root/"metrics.json"), raw, completion
