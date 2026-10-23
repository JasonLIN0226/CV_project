"""Static original metric loop: cached input/targets and all conditions unchanged."""
from pathlib import Path
import numpy as np
import torch
from phase1.config import CONDITIONS,SPLITS
from phase1.data import load_episode
from phase1.inputs import FeatureStore,collate
from phase1.probes import load_decoder
from phase1.evaluation import active_tasks,pair_calibration,pair_diagnostic,summarize,summarize_calibration
from phase1.utils import read_json,write_json,write_csv,file_hash,code_hash
from intervention import both,state_digest
COMPLETION=None
LOADER=None
STATE_PROOFS={}
BRANCH_RESIDUALS={}
class FrozenOff(torch.nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.initial_digest=state_digest(base);self.max_residual=0.
 @property
 def projection(self):return self.base.projection
 def forward(self,batch):
  values,residual=both(self.base,batch);self.max_residual=max(self.max_residual,residual)
  return torch.from_numpy(values['transport_off'])
def load_locked(seed,store):
 model,ck=LOADER(seed,store);return FrozenOff(model).eval().requires_grad_(False),ck
def evaluate_candidate(c, control, training_root, output):
    device = torch.device("cpu")
    store = FeatureStore(c, max_cached=2*max(c["data"]["pairs"][s] for s in SPLITS[1:]))
    completion = COMPLETION
    if completion is not None and not completion["complete"]:
        raise ValueError("Held-out evaluation requires all prescribed training to complete: "+"; ".join(completion["issues"]))
    decoders = {task: load_decoder(c, task) for task in active_tasks(c)}
    raw, latencies, checkpoints, calibration, pending = [], [], {}, [], {}
    for seed in c["training"]["seeds"]:
        for task in active_tasks(c):
            model, ck = load_locked(seed, store)
            checkpoints[f"{task}/{seed}"] = {"task": task, "seed": seed, "best_epoch": ck["epoch"], "training_seconds": ck["seconds"],
                "best_sha256": file_hash(Path(training_root)/"predictor"/task/f"seed_{seed}"/"best.pt")}
            for split in SPLITS[1:]:
                rows = [r for r in store.manifest["records"] if r["task"] == task and r["split"] == split]
                for condition in CONDITIONS+["persistence"]:
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
                    write_json(Path(output)/"progress.json",dict(stage="original_prediction",seed=seed,split=split,condition=condition,rows=len(raw)))
            assert state_digest(model.base)==model.initial_digest
            STATE_PROOFS[str(seed)]=model.initial_digest
            BRANCH_RESIDUALS[str(seed)]=model.max_residual
    root = Path(output)/"evaluation"
    if pending:
        raise ValueError("Calibration encountered an incomplete gate sibling pair")
    write_csv(root/"raw_records.csv", raw)
    summary = summarize(c, raw)
    write_json(root/"metrics.json", {"mode": c["mode"], "cache_signature": store.ix["signature"],
        "raw_sha256": file_hash(root/"raw_records.csv"), "legacy_metric_helpers_code_hash": code_hash("evaluation.py", "stats.py"),
        "checkpoints": checkpoints, "training_completion": completion, "data_source_experiment": c["experiment"],
        "reader": "FrozenCenteredTransportPredictor_transport_values_zero", "candidate_evaluation_code_hash": code_hash("repair_evaluation.py", "repair_trial.py", "relational_memory.py", "evaluation.py", "stats.py"),
        "phase1_go": False, **summary})
    write_json(root/"latency_memory.json", latencies)
    if calibration:
        write_json(root/"calibration_summary.json", {"cache_signature": store.ix["signature"], **summarize_calibration(c, calibration)})
    return read_json(root/"metrics.json"), raw, completion
