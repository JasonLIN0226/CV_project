"""Fixed 512-pair reader and TRAIN-only use-time ridge pilot, CPU execution.

Uses the completed reader-scale A100 archive. No encoding, reader training,
hyperparameter search, GPU access, TD, or distractor-task modification occurs.
Evaluation scenes have already been inspected for reader diagnosis: this is a
pilot, not independent confirmation. Reader and scorer training coverage both
increase relative to the earlier experiment, so their effects are not isolated.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import MappingProxyType

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/"experiments/predictor/control/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
import numpy as np
import torch
from phase1.model import DynamicsPredictor,project
from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.deletion import EvaluatorProvenance
from experiments.phase2.diagnose_gate_oracle_metric import paired_components
from experiments.phase2.diagnose_oracle_learnability import RidgeProbe,ProbeRetention,causal_features,parent_from_arrays
from experiments.phase2.frozen_predictor import FrozenPredictorEvaluator,_immutable,_state_hash,_versions
from experiments.phase2.generate_deletion_pilot import sha256,write
from experiments.phase2.run_literature_baselines import QueryOracle,attention_teacher,online_policy,overflow_parent
from experiments.phase2.run_retention_baseline_pilot import OnlineRetention
from experiments.phase2.run_use_time_probe import AttentionContinuation,deletion_labels,pair_ci

ARMS=("local_cost","use_time_cost","future_attention","shuffled_use_time_cost")
PARENT_POLICIES=("fifo","reservoir","visual_diversity","attention","wr")


class ScaledReader(FrozenPredictorEvaluator):
    """Bind only the completed scale-512 model; inherits frozen CPU checks."""
    def __init__(self,run,inputs,completion):
        checkpoint=run/"pairs-512/selected.pt"
        paths=[checkpoint,inputs/"basis.npz",inputs/"normalization.npz",inputs/"model_config.json",
               run/"protocol.json",run/"readers_locked.json"]
        if sha256(checkpoint)!=completion["files"]["pairs-512/selected.pt"]:
            raise ValueError("Scale-512 checkpoint differs from completed execution")
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if saved["train_pairs"]!=512 or saved["basis_sha256"]!=sha256(inputs/"basis.npz") or saved["normalization_sha256"]!=sha256(inputs/"normalization.npz"):
            raise ValueError("Checkpoint training/basis/normalization binding mismatch")
        with np.load(inputs/"basis.npz",allow_pickle=False) as a:
            basis={k:a[k].copy() for k in a.files}
        for key in ("input_mean","projection","target_mean","output_basis"):
            if not np.array_equal(saved["model"][key].numpy(),basis[key]):
                raise ValueError("Checkpoint buffer differs from fixed basis: "+key)
        config=json.loads((inputs/"model_config.json").read_text())
        model=DynamicsPredictor(5120,config,(3,6),basis)
        model.load_state_dict(saved["model"],strict=True)
        self._model=model.cpu().eval().requires_grad_(False)
        self._basis=MappingProxyType({k:_immutable(basis[k]) for k in ("input_mean","projection")})
        self._artifacts=MappingProxyType({p:sha256(p) for p in paths})
        self._initial_hash,self._initial_versions=_state_hash(model),_versions(model)
        self._calls={"count":0}
        self.task,self.seed,self.horizons,self.feature_dimension="gate",17,(3,6),5120
        self.provenance=EvaluatorProvenance(sha256(checkpoint),"reader_scale_512_fixed_cpu")
        self.audit()


def run(args):
    work=args.work.resolve();source=work/"run";inputs=(args.inputs or work/"bundle/inputs").resolve()
    out=(args.output or work/"scaled-use-time-probe").resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output")
    out.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();torch.set_num_threads(1)
    completion=json.loads((source/"completion.json").read_text())
    if completion["status"]!="completed":
        raise ValueError("Reader-scale experiment incomplete")
    checked={}
    def verify(relative):
        p=(source/relative).resolve()
        if not p.is_relative_to(source) or relative not in completion["files"] or sha256(p)!=completion["files"][relative]:
            raise ValueError("Completed source hash mismatch: "+relative)
        checked[relative]=completion["files"][relative]
        return p
    for name in ("protocol.json","readers_locked.json","data/records.json","data/manifest.json"):
        verify(name)
    reader_protocol=json.loads((source/"protocol.json").read_text())
    if sha256(inputs/"manifest.json")!=reader_protocol["inputs_manifest_sha256"]:
        raise ValueError("Input manifest differs from reader training")
    im=json.loads((inputs/"manifest.json").read_text())
    for name in ("basis.npz","normalization.npz","model_config.json"):
        if sha256(inputs/name)!=im["files"][name]:
            raise ValueError("Fixed input differs: "+name)
    records=json.loads((source/"data/records.json").read_text());by_id={r["id"]:r for r in records}
    training=[r for r in records if r["split"]=="train"]
    evaluation=[r for r in records if r["split"]=="evaluation"]
    if len(training)!=1024 or len(evaluation)!=96 or len({r["pair"] for r in training})!=512 or len({r["pair"] for r in evaluation})!=48:
        raise ValueError("Unexpected fixed training/evaluation population")
    if {r["pair"] for r in training}&{r["pair"] for r in evaluation}:
        raise ValueError("Train/evaluation overlap")
    reader=ScaledReader(source,inputs,completion);scores=CausalModelScores(reader)
    def episode(rec):
        with np.load(verify("data/"+rec["feature_path"]),allow_pickle=False) as a:
            z,actions=a["z"].copy(),a["actions"].copy()
        if z.dtype!=np.float32 or z.shape!=(rec["frames"],5120) or actions.dtype!=np.float32 or not np.isfinite(z).all() or not np.isfinite(actions).all():
            raise ValueError("Invalid normalized episode")
        return z,actions
    sources=[Path(__file__)]+[ROOT/"experiments/phase2"/f for f in (
        "run_use_time_probe.py","diagnose_oracle_learnability.py","memory.py","run_literature_baselines.py",
        "literature_baselines.py","causal_model_scores.py","run_retention_baseline_pilot.py")]
    protocol={"version":"scaled-use-time-pilot-v1","started_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Pilot on already-inspected reader evaluation scenes; not untouched confirmation",
        "source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        "reader_completion_sha256":sha256(source/"completion.json"),"reader":reader.audit(),
        "train_pairs":512,"evaluation_pairs":48,"parent_policies":PARENT_POLICIES,
        "parents":"All unique episode/time/three-timestamp parents reached by the five fixed policies",
        "teacher":"one deletion followed by fixed attention continuation to recorded query; mean q+3/q+6",
        "student":"same 903 causal features, float32 storage, parent-centered ridge lambda=1; TRAIN RMS support mask<=1e-6",
        "input_numerics":"Causal features are computed in float64 then stored float32; ridge fitting casts back to float64. All new arms share these exact inputs. Earlier 16-pair probe retained float64 features, so cross-experiment numerical identity is not claimed.",
        "arms":ARMS,"controls":["attention","visual_diversity"],"permutation_seed":17,
        "R":4,"K":2,"horizons":[3,6],"hyperparameter_selection":False,"new_gpu_seconds":0,
        "confounds":["Reader and scorer training populations both expand versus the 16-pair experiment",
                     "Reader was trained at query; local labels at other times remain outside that origin distribution",
                     "Frozen old-16-pair fitted feature transforms remain unchanged"]}
    write(out/"protocol.json",protocol)
    features=[];local_labels=[];use_labels=[];attention_labels=[];meta=[]
    label_started=time.monotonic();memo_states=0
    for index,rec in enumerate(training):
        z,actions=episode(rec);projected=project(z,scores._basis)
        continuation=AttentionContinuation(rec["id"],z,rec["query"],scores.attention)
        query_oracle=QueryOracle(rec["id"],z,actions,rec["query"],scores)
        local_oracles={};seen=set()
        for policy in PARENT_POLICIES:
            online=online_policy(rec["id"],policy,scores,seed=17)
            for t in range(rec["query"]+1):
                parent=overflow_parent(online.stream.snapshot,z[t],rec["id"],t)
                if parent is not None:
                    times=tuple(sorted(e.timestamp for e in parent[0]));key=(t,times)
                    if key not in seen:
                        seen.add(key)
                        candidates,recent,past=parent_from_arrays(rec,z,actions,t,times)
                        features.append(causal_features(candidates,recent,t,scores._basis,past).astype(np.float32))
                        local_oracles.setdefault(t,QueryOracle(rec["id"],z,actions,t,scores))
                        local=[float(local_oracles[t].evaluate([k for j,k in enumerate(times) if j!=d])[0].mean()) for d in range(3)]
                        use,detail=deletion_labels(rec,z,actions,t,times,scores,continuation,query_oracle)
                        teacher,_=attention_teacher(scores,projected,list(times),t)
                        local_labels.append(local);use_labels.append(use);attention_labels.append(teacher)
                        meta.append({"episode_id":rec["id"],"pair":rec["pair"],"split":"train","time":t,
                            "query":rec["query"],"candidate_times":times,"parent_policy":policy,"use_time_final_kept":detail["final_kept"]})
                online.observe(z[t],executed_action=None if t==0 else actions[t-1])
        memo_states+=len(continuation.cache)
        if (index+1)%32==0:
            print(json.dumps({"stage":"labels","episodes":index+1,"parents":len(meta),"seconds":time.monotonic()-label_started}),flush=True)
    x=np.stack(features);del features
    local=np.asarray(local_labels);use=np.asarray(use_labels);attention=np.asarray(attention_labels)
    rng=np.random.default_rng(17);null=np.array([r[rng.permutation(3)] for r in use])
    labels=dict(zip(ARMS,(local,use,attention,null)))
    # Keep exact input/label bytes for later no-model audits; the file may be large.
    np.save(out/"training_inputs.npy",x)
    np.savez_compressed(out/"training_labels.npz",**labels)
    write(out/"training_metadata.json",meta)
    label_seconds=time.monotonic()-label_started;fit_started=time.monotonic();probes={};fits={}
    for name,y in labels.items():
        p=RidgeProbe.fit(x,y,meta,name)
        probes[name]=RidgeProbe(p.scale,np.where(p.scale<=1e-6,0.,p.weights),p.label_scale,name)
        predicted=np.array([probes[name].predict(v) for v in x]);chosen=predicted.argmin(1)
        regret=y[np.arange(len(y)),chosen]-y.min(1)
        fits[name]={"mean_label_regret":float(regret.mean()),"optimal_within_1e-8":int(np.sum(regret<=1e-8)),
            "tied_minimum_parents":int(np.sum(np.sum(y<=y.min(1,keepdims=True)+1e-8,axis=1)>1)),
            "centered_label_rms":float(np.sqrt(np.mean((y-y.mean(1,keepdims=True))**2)))}
    fit_seconds=time.monotonic()-fit_started;del x
    np.savez_compressed(out/"probe_parameters.npz",**{name+"_"+k:getattr(p,k) for name,p in probes.items() for k in ("scale","weights","label_scale")})
    write(out/"training_fit.json",fits)
    rows=[];traces=[]
    for index,rec in enumerate(evaluation):
        z,actions=episode(rec);sibling,_=episode(by_id[rec["sibling"]]);q=rec["query"]
        cue=set(np.flatnonzero(np.any(z[:q-3]!=sibling[:q-3],axis=1)).tolist())
        if not cue:
            raise ValueError("Evaluation sibling has no visible historical cue contrast")
        oracle=QueryOracle(rec["id"],z,actions,q,scores);episode_rows=[]
        for name in (*ARMS,"attention","visual_diversity"):
            online=ProbeRetention(rec["id"],scores,probes[name]) if name in probes else OnlineRetention(rec["id"],name,scores,seed=17)
            events=[]
            for t in range(q+1):
                snap,event=online.observe(z[t],executed_action=None if t==0 else actions[t-1]);events.append(event)
            kept=[e.timestamp for e in snap.retained];loss,pred=oracle.evaluate(kept)
            metrics=paired_components(pred,z[q+np.array([3,6])],sibling[q+np.array([3,6])])
            row={"episode_id":rec["id"],"pair":rec["pair"],"condition":rec["condition"],"arm":name,
                "query":q,"kept_times":kept,"cue_times":sorted(cue),"cue_retained":float(bool(cue.intersection(kept))),
                "horizon_losses":loss.tolist(),**metrics,"online_seconds":online.seconds,"logical_memory_maxima":online.maxima,
                "probe_parameter_bytes":probes[name].bytes if name in probes else 0}
            episode_rows.append(row);traces.extend({"episode_id":rec["id"],"arm":name,**e} for e in events)
        by_arm={r["arm"]:r for r in episode_rows}
        for row in episode_rows:
            for control in (*ARMS,"attention","visual_diversity"):
                row["minus_"+control]=row["mse"]-by_arm[control]["mse"]
        rows.extend(episode_rows)
        if (index+1)%16==0:
            print(json.dumps({"stage":"evaluation","episodes":index+1,"seconds":time.monotonic()-began}),flush=True)
    write(out/"online_results.json",rows)
    with (out/"online_trace.jsonl").open("w") as f:
        for row in traces:f.write(json.dumps(row,allow_nan=False)+"\n")
    stats={}
    for name in (*ARMS,"attention","visual_diversity"):
        stats[name]={}
        for condition in ("all","id","long_gap","distractor_shift"):
            selected=[r for r in rows if r["arm"]==name and (condition=="all" or r["condition"]==condition)]
            fields=("mse","cue_retained","feature_branch_correct","minus_local_cost","minus_future_attention",
                    "minus_shuffled_use_time_cost","minus_attention","minus_visual_diversity")
            stats[name][condition]={key:pair_ci(selected,key) for key in fields}
    if any(sha256(ROOT/name)!=h for name,h in protocol["source_sha256"].items()):
        raise ValueError("Source changed during experiment")
    final_audit=reader.audit()
    summary={"status":"completed_scaled_use_time_pilot","elapsed_seconds":time.monotonic()-began,
        "training_parents":len(meta),"label_seconds":label_seconds,"fit_seconds":fit_seconds,
        "feature_width":903,"supported_features":int(np.sum(probes["local_cost"].scale>1e-6)),
        "teacher_memo_states":memo_states,"training_fit":fits,"statistics":stats,"reader_audit":final_audit,
        "checked_source_artifacts":checked,"new_gpu_seconds":0,"new_reader_training":False,"new_scorer_fitting":True,
        "limits":protocol["confounds"]+["Already inspected reader evaluation set; pilot only",
            "Same K and matched ridge parameters; heuristic controls differ in controller memory and compute",
            "Sibling feature-contrast sign is not decoded physical lane accuracy"]}
    summary["result_sha256"]={p.name:sha256(p) for p in out.iterdir() if p.is_file()}
    write(out/"summary.json",summary)
    print(json.dumps({"stage":"complete","parents":len(meta),"elapsed_seconds":summary["elapsed_seconds"],
        "online":{name:{k:stats[name]["all"][k]["mean"] for k in ("mse","cue_retained","feature_branch_correct")} for name in stats}},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--inputs",type=Path)
    p.add_argument("--output",type=Path)
    run(p.parse_args())
