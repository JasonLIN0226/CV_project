"""One post-hoc, train-derived support-mask ablation of frozen ridge probes.

This addresses numerical extrapolation, not a new fitted memory method. The
unmasked run is preserved. Mask all TRAIN-RMS-floor columns, identically for
cost/attention/null, without selecting columns or thresholds on fresh outcomes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/"experiments/predictor/control/src"):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
import numpy as np
from experiments.phase2.diagnose_oracle_learnability import (
    ARMS,RidgeProbe,ProbeRetention,causal_features,parent_from_arrays,paired_summary,
)
from experiments.phase2.generate_deletion_pilot import sha256,write
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.run_literature_baselines import overflow_parent


def no_model_call(*args,**kwargs):
    raise AssertionError("This support ablation does not call the world model")


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Use a new empty output")
    output.mkdir(parents=True);began=time.monotonic()
    summary=json.loads((source/"summary.json").read_text());old_protocol=json.loads((source/"protocol.json").read_text())
    assert sha256(source/"protocol.json")==summary["protocol_sha256"]
    for path,digest in old_protocol["source_sha256"].items():assert sha256(ROOT/path)==digest
    with np.load(source/"probe_parameters.npz") as data:
        original={name:RidgeProbe(data[name+"_scale"].copy(),data[name+"_weights"].copy(),float(data[name+"_label_scale"]),name) for name in ARMS}
    mask=original["cost_ridge"].scale<=1e-6
    for probe in original.values():assert np.array_equal(probe.scale,original["cost_ridge"].scale)
    # The support is determined entirely by existing train-only RMS, not labels.
    probes={name:RidgeProbe(p.scale.copy(),np.where(mask,0.,p.weights),p.label_scale,name) for name,p in original.items()}
    basis_path=ROOT/"build/history-diagnosis-20261020/followup-blas1/gate/unit_columns/basis.npz"
    with np.load(basis_path) as b:basis={k:b[k].copy() for k in ("input_mean","projection")}
    norm_path=ROOT/"build/hpc-20261020/development-assets/data/features/normalization.npz"
    protocol={"version":"posthoc-train-support-mask-v1","created_utc":datetime.now(timezone.utc).isoformat(),
       "reason":"Unmasked probe train-RMS floor creates very large fresh standardized inputs; one numerical-support diagnosis, not model selection",
       "source_run":str(source.relative_to(ROOT)),"source_summary_sha256":sha256(source/"summary.json"),
       "source_parameters_sha256":sha256(source/"probe_parameters.npz"),"basis_sha256":sha256(basis_path),
       "normalization_sha256":sha256(norm_path),"diagnostic_source_sha256":sha256(Path(__file__)),
       "mask_rule":"set coefficient to exactly zero iff old TRAIN parent-centered feature RMS scale <= 1e-6; no refit",
       "mask_columns":np.flatnonzero(mask).tolist(),"masked_columns":int(mask.sum()),"total_columns":len(mask),
       "all_arms_share_mask":True,"hyperparameters_changed":False,"fitting_calls":0,
       "evaluations":["all 1121 original common parents","all192episodes irreversible R4K2 replay for all3maskedarms"],
       "scope":"Post-hoc exploratory numerical diagnostic; keep original negative run; no final-selector or confirmation claim"}
    write(output/"protocol.json",protocol)  # Before masked choices or future-loss lookup.
    fresh=FreshGateDataset(ROOT/"build/literature-baselines-20261022/fresh-data")
    with np.load(norm_path) as n:fresh.verify_old_normalization({k:n[k] for k in ("mean","std")})
    scores=SimpleNamespace(_basis=basis,attention=no_model_call)
    cache=defaultdict(dict)
    for row in json.loads((ROOT/"build/literature-baselines-20261022/fresh-eval/oracle_subsets.json").read_text()):
        cache[row["episode_id"]][tuple(row["kept_times"])]=float(np.mean(row["feature_mse"]))
    old_common=json.loads((source/"common_parent_results.json").read_text())
    old_online={(r["episode_id"],r["arm"]):r for r in json.loads((source/"online_results.json").read_text())}
    baseline=defaultdict(dict)
    for row in json.loads((ROOT/"build/literature-baselines-20261022/fresh-eval/episode_results.json").read_text()):
        baseline[row["episode_id"]][(row["policy"],row["seed"])]=row
    common=[];online=[];traces=[];score_decomposition=[]
    for oldrow in old_common:
        rec=fresh.by_id[oldrow["episode_id"]];z,actions=fresh.episode(rec)
        c,r,p=parent_from_arrays(rec,z,actions,rec["query"],oldrow["candidate_times"])
        x=causal_features(c,r,rec["query"],basis,p);loss=np.array(oldrow["losses"])
        row={k:oldrow[k] for k in ("episode_id","pair","condition","candidate_times","losses","margin")}
        for name,probe in probes.items():
            score=probe.predict(x);original_score=original[name].predict(x);deleted=int(np.argmin(score))
            row[name+"_choice"]=deleted;row[name+"_mse"]=float(loss[deleted]);row[name+"_regret"]=float(loss[deleted]-min(loss))
            row[name+"_optimal"]=float(loss[deleted]<=min(loss)+1e-8)
            row[name+"_minus_original"]=float(loss[deleted]-oldrow[name+"_mse"])
            score_decomposition.append({"episode_id":rec["id"],"condition":rec["condition"],"arm":name,
                "original_score_range":float(np.ptp(original_score)),"supported_score_range":float(np.ptp(score)),
                "floor_column_score_range":float(np.ptp(original_score-score)),"choice_changed":bool(deleted!=oldrow["choices"][name])})
        row["cost_minus_attention"]=row["cost_ridge_mse"]-row["attention_ridge_mse"]
        row["cost_minus_null"]=row["cost_ridge_mse"]-row["shuffled_cost_ridge_mse"]
        common.append(row)
    for rec in fresh.records:
        z,actions=fresh.episode(rec);q=rec["query"]
        for name,probe in probes.items():
            policy=ProbeRetention(rec["id"],scores,probe)
            for t in range(q+1):
                if t==q:parent=overflow_parent(policy.stream.snapshot,z[t],rec["id"],t)
                snap,event=policy.observe(z[t],executed_action=None if t==0 else actions[t-1])
                traces.append({"episode_id":rec["id"],"arm":name,**event})
            kept=tuple(sorted(e.timestamp for e in snap.retained));times=sorted(e.timestamp for e in parent[0])
            mse=cache[rec["id"]][kept]
            local=min(cache[rec["id"]][tuple(k for j,k in enumerate(times) if j!=d)] for d in range(3))
            b=baseline[rec["id"]];groups=b[("attention",17)]["evidence_equivalent_feature_times"]
            row={"episode_id":rec["id"],"pair":rec["pair"],"condition":rec["condition"],"arm":name,
                 "kept_times":list(kept),"candidate_times":times,"mse":mse,"local_regret":mse-local,
                 "hindsight_regret":mse-min(v for k,v in cache[rec["id"]].items() if len(k)==2),
                 "cue_retained":float(all(set(kept).intersection(group) for group in groups)),
                 "minus_original":mse-old_online[(rec["id"],name)]["mse"],"parameter_bytes":probe.bytes,
                 "dynamic_logical_maxima":policy.maxima,"online_seconds":policy.seconds}
            for control in ("fifo","attention","visual_diversity","wr"):
                row["minus_"+control]=mse-b[(control,17)]["mean_mse"]
            row["minus_pafu"]=mse-np.mean([b[("pafu",seed)]["mean_mse"] for seed in (17,23,41)])
            online.append(row)
    grouped=defaultdict(dict)
    for row in online:grouped[row["episode_id"]][row["arm"]]=row
    for arms in grouped.values():
        for row in arms.values():
            row["minus_attention_ridge"]=row["mse"]-arms["attention_ridge"]["mse"]
            row["minus_shuffled_cost_ridge"]=row["mse"]-arms["shuffled_cost_ridge"]["mse"]
    conditions=["all","id","long_gap","distractor_shift"]
    fields=[name+"_"+metric for name in ARMS for metric in ("mse","regret","optimal","minus_original")]+["cost_minus_attention","cost_minus_null"]
    stats={"common_parent":paired_summary(common,fields,conditions),"online":{name:paired_summary([r for r in online if r["arm"]==name],
        ["mse","local_regret","hindsight_regret","cue_retained","minus_original","minus_fifo","minus_attention","minus_visual_diversity","minus_wr","minus_pafu","minus_attention_ridge","minus_shuffled_cost_ridge"],conditions) for name in ARMS}}
    write(output/"common_parent_results.json",common);write(output/"online_results.json",online);write(output/"statistics.json",stats)
    write(output/"score_decomposition.json",score_decomposition)
    with (output/"online_trace.jsonl").open("w") as file:
        for row in traces:file.write(json.dumps(row,allow_nan=False)+"\n")
    assert sha256(Path(__file__))==protocol["diagnostic_source_sha256"]
    for path,digest in old_protocol["source_sha256"].items():assert sha256(ROOT/path)==digest
    result={"status":"completed_posthoc_train_support_mask","elapsed_seconds":time.monotonic()-began,
            "protocol_sha256":sha256(output/"protocol.json"),"model_fits":0,"world_model_calls":0,"new_GPU_seconds":0,
            "common_parent_rows":len(common),"online_rows":len(online),"trace_events":len(traces),
            "masked_columns":int(mask.sum()),"all_arms_fixed_numeric_storage_bytes":next(iter(probes.values())).bytes,
            "online_statistics":stats["online"],"common_parent_statistics":stats["common_parent"],
            "source_run_kept_unchanged":sha256(source/"summary.json")==protocol["source_summary_sha256"],
            "artifact_sha256":{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write(output/"summary.json",result)
    print(json.dumps({k:result[k] for k in ("status","elapsed_seconds","masked_columns","common_parent_rows","online_rows")},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
