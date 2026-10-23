"""Audit and publish the fixed learnability diagnostic without fitting a model."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/"experiments/predictor/control/src"):
    if str(directory) not in sys.path: sys.path.insert(0,str(directory))
import numpy as np
from experiments.phase2.diagnose_oracle_learnability import (
    ARMS, RidgeProbe, causal_features, parent_from_arrays, parent_hashes, ambiguity_report,
)
from experiments.phase2.generate_deletion_pilot import sha256, write
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset


def quantiles(values):
    a=np.asarray(values)
    return {str(q):float(np.quantile(a,q)) for q in (0,.25,.5,.75,.95,.99,1)}


def sanitize(value):
    if isinstance(value,str): return value.replace(str(ROOT),"<CV_PROJECT>")
    if isinstance(value,dict):return {sanitize(k):sanitize(v) for k,v in value.items()}
    if isinstance(value,list):return [sanitize(v) for v in value]
    return value


def independent_pair_means(rows, field, condition="all"):
    episodes=defaultdict(list); pairs=defaultdict(list)
    for row in rows:
        if condition=="all" or row["condition"]==condition:
            episodes[(row["pair"],row["episode_id"])].append(row[field])
    for (pair,_),values in episodes.items():pairs[pair].append(float(np.mean(values)))
    return np.array([np.mean(values) for values in pairs.values()])


def audit_and_export(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError("Use an empty portable output")
    summary=json.loads((source/"summary.json").read_text());protocol=json.loads((source/"protocol.json").read_text())
    assert sha256(source/"protocol.json")==summary["protocol_sha256"]
    for name,digest in summary["result_sha256"].items():assert sha256(source/name)==digest,name
    for name,digest in protocol["source_sha256"].items():assert sha256(ROOT/name)==digest,name
    for name,digest in protocol["input_sha256"].items():assert sha256(ROOT/name)==digest,name
    common=json.loads((source/"common_parent_results.json").read_text())
    online=json.loads((source/"online_results.json").read_text())
    ambiguity=json.loads((source/"ambiguity.json").read_text())
    stat=json.loads((source/"statistics.json").read_text())
    params=np.load(source/"probe_parameters.npz")
    probes={name:RidgeProbe(params[name+"_scale"],params[name+"_weights"],float(params[name+"_label_scale"]),name) for name in ARMS}
    old=PortableSnapshot(ROOT/"build/hpc-20261020/development-assets")
    fresh=FreshGateDataset(ROOT/"build/literature-baselines-20261022/fresh-data")
    fresh.verify_old_normalization(old.norm)
    with np.load(ROOT/"build/history-diagnosis-20261020/followup-blas1/gate/unit_columns/basis.npz") as b:
        basis={k:b[k].copy() for k in ("input_mean","projection")}
    with np.load(source/"training_inputs_labels.npz") as data:
        tx,tc,ta,tn=(data[k].copy() for k in ("inputs","cost","attention","shuffled_cost"))
    metadata=json.loads((source/"training_metadata.json").read_text())
    # Rebuild every train input from actual old TRAIN observations/actions.
    for x,m in zip(tx,metadata):
        rec=old.by_id[m["episode_id"]];assert rec["split"]==m["split"]=="train"
        c,r,p=parent_from_arrays(rec,old.features(rec),old.actions(rec),m["time"],m["candidate_times"])
        np.testing.assert_array_equal(causal_features(c,r,m["time"],basis,p),x)
    tc_center=tc-tc.mean(1,keepdims=True); tx_center=tx-tx.mean(1,keepdims=True)
    training_metrics={}
    for name,y in zip(ARMS,(tc,ta,tn)):
        y=np.asarray(y,np.float64)
        p=probes[name]; a=(tx_center/p.scale).reshape(-1,tx.shape[-1]); b=((y-y.mean(1,keepdims=True))/p.label_scale).ravel()
        residual=(a.T@a/len(a)+np.eye(a.shape[1]))@p.weights-a.T@b/len(a)
        assert np.max(np.abs(residual))<1e-10
        pred=(tx_center/p.scale)@p.weights*p.label_scale; choices=pred.argmin(1)
        selected=tc[np.arange(len(tc)),choices]
        training_metrics[name]={"training_parent_optimal_fraction":float(np.mean(selected<=tc.min(1)+1e-8)),
            "training_parent_mean_regret":float(np.mean(selected-tc.min(1))),
            "normal_equation_max_residual":float(np.max(np.abs(residual))),
            "fit_target_centered_mse":float(np.mean((pred-(y-y.mean(1,keepdims=True)))**2))}
    cache=defaultdict(dict)
    for row in json.loads((ROOT/"build/literature-baselines-20261022/fresh-eval/oracle_subsets.json").read_text()):
        cache[row["episode_id"]][tuple(row["kept_times"])]=float(np.mean(row["feature_mse"]))
    fresh_inputs=[]
    for row in common:
        rec=fresh.by_id[row["episode_id"]];z,ac=fresh.episode(rec);times=row["candidate_times"];q=rec["query"]
        c,r,p=parent_from_arrays(rec,z,ac,q,times);x=causal_features(c,r,q,basis,p);fresh_inputs.append(x)
        assert row["hashes"]==parent_hashes(z,ac,q,times,x)
        loss=[cache[rec["id"]][tuple(k for j,k in enumerate(times) if j!=d)] for d in range(3)]
        np.testing.assert_array_equal(loss,row["losses"])
        for name,probe in probes.items():
            choice=int(np.argmin(probe.predict(x)));assert choice==row["choices"][name]
            assert row[name+"_mse"]==loss[choice]
            assert row[name+"_regret"]==loss[choice]-min(loss)
    for row in online:
        assert row["mse"]==cache[row["episode_id"]][tuple(row["kept_times"])]
    for key in ambiguity:
        assert ambiguity[key]==ambiguity_report(common,key)
    # Independent confidence-interval recomputation, with episode means first.
    max_ci_difference=0.
    for source_rows,groups in [(common,stat["common_parent"])]+[
        ([r for r in online if r["arm"]==name],stat["online"][name]) for name in ARMS]:
        for condition,fields in groups.items():
            for field,expected in fields.items():
                values=independent_pair_means(source_rows,field,condition)
                rng=np.random.default_rng(20261022)
                boot=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
                actual=[values.mean(),np.quantile(boot,.025),np.quantile(boot,.975)]
                delta=max(abs(a-expected[k]) for a,k in zip(actual,("mean","low","high")))
                assert delta<1e-12;max_ci_difference=max(max_ci_difference,float(delta))
    # Norm and ridge leverage are post-run input-shift diagnostics, not new fits.
    fx=np.array(fresh_inputs);fxc=fx-fx.mean(1,keepdims=True)
    a=(tx_center/probes["cost_ridge"].scale).reshape(-1,tx.shape[-1])
    b=(fxc/probes["cost_ridge"].scale).reshape(-1,tx.shape[-1])
    covariance=a.T@a/len(a)+np.eye(a.shape[1])
    trlev=np.sum(a*np.linalg.solve(covariance,a.T).T,1)
    flev=np.sum(b*np.linalg.solve(covariance,b.T).T,1)
    train_norm=np.linalg.norm(a,axis=1);fresh_norm=np.linalg.norm(b,axis=1)
    shift={"scope":"Post-run descriptive input shift only; no regularizer or model selection", "train_norm":quantiles(train_norm),
           "fresh_norm":quantiles(fresh_norm),"train_ridge_leverage":quantiles(trlev),"fresh_ridge_leverage":quantiles(flev),
           "fresh_rows_over_train_norm_p99":float(np.mean(fresh_norm>np.quantile(train_norm,.99))),
           "fresh_rows_over_train_leverage_p99":float(np.mean(flev>np.quantile(trlev,.99))),
           "by_condition":{}}
    for condition in ("id","long_gap","distractor_shift"):
        mask=np.repeat([r["condition"]==condition for r in common],3)
        shift["by_condition"][condition]={"norm":quantiles(fresh_norm[mask]),"leverage":quantiles(flev[mask]),
             "over_train_norm_p99":float(np.mean(fresh_norm[mask]>np.quantile(train_norm,.99))),
             "over_train_leverage_p99":float(np.mean(flev[mask]>np.quantile(trlev,.99)))}
    trace_count=0; deleted=defaultdict(set); last_time={}
    with (source/"online_trace.jsonl").open() as file:
        for line in file:
            row=json.loads(line);key=(row["episode_id"],row["arm"])
            assert row["time"]==last_time.get(key,-1)+1;last_time[key]=row["time"]
            resident=set(row["retained_timestamps"]+row["recent_timestamps"])
            assert not resident.intersection(deleted[key])
            assert len(row["retained_timestamps"])<=2 and len(row["recent_timestamps"])<=4
            if row["deleted_timestamp"] is not None:deleted[key].add(row["deleted_timestamp"])
            trace_count+=1
    conflicts=ambiguity["bounded_observed_state"]["conflict_examples"]
    worst=max(conflicts,key=lambda r:r["empirical_shared_action_regret"])
    all_same_future_action=sum(len(r["future_action_hashes"])==1 for r in conflicts)
    matching=[r for r in common if r["episode_id"] in worst["episodes"] and r["candidate_times"]==worst["candidate_times"]]
    best_online=max((r for r in online if r["arm"]=="cost_ridge"),key=lambda r:r["minus_attention_ridge"])
    examples={"selection":"Largest empirical shared-action regret among exact bounded-state conflicts; largest cost-minus-attention online loss",
              "bounded_alias":worst,"bounded_alias_rows":matching,"online_cost_failure":best_online,
              "conflict_groups_with_identical_recorded_future_actions":all_same_future_action,
              "total_conflict_groups":len(conflicts)}
    audit={"training_input_rows_exact":len(tx),"training_source_pairs":len({m['pair'] for m in metadata}),
           "common_parent_cost_rows_exact":len(common),"probe_choices_recomputed":len(common)*len(ARMS),
           "online_query_losses_exact":len(online),"irreversible_trace_events_checked":trace_count,
           "max_pair_CI_recomputation_difference":max_ci_difference,"fresh_artifact_hashes":len(fresh.loaded),
           "models_refit":0,"world_model_calls":0,"new_GPU_seconds":0,"training_metrics":training_metrics,
           "input_shift":shift,"limits":"Cost-label world-model inference was performed in the frozen run; this export checks its provenance and cached fresh losses, not an independent new training-label inference."}
    output.mkdir(parents=True)
    names=("protocol.json","summary.json","statistics.json","common_parent_results.json","online_results.json",
           "ambiguity.json","training_metadata.json")
    raw={}
    for name in names:
        raw[name]={"sha256":sha256(source/name),"bytes":(source/name).stat().st_size}
        write(output/name,sanitize(json.loads((source/name).read_text())))
    write(output/"audit.json",audit);write(output/"bad_cases.json",examples)
    write(output/"probe_parameters.json",{name:{"scale":p.scale.tolist(),"weights":p.weights.tolist(),"label_scale":p.label_scale}
                                         for name,p in probes.items()})
    source_dir=output/"source";source_dir.mkdir()
    for file in (Path(__file__),ROOT/"experiments/phase2/diagnose_oracle_learnability.py"):
        shutil.copy2(file,source_dir/file.name)
    render(output)
    (output/"README.md").write_text("""# Causal oracle-learnability diagnostic

Fixed old-TRAIN-only ridge probes; 16 training pairs and 96 fresh development
pairs. `protocol.json` was written before labels/fitting. Cost, attention and
shuffled-cost probes share 903 features/weights and lambda=1. The three retained
parameter/scale arrays are published as JSON. This is not the final proposed
selector or an untouched confirmation result.

`audit.json` independently reconstructs causal training inputs, fresh parent
costs, probe choices, grouped confidence intervals and irreversible trace
invariants. It also describes training fit and input shift without model tuning.
`ambiguity.json` separates complete past from bounded state: the repeated states
are after information has been lost; a better earlier retention policy could
avoid them. `bad_cases.json` gives deterministic selection rules and complete
three-way losses. The full traces and binary training arrays remain in ignored
`build/literature-diagnosis-20261022/learnability-v2`, hash-bound by the raw summary.

The earlier v1 run was superseded only to fix sibling weighting in summary
statistics (average parents within each episode before averaging siblings).
The v2 protocol/hyperparameters were unchanged; fits and per-case predictions
are identical. No outcome-driven model selection was performed.

Run on the recorded local development assets:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/phase2/diagnose_oracle_learnability.py --output build/learnability-NEW
.venv/bin/python experiments/phase2/export_oracle_learnability.py --source build/learnability-NEW --output results/learnability-NEW
```
""")
    manifest={"raw_source_directory":str(source.relative_to(ROOT)),"raw_json":raw,
              "transformations":["Absolute repository prefixes redacted to <CV_PROJECT>; scalar values unchanged","Parameters converted losslessly from NumPy float64 to JSON"],
              "exporter_sha256":sha256(Path(__file__)),"published_files":{str(p.relative_to(output)):{"sha256":sha256(p),"bytes":p.stat().st_size}
                    for p in sorted(output.rglob('*')) if p.is_file()}}
    write(output/"manifest.json",manifest)
    print(json.dumps({"output":str(output),"audit":{k:v for k,v in audit.items() if k not in ('training_metrics','input_shift')},"input_shift":shift},indent=2))


def render(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats=json.loads((output/"statistics.json").read_text())["online"]
    conditions=("id","long_gap","distractor_shift")
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    colors=("#bb6545","#327b99","#a5adb1")
    names=("Cost ridge","Attention ridge","Shuffled cost")
    x=np.arange(3)
    for j,(arm,color,label) in enumerate(zip(ARMS,colors,names)):
        ys=[stats[arm][c]["mse"]["mean"] for c in conditions]
        lo=[y-stats[arm][c]["mse"]["low"] for y,c in zip(ys,conditions)]
        hi=[stats[arm][c]["mse"]["high"]-y for y,c in zip(ys,conditions)]
        axes[0].bar(x+(j-1)*.24,ys,.23,label=label,color=color,yerr=[lo,hi],capsize=3)
        axes[1].bar(x+(j-1)*.24,[stats[arm][c]["cue_retained"]["mean"] for c in conditions],.23,color=color,label=label)
    for ax in axes:
        ax.set_xticks(x,["ID","Long gap","Distractor shift"]);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel("Mean feature MSE (h3/h6)");axes[1].set_ylabel("Cue information retained");axes[1].set_ylim(0,1)
    axes[0].legend(fontsize=8);fig.suptitle("Past-only fixed ridge feasibility probes: irreversible R4 / K2 replay")
    fig.text(.5,.015,"Old 16 pairs train only; 96 already-inspected fresh development pairs. Lambda=1, matched features/capacity; no selection.",ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.96));(output/"figures").mkdir(exist_ok=True)
    for ext in ("png","pdf"):fig.savefig(output/"figures"/("online_probe_comparison."+ext),dpi=170,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();audit_and_export(a.source,a.output)
