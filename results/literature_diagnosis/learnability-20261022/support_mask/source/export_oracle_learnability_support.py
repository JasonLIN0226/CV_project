"""Independently audit and publish the one fixed support-mask ablation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/"experiments/predictor/control/src"):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
import numpy as np
from experiments.phase2.diagnose_oracle_learnability import ARMS,RidgeProbe,causal_features,parent_from_arrays
from experiments.phase2.export_oracle_learnability import independent_pair_means,quantiles,sanitize
from experiments.phase2.generate_deletion_pilot import sha256,write
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset


def audit_export(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Use an empty output")
    summary=json.loads((source/"summary.json").read_text());protocol=json.loads((source/"protocol.json").read_text())
    assert summary["protocol_sha256"]==sha256(source/"protocol.json")
    for name,digest in summary["artifact_sha256"].items():assert sha256(source/name)==digest,name
    origin=ROOT/protocol["source_run"]
    assert sha256(origin/"summary.json")==protocol["source_summary_sha256"]
    assert sha256(origin/"probe_parameters.npz")==protocol["source_parameters_sha256"]
    assert sha256(ROOT/"experiments/phase2/diagnose_oracle_learnability_support.py")==protocol["diagnostic_source_sha256"]
    with np.load(origin/"probe_parameters.npz") as p:
        original={name:RidgeProbe(p[name+"_scale"].copy(),p[name+"_weights"].copy(),float(p[name+"_label_scale"]),name) for name in ARMS}
    mask=original["cost_ridge"].scale<=1e-6
    assert list(np.flatnonzero(mask))==protocol["mask_columns"]
    probes={name:RidgeProbe(p.scale,np.where(mask,0.,p.weights),p.label_scale,name) for name,p in original.items()}
    with np.load(ROOT/"build/history-diagnosis-20261020/followup-blas1/gate/unit_columns/basis.npz") as b:
        basis={k:b[k].copy() for k in ("input_mean","projection")}
    fresh=FreshGateDataset(ROOT/"build/literature-baselines-20261022/fresh-data")
    with np.load(ROOT/"build/hpc-20261020/development-assets/data/features/normalization.npz") as n:
        fresh.verify_old_normalization({k:n[k] for k in ("mean","std")})
    common=json.loads((source/"common_parent_results.json").read_text())
    online=json.loads((source/"online_results.json").read_text());stats=json.loads((source/"statistics.json").read_text())
    decomposition=json.loads((source/"score_decomposition.json").read_text())
    cache=defaultdict(dict)
    for row in json.loads((ROOT/"build/literature-baselines-20261022/fresh-eval/oracle_subsets.json").read_text()):
        cache[row["episode_id"]][tuple(row["kept_times"])]=float(np.mean(row["feature_mse"]))
    for index,row in enumerate(common):
        rec=fresh.by_id[row["episode_id"]];z,ac=fresh.episode(rec);times=row["candidate_times"];q=rec["query"]
        c,r,p=parent_from_arrays(rec,z,ac,q,times);x=causal_features(c,r,q,basis,p)
        losses=[cache[rec["id"]][tuple(k for j,k in enumerate(times) if j!=d)] for d in range(3)]
        np.testing.assert_array_equal(losses,row["losses"])
        for j,(name,probe) in enumerate(probes.items()):
            score=probe.predict(x);oldscore=original[name].predict(x);choice=int(np.argmin(score))
            assert choice==row[name+"_choice"] and losses[choice]==row[name+"_mse"]
            dec=decomposition[index*3+j]
            assert dec["original_score_range"]==float(np.ptp(oldscore))
            assert dec["supported_score_range"]==float(np.ptp(score))
            assert dec["floor_column_score_range"]==float(np.ptp(oldscore-score))
            assert dec["choice_changed"]==(choice!=int(np.argmin(oldscore)))
    for row in online:assert row["mse"]==cache[row["episode_id"]][tuple(row["kept_times"])]
    max_ci_difference=0.
    for rows,groups in [(common,stats["common_parent"])]+[([r for r in online if r["arm"]==name],stats["online"][name]) for name in ARMS]:
        for condition,fields in groups.items():
            for field,expected in fields.items():
                values=independent_pair_means(rows,field,condition);rng=np.random.default_rng(20261022)
                boot=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
                actual=[values.mean(),np.quantile(boot,.025),np.quantile(boot,.975)]
                delta=max(abs(a-expected[k]) for a,k in zip(actual,("mean","low","high")))
                assert delta<1e-12;max_ci_difference=max(max_ci_difference,float(delta))
    deleted=defaultdict(set);last={};count=0;choices=0
    for line in (source/"online_trace.jsonl").read_text().splitlines():
        row=json.loads(line);key=(row["episode_id"],row["arm"])
        assert row["time"]==last.get(key,-1)+1;last[key]=row["time"]
        resident=set(row["retained_timestamps"]+row["recent_timestamps"])
        assert not resident.intersection(deleted[key])
        assert len(row["retained_timestamps"])<=2 and len(row["recent_timestamps"])<=4
        if row["deleted_timestamp"] is not None:
            rec=fresh.by_id[row["episode_id"]];z,ac=fresh.episode(rec)
            times=sorted(row["retained_timestamps"]+[row["deleted_timestamp"]]);t=row["time"]
            c,r,p=parent_from_arrays(rec,z,ac,t,times);x=causal_features(c,r,t,basis,p)
            assert times[int(np.argmin(probes[row["arm"]].predict(x)))]==row["deleted_timestamp"]
            deleted[key].add(row["deleted_timestamp"]);choices+=1
        count+=1
    training={}
    with np.load(origin/"training_inputs_labels.npz") as t:
        x=t["inputs"].copy();cost=t["cost"].copy()
    for name,probe in probes.items():
        pred=np.array([probe.predict(a) for a in x]);selected=cost[np.arange(len(cost)),pred.argmin(1)]
        training[name]={"optimal_fraction":float(np.mean(selected<=cost.min(1)+1e-8)),"mean_regret":float(np.mean(selected-cost.min(1)))}
    ranges={}
    for name in ARMS:
        rows=[r for r in decomposition if r["arm"]==name]
        ranges[name]={"choice_changed_fraction":float(np.mean([r["choice_changed"] for r in rows])),
            **{key:quantiles([r[key] for r in rows]) for key in ("original_score_range","supported_score_range","floor_column_score_range")}}
    audit={"common_parent_choices_and_costs_exact":len(common)*3,"online_query_losses_exact":len(online),
           "irreversible_trace_events_checked":count,"all_online_deletion_choices_recomputed":choices,
           "max_pair_CI_recomputation_difference":max_ci_difference,"mask_columns":int(mask.sum()),
           "training_fit_after_fixed_mask":training,"common_parent_score_decomposition":ranges,
           "world_model_calls":0,"model_fits":0,"GPU_seconds":0}
    output.mkdir(parents=True)
    names=("protocol.json","summary.json","statistics.json","common_parent_results.json","online_results.json","score_decomposition.json")
    for name in names:write(output/name,sanitize(json.loads((source/name).read_text())))
    write(output/"audit.json",audit)
    (output/"source").mkdir()
    for path in (Path(__file__),ROOT/"experiments/phase2/diagnose_oracle_learnability_support.py"):
        shutil.copy2(path,output/"source"/path.name)
    render(output,origin)
    (output/"README.md").write_text("""# Post-hoc train-support numerical ablation

Preserves the preceding unmasked run. No refitting: each arm's 903-coefficient
vector has coefficients zeroed where its old-TRAIN parent-centered feature RMS
is at the fixed 1e-6 numerical floor (710 columns). The mask, architecture,
causal observations, normalization and byte allocation match across all arms.
This is one exploratory numerical diagnosis after seeing unstable input norms,
not a selected method or an untouched confirmation experiment.

The protocol amendment was saved before masked choices and losses were read.
`audit.json` independently recomputes all common-parent and online deletion
choices, cached losses, irreversible memory invariants and pair confidence
intervals. The unmasked coefficients remain in the parent directory; this mask
and those coefficients exactly reproduce the masked parameters. No extra
training, checkpoint changes or GPU execution were involved.

Cost vs attention changes sign after masking, but cost does not reliably beat
the shuffled-label null or strong existing controls. Neither variant establishes
oracle learnability or unlearnability. See the joint diagnosis document.
""")
    write(output/"manifest.json",{"raw_source_directory":str(source.relative_to(ROOT)),
        "raw_sha256":{p.name:sha256(p) for p in source.iterdir() if p.is_file()},
        "published_files":{str(p.relative_to(output)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps(audit,indent=2))


def render(output,origin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    old=json.loads((origin/"statistics.json").read_text())["online"]
    new=json.loads((output/"statistics.json").read_text())["online"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));x=np.arange(3)
    labels=("Cost ridge","Attention ridge","Shuffled cost")
    for data,offset,color,label in ((old,-.18,"#a56d59","Original"),(new,.18,"#347d91","Train-support mask")):
        means=[data[a]["all"]["mse"]["mean"] for a in ARMS]
        lo=[m-data[a]["all"]["mse"]["low"] for a,m in zip(ARMS,means)]
        hi=[data[a]["all"]["mse"]["high"]-m for a,m in zip(ARMS,means)]
        axes[0].bar(x+offset,means,.34,yerr=[lo,hi],capsize=3,color=color,label=label)
        axes[1].bar(x+offset,[data[a]["distractor_shift"]["cue_retained"]["mean"] for a in ARMS],.34,color=color,label=label)
    for ax in axes:
        ax.set_xticks(x,labels);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel("Mean feature MSE (all 96 pairs)");axes[0].legend(fontsize=9)
    axes[1].set_ylabel("Cue survival (32 distractor-shift pairs)");axes[1].set_ylim(0,1.05)
    fig.suptitle("Fixed train-support mask changes the probe ranking")
    fig.text(.5,.01,"No refit or fresh-set tuning. Masked cost still does not reliably beat shuffled labels. Error bars: pair-bootstrap 95% CI.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.96));(output/"figures").mkdir()
    for ext in ("png","pdf"):fig.savefig(output/"figures"/("support_mask_comparison."+ext),dpi=170,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();audit_export(a.source,a.output)
