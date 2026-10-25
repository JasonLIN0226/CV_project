"""Independent saved-data audit of scaled scorer pilot; no fitting or inference."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/"experiments/predictor/control/src"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:24]


def cue_catalog(records,data_manifest,old_root):
    """Regenerate scene timing, not images/features; no salience inference."""
    from phase1 import envs
    from experiments.phase2.encode_fresh_gate_diagnostic import DATA_CONFIG
    from experiments.phase2.portable_snapshot import PortableSnapshot
    for name in ("envs.py","utils.py"):
        if sha(ROOT/"experiments/predictor/control/src/phase1"/name)!=data_manifest["source_sha256"][name]:
            raise ValueError("Generator source changed: "+name)
    old=PortableSnapshot(old_root)
    catalog={}
    for r in records:
        if r["legacy_training"]:
            original=old.by_id[r["id"]]
            scene=envs.scene(old.config,"gate","train",original["scene_index"])
            if any(original[k]!=r[k] for k in ("id","pair","sibling","seed","query","frames")):
                raise ValueError("Legacy episode identity differs")
            if int(old.evidence(original)[0])!=scene["start"]:
                raise ValueError("Legacy cue annotation differs from generator")
            z=old.features(original)
            expected=list(range(scene["start"],scene["end"]+1))
            same=[t for t in range(r["query"]-3) if np.array_equal(z[t],z[scene["start"]])]
            if same!=expected:
                raise ValueError("Legacy feature-equivalent cue times differ from generator")
        else:
            if r["split"]=="train":
                index=r["train_index"]-16
            else:
                limit=32 if r["split"]=="validation" else 16
                candidates=[i for i in range(limit) if identity(["reader-scale-pilot-v1",r["data_seed"],r["split"],r["condition"],i])==r["pair"]]
                if len(candidates)!=1:raise ValueError("No exact generated pair identity")
                index=candidates[0]
            pair=identity(["reader-scale-pilot-v1",r["data_seed"],r["split"],r["condition"],index])
            episode_ids={identity([pair,j]) for j in (0,1)}
            if pair!=r["pair"] or {r["id"],r["sibling"]}!=episode_ids:
                raise ValueError("Generated episode IDs differ")
            split=r["split"] if r["split"] in ("train","validation") else {"id":"test_id","long_gap":"test_long_gap","distractor_shift":"test_distractor_shift"}[r["condition"]]
            scene=envs.scene({"data":dict(DATA_CONFIG,seed=r["data_seed"])},"gate",split,index)
            rgb=r["rgb_indices"]
            cue_indices=rgb[scene["start"]:scene["end"]+1]
            if len(set(cue_indices))!=1 or cue_indices[0]==rgb[0]:
                raise ValueError("Saved RGB bank indices contradict regenerated cue timing")
        if scene["seed"]!=r["seed"] or scene["query"]!=r["query"] or scene["length"]!=r["frames"]:
            raise ValueError("Regenerated seed/query/length mismatch")
        catalog[r["id"]]=list(range(scene["start"],scene["end"]+1))
    return catalog


def bootstrap(rows,key):
    by_pair={}
    for r in rows:by_pair.setdefault(r["pair"],[]).append(float(r[key]))
    values=np.array([np.mean(v) for v in by_pair.values()])
    rng=np.random.default_rng(20261024)
    draws=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return {"mean":float(values.mean()),"low":float(np.quantile(draws,.025)),"high":float(np.quantile(draws,.975)),
            "scene_pairs":len(values),"episodes":len(rows)}


def run(args):
    package=args.package.resolve();out=args.output.resolve()
    if out.exists():raise ValueError("Use a new verification output path")
    manifest=read(package/"package_manifest.json")
    for name,info in manifest["files"].items():
        p=(package/name).resolve()
        if not p.is_relative_to(package) or p.stat().st_size!=info["bytes"] or sha(p)!=info["sha256"]:
            raise ValueError("Full package member mismatch: "+name)
    result=package/"results";summary=read(result/"summary.json");protocol=read(result/"protocol.json")
    if summary["status"]!="completed_scaled_use_time_pilot":raise ValueError("Incomplete pilot")
    for name,h in summary["result_sha256"].items():
        if sha(result/name)!=h:raise ValueError("Completed result digest mismatch: "+name)
    for name,h in protocol["source_sha256"].items():
        if sha(package/"source"/name)!=h:raise ValueError("Pinned source mismatch: "+name)
    records=read(package/"reader_provenance/data/records.json");by_id={r["id"]:r for r in records}
    data_manifest=read(package/"reader_provenance/data/manifest.json")
    cues=cue_catalog(records,data_manifest,args.old_snapshot)
    rows=read(result/"online_results.json");meta=read(result/"training_metadata.json")
    arms=tuple(protocol["arms"]);all_arms=arms+tuple(protocol["controls"])
    if len(rows)!=96*len(all_arms) or len({(r["episode_id"],r["arm"]) for r in rows})!=len(rows):
        raise ValueError("Online evaluation population mismatch")
    parameters=np.load(result/"probe_parameters.npz",allow_pickle=False)
    labels=np.load(result/"training_labels.npz",allow_pickle=False)
    x=np.load(result/"training_inputs.npy",mmap_mode="r",allow_pickle=False)
    if x.shape!=(len(meta),3,903) or x.dtype!=np.float32 or len(meta)!=summary["training_parents"]:
        raise ValueError("Training input/metadata shape mismatch")
    for start in range(0,len(x),1024):
        if not np.isfinite(x[start:start+1024]).all():raise ValueError("Nonfinite training inputs")
    for name in labels.files:
        if labels[name].shape!=(len(meta),3) or not np.isfinite(labels[name]).all():raise ValueError("Invalid labels")
    parameter_bytes={}
    for arm in arms:
        scale,weight,ls=[parameters[arm+"_"+k] for k in ("scale","weights","label_scale")]
        if scale.shape!=(903,) or weight.shape!=(903,) or not np.isfinite(scale).all() or not np.isfinite(weight).all() or not np.isfinite(ls).all():
            raise ValueError("Invalid probe parameters")
        parameter_bytes[arm]=scale.nbytes+weight.nbytes+8
        if not np.array_equal(scale,parameters[arms[0]+"_scale"]):raise ValueError("Arm feature scalings differ")
        if np.any(weight[scale<=1e-6]!=0):raise ValueError("TRAIN support mask violated")
    lookup={(r["episode_id"],r["arm"]):r for r in rows}
    for row in rows:
        rec=by_id[row["episode_id"]]
        if rec["split"]!="evaluation" or row["pair"]!=rec["pair"] or row["query"]!=rec["query"]:
            raise ValueError("Online row identity/query mismatch")
        if row["cue_times"]!=cues[rec["id"]] or row["cue_retained"]!=float(bool(set(row["kept_times"])&set(cues[rec["id"]]))):
            raise ValueError("Cue survival differs from regenerated scene evidence")
        if row["probe_parameter_bytes"]!=parameter_bytes.get(row["arm"],0):raise ValueError("Parameter byte mismatch")
        if not np.isclose(row["mse"],np.mean(row["horizon_losses"]),rtol=1e-12,atol=1e-12):raise ValueError("Horizon aggregation mismatch")
        if not np.isclose(row["mse"],row["contrast_mse"]+row["orthogonal_mse"],rtol=1e-12,atol=1e-12):raise ValueError("Metric decomposition mismatch")
        if row["feature_branch_correct"]!=float(np.mean(np.asarray(row["contrast_amplitude"])>0)):raise ValueError("Feature-direction metric mismatch")
        for control in all_arms:
            if not np.isclose(row["minus_"+control],row["mse"]-lookup[(rec["id"],control)]["mse"],rtol=1e-12,atol=1e-12):
                raise ValueError("Paired policy difference mismatch")
    max_ci_error=0.;ci_fields=0;online={}
    for arm,conditions in summary["statistics"].items():
        online[arm]={}
        for condition,statistics in conditions.items():
            selected=[r for r in rows if r["arm"]==arm and (condition=="all" or r["condition"]==condition)]
            online[arm][condition]={"episodes":len(selected),"cue_retained":sum(int(r["cue_retained"]) for r in selected),"mse":float(np.mean([r["mse"] for r in selected]))}
            for key,saved in statistics.items():
                calculated=bootstrap(selected,key)
                for field,v in calculated.items():
                    max_ci_error=max(max_ci_error,abs(v-saved[field]));ci_fields+=1
                    if not np.isclose(v,saved[field],rtol=1e-12,atol=1e-12):raise ValueError("Pair bootstrap mismatch")
    timing=defaultdict(lambda:{"parents":0,"cue_risk_parents":0,"labels":{a:{"minimum_can_destroy_cue":0,"every_minimum_destroys_cue":0,"argmin_destroys_cue":0,"tied_minima":0} for a in arms}})
    for i,m in enumerate(meta):
        rec=by_id[m["episode_id"]];t=int(m["time"]);times=m["candidate_times"]
        if rec["split"]!="train" or m["split"]!="train" or m["pair"]!=rec["pair"] or m["query"]!=rec["query"] or len(set(times))!=3 or max(times)>=t-3:
            raise ValueError("Training parent identity/causality mismatch")
        regime="pre_use" if t+6<=rec["query"] else "post_use" if t+3>rec["query"] else "straddle"
        group=timing[regime];group["parents"]+=1
        cue=set(cues[rec["id"]]);destroy={j for j,k in enumerate(times) if k in cue}
        risk=len(destroy)==1 and not set(range(t-3,t+1))&cue
        if not risk:continue
        group["cue_risk_parents"]+=1
        for arm in arms:
            values=labels[arm][i];minimum=set(np.flatnonzero(values<=values.min()+1e-8).tolist());report=group["labels"][arm]
            report["minimum_can_destroy_cue"]+=int(bool(minimum&destroy));report["every_minimum_destroys_cue"]+=int(minimum<=destroy)
            report["argmin_destroys_cue"]+=int(int(values.argmin()) in destroy);report["tied_minima"]+=int(len(minimum)>1)
    traces=defaultdict(list)
    with (result/"online_trace.jsonl").open() as f:
        for line in f:
            event=json.loads(line);traces[(event["episode_id"],event["arm"])].append(event)
    if set(traces)!=set(lookup):raise ValueError("Trace/result coverage differs")
    for key,events in traces.items():
        deleted=set();q=lookup[key]["query"]
        if [e["time"] for e in events]!=list(range(q+1)):raise ValueError("Non-contiguous trace")
        for event in events:
            t=event["time"];recent=event["recent_timestamps"];kept=event["retained_timestamps"]
            if recent!=list(range(max(0,t-3),t+1)) or len(kept)>2 or set(kept)&deleted or set(kept)&set(recent):raise ValueError("Bounded/irreversible trace contract failed")
            if event["deleted_timestamp"] is not None:deleted.add(event["deleted_timestamp"])
        if events[-1]["retained_timestamps"]!=lookup[key]["kept_times"]:raise ValueError("Final trace retention differs")
    report={"status":"passed_saved_scaled_probe_audit","created_utc":datetime.now(timezone.utc).isoformat(),
        "verifier_sha256":sha(__file__),"package_manifest_sha256":sha(package/"package_manifest.json"),
        "files_verified":len(manifest["files"]),"training_parents":len(meta),"online_rows":len(rows),
        "bootstrap_fields_verified":ci_fields,"max_bootstrap_absolute_error":max_ci_error,
        "parameter_bytes":parameter_bytes,"online":online,"training_label_cue_audit":dict(timing),
        "cue_definition":"Deterministically regenerated simulator cue interval; seed/query/length/pair/id/RGB indices guarded; legacy cue features checked against verified archive",
        "cue_regeneration_episodes":len(cues),"irreversible_traces_verified":len(traces),
        "new_model_calls":0,"new_fitting":False,"gpu_used":False,
        "limits":["Recomputes saved online summary arithmetic and cue/trace semantics; does not rerun reader predictions or encoder",
                  "Cue-label audit is evaluator-only and did not enter scorer training",
                  "Already inspected evaluation scenes; pilot, not independent confirmation"]}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("online",)},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--old-snapshot",type=Path,default=ROOT/"build/hpc-20261020/development-assets")
    run(parser.parse_args())
