"""Recompute saved reader-scale metrics using NumPy only; no model calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse overwrite of arithmetic verification")
    completion=json.loads((args.run/"completion.json").read_text())
    files=("evaluation_predictions.npz","evaluation_rows.json","evaluation_summary.json","data/records.json")
    for name in files:
        if sha(args.run/name)!=completion["files"][name]:
            raise ValueError("Saved execution hash mismatch: "+name)
    manifest=json.loads((args.evidence/"execution_evidence_manifest.json").read_text())["files"]
    for name,expected in manifest.items():
        if sha(args.evidence/name)!=expected:
            raise ValueError("Export hash mismatch: "+name)
    receipt=json.loads((args.evidence/"evaluation_targets_receipt.json").read_text())
    if receipt["completion_sha256"]!=sha(args.run/"completion.json"):
        raise ValueError("Target export bound to a different execution")
    if sha(args.evidence/"evaluation_targets.npz")!=receipt["target_sha256"]:
        raise ValueError("Target digest mismatch")
    records={r["id"]:r for r in json.loads((args.run/"data/records.json").read_text()) if r["split"]=="evaluation"}
    with np.load(args.evidence/"evaluation_targets.npz",allow_pickle=False) as a:
        ids=a["episode_ids"].tolist();siblings=a["sibling_ids"].tolist();queries=a["queries"]
        targets=a["targets"].copy();std=a["normalization_std"].copy() if "normalization_std" in a.files else None
    if len(ids)!=96 or set(ids)!=set(records) or targets.shape!=(96,2,5120) or not np.isfinite(targets).all():
        raise ValueError("Target population/schema mismatch")
    for i,eid in enumerate(ids):
        if records[eid]["sibling"]!=siblings[i] or records[eid]["query"]!=queries[i]:
            raise ValueError("Target identity/timing mismatch")
    indices={eid:i for i,eid in enumerate(ids)}
    rows=json.loads((args.run/"evaluation_rows.json").read_text())
    with np.load(args.run/"evaluation_predictions.npz",allow_pickle=False) as a:
        if a.files!=["predictions"]:
            raise ValueError("Unexpected prediction fields")
        predictions=a["predictions"].copy()
    if len(rows)!=1920 or predictions.shape!=(1920,2,5120) or not np.isfinite(predictions).all():
        raise ValueError("Prediction population/schema mismatch")
    if len({(r["reader"],r["episode_id"],r["control"]) for r in rows})!=1920:
        raise ValueError("Duplicate saved evaluation identity")
    maxima={k:0. for k in ("mse","contrast_mse","orthogonal_mse","contrast_amplitude","feature_branch_correct")}
    if std is not None:
        maxima["raw_feature_mse"]=0.
    reconstructed=[]
    for row,pred32 in zip(rows,predictions):
        i=indices[row["episode_id"]]
        y32=targets[i];other32=targets[indices[siblings[i]]]
        pred,y,other=[v.astype(np.float64) for v in (pred32,y32,other32)]
        half=(y-other)/2;mid=(y+other)/2;energy=(half*half).sum(-1)
        if np.any(energy<=1e-20):
            raise ValueError("Degenerate sibling contrast")
        alpha=((pred-mid)*half).sum(-1)/energy
        metrics={"mse":float(np.mean((pred-y)**2)),
                 "contrast_mse":float(np.mean((alpha-1)**2*energy/y.shape[-1])),
                 "orthogonal_mse":float(np.mean((pred-mid-alpha[:,None]*half)**2)),
                 "contrast_amplitude":alpha.tolist(),"feature_branch_correct":float(np.mean(alpha>0))}
        if std is not None:
            # Preserve the archived float32 arithmetic for this specific metric.
            metrics["raw_feature_mse"]=float(np.mean(((pred32-y32)*std)**2))
        for key,value in metrics.items():
            error=float(np.max(np.abs(np.asarray(value)-np.asarray(row[key]))))
            maxima[key]=max(maxima[key],error)
            if not np.allclose(value,row[key],rtol=1e-10,atol=1e-12):
                raise ValueError("Metric mismatch: "+str((row["reader"],row["episode_id"],row["control"],key,error)))
        reconstructed.append({**row,**metrics})
    summary=json.loads((args.run/"evaluation_summary.json").read_text())
    aggregate_max=0.;aggregate_count=0
    for reader,controls in summary["arms"].items():
        for control,metrics in controls.items():
            selected=[r for r in reconstructed if r["reader"]==reader and r["control"]==control]
            if len(selected)!=96:
                raise ValueError("Unbalanced reader/control population")
            for key,saved in metrics.items():
                if key=="raw_feature_mse" and std is None:
                    continue
                value=float(np.mean([r[key] for r in selected]));aggregate_max=max(aggregate_max,abs(value-saved));aggregate_count+=1
                if not np.isclose(value,saved,rtol=1e-10,atol=1e-12):
                    raise ValueError("Aggregate mismatch: "+str((reader,control,key)))
    result={"status":"passed_saved_prediction_arithmetic", "created_utc":datetime.now(timezone.utc).isoformat(),
        "script_sha256":sha(__file__), "completion_sha256":sha(args.run/"completion.json"),
        "prediction_sha256":sha(args.run/"evaluation_predictions.npz"), "targets_sha256":receipt["target_sha256"],
        "rows":1920,"episodes":96,"scene_pairs":48,"max_absolute_metric_errors":maxima,
        "aggregate_values_verified":aggregate_count,"aggregate_max_absolute_error":aggregate_max,
        "raw_feature_mse_verified":std is not None,"new_model_calls":0,"gpu_used":False,
        "limits":["Verifies arithmetic and artifact binding; does not independently rerun encoder/model or validate task semantics",
                  "Feature branch correctness is sibling-contrast sign, not physical lane decoding"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
