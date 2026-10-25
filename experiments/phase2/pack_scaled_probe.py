"""Verify and package the completed scaled scorer pilot without rerunning it."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--logs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    work=args.work.resolve();result=work/"scaled-use-time-probe";bundle=work/"bundle"
    output=(args.output or work/"scaled-use-time-probe-full.zip").resolve()
    receipt=output.with_suffix(".receipt.json")
    if output.exists() or receipt.exists():
        raise ValueError("Refuse overwrite of an existing package or receipt")
    summary=json.loads((result/"summary.json").read_text())
    if summary.get("status")!="completed_scaled_use_time_pilot":
        raise ValueError("Scaled scorer pilot is not completed")
    protocol=json.loads((result/"protocol.json").read_text())
    for name,expected in summary["reader_audit"]["artifact_sha256"].items():
        p=Path(name).resolve()
        if not p.is_relative_to(work) or not p.is_file() or sha(p)!=expected:
            raise ValueError("Frozen reader/input artifact changed: "+name)
    required={"training_inputs.npy","training_labels.npz","training_metadata.json","probe_parameters.npz",
              "training_fit.json","online_results.json","online_trace.jsonl","protocol.json"}
    if not required.issubset(summary["result_sha256"]):
        raise ValueError("Completion result hashes omit required full-data artifacts")
    entries={}
    for name,expected in summary["result_sha256"].items():
        p=(result/name).resolve()
        if not p.is_relative_to(result) or not p.is_file() or sha(p)!=expected:
            raise ValueError("Completed result hash mismatch: "+name)
        entries["results/"+name]=(p,expected)
    entries["results/summary.json"]=(result/"summary.json",sha(result/"summary.json"))
    for name,expected in protocol["source_sha256"].items():
        p=(bundle/name).resolve()
        if not p.is_relative_to(bundle) or not p.is_file() or sha(p)!=expected:
            raise ValueError("Experiment source changed: "+name)
        entries["source/"+name]=(p,expected)
    actual=work/"scaled-probe-actual.log"
    if args.logs:
        logs=[p.resolve() if p.is_absolute() else (work/p).resolve() for p in args.logs]
    else:
        logs=sorted(p.resolve() for p in work.glob("scaled*probe*.log"))
    if not actual.is_file() or actual.resolve() not in logs or len(logs)<2:
        raise ValueError("Need actual and first-failed execution logs; use --logs if their names differ")
    for i,p in enumerate(logs):
        if not p.is_file() or not p.is_relative_to(work):
            raise ValueError("Invalid execution log path: "+str(p))
        entries["execution_logs/"+str(i)+"_"+p.name]=(p,sha(p))
    # Bind the reader/input experiment that supplies this scorer pilot. Large
    # model/feature archives remain at their original locations; training arrays
    # and all scorer labels/parameters are included in this package.
    for name in ("completion.json","protocol.json","readers_locked.json","data/records.json","data/manifest.json"):
        p=work/"run"/name
        if not p.is_file():
            raise ValueError("Missing reader provenance: "+name)
        entries["reader_provenance/"+name]=(p,sha(p))
    if sha(work/"run/completion.json")!=protocol["reader_completion_sha256"]:
        raise ValueError("Reader completion changed")
    for name in ("manifest.json","normalization.npz","basis.npz","model_config.json"):
        p=bundle/"inputs"/name
        if not p.is_file():
            raise ValueError("Missing fixed input: "+name)
        entries["fixed_inputs/"+name]=(p,sha(p))
    # Include the exact frozen scale-512 checkpoint so the archive can support
    # arithmetic/fit analysis without requesting another model upload.
    checkpoint=work/"run/pairs-512/selected.pt"
    reader_completion=json.loads((work/"run/completion.json").read_text())
    if sha(checkpoint)!=reader_completion["files"]["pairs-512/selected.pt"]:
        raise ValueError("Reader checkpoint changed")
    entries["reader_provenance/pairs-512/selected.pt"]=(checkpoint,sha(checkpoint))
    manifest={"status":"verified_full_scaled_probe_package","created_utc":datetime.now(timezone.utc).isoformat(),
        "packer_sha256":sha(__file__),"work":str(work),"new_model_calls":0,"gpu_used":False,
        "files":{name:{"sha256":h,"bytes":p.stat().st_size} for name,(p,h) in entries.items()},
        "source_hashes_unchanged":True,"result_hashes_matched":True,"remote_originals_preserved":True,
        "limits":["Includes full scorer training inputs/labels/parameters and online traces, plus fixed reader/transforms",
                  "Full raw episode RGB and per-frame feature archive remain in the original reader-scale run; complete online reruns require them"]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for name,(p,expected) in sorted(entries.items()):
            archive.write(p,name)
            # Detect accidental edits, including logs still being written,
            # rather than silently packaging a mixed execution state.
            if sha(p)!=expected:
                raise ValueError("File changed while packaging: "+str(p))
        archive.writestr("package_manifest.json",json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    with zipfile.ZipFile(output) as archive:
        bad=archive.testzip()
        if bad is not None:
            raise ValueError("Zip CRC failed: "+bad)
    value={"status":"verified_package_created","path":str(output),"bytes":output.stat().st_size,
        "sha256":sha(output),"entries":len(entries)+1,"uncompressed_bytes":sum(p.stat().st_size for p,h in entries.values()),
        "zip_crc_passed":True,"script_sha256":sha(__file__),"remote_originals_preserved":True}
    receipt.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(json.dumps(value),flush=True)


if __name__=="__main__":
    main()
