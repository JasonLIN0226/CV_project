"""CPU-only export of saved evaluation targets and execution receipts; no model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    work = args.work.resolve()
    run = work / "run"
    completion = json.loads((run/"completion.json").read_text())
    if completion["status"] != "completed":
        raise ValueError("Require completed archived execution")
    checked = {}

    def verify(relative):
        p = (run / relative).resolve()
        if not p.is_relative_to(run) or relative not in completion["files"]:
            raise ValueError("Not bound by completion manifest: "+relative)
        digest = sha(p)
        if digest != completion["files"][relative]:
            raise ValueError("Completion hash mismatch: "+relative)
        checked[relative] = digest
        return p

    for name in ("protocol.json", "data/manifest.json", "data/records.json", "evaluation_rows.json", "evaluation_summary.json", "evaluation_predictions.npz"):
        verify(name)
    records = json.loads((run/"data/records.json").read_text())
    evaluation = [r for r in records if r["split"] == "evaluation"]
    if len(evaluation) != 96 or len({r["id"] for r in evaluation}) != 96:
        raise ValueError("Expected 96 unique evaluation episodes")
    ids = {r["id"] for r in evaluation}
    if any(r["sibling"] not in ids for r in evaluation):
        raise ValueError("Incomplete sibling pairs")
    targets = []
    for rec in evaluation:
        with np.load(verify("data/"+rec["feature_path"]), allow_pickle=False) as values:
            z = values["z"]
            target = z[rec["query"]+np.array([3,6])].copy()
            if target.shape != (2,5120) or target.dtype != np.float32 or not np.isfinite(target).all():
                raise ValueError("Invalid normalized evaluation targets")
            targets.append(target)
    with np.load(run/"evaluation_predictions.npz", allow_pickle=False) as archive:
        if archive.files != ["predictions"] or archive["predictions"].shape != (1920,2,5120):
            raise ValueError("Unexpected archived prediction schema")
    destination = work/"execution-evidence"
    package = work/"execution-evidence.zip"
    if destination.exists() or package.exists():
        raise ValueError("Refuse overwrite; export destination already exists")
    destination.mkdir()
    arrays = {"episode_ids": np.array([r["id"] for r in evaluation]),
              "sibling_ids": np.array([r["sibling"] for r in evaluation]),
              "queries": np.array([r["query"] for r in evaluation]),
              "targets": np.stack(targets)}
    protocol = json.loads((run/"protocol.json").read_text())
    inputs = work/"bundle/inputs"
    if (inputs/"manifest.json").exists():
        if sha(inputs/"manifest.json") != protocol["inputs_manifest_sha256"]:
            raise ValueError("Inputs manifest hash differs")
        im = json.loads((inputs/"manifest.json").read_text())
        expected = im["files"]["normalization.npz"]
        if isinstance(expected, dict):
            expected = expected["sha256"]
        if sha(inputs/"normalization.npz") != expected:
            raise ValueError("Normalization hash differs")
        with np.load(inputs/"normalization.npz", allow_pickle=False) as a:
            arrays["normalization_std"] = a["std"].copy()
        checked["inputs/normalization.npz"] = expected
    np.savez_compressed(destination/"evaluation_targets.npz", **arrays)
    job = str(completion["hardware"]["slurm_job_id"])
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Invalid job ID")
    sacct = shutil.which("sacct") or "/opt/slurm/bin/sacct"
    command = [sacct, "-j", job, "--parsable2", "--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES,NodeList"]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    (destination/("sacct-"+job+".txt")).write_text(proc.stdout)
    (destination/("sacct-"+job+".stderr.txt")).write_text(proc.stderr)
    logs = list(work.glob("slurm-"+job+".log")) + list(run.glob("slurm-"+job+".log"))
    if not logs:
        raise FileNotFoundError("Cannot locate recorded Slurm log in work or run")
    shutil.copy2(logs[0], destination/logs[0].name)
    receipts = sorted(set(work.glob("*receipt*.json")) | set(run.glob("*receipt*.json")))
    for receipt in receipts:
        target = destination / ("source_"+receipt.parent.name+"_"+receipt.name)
        shutil.copy2(receipt, target)
    for name in ("completion.json", "protocol.json", "evaluation_rows.json", "evaluation_summary.json"):
        shutil.copy2(run/name, destination/name)
    write(destination/"evaluation_targets_receipt.json", {"status": "exported_saved_targets_only", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "completion_sha256": sha(run/"completion.json"), "hardware": completion["hardware"],
        "checked_source_sha256": checked, "target_sha256": sha(destination/"evaluation_targets.npz"),
        "episodes": 96, "targets_shape": list(arrays["targets"].shape), "prediction_schema": {"predictions": [1920,2,5120]},
        "normalization_std_exported": "normalization_std" in arrays, "new_model_calls": 0, "gpu_used_by_export": False,
        "sacct_command": command, "sacct_returncode": proc.returncode, "source_receipts": [str(p.relative_to(work)) for p in receipts]})
    manifest = {p.name:sha(p) for p in destination.iterdir() if p.is_file()}
    write(destination/"execution_evidence_manifest.json", {"files": manifest})
    with zipfile.ZipFile(package, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(destination.iterdir()):
            archive.write(p, p.name)
    receipt = {"zip": str(package), "bytes": package.stat().st_size, "sha256": sha(package), "files": len(manifest)+1}
    write(work/"execution_evidence_zip_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
