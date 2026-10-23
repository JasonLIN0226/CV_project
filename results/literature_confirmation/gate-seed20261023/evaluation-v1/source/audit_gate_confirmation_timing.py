"""Append independent preregistration/submission/encoding chronology evidence."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase2.generate_deletion_pilot import sha256, write


def run(package):
    package = Path(package).resolve()
    protocol = json.loads((package/"protocol.json").read_text())
    run_receipt = json.loads((package/"run_receipt.json").read_text())
    manifest = json.loads((package/"data_manifest.json").read_text())
    source = ROOT/run_receipt["confirmation_directory"]/"execution_source/submission_receipt.json"
    receipt = json.loads(source.read_text())
    locked, submitted, started = (protocol["locked_utc"], receipt["created_utc"], manifest["started_utc"])
    assert datetime.fromisoformat(locked) < datetime.fromisoformat(submitted) < datetime.fromisoformat(started)
    assert receipt["encoder_sha256"] == protocol["encoder_sha256"] == manifest["helper_sha256"]
    assert receipt["job_id"] == manifest["hardware"]["slurm_job_id"]
    assert receipt["returncode"] == 0 and manifest["status"] == "completed"
    timing = {"locked_utc": locked, "submission_created_utc": submitted, "encoding_started_utc": started,
              "encoding_ended_utc": manifest["ended_utc"], "strict_order_passed": True,
              "checks": "locked_protocol.locked_utc < submission_receipt.created_utc < manifest.started_utc",
              "job_id": receipt["job_id"], "actual_gpu_uuid": manifest["hardware"]["actual_gpu_uuid"],
              "source_sha256": {"protocol.json": sha256(package/"protocol.json"),
                                "submission_receipt.json": sha256(source),
                                "data_manifest.json": sha256(package/"data_manifest.json")},
              "scope": "Post-export independent timestamp/provenance check; no model execution or change to locked analysis."}
    write(package/"temporal_audit.json", timing)
    shutil.copy2(source, package/"submission_receipt.json")
    audit = json.loads((package/"audit.json").read_text())
    audit["temporal_order_audit"] = timing
    audit["independent_reader_inference_rerun"] = False
    write(package/"audit.json", audit)
    shutil.copy2(Path(__file__), package/"source"/Path(__file__).name)
    readme = package/"README.md"
    addition = "\n`temporal_audit.json` and the original `submission_receipt.json` additionally verify\nthat protocol lock preceded submission, which preceded encoding. This post-export\naudit does not independently rerun the reader; its arithmetic/trace scope is\nexplicit in `audit.json`. Submission-time verification flags remain historical.\n"
    if addition not in readme.read_text():
        readme.write_text(readme.read_text()+addition)
    published = json.loads((package/"manifest.json").read_text())
    published["post_export_chronology_audit_source_sha256"] = sha256(Path(__file__))
    published["published_files"] = {
        str(p.relative_to(package)): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(package.rglob('*')) if p.is_file() and p != package/"manifest.json"}
    write(package/"manifest.json", published)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    run(parser.parse_args().package)
