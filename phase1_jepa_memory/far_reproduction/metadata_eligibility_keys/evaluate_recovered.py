"""Execute unchanged official evaluation with provenance and budget guards."""
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

RECOVERY = Path(__file__).resolve().parent
ROOT = Path("/scratch/xl4358/phase1_far_reference")
UPSTREAM = ROOT / "upstream"
OUT = RECOVERY / "evaluation" / ("job_" + os.environ["SLURM_JOB_ID"])
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic() - 1580
import faulthandler
faulthandler.dump_traceback_later(300, repeat=True)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def save(name, d):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2) + "\n")
    tmp.replace(p)


def source():
    lock = json.loads((ROOT / "upstream_source_lock.json").read_text())
    for n, h in lock["files"].items():
        if sha(UPSTREAM / n) != h:
            raise RuntimeError("upstream source mismatch: " + n)
    for n, h in json.loads((ROOT / "evaluation_source_lock.json").read_text())["files"].items():
        if sha(ROOT / n) != h:
            raise RuntimeError("evaluation/controller source mismatch: " + n)


def main():
    import torch
    if os.environ.get("SLURM_JOB_ACCOUNT") != "csci_ga_2271-2026fa":
        raise RuntimeError("course allocation required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one actual CUDA GPU required; CPU fallback forbidden")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("actual A100 required")
    if torch.__version__ != "2.7.0+cu128":
        raise RuntimeError("author Torch/CUDA dependency target not active")
    torch.set_num_threads(4)
    source()
    recovery_lock = json.loads((RECOVERY / "source_lock.json").read_text())
    assert all(sha(RECOVERY / n) == h for n,h in recovery_lock["files"].items())
    recovery_binding = json.loads((RECOVERY / "evaluation_binding.json").read_text())
    certificate_path = Path(recovery_binding["certificate_path"])
    assert sha(certificate_path) == recovery_binding["certificate_sha256"]
    cert = json.loads(certificate_path.read_text())
    assert cert["status"] == "CURRENT_IDENTITIES_AND_OFFICIAL_ELIGIBILITY_COMPLETE"
    assert cert["source_lock_sha256"] == sha(RECOVERY / "source_lock.json")
    for entry in cert["files"] + cert["metrics"]:
        st = (UPSTREAM / entry["path"]).stat()
        assert (st.st_size, st.st_mtime_ns) == (entry["bytes"], entry["mtime_ns"])
    binding = json.loads((ROOT / "evaluation_binding.json").read_text())
    if binding["protocol_sha256"] != sha(ROOT / "reference_protocol.json"):
        raise RuntimeError("actual controller protocol binding differs")
    if binding["evaluation_source_sha256"] != sha(ROOT / "evaluation_source_lock.json"):
        raise RuntimeError("actual controller source binding differs")
    protocol = json.loads((ROOT / "reference_protocol.json").read_text())
    original_data_proof = ROOT / "evaluation/job_7403/executed_data_proof.json"
    assert sha(original_data_proof) == cert["original_numeric_proof_sha256"]
    old = json.loads(original_data_proof.read_text())
    assert old["test_records"] == cert["test_records"] == 3840
    assert old["frames"] == cert["frames"] == 2515992
    inputs = old["files"]
    save("executed_data_proof.json", {"test_records": 3840, "frames": cert["frames"],
         "scope": "Original complete finite/axis proof reused after completed7412 CPU full current SHA and current all-file stat boundary; metadata-only eligibility certificate; GPU stat identities verified",
         "certificate_sha256": sha(certificate_path), "original_proof_sha256": sha(original_data_proof),
         "new_GPU_full_tensor_traversal_claimed": False, "files": inputs})
    bundles = json.loads((ROOT / "bundles.json").read_text())["bundles"]
    selected = [e for e in bundles if e.get("corpus") == "loopnav"
                or e["path"] == "oasis_500m_vit_vae.pth"]
    for e in selected:
        q = UPSTREAM / "models" / e["path"]
        if q.stat().st_size != e["size_bytes"] or sha(q) != e["sha256"]:
            raise RuntimeError("actual model/retriever/tokenizer release mismatch")
        if e.get("role") != "asset":
            c = torch.load(q, map_location="cpu", weights_only=False)
            def finite(x):
                if isinstance(x, torch.Tensor):
                    return bool(torch.isfinite(x).all())
                if isinstance(x, dict):
                    return all(finite(v) for v in x.values())
                if isinstance(x, (list, tuple)):
                    return all(finite(v) for v in x)
                return True
            if not finite(c):
                raise RuntimeError("nonfinite released checkpoint")
            if e["role"] != "retriever" and c.get("train_steps") != 700000:
                raise RuntimeError("released evaluation step mismatch")
            del c
    save("executed_model_proof.json", {
        "protocol_sha256": sha(ROOT / "reference_protocol.json"),
        "models": selected, "all_finite": True,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": sys.version, "science_threads": torch.get_num_threads(),
        "inference_started": False})
    spec = importlib.util.spec_from_file_location("far_official_eval", UPSTREAM / "scripts/world/eval_rollout.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    real_rollout = official.rollout_clip
    first_work = None
    completed_steps = 0
    records = 0
    original_print = builtins.print
    scalar_output = OUT / "scalars"
    scalar_output.mkdir(exist_ok=True)
    eligible_path = certificate_path.parent / "eligible_clips.json"
    assert sha(eligible_path) == cert["eligible_clips_sha256"]
    import shutil
    shutil.copyfile(eligible_path, scalar_output / "eligible_clips.json")
    save("launch_status.json", {"status": "ORIGINAL_MODEL_AND_METRIC_INITIALIZATION", "prior_GPU_seconds": 1580,
                               "remaining_soft_seconds": 13800 - (time.monotonic() - START)})

    def rollout(*a, **kw):
        nonlocal first_work
        if first_work is None:
            first_work = time.monotonic()
        save("launch_status.json", {"status": "OFFICIAL_ROLLOUT_ENTERED", "prior_GPU_seconds": 1580,
                                    "elapsed_including_prior_GPU_seconds": time.monotonic() - START})
        return real_rollout(*a, **kw)

    def print_guard(*a, **kw):
        nonlocal completed_steps, records
        original_print(*a, **kw)
        msg = " ".join(str(v) for v in a)
        match = re.match(r"^\[([^\]]+)\] clip\s+\d+ .*?: (\d+) steps\s", msg)
        if not match:
            return
        records += 1
        completed_steps += int(match.group(2))
        elapsed = time.monotonic() - START
        if records == len(protocol["arms"]):
            eligible = json.loads((scalar_output / "eligible_clips.json").read_text())
            if len(eligible["clips"]) != protocol["expected_official_eligible"] or eligible["scanned"] != 3840:
                raise RuntimeError("official eligible list differs; no silent omission")
            steps = sum(int(x[1]) for x in eligible["clips"]) * len(protocol["arms"])
            rate = (time.monotonic() - first_work) / completed_steps
            projection = elapsed + (steps - completed_steps) * rate
            report = {"elapsed_seconds": elapsed, "completed_records": records,
                      "completed_steps": completed_steps, "total_steps": steps,
                      "full_stage_projection_seconds": projection,
                      "soft_budget_seconds": 13800, "passed": projection <= 13800,
                      "forecast_not_completion": True, "all_data_and_arms_preserved": True}
            save("first_complete_record_budget.json", report)
            if projection > 13800:
                raise RuntimeError("FULL_REFERENCE_BUDGET_EXCEEDED; retained first complete record, full replication incomplete")
        save("progress.json", {"status": "EVALUATION_IN_PROGRESS", "records": records,
                               "steps": completed_steps, "elapsed_seconds": elapsed})
        if elapsed >= 13800:
            raise RuntimeError("soft stage budget reached; no additional allocation")

    official.rollout_clip = rollout
    builtins.print = print_guard
    sys.argv = [str(UPSTREAM / "scripts/world/eval_rollout.py"), "--config-name", "eval_rollout_loopnav",
                "output_dir=" + str(scalar_output), "shard=0", "num_shards=1"]
    try:
        official.main()
    finally:
        builtins.print = original_print
    source()
    for entry in cert["files"] + cert["metrics"]:
        st = (UPSTREAM / entry["path"]).stat()
        assert (st.st_size, st.st_mtime_ns) == (entry["bytes"], entry["mtime_ns"])
    expected = protocol["expected_official_eligible"] * len(protocol["arms"])
    if records != expected:
        raise RuntimeError("incomplete official record count")
    save("completion.json", {"status": "ALL_OFFICIAL_ROLLOUT_RECORDS_COMPLETE_FINAL_CPU_SHA_AND_AGGREGATION_PENDING",
                             "records": records, "elapsed_seconds": time.monotonic() - START,
                             "data_or_seed_or_method_reduction": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save("failure.json", {"status": "REFERENCE_EVALUATION_INCOMPLETE", "error": str(e),
                              "type": type(e).__name__, "elapsed_seconds": time.monotonic() - START,
                              "is_Phase1_scientific_verdict": False})
        raise
