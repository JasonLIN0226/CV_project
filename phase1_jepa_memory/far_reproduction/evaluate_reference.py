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

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
OUT = ROOT / "evaluation" / ("job_" + os.environ["SLURM_JOB_ID"])
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()


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
    binding = json.loads((ROOT / "evaluation_binding.json").read_text())
    if binding["protocol_sha256"] != sha(ROOT / "reference_protocol.json"):
        raise RuntimeError("actual controller protocol binding differs")
    if binding["evaluation_source_sha256"] != sha(ROOT / "evaluation_source_lock.json"):
        raise RuntimeError("actual controller source binding differs")
    protocol = json.loads((ROOT / "reference_protocol.json").read_text())
    import numpy as np
    index = json.loads((UPSTREAM / "results/indices/loopnav/loopnav_latent_test.json").read_text())
    ids = index["sample_ids"]
    if len(ids) != 3840 or len(set(ids)) != 3840:
        raise RuntimeError("official test split count/identity mismatch")
    inputs, total_frames = [], 0
    for name in ids:
        path = UPSTREAM / name
        lat = np.load(path, mmap_mode="r", allow_pickle=False)
        frames = json.loads(path.with_suffix(".json").read_text())
        if lat.ndim != 4 or tuple(lat.shape[1:]) != (16, 18, 32) or lat.dtype != np.float32:
            raise RuntimeError("actual observation shape/dtype mismatch")
        if len(frames) != lat.shape[0]:
            raise RuntimeError("frame log and observation array are not aligned")
        for i in range(0, len(lat), 128):
            if not np.isfinite(lat[i:i + 128]).all():
                raise RuntimeError("nonfinite actual observation array")
        poses = np.array([[f[k] for k in ("x", "y", "z", "pitch", "yaw")] for f in frames])
        if not np.isfinite(poses).all():
            raise RuntimeError("nonfinite actual camera metadata")
        total_frames += len(lat)
        inputs.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size,
                       "frames": len(lat), "frame_log_sha256": sha(path.with_suffix(".json"))})
        del lat, frames, poses
    save("executed_data_proof.json", {"test_records": len(ids), "frames": total_frames,
                                     "all_finite_and_frame_logs_aligned": True, "files": inputs})
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

    def rollout(*a, **kw):
        nonlocal first_work
        if first_work is None:
            first_work = time.monotonic()
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
    for e in inputs:
        p = UPSTREAM / e["path"]
        if sha(p) != e["sha256"] or sha(p.with_suffix(".json")) != e["frame_log_sha256"]:
            raise RuntimeError("actual observations changed during evaluation")
    expected = protocol["expected_official_eligible"] * len(protocol["arms"])
    if records != expected:
        raise RuntimeError("incomplete official record count")
    save("completion.json", {"status": "ALL_OFFICIAL_ROLLOUT_RECORDS_COMPLETE_AGGREGATION_PENDING",
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
