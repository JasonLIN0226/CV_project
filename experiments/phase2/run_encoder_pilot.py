"""Bounded CUDA-only PushT/I-JEPA encoding pilot inside an existing A100 job.

This runner never submits, extends, or requeues a Slurm job. The child runs the
existing PushT CLI with an audited real FrozenIJepa constructor; no weights are
trained and no baseline planning claim is made.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments/predictor/control/src"
REFERENCE = ROOT / "results/reference/gate/resolved_config.json"
PINNED_REPOSITORY = "facebook/ijepa_vith14_1k"
PINNED_REVISION = "f157467ea509bc356ff9f61fd3c0d840eec5e04e"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def smi_rows(query, fields):
    command = ["nvidia-smi", f"--query-{query}=" + ",".join(fields), "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    return [{key: value.strip() for key, value in zip(fields, row)}
            for row in csv.reader(result.stdout.splitlines()) if len(row) == len(fields)]


def uuid_key(value):
    text = str(value).strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def verify_hardware(output):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Require exactly one visible CUDA device from the existing single-GPU allocation")
    torch.cuda.set_device(0)
    marker = torch.zeros(1, device="cuda:0")  # materialize this process's CUDA context
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    inventory = smi_rows("gpu", ["index", "name", "uuid", "memory.total", "driver_version"])
    used_uuid = getattr(properties, "uuid", None)
    uuid_source = "torch.cuda.get_device_properties(0).uuid"
    if used_uuid is None:
        # Never guess from machine GPU order. Resolve THIS process's CUDA context.
        rows = smi_rows("compute-apps", ["pid", "gpu_uuid"])
        matches = {r["gpu_uuid"] for r in rows if r["pid"] == str(os.getpid())}
        if len(matches) != 1:
            raise RuntimeError("Torch UUID unavailable and nvidia-smi cannot uniquely map this CUDA process")
        used_uuid = matches.pop()
        uuid_source = "nvidia-smi compute-apps mapping for current worker PID"
    matching = [r for r in inventory if uuid_key(r["uuid"]) == uuid_key(used_uuid)]
    if len(matching) != 1:
        raise RuntimeError("Actual Torch CUDA device UUID does not uniquely match nvidia-smi inventory")
    gpu = matching[0]
    if "A100" not in properties.name.upper() or "A100" not in gpu["name"].upper():
        raise RuntimeError("This pilot explicitly requires an A100; refusing another device")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SLURM_JOB_ID is absent; execute inside the existing course allocation")
    record = {
        "host": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "worker_pid": os.getpid(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_device": "cuda:0", "visible_cuda_devices": torch.cuda.device_count(),
        "torch_device_name": properties.name, "torch_total_memory_bytes": properties.total_memory,
        "actual_device_uuid": gpu["uuid"], "uuid_source": uuid_source,
        "matched_nvidia_smi_device": gpu,
    }
    write_json(output/"hardware.json", record)
    del marker
    return record


def worker(args):
    import numpy as np
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SRC))
    import phase1.encoders as encoders
    from experiments.phase2 import pusht

    output = args.output.resolve()
    hardware = verify_hardware(output)
    print(json.dumps({"hardware_verified": hardware}, indent=2), flush=True)
    original = encoders.FrozenIJepa
    instances = []

    class AuditedFrozenIJepa(original):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            parameters = list(self.model.parameters())
            if any(module.training for module in self.model.modules()):
                raise RuntimeError("Frozen encoder contains a module in training mode")
            if any(p.requires_grad for p in parameters):
                raise RuntimeError("Frozen encoder still has trainable parameters")
            if any(p.device.type != "cuda" or p.device.index != 0 for p in parameters):
                raise RuntimeError("Encoder parameters are not on the verified CUDA device")
            if any(str(p.dtype) != "torch.float32" for p in parameters):
                raise RuntimeError("Encoder dtype differs from the declared float32 pilot")
            self._pilot_versions = tuple(p._version for p in parameters)
            instances.append(self)
            write_json(output/"frozen_encoder_audit.json", {
                "all_modules_eval": True, "trainable_parameters": 0,
                "parameter_count": sum(p.numel() for p in parameters),
                "parameter_bytes": sum(p.numel()*p.element_size() for p in parameters),
                "parameter_dtype": "float32", "device": "cuda:0",
                "device_uuid": hardware["actual_device_uuid"], "encoder_metadata": self.metadata,
            })

    # The CLI imports this symbol at call time. It still uses the original loader
    # and encoder forward; the subclass only records/asserts its actual state.
    encoders.FrozenIJepa = AuditedFrozenIJepa
    sys.argv = [str(ROOT/"experiments/phase2/pusht.py"), "--output", str(output/"pusht"),
                "--seed", str(args.seed), "--steps", "24", "--encoder-config", str(output/"encoder_config.json"),
                "--device", "cuda:0", "--batch-size", "2"]
    pusht.main()
    if len(instances) != 1:
        raise RuntimeError("Expected exactly one real frozen I-JEPA instance")
    parameters = list(instances[0].model.parameters())
    if tuple(p._version for p in parameters) != instances[0]._pilot_versions:
        raise RuntimeError("Encoder parameter version changed during inference")
    with np.load(output/"pusht/encoded_trace.npz", allow_pickle=False) as f:
        observed, control = f["observed"], f["control"]
        if observed.shape != (25, 5120) or control.shape != (25, 5120):
            raise RuntimeError(f"Unexpected real encoded trace shape: {observed.shape}, {control.shape}")
        if not np.isfinite(observed).all() or not np.isfinite(control).all():
            raise RuntimeError("Nonfinite encoded features")
        with np.load(output/"pusht/paired_trace.npz", allow_pickle=False) as raw:
            masked = raw["masks"].any(axis=(1, 2))
        if not np.allclose(observed[~masked], control[~masked], rtol=1e-5, atol=1e-6):
            raise RuntimeError("Identical visible RGB did not retain matching encoded features")
        if not np.any(observed[masked] != control[masked]):
            raise RuntimeError("Occluded and control RGB produced identical features throughout the mask interval")
    check = {
        "passed": True, "encoded_shape": [25, 5120], "dtype": str(observed.dtype), "finite": True,
        "unchanged_parameter_versions": True, "all_modules_eval": True, "trainable_parameters": 0,
        "visible_control_features_match": True, "masked_features_differ": True,
        "history_necessity_validated": False, "baseline_model_planning_verified": False,
        "predictor_trained": False, "training_gpu_seconds": 0,
    }
    write_json(output/"encoding_check.json", check)
    print(json.dumps(check, indent=2), flush=True)


def terminate_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run(args):
    output, cache = args.output.expanduser().resolve(), args.cache_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; prior run evidence is never overwritten")
    if args.timeout_seconds < 1 or args.timeout_seconds > 3000:
        raise ValueError("Timeout must be between 1 and 3000 seconds inside the one-hour allocation")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    manifest = {
        "schema_version": 1, "status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "real_frozen_ijepa_pusht_encoding_only", "timeout_seconds": args.timeout_seconds,
        "requested_device": "single_A100_cuda0", "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python_executable": sys.executable,
        "repository": PINNED_REPOSITORY, "revision": PINNED_REVISION,
        "history_necessity_validated": False, "baseline_model_planning_verified": False,
        "predictor_trained": False, "slurm_submitted_or_requeued": False,
    }
    process = None
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted_by_sigterm(_signum, _frame):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, interrupted_by_sigterm)
    try:
        if cache == output or cache.is_relative_to(output):
            raise ValueError("Keep cache outside the output directory; cached weights are not run trace artifacts")
        cache.mkdir(parents=True, exist_ok=True)
        config = json.loads(REFERENCE.read_text())
        encoder = config["encoder"]
        if encoder["repository"] != PINNED_REPOSITORY or encoder["revision"] != PINNED_REVISION:
            raise ValueError("Reference encoder does not match the reviewed immutable official checkpoint")
        encoder.update(dtype="float32", pooling="grid2", backend="ijepa", batch_size=2,
                       cache_dir=str(cache/"ijepa"), local_files_only=False)
        write_json(output/"encoder_config.json", {"encoder": encoder})
        paths = [Path(__file__), ROOT/"experiments/phase2/pusht.py", SRC/"phase1/encoders.py", REFERENCE]
        manifest["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        manifest["encoder_config_sha256"] = sha256(output/"encoder_config.json")
        env = os.environ.copy()
        env.update(HF_HOME=str(cache/"huggingface"), HF_HUB_CACHE=str(cache/"huggingface/hub"),
                   XDG_CACHE_HOME=str(cache/"xdg"), TORCH_HOME=str(cache/"torch"), PYTHONUNBUFFERED="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output),
                   "--cache-dir", str(cache), "--seed", str(args.seed)]
        manifest["command"] = command
        manifest["status"] = "running"
        write_json(output/"run_manifest.json", manifest)
        with (output/"worker.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       cwd=ROOT, start_new_session=True)
            code = process.wait(timeout=args.timeout_seconds)
        manifest["worker_exit_code"] = code
        if code != 0:
            raise RuntimeError(f"Encoding worker failed with exit code {code}; inspect worker.log")
        check = json.loads((output/"encoding_check.json").read_text())
        if not check["passed"]:
            raise RuntimeError("Worker did not produce a passing encoding check")
        manifest["status"] = "completed"
        manifest["encoding_verified"] = True
    except subprocess.TimeoutExpired:
        terminate_group(process)
        manifest.update(status="failed", encoding_verified=False, failure="worker_timeout_process_group_terminated")
    except KeyboardInterrupt:
        if process is not None:
            terminate_group(process)
        manifest.update(status="interrupted", encoding_verified=False, failure="interrupted_process_group_terminated")
    except Exception as exc:
        if process is not None:
            terminate_group(process)
        manifest.update(status="failed", encoding_verified=False, failure=f"{type(exc).__name__}: {exc}")
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic()-began
        manifest["artifact_sha256"] = {str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_manifest.json"}
        write_json(output/"run_manifest.json", manifest)
        signal.signal(signal.SIGTERM, old_sigterm)
    print(json.dumps({"output": str(output), "status": manifest["status"],
                      "encoding_verified": manifest.get("encoding_verified", False),
                      "failure": manifest.get("failure"), "baseline_model_planning_verified": False}, indent=2))
    return 0 if manifest["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
