"""Bounded Apple-MPS-only validation of the actual frozen PushT/I-JEPA path.

This is explicitly LOCAL Apple MPS, not an A100 run. No model is trained. The
child reuses the existing PushT CLI and original frozen encoder implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase2.run_encoder_pilot import PINNED_REPOSITORY, PINNED_REVISION, sha256, write_json, terminate_group

SRC = ROOT/"experiments/predictor/control/src"
REFERENCE = ROOT/"results/reference/gate/resolved_config.json"


def worker(output):
    import numpy as np
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS unavailable; no CPU or A100 substitution")
    sys.path.insert(0, str(SRC))
    import phase1.encoders as encoders
    from experiments.phase2 import pusht
    hardware = {
        "device": "Apple MPS", "a100_used": False, "host": socket.gethostname(),
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "huggingface-hub", "safetensors", "numpy", "Pillow", "gym-pusht")},
        "mps_available": True, "torch_cuda": torch.version.cuda,
        "mps_cpu_operator_fallback_enabled": False,
    }
    write_json(output/"hardware.json", hardware)
    print(json.dumps(hardware, indent=2), flush=True)
    original = encoders.FrozenIJepa
    instances = []

    class AuditedMPS(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            parameters = list(self.model.parameters())
            if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in parameters):
                raise RuntimeError("Encoder must be eval and frozen")
            if any(p.device.type != "mps" or p.dtype != torch.float32 for p in parameters):
                raise RuntimeError("Every encoder parameter must be float32 on Apple MPS")
            self._pilot_versions = tuple(p._version for p in parameters)
            instances.append(self)
            write_json(output/"frozen_encoder_audit.json", {
                "device": "Apple MPS", "a100_used": False, "all_modules_eval": True,
                "trainable_parameters": 0, "parameter_dtype": "float32",
                "parameter_count": sum(p.numel() for p in parameters),
                "parameter_bytes": sum(p.numel()*p.element_size() for p in parameters),
                "metadata": self.metadata,
            })
            print("Actual pretrained I-JEPA loaded, frozen and on Apple MPS", flush=True)

    encoders.FrozenIJepa = AuditedMPS
    sys.argv = [str(ROOT/"experiments/phase2/pusht.py"), "--output", str(output/"pusht"),
                "--steps", "24", "--seed", "17", "--encoder-config", str(output/"encoder_config.json"),
                "--device", "mps", "--batch-size", "2"]
    pusht.main()
    torch.mps.synchronize()
    if len(instances) != 1:
        raise RuntimeError("Expected one real encoder")
    if tuple(p._version for p in instances[0].model.parameters()) != instances[0]._pilot_versions:
        raise RuntimeError("Encoder parameters changed during inference")
    with np.load(output/"pusht/encoded_trace.npz", allow_pickle=False) as f:
        observed, control = f["observed"], f["control"]
        if observed.shape != (25, 5120) or control.shape != (25, 5120):
            raise RuntimeError("Expected paired (25,5120) real I-JEPA features")
        if not np.isfinite(observed).all() or not np.isfinite(control).all():
            raise RuntimeError("Nonfinite feature values")
        with np.load(output/"pusht/paired_trace.npz", allow_pickle=False) as f:
            masked = f["masks"].any(axis=(1, 2))
        if not np.allclose(observed[~masked], control[~masked], atol=1e-6, rtol=1e-5):
            raise RuntimeError("Visible control features differ")
        if not np.any(observed[masked] != control[masked]):
            raise RuntimeError("Masked RGB did not change any feature")
    result = {
        "passed": True, "device": "Apple MPS", "a100_used": False,
        "encoded_shape_per_variant": [25, 5120], "dtype": str(observed.dtype), "finite": True,
        "all_modules_eval": True, "trainable_parameters": 0, "unchanged_parameter_versions": True,
        "visible_control_features_match": True, "masked_features_differ": True,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "history_necessity_validated": False, "baseline_model_planning_verified": False,
        "predictor_trained": False,
    }
    write_json(output/"encoding_check.json", result)
    print(json.dumps(result, indent=2), flush=True)


def run(args):
    output, cache = args.output.resolve(), args.cache_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new and empty")
    if not 1 <= args.timeout_seconds <= 1200:
        raise ValueError("Local validation is bounded to at most 1200 seconds")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {
        "schema_version": 1, "status": "starting", "device": "Apple MPS", "a100_used": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": args.timeout_seconds,
        "scope": "local_frozen_ijepa_pusht_encoder_validation_only",
        "predictor_trained": False, "baseline_model_planning_verified": False, "history_necessity_validated": False,
        "slurm_submitted_or_requeued": False,
    }
    process = None
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if cache == output or cache.is_relative_to(output):
            raise ValueError("Cache must be outside the trace output")
        cache.mkdir(parents=True, exist_ok=True)
        config = json.loads(REFERENCE.read_text())["encoder"]
        if config["repository"] != PINNED_REPOSITORY or config["revision"] != PINNED_REVISION:
            raise ValueError("Pinned official reference changed")
        config.update(dtype="float32", pooling="grid2", backend="ijepa", cache_dir=str(cache/"ijepa"),
                      local_files_only=False, batch_size=2)
        write_json(output/"encoder_config.json", {"encoder": config})
        paths = [Path(__file__), ROOT/"experiments/phase2/pusht.py", ROOT/"experiments/phase2/run_encoder_pilot.py",
                 SRC/"phase1/encoders.py", REFERENCE]
        manifest["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        manifest["encoder_config_sha256"] = sha256(output/"encoder_config.json")
        manifest["checkpoint"] = {"repository": PINNED_REPOSITORY, "revision": PINNED_REVISION}
        env = os.environ.copy()
        env.update(HF_HOME=str(cache/"huggingface"), HF_HUB_CACHE=str(cache/"huggingface/hub"),
                   TORCH_HOME=str(cache/"torch"), XDG_CACHE_HOME=str(cache/"xdg"),
                   PYTORCH_ENABLE_MPS_FALLBACK="0", PYTHONUNBUFFERED="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output),
                   "--cache-dir", str(cache)]
        manifest.update(status="running", command=command)
        write_json(output/"run_manifest.json", manifest)
        with (output/"worker.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            code = process.wait(timeout=args.timeout_seconds)
        manifest["worker_exit_code"] = code
        if code != 0:
            raise RuntimeError(f"MPS worker exited {code}; see worker.log")
        result = json.loads((output/"encoding_check.json").read_text())
        if not result["passed"]:
            raise RuntimeError("Encoding verification did not pass")
        manifest.update(status="completed", encoding_verified=True)
    except subprocess.TimeoutExpired:
        terminate_group(process)
        manifest.update(status="failed", encoding_verified=False, failure="1200-second bounded child timeout; group terminated")
    except KeyboardInterrupt:
        if process is not None:
            terminate_group(process)
        manifest.update(status="interrupted", encoding_verified=False, failure="child process group terminated")
    except Exception as exc:
        if process is not None:
            terminate_group(process)
        manifest.update(status="failed", encoding_verified=False, failure=f"{type(exc).__name__}: {exc}")
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic()-started
        manifest["artifact_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                                       if p.is_file() and p.name != "run_manifest.json"}
        write_json(output/"run_manifest.json", manifest)
        signal.signal(signal.SIGTERM, old_sigterm)
    print(json.dumps({"output": str(output), "device": "Apple MPS", "a100_used": False,
                      "status": manifest["status"], "elapsed_seconds": manifest["elapsed_seconds"],
                      "failure": manifest.get("failure")}, indent=2))
    return 0 if manifest["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.output.resolve())
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
