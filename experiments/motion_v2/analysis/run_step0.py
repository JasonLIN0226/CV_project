"""Run each preregistered reference check once, preserving stdout and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(expected, actual, path="$"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                out.append({"path": f"{path}.{key}", "expected": expected.get(key), "actual": actual.get(key)})
            else:
                out.extend(differences(expected[key], actual[key], f"{path}.{key}"))
        return out
    return [] if expected == actual else [{"path": path, "expected": expected, "actual": actual}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["v2_1_reference", "phase1_headroom"])
    args = parser.parse_args()
    protocol_path = Path("experiments/motion_v2/protocol_step0_v2_1.json")
    protocol = json.loads(protocol_path.read_text())
    frozen = {**protocol["source_sha256"], str(protocol_path): sha(protocol_path)}
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise RuntimeError(f"Source hash mismatch: {path}")
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"])
        if hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError(f"Source/protocol must be committed before execution: {path}")
    root = Path(protocol["output_directory"])
    previous = list(root.glob("*/status.json")) if root.exists() else []
    if any(json.loads(p.read_text())["status"] != "passed" for p in previous):
        raise RuntimeError("A prior check did not pass; stop without retrying")
    out = root / args.check
    out.mkdir(parents=True, exist_ok=False)
    spec = protocol["checks"][args.check]
    command = [sys.executable, "-I", *spec["arguments"]]
    env = dict(os.environ, **protocol["thread_environment"])
    status = {"check": args.check, "status": "running", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "protocol_sha256": sha(protocol_path), "command": command, "python": sys.version,
              "platform": platform.platform(), "gpu_seconds": 0, "runtime_test_data_opened_or_generated": False,
              "reference_scene_metadata_reconstruction_only": args.check == "phase1_headroom"}
    def write_status():
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    write_status()
    started = time.perf_counter()
    try:
        with (out / "stdout.json").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
            run = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=protocol["timeout_seconds_per_check"])
        status["exit_code"] = run.returncode
        if run.returncode:
            status["status"] = "stopped_runtime_failure"
        else:
            expected = json.loads(Path(spec["reference_json"]).read_text())
            actual = json.loads((out / "stdout.json").read_text())
            diff = differences(expected, actual)
            (out / "comparison.json").write_text(json.dumps({"exact_json_match": not diff, "differences": diff}, indent=2) + "\n")
            status["status"] = "passed" if not diff else "stopped_reference_mismatch"
            status["exact_json_match"] = not diff
    except Exception as exc:
        status.update(status="stopped_runtime_failure", error=f"{type(exc).__name__}: {exc}")
    finally:
        status["cpu_wall_seconds"] = time.perf_counter() - started
        write_status()
        files = sorted(p for p in out.iterdir() if p.is_file())
        (out / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.as_posix()}\n" for p in files))
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
