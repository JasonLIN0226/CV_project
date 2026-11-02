"""Generate once and run the preregistered real-data pytest audit once, on CPU."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROTOCOL = Path("experiments/motion_v2/protocol_step1_v2_2.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = json.loads(PROTOCOL.read_text())
    for name, expected in {**p["source_sha256"], str(PROTOCOL): sha(PROTOCOL)}.items():
        if sha(name) != expected or hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{name}"])).hexdigest() != expected:
            raise RuntimeError(f"Source/protocol changed or not committed: {name}")
    for name in p["prerequisite_status_paths"]:
        if json.loads(Path(name).read_text())["status"] != "passed":
            raise RuntimeError(f"Prior gate did not pass: {name}")
    out = Path(p["output_root"])
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    status = dict(status="running", protocol_sha256=sha(PROTOCOL),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  python=sys.version, platform=platform.platform(), gpu_seconds=0,
                  test_data_opened_or_generated=False, generator_runs=0, audit_runs=0)
    save(out / "status.json", status)
    env = dict(os.environ, **p["thread_environment"], **p["audit_environment"])
    try:
        for step, argv in p["commands"].items():
            command = [sys.executable, "-I", "-B", *argv]
            status["generator_runs" if step == "generate" else "audit_runs"] += 1
            with (out / f"{step}.stdout.log").open("w") as stdout, (out / f"{step}.stderr.log").open("w") as stderr:
                run = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=3600)
            status[step] = {"command": command, "exit_code": run.returncode}
            if run.returncode:
                status["status"] = f"stopped_{step}_failure"
                break
        else:
            audit = json.loads(Path(p["audit_environment"]["MOTION_V2_AUDIT_OUTPUT"]).read_text())
            status["status"] = "passed" if audit["status"] == "passed" else "stopped_audit_failure"
    except Exception as exc:
        status.update(status="stopped_runtime_failure", error=f"{type(exc).__name__}: {exc}")
    status["cpu_wall_seconds"] = time.perf_counter() - began
    save(out / "status.json", status)
    # The dataset manifest separately hashes every observation/evaluation NPZ.
    files = sorted(x for x in out.rglob("*") if x.is_file() and x.suffix not in {".npz", ".npy"})
    (out / "SHA256SUMS").write_text("".join(f"{sha(x)}  {x.as_posix()}\n" for x in files))
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0 if status["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
