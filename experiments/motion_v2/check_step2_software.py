"""Run preregistered statistics-only checks once; never import model code."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "experiments/motion_v2/protocol_step2_v2_2.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "FROZEN_STEP2_PROTOCOL"
    assert protocol["registered_before_run"] is True
    for name, expected in {**protocol["source_sha256"], str(PROTOCOL.relative_to(REPO)): sha(PROTOCOL)}.items():
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=REPO)
        if sha(REPO / name) != expected or hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError(f"Uncommitted or changed registered file: {name}")
    settings = protocol["software_preflight"]
    output = REPO / settings["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **settings["environment"])
    env["MOTION_V2_SOFTWARE_TEST_OUTPUT"] = str(output / "retained_array_fixture")
    command = [sys.executable, "-I", "-B", *settings["pytest_arguments"]]
    started = time.perf_counter()
    status = {
        "status": "not_run", "protocol_sha256": sha(PROTOCOL),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "command": command, "software_only": True, "experiment_runs": 0,
        "model_or_decoder_fits": 0, "gpu_seconds": 0, "test_episode_access": False,
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pytest")},
    }
    try:
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            run = subprocess.run(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, timeout=300)
        status.update(exit_code=run.returncode, status="passed" if run.returncode == 0 else "stopped_software_failure")
    except Exception as exc:
        status.update(status="stopped_software_failure", error=f"{type(exc).__name__}: {exc}")
    status["cpu_wall_seconds"] = time.perf_counter() - started
    save(output / "status.json", status)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    with (output / "SHA256SUMS").open("x") as destination:
        destination.write("".join(f"{sha(path)}  {path.relative_to(REPO).as_posix()}\n" for path in files))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
