"""Retained, CPU-only Slurm-record software replay after protocol registration."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
AMENDMENT = REPO / "experiments/motion_v2/amendment_v2_3_1.json"
TEST = "experiments/motion_v2/test_slurm_records_v2_3_1.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")


def committed(path):
    path = Path(path).resolve()
    relative = path.relative_to(REPO).as_posix()
    result = subprocess.run(["git", "show", "HEAD:" + relative], cwd=REPO,
                            check=True, capture_output=True, timeout=30)
    if result.stdout != path.read_bytes():
        raise ValueError("Software source/protocol differs from committed HEAD: " + relative)
    return relative


def registration(protocol_path):
    committed(protocol_path)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if (protocol.get("status") != "FROZEN_SLURM_REPLAY_PROTOCOL" or
            protocol.get("version") != "v2.3.1" or
            protocol.get("registered_before_run") is not True):
        raise ValueError("Expected the committed v2.3.1 replay protocol")
    committed(AMENDMENT)
    if protocol.get("amendment_sha256") != sha256(AMENDMENT):
        raise ValueError("Replay amendment SHA256 mismatch")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    sources = protocol["source_sha256"]
    required = {TEST, Path(__file__).resolve().relative_to(REPO).as_posix(),
                "experiments/motion_v2/slurm_records.py", "experiments/motion_v2/run_step2.py",
                "experiments/motion_v2/submit_step2_v2_3_1.py"}
    if not required <= set(sources):
        raise ValueError("Replay must lock tests, driver and the actual runtime validators")
    for relative, expected in sources.items():
        path = REPO / relative
        committed(path)
        if sha256(path) != expected:
            raise ValueError("Registered source SHA256 mismatch: " + relative)
    fixtures = protocol["fixture_sha256"]
    if not amendment["fixture_sha256"].items() <= fixtures.items():
        raise ValueError("Replay must retain all three original amendment fixtures")
    for relative, expected in fixtures.items():
        path = (REPO / relative).resolve()
        path.relative_to(REPO)
        if sha256(path) != expected:
            raise ValueError("Registered fixture SHA256 mismatch: " + relative)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise ValueError("Invoke this replay with Python -I -B")
    protocol = registration(args.protocol)
    output = args.output.absolute()
    output.relative_to(REPO / "results/motion_v2/v2_3_1/software")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh retained attempt directory; never overwrite or remove a run")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
               MOTION_SLURM_REPLAY_OUTPUT=str(output),
               MOTION_SLURM_REPLAY_PROTOCOL=str(args.protocol.resolve()))
    command = [sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-o", "addopts=", "-o", "pythonpath=", TEST,
               "--junitxml=" + str(output / "pytest.xml")]
    status = {"status": "not_run", "version": "v2.3.1", "software_only": True,
              "source_sha256": protocol["source_sha256"], "fixture_sha256": protocol["fixture_sha256"],
              "protocol_sha256": sha256(args.protocol), "amendment_sha256": sha256(AMENDMENT),
              "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                                       text=True, timeout=30).strip(),
              "command": command, "python": sys.version, "platform": platform.platform(),
              "pytest_version": importlib.metadata.version("pytest"),
              "scientific_runner_invocations": 0, "native_cluster_commands": 0,
              "real_polling_sleeps": 0, "gpu_seconds": 0, "model_imports_or_fits": 0,
              "dataset_arrays_opened": False, "test_episode_access": False}
    save(output / "before_run.json", status)
    start = time.perf_counter()
    try:
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            completed = subprocess.run(command, cwd=REPO, env=env, stdout=stdout,
                                       stderr=stderr, timeout=300)
        status["exit_code"] = completed.returncode
        report_path = output / "fixture_replay.json"
        if not report_path.is_file():
            raise RuntimeError("The real-fixture field-by-field replay report is missing")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status["fixture_replay_status"] = report["status"]
        xml = ET.parse(output / "pytest.xml").getroot()
        suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
        status["pytest"] = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                            for key in ("tests", "failures", "errors", "skipped")}
        passed = (completed.returncode == 0 and report["status"] == "passed" and
                  status["pytest"]["tests"] > 0 and
                  all(status["pytest"][key] == 0 for key in ("failures", "errors", "skipped")))
        status["status"] = "passed" if passed else "stopped_software_failure"
    except Exception as error:
        status.update(status="stopped_software_failure", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    status["cpu_wall_seconds"] = time.perf_counter() - start
    save(output / "status.json", status)
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as destination:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                destination.write(sha256(path) + "  " + path.relative_to(REPO).as_posix() + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if status["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
