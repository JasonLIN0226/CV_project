"""Run only preregistered local software checks; never contact Slurm or a model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results/motion_v2/v2_3_1_resubmission/software"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def committed(path):
    relative = path.resolve().relative_to(REPO).as_posix()
    content = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=REPO, timeout=30)
    if content != path.read_bytes():
        raise ValueError("File differs from committed HEAD: " + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise ValueError("Use Python -I -B")
    protocol_path = args.protocol.resolve()
    committed(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "FROZEN_RUNTIME_SOFTWARE_PROTOCOL" or protocol.get("registered_before_run") is not True:
        raise ValueError("Expected preregistered runtime-only software protocol")
    if protocol.get("test_paths") != ["experiments/motion_v2/test_resubmit_step2_v2_3_1.py"]:
        raise ValueError("Only the registered runtime test module may run")
    required = set(protocol["test_paths"]) | {Path(__file__).relative_to(REPO).as_posix()}
    if not required <= set(protocol["source_sha256"]):
        raise ValueError("Protocol must lock the tests and this driver")
    for relative, expected in protocol["source_sha256"].items():
        path = REPO / relative
        committed(path)
        if sha(path) != expected:
            raise ValueError("Registered source hash mismatch: " + relative)
    output = args.output.absolute()
    output.relative_to(RESULTS)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse an existing software output directory")
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-o", "addopts=", "-o", "pythonpath=", *protocol["test_paths"], "--junitxml=" + str(output / "pytest.xml")]
    record = {"status": "not_run", "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(protocol_path), "source_sha256": protocol["source_sha256"],
              "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "command": command, "python": sys.version, "platform": platform.platform(),
              "scientific_entrypoint_invoked": False, "slurm_contacted": False, "actual_a100_gpu_seconds": 0}
    save(output / "before_run.json", record)
    start = time.monotonic()
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    record["status"] = "failed"
    try:
        with (output / "stdout.txt").open("x") as stdout, (output / "stderr.txt").open("x") as stderr:
            result = subprocess.run(command, cwd=REPO, env=environment, stdout=stdout, stderr=stderr, timeout=120)
        record["returncode"] = result.returncode
        xml = ET.parse(output / "pytest.xml")
        suites = list(xml.getroot().iter("testsuite"))
        counts = {name: sum(int(suite.get(name, "0")) for suite in suites) for name in ("tests", "errors", "failures", "skipped")}
        record["pytest"] = counts
        record["status"] = "passed" if result.returncode == 0 and counts["tests"] > 0 and not any(counts[k] for k in ("errors", "failures", "skipped")) else "failed"
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    record.update(elapsed_seconds=time.monotonic() - start,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save(output / "status.json", record)
    with (output / "SHA256SUMS").open("x") as stream:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(sha(path) + "  " + path.name + "\n")
    print(json.dumps(record, ensure_ascii=False))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
