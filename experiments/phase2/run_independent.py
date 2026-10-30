"""Run only asset-independent CPU checks and an actual PushT interface trace.

No encoder download, trained-model inference, Slurm submission, or training is
performed. The manifest deliberately cannot certify scientific Phase 2 results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], log: Path) -> dict:
    started = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                check=False)
    return {"argv": argv, "exit_code": result.returncode,
            "seconds": time.monotonic() - started, "log": log.name,
            "log_sha256": sha256(log)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty output directory; prior results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {
        "schema": "phase2_independent_checks_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU software contracts and real PushT environment interface only",
        "python": sys.version, "platform": platform.platform(),
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "gpu_allocations_submitted": 0, "gpu_allocation_seconds": 0,
        "trained_model_evaluation": False, "predictor_training": False,
        "scientific_deletion_dataset_generated": False,
        "six_model_bound_baseline_results": False,
        "baseline_model_planning_verified": False,
        "phase2_completed": False, "checks": [],
    }
    # Include source identity even for files not yet committed to the work branch.
    sources = sorted((REPO / "experiments/phase2").glob("*.py"))
    sources += sorted((REPO / "tests").glob("test_phase2_*.py"))
    manifest["source_sha256"] = {str(p.relative_to(REPO)): sha256(p) for p in sources}
    with (output / "environment.txt").open("x") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=REPO,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    jobs = [
        ("dependencies", [sys.executable, "-m", "pip", "check"]),
        ("published_latest_evidence", [sys.executable, "tools/verify_phase1_latest.py"]),
        ("published_earlier_evidence", [sys.executable, "tools/verify_results.py"]),
        ("software_tests", [sys.executable, "-m", "pytest", "-q", "--junitxml", str(output / "tests.xml")]),
        ("pusht_interface", [sys.executable, "experiments/phase2/pusht.py", "--output",
                              str(output / "pusht"), "--seed", "17", "--steps", "24"]),
    ]
    for name, command in jobs:
        result = {"name": name, **run(command, output / f"{name}.log")}
        manifest["checks"].append(result)
        print(f"{name}: {'PASS' if result['exit_code'] == 0 else 'FAIL'} ({result['seconds']:.2f}s)", flush=True)
        if result["exit_code"]:
            break
    if (output / "tests.xml").exists():
        suites = ET.parse(output / "tests.xml").getroot().iter("testsuite")
        counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.attrib.get(key, 0))
        manifest["software_test_counts"] = counts
    if (output / "pusht/manifest.json").exists():
        manifest["pusht_manifest_sha256"] = sha256(output / "pusht/manifest.json")
    manifest["independent_checks_passed"] = len(manifest["checks"]) == len(jobs) and all(
        c["exit_code"] == 0 for c in manifest["checks"])
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(output / "manifest.json"), flush=True)
    return 0 if manifest["independent_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
