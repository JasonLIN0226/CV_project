"""Freeze and execute the user-authorized one-shot CONF without training.

``lock`` only inspects passed DEV evidence and frozen inputs; it does not read
CONF. ``run`` verifies the lock, checks the downloaded transfer and appearance,
then records opening before reader/student predictions. Any failure preserves
all artifacts and ends the invocation; an existing attempt/opening is never
automatically resumed or overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
RES = HERE.parent
REPO = HERE.parents[3]
CODE = RES / "code"
RUN = REPO / "build/terminal-download-20261025/extracted/run"
AGG = REPO / "build/reader-aggregation-20261025/hpc/extracted/run"
DEV_XFER = REPO / "build/_claude_transfer_20261025"
BUILD = REPO / "build/qualified-reader-conf-20261026"
DEV_REPORT = BUILD / "dev-preflight-v2/dev_reproduction_v2.json"
STUDENTS = BUILD / "dev-preflight/students"
LOCK = HERE / "confirmation_execution_lock.json"
OPENING = HERE / "confirmation_opened.json"
ATTEMPT = HERE / "confirmation_attempt.json"
READERS = ("multihead", "item_attention")
GROUPS = tuple(r + "_" + m for r in READERS for m in ("standard", "distilled"))
README_SHA = "4c0c5898376e32801323a6ce3e3dad24ad0e67accc6828376ecc30a2867c68d8"
FTS_SHA = "08021072e0dd9b3a74dc6d397ba6d8a3f90c7db1c36fbfed6589a6a27221bcd7"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def require_absent(paths):
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise ValueError("Refusing to rerun/overwrite existing CONF artifacts: " + ", ".join(existing))


def verify_dev():
    report = read(DEV_REPORT)
    if report.get("status") != "passed" or report.get("confirmation_accessed") is not False:
        raise ValueError("Require successful v2 DEV preflight before CONF access")
    if report["preregistration"]["readme_sha256"] != README_SHA or sha(HERE / "README.md") != README_SHA:
        raise ValueError("Predeclared v2 rule text changed")
    if set(report["stageA"]) != set(READERS) or not all(report["stageA"][r]["passed"] for r in READERS):
        raise ValueError("Both DEV reader replays must pass")
    if set(report["students"]) != set(GROUPS) or not all(report["students"][g]["passed"] for g in GROUPS):
        raise ValueError("All four frozen student groups must pass")
    for name, digest in report["source_sha256"].items():
        if sha(CODE / name) != digest:
            raise ValueError("Code changed after the successful DEV replay: " + name)
    if sha(REPO / "experiments/phase2/fit_terminal_students.py") != FTS_SHA or report["frozen_student_source_sha256"] != FTS_SHA:
        raise ValueError("Frozen student implementation changed")
    if sha(report["tie_robustness"]["path"]) != report["tie_robustness"]["sha256"]:
        raise ValueError("DEV tie reference changed")
    return report


def frozen_input_paths(report):
    """Validate models and canonical archives, returning exact binding paths."""
    paths = {DEV_REPORT, HERE / "README.md", RES / "manifest_sha256.txt"}
    paths.update(CODE.glob("*.py"))
    paths.update(HERE.glob("*.py"))
    paths.update({HERE / "conf.sbatch", RUN / "protocol.json", RUN / "basis.npz",
                  RUN / "data/normalization.npz", RUN / "data/action_banks.npz",
                  DEV_XFER / "target_mean_coords.npy", Path(report["tie_robustness"]["path"])})
    tracked = subprocess.check_output(["git", "ls-files", "experiments/phase2", "experiments/predictor/control/src/phase1"], cwd=REPO, text=True)
    names = [n for n in tracked.splitlines() if n.endswith((".py", ".json"))]
    changed = subprocess.run(["git", "diff", "--quiet", "200a7d8", "--", *names], cwd=REPO)
    if changed.returncode:
        raise ValueError("Protected experiment dependencies differ from HEAD 200a7d8")
    paths.update(REPO / n for n in names)
    historical = {}
    for line in (RES / "manifest_sha256.txt").read_text().splitlines():
        digest, size, name = line.split(maxsplit=2)
        historical[name.removeprefix("./")] = (digest, int(size))
    for group in GROUPS:
        archive = RES / "students" / (group + ".zip")
        expected, size = historical["students/" + archive.name]
        if sha(archive) != expected or archive.stat().st_size != size:
            raise ValueError("Frozen student ZIP mismatch: " + group)
        paths.add(archive)
        model_dir = STUDENTS / group
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Invalid student ZIP CRC")
            for member in z.infolist():
                if member.is_dir():
                    continue
                rel = Path(member.filename)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Unsafe archive path")
                extracted = model_dir / rel
                if sha(extracted) != hashlib.sha256(z.read(member)).hexdigest():
                    raise ValueError("Extracted student differs from frozen ZIP: " + group + "/" + str(rel))
                paths.add(extracted)
        manifest = read(model_dir / "manifest.json")
        prefit = read(model_dir / "prefit_manifest.json")
        if sha(model_dir / "prefit_manifest.json") != manifest["prefit_manifest_sha256"] or prefit["source_sha256"] != FTS_SHA:
            raise ValueError("Student fit/source binding mismatch")
        if manifest["training_seeds"] != [17, 29, 43]:
            raise ValueError("Unexpected student seeds")
        if sha(model_dir / "manifest.json") != report["students"][group]["manifest_sha256"]:
            raise ValueError("Student manifest differs from successful DEV replay")
    parent = read(AGG / "parent_binding.json")
    for name in ("basis.npz", "protocol.json", "data/normalization.npz", "data/action_banks.npz"):
        if sha(RUN / name) != parent["verified_files"][name]["sha256"]:
            raise ValueError("Frozen TRAIN input differs from reader source: " + name)
    paths.add(AGG / "parent_binding.json")
    for reader in READERS:
        receipt = AGG / "readers" / reader / "reader_locked.json"
        locked = read(receipt)
        checkpoint = AGG / locked["checkpoint"]
        if sha(checkpoint) != locked["checkpoint_sha256"] or sha(RUN / "basis.npz") != locked["basis_sha256"]:
            raise ValueError("Reader/basis binding mismatch: " + reader)
        paths.update((receipt, checkpoint))
    return sorted(paths)


def make_lock():
    require_absent((LOCK, OPENING, ATTEMPT, HERE / "claims.json", HERE / "stageA", HERE / "predictions", HERE / "eval"))
    report = verify_dev()
    paths = frozen_input_paths(report)
    import numpy as np
    import torch
    if np.__version__ != report["versions"]["numpy"] or torch.__version__ != report["versions"]["torch"] or sys.version != report["versions"]["python"]:
        raise ValueError("Use the exact successful DEV Python/NumPy/Torch runtime")
    lock = {"status": "locked_before_confirmation_prediction", "utc": now(), "seed": 20261104,
            "families": 48, "no_confirmation_read": True, "dev_reproduction_passed": True,
            "dev_report": str(DEV_REPORT), "dev_report_sha256": sha(DEV_REPORT),
            "preregistered_readme_sha256": README_SHA, "versions": report["versions"],
            "execution_settings": report["execution_settings"],
            "bindings": {str(p.relative_to(REPO)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths}}
    write_new(LOCK, lock)
    print(json.dumps({"status": lock["status"], "lock": str(LOCK), "sha256": sha(LOCK), "bound_files": len(paths)}))


def verify_lock():
    lock = read(LOCK)
    if lock.get("status") != "locked_before_confirmation_prediction" or lock.get("seed") != 20261104:
        raise ValueError("Invalid execution lock")
    for relative, expected in lock["bindings"].items():
        path = REPO / relative
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Input/source changed after freeze: " + relative)
    report = verify_dev()
    if sha(DEV_REPORT) != lock["dev_report_sha256"]:
        raise ValueError("DEV receipt changed after freeze")
    import numpy as np
    import torch
    if np.__version__ != lock["versions"]["numpy"] or torch.__version__ != lock["versions"]["torch"] or sys.version != lock["versions"]["python"]:
        raise ValueError("Python/NumPy/Torch runtime differs from frozen DEV replay")
    return lock, report


def run_script(script, args, environment, log, torch_setup=False):
    """Subprocess uses the same thread policy as the passed DEV reader replay."""
    verify_lock()
    attempt = read(ATTEMPT)
    if OPENING.exists() and read(OPENING)["attempt_sha256"] != sha(ATTEMPT):
        raise ValueError("CONF attempt binding changed after opening")
    transfer = Path(attempt["transfer_directory"])
    for name, digest in attempt["transfer_bindings"].items():
        if Path(name).name != name or sha(transfer / name) != digest:
            raise ValueError("CONF input changed during one-shot execution")
    if torch_setup:
        bootstrap = "import runpy,sys,torch; torch.set_num_threads(2); torch.use_deterministic_algorithms(True); p=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(p,run_name='__main__')"
        command = [sys.executable, "-I", "-c", bootstrap, str(script), *map(str, args)]
    else:
        command = [sys.executable, "-I", str(script), *map(str, args)]
    with log.open("x") as f:
        subprocess.run(command, cwd=REPO, env=environment, stdout=f, stderr=subprocess.STDOUT, check=True)


def run_once(transfer, archive, expected_zip_sha, job_id):
    require_absent((ATTEMPT, OPENING, HERE / "claims.json", HERE / "stageA", HERE / "predictions",
                    HERE / "eval", HERE / "integrity.json", HERE / "receipt.json", HERE / "logs",
                    HERE / "appearance.npz", HERE / "confirmation_failed.json", HERE / "confirmation_completed.json"))
    lock, report = verify_lock()
    if not re.fullmatch("[0-9a-f]{64}", expected_zip_sha) or sha(archive) != expected_zip_sha:
        raise ValueError("Downloaded ZIP does not match the SHA printed by the A100 job")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Transfer ZIP CRC failed")
        names = [m.filename for m in z.infolist() if not m.is_dir()]
        if len(names) != len(set(names)) or set(names) != {p.name for p in transfer.iterdir() if p.is_file()}:
            raise ValueError("Transfer directory and ZIP members differ")
        for name in names:
            if Path(name).name != name or sha(transfer / name) != hashlib.sha256(z.read(name)).hexdigest():
                raise ValueError("Extracted transfer member differs from verified ZIP")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("QR_")}
    for name, value in lock["execution_settings"]["environment"].items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = str(value)
    environment.update({"QR_REPO": str(REPO), "QR_RUN": str(RUN), "QR_XFER": str(transfer),
                        "QR_AGG": str(AGG), "QR_RECORDS": str(transfer / "records.json"),
                        "QR_ACTIONS": str(transfer / "actions.npz"), "QR_RGB_INDEX": str(transfer / "rgb_index.json"),
                        "QR_APPEARANCE_OUTPUT": str(HERE / "appearance.npz"),
                        "QR_SPLIT": "confirmation", "QR_SPLITS": "confirmation"})
    write_new(ATTEMPT, {"utc": now(), "split": "confirmation", "seed": 20261104,
                        "phase": "pre-prediction integrity", "execution_lock_sha256": sha(LOCK),
                        "zip_sha256": expected_zip_sha, "job_id": str(job_id),
                        "transfer_directory": str(transfer),
                        "transfer_bindings": {name: sha(transfer / name) for name in names},
                        "automatic_resume_allowed": False, "dev_reproduction_passed": True})
    logs = HERE / "logs"
    logs.mkdir(exist_ok=False)
    try:
        run_script(CODE / "appearance.py", [], environment, logs / "appearance.log")
        run_script(HERE / "conf_integrity.py",
                   ["--transfer", transfer, "--train-run", RUN, "--dev-transfer", DEV_XFER,
                    "--appearance", HERE / "appearance.npz", "--appearance-log", logs / "appearance.log",
                    "--output", HERE / "integrity.json"], environment, logs / "integrity.log")
        integrity = read(HERE / "integrity.json")
        if integrity.get("status") != "passed":
            raise ValueError("CONF integrity did not pass")
        verify_lock()
        write_new(HERE / "receipt.json", {"job_id": str(job_id), "zip_sha256": expected_zip_sha,
                                           "execution_lock_sha256": sha(LOCK), "dev_preflight_sha256": sha(DEV_REPORT),
                                           "integrity_sha256": sha(HERE / "integrity.json"),
                                           "encoding_receipt": read(transfer / "receipt.json")})
        write_new(OPENING, {"utc": now(), "split": "confirmation", "seed": 20261104, "families": 48,
                           "dev_reproduction_passed": True, "integrity_passed": True,
                           "execution_lock_sha256": sha(LOCK), "receipt_sha256": sha(HERE / "receipt.json"),
                           "attempt_sha256": sha(ATTEMPT),
                           "before_first_reader_or_student_prediction": True,
                           "automatic_resume_allowed": False})
        (HERE / "stageA").mkdir()
        (HERE / "predictions").mkdir()
        (HERE / "eval").mkdir()
        for reader in READERS:
            run_script(CODE / "run_main.py", [reader, HERE / "stageA"], environment,
                       logs / (reader + "_stageA.log"), torch_setup=True)
            run_script(HERE / "conf_predict.py",
                       ["--reader", reader, "--students", STUDENTS, "--appearance", HERE / "appearance.npz",
                        "--opening-receipt", OPENING, "--output", HERE / "predictions"], environment,
                       logs / (reader + "_students.log"))
            run_script(CODE / "evaluate_dev.py", [reader, HERE / "stageA", HERE / "predictions",
                                                  HERE / "eval" / (reader + "_conf.json")],
                       environment, logs / (reader + "_evaluation.log"))
        run_script(HERE / "conf_claims.py",
                   ["--records", transfer / "records.json", "--stage-a", HERE / "stageA",
                    "--predictions", HERE / "predictions", "--evaluations", HERE / "eval",
                    "--output", HERE / "claims.json"], environment, logs / "claims.log")
        verify_lock()
        result_paths = list((HERE / "stageA").glob("*.npz")) + list((HERE / "predictions").rglob("*")) + list((HERE / "eval").glob("*.json"))
        result_paths += [HERE / "claims.json", HERE / "receipt.json", HERE / "integrity.json", OPENING]
        write_new(HERE / "confirmation_completed.json", {
            "utc": now(), "status": "completed", "training_performed": False, "split": "confirmation",
            "execution_lock_sha256": sha(LOCK), "opening_sha256": sha(OPENING),
            "results": {str(p.relative_to(HERE)): {"sha256": sha(p), "bytes": p.stat().st_size}
                        for p in result_paths if p.is_file()}})
        print(json.dumps({"status": "completed", "claims": str(HERE / "claims.json")}), flush=True)
    except Exception as exc:
        failure = HERE / "confirmation_failed.json"
        if not failure.exists():
            write_new(failure, {"utc": now(), "error": repr(exc), "opening_exists": OPENING.exists(),
                                "automatic_resume_allowed": False, "artifacts_preserved": True})
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("lock")
    run = sub.add_parser("run")
    run.add_argument("--transfer", type=Path, required=True)
    run.add_argument("--zip", type=Path, required=True)
    run.add_argument("--expected-zip-sha256", required=True)
    run.add_argument("--job-id", required=True)
    args = ap.parse_args()
    if args.command == "lock":
        make_lock()
    else:
        run_once(args.transfer.resolve(), args.zip.resolve(), args.expected_zip_sha256, args.job_id)


if __name__ == "__main__":
    main()
