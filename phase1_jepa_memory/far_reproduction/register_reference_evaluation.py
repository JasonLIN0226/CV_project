"""Exactly-once registration after actual complete public-release preparation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if os.environ.get("SLURM_JOB_ACCOUNT") != "csci_ga_2271-2026fa":
        raise RuntimeError("course allocation required")
    for name, h in json.loads((ROOT / "evaluation_source_lock.json").read_text())["files"].items():
        if sha(ROOT / name) != h:
            raise RuntimeError("evaluation/controller source differs")
    r = subprocess.check_output(["/opt/slurm/bin/sacct", "-j", "7385", "-X", "-n", "-P",
                                 "-o", "JobID,State,ExitCode,Account,AllocTRES"], text=True)
    row = r.strip().split("|")
    if row[:4] != ["7385", "COMPLETED", "0:0", "csci_ga_2271-2026fa"]:
        raise RuntimeError("actual data preparation not completed0:0: " + r)
    prep = ROOT / "preparation/job_7385/preparation_report.json"
    d = json.loads(prep.read_text())
    if d["status"] != "PUBLIC_RELEASE_FILES_AND_DEPENDENCIES_PREPARED" or d["test_clips_declared"] != 3840:
        raise RuntimeError("actual prepared release incomplete")
    q = subprocess.check_output(["/opt/slurm/bin/squeue", "-u", "xl4358", "-h", "-o", "%i %j %b"], text=True)
    if any(line.split()[0] != os.environ["SLURM_JOB_ID"] for line in q.splitlines()):
        raise RuntimeError("another allocation exists; refuse parallel science")
    binding = {"preparation_job": 7385, "metadata_job": int(os.environ["SLURM_JOB_ID"]),
               "preparation_report_sha256": sha(prep),
               "protocol_sha256": sha(ROOT / "reference_protocol.json"),
               "evaluation_source_sha256": sha(ROOT / "evaluation_source_lock.json")}
    (ROOT / "evaluation_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    attempt = ROOT / "evaluation_submission_attempt.json"
    fd = os.open(attempt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"status": "ATTEMPT_STARTED_UNKNOWN_OUTCOME", **binding}, f)
    x = subprocess.run(["/opt/slurm/bin/sbatch", "--parsable", str(ROOT / "evaluate_reference.sbatch")],
                       capture_output=True, text=True)
    receipt = {"returncode": x.returncode, "stdout": x.stdout.strip(), "stderr": x.stderr.strip(), **binding}
    if x.returncode == 0:
        value = x.stdout.strip().split(";")[0]
        if not value.isdigit():
            raise RuntimeError("unknown successful submission receipt; do not retry")
        receipt.update(status="SUBMITTED_ACTUAL_RECEIPT", job_id=int(value))
    else:
        receipt.update(status="SUBMISSION_REJECTED_NO_JOB_ID", job_id=None)
    (ROOT / "evaluation_submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    attempt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if x.returncode:
        raise RuntimeError("actual GPU submission rejected")


if __name__ == "__main__":
    main()
