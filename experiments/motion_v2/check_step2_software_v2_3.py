"""One-shot synthetic-array tests authorized by the pushed v2.3 amendment."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
AMENDMENT = REPO / "experiments/motion_v2/amendment_v2_3.json"
OUTPUT = REPO / "results/motion_v2/v2_3/software"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    amendment = json.loads(AMENDMENT.read_text())
    if amendment["version"] != "v2.3":
        raise RuntimeError("Expected the preregistered v2.3 amendment")
    if sha(REPO / amendment["amendment_path"]) != amendment["amendment_sha256"]:
        raise RuntimeError("Amendment document hash mismatch")
    names = ["experiments/motion_v2/amendment_v2_3.json", amendment["amendment_path"],
             "experiments/motion_v2/step2_metrics.py",
             "experiments/motion_v2/test_step2_metrics.py",
             "experiments/motion_v2/check_step2_software_v2_3.py"]
    hashes = {}
    for name in names:
        raw = (REPO / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=REPO)
        if raw != committed:
            raise RuntimeError(f"Test source not identical to committed HEAD: {name}")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               BLIS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               MOTION_V2_SOFTWARE_TEST_OUTPUT=str(OUTPUT / "retained_array_fixture"))
    command = [sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-o", "addopts=", "-o", "pythonpath=",
               "experiments/motion_v2/test_step2_metrics.py",
               "--junitxml=" + str(OUTPUT / "pytest.xml")]
    status = {"status": "not_run", "amendment_sha256": sha(AMENDMENT),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "command": command, "source_sha256": hashes, "software_only": True,
              "experiment_runs": 0, "model_or_decoder_fits": 0, "gpu_seconds": 0,
              "actual_validation_access": False, "test_episode_access": False,
              "python": sys.version, "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pytest")}}
    save(OUTPUT / "before_run.json", status)
    started = time.perf_counter()
    try:
        with (OUTPUT / "stdout.log").open("x") as stdout, (OUTPUT / "stderr.log").open("x") as stderr:
            run = subprocess.run(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, timeout=300)
        status.update(exit_code=run.returncode,
                      status="passed" if run.returncode == 0 else "stopped_software_failure")
    except Exception as exc:
        status.update(status="stopped_software_failure", error=f"{type(exc).__name__}: {exc}")
    status["cpu_wall_seconds"] = time.perf_counter() - started
    save(OUTPUT / "status.json", status)
    with (OUTPUT / "SHA256SUMS").open("x") as handle:
        for path in sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
            handle.write(f"{sha(path)}  {path.relative_to(REPO).as_posix()}\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
