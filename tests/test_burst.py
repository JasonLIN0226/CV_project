"""The scripts in burst/ (course cluster), run here with stand-ins for sbatch and conda.

They have not run on the course cluster yet. These tests check what can be checked without it:
the settings file, the sbatch call that submit.sh makes, that job.sbatch runs the tasks and
returns their exit code, how the driver version is read, and the result archive.
"""
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("BM_ACCOUNT", "BM_GPU_PARTITION", "BM_GPU_GRES", "BM_CPUS", "BM_MEM", "BM_TIME", "BM_PARALLEL", "BM_SCRATCH", "BM_REPO",
         "HF_HOME", "BM_CONDA_SH", "BM_CONDA_ENV", "BM_SBATCH_ARGS")


@pytest.fixture
def checkout(tmp_path):
    """A checkout in a temporary folder: burst/ copied, the code folders linked."""
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "burst", repo / "burst", ignore=shutil.ignore_patterns(".env"))
    for name in ("scripts", "torch", "src", "configs"):
        (repo / name).symlink_to(ROOT / name)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").symlink_to(sys.executable)
    return repo


def base_env(checkout, **more):
    tmp = checkout.parent
    return {"PATH": f"{tmp / 'bin'}:/usr/bin:/bin", "USER": "alice", "HOME": str(tmp / "home"), **more}


def settings(checkout, **exported):
    show = 'set -euo pipefail; source "$1/burst/cluster.sh"; ' + 'echo "' + "|".join(f"${n}" for n in NAMES) + '"'
    out = subprocess.run(["bash", "-c", show, "_", str(checkout)], capture_output=True, text=True, env=base_env(checkout, **exported), check=True)
    return dict(zip(NAMES, out.stdout.strip().split("|")))


def test_defaults_are_the_course_account_and_one_a100(checkout):
    assert settings(checkout) == {
        "BM_ACCOUNT": "csci_ga_2271-2026fa", "BM_GPU_PARTITION": "c12m85-a100-1", "BM_GPU_GRES": "gpu:1", "BM_CPUS": "12", "BM_MEM": "80G",
        "BM_TIME": "05:00:00", "BM_PARALLEL": "3", "BM_SCRATCH": "/scratch/alice/cv_project_runs", "BM_REPO": str(checkout.resolve()),
        "HF_HOME": "/scratch/alice/hf_cache", "BM_CONDA_SH": "/scratch/alice/miniforge3/etc/profile.d/conda.sh",
        "BM_CONDA_ENV": "/scratch/alice/miniforge3/envs/cvmem", "BM_SBATCH_ARGS": ""}


def test_personal_file_replaces_defaults_and_exported_variables_replace_both(checkout):
    (checkout / "burst/.env").write_text("# mine\nBM_GPU_PARTITION=g2-standard-12\nBM_SCRATCH=/data/alice\nBM_PARALLEL=2\n")
    s = settings(checkout)
    assert (s["BM_GPU_PARTITION"], s["BM_SCRATCH"], s["BM_PARALLEL"], s["BM_MEM"]) == ("g2-standard-12", "/data/alice", "2", "80G")
    s = settings(checkout, BM_PARALLEL="4", BM_MEM="60G")
    assert (s["BM_GPU_PARTITION"], s["BM_PARALLEL"], s["BM_MEM"]) == ("g2-standard-12", "4", "60G")


def stand_ins(checkout):
    """A conda environment and an sbatch that only record what they were given."""
    tmp = checkout.parent
    env_dir = tmp / "conda/envs/cvmem/bin"
    env_dir.mkdir(parents=True)
    (env_dir / "python").symlink_to(sys.executable)
    (tmp / "conda/conda.sh").write_text("conda() { :; }\n")
    sbatch = tmp / "bin/sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$RECORD/args"\nprintf "%s" "$BM_PACK_ARGS" > "$RECORD/pack_args"\necho 4242\n')
    sbatch.chmod(0o755)
    (tmp / "record").mkdir()
    return base_env(checkout, RECORD=str(tmp / "record"), BM_SCRATCH=str(tmp / "scratch"), BM_HF_HOME=str(tmp / "hf"),
                    BM_CONDA_SH=str(tmp / "conda/conda.sh"), BM_CONDA_ENV=str(tmp / "conda/envs/cvmem"))


def test_submit_asks_for_one_gpu_job_and_passes_the_arguments_through(checkout):
    env = stand_ins(checkout)
    done = subprocess.run(["bash", str(checkout / "burst/submit.sh"), "--run-id", "r1", "--select", "gate.retrain_gpu", "synergy.retrain_gpu",
                           "--skip", "gate.retrain_gpu.compare_cpu", "--parallel", "2"], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    record = checkout.parent / "record"
    args = (record / "args").read_text().split("\n")
    for wanted in ("--account=csci_ga_2271-2026fa", "--partition=c12m85-a100-1", "--gres=gpu:1", "--cpus-per-task=12", "--mem=80G",
                   "--time=05:00:00", "--export=ALL", str(checkout / "burst/job.sbatch")):
        assert wanted in args, wanted
    # exactly what was typed: the runner takes --parallel from the command line, else from BM_PARALLEL
    assert (record / "pack_args").read_text() == "--run-id r1 --select gate.retrain_gpu synergy.retrain_gpu --skip gate.retrain_gpu.compare_cpu --parallel 2"
    assert "submitted job 4242" in done.stdout and "runs/r1/pack_status.txt" in done.stdout


def test_submit_refuses_without_a_run_id_or_an_environment(checkout):
    env = stand_ins(checkout)
    assert subprocess.run(["bash", str(checkout / "burst/submit.sh"), "--select", "gate"], env=env, capture_output=True).returncode == 1
    done = subprocess.run(["bash", str(checkout / "burst/submit.sh"), "--run-id", "r1", "--select", "gate"],
                          env=dict(env, BM_CONDA_ENV="/nowhere"), capture_output=True, text=True)
    assert done.returncode == 1 and "burst/setup.sh" in done.stderr
    assert not (checkout.parent / "record/args").exists()


WORK = "import sys, pathlib\npathlib.Path(sys.argv[1]).write_text('ran')\nsys.exit(int(sys.argv[2]))\n"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_the_job_runs_the_selected_tasks_and_returns_their_exit_code(checkout, exit_code):
    env = stand_ins(checkout)
    tmp = checkout.parent
    (checkout / "work.py").write_text(WORK)
    (tmp / "manifest.yaml").write_text(yaml.safe_dump({"tasks": {"x.a": {
        "script": "work.py", "args": f"{tmp / 'ran'} {exit_code}", "resources": {"cpus": 1, "mem": "1G", "time": "00:01:00"}}}}))
    env.update(BM_PACK_ARGS=f"--run-id r1 --select x --manifest {tmp / 'manifest.yaml'}", SLURM_JOB_ID="1", BM_POLL_SECONDS="0.05")
    script = f'source "{checkout}/burst/cluster.sh"; bash "{checkout}/burst/job.sbatch"'
    done = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == exit_code, done.stdout + done.stderr
    assert (tmp / "ran").read_text() == "ran"
    assert "x.a[0]" in (checkout / "runs/r1/pack_status.txt").read_text()
    assert f"exit code {exit_code}" in done.stdout


@pytest.mark.parametrize("line, version", [
    ("NVRM version: NVIDIA UNIX x86_64 Kernel Module  610.43.02  Thu Oct  1 00:00:00 UTC 2026", "610.43.02"),
    ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  575.57.08  Release Build  (dvs-builder@U22)", "575.57.08"),
])
def test_the_driver_folder_is_named_after_the_kernel_module_version(checkout, line, version):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this machine has a working GPU; nothing to unpack")
    tmp = checkout.parent
    (tmp / "version").write_text(line + "\nGCC version:  gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04)\n")
    (tmp / "home" / f"nvidia-driver-{version}").mkdir(parents=True)          # already unpacked: nothing is downloaded
    script = f'source "{checkout}/burst/gpu_driver.sh"; echo "LD=$LD_LIBRARY_PATH"'
    done = subprocess.run(["bash", "-c", script], env=base_env(checkout, BM_NVIDIA_VERSION_FILE=str(tmp / "version")), capture_output=True, text=True)
    assert f"LD={tmp / 'home'}/nvidia-driver-{version}" in done.stdout, done.stdout + done.stderr


def test_pack_results_archives_the_run_folder(checkout):
    env = stand_ins(checkout)
    (checkout / "runs/r1/aggregate").mkdir(parents=True)
    (checkout / "runs/r1/aggregate/tables.md").write_text("table\n")
    (checkout.parent / "home").mkdir(exist_ok=True)
    done = subprocess.run(["bash", str(checkout / "burst/pack_results.sh"), "r1"], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(checkout.parent / "home/r1_results.tar.gz") as archive:
        assert "runs/r1/aggregate/tables.md" in archive.getnames()
    assert subprocess.run(["bash", str(checkout / "burst/pack_results.sh"), "missing"], env=env, capture_output=True).returncode == 1


def test_no_script_in_burst_names_a_person(checkout):
    for path in sorted((ROOT / "burst").iterdir()):
        if path.name != ".env":
            text = path.read_text()
            assert "ys4907" not in text and "torch_pr_" not in text, path.name
