"""torch/submit_pack.sh: one GPU job on Torch that runs many tasks, with stand-ins for sbatch and conda."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = "import sys, pathlib\npathlib.Path(sys.argv[1]).write_text(pathlib.Path.cwd().name)\nsys.exit(int(sys.argv[2]))\n"


@pytest.fixture
def cluster(tmp_path):
    """A small checkout, a conda environment and an sbatch that runs the job at once."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/run_tasks.py", repo / "scripts/run_tasks.py")
    shutil.copytree(ROOT / "torch", repo / "torch", ignore=shutil.ignore_patterns(".env", "generated", "__pycache__"))
    (repo / "src").mkdir(); (repo / "configs").mkdir()
    (repo / "scripts/work.py").write_text(WORK)
    env_bin = tmp_path / "conda/envs/cvmem/bin"
    env_bin.mkdir(parents=True)
    (env_bin / "python").symlink_to(sys.executable)
    (tmp_path / "conda/conda.sh").write_text("conda() { :; }\n")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").symlink_to(sys.executable)
    sbatch = tmp_path / "bin/sbatch"                 # records the options, then runs the job script like Slurm would
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$RECORD/args"\nSLURM_JOB_ID=77 bash "${@: -1}" > "$RECORD/job.log" 2>&1\n'
                      'echo $? > "$RECORD/job.exit"\necho 77\n')
    sbatch.chmod(0o755)
    (tmp_path / "record").mkdir()
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "USER": "alice", "HOME": str(tmp_path), "RECORD": str(tmp_path / "record"),
           "BM_SCRATCH": str(tmp_path / "scratch"), "BM_HF_HOME": str(tmp_path / "hf"), "BM_CONDA_SH": str(tmp_path / "conda/conda.sh"),
           "BM_CONDA_ENV": str(tmp_path / "conda/envs/cvmem"), "BM_POLL_SECONDS": "0.05"}
    return repo, env


def submit(cluster, exit_code=0, **more):
    repo, env = cluster
    tmp = repo.parent
    (tmp / "manifest.yaml").write_text(yaml.safe_dump({"tasks": {"x.a": {
        "script": "scripts/work.py", "args": f"{tmp / 'ran'} {exit_code}", "resources": {"cpus": 1, "mem": "1G", "time": "00:01:00"}}}}))
    done = subprocess.run(["bash", str(repo / "torch/submit_pack.sh"), "--run-id", "r1", "--select", "x", "--manifest", str(tmp / "manifest.yaml")],
                          env=dict(env, **more), capture_output=True, text=True, timeout=300)
    record = tmp / "record"
    return done, (record / "args").read_text().split("\n") if (record / "args").exists() else None, record


def test_one_gpu_job_with_the_personal_account_and_gpu_model(cluster):
    done, args, record = submit(cluster, BM_ACCOUNT="my_account", BM_GPU_TYPE="l40s", BM_SBATCH_ARGS="--qos=q1")
    assert done.returncode == 0, done.stderr
    for wanted in ("--account=my_account", "--qos=q1", "--gres=gpu:l40s:1", "--cpus-per-task=8", "--mem=64G", "--time=08:00:00"):
        assert wanted in args, wanted
    assert "pack\t77\t" in done.stdout and "runs/r1/pack_status.txt" in done.stdout


def test_defaults_ask_for_any_gpu_and_no_account(cluster):
    done, args, _ = submit(cluster)
    assert done.returncode == 0, done.stderr
    assert "--gres=gpu:1" in args and not any(a.startswith("--account") for a in args)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_the_job_runs_from_the_code_snapshot_and_returns_the_exit_code(cluster, exit_code):
    repo, _ = cluster
    done, _, record = submit(cluster, exit_code=exit_code)
    assert done.returncode == 0, done.stderr
    assert (record / "job.exit").read_text().strip() == str(exit_code), (record / "job.log").read_text()
    assert (repo.parent / "ran").read_text().startswith("code_")            # the task ran inside the snapshot
    assert "x.a[0]" in (repo / "runs/r1/pack_status.txt").read_text()       # the status is in the checkout, which Mutagen syncs


def test_it_refuses_without_a_run_id_or_an_environment(cluster):
    repo, env = cluster
    assert subprocess.run(["bash", str(repo / "torch/submit_pack.sh"), "--select", "x"], env=env, capture_output=True).returncode == 1
    done = subprocess.run(["bash", str(repo / "torch/submit_pack.sh"), "--run-id", "r1", "--select", "x"],
                          env=dict(env, BM_CONDA_ENV="/nowhere"), capture_output=True, text=True)
    assert done.returncode == 1 and "submit_setup_env.sh" in done.stderr
    assert not (repo.parent / "record/args").exists()
