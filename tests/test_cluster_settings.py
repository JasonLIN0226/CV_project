"""Cluster settings live in one file and nowhere else.

torch/cluster.sh gives every setting a default, lets torch/.env (personal, ignored by git) replace
it, and lets an exported variable replace both. Nothing else under torch/, scripts/ or
configs/<environment>/ names an account or a personal path.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHOW = ('set -euo pipefail; source "$1/torch/cluster.sh"; '
        'echo "$BM_ACCOUNT|$BM_SCRATCH|$BM_REPO|$HF_HOME|$BM_CONDA_SH|$BM_CONDA_ENV|$BM_SBATCH_ARGS|$BM_GPU_TYPE|$BM_GPU_SBATCH_ARGS"')


def settings(checkout, **exported):
    env = {"PATH": "/usr/bin:/bin", "USER": "alice", **exported}
    out = subprocess.run(["bash", "-c", SHOW, "_", str(checkout)], capture_output=True, text=True, env=env, check=True).stdout.strip()
    return dict(zip(("account", "scratch", "repo", "hf_home", "conda_sh", "conda_env", "sbatch_args", "gpu_type", "gpu_args"), out.split("|")))


@pytest.fixture
def checkout(tmp_path):
    (tmp_path / "torch").mkdir()
    shutil.copy(ROOT / "torch/cluster.sh", tmp_path / "torch/cluster.sh")
    return tmp_path


def test_defaults_follow_the_cluster_user(checkout):
    s = settings(checkout)
    assert s == {"account": "", "scratch": "/scratch/alice/cv_project_runs", "repo": str(checkout.resolve()),
                 "hf_home": "/scratch/alice/hf_cache", "conda_sh": "/scratch/alice/miniconda3/etc/profile.d/conda.sh",
                 "conda_env": "/scratch/alice/miniconda3/envs/cvmem", "sbatch_args": "", "gpu_type": "", "gpu_args": ""}


def test_personal_file_replaces_defaults_and_exported_variables_replace_both(checkout):
    (checkout / "torch/.env").write_text("# mine\nBM_ACCOUNT=from_file\nBM_SCRATCH=/data/alice\nBM_SBATCH_ARGS=--partition=p1 --qos=q1\n\nnot a setting\n")
    s = settings(checkout)
    assert (s["account"], s["scratch"], s["sbatch_args"]) == ("from_file", "/data/alice", "--partition=p1 --qos=q1")
    assert s["hf_home"] == "/scratch/alice/hf_cache"                       # untouched settings keep their default
    s = settings(checkout, BM_ACCOUNT="from_env", BM_HF_HOME="/data/hf", BM_GPU_TYPE="l40s")
    assert (s["account"], s["scratch"], s["hf_home"], s["gpu_type"]) == ("from_env", "/data/alice", "/data/hf", "l40s")


def test_gpu_tasks_take_their_gpu_model_from_the_settings():
    """No generated job file fixes a GPU model; submit.sh adds --gres from BM_GPU_TYPE."""
    import yaml
    text = (ROOT / "torch/make_jobs.py").read_text()
    assert "#SBATCH --gres" not in text and "BM_GPU_TYPE" in text
    tasks = yaml.safe_load((ROOT / "configs/reproduction.yaml").read_text())["tasks"]
    for name, task in tasks.items():       # a task that asks for cuda must also ask for a GPU, and the other way round
        wants_cuda = "--device cuda" in task.get("args", "") or task.get("device") == "cuda"
        assert wants_cuda == bool(task["resources"].get("gpu")), name


@pytest.mark.parametrize("cpu, gpu, differing", [
    ("configs/synergy/release.yaml", "configs/synergy/gpu_features.yaml", {"train_features"}),
    ("configs/synergy/retrained_readers.yaml", "configs/synergy/retrained_readers_gpu.yaml", {"train_features", "readers"}),
    ("configs/gate/rebuilt.yaml", "configs/gate/rebuilt_gpu.yaml", {"root", "predictor_sha256", "students"}),
])
def test_gpu_configs_differ_from_their_cpu_twins_only_in_folders(cpu, gpu, differing):
    import yaml
    a, b = yaml.safe_load((ROOT / cpu).read_text()), yaml.safe_load((ROOT / gpu).read_text())
    assert {k for k in a["assets"] if a["assets"][k] != b["assets"][k]} == differing
    assert all(a[k] == b[k] for k in a if k != "assets") and a.keys() == b.keys()
