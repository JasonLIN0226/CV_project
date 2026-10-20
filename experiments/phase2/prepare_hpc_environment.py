"""Prepare an isolated scratch environment on CPU; never submit a GPU job.

Bootstrap is Python 3.9 compatible because the course node exposes only 3.9.
uv/Python/packages come from PyPI, Astral and the official PyTorch wheel index.
Existing installations and the user's home configuration are not changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(root/'cache/uv'), UV_PYTHON_INSTALL_DIR=str(root/'python'),
               UV_PYTHON_BIN_DIR=str(root/'python-bin'), PIP_CACHE_DIR=str(root/'cache/pip'),
               XDG_CACHE_HOME=str(root/'cache/xdg'), HF_HOME=str(root/'cache/huggingface'))
    commands = []
    began = time.monotonic()
    def run(command, timeout=1200):
        print('RUN', command, flush=True)
        subprocess.run([str(a) for a in command], env=env, cwd=repo, check=True, timeout=timeout)
        commands.append([str(a) for a in command])
    bootstrap = root/'bootstrap'
    if not (bootstrap/'bin/python').exists():
        run([sys.executable, '-m', 'venv', bootstrap])
    run([bootstrap/'bin/python', '-m', 'pip', 'install', 'uv==0.9.11'])
    uv = bootstrap/'bin/uv'
    venv = root/'venv'
    if not (venv/'bin/python').exists():
        run([uv, 'venv', '--python', '3.11', '--seed', venv])
    py = venv/'bin/python'
    run([uv, 'pip', 'install', '--python', py, 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    run([uv, 'pip', 'install', '--python', py, '-r', repo/'requirements-phase2-hpc.txt'])
    run([uv, 'pip', 'install', '--python', py, '--no-deps', '-e', repo])
    run([py, '-m', 'pip', 'check'])
    run([py, '-c', 'import torch, torchvision, transformers, gym_pusht; from transformers import AutoImageProcessor, IJepaModel; print(torch.__version__, torchvision.__version__, transformers.__version__)'])
    packages = subprocess.check_output([str(py), '-m', 'pip', 'freeze'], env=env, text=True)
    (root/'environment.txt').write_text(packages)
    result = {'status':'ready', 'scope':'CPU dependency preparation only',
              'python':str(py), 'commands':commands, 'elapsed_seconds':time.monotonic()-began,
              'gpu_job_submitted':False, 'gpu_encoding_verified':False}
    (root/'environment_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    main()
