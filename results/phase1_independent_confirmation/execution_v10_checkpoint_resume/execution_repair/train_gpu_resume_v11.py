#!/usr/bin/env python3
"""Resume V10 training after the OOD-session SIGTERM without re-encoding."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["TZ"] = "UTC"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HOME"] = "/scratch/xl4358/phase1_hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONPATH"] = os.pathsep.join(
    (str(ROOT / "original_src"), str(ROOT))
)
sys.path[:0] = [str(ROOT), str(ROOT / "original_src"), str(ROOT / "execution_repair")]

import orchestrator as workflow
import fresh_training
import fresh_training_resume


attempt = ROOT / "stage_attempts" / "train_gpu.json"
if attempt.exists():
    raise SystemExit("Launcher failed to archive the previous train_gpu attempt")
failure = ROOT / "stage_failures" / "train_gpu.json"
if failure.exists():
    raise SystemExit("Previous train_gpu has an unreviewed application failure")

workflow.CHAIN = [
    "train_gpu",
    "lock_cpu",
    "encode_test",
    "evaluate_cpu",
    "audit_cpu",
    "evidence_cpu",
]
fresh_training.run_one_model = fresh_training_resume.run_one_model
workflow.main("train_gpu")
