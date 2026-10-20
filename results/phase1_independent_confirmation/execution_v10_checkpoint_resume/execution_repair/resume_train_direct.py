#!/usr/bin/env python3
"""Resume the frozen V10 training stage after the verified resume_cpu proof.

The original stage chain failed only while submitting the next stage: its GPU
pre-submit guard looked for ``fresh/sequence_budget.json`` under the execution
root instead of the data root recorded in ``config.json``.  This wrapper changes
only the stage-chain position used for the already verified resume boundary.
"""
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
sys.path[:0] = [str(ROOT), str(ROOT / "original_src")]

import orchestrator as workflow

workflow.CHAIN = [
    "train_gpu",
    "lock_cpu",
    "encode_test",
    "evaluate_cpu",
    "audit_cpu",
    "evidence_cpu",
]
workflow.main("train_gpu")
