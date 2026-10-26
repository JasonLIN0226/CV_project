"""Filesystem/split configuration only; no reader or metric computation."""
import os
from pathlib import Path

REPO = Path(os.environ.get("QR_REPO", Path(__file__).resolve().parents[4])).resolve()
RUN = Path(os.environ.get("QR_RUN", REPO / "build/terminal-download-20261025/extracted/run"))
XFER = Path(os.environ.get("QR_XFER", REPO / "build/_claude_transfer_20261025"))
AGG = Path(os.environ.get("QR_AGG", REPO / "build/reader-aggregation-20261025/hpc/extracted/run"))
RECORDS = Path(os.environ.get("QR_RECORDS", RUN / "data/records.json"))
ACTIONS = Path(os.environ.get("QR_ACTIONS", RUN / "data/actions.npz"))
RGB_INDEX = Path(os.environ.get("QR_RGB_INDEX", RUN / "data/rgb_index.json"))
BASELINE_DECISIONS = Path(os.environ.get("QR_BASELINE_DECISIONS", RUN / "development/baseline_decisions.json"))
APPEARANCE_OUTPUT = Path(os.environ.get("QR_APPEARANCE_OUTPUT", REPO / "build/qualified-reader-conf-20261026/dev-preflight/appearance_all.npz"))
SPLIT = os.environ.get("QR_SPLIT", "development")
SPLITS = tuple(os.environ.get("QR_SPLITS", "train,validation,development").split(","))

