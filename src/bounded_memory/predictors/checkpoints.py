"""Checksum-verified loading of frozen predictors.

The historical loader (phase1/training.py::load_predictor) accepted a checkpoint only when a
signature built from its own source-file hashes matched. Code that has moved cannot reproduce
that signature, so this loader pins the checkpoint bytes instead, as the team's released
tools/load_predictor.py does. The original checkpoints hold pickled training state: nothing is
unpickled before its SHA256 matches the recorded value.
"""
from pathlib import Path

import numpy as np
import torch

from ..data.io import require_sha256
from .dynamics import DynamicsPredictor

BASIS_KEYS = ("input_mean", "projection", "target_mean", "output_basis")


def load_basis(path, sha256=None):
    if sha256 is not None:
        require_sha256(path, sha256, "predictor basis")
    with np.load(path, allow_pickle=False) as src:
        return {key: src[key] for key in BASIS_KEYS}


def load_dynamics_predictor(path, basis, *, sha256, seed, task, horizons, device="cpu"):
    """Load one frozen DynamicsPredictor; `sha256` is required and checked before deserialisation."""
    path = require_sha256(Path(path), sha256, "predictor checkpoint")
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["seed"] != seed or ck["task"] != task or tuple(ck["horizons"]) != tuple(horizons):
        raise ValueError("checkpoint metadata does not match the requested seed, task or horizons")
    model = DynamicsPredictor(ck["feature_dim"], ck["model_config"], ck["horizons"], basis).to(device)
    model.load_state_dict(ck["model"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("predictor contains nonfinite weights")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
