"""Conditional Gate training helper; never used by single-trajectory inference.

This module does not activate a candidate or modify the original predictor.
The caller must first satisfy the paired development diagnostic.  The teacher
is the current shared predictor's recent-only output, not a fixed pretrained
teacher or a true-future target.  Its stop-gradient blocks this loss's teacher
path; ordinary shared-parameter updates can still change it at the next step.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch


COMMON_ANCHOR_WEIGHT = 1.0
HISTORY_CONDITIONS = frozenset({"full_history", "privileged_small_budget", "random_small_budget"})
_REQUIRED_INPUTS = frozenset({"z", "age", "type", "past_actions", "actions", "current", "mask"})
_PROJECTED_INPUTS = frozenset({"projected_z", "projected_current"})


@dataclass
class TeacherForwardAccounting:
    """Additional forwards only; the caller records its ordinary training cost."""
    extra_forward_calls: int = 0
    extra_forward_examples: int = 0


def common_anchor_loss(history_prediction, recent_teacher, rows, *, condition, task="gate"):
    """Fixed-lambda common-mode MSE over complete adjacent TRAIN sibling pairs.

    Predictions have shape (episodes, horizons, normalized feature dimensions).
    Metadata is used only to validate the pair boundary and never enters the
    model.  Both recent predictions must be identical for the audited shared
    context.  No target, hidden state, future RGB, or evaluation pair is read.
    """
    if task != "gate" or condition not in HISTORY_CONDITIONS:
        raise ValueError("Common anchor requires gate training and a history condition")
    if not torch.is_tensor(history_prediction) or not torch.is_tensor(recent_teacher):
        raise ValueError("Common anchor requires prediction tensors")
    shape = history_prediction.shape
    if (len(shape) != 3 or any(size == 0 for size in shape) or shape[0] % 2
            or recent_teacher.shape != shape or not history_prediction.is_floating_point()
            or recent_teacher.dtype != history_prediction.dtype
            or recent_teacher.device != history_prediction.device):
        raise ValueError("Common anchor requires matching finite (even episodes, horizons, features) tensors")
    if len(rows) != shape[0]:
        raise ValueError("Common anchor requires one metadata row per prediction")
    seen_ids, seen_pairs = set(), set()
    for a, b in zip(rows[::2], rows[1::2]):
        if (a["task"] != task or b["task"] != task or a["split"] != "train" or b["split"] != "train"
                or a["pair"] != b["pair"] or a["id"] == b["id"]
                or a["sibling"] != b["id"] or b["sibling"] != a["id"]
                or a["id"] in seen_ids or b["id"] in seen_ids or a["pair"] in seen_pairs):
            raise ValueError("Common anchor requires distinct complete adjacent gate TRAIN sibling pairs")
        seen_ids.update((a["id"], b["id"]))
        seen_pairs.add(a["pair"])
    teacher = recent_teacher.detach()
    if not torch.isfinite(history_prediction).all() or not torch.isfinite(teacher).all():
        raise ValueError("Non-finite common anchor prediction")
    if not torch.equal(teacher[::2], teacher[1::2]):
        raise ValueError("Common anchor requires identical sibling recent-only predictions")
    common = (history_prediction[::2] + history_prediction[1::2]) / 2
    return COMMON_ANCHOR_WEIGHT * (common - teacher[::2]).square().mean()


def recent_only_teacher_batch(history_batch):
    """Construct legal recent-only tensors without sampling or reading targets.

    History tokens are physically removed.  A single trajectory is supported;
    sibling metadata is not needed for this input transform or for inference.
    Planned actions remain legal predictor inputs.  Unknown keys (including
    targets or hidden states) and future-dated visual tokens are rejected.
    """
    keys = frozenset(history_batch)
    if (not _REQUIRED_INPUTS <= keys or keys - (_REQUIRED_INPUTS | _PROJECTED_INPUTS)
            or bool(keys & _PROJECTED_INPUTS) != (_PROJECTED_INPUTS <= keys)):
        raise ValueError("Recent teacher accepts only the predictor input whitelist")
    if not all(torch.is_tensor(value) for value in history_batch.values()):
        raise ValueError("Recent teacher inputs must be tensors")
    z, mask = history_batch["z"], history_batch["mask"]
    if z.ndim != 3 or not z.is_floating_point() or z.shape[0] == 0 or mask.dtype != torch.bool or mask.shape != z.shape[:2]:
        raise ValueError("Invalid recent teacher visual/mask shape")
    n, tokens, dim = z.shape
    if any(value.device != z.device for value in history_batch.values()):
        raise ValueError("Recent teacher inputs must share one device")
    if (history_batch["age"].shape != (n, tokens) or history_batch["type"].shape != (n, tokens)
            or history_batch["past_actions"].shape != (n, tokens, 2)
            or history_batch["current"].shape != (n, dim)
            or history_batch["actions"].ndim != 3 or history_batch["actions"].shape[0] != n
            or history_batch["actions"].shape[1] == 0 or history_batch["actions"].shape[2] != 2):
        raise ValueError("Invalid recent teacher predictor input shape")
    age, token_type = history_batch["age"], history_batch["type"]
    if (not age.is_floating_point() or token_type.dtype not in {torch.int32, torch.int64}
            or not torch.isfinite(age[mask]).all() or (age[mask] < 0).any()
            or not ((token_type[mask] == 0) | (token_type[mask] == 1)).all()):
        raise ValueError("Recent teacher requires legal past visual timestamps and token types")
    for key in ("z", "past_actions"):
        if not torch.isfinite(history_batch[key][mask]).all():
            raise ValueError("Non-finite recent teacher observation")
    for key in ("actions", "current"):
        if not history_batch[key].is_floating_point() or not torch.isfinite(history_batch[key]).all():
            raise ValueError("Non-finite recent teacher action/current input")
    recent = mask & (token_type == 0)
    counts = recent.sum(1)
    if (counts == 0).any() or not (counts == counts[0]).all():
        raise ValueError("Recent teacher requires the same nonempty recent buffer per trajectory")
    indices = [torch.where(row)[0] for row in recent]
    length = int(counts[0])
    expected = torch.arange(length-1, -1, -1, device=z.device, dtype=age.dtype)
    for i, index in enumerate(indices):
        if (not torch.equal(age[i, index], expected)
                or (age[i, mask[i] & (token_type[i] == 1)] < length).any()
                or not torch.equal(history_batch["current"][i], z[i, index[-1]])):
            raise ValueError("Recent teacher requires contiguous past recent frames and matching current")
    if keys & _PROJECTED_INPUTS:
        p, pc = history_batch["projected_z"], history_batch["projected_current"]
        if (p.ndim != 3 or p.shape[:2] != (n, tokens) or pc.shape != (n, p.shape[-1])
                or not torch.isfinite(p[mask]).all() or not torch.isfinite(pc).all()
                or any(not torch.equal(pc[i], p[i, index[-1]]) for i, index in enumerate(indices))):
            raise ValueError("Invalid recent teacher projected inputs")
    result = {key: torch.stack([history_batch[key][i, index] for i, index in enumerate(indices)]).detach()
              for key in ("z", "age", "type", "past_actions", "projected_z") if key in keys}
    result["mask"] = torch.ones((n, length), dtype=torch.bool, device=z.device)
    for key in ("actions", "current", "projected_current"):
        if key in keys:
            result[key] = history_batch[key].detach()
    return result


def recent_teacher_prediction(model, history_batch, *, accounting=None):
    """One current-parameter, no-grad, eval-mode teacher forward per call.

    Every module's prior training flag and Python/NumPy/Torch RNG state is
    restored, including on exceptions.  No cached/stale prediction is used.
    The existing DynamicsPredictor has no mutable eval-time running statistics.
    """
    batch = recent_only_teacher_batch(history_batch)
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    tensors = list(batch.values()) + list(model.parameters()) + list(model.buffers())
    cuda_devices = sorted({value.device.index for value in tensors if value.device.type == "cuda"})
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            with torch.no_grad():
                if accounting is not None:
                    accounting.extra_forward_calls += 1
                    accounting.extra_forward_examples += batch["z"].shape[0]
                prediction = model(batch)
                if not torch.is_tensor(prediction) or prediction.ndim != 3 or prediction.shape[0] != batch["z"].shape[0]:
                    raise ValueError("Recent teacher must return an episode/horizon/feature tensor")
                if not torch.isfinite(prediction).all():
                    raise ValueError("Non-finite recent teacher prediction")
                return prediction.detach()
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, training in modes:
            module.training = training
