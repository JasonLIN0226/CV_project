"""The predictor input boundary: which frames may enter memory and how a batch is assembled.

`selected_evidence`, `memory_indices` and `collate` are extracted unchanged from
experiments/predictor/control/src/phase1/inputs.py. No labels, states or IDs enter the tensors.
"""
import numpy as np
import torch


def selected_evidence(c, rec, raw_evidence):
    """Fixed privileged timestamps; reads no hidden state, label, or future image.

    The wide motion policy uses the already visible frame at zero and the same
    pre-occlusion end frame.  It changes neither the legal history boundary nor
    the random/full/recent memory rules.
    """
    evidence = np.asarray(raw_evidence)
    if evidence.ndim != 1 or not np.issubdtype(evidence.dtype, np.integer):
        raise ValueError("Evidence timestamps must be a one-dimensional integer array")
    evidence = evidence.astype(np.int64, copy=True)
    stop = rec["query"]-c["data"]["recent"]+1
    if not len(evidence) or not np.all((evidence >= 0) & (evidence < stop)):
        raise ValueError("Evidence must precede recent buffer")
    if np.any(np.diff(evidence) <= 0):
        raise ValueError("Evidence timestamps must be distinct and increasing")
    experiment = c.get("experiment", {})
    wide = experiment.get("arm") in {"motion_wide_span", "confirm_candidate"}
    if rec["task"] == "motion":
        if len(evidence) != 2:
            raise ValueError("Motion evidence requires exactly two timestamps")
        if wide:
            evidence = np.array([0, evidence[-1]], np.int64)
            if evidence[-1] <= 0:
                raise ValueError("Wide motion evidence requires a positive end timestamp")
    return evidence


def memory_indices(q, recent, budget, evidence, condition, rng):
    stop = q-recent+1
    valid = np.arange(stop, dtype=np.int64)
    evidence = np.asarray(evidence, np.int64)
    if not np.all((evidence >= 0) & (evidence < stop)):
        raise ValueError("Evidence must precede recent buffer")
    if condition == "recent_only":
        return np.empty(0, np.int64)
    if condition == "full_history":
        return valid
    if len(valid) < budget:
        raise ValueError("Insufficient history: no duplication or crossing recent boundary")
    if condition in {"privileged_small_budget", "wrong_evidence"}:
        if len(evidence) > budget:
            raise ValueError("Evidence does not fit budget")
        filler = [int(i) for i in valid[::-1] if i not in evidence][:budget-len(evidence)]
        return np.sort(np.concatenate([evidence, filler])).astype(np.int64)
    if condition == "random_small_budget":
        return np.sort(rng.choice(valid, size=budget, replace=False))
    if condition == "random_subset":
        return np.sort(rng.choice(valid, size=int(rng.integers(0, len(valid)+1)), replace=False))
    raise ValueError(f"Unknown condition {condition}")


def collate(samples, device):
    # Padding is sanitized before linear layers; both encoder and cross attention mask it.
    n = max(len(s["z"]) for s in samples)
    batch = {}
    for key in ["z", "age", "type", "past_actions"]+(['projected_z'] if 'projected_z' in samples[0] else []):
        a = samples[0][key]
        out = np.zeros((len(samples), n)+a.shape[1:], dtype=a.dtype)
        for i, s in enumerate(samples):
            out[i, :len(s[key])] = s[key]
        batch[key] = torch.from_numpy(out).to(device)
    batch["mask"] = torch.tensor([[j < len(s["z"]) for j in range(n)] for s in samples], dtype=torch.bool, device=device)
    for key in ["actions", "current"]+(['projected_current'] if 'projected_current' in samples[0] else []):
        batch[key] = torch.from_numpy(np.stack([s[key] for s in samples])).to(device)
    return batch
