"""Observe one Gate validation replay without changing its retention decisions.

Scores are deletion logits, not teacher losses or probabilities. Cue annotations
are attached after replay and never passed to the online selector. As in rollout,
this audits logical B=2 retention, not the physical memory used by offline replay.
"""
from __future__ import annotations

import numpy as np
import torch

from ..environments.gate.protocol import HORIZONS, R, predict_selection
from ..methods.deletion_cost.retention import rollout
from ..methods.deletion_cost.scorer import score_update
from .metrics.gate import metric_rows

POLICY = "deletion_cost_student"


@torch.inference_mode()
def trace_episode(assets, episode, scorer, predictor, seed):
    """Return JSON-ready decisions, predictions and metrics for one validation episode.

The caller supplies the same loaded assets/models as the ordinary Gate evaluation.
Only observed prefixes reach rollout; future targets and cue metadata are used
after selection. The temporary forward hook returns nothing and is always removed.
"""
    if episode.split != "validation":
        raise ValueError("gate trace accepts validation episodes only")
    if episode.q < R:
        raise ValueError("query must include four recent frames and eligible history")
    if tuple(predictor.horizons) != HORIZONS:
        raise ValueError("Gate predictor must use horizons (3, 6)")

    captured, scores_by_clock = [], {}

    def capture(_model, _inputs, output):
        captured.append(output.detach().cpu().numpy().copy())

    def update(state):
        captured.clear()
        selected = score_update(scorer, state)
        if len(captured) != 1 or captured[0].shape != (1, 3):
            raise ValueError("expected exactly one [1, 3] scorer output per eviction")
        scores_by_clock[state.clock] = captured[0][0].tolist()
        return selected

    handle = scorer.register_forward_hook(capture)
    try:
        kept, trace = rollout(
            episode.p[:episode.q + 1].copy(), episode.actions[:episode.q].copy(),
            episode.q, update,
        )
    finally:
        handle.remove()

    # Evaluation-only metadata is deliberately added after every decision is made.
    for row in trace:
        pool = row["previous"] + [row["arrival"]]
        removed = [t for t in pool if t not in row["selected"]]
        row.update(
            recent=list(range(row["clock"] - R + 1, row["clock"] + 1)),
            candidates=pool,
            deletion_logits=scores_by_clock.get(row["clock"]),
            removed=removed[0] if removed else None,
            cue_candidates=[t for t in pool if episode.cue[0] <= t <= episode.cue[1]],
            cue_kept=[t for t in row["selected"] if episode.cue[0] <= t <= episode.cue[1]],
        )

    predictor.eval()
    prediction = predict_selection(predictor, [episode], [kept], assets.device)
    if prediction.shape != (1, len(HORIZONS), episode.z.shape[1]) or not np.isfinite(prediction).all():
        raise ValueError("invalid single-episode Gate prediction")
    rows = metric_rows(assets, [episode], prediction, [kept], POLICY, seed)
    decoded = np.asarray(assets.decoder.predict(prediction[0]))
    if decoded.ndim != 2 or decoded.shape[0] != len(HORIZONS) or decoded.shape[1] < 1 \
            or not np.isfinite(decoded).all():
        raise ValueError("invalid Gate route decoder output")
    routes = [
        {"horizon": h, "target_clock": int(episode.q + h),
         "decoder_score": float(decoded[i, 0]),
         "predicted_route_positive": bool(decoded[i, 0] > 0),
         "actual_route_positive": bool(episode.classes[episode.q + h] > 0)}
        for i, h in enumerate(HORIZONS)
    ]
    return {
        "schema_version": 1, "environment": "gate", "policy": POLICY,
        "episode": {"id": episode.id, "pair": episode.pair, "split": episode.split,
                    "query": int(episode.q), "cue_interval": [int(t) for t in episode.cue]},
        "trace": trace, "kept": kept,
        "prediction_input_frames": list(range(episode.q - R + 1, episode.q + 1)) + kept,
        "horizons": list(HORIZONS), "prediction": prediction[0].tolist(),
        "routes": routes, "rows": rows,
    }


def format_trace(result):
    """Compact text view; full-precision logits and latents remain in JSON."""
    e = result["episode"]
    lines = [
        f"Gate validation episode {e['id']} | query={e['query']} | cue={e['cue_interval']}",
        "Deletion logits: largest is removed; exact ties remove the oldest frame.",
        "Cue columns are evaluation-only annotations added after selection.",
        "clock | recent | candidates | deletion logits | removed | kept | cue kept",
    ]
    for row in result["trace"]:
        scores = row["deletion_logits"]
        score_text = "not scored (filling slots)" if scores is None else ", ".join(
            f"{t}:{s:.6g}" for t, s in zip(row["candidates"], scores))
        lines.append(f"{row['clock']} | {row['recent']} | {row['candidates']} | {score_text} | "
                     f"{row['removed']} | {row['selected']} | {row['cue_kept']}")
    lines.append(f"Prediction input frames: {result['prediction_input_frames']}")
    lines.append("Route positive means decoder/true class > 0; outputs are latent features, not video.")
    for route, metrics in zip(result["routes"], result["rows"]):
        lines.append(f"+{route['horizon']} (frame {route['target_clock']}): "
                     f"predicted_positive={route['predicted_route_positive']}, "
                     f"actual_positive={route['actual_route_positive']}, "
                     f"correct={bool(metrics['route_accuracy'])}, "
                     f"normalized_mse={metrics['normalized_mse']:.8g}")
    return "\n".join(lines)
