"""Preview renderer for the structured arrow-plus-line retention audit.

This module is deliberately preview-only.  It does not materialize a dataset,
load a model, or define a scientific result.  The purpose is to make the
visual cue-identification rule reviewable before any GPU work.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SOURCE_DIR = Path(__file__).resolve().parents[1] / "retention_position_noise_audit"
CONTROL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "predictor"
    / "control"
    / "src"
)
for source in (CONTROL_SOURCE, SOURCE_DIR):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from design import paired_noisy_episodes  # noqa: E402
from phase1.utils import stable_seed  # noqa: E402


BACKGROUND = np.asarray((24, 29, 38), dtype=np.uint8)
ARENA_BACKGROUND = np.asarray((44, 49, 60), dtype=np.uint8)
PANEL_BACKGROUND = (70, 79, 92)
STRUCTURED_LINE = (190, 194, 200)
FULL_IMAGE_LINE_PLACEMENT = "uniform_center_and_length_over_complete_image_v2"

ARENA_BOX = (0.055, 0.255, 0.945, 0.775)
LINE_BOX = (0.02, 0.02, 0.98, 0.98)
ARROW_PANEL = (0.31, 0.04, 0.69, 0.22)


@dataclass(frozen=True)
class StructuredFrame:
    rgb: np.ndarray
    is_cue: bool
    vertical_line_count: int
    line_mask: np.ndarray


def render_structured_siblings(
    rgb_by_label: dict[int, np.ndarray],
    *,
    scene_seed: int,
    frame_index: int,
    cue_start: int,
    cue_end: int,
    include_vertical_lines: bool = True,
    full_image_lines: bool = False,
) -> dict[int, StructuredFrame]:
    """Render a matched sibling frame with one shared nuisance-line mask.

    The mask is intersected across both siblings' arena backgrounds so the
    visible line stream is bit-identical while cue, track, ball, and distractor
    pixels stay unchanged.
    """
    labels = sorted(int(label) for label in rgb_by_label)
    if labels != [0, 1]:
        raise ValueError("A matched sibling frame requires labels 0 and 1")
    frames = {label: np.asarray(rgb_by_label[label], dtype=np.uint8) for label in labels}
    if include_vertical_lines:
        line_mask, _, line_count = _vertical_line_layer(
            frames[0],
            scene_seed=scene_seed,
            frame_index=frame_index,
            sibling_rgbs=[frames[0], frames[1]],
            full_image_lines=full_image_lines,
        )
    else:
        line_mask = np.zeros(frames[0].shape[:2], dtype=bool)
        line_count = 0
    return {
        label: render_structured_frame(
            frames[label],
            scene_seed=scene_seed,
            frame_index=frame_index,
            cue_start=cue_start,
            cue_end=cue_end,
            label=label,
            include_vertical_lines=include_vertical_lines,
            full_image_lines=full_image_lines,
            shared_line_mask=line_mask,
            shared_line_count=line_count,
        )
        for label in labels
    }


def _box(values: tuple[float, float, float, float], size: int) -> tuple[int, ...]:
    return tuple(int(round(value * size)) for value in values)


def structured_line_width(size: int) -> int:
    """Shared shaft and nuisance-line width."""
    return max(1, int(size) // 36)


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    tip: tuple[float, float],
    previous: tuple[float, float],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a compact triangular head whose apex is ``tip``."""
    tx, ty = tip
    px, py = previous
    vx, vy = tx - px, ty - py
    norm = max((vx * vx + vy * vy) ** 0.5, 1e-9)
    ux, uy = vx / norm, vy / norm
    # Perpendicular unit vector.
    qx, qy = -uy, ux
    length = 0.048
    half_width = 0.028
    base_x, base_y = tx - length * ux, ty - length * uy
    left = (base_x + half_width * qx, base_y + half_width * qy)
    right = (base_x - half_width * qx, base_y - half_width * qy)
    points = [
        (int(round(tx * size)), int(round(ty * size))),
        (int(round(left[0] * size)), int(round(left[1] * size))),
        (int(round(right[0] * size)), int(round(right[1] * size))),
    ]
    draw.polygon(points, fill=color)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    size: int,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    draw.line(
        (*_box((start[0], start[1], end[0], end[1]), size),),
        fill=color,
        width=width,
    )
    _arrow_head(draw, end, start, size, color)


def _draw_fixed_arrow_panel(
    draw: ImageDraw.ImageDraw,
    size: int,
    *,
    is_cue: bool,
    label: int,
) -> None:
    """Draw two parallel arrows whose direction carries the retained cue.

    Neutral frames show the same two right-pointing arrows as every sibling.
    During evidence, both arrows rotate together: up for label 0 or down for
    label 1.  Object presence never changes, so the task requires remembering
    a local directional event rather than merely detecting an added object.
    """
    draw.rectangle(
        _box(ARROW_PANEL, size),
        fill=PANEL_BACKGROUND,
        outline=(105, 113, 127),
        width=max(1, size // 64),
    )
    if not is_cue:
        for y in (0.105, 0.158):
            _arrow(
                draw,
                (0.365, y),
                (0.635, y),
                size,
                color=STRUCTURED_LINE,
                width=structured_line_width(size),
            )
        return

    # Both arrows rotate as a parallel pair.  Keeping the shafts vertical
    # makes the event legible at the 96-pixel source resolution.
    start_y, end_y = (0.182, 0.078) if label == 0 else (0.078, 0.182)
    for x in (0.425, 0.575):
        _arrow(
            draw,
            (x, start_y),
            (x, end_y),
            size,
            color=STRUCTURED_LINE,
            width=structured_line_width(size),
        )


def _vertical_line_layer(
    rgb: np.ndarray,
    *,
    scene_seed: int,
    frame_index: int,
    sibling_rgbs: list[np.ndarray] | None = None,
    full_image_lines: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Render deterministic nuisance lines over legal background pixels.

    ``full_image_lines`` uses independently sampled line centers and lengths
    over the complete image.  Lines therefore occur in the top, middle, and
    bottom bands instead of being dominated by long segments that start in the
    arena and run toward the bottom edge.
    """
    size = int(rgb.shape[0])
    rng = np.random.default_rng(
        stable_seed(scene_seed, frame_index, "structured-vertical-lines-v1")
    )
    line_count = int(rng.integers(7, 15))
    mask_image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_image)
    left, top, right, bottom = LINE_BOX
    for _ in range(line_count):
        x = float(rng.uniform(left, right))
        if full_image_lines:
            center_y = float(rng.uniform(top, bottom))
            half_length = float(rng.uniform(0.05, 0.15))
            y1 = max(top, center_y - half_length)
            y2 = min(bottom, center_y + half_length)
        else:
            y1 = float(rng.uniform(top, bottom - 0.10))
            y2 = float(rng.uniform(y1 + 0.08, bottom))
        draw.line(
            _box((x, y1, x, y2), size),
            fill=255,
            width=structured_line_width(size),
        )
    line_mask = np.asarray(mask_image, dtype=np.uint8) > 0
    compared = sibling_rgbs if sibling_rgbs is not None else [rgb]
    scene_background: np.ndarray | None = None
    for item in compared:
        item_background = (
            np.all(item == BACKGROUND, axis=2)
            | np.all(item == ARENA_BACKGROUND, axis=2)
        )
        scene_background = (
            item_background
            if scene_background is None
            else scene_background & item_background
        )
    assert scene_background is not None
    allowed = line_mask & scene_background
    panel_x0, panel_y0, panel_x1, panel_y1 = _box(ARROW_PANEL, size)
    allowed[panel_y0:panel_y1, panel_x0:panel_x1] = False
    return allowed, scene_background, line_count


def render_structured_frame(
    rgb: np.ndarray,
    *,
    scene_seed: int,
    frame_index: int,
    cue_start: int,
    cue_end: int,
    label: int,
    include_vertical_lines: bool = True,
    full_image_lines: bool = False,
    shared_line_mask: np.ndarray | None = None,
    shared_line_count: int | None = None,
) -> StructuredFrame:
    """Replace the original cue panel and add deterministic nuisance lines."""
    base = np.asarray(rgb, dtype=np.uint8)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB frame")
    if include_vertical_lines and (shared_line_mask is None) != (shared_line_count is None):
        raise ValueError("Shared line mask and count must be supplied together")
    if not include_vertical_lines:
        line_pixels = np.zeros(base.shape[:2], dtype=bool)
        line_count = 0
    elif shared_line_mask is None:
        line_pixels, _, line_count = _vertical_line_layer(
            base,
            scene_seed=scene_seed,
            frame_index=frame_index,
            full_image_lines=full_image_lines,
        )
    else:
        line_pixels = np.asarray(shared_line_mask, dtype=bool)
        line_count = int(shared_line_count)
        if line_pixels.shape != base.shape[:2]:
            raise ValueError("Shared line mask has the wrong shape")
    structured = base.copy()
    structured[line_pixels] = STRUCTURED_LINE

    image = Image.fromarray(structured)
    draw = ImageDraw.Draw(image)
    is_cue = int(cue_start) <= int(frame_index) <= int(cue_end)
    _draw_fixed_arrow_panel(
        draw,
        int(base.shape[0]),
        is_cue=is_cue,
        label=int(label),
    )
    return StructuredFrame(
        rgb=np.asarray(image, dtype=np.uint8).copy(),
        is_cue=is_cue,
        vertical_line_count=line_count,
        line_mask=line_pixels,
    )


def structured_pair_bundle(
    source_config: dict,
    split: str,
    scene_index: int,
    *,
    include_vertical_lines: bool = True,
    full_image_lines: bool = False,
) -> tuple[dict, object, dict[int, dict]]:
    """Return source metadata and structured frames for both siblings."""
    scene, design, pairs = paired_noisy_episodes(
        source_config,
        split,
        scene_index,
        sigma=0.0,
    )
    by_label = {int(episode["label"]): episode for episode in pairs}
    rendered = {
        label: {"episode": by_label[label], "frames": []}
        for label in (0, 1)
    }
    frame_count = next(iter(by_label.values()))["rgb"].shape[0]
    for frame_index in range(frame_count):
        base_frames = {
            label: by_label[label]["rgb"][frame_index]
            for label in (0, 1)
        }
        sibling_frames = render_structured_siblings(
            base_frames,
            scene_seed=scene["seed"],
            frame_index=frame_index,
            cue_start=design.cue_start,
            cue_end=design.cue_end,
            include_vertical_lines=include_vertical_lines,
            full_image_lines=full_image_lines,
        )
        for label in (0, 1):
            rendered[label]["frames"].append(sibling_frames[label])
    return scene, design, rendered


def structured_pair(
    source_config: dict,
    split: str,
    scene_index: int,
    *,
    include_vertical_lines: bool = True,
    full_image_lines: bool = False,
) -> tuple[dict, object, dict[int, list[StructuredFrame]]]:
    """Return both labels' structured frames after a shared nuisance stream."""
    scene, design, bundle = structured_pair_bundle(
        source_config,
        split,
        scene_index,
        include_vertical_lines=include_vertical_lines,
        full_image_lines=full_image_lines,
    )
    return (
        scene,
        design,
        {
            label: item["frames"]
            for label, item in bundle.items()
        },
    )
