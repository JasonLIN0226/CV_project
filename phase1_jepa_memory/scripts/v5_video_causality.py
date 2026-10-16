"""Model-independent exact-time rules for v5 video ingestion.

No decoder, model, label, or platform dependencies are imported here.
"""
from bisect import bisect_right
from fractions import Fraction


def as_pair(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def causal_index(presentation_times, requested_time):
    """Latest presented frame at or before the requested observation clock."""
    if not presentation_times:
        raise ValueError("empty presentation timeline")
    if any(t is None for t in presentation_times):
        raise ValueError("missing PTS must not be synthesized")
    if any(b <= a for a, b in zip(presentation_times, presentation_times[1:])):
        raise ValueError("presentation timeline must be strictly increasing")
    index = bisect_right(presentation_times, requested_time) - 1
    if index < 0:
        raise ValueError("no frame has been presented by this clock")
    return index


def annotation_frame_at(relative_time):
    """Map seconds to the official 30-fps annotation grid, never raw 60-fps IDs.

    Segment endpoint inclusion is intentionally not decided by this mapping.
    """
    relative_time = Fraction(relative_time)
    if relative_time < 0:
        raise ValueError("negative annotation time")
    return (relative_time * 30).__floor__()


def episode_starts(last_relative_time, tokens=256, fps=2, maximum=4):
    if tokens != 256 or fps != 2 or maximum != 4:
        raise ValueError("v5 episode specification changed")
    span = Fraction(tokens, fps)
    last_offset = Fraction(tokens - 1, fps)
    return [i * span for i in range(maximum)
            if i * span + last_offset <= last_relative_time]


def query_steps(tokens=256, start=40, stride=4, horizons=(1, 4, 16, 64)):
    if (tokens, start, stride, tuple(horizons)) != (256, 40, 4, (1, 4, 16, 64)):
        raise ValueError("v5 query specification changed")
    return list(range(start, tokens - max(horizons), stride))


def selected_video_names(manifest):
    """Selection is fixed by acquisition metadata, never by official CSV split."""
    names = {}
    for row in manifest["development_rgb"]:
        if row["split"] not in ("train", "validation"):
            raise ValueError("sealed participant in development importer")
        expected = f"recordings/{row['recording']}/C10095_rgb.mp4"
        if row["path"] != expected or row["camera"] != "C10095_rgb":
            raise ValueError("unexpected camera or path")
        video = expected.removeprefix("recordings/")
        if video in names:
            raise ValueError("duplicate recording")
        names[video] = {"split": row["split"], "participant": row["participant"]}
    return names


def filter_annotation_rows(raw_rows, manifest):
    """Quarantine filter BEFORE interpreting any numeric labels.

    This helper is software-tested only; the present PTS audit downloads no labels.
    """
    names = selected_video_names(manifest)
    for raw in raw_rows:
        video = raw.get("video")
        if video not in names:
            continue
        row = dict(raw)
        row.update(names[video])
        yield row
