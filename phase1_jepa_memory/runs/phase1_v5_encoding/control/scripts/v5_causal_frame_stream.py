"""Sequential single-frame extraction, prepared for a later locked encoding run.

This module has no model import and has not been run on the scientific videos.
"""
from fractions import Fraction


def presentation_time(frame):
    if frame.pts is None or frame.time_base is None or frame.is_corrupt:
        raise ValueError("missing/corrupt observed frame; no inferred timestamp")
    return Fraction(frame.pts) * frame.time_base


def causal_frames(decoded_frames, observation_clocks, max_lag=Fraction(1, 30)):
    """Yield (clock, actual PTS, frame) without nearest-future selection.

    Decode once in presentation order instead of repeatedly seeking for each
    feature. A later frame closes the temporal bracket but is never substituted
    for the requested observation. Stop when the last requested clock is served.
    """
    clocks = tuple(Fraction(t) for t in observation_clocks)
    if not clocks:
        return
    if any(t < 0 for t in clocks) or any(b <= a for a, b in zip(clocks, clocks[1:])):
        raise ValueError("observation clocks must be nonnegative and strictly increasing")
    index = 0
    previous = None
    previous_time = None
    for frame in decoded_frames:
        stamp = presentation_time(frame)
        if previous_time is not None and stamp <= previous_time:
            raise ValueError("decoded frames not in strict presentation order")
        while index < len(clocks) and clocks[index] < stamp:
            clock = clocks[index]
            if previous is None or clock - previous_time > max_lag:
                raise ValueError("missing or stale causal observation")
            yield clock, previous_time, previous
            index += 1
        if index == len(clocks):
            return
        previous, previous_time = frame, stamp
    # EOF is valid only when the final requested observations were actually seen.
    while index < len(clocks):
        clock = clocks[index]
        if previous is None or clock != previous_time:
            raise ValueError("requested observation extends beyond actual video EOF")
        yield clock, previous_time, previous
        index += 1


def single_frame_rgb(av, path, observation_clocks, threads=4):
    """RGB conversion occurs on only the chosen frame, never a temporal window."""
    if threads != 4:
        raise ValueError("v5 CPU decode thread budget changed")
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError("expected one observed camera stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        stream.codec_context.thread_type = "SLICE"
        for clock, stamp, frame in causal_frames(container.decode(stream), observation_clocks):
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or str(rgb.dtype) != "uint8":
                raise ValueError("unexpected RGB representation")
            yield clock, stamp, rgb
