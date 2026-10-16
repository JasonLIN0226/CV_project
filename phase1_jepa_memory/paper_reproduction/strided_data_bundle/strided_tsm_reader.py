"""Read the exact author-selected indices without fetching discarded values."""
from __future__ import annotations

import re
import numpy as np
import torch

FRAME_PATTERN = re.compile(rb"_(\d+)\.jpg$")
CAMERA_ID = "C10095_rgb"


def read_recording(environment, recording_id: str, frame_stride: int):
    if frame_stride <= 0:
        raise ValueError("positive stride required")
    prefix = f"{recording_id}/{CAMERA_ID}/{CAMERA_ID}_".encode()
    frame_ids, keys = [], []
    with environment.begin(buffers=True) as transaction:
        cursor = transaction.cursor()
        present = cursor.set_range(prefix)
        while present:
            key = bytes(cursor.key())
            if not key.startswith(prefix):
                break
            match = FRAME_PATTERN.search(key)
            if match is None:
                raise ValueError(f"malformed Assembly101 TSM key: {key!r}")
            frame_ids.append(int(match.group(1)))
            keys.append(key)
            present = cursor.next()
        if not keys:
            raise KeyError(f"no TSM features found for {recording_id}/{CAMERA_ID}")
        # Match the upstream numeric sorting and every-fifteenth selection.
        order = np.argsort(np.asarray(frame_ids))[::frame_stride]
        frame_array = np.asarray(frame_ids, dtype=np.int64)[order]
        if len(np.unique(frame_array)) != len(frame_array):
            raise ValueError(f"duplicate frame ids in {recording_id}")
        selected = []
        for index in order:
            raw = transaction.get(keys[int(index)])
            if raw is None:
                raise RuntimeError("indexed value disappeared")
            value = np.frombuffer(raw, dtype=np.float32)
            if value.shape != (2048,):
                raise ValueError(f"expected a 2048-D TSM feature, got {value.shape}")
            selected.append(value.copy())
    feature_array = np.stack(selected)
    feature_array /= np.maximum(
        np.linalg.norm(feature_array, axis=1, keepdims=True), 1e-8
    )
    return torch.from_numpy(feature_array), torch.from_numpy(frame_array)
