"""Tests target causality/leakage boundaries, not scientific success."""
import sys
from fractions import Fraction as F
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v5_video_causality import (causal_index, annotation_frame_at, episode_starts,
                                query_steps, filter_annotation_rows)


def test_nearest_future_frame_is_forbidden():
    times = [F(0), F(1, 60), F(2, 60)]
    assert causal_index(times, F(99, 6000)) == 0
    assert causal_index(times, F(1, 60)) == 1
    assert causal_index(times, F(1000)) == 2
    with pytest.raises(ValueError):
        causal_index(times, F(-1))


@pytest.mark.parametrize("times", [[], [None], [F(0), F(0)], [F(1), F(0)]])
def test_missing_or_reindexed_timestamps_fail_closed(times):
    with pytest.raises(ValueError):
        causal_index(times, F(1))


def test_annotation_grid_uses_30_not_raw_60():
    assert annotation_frame_at(F(1, 60)) == 0
    assert annotation_frame_at(F(2, 60)) == 1
    assert annotation_frame_at(F(1, 2)) == 15
    with pytest.raises(ValueError):
        annotation_frame_at(F(-1))


def test_short_clips_not_padded_and_episode_boundaries_are_common():
    assert episode_starts(F(12749, 100)) == []
    assert episode_starts(F(255, 2)) == [F(0)]
    assert episode_starts(F(511, 2)) == [F(0), F(128)]
    assert episode_starts(F(10000)) == [F(0), F(128), F(256), F(384)]
    with pytest.raises(ValueError):
        episode_starts(F(10000), tokens=128)


def test_all_horizons_stay_inside_same_episode_queries():
    q = query_steps()
    assert len(q) == 38 and q[0] == 40 and q[-1] == 188
    assert all(0 <= t + h < 256 for t in q for h in (1, 4, 16, 64))


def test_sealed_and_other_camera_labels_filtered_before_interpretation():
    manifest = {"development_rgb": [{"recording": "chosen", "camera": "C10095_rgb",
                 "path": "recordings/chosen/C10095_rgb.mp4", "split": "train", "participant": "dev"}]}
    rows = [{"video": "sealed/C10095_rgb.mp4", "verb_id": "INVALID DO NOT PARSE"},
            {"video": "chosen/other.mp4", "verb_id": "INVALID DO NOT PARSE"},
            {"video": "chosen/C10095_rgb.mp4", "verb_id": "7", "split": "test"}]
    filtered = list(filter_annotation_rows(rows, manifest))
    assert len(filtered) == 1 and filtered[0]["verb_id"] == "7"
    assert filtered[0]["split"] == "train"  # Our fixed participant split wins.
    manifest["development_rgb"][0]["split"] = "test"
    with pytest.raises(ValueError):
        list(filter_annotation_rows(rows, manifest))


def test_real_decoder_b_frame_order_and_causal_boundary(tmp_path):
    av = pytest.importorskip("av")
    import hashlib
    import time
    from audit_v5_video_data import audit_video
    path = tmp_path / "artificial.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=60)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "fast", "x264-params": "bframes=2:b-adapt=0:scenecut=0"}
        for index in range(7680):
            frame = av.VideoFrame(32, 32, "rgb24")
            frame.planes[0].update(bytes([index % 256]) * frame.planes[0].buffer_size)
            frame.pts = index
            frame.time_base = F(1, 60)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    row = {"recording": "artificial", "participant": "fixture", "split": "train",
           "path": "artificial.mp4", "bytes": path.stat().st_size,
           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = audit_video(av, path, row, time.monotonic() + 60)
    assert report["pts_reordering_present"]
    assert report["packet_frames"] == 7680 and report["sampled_frame_count"] == 256
    boundary = next(p for p in report["decode_probes"] if p["requested_seconds"] == [1, 120])
    assert boundary["selected_seconds"] == [0, 1]
    assert report["episodes"][0]["last_observation_seconds"] == [255, 2]
    assert not report["annotation_segment_endpoints_verified"]
