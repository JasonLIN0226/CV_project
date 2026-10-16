"""Artificial software checks only; no actual video or official inference."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from phase1 import v5_encoding as frozen, v5_encoding_cpu as cpu


def protocol():
    root = Path(__file__).parents[1]
    path = root / "protocol.json"
    if not path.is_file():
        path = root / "configs/phase1_v5_real_encoding_cpu_protocol.json"
    return json.loads(path.read_text())


def test_shared_encoding_core_retains_original_causal_and_payload_checks():
    for name in ("encoding_plan", "encode_record", "checkpoint_files", "prefetch",
                 "file_sha", "no_symlink", "atomic_json", "work_projection"):
        assert getattr(cpu, name) is getattr(frozen, name)


def test_declared_zero_gpu_cpu_allocation():
    cpu.require_cpu_resource(cpu.COURSE, "910001", "b-software", "n2c48m24",
                             "4", "", "0", False, 0)


@pytest.mark.parametrize("index,value", [
    (0, "another-course"), (1, ""), (2, "torch-login"),
    (3, "g2-standard-12"), (4, "8"), (5, "0"),
    (6, "1"), (7, True), (8, 1)])
def test_unapproved_allocation_or_hidden_gpu_rejected(index, value):
    args = [cpu.COURSE, "910001", "b-software", "n2c48m24", "4", "", "0", False, 0]
    args[index] = value
    with pytest.raises(RuntimeError):
        cpu.require_cpu_resource(*args)


def test_original_full_data_cpu_protocol():
    cpu.require_cpu_protocol(protocol())


@pytest.mark.parametrize("section,key,value", [
    ("encoder", "batch_size", 1), ("encoder", "dtype", "bfloat16"),
    ("encoder", "revision", "main"), ("encoder", "frozen", False),
    ("budget", "actual_frames", 512), ("budget", "soft_stop_seconds", 28800),
    ("budget", "gpu_count", 1), ("budget", "memory_gb", 32),
    ("budget", "partition", "interactive"), ("decoder", "threads", 8)])
def test_budget_data_or_model_cannot_silently_change(section, key, value):
    p = copy.deepcopy(protocol())
    p[section][key] = value
    with pytest.raises(ValueError):
        cpu.require_cpu_protocol(p)


def test_fixed512_reuse_and_total_projection():
    assert cpu.cpu_projection(20, 100) == 9920
    assert cpu.cpu_projection(0, 13800 * 512 / 50688) == pytest.approx(13800)


@pytest.mark.parametrize("setup,work", [
    (float("nan"), 1), (1, float("inf")), (-1, 1), (1, -1)])
def test_nonfinite_or_negative_budget_measurement_fails_closed(setup, work):
    with pytest.raises(ValueError):
        cpu.cpu_projection(setup, work)


def test_budget_veto_preserves_first512_without_claiming_complete_cache(tmp_path, monkeypatch):
    p = protocol()
    control = tmp_path / "control"
    control.mkdir()
    for name, obj in [("protocol.json", p), ("manifest.json", {}), ("pts_report.json", {})]:
        (control / name).write_text(json.dumps(obj))
    (control / "source_lock.json").write_text("{}")
    raw = tmp_path / "rgb"
    raw.mkdir()
    video = raw / "video.mp4"
    video.write_bytes(b"artificial software fixture")
    cache_root = tmp_path / "CPU_CACHE_TEST_ONLY"
    monkeypatch.setattr(cpu, "CPU_OUTPUT_ROOT", cache_root)
    p["cache_namespace"] = str(cache_root)
    (control / "protocol.json").write_text(json.dumps(p))
    for key, value in {
        "SLURM_JOB_ACCOUNT": cpu.COURSE, "SLURM_JOB_ID": "910001",
        "SLURM_JOB_PARTITION": "n2c48m24", "SLURM_CPUS_PER_TASK": "4",
        "SLURM_JOB_GPUS": "", "SLURM_GPUS_ON_NODE": "0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(cpu.socket, "gethostname", lambda: "b-software-test")
    monkeypatch.setattr(cpu.shutil, "disk_usage", lambda _: types.SimpleNamespace(free=10**10))
    monkeypatch.setattr(cpu, "checkpoint_files", lambda *args: {"TEST_ONLY": {}})
    digest = __import__("hashlib").sha256()
    for i in range(512):
        t = Fraction(i, 2)
        digest.update(f"{i//256}:{i%256}:{t.numerator}/{t.denominator}\n".encode())
    row = {"path": "video.mp4", "bytes": video.stat().st_size,
           "sha256": cpu.file_sha(video), "count": 512,
           "timestamp_sha256": digest.hexdigest(), "recording": "TEST_ONLY",
           "participant": 0, "split": "train", "episodes": [{}, {}]}
    monkeypatch.setattr(cpu, "encoding_plan", lambda *args: [row])
    import v5_causal_frame_stream

    def images(*args, **kwargs):
        for i in range(512):
            yield Fraction(i, 2), Fraction(i, 2), np.zeros((4, 4, 3), np.uint8)
    monkeypatch.setattr(v5_causal_frame_stream, "single_frame_rgb", images)

    class TestEncoder:
        def __init__(self, cfg, device):
            assert device.type == "cpu"
            self.model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
            self.metadata = {"resolved_revision": cfg["revision"], "TEST_ONLY": True}

        def encode(self, rgb):
            assert rgb.shape == (2, 4, 4, 3)
            return np.zeros((2, 5120), dtype=np.float32)
    monkeypatch.setattr(cpu, "FrozenIJepa", TestEncoder)
    monkeypatch.setattr(cpu, "cpu_projection", lambda *args: 13801)
    output = cache_root / "job_910001"
    assert cpu.main(control, raw, output, types.SimpleNamespace(__version__="18.0.0")) == 2
    result = json.loads((output / "encoding_report.json").read_text())
    assert result["status"] == "STOPPED_FIXED_FIRST512_PROJECTION_EXCEEDS_BUDGET"
    assert result["frames_encoded"] == 512 and result["preflight_passed"] is False
    assert result["predictor_trained"] is False and result["phase1_passed"] is False
    assert len(result["completed_recordings"]) == 1
    from phase1.v5_real_training_l4 import validate_cache_report
    with pytest.raises(ValueError, match="incomplete"):
        validate_cache_report(result, {}, "910001")
