"""PushT pixel-only boundary and paired pre-encoding occlusion smoke.

This module does not claim that a trained PushT predictor or MPC exists. Gym
dependencies are lazy so the boundary can be tested independently on CPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rgb(value: np.ndarray) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
        raise ValueError("PushT observations must be HWC uint8 RGB pixels")
    return a


@dataclass(frozen=True)
class Occlusion:
    """Fixed half-open [start, stop) time interval; rectangle is x0,y0,x1,y1."""

    start: int = 8
    stop: int = 13
    rectangle: tuple[float, float, float, float] = (0., 0., 1., 1.)
    fill: int = 0
    enabled: bool = True

    def __post_init__(self):
        x0, y0, x1, y1 = self.rectangle
        if not (0 <= self.start < self.stop and 0 <= self.fill <= 255):
            raise ValueError("Invalid occlusion interval or fill")
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("Occlusion rectangle must be inside the image")

    def mask(self, shape: tuple[int, ...], timestamp: int) -> np.ndarray:
        if timestamp < 0:
            raise ValueError("Negative observation timestamp")
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        if self.enabled and self.start <= timestamp < self.stop:
            x0, y0, x1, y1 = self.rectangle
            mask[int(np.floor(y0*h)):int(np.ceil(y1*h)),
                 int(np.floor(x0*w)):int(np.ceil(x1*w))] = True
        return mask

    def apply(self, rgb: np.ndarray, timestamp: int) -> np.ndarray:
        result = _rgb(rgb).copy()
        result[self.mask(result.shape, timestamp)] = self.fill
        return result


@dataclass(frozen=True)
class VisualObservation:
    """Only this object is permitted to enter an encoder/model boundary."""

    timestamp: int
    rgb: np.ndarray

    def __post_init__(self):
        if self.timestamp < 0:
            raise ValueError("Negative observation timestamp")
        pixels = _rgb(self.rgb).copy()
        pixels.setflags(write=False)
        object.__setattr__(self, "rgb", pixels)


@dataclass(frozen=True)
class Diagnostic:
    """Privileged simulator info/reward; keep outside model/policy inputs."""

    timestamp: int
    reward: float | None
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def absolute_actions(value: np.ndarray) -> np.ndarray:
    a = np.asarray(value, dtype=np.float32)
    if a.ndim < 1 or a.shape[-1] != 2 or not np.isfinite(a).all():
        raise ValueError("Absolute actions must have final dimension 2 and be finite")
    if ((a < 0) | (a > 512)).any():
        raise ValueError("Absolute PushT action outside [0,512]; no silent clipping")
    return a.copy()


def normalize_actions(value: np.ndarray) -> np.ndarray:
    """Preserve absolute-coordinate action sequence, mapping [0,512] to [-1,1]."""
    return absolute_actions(value) / np.float32(256.) - np.float32(1.)


def denormalize_actions(value: np.ndarray) -> np.ndarray:
    a = np.asarray(value, dtype=np.float32)
    if a.ndim < 1 or a.shape[-1] != 2 or not np.isfinite(a).all() or (np.abs(a) > 1).any():
        raise ValueError("Normalized absolute actions must be finite and in [-1,1]")
    return absolute_actions((a + np.float32(1.)) * np.float32(256.))


class PushTVisualAdapter:
    """Actual environment observations, never privileged state or render overlays."""

    def __init__(self, env, occlusion: Occlusion):
        self._env, self.occlusion = env, occlusion
        self.timestamp, self._done, self._reset = 0, False, False

    @classmethod
    def create(cls, occlusion: Occlusion | None = None):
        import gymnasium as gym
        import gym_pusht  # noqa: F401 -- registers the official environment
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array",
                       observation_width=96, observation_height=96)
        return cls(env, occlusion or Occlusion())

    def reset(self, *, seed: int):
        raw, info = self._env.reset(seed=seed)
        self.timestamp, self._done, self._reset = 0, False, True
        obs = VisualObservation(0, self.occlusion.apply(raw, 0))
        return obs, Diagnostic(0, None, False, False, dict(info))

    def step(self, action: np.ndarray):
        if not self._reset or self._done:
            raise RuntimeError("Reset before stepping a new or terminated episode")
        action = absolute_actions(action)
        if action.shape != (2,):
            raise ValueError("Environment step requires one absolute action of shape (2,)")
        raw, reward, terminated, truncated, info = self._env.step(action)
        self.timestamp += 1
        self._done = bool(terminated or truncated)
        obs = VisualObservation(self.timestamp, self.occlusion.apply(raw, self.timestamp))
        diagnostic = Diagnostic(self.timestamp, float(reward), bool(terminated), bool(truncated), dict(info))
        return obs, diagnostic

    def close(self):
        self._env.close()


def encode_observations(encoder, observations: list[VisualObservation], *, batch_size=2) -> np.ndarray:
    """Encoder receives only already-masked RGB; no access to env/info/control."""
    if not observations or batch_size < 1:
        raise ValueError("Nonempty observations and positive batch size required")
    if any(not isinstance(o, VisualObservation) for o in observations):
        raise TypeError("Only VisualObservation objects may enter the encoder")
    result = []
    for offset in range(0, len(observations), batch_size):
        pixels = np.stack([o.rgb for o in observations[offset:offset+batch_size]])
        features = np.asarray(encoder.encode(pixels))
        if features.ndim != 2 or len(features) != len(pixels) or not np.isfinite(features).all():
            raise ValueError("Encoder must return finite [batch, feature] representations")
        result.append(features)
    return np.concatenate(result)


def later_visible_targets(observations: list[VisualObservation], occlusion: Occlusion,
                          horizons=(3, 6), recent_length=4) -> list[dict[str, int]]:
    """Offline target audit only; its future visibility is not policy input."""
    by_time = {o.timestamp: o for o in observations}
    if len(by_time) != len(observations):
        raise ValueError("Duplicate observation timestamps")
    valid = []
    for t, obs in by_time.items():
        recent_times = range(t-recent_length+1, t+1)
        if any(s not in by_time or not occlusion.mask(by_time[s].rgb.shape, s).all() for s in recent_times):
            continue
        for h in horizons:
            target = by_time.get(t+h)
            if h > 0 and target is not None and not occlusion.mask(target.rgb.shape, t+h).any():
                valid.append({"query": t, "horizon": h, "target": t+h})
    return valid


def collect_paired_trace(*, seed=17, steps=24, occlusion=None, factory=PushTVisualAdapter.create):
    schedule = occlusion or Occlusion()
    masked = factory(schedule)
    control = factory(Occlusion(enabled=False))
    observations, controls, diagnostics, actions = [], [], [], []
    rng = np.random.default_rng(seed + 100_000)
    try:
        mo, md = masked.reset(seed=seed)
        co, cd = control.reset(seed=seed)
        observations.append(mo)
        controls.append(co)
        diagnostics.append({"masked": asdict(md), "control": asdict(cd)})
        if not np.array_equal(mo.rgb, schedule.apply(co.rgb, co.timestamp)):
            raise AssertionError("Same seed control diverged at reset")
        for _ in range(steps):
            action = rng.uniform(40., 472., size=2).astype(np.float32)
            mo, md = masked.step(action)
            co, cd = control.step(action)
            if not np.array_equal(mo.rgb, schedule.apply(co.rgb, co.timestamp)):
                raise AssertionError("Same seed/action control diverged beyond the prescribed RGB mask")
            if (md.reward, md.terminated, md.truncated) != (cd.reward, cd.terminated, cd.truncated):
                raise AssertionError("Observation masking altered physical transition or reward")
            observations.append(mo)
            controls.append(co)
            diagnostics.append({"masked": asdict(md), "control": asdict(cd)})
            actions.append(action)
            if md.terminated or md.truncated:
                break
    finally:
        masked.close()
        control.close()
    targets = later_visible_targets(observations, schedule)
    return observations, controls, np.asarray(actions, dtype=np.float32), diagnostics, targets


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--encoder-config", type=Path,
                        help="Optional existing JSON config with an official I-JEPA encoder section")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory; existing evidence will not be overwritten")
    schedule = Occlusion()
    observations, controls, actions, diagnostics, targets = collect_paired_trace(
        seed=args.seed, steps=args.steps, occlusion=schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    masked = np.stack([o.rgb for o in observations])
    control = np.stack([o.rgb for o in controls])
    masks = np.stack([schedule.mask(o.rgb.shape, o.timestamp) for o in observations])
    np.savez_compressed(args.output / "paired_trace.npz", observed_rgb=masked, control_rgb=control,
                        masks=masks, timestamps=np.array([o.timestamp for o in observations]),
                        actions_absolute=actions, actions_normalized=normalize_actions(actions))
    (args.output / "diagnostics_only.json").write_text(json.dumps(diagnostics, default=_json_default, indent=2)+"\n")
    manifest = {
        "schema_version": 1, "status": "interface_trace_only", "seed": args.seed,
        "environment": "gym_pusht/PushT-v0", "obs_type": "pixels", "observations": len(observations),
        "rgb_shape": list(masked.shape[1:]), "rgb_dtype": str(masked.dtype),
        "occlusion": asdict(schedule), "action_semantics": "absolute target XY in [0,512]; NOT displacement",
        "same_seed_actions_control_verified": True, "pixel_changes": int((masked != control).sum()),
        "later_visible_targets_after_fully_occluded_recent4": targets,
        "history_necessity_validated": False, "baseline_model_planning_verified": False,
        "encoder_verified": False, "source_sha256": _sha(__file__),
        "dependencies": {},
    }
    for package in ("gym-pusht", "gymnasium", "pygame", "pymunk", "shapely", "numpy", "opencv-python"):
        manifest["dependencies"][package] = importlib.metadata.version(package)
    try:
        from PIL import Image
        chosen = [t for t in (0, 7, 8, 11, 12, 13, 17) if t < len(observations)]
        sheet = np.concatenate([np.concatenate([control[t], masked[t]], axis=0) for t in chosen], axis=1)
        Image.fromarray(sheet).save(args.output / "paired_trace.png")
        manifest["contact_sheet"] = {"file": "paired_trace.png", "columns": chosen,
                                      "top": "unoccluded control", "bottom": "encoder input"}
    except ImportError:
        manifest["contact_sheet"] = None
    if args.encoder_config:
        import torch
        from phase1.encoders import FrozenIJepa
        config = json.loads(args.encoder_config.read_text())["encoder"]
        if config.get("backend") != "ijepa":
            raise ValueError("Encoder proof requires explicit backend='ijepa'; smoke pixels are not JEPA")
        encoder = FrozenIJepa(config, torch.device(args.device))
        z = encode_observations(encoder, observations, batch_size=args.batch_size)
        z_control = encode_observations(encoder, controls, batch_size=args.batch_size)
        np.savez_compressed(args.output / "encoded_trace.npz", observed=z, control=z_control)
        manifest.update(encoder_verified=True, encoder=encoder.metadata, encoded_shape=list(z.shape),
                        encoder_config_sha256=_sha(args.encoder_config))
    manifest["files"] = {p.name: _sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    manifest["checks_passed"] = bool(targets and manifest["pixel_changes"] > 0)
    (args.output / "manifest.json").write_text(json.dumps(manifest, default=_json_default, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "observations": len(observations),
                      "checks_passed": manifest["checks_passed"], "encoder_verified": manifest["encoder_verified"],
                      "baseline_model_planning_verified": False}, indent=2))
    if not manifest["checks_passed"]:
        raise SystemExit("Trace did not reach a masked recent buffer and later visible target; see manifest")


if __name__ == "__main__":
    main()
