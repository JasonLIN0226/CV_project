# PushT interface and remaining baseline planning work

This addition closes part of the proposal's Phase 1 integration gap. The inherited
repository contains no PushT interface, PushT-trained world model, or MPC runner.
`experiments/phase2/pusht.py` adds the actual environment's RGB/action boundary and
a reproducible paired occlusion/control trace. It does **not** establish history
necessity, encoder execution, or baseline-model planning merely by passing CPU tests.

## Install and run

Use Python 3.10–3.12 in the task environment. Keep NumPy 1.26.4 compatible with
the inherited project; OpenCV 4.12+ may require NumPy 2 on current Python versions.

```bash
.venv/bin/python -m pip install numpy==1.26.4 gym-pusht==0.1.8 opencv-python==4.11.0.86 Pillow
.venv/bin/python -m pytest -q tests/test_phase2_pusht.py
.venv/bin/python experiments/phase2/pusht.py --output build/phase2-independent-20261020/verified/pusht
```

The output is a compressed paired RGB/action trace, masks, a separate privileged
diagnostic file, an optional contact sheet, and a hashed manifest. The contact
sheet's top row is the unoccluded control and bottom row the actual encoder input;
columns are identified in the manifest. Existing output is never overwritten.
All of this is CPU environment work; no GPU job or predictor training is started.

An optional encoder check uses the existing `phase1.encoders.FrozenIJepa`, with an
explicit JSON `encoder` configuration whose backend is `ijepa`, and real pinned
weights. A missing weight/configuration is an error, never a random-weight fallback:

```bash
PYTHONPATH=experiments/predictor/control/src .venv/bin/python experiments/phase2/pusht.py \
  --output build/pusht-encoded-interface --encoder-config /actual/config.json --device cuda
```

The plain CPU trace records `encoder_verified=false` and
`baseline_model_planning_verified=false`. The optional encoding command only changes
the first flag after real execution; planning remains false. Encoded observed/control
arrays and resolved encoder metadata are saved separately.

## Contract and chosen occlusion variant

- Environment: `gym_pusht/PushT-v0`, explicitly `obs_type="pixels"` and
  `render_mode="rgb_array"`. Pixel observations are HWC uint8 RGB, 96×96×3. Use the
  observation returned by `reset`/`step`, never `render()`, which uses visualization
  resolution and can overlay the previous action.
- The environment's native 2D action is the **absolute target XY position**, each
  coordinate in [0,512]. Sequence order is preserved. Normalization maps it to
  [-1,1]; it is not a displacement. Invalid/nonfinite actions fail rather than clip.
- The inherited Gate/Motion input builder sums past additive controls. That summary
  and the corresponding weights are not automatically compatible with PushT merely
  because both use two action coordinates. A future PushT model must explicitly
  consume absolute-action sequences or a documented compatible encoding.
- Model-facing observations contain only timestamp and already-masked RGB. The
  environment's `info` includes agent/block/goal state; rewards, termination, and
  info are returned as a distinct `Diagnostic` object and saved separately. They
  never enter the encoder. The frozen encoder receives RGB batches only.
- The initial engineering variant blacks out the complete RGB image at timestamps
  8 through 12 inclusive (half-open interval [8,13)). The mask is applied before
  encoding. The unoccluded control runs the same seed and actions in a separate
  environment and must match exactly outside the prescribed pixel transformation.
  This is a simple starting variant, not a proposal-mandated geometry or evidence
  that the task needs memory. Rectangle masking is also supported by configuration.
- With recent length 4, queries 11 and 12 have a fully occluded recent buffer.
  The trace audits horizons 3/6 whose later targets are visible. It fails its smoke
  check if termination prevents observing such a target. Nothing uses future
  visibility to make a retention/planning decision.
- Every reset clears time and termination state. Stepping after termination or
  truncation is rejected until reset. Environment/control reward and termination
  must agree; observation masking must not alter physical evolution.

## Baseline-model planning is still outstanding

The portable next connection is recent/bounded encoded observations plus timestamped
absolute-action history → a **PushT-trained action-conditioned baseline predictor** →
predicted features for planner-generated candidate future action sequences → goal
feature objective → selected action → environment step. Actual goal RGB is legal;
simulator states, future recorded actions, and simulator rollout predictions are not
substitutes for this learned visual prediction path.

Do not count the random actions used to collect this interface trace as planning.
Do not count a test encoder, an untrained predictor, or a transplanted Gate/Motion
checkpoint as the required PushT baseline. A real smoke must record its trained
checkpoint, visual goal, action search/horizon/budget, model calls and predicted
objectives, selected actions, actual transitions, and whether changing predictions
changes selection. Its scope is end-to-end operation; formal PushT success-rate
comparison belongs to the later proposal phase.

## Official API sources checked on 2026-10-20

- [Hugging Face gym-pusht environment source](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)
- [Official package dependencies/version](https://github.com/huggingface/gym-pusht/blob/main/pyproject.toml)
- [Environment registration and 300-step limit](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/__init__.py)

The run manifest records installed versions; the links above are mutable upstream
references, not immutable proofs of the exact locally installed source.
