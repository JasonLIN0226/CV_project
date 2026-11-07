# Hidden-CoG PushT: a history-dependent PushT variant — 7 November 2026

Plain PushT gives memory nothing to do. gym-pusht has zero damping (velocity is
discarded every physics step) and a kinematic pusher, so the current frame and
the next actions determine the future. These variants hide a **discrete center of
gravity (CoG)** in the T block: left (−40, 20), center (0, 45) = gym-pusht default,
right (40, 20) in body-local coordinates (T scale 30). The block looks identical
under every class; the class shows only in how the block turns when pushed. A
retention method is useful here if it keeps the frames that show a push and its
response. The PushT end goal is planning; before that, the I-JEPA latent prediction
accuracy under the different CoG classes is evaluated with only retained history
frames as cues.

## Versions

| | Design | Status | Code |
|---|---|---|---|
| **v4** | Natural continuous episodes; shared start and probe per family | **current** | `pusht_cog_natural.py`, `prepare_pusht_cog_v4_data.py` |
| v3 | Matched probes (each class pushed differently to a shared pose) | abandoned (dev only) | `pusht_cog_matched.py` (+ probe library) |
| v2 | Continuous; class-specific start poses calibrated to a shared post-probe pose | superseded | `pusht_cog_coherent.py`, `prepare_pusht_cog_v2_data.py` |
| v1 | Probe, then a scene cut (teleport) to a fixed test pose | kept | `pusht_cog_cue.py`, `prepare_pusht_cog_v1_data.py` |

All four share: probe push → pusher leaves → gap without contact → query → top-face test push
→ target at query + 12 (recent window query−3..query). Model-facing archives hold exactly `rgb`,
`timestamps`, `actions_absolute`, `actions_normalized`; class labels, simulator state and checks live
in `evaluation_only/`. Siblings are stored as `s0/s1/s2` under a seeded shuffle.

Common implementation notes:
- Placing the block sets its angle before its position; pymunk rotates about the CoG, so the reverse
  order shifts the block by a class-dependent amount.
- Chipmunk keeps an overlap-correction (bias) velocity from the last solve and applies it on the next
  step. `place()` takes one step at the new pose to spend it, then places again; without this the block
  drifted 1–2 px with no contact.
- Only top-face test pushes within ±20 px of the stem separate all three classes; underside and side pushes
  leave two CoGs on the same side of the push line.
- On headless Linux use `opencv-python-headless==4.11.0.86` (the regular wheel needs `libGL.so.1`).

## v4 (current): natural episodes

A family shares the start pose, the pusher start and one probe push across the three classes, so the
first frame, every frame before contact and the probe actions are identical; only the block's response
differs. After the probe, everything is continuous physics: the pusher's transit, hold, gap and test push
are planned from the block's actual pose. Recent frames may differ between classes (the turned block
stays turned). Requirement (from the user, 7 November): the setting must let methods be tested on whether
they retain frames that hint the dynamics; identical recent windows are not required.

- **Probe:** drawn from the v3 probe library (any face, offset, tilt, depth; jittered), restricted to pushes
  whose class turns differ by ≥ 20° and whose shift is ≤ 70 px.
- **Start:** block centre uniform in [195, 317]² px, angle uniform.
- **Selection:** candidate seeds are tried in order; a family is rejected if any class fails validity
  (contact before the probe or in transit/gap, leaving the arena, test contact before the query).
  Class-symmetric; never on outcomes.
- **Counterfactual check:** from each episode's exact query state the test push is replayed under every
  CoG; the true-class replay reproduces the run to 3e-13 px.

**Dev (seeds 2026140000–191, simulator-state classifiers, family-grouped 5-fold CV, chance 0.33):**

| Start box | Valid | Present | First frame | End of probe | First frame + present |
|---|---:|---:|---:|---:|---:|
| [180, 332], shift ≤ 90 | 125/192 | 0.34 | 0.33 | 0.50 | 0.75 |
| [195, 317], shift ≤ 70 (used) | 168/192 | 0.40 | 0.33 | 0.60 | 0.83 |

Counterfactual spread at +12 (closest pair of CoGs from the same present): 21 / 26 / 30 px
(10th/50th/90th). The present-only leak comes from the arena: the same probe moves each class
differently, walls and the fixed goal are visible, so the block's arena position in its own frame depends
on the class. It grows as the start box narrows. Fixes if the encoder picks it up: widen the box (pull the
transit orbit in from the walls) and spread probe depths so class translations overlap.

**Corpus `pusht-cog-v4-20261107`** (job 22151809, 85 s on 32 CPUs, 62 MB under ignored `build/`):

| Split | Families | Episodes | Frames | Rejected candidates |
|---|---:|---:|---:|---:|
| Train | 128 | 384 | 34,095 | 40 |
| Validation | 32 | 96 | 8,526 | 7 |
| Evaluation | 32 | 96 | 8,532 | 3 |

Rejections: 48 out of bounds, 2 approach contact, 2 transit/gap contact. Seeds: train 2026151000+,
validation 2026152000+, evaluation 2026153000+. I-JEPA encoding runs on the course cloud-bursting cluster
from the package built by `experiments/phase2/slurm/cloud_burst/make_package.sh` (train+validation only).

## v3 (abandoned): matched probes

Every sibling shares the start pose; each class gets its own probe, solved by least squares (offset, tilt,
depth on one face) on the real episode prefix so the block lands on the anchor class's post-probe pose.
When it converges the match is exact (≤ 5e-13 px) and the reset frame is identical across classes, but
dev runs matched only 6/32 families (anchors from the whole library) and 11/32 (anchors restricted to
motions every class nearly reproduces); failed solves stalled at a median 1.4 px. All matched families had
turns within ±2°: pure translation, which requires pushing through each class's own CoG, so the contact
location alone revealed the class. Jobs 22139069, 22141060.

## v2 (superseded): calibrated start poses

The family samples the post-probe pose; each class starts at that pose with its own probe motion undone
(PushT is invariant to rigid motions of the scene), so all classes end the probe at one pose (max mismatch
1.7e-12 px), pushers rendezvous and then share every action; frames are pixel-identical from
`converge_frame` through the first test contact. Corpus `pusht-cog-v2-20261102` (128/32/32 families).
I-JEPA probes (job 21664867, validation, linear/MLP): recent window 0.333/0.333; reset frame 0.625/0.573;
end of probe 0.667/0.844; reset + query 0.698/0.979; target 0.385/0.333; query + target 0.271/0.917.
Superseded because the start pose (relative to the arena, in the block frame) reveals the class: 0.76 from
two state numbers.

## v1 (kept): scene cut

`pusht_cog_cue.py` teleports block and pusher to a fixed test pose after the probe, so siblings share
actions throughout and are pixel-identical from the cut through the query. Corpus `pusht-cog-v1-20261101`
(seeds 2026111000+/2026112000+/2026113000+; dev 2026110000–127); `prepare_pusht_cog_v1_data.py` is
byte-identical to the collector that produced it (SHA-256 23aaaa10…612161 in its manifest). I-JEPA probes
(job 21615935, validation, linear/MLP): recent window 0.333/0.333; cue frame 0.792/0.812; reset + cue
0.667/1.000; target 0.333/0.344; query + target 0.302/0.875.

## Reproduce

```bash
.venv/bin/python -m pytest -q tests/test_pusht_cog_cue.py tests/test_pusht_cog_coherent.py \
  tests/test_prepare_pusht_cog_v2_data.py tests/test_pusht_cog_natural.py
# Probe library (CPU; cached JSON), then v4 dev check and corpus (see experiments/phase2/slurm/)
sbatch experiments/phase2/slurm/run_pusht_cog_v4_collect.sbatch
# I-JEPA encoding + probes: lab GPU (run_pusht_cog_encode_probe.sbatch) or the course package
experiments/phase2/slurm/cloud_burst/make_package.sh build/pusht-cog-v4-20261107 HF_CACHE OUT_DIR
```

Encoding used torch 2.5.1+cu121, transformers 5.5.4, numpy 1.26.4, `facebook/ijepa_vith14_1k@f157467`,
grid2 pooling (5120 features).

## Limits

The cue is the block's response to a push, not momentum. Moment of inertia is the default for every class.
Probe readability does not show that a predictor learns to use history; the latent prediction evaluation
with retained history and the planning study have not been run.
