# Hidden-CoG PushT: a history-dependent PushT variant — 9 November 2026

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
| **v5** | CoG shown as a dot for 3 early frames; no probe; siblings identical except those frames | **current (Gate-like testbed)** | `pusht_cog_marker.py`, `prepare_pusht_cog_marker_data.py` |
| **v4** | Natural continuous episodes; shared start and probe per family | **current (harder follow-up)** | `pusht_cog_natural.py`, `prepare_pusht_cog_v4_data.py` |
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

## v5 (current, Gate-like testbed): marker

The PushT analogue of Gate. The hidden CoG class is drawn as a red dot (radius 2 px at 96×96, RGB
220/20/20) at the CoG's position on the block for 3 consecutive frames, starting uniformly at frame 2–10.
There is no probe push: the block is untouched until the test push, and every action comes from one plan
per family (orbit wander, transit, hold, gap, top-face test push), so siblings share all actions and are
pixel-identical except the marker frames and after the first test contact. The recent window
(query−3..query, query ≈ frame 56) is identical across classes; the future depends on the class through the
test push (closest pair ≈ 25 px apart at +12 in a test family). Marker frames and positions are oracle
information and live only in `evaluation_only/`. The dot is drawn on the observation after rendering; it is
not physics.

Why: v4's cue is an interaction (a push and the block's response, read together with the present), which the
current Gate scorer is not trained for (teacher states only at the query, from baseline-policy candidates;
local credit). v5 fits the existing single-cue retention pipeline and gives the PushT predictor and planning
work a clean first testbed. v4 remains the harder follow-up, to be trained DAgger-style with the deletion
cost evaluated at the target frame.

Corpus `pusht-cog-v5-marker-20261109` (128/32/32 families; seeds train 2026161000+, validation 2026162000+,
evaluation 2026163000+; test-only 2026159000–029): collection job 22486446.

## v4 (harder follow-up): natural episodes

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

**I-JEPA readability (course cloud-bursting A100, 940 s; validation, chance 0.33, linear/MLP):** recent
window 0.30/0.29; first frame 0.33/0.33; end of probe 0.27/0.30; first frame + query 0.32/0.31; target
0.26/0.29; query + target 0.30/0.83. Probes over every probe frame (+ first frame + query) also stay at
0.28–0.38: with 102 probe types and 384 labelled episodes the end-to-end probes are data-starved.

**Pose decoding** (`diagnose_pusht_cog_pose_decoding.py`, `results/pusht_cog/v4_pose_decoding_20261108.json`):
a per-frame regressor from I-JEPA features (StandardScaler, PCA 256, MLP 512-256) trained on all ~34k
train frames with privileged simulator poses decodes block centre / angle / pusher at a median 4.2 px /
3.1° / 10.7 px on validation (ridge: 7.0 px / 9.1° / 18.5 px). Classifying the CoG from decoded poses:
present 0.33, first frame + present 0.59, end of probe + present 0.52 (true poses: 0.36 / 0.81 / 0.63).
The cue survives the encoder; pose precision (probe turns differ by roughly 10–40°) limits it. Stage 1
uses simulator labels and is a diagnostic only.

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

## Handoff (9 November 2026)

Data are not in Git (`build/` is ignored). On Torch, under `/projects/work/yang-lab/users/yz12129/CV_project/build/`:

| Path | Size | Contents |
|---|---:|---|
| `pusht-cog-v5-marker-20261109/` | ~60 MB (expected) | v5 corpus, all three splits |
| `pusht-cog-v4-20261107/` | 60 MB | v4 corpus, all three splits |
| `pusht-cog-v4-20261107-ijepa/` | 833 MB | v4 train+validation I-JEPA features (5120/frame), probes, burst log |
| `pusht_cog_probe_library_v3.json` | 0.2 MB | probe library used by v3/v4 |
| `pusht-cog-v1-20261101*`, `pusht-cog-v2-20261102*` | ~45 MB + ~1 GB features | earlier versions |

I-JEPA weights: `facebook/ijepa_vith14_1k@f157467` (2.4 GB), downloadable from Hugging Face. All corpora
regenerate on CPU in minutes from the collectors (seeds and protocols are fixed); the evaluation splits have
not been read by any model.

Course GPU (`csci_ga_2271-2026fa`): only through ood.burst.hpc.nyu.edu code-server sessions. Their
Singularity container exposes `/dev/nvidia*` but not the NVIDIA user-space driver: extract the driver
package matching `/proc/driver/nvidia/version` (610.43.02 on 8 November;
`https://download.nvidia.com/XFree86/Linux-x86_64/610.43.02/NVIDIA-Linux-x86_64-610.43.02.run`) with
`--extract-only` into the home directory, symlink `lib*.so.<ver>` to `.so.1`, and prepend it to
`LD_LIBRARY_PATH`. `slurm/cloud_burst/make_package.sh` builds the full package (code, data, weights);
`make_data_addon.sh` adds another dataset to an existing package folder.

Next steps:
1. Encode v5 (train+validation) and run the readability probes (marker frame + query should be high,
   recent window at chance).
2. PushT predictor: absolute-action conditioning, 4-frame present plus B = 2 history slots, trained with
   randomly retained history; evaluate latent error at the target per CoG class for recent-only, fixed
   baselines, the marker-frame oracle and an exhaustive pair oracle.
3. Gate-style deletion-cost scorer on v5; then v4 with DAgger-style states and target-time credit.
4. Planning (MPC with the learned predictor).

## Reproduce

```bash
.venv/bin/python -m pytest -q tests/test_pusht_cog_cue.py tests/test_pusht_cog_coherent.py \
  tests/test_prepare_pusht_cog_v2_data.py tests/test_pusht_cog_natural.py tests/test_pusht_cog_marker.py
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
