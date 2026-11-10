# Hidden-CoG PushT: history-dependent PushT for bounded visual memory — 9 November 2026

This document explains the PushT variants built for the retention study, what each one tests, what the
current retention scheme needs in order to work on them, and why the natural variant (v4) is much harder
than the marker variant (v5), with the ways to address it. Results, data locations and reproduction steps
follow at the end.

**Status.** v5 (marker) is the Gate-like PushT testbed to build on first; its corpus is being collected
(job 22486446). v4 (natural) is collected and encoded; it is the harder follow-up. v1–v3 are kept for
provenance. No PushT predictor, retention scorer or planner has been trained yet. The PushT end goal is
planning; before that, the latent prediction accuracy of an I-JEPA-feature predictor is evaluated under the
different hidden dynamics with only the retained history frames as cues.

## 1. What is being tested

### 1.1 The retention scheme (as built for Gate)

The project studies a model that sees the last **R = 4** frames (the *present*) plus **B = 2** retained
*history* frames, chosen online and irrevocably: when a frame leaves the recent window it competes with the
two stored frames, and one of the three is deleted for good.

- **Encoder and predictor.** A frozen I-JEPA ViT-H/14 encodes each 96×96 frame (resized to 224) into 5,120
  features (patch tokens average-pooled into four image quadrants, "grid2"). A predictor
  (`DynamicsPredictor`, `experiments/predictor/control/src/phase1/model.py`) maps the present, the retained
  slots (with their ages) and the actions to future latents. It summarizes the slots as a content-weighted
  memory and a fitted feature velocity over time, so it can relate history to the present.
- **Deletion-cost teacher** (`gate_supervision_comparison_design_v2/teachers.py`). For a decision among three
  candidates (two stored frames and the arrival), the frozen predictor is run three times, each time with one
  candidate deleted, and scored against the real future latents (query + 3 and + 6). The three errors become
  a softmax target over "which one to delete". Deleting a frame that the prediction needs costs a lot.
- **Student scorer** (`core.py`). A set-conditioned network scores the three candidates jointly (each
  candidate's features, plus a mean-pooled context over all three) and is trained with cross-entropy to the
  teacher's distribution. Per-candidate inputs: its projected latent, difference to the current frame, log
  age, cumulative action since then, the current frame, the recent mean and the last action.
- **Training states** (`states.py`). Teacher labels are produced only at the **query**: for every training
  episode, each baseline policy (FIFO, reservoir, temporal buckets, visual diversity) is rolled out to the
  query, and its final three candidates are labelled. At test time the student makes every online decision.

On Gate this works: the deletion-cost scorer keeps the cue in 100% of validation episodes and reaches
94.5% route accuracy, versus ~56% for FIFO and the attention-proxy scorer (`docs/GATE_RETENTION_COMPARISON_LATEST.md`).

### 1.2 What a PushT setup must provide

1. **The future depends on a hidden variable** (here, the T block's centre of gravity).
2. **The present does not reveal it**, so a recent-only model cannot predict well.
3. **Some past frames do reveal it**, so retaining the right ones helps.
4. **The encoder and predictor can actually use those frames.** The information must survive I-JEPA encoding
   and be learnable from the available data.
5. For the later planning study, the hidden variable must change which actions achieve a goal.

## 2. PushT physics and the hidden CoG

**Why plain PushT needs no memory.** In gym-pusht the space has no gravity and zero damping, so every body
loses all its velocity at every physics step: the T block moves only while it is being pushed and stops the
instant contact ends. The pusher is a kinematic body (infinitely heavy) driven by a PD controller toward an
**absolute target position** (actions are targets in [0, 512]², not velocities). Contacts are frictionless,
so a push acts only along the contact normal. The current frame plus the upcoming actions therefore determine
the future; old frames add nothing.

**The hidden CoG.** We move the block's centre of gravity to one of three body-local positions (T scale 30;
crossbar x ∈ [−60, 60], y ∈ [0, 30]; stem x ∈ [−15, 15], y ∈ [30, 120]):

| Class | CoG (body-local) | Note |
|---|---|---|
| left | (−40, 20) | under the left part of the crossbar |
| center | (0, 45) | the gym-pusht default |
| right | (40, 20) | under the right part of the crossbar |

The block's appearance is identical in every class. Under a frictionless push, the block turns according to
which side of the push line the CoG lies on and how far: the same push turns a left-CoG block one way, a
right-CoG block the other way, and a centre-CoG block little. Moment of inertia is left at the default.

**The test push.** Every variant ends with a *test push* on the crossbar's top face, toward the stem, at a
lateral offset uniform in ±20 px. Within this range the push line passes between the CoG positions, so the
three classes respond clearly differently. Underside and side pushes leave two CoGs on the same side of the
push line and were rejected.

## 3. Terminology and episode anatomy

Frame t is the observation after action t−1; frame 0 is the reset. A **family** is one sampled scenario run
once per CoG class; the three runs are **siblings**, stored as `s0/s1/s2` in a seeded random order.

| Term | Meaning |
|---|---|
| **First frame** | Frame 0: the scene before anything happens. |
| **Cue frames** | The frames that reveal the class: the marker frames (v5) or the probe frames (v4). |
| **Probe** (v4) | A deliberate push, identical in every sibling, whose effect on the block depends on the CoG. |
| **End of probe** (v4) | The last frame of the probe's retreat: block at its post-probe pose, pusher just backed off. |
| **Transit / hold / gap** | The pusher circles to the test side, waits, then wanders slightly, never touching the block. |
| **Query** | The last gap frame: the moment of prediction. |
| **Present** (recent window) | Frames query−3 … query, always visible to the model. Everything earlier is history. |
| **Test push** | Starts right after the query; first contact about 4 frames later. |
| **Target** | Frame query + 12: the future latent to predict; its content depends on the class. |

Model-facing archives hold exactly `rgb`, `timestamps`, `actions_absolute`, `actions_normalized`. Class labels,
simulator state, cue-frame indices and checks live in `evaluation_only/`.

## 4. v5 (marker): the Gate-like testbed

```text
frame  0 | 1 … 20 (marker for 3 frames starting at 2–10) | transit, hold, gap | query ≈ 56 | test push | target = query+12
        first frame   pusher wanders on an orbit             no contact          present        block turns by class
```

- **Cue:** a red dot (radius 2 px at 96×96, RGB 220/20/20) drawn at the CoG's image position for 3 consecutive
  frames starting uniformly at frame 2–10 (random, so a policy cannot keep a fixed frame index). The dot is
  drawn on the observation after rendering; it is not physics.
- **No probe:** the block is untouched until the test push. All actions come from one plan per family (wander
  on an orbit, transit, hold, gap, test push), computed from the shared start pose, so **all three siblings
  share every action**.
- **Pairing:** siblings are pixel-identical everywhere except the three marker frames, until the test push
  touches the block. The present is identical across classes (exact chance for a recent-only model) and
  sibling-history swaps are valid counterfactuals, as in Gate.
- **Dynamics:** after the test contact the classes separate (closest pair ≈ 25 px mean keypoint distance at
  the target in a test family). The predictor must turn "where the dot was" into how the block will turn.
- Marker frames and positions are oracle information, stored only in `evaluation_only/`.

Corpus `pusht-cog-v5-marker-20261109` (128/32/32 families; seeds train 2026161000+, validation 2026162000+,
evaluation 2026163000+; test-only 2026159000–029): collection job 22486446, results pending.

## 5. v4 (natural): the harder follow-up

```text
frame 0 | 1–17 approach + pause | 18–33 probe push, 34–36 retreat | 37–58 transit + hold | gap | query ≈ 66–72 | test | target
 first     pusher lines up         block turns by class              block stays turned        present
```

- **Shared start and probe:** in a family, the start pose, the pusher start and one probe push are identical
  for all three classes, so the first frame, every frame before contact and the probe actions are
  class-neutral; only the block's response differs.
- **Probe:** drawn from a library of 360 pushes (any face, offset, tilt, depth; `pusht_cog_matched.py`),
  restricted to pushes whose class turns differ by ≥ 20° and whose shift is ≤ 70 px (102 pushes), then
  jittered.
- **Afterwards:** continuous physics. The pusher's transit, hold, gap and test push are planned from the
  block's actual post-probe pose, so later actions and the present differ between siblings (the turned block
  stays turned).
- **Start:** block centre uniform in [195, 317]² px, angle uniform. A family is rejected if any class fails
  validity (contact outside the probe/test, leaving the arena, test contact before the query); the rule is
  class-symmetric and never looks at outcomes.

**Which history frames carry the cue.** Since the block does not move between the end of the probe and the
test push, the present shows the post-probe pose ("after"). What the present cannot show is where the push
happened: by the query the pusher has gone round to the test side.

| Retained frame | Block in it | Pusher in it | Together with the present |
|---|---|---|---|
| First frame | before the push | on the approach orbit | net motion, but not where it was pushed |
| Pause (≈ 15–17) | before the push | lined up on the push line | net motion **and** push line: strongest single frame |
| Mid-push (≈ 20–33) | partly moved | at the contact point | contact point and remaining motion (direction from that step's action) |
| End of probe (36) | same as the present | backed off along the push line | push line, but no motion |
| Two adjacent probe frames | ~1–2° apart | in contact | too little motion: below I-JEPA pose precision (~3°) |

## 6. What it takes for the scheme to work

### 6.1 Needed for both variants

1. **A PushT predictor.** The Gate predictor summarizes actions as cumulative additive displacements, which is
   wrong for PushT's absolute targets. It needs the ordered absolute targets: the future targets up to the
   horizon, and the target at each retained frame's time (the push direction is implied by target minus
   pusher position). Inputs: the 4-frame present plus up to B = 2 history slots with ages. Train it with
   randomly retained history subsets (none, random pairs, cue frames) so one model is fair to every retention
   policy, as the repo does. Predict the target at query + 12 (the test-push response), plus shorter horizons.
2. **Scorer inputs adapted the same way** (absolute actions instead of cumulative sums).
3. **Evaluation per CoG class:** latent error at the target for recent-only, FIFO, reservoir, temporal
   buckets, visual diversity, prediction surprise, the learned scorer, a fixed cue-frame reference and an
   exhaustive pair oracle (every pair of history frames, scored by the trained predictor). The study is
   meaningful only if the oracle clearly beats recent-only; that gap is what retention can earn.

### 6.2 v5: why it should behave like Gate

v5 has exactly Gate's structure: one short burst of cue frames, visually distinctive, far before the query,
with an identical present. The cue is self-contained in a single frame, so per-frame deletion cost is a
faithful label: deleting a marker frame (if no other marker frame is kept) loses the class, deleting a gap
frame loses nothing. Teacher states at the query work because the baseline rollouts sometimes still hold a
marker frame (reservoir and temporal buckets keep old frames; visual diversity is drawn to the red dot), so
the teacher sees "marker vs ordinary frame" choices to learn from.

Risks to check first:
- **Dot visibility after encoding.** A 2-px dot becomes about one 14-px patch after resizing to 224, and grid2
  pooling averages it with ~64 patches per quadrant. If marker frames are not distinguishable in the features
  (the readability probe "marker frame + query" will show this), enlarge the dot, show it for more frames, or
  use finer pooling (e.g. 4×4 cells).
- **Mapping position to dynamics.** The predictor must learn that a dot on the left of the crossbar means a
  particular turn under the test push. With 128 training families and three classes this is a
  three-way lookup combined with the test-push geometry, similar to Gate's cue-to-lane rule.

### 6.3 v4: why it is much harder

1. **The cue is relational, not a frame.** No single frame contains the class. It is in the relation between
   a push (where and in which direction the pusher acts) and the block's response, read against the present.
   A frame's value depends on which other frame is kept and on what the present shows. Gate's cue frame, by
   contrast, contains the answer.
2. **Cue frames look ordinary.** There is no symbol. A probe frame is a pusher touching a block, which looks
   like any other contact frame, so visual salience does not point to it.
3. **Many probe types, few examples.** The probe is drawn from 102 pushes, so each push type appears in about
   one training family. Learning "push type × response → class" from 128 families is data-starved. Evidence:
   end-to-end I-JEPA probes stay at chance (0.28–0.38) for every history input, while the same probes reach
   0.83 for query + target, where the push type (the test push) is always the same.
4. **The signal is subtle in I-JEPA features.** Decoding pose from the features (with privileged labels, as a
   diagnostic) gives a median 3.1° angle error and 10.7 px pusher error; the classes' turns differ by roughly
   10–40°, and adjacent probe frames differ by ~1–2°. Classifying from decoded poses reaches 0.59
   (first frame + present) against 0.81 from exact poses, with 0.33 chance.
5. **Training states never contain good pairs.** Teacher labels are made only at the query from baseline
   rollouts. FIFO and diversity end the episode holding gap frames; at most a random old frame survives. The
   teacher almost never sees a decision where keeping a complementary probe pair matters, and the student is
   then applied during the probe, a situation it never trained on.
6. **Local credit undervalues the cue.** During the probe the present shows the push and the response
   directly, so history barely helps near-term prediction. A teacher scored on the near future (+3, +6 from
   the decision time) would call probe frames dispensable. The value appears only at the test push, long
   after. Gate hit the same issue: some oracles deleted the cue to lower local error, and use-time labels
   fixed the ranking (`docs/GATE_USE_TIME_PROBE.md`, `docs/GATE_DELAYED_CREDIT_FEASIBILITY_ZH.md`).
7. **Greedy, irrevocable decisions must plan ahead.** To keep a pre-contact frame and a later probe frame, the
   policy must hold the early frame while many similar probe frames arrive, and must not keep two adjacent
   frames (redundant, too little motion).
8. **A small arena leak into the present.** The same probe moves each class differently and the walls and the
   fixed goal are visible, so the block's arena position in its own frame depends on the class: 0.37–0.40
   from exact simulator state. It does not show in I-JEPA features (present-only probes 0.29–0.33), but it is
   a confound to watch.

### 6.4 v4: ways to solve it

**A. Make the cue learnable** (data and encoding)
- Fewer probe types with more examples each: a few push families (e.g. stem tip, crossbar faces) with jitter
  in offset, tilt and depth, instead of 102 library pushes. This trades a little present leak for a much
  more learnable cue (v2's fixed stem-tip probe gave 0.98 for first frame + query).
- More families: the collector is cheap (minutes on CPU); 1–2k training families mainly cost encoding time
  (~16 min per 50k frames on an A100).
- Finer spatial features: grid2 pooling discards most geometry. 4×4 cells or patch tokens would improve pose
  precision; this changes the predictor input size, so it needs its own basis fit.
- Longer or slower probes (more turn per push) widen the class separation relative to pose noise.

**B. A predictor that can use relational history**
- Give each slot its absolute action at that time, so the push direction is explicit.
- Let the slots and the present interact (e.g. attention between slots and the current frame) rather than
  only a weighted average; the existing velocity term relates slots to the present, which may be enough. The
  pair oracle tells whether it is.
- Train with retained sets that include cue pairs (pause + mid-push, two well-separated probe frames) as well
  as random ones, so the predictor learns to use them.

**C. Target-time credit for the teacher**
- Score every deletion by the predictor's error at the episode's target (query + 12), assuming the kept frames
  survive until the query, instead of the near future of the decision time. This values probe frames for what
  they do at the test push.

**D. DAgger-style training states**
- Roll out the current student (mixed with an oracle or random-keep policy with decreasing probability),
  record the candidate triple at **every** arrival, label each with the target-time teacher, add to the
  dataset, retrain, and repeat. The student then learns from the states it actually visits, including
  "mid-probe, holding the pause frame".
- Seed the first round with rollouts that keep cue frames (the oracle pair), so good states exist from the
  start.

**E. Scorer architecture**
- The scorer already sees all three candidates with a pooled context; pairwise interaction (attention among
  candidates and with the present) would let it judge complementarity ("these two frames span the push")
  rather than per-frame salience.

**F. Order of work and success criterion**
1. Train the predictor with random history; measure the target error of recent-only versus the exhaustive
   pair oracle and the fixed references (pause + present, mid-push + present, first frame + present, end of
   probe + present). If the oracle barely beats recent-only, fix A and B before any retention work.
2. Check the target-time teacher: on states that contain a cue pair, does it rank the cue frames as costly to
   delete?
3. Train the scorer with DAgger and compare.

v4 succeeds if the learned scorer approaches the pair oracle and beats the best single-frame reference;
matching only single-frame references would mean it is keeping a salient frame, not an interaction.

## 7. Version history and results

| | Design | Status | Code |
|---|---|---|---|
| **v5** | CoG shown as a dot for 3 early frames; no probe; siblings identical except those frames | current (Gate-like testbed) | `pusht_cog_marker.py`, `prepare_pusht_cog_marker_data.py` |
| **v4** | Natural continuous episodes; shared start and probe per family | current (harder follow-up) | `pusht_cog_natural.py`, `prepare_pusht_cog_v4_data.py` |
| v3 | Matched probes (each class pushed differently to a shared pose) | abandoned (dev only) | `pusht_cog_matched.py` (+ probe library) |
| v2 | Continuous; class-specific start poses calibrated to a shared post-probe pose | superseded | `pusht_cog_coherent.py`, `prepare_pusht_cog_v2_data.py` |
| v1 | Probe, then a scene cut (teleport) to a fixed test pose | kept | `pusht_cog_cue.py`, `prepare_pusht_cog_v1_data.py` |

**v4 results.**
- Dev (seeds 2026140000–191), simulator-state classifiers, family-grouped 5-fold CV, chance 0.33:

  | Start box | Valid | Present | First frame | End of probe | First frame + present |
  |---|---:|---:|---:|---:|---:|
  | [180, 332], shift ≤ 90 | 125/192 | 0.34 | 0.33 | 0.50 | 0.75 |
  | [195, 317], shift ≤ 70 (used) | 168/192 | 0.40 | 0.33 | 0.60 | 0.83 |

  Counterfactual spread at +12 (closest pair of CoGs replayed from the same present): 21 / 26 / 30 px
  (10th/50th/90th); the true-class replay reproduces the run to 3e-13 px.
- Corpus `pusht-cog-v4-20261107` (job 22151809, 85 s on 32 CPUs): 128 / 32 / 32 families (384 / 96 / 96
  episodes; 34,095 / 8,526 / 8,532 frames); 50 candidates rejected (48 out of bounds, 2 approach contact,
  2 transit/gap contact). Seeds: train 2026151000+, validation 2026152000+, evaluation 2026153000+.
- I-JEPA readability (course A100, 940 s; validation, linear/MLP): recent window 0.30/0.29; first frame
  0.33/0.33; end of probe 0.27/0.30; first frame + query 0.32/0.31; target 0.26/0.29; query + target
  0.30/0.83. Probes over every probe frame (+ first frame + query) stay at 0.28–0.38.
- Pose decoding (`diagnose_pusht_cog_pose_decoding.py`, `results/pusht_cog/v4_pose_decoding_20261108.json`):
  per-frame regressor from I-JEPA features (StandardScaler, PCA 256, MLP 512-256) on all ~34k train frames
  with privileged simulator poses; validation median error 4.2 px / 3.1° / 10.7 px (block centre / angle /
  pusher; ridge 7.0 px / 9.1° / 18.5 px). CoG from decoded poses: present 0.33, first frame + present 0.59,
  end of probe + present 0.52 (exact poses 0.36 / 0.81 / 0.63). Diagnostic only.

**v3 (abandoned).** Every sibling shares the start pose; each class gets its own probe, solved by least
squares (offset, tilt, depth) on the real episode prefix so the block lands on the anchor class's post-probe
pose. Exact when it converges (≤ 5e-13 px), but only 6/32 and then 11/32 dev families matched, failed solves
stalled at a median 1.4 px, and every matched family had a turn within ±2°: pure translation, which requires
pushing through each class's own CoG, so the contact location alone revealed the class (jobs 22139069,
22141060).

**v2 (superseded).** Each class starts at the shared post-probe pose with its own probe motion undone (PushT
is invariant to rigid motions of the scene), so all classes end the probe at one pose (≤ 1.7e-12 px) and then
share every action. Corpus `pusht-cog-v2-20261102`. I-JEPA probes (job 21664867, validation, linear/MLP):
recent window 0.333/0.333; first frame 0.625/0.573; end of probe 0.667/0.844; first frame + query
0.698/0.979; target 0.385/0.333; query + target 0.271/0.917. Superseded because the start pose relative to
the arena reveals the class (0.76 from two state numbers).

**v1 (kept).** A stem-tip probe, then a scene cut: block and pusher are teleported to a fixed test pose, so
siblings share all actions and are identical from the cut through the query. Corpus
`pusht-cog-v1-20261101`; `prepare_pusht_cog_v1_data.py` is byte-identical to the collector that produced it
(SHA-256 23aaaa10…612161). I-JEPA probes (job 21615935, validation, linear/MLP): recent window 0.333/0.333;
cue frame 0.792/0.812; first frame + cue 0.667/1.000; target 0.333/0.344; query + target 0.302/0.875.

## 8. Implementation notes

- Placing the block sets its angle before its position; pymunk rotates about the CoG, so the reverse order
  shifts the block by a class-dependent amount.
- Chipmunk keeps an overlap-correction (bias) velocity from the last solve and applies it on the next step.
  `place()` takes one step at the new pose to spend it, then places again; without this the block drifted
  1–2 px with no contact.
- On headless Linux use `opencv-python-headless==4.11.0.86` (the regular wheel needs `libGL.so.1`).
- The I-JEPA image processor in transformers 5.5.4 needs torchvision installed.

## 9. Data and handoff

Data are not in Git (`build/` is ignored). On Torch, under `/projects/work/yang-lab/users/yz12129/CV_project/build/`:

| Path | Size | Contents |
|---|---:|---|
| `pusht-cog-v5-marker-20261109/` | ~60 MB (expected) | v5 corpus, all three splits |
| `pusht-cog-v4-20261107/` | 60 MB | v4 corpus, all three splits |
| `pusht-cog-v4-20261107-ijepa/` | 833 MB | v4 train+validation I-JEPA features (5120/frame), probes, burst log |
| `pusht_cog_probe_library_v3.json` | 0.2 MB | probe library used by v3/v4 |
| `pusht-cog-v1-20261101*`, `pusht-cog-v2-20261102*` | ~45 MB + ~1 GB features | earlier versions |

I-JEPA weights: `facebook/ijepa_vith14_1k@f157467` (2.4 GB) from Hugging Face. All corpora regenerate on CPU
in minutes from the collectors (seeds and protocols are fixed); the evaluation splits have not been read by
any model.

**Course GPU** (`csci_ga_2271-2026fa`): only through ood.burst.hpc.nyu.edu code-server sessions. Their
Singularity container exposes `/dev/nvidia*` but not the NVIDIA user-space driver: extract the driver package
matching `/proc/driver/nvidia/version` (610.43.02 on 8 November;
`https://download.nvidia.com/XFree86/Linux-x86_64/610.43.02/NVIDIA-Linux-x86_64-610.43.02.run`) with
`--extract-only` into the home directory, symlink `lib*.so.<ver>` to `.so.1`, and prepend it to
`LD_LIBRARY_PATH`. `experiments/phase2/slurm/cloud_burst/make_package.sh` builds the full package (code, data,
weights); `make_data_addon.sh` adds another dataset to an existing package folder.

**Next steps:**
1. Encode v5 (train+validation) and run the readability probes (marker frame + query should be high, the
   recent window at chance).
2. PushT predictor (section 6.1) and the latent prediction evaluation per CoG class with recent-only,
   baselines, the marker-frame reference and the pair oracle.
3. Deletion-cost scorer on v5; then v4 following section 6.4.
4. Planning (MPC with the learned predictor).

## 10. Reproduce

```bash
.venv/bin/python -m pytest -q tests/test_pusht_cog_cue.py tests/test_pusht_cog_coherent.py \
  tests/test_prepare_pusht_cog_v2_data.py tests/test_pusht_cog_natural.py tests/test_pusht_cog_marker.py
# Corpora (CPU, Slurm scripts in experiments/phase2/slurm/)
sbatch experiments/phase2/slurm/run_pusht_cog_marker_collect.sbatch   # v5
sbatch experiments/phase2/slurm/run_pusht_cog_v4_collect.sbatch       # v4 (dev check + corpus)
# I-JEPA encoding + probes: lab GPU (run_pusht_cog_encode_probe.sbatch) or the course package
experiments/phase2/slurm/cloud_burst/make_package.sh build/pusht-cog-v4-20261107 HF_CACHE OUT_DIR
```

Encoding used torch 2.5.1+cu121, torchvision 0.20.1, transformers 5.5.4, numpy 1.26.4,
`facebook/ijepa_vith14_1k@f157467`, grid2 pooling (5120 features).

## 11. Limits

The cue is the block's response to a push (v4) or a drawn symbol (v5), not momentum. Moment of inertia is the
default for every class. Readability probes do not show that a predictor learns to use history; the latent
prediction evaluation with retained history and the planning study have not been run.
