# Hidden-CoG PushT: predictor and retention (run of 20–24 November 2026)

Plain PushT needs no memory: the block moves only while pushed (zero damping) and actions are absolute
pusher targets, so the present frame and the actions determine the future. We hide a **discrete centre of
gravity** (left, centre or right; body-local (−40, 20), (0, 45), (40, 20)) in the T block. The block looks
the same in every class, but a later **test push** on the crossbar turns it differently, so predicting the
future needs a past cue. The PushT end goal is planning; this run evaluates latent prediction and retention.
The design history up to 9 November is in `docs/PUSHT_COG_CUE.md` (frozen); the evolving write-up lives on
branch `pusht-cog-cue`.

A *family* is one scenario run once per class (three *siblings*). The model sees the 4-frame present
(query−3 … query) plus B = 2 retained history frames; the target is the I-JEPA latent at query + 12.

| Version | Cue | Role |
|---|---|---|
| **v5c** | the CoG drawn as a red dot (radius 4 px at 96×96) on the block for 3 frames starting at frame 2–10, plus 1–2 distractor dots (black, orange, purple or the marker red) on empty background in every history frame | **main testbed** |
| v5b | the same dot without distractors | the dot is the most salient frame |
| v5 | a 2-px dot | too faint in I-JEPA features |
| v4 | no symbol: the block's response to a shared probe push | harder follow-up |

In v5–v5c the block is untouched until the test push and all actions are shared, so siblings differ only in
the marker frames: the present carries **no** class information (probes at exactly 1/3) and the future depends
strongly on the class (query + target probe 0.99). Distractors are seeded per family and frame, so siblings
share them; the present and the targets are clean.

## Files

| File | What |
|---|---|
| `pusht-cog-v5c-marker-r4-d2-t960-20261123_retention.json` | v5c retention, 3 seeds (jobs 24248904 collect, 24248908 encode, 24248947 predictor, 24248950 retention) |
| `pusht-cog-v5c-marker-r4-d2-t960-20261123_predictor.json` | v5c predictor, fixed memory conditions, 3 seeds |
| `pusht-cog-v5b-marker-r4-t960-20261120_retention.json` | v5b retention, 3 seeds (jobs 24128853, 24128855) |
| `pusht-cog-v5b-marker-r4-t960-20261120_predictor.json` | v5b predictor of that retention run |
| `pusht-cog-v5b-marker-r4-t128-20261120_predictor.json` | v5b predictor with 128 training families (job 23856857) |
| `v5_marker_basis.json`, `v5b_marker_basis.json` | does a predictor-style PCA basis keep the dot? (2 px vs 4 px) |

All on Torch CPU nodes, encoding on an H200. Corpora, features and checkpoints (~7 GB for v5c) are on a
teammate's project storage (`/projects/work/yang-lab/users/yz12129/CV_project/build/`); they regenerate with the
tasks `pusht.prepare_data`, `pusht.encode`, `pusht.train_predictor`, `pusht.retention` in `configs/reproduction.yaml`.

## Predictor

`src/bounded_memory/environments/pusht/train_pusht_cog_predictor.py`: predicts standardized I-JEPA latents at
+6/+12 from the present, up to two history slots (latent, age, pusher target) and future absolute pusher
targets, through a 128-dim input basis that includes sibling frame differences (non-zero only at marker frames,
so the faint marker direction is kept). Trained with random memory conditions (none, random pair, random single,
marker frames) so one model serves every policy. It needed data: with 128 training families keeping the marker
lowered the +12 error by 7% (sibling identification 0.33 → 0.59); with **960** families by 39% (0.33 → 0.91).

## Retention

`src/bounded_memory/environments/pusht/retention_pusht_cog.py`: online B = 2 memory; baselines use the Gate
rules of `evaluation/heuristics/gate.py` (FIFO, reservoir, temporal buckets, visual diversity) plus random
deletion; prediction-surprise is not included. The deletion-cost scorer is Gate v2's set-conditioned student
with a **target-time teacher** (each deletion scored by the frozen predictor's error at the episode's +6/+12
targets with the two kept frames), **training states along whole rollouts** of the baselines (~184k), and two
**DAgger** rounds on the student's own states (~257k). Validation (96 episodes), mean of seeds 17/29/43, 960
training families. Error: mean squared error over 5,120 standardized features. Sibling ID: how often the
prediction is closest to its own sibling's future (chance 1/3).

| Policy | v5b: +12 error / sibling ID / marker kept | **v5c: +12 error / sibling ID / marker kept** |
|---|---|---|
| Recent only | 0.810 / 0.33 / 0% | 0.743 / 0.33 / 0% |
| FIFO | 0.810 / 0.33 / 0% | 0.743 / 0.33 / 0% |
| Temporal buckets | 0.810 / 0.33 / 0% | 0.744 / 0.33 / 0% |
| Random deletion | 0.810 / 0.34 / 0% | 0.745 / 0.33 / 0% |
| Reservoir | 0.764 / 0.41 / 13% | 0.702 / 0.40 / 13% |
| Visual diversity | 0.498 / 0.89 / 99% | 0.618 / 0.57 / 44% |
| **Deletion-cost scorer** | **0.501 / 0.90 / 100%** | **0.473 / 0.87 / 99.7%** |
| Marker frames (privileged reference) | 0.499 / 0.90 / 100% | 0.474 / 0.86 / 100% |

"Marker kept" counts episodes whose memory at the query holds at least one marker frame. On v5b the scorer keeps
one in all 288 episode runs (96 episodes × 3 seeds); on v5c in 287 of 288 (96 / 96 / 95; one miss in seed 43),
with +12 error 0.469 / 0.486 / 0.465 per seed, matching the privileged reference. Visual diversity, which tied it
on v5b (285 of 288), keeps a marker frame in 45 / 41 / 40 of 96 episodes on v5c once distractors make other
frames just as distinctive.

**Why ours lands on the oracle.** The "oracle" here is a fixed privileged rule (keep the first and last marker
frames), not the best possible memory: the best pair per episode, chosen with the target, reaches ~0.35 at +12.
That best-pair value is not attainable by a retention policy: beyond keeping a marker frame, its gain comes from
using the target to pick frames that sway the predictor, partly through predictor-specific quirks.
What the v5c scorer keeps at the query (validation, from the saved
round-2 scorers):

| Seed | 2 marker frames | 1 marker + 1 other frame | No marker frame |
|---|---:|---:|---:|
| 17 | 72 | 24 | 0 |
| 29 | 73 | 23 | 0 |
| 43 | 67 | 28 | 1 |

+12 error with the same predictor for different memory contents:

| Seed | Ours | Reference (first + last marker frame) | Only the middle marker frame, other slot empty |
|---|---:|---:|---:|
| 17 | 0.4689 | 0.4691 | 0.4637 |
| 29 | 0.4861 | 0.4886 | 0.4795 |
| 43 | 0.4652 | 0.4629 | 0.4856 |

A single marker frame carries almost all the value: the error with one marker frame and an empty second slot is
within ±0.02 of the reference's two marker frames, in either direction depending on the seed. The three marker
frames are near-duplicates (same dot, same block pose) and the predictor ignores ordinary and distractor frames.
So the scorer and the reference reach almost the same error although they often keep different frames: in about
a quarter of episodes the scorer keeps one marker frame plus another frame, and when it keeps two, often a
different pair than first + last.

## Why the deletion-cost scorer works here

1. **The teacher measures what matters.** `teacher_energy` runs the frozen predictor once per candidate with
   the other two kept, at the query, against the episode's real +6/+12 latents; `soft_target` turns the three
   errors into a deletion distribution. The predictor's +12 error depends almost only on whether a marker frame
   is kept (0.47 with, 0.74 without), so deleting the marker is costly and deleting anything else is nearly free.
2. **Prediction-aligned, not salience-driven.** The scorer and visual diversity read the same 128-dim
   projection. Visual diversity keeps the most spread-out pair, which distractors inflate. The predictor learned
   to ignore distractors (they never appear in targets; keeping one never lowers the error), so the teacher never
   rewards them.
3. **Target-time credit.** The marker arrives ~40 frames before the query; scoring every decision at the
   episode's target (not the near future of the decision clock) values it when the decision is made, the
   use-time fix Gate also needed.
4. **The right training states.** Every baseline rollout passes the step where the marker frame arrives, so
   that decision is labelled in every episode; DAgger adds the student's own "hold the marker as each new frame
   arrives" states, which it must get right ~40 times per episode.
5. **Set-conditioned scoring.** Each candidate is judged relative to the other two; once one marker frame is
   kept the second slot barely matters, so the scorer reaches the reference exactly.
6. **An enabler: the input basis** includes sibling frame differences, which make the marker's direction
   explicit. Visual diversity uses the same projection, so this is not an advantage between the two, but it
   relies on the dataset's sibling pairing.

## Caveats and next steps

The teacher uses the frozen predictor and the true future latents at training time (run-time inputs are past
frames and actions only). v5c's cue fits in a single frame, whereas v4's is relational (a push and its
response, read against the present), which tests set conditioning and DAgger harder. All numbers are on 96
validation episodes; the test split is untouched. The literature baselines (PaFu-KV, Compress to Remember,
FrameMorrow) have not been run on PushT. Next: v4 with the same pipeline, then planning with the learned
predictor.
