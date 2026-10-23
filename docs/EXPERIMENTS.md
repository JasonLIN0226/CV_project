> Historical development design. For the completed fresh confirmation, use the current report and PHASE1_LATEST.md.

# Experimental design

## Overall goal and this stage

The overall goal is a past-only, bounded-memory retention decision. The project asks whether additional prediction error after deleting an observation can supervise a retention scorer.

The original early phase tests prerequisites: the encoder preserves evidence, historical evidence can improve prediction, and prediction responds to that evidence. The later equal-capacity deletion assay is a bridge toward retention research. It is not a learned retention policy.

## Data and processing

We use two controlled synthetic tasks, not a public video benchmark:

- **Gate:** an early geometric cue disappears; the cue and the known action determine the later lane.
- **Motion:** two earlier observations reveal movement; a shutter then hides the object, requiring prediction from past evidence and actions.

Each matched pair has identical recent RGB and action sequences but different earlier evidence and future outcomes. The statistical unit is a base pair, not each sibling or each deletion fork.

Per task: 512 training pairs, 96 validation pairs, and 128 pairs in each of three evaluation conditions. That is 992 pairs / 1,984 episodes per task, or 3,968 episodes across both tasks. Data seed 6301 has been inspected during development and is not untouched confirmation data.

Frames are rendered at 96 × 96 RGB. The frozen official I-JEPA encoder is pooled into four spatial cells, yielding 5,120 features. Normalization and the 128-dimensional input basis are fitted using training frames only. Prediction is scored in the full normalized 5,120-dimensional output. Hidden simulator state and labels are diagnostic/scoring information, not predictor inputs.

## What the three conditions mean

**Ordinary** uses familiar generation settings. After the final historical evidence frame, 5–10 intervening steps precede the four-frame recent window. Motion is already occluded in this condition; ordinary does not mean fully visible.

**Long gap** changes that interval to 20–32 steps. Gate must use a cue that disappeared longer ago; Motion must extrapolate through a longer hidden interval. The recent window and future horizons are unchanged.

**Distractor shift** keeps ordinary timing and the task rules. Irrelevant objects outside the arena change from three circles to seven rectangles with a different palette. It changes visual distractions, not the correct cue or movement rule.

```text
Earlier evidence → intervening gap → 4 recent frames → query → future +3 / +6
Ordinary                 5–10
Long gap                20–32
Distractor shift         5–10, with changed irrelevant scenery
```

Gap is elapsed time **before** the recent window; horizon is the forecast distance **after** the query.

## Tests and their meaning

| Test | Comparison | What it establishes |
|---|---|---|
| T1 Evidence readout | A supervised linear readout of historical versus recent features | Whether the clue is readable after encoding |
| T2 History benefit | Privileged two-memory versus recent-only prediction | Whether supplied evidence improves future prediction |
| T3 History replacement | Correct versus paired wrong history with the same recent inputs/actions | Whether historical content affects the forecast |
| T4 Fixed reference | Current predictor versus a fixed earlier predictor | Whether repair improves a stable comparison |
| T5 Critical deletion | Delete critical evidence versus nuisance content, always retaining two entries | Whether deletion cost distinguishes useful memories at equal capacity |
| T6 Cross horizon | Select a removal at +3, score at +6, and reverse | Whether the oracle deletion choice transfers across the two horizons |
| T7 Policy advantage | Learned past-only scorer versus strong streaming rules | Not run; requires a deployable retention policy |
| T8 Fresh confirmation | Locked procedure with fresh data and newly fitted models/transforms | Not run; requires independent confirmation |

### Why T1 is exactly 50% versus 100%

Recent inputs in a pair are identical and labels are opposite. A deterministic recent-only classifier returns the same answer for both, so exactly one is correct: 50%. The historical readout is trained to decode deliberately clear, explicitly supplied evidence and achieved 100% on the evaluated examples. The original values are 0.5 and 1.0, not rounded values.

T1 directly accesses cached earlier features. A longer later delay does not erase those features. Thus long gap can remain easy for readout while making future prediction hard. This result does not establish exact velocity estimation, accurate future position, reliable retention, or perfect performance on new data.

## Predictor and deletion settings

The latest complete predictor experiment uses both tasks and training seeds 17/29/43: six models, 120 epochs and 7,680 steps each. The centered historical correction reads every retained entry; no selector or retrieval policy is trained. The earliest validation minimum over four balanced conditions and two horizons selects the checkpoint.

The deletion assay reuses these models. Each privileged parent has three real historical entries. Gate uses a cue plus two post-cue nuisance frames. Motion uses two motion-evidence frames plus one shutter frame. Every fork deletes exactly one entry, leaving two memories and identical recent context/actions. Two nuisance-time variants and both siblings are averaged within each base pair.

T5 requires at least 10% aggregate benefit and a positive paired 95% lower confidence bound in both feature and task error for each critical role/seed/horizon. T6 uses future outcomes to choose a deletion and therefore is an oracle diagnostic, not an online policy. Its advantage must transfer across horizons in both metrics. The primary joint readiness rule is ID-bounded; the other conditions separately delimit scope.

All original thresholds and failures are retained. No majority-pass threshold is substituted after inspecting outcomes.
