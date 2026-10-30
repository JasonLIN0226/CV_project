# Gate TRAIN label timing audit — 2026-10-24

This is an audit of already saved labels, with no predictor calls, GPU use,
new fitting, confirmation-set access, or changes to frozen confirmation.

**The saved local deletion-loss labels frequently favor completely losing the
cue before the outcome is visible.** Among 130 old TRAIN parents whose two
targets are no later than query and where a single deletion can remove the
last resident cue, 111 (85.4%) have their minimum saved loss at that deletion.
The saved future-attention teacher favors that deletion in 0/130 parents.
This is direct evidence about the supervision; it does not yet establish the
cause of the learned scorer's confirmation failures or a useful new method.

## Inputs and boundaries

- Inputs: `build/literature-diagnosis-20261022/learnability-v2/training_inputs_labels.npz`
  and `training_metadata.json`, plus the verified old TRAIN snapshot at
  `build/hpc-20261020/development-assets`.
- Population: 1,282 K+1 parents from 32 episodes / 16 scene pairs. Multiple
  parents from an episode are correlated; percentages here are descriptive
  parent counts, not independent-sample estimates.

The previously local-only input/label file is now exported byte-for-byte as
[`training_inputs_labels.npz`](../results/literature_diagnosis/use-time-label-audit-20261024/training_inputs_labels.npz)
(1,184,698 bytes; SHA256 `a5df502b955f03a2184c48b383758f3e2b13e4657b171e8fa8070f655a468976`).
It contains float64 inputs of shape `(1282,3,903)` and the original cost,
attention and shuffled-cost labels; this matches the pre-existing source hash.
Parent metadata and regenerated cue annotations remain in the adjacent audit
files. Exporting it does not change any fitted result.
- Saved cost column j is mean h3/h6 feature MSE after deleting sorted candidate j.
  The training code anchors the two horizons at each overflow decision time t,
  not just at the episode's final query q.
- Saved attention column j is the original future-attention teacher's salience
  for candidate j. All 1,282 columns were checked against original teacher
  targets after sorting candidate times. This audits teacher labels, not a
  learned PaFu student's predictions or the online attention baseline's score.
- Both policies prefer deleting the lowest score. Minimum sets include all
  entries within absolute tolerance 1e-8 of the minimum, with relative
  tolerance zero. No arbitrary tie break is used to claim cue destruction.
- Cue annotations are used only for these diagnostic strata. A cue copy means
  exact equality of the full normalized feature vector to the annotated cue
  frame. A deletion completely loses the resident cue only when the parent
  has exactly one such historical candidate and none in its R=4 recent window.

The old TRAIN evaluation annotations confirm that outcome classes first become
visible at **q+1**, while the query frame q still precedes the outcome. Thus the
previous shorthand “520 targets all before query” needs a boundary correction:

| Relationship of the two targets t+3 / t+6 to query q | Parents |
|---|---:|
| Before / before | 418 |
| Before / exactly at | 102 |
| Before / after | 231 |
| Exactly at / after | 129 |
| After / after | 402 |

The first two rows form the 520 parents with **both targets before outcome
visibility**. Only 418 are strictly before query.

## What the saved labels prefer

The “last-cue risk” denominator includes only parents where one deletion can
completely erase the resident cue. It is not all 1,282 parents.

| Target timing | All parents | Last-cue risk parents | Cost minimum deletes last cue | Attention minimum deletes last cue | Median loss of deleting cue minus best preserving deletion |
|---|---:|---:|---:|---:|---:|
| Both at/before q | 520 | 130 | 111 / 130 (85.4%) | 0 / 130 | −0.031818 |
| t+3 ≤ q < t+6 | 360 | 162 | 85 / 162 (52.5%) | 5 / 162 | −0.001984 |
| Both after q | 402 | 182 | 13 / 182 (7.1%) | 6 / 182 | +0.058934 |
| All | 1,282 | 474 | 209 / 474 (44.1%) | 11 / 474 | +0.007544 |

A negative margin means the actual saved reader loss is lower after destroying
the cue. A positive margin means a preserving deletion is better. It is a
comparison between choices in the same K+1 parent; it is not a controlled cue
content intervention with timestamps and actions held fixed.

Candidate cue counts across all parents are 0: 510, 1: 524, 2: 234, and 3: 14.
Of the 524 single-cue-candidate parents, 50 still have a cue copy in recent
memory, so deleting that candidate does not yet erase all resident evidence.

The cost labels have no tied minima at tolerance 1e-8. Their median within-parent
range is 0.009673 (minimum 0.000130, maximum 0.310445). In the 130 pre-outcome
last-cue-risk parents, the median range is 0.037260. The observed preference is
therefore not merely an arbitrary resolution of numerical ties. Attention has
1,076 tied minimum sets, often among redundant non-cue candidates; tied sets
are preserved in the per-parent output. Cost and attention values have different
units and should not be compared by numerical scale.

## First illustrative parent

Selection is post hoc but deterministic: take the first saved parent where
both targets are before outcome visibility, cost prefers erasing the last cue,
and attention prefers preserving it. This is parent index **5**, not the parent
with the largest gap.

- Episode `6064f0d6f5f5f29cc902acff`, scene pair `1c8791cf7ed67b87bc9b649d`.
- Cue at frame 7; current decision t=11; candidates [5, 6, 7]; recent [8,9,10,11].
- Query q=21; outcome first appears at 22. The current label targets are 14 and 17.

| Delete frame | What survives | Saved local loss | Saved attention salience |
|---|---|---:|---:|
| 5 | 6 and cue 7 | 0.165912 | 0.151694 |
| 6 | 5 and cue 7 | 0.166116 | 0.151694 |
| 7 | 5 and 6; cue completely gone | **0.106379** | **0.696611** |

The local cost label favors removing the cue by 0.059533 relative to the best
preserving deletion. The attention teacher protects it. The two target feature
vectors happen to be identical, but the three actual saved losses are plainly
different. **Equal future target features do not imply equal reader losses.**

This is an example of competing supervision signals. It is not yet evidence
that the attention choice has lower eventual query loss for this same parent;
that requires a continuation intervention.

## What this supports, and what remains unresolved

The result motivates checking whether a decision-local horizon supervises the
wrong behavior when a cue matters only later. It strengthens the hypothesis
beyond observing early cue loss in an online trace: saved TRAIN labels really
do prefer that loss in many relevant parents.

The timing strata contain different parents, candidate sets, decision times,
and episodes. They are not a causal intervention on horizon. Reader misuse,
the feature-MSE objective, and changing candidate content/timestamps/actions
can still contribute. The 13/182 post-outcome parents that prefer complete cue
loss also rule out treating timing as the sole explanation.

There is an additional reader-domain confound. The frozen reader's training
path, `train_predictor_followup.py:129–134`, calls `sample_predictor` only with
`query=rec["query"]`. Its supervised targets are therefore the recorded
query+3/query+6 outcomes. In contrast, the local cost teacher calls this reader
at earlier overflow decisions t, including pre-outcome static targets that
were not supervised as prediction origins in that training path. A label that
actively rewards dropping a cue can consequently reflect the reader's behavior
outside its training-origin distribution as well as a short utility horizon.

A use-time teacher which continues to the recorded query changes both the
target time and the reader's prediction-origin domain. Even a clear same-parent
ranking reversal would not isolate a pure horizon effect. Separating these
requires, for example, a reader trained/validated across decision origins,
followed by a controlled objective comparison. Identical target features, or
identical targets across siblings before branching, do not establish noiseless
deletion labels, equal reader losses, or that deleting the cue is safe for the
eventual task.

The next controlled comparison should start from the same K+1 state, force
each of its three deletions, continue with one fixed causal policy without
restoring evicted evidence, and evaluate at the same eventual query. Compare
that use-time ranking with the saved local ranking. Keep the reader, capacity,
training feature map, model capacity, and held-out evaluation protocol fixed.
If labels improve but a matched scorer does not, distinguish label quality
from causal learnability and approximation failure.

## Reproduction and outputs

Run from the repository root, using a new empty output directory:

```bash
python3 experiments/phase2/audit_use_time_labels.py \
  --output build/use-time-label-audit-rerun
```

The saved audit outputs are under
`results/literature_diagnosis/use-time-label-audit-20261024/`:

- `parent_records.jsonl`: all 1,282 rows, index aligned with the saved training
  arrays, including candidate cue masks, recent cue copies, target timing,
  all three labels, tie sets, margins, and complete-loss flags.
- `summary.json`: aggregate counts, label distributions, timing/cue-count/
  complete-loss strata and their intersections.
- `protocol.json`: definitions, tolerance, scope, and alignment checks.
- `sourcehashes.json`: SHA256 of every read source/input and the output files.

The script verifies snapshot checksums, TRAIN-only identities, producer source
hashes, candidate-column alignment, and q+1 outcome onset. No existing
confirmation protocol or result was modified.
