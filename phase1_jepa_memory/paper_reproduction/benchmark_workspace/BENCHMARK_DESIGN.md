# Assembly101 paper-based benchmark: prespecified reconstruction

This supplements the author release's missing decoder and benchmark interfaces.
It is not a claim that unpublished author implementation details are known.
No action score, decoder fit, or policy benchmark has yet been run.

The predictor uses all 219 training recordings and all 75 validation recordings.
Only one training recording lacks released annotation rows. Its 411 observations
remain in predictor/basis/bank training, but have a false label-availability mask.
All 75 validation recordings and their 75,039 observations have verified indexed
labels. The decoder must never interpret an unknown label as background. This
coverage policy was fixed before any semantic result or decoder fitting.

## Train-only task decoder

Use a linear 2048-to-24 multi-label decoder for each predictor/policy seed
0, 1, and 2. Its architecture is a disclosed reconstruction because the paper
and release do not specify the decoder training implementation. Fit only the
172,887 labelled training observations (218 recordings), using L2-normalized
true TSM features and current 24-verb multi-hot labels. Do not fit on validation
frames or future validation labels. Normalize predicted future features to the
same unit-vector domain before applying the frozen decoder. Use 30 epochs,
batch128, AdamW3e-4/wd0.05, five-percent linear warmup then cosine decay,
BF16, and gradient clipping at1.0. Select the earliest minimum full-training BCE
checkpoint; validation labels are not used to select decoder hyperparameters.
All seeds, epochs, and available supervised observations remain fixed.

## Matched online comparisons

The main reproduction scope is the core six methods: recent-only, FIFO,
Reservoir, SPACE, utility-only, and state-only. All share the same released
features, frozen predictor, decoder, memory capacity16, recent buffer8, and
horizons1/4/16/64. Score every full-memory, complete-future anchor: observation
indices23 through L-65 inclusive, giving L-87 queries per recording. All earlier
observations still update memory causally; they are not a selected warm-up subset. Use the published SPACE configuration directly, without a
validation threshold search. Each policy receives only the incoming observed
event, recent context, and its own retained memory; it cannot reload deleted
validation payloads. Utility-only selects the lowest retrieved mean cost among actions satisfying
the original conservative advantage/support thresholds against Reservoir,
without departure/reachability geometry; exact cost ties use action index. State-only removes utility filtering
while retaining the same departure and reachability logic. These two comparison
adapters are disclosed reconstructions. Other literature adapters and the
paper's cross-dataset appendices are outside this primary Assembly101 scope.

## Scores and aggregation

Action NLL is the sum of24 binary negative log-likelihoods, averaged uniformly
across horizons and eligible queries. Report each horizon and the overall score.
Report absolute and relative NLL changes against matched Reservoir, together
with mAP, LRAP, and Recall@5. Primary mAP averages all24 classes per recording;
a class with no positive label receives AP0. LRAP and Recall@5 use full known
streams; an empty target label set receives1 under the no-ranking-error
convention. Also report the background fraction and foreground-only ranking
metrics descriptively so this convention cannot conceal task weakness.
These conventions are disclosed reconstructions, fixed before any score.
Also report the frozen decoder NLL on true future features at exactly the same
queries as an evaluator-only task-readout reference; it is not a deployable
policy or an acceptance criterion.
Average query measurements within a recording, recordings within each
participant, and participants equally. Report mean and sample SD over the three
complete seeds. Paired inference first differences methods within recording
and seed, averages within participant and across seeds, then bootstraps the10
participants with10,000 seed0 replicates. Also report an exact one-sided sign
flip test. These are development comparisons, not sealed confirmatory results.

Use the frozen shared slow predictive state for drift: consecutive states with
the same nonempty active-verb set are within-regime transitions. Onsets and
changes to another nonempty set are semantic boundaries. Exclude foreground
to background from boundary adaptation. Report absolute within-regime drift
and log boundary-to-within displacement ratio, not just a normalized gain.

## Controlled recovery

Prespecify four uniformly spaced eligible anchors per validation recording,
using label/chronology eligibility only. Anchors must have a nonempty unchanged
verb set through16 subsequent updates, all four forecast targets, and an earlier
full-memory state with a different nonempty verb set. If no such state exists,
report the recording as ineligible; never manufacture a donor. Replace25% or50%
of slots with past donor payloads, retaining timestamps and slot metadata.
The evaluator alone performs this intervention. Recent-only has no retained
payload to corrupt, so its memory-corruption recovery is not applicable; all
five memory policies receive the same eligible anchors and corruption levels. Online policies still have no
archive/re-entry operation. Clone clean and perturbed policy state and use the
same subsequent observations/counter draws. Measure errors at0/1/2/4/8/16,
report raw discrepancies alongside normalized recovery, and report denominator
floor counts and eligible recording/participant counts. Use the published
Rec(h)=(d0-dh)/max(d0,1e-8), and normalize trapezoidal AUC over offsets1..16
by the15-update span. For memory recovery, compare multisets of payload-origin
identities tracked by the evaluator, not timestamps that the corruption keeps
unchanged. For task recovery, use mean absolute differences between clean and
perturbed sigmoid action probabilities across the four horizons and24 verbs;
this task-distance choice is an explicitly reconstructed detail. Report raw
initial and final distances alongside every normalized measure.

## Evidence and budget

Before any model inference, bind actual completed GPU training and artifact
jobs, six model hashes, all source locks, actual train-only basis/bank hashes,
and the independent label-axis audit. Formal protocols are created only after
those actual proofs exist; this design file is not a submitted protocol.
Prepare and verify the benchmark software before allocation. Use one A100,
four science threads,60G, and one four-hour stage with13,800s soft budget. A full
recording with all seeds/methods must support the complete-work projection.
If it cannot fit, preserve real partial results and stop. Do not reduce data,
methods, seeds, or recovery cases, extend the budget, or quietly spawn another
identical stage. A different execution design requires an explicit predeclared
performance justification, source/protocol lock, and unchanged mathematical
comparisons before execution.

After the benchmark, compare actual scores and implementation deviations with
the published paper, then freeze a separate transfer experiment for our causal
I-JEPA task. The unresolved raw60fps annotation origin does not become solved
by this official indexed TSM mapping. Existing results and acceptance rules
remain unchanged; reproduction is not a guaranteed first-round pass.

Immutable utility-bank normalization is cached once using the released algebra.
Synthetic distribution equality and public SPACE stream-equivalence checks
precede deployment. This is an execution optimization, not a changed metric,
retrieval rule, bank, or policy configuration. Large score/label/trace arrays
remain on HPC; only aggregate result metadata and small figures are transferred.
