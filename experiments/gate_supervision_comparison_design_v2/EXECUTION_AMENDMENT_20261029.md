# Execution amendment — Gate matched scorers

Date: 2026-10-29, America/New_York. This amendment is recorded before any
actual teacher labels or scorer training for v2.

The user authorized starting the experiment and requested more budget to avoid
interruptions. The previously proposed 1,800 GPU-second cap becomes **3,600
new GPU-seconds**, counting every GPU allocation's startup/configuration,
profiling, failed attempts, teacher calculation, scorer training and validation.
Queueing without an allocation does not count. The cap covers the whole paired
run, all three seeds and both methods; it is not 3,600 seconds per model. The
actual scheduler allocation/restart ledger takes precedence, and unknown usage
cannot be treated as zero. Do not expand the cap again without user instruction.

This authorizes real training/validation of the two new v2 scorers using only
the existing train/validation split. The encoder, original predictors, trained
transforms, cached encodings, seeds, 1,000 student updates, metrics and original
Gate thresholds are unchanged. The original Gate result remains failed 42/51.
No fresh independent test is generated/opened by this authorization. Existing
9137 and 2026102317 test outputs cannot select parameters or checkpoints.

Before any real target generation or training: freeze code and this amendment;
verify the frozen assets; profile the fixed 32 train-only states; and verify the
estimated full paired run, including startup and validation, fits the cap.
One scheduler allocation with no automatic requeue, a fixed wall limit no greater
than the remaining budget, and durable per-seed checkpoints is preferred. The
scheduler remains authoritative; any requeue, interruption or missing ledger
requires inspection before resumption. Preserve previous outcomes and do not
repeat completed scorers or generated packs. The cap controls resources; it is
not a guarantee against external termination.
