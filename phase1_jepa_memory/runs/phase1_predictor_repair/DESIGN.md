> Current budget amendment: the human added three hours, so the cumulative cap is now 28800 GPU seconds. All old consumption remains charged. See budget_authorization_8h.json and the latest workflow for native settlement and exact partial-resume provenance. The historical five-hour stop below remains part of the record. The scientific design and every passing rule are unchanged.

# Proposal Phase1: one bounded predictor repair

The goal is to establish that the predictor reliably uses older visual evidence on both original controlled tasks. A successful FAR experiment, a software audit, or completed training does not establish this premise. All original passing rules remain in force.

## Actual diagnosis

Read-only course CPU job 7518 completed with exit 0:0 in 161 seconds. The three diagnostic sources, all 48 prior scientific sources, and all 12 current best/latest checkpoint hashes matched before accepting the output. All 144 diagnostic slices were finite; the four compact result files have matching native-before/local/native-after SHA256 values. No GPU was used.

Each diagnostic slice uses the predetermined first 32 complete pairs; these diagnostic numbers do not replace the original full evaluation. On the Gate validation slice, the best seed 17 model preserves only 3.85% of the target sibling prediction amplitude, compared with 97.71% and 97.87% for seeds 29 and 43. Seed 17's later correction grows while validation error worsens. This supports addressing reader learning and optimization stability.

On Motion, the corresponding validation amplitude ratios are 0.807–0.848 and fall to 0.341–0.462 on long gaps. The old time embeddings are about 1.96–2.31 times the visual embedding RMS on those long-gap slices. Increasing age by 16 changes the prediction on unchanged pixels. That counterfactual is a sensitivity probe, not a physically valid new performance test; it does not prove that the time embedding is the sole cause.

## Fixed repair choice

Use the single `RelationalMemoryPredictor` already prepared before this native diagnosis. Learned seed attention pooling follows the PMA principle of [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html); observed-frame relation features are an adaptation motivated by [Relation Networks](https://arxiv.org/abs/1706.01427). The official Set Transformer MIT license is included. This is an adapted composite predictor, not a reproduction of either paper.

The predictor reads every supplied historical entry, pools frame content and adjacent chronological relations, and includes explicit linear transport from observed feature differences and elapsed time. Learned age inputs are bounded relative quantities. A single direct history correction replaces subtraction of two learned side outputs. Empty history remains exactly equal to the frozen recent reference. A small nonzero readout allows reader gradients on the first update. No hidden simulator velocity, target, sibling identity, future RGB, retrieval, or retention selector is supplied.

Fix AdamW learning rate 0.0003, weight decay 0.0001, gradient clipping 1, five-epoch linear warmup and cosine decay to 0.1 of the base rate by epoch 120. Keep original losses, Gate common-anchor weight 1, Motion weight 0, balanced conditions, batch size 64, and earliest minimum of the equally weighted four-condition/two-horizon validation MSE. These optimization choices are one fixed development repair, not a candidate sweep or a claim isolating the effect of each component.

## Unchanged validation and independent confirmation

Development remains Gate and occluded Motion, seeds 17/29/43, 120 epochs and 7680 steps per model, data seed 6301, 512 train/96 validation pairs per task, 128 pairs in each of the three tests, recent 4, B2, and horizons 3/6. Preserve the original task rules and every one of the 24 fixed-recent and eight critical absolute-MSE checks per task. Keep all failed seeds and difficult slices. The previous negative result remains intact.

Both development tasks and the independent raw-scalar audit must pass before fresh confirmation. Fresh seed 9137 remains closed. Confirmation requires new data, new train-only normalization/basis/readout, 512 train/96 validation/256 pairs per test and task, and 12 newly trained reference/candidate models across both tasks and all three seeds. Lock every model before opening confirmation tests. Do not substitute development weights or retune after confirmation failure.

## Budget and execution boundary

Native allocation accounting reconfirms 6247 of the authorized 18000 GPU seconds used, leaving 11753 seconds. Every later allocation startup, preload, encoding, warmup, proof, profile, partial failure, development model, and confirmation model shares this balance. The resource preparation code rejects missing/nonfinite timing measurements and reserves the entire confirmation stage before continuing development.

The current frozen renderer gives a conservative full-frame confirmation encoding bound of 198656 RGB frames without generating any fresh samples. Exact RGB deduplication could reduce work, but its smaller bound may only be used after actual execution equivalence is demonstrated. No frame, episode, seed, epoch, or passing threshold may be removed to fit the cap.

The architecture/optimizer choice is fixed in `repair_decision.json`. The new development training/evaluation runner, signatures, independent audit and executable protocol are now frozen as55source files, source lock9ab65cd2e422085672d753a9b8bb1ff462861271cfbb42899ce54c116e838ab4. All28local software checks passed. A vectorized causal boundary preserves original per-trajectory predicates and exact returned tensors, with nativeCPU/CUDAproof mandatory before training. New entrypoints are run_repair.py, native_preflight.py and repair_train/evaluate.sbatch; copied obsolete launch stubs were removed. Native input proofs, throughput and the full-sequence cost guard are still pending; no native GPU result is claimed. Fresh confirmation execution must be separately frozen after both development tasks pass. No new GPU job has been submitted. A budget prediction is not Phase1_GO. Full independent confirmation is required for that conclusion.
