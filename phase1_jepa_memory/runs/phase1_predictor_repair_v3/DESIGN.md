# Third repair: diagnosis before one fixed trial

Status:120 native read-only diagnostic slices reviewed;9compact three-SHA matches. One candidate frozen in58science sources. CPU7581 preflight has a single actual receipt; GPU has not started. See execution_ready.json and diagnostic_review.json.

The original Proposal Phase1 acceptance is unchanged: both tasks must pass all 51 hard checks and all 32 fixed-reference checks each, followed by an independent raw-scalar audit. Revealed6301 is exploratory development. Only complete support permits the original independent9137 confirmation, including six new references and six new candidates, fresh train-only transforms, and locking every model before opening tests. Failure on confirmation is final and is not retuned.

## Questions

Motion: quantify actual trajectory reflections, visible velocity readout error, original visual-feature secant error, and common-vs-pair-difference prediction error. Compare the frame correction with the explicit transport correction. Ground-truth states are evaluation metadata, never predictor inputs or a new training target.

Gate: the single seed43/test_id/h6 privileged-benefit hard check fails the state-accuracy confidence bound. Normalized latent-MSE and decoded-position gains pass. The previous decoder is linear and future action primarily changes positive attention weights. A post-pooling action-conditioned nonlinear decoder can represent cue-action interaction more directly, but this is a hypothesis, not a unique causal explanation.

## Provisional repair

Encode centered observed features through a small nonlinear coordinate map. Estimate elapsed transport using observed coordinate differences, then decode nonlinearly into the frozen128-dimensional output basis. Use action-conditioned featurewise modulation after history pooling. The coordinate origin and decoder origin are subtracted exactly. Empty history retains the exact frozen recent-only reference. All retained observations are read, with no selector or TopK. No hidden simulator state, labels, sibling ID, or future image is provided.

Borrowed principles: E2C (https://arxiv.org/abs/1506.07365), nonlinear dynamical embeddings (https://arxiv.org/abs/1712.09707), and FiLM conditioning (https://arxiv.org/abs/1709.07871). This is an adaptation of these principles, not reproduction of their complete architectures, proof of learned physical state, or a novelty claim.

Both tasks retain the same fixed candidate architecture and original loss, learning rate, optimizer, sampling, seeds17/29/43, all120epochs and earliest validation selection. No test-specific or per-seed tuning. One composite trial does not identify which component causes any improvement.

## Resources

Native settled GPU use12604s of28800s. Remaining16196s includes every new GPU allocation second, initialization, failed partials, proof, encoding and confirmation. Before continuing beyond the first full Gate17 and Motion17 epochs, the existing native full-sequence resource guard reserves all remaining development plus12-model confirmation plus198656 encoded frames,15% margin,900s fixed overhead and two encoder loads. Rejecting that estimate stops with partial evidence; it cannot shorten the experiment or relax a gate.

## Local checks

Ten synthetic-only software fixtures pass: exact empty-history/reference behavior even after a toy update; frozen reference; first-step internal gradients; action-prefix causality; padding/permutation; forbidden future/target boundary; finite long gap; common-offset invariance; exact zero visual evidence; every retained-value gradient; cached/raw projection equivalence. These are software checks, not scientific validation.

## Actual diagnostic interpretation

In the diagnosed64-pair-per-split subset, geometric reflections were0%. Motion full-history long-gap feature secant MSE was4.94/5.86 for h3/h6, versus0.998/1.380 onID. Full-history long-gap pair-common MSE increased from reference0.0776/0.0789 to0.0996/0.1056. At h6, removing the transport component changed total MSE0.1871 to0.1727. Predicted-feature position error remained higher than true-target-feature readout error. This supports testing nonlinear coordinates/decoding; it does not establish a unique cause or guarantee success. These are diagnostic subset values, not replacement acceptance statistics.
