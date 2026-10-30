# Frozen Gate input-support diagnosis

Completed exploratory diagnosis of all 192 previously observed fresh development episodes. Read the [Chinese report](../../../docs/GATE_INPUT_SUPPORT_DIAGNOSIS.md).

The fixed old-TRAIN-prefix second-moment mask removes 81 of 128 projected input coordinates. With original history sets fixed, the predictor's attention MSE changes only from 0.110064 to 0.109824, and all 27 originally harmful local cue comparisons remain harmful. The same support mask applied to frozen cue-probe coefficient rows changes cue−current label accuracy from 50% to 100% / 98.4375% / 100% across ID / long gap / distractor shift. The previous unstable probe does not establish missing cue information.

- `protocol.json`: fixed rule, complete-set evaluation, source and input hashes, runtime.
- `summary.json`: full scene-pair confidence intervals and input/probe audits.
- `training_support.*`: old-TRAIN-only mask, rank, moments and fit episode IDs.
- `episode_rows.json`, `mask_minus_original.json`, `cue_gaps.json`: every predictor comparison; original sets never reoptimized.
- `cue_probe.json`, `*_readout.npz`: every probe score and reconstructed original frozen coefficients.
- `independent_audit.json`: saved-vector and metric verification; no additional model calls.
- `source/`: exact executed source snapshots and derivation-only exporter.
- `failed-first-runtime/`: preserved first attempt rejected by exact replay due to selecting system Python.
- `manifest.json`: portable artifact hashes; excluded large prediction vectors remain under ignored build.

These are diagnostic interventions, not repaired benchmark performance, new retention methods, or untouched confirmation. The cue probes use evaluator-provided cue positions. Global hindsight choices remain privileged, and the world-model input and evaluation-probe support interventions must not be conflated.
