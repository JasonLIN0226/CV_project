# Locked before scoring: training-free retention rules on qualified readers

Written 2026-10-25 (America/New_York) before computing any selection result below.

## Data (already-viewed development data; not CONF)
- Readers: `multihead` (primary), `item_attention` (secondary), `original` (reference only).
- Splits: `existing_development` (seed 20261103) and `fresh_diagnostic` (seed 20261120); both were used to qualify readers, so this is development analysis, not confirmation. CONF (20261104) stays closed.
- Inputs: reader-aggregation release `reader_qualification_coefficients.npy` (rank-128 coefficients, h3/h6), `reader_qualification_rows.json`, `pooling_weights.json`, reader checkpoints (output basis buffer).

## Rules (one terminal deletion among three candidates, K=2)
- Forecast difference between two retained sets of the same parent: `(c_i - c_j) @ output_basis.T`, squared, averaged over horizons and feature dims.
- **oracle**: argmin of saved native-time MSE over the three deletions (actual episode plan).
- **medoid**: argmin_i sum_j D(native_i, native_j).
- **full-set sensitivity**: argmin_i D(full, native_i).
- **attention**: delete the candidate with the lowest saved pooling weight (mean over pooling paths, as saved).
- **FIFO**: delete slot 0. **uniform**: expected value (1/3 per deletion).
- Ties: first index of exact float64 argmin.

## Known limitation (stated before results)
Selection uses forecasts under the episode's actual future plan, not the causal 16-plan TRAIN reference bank (that needs reader inference on features that are not in the release). In Gate2 the two agreed on 189/192 choices; here it is an approximation to verify later.

## Endpoints
Necessary-evidence loss (choice not in `semantic_correct_delete_set`) per task and condition; regret = MSE(choice) - min MSE; family bootstrap (24 families, 2000 resamples, seed 20261107).

## Predictions
| rule | R redundancy | D decoy | S synergy |
|---|---|---|---|
| medoid | near 0 loss | near 0 loss (if missing-info forecasts hedge) | **high loss** (two incomplete forecasts agree) |
| full-set sensitivity | near 0 | near 0 | near 0 (K3 reader now qualified) |
| attention | unknown | unknown | unknown |
| oracle evidence loss on S | below the original reader's 159/1152 | | |
