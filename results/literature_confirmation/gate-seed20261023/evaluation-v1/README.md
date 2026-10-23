# Frozen Gate confirmation, seed 20261023

The locked protocol predates generation/readout of this separate 96-pair set.
All six existing probes, WR/PaFu, FIFO/diversity/attention, reader and normalization
are frozen. No training, support selection, hyperparameter selection or arm
selection uses these observations. See `primary_decision.json` for the single
conjunctive decision against shuffled cost, reader-attention and diversity.

`online_results.json` and `common_parent_results.json` retain every result.
`statistics.json` uses independent sibling pairs, not repeated parent rows.
`task_contrast_comparisons.json` is descriptive, evaluator-only, and cannot replace
the primary comparison. A positive feature-MSE result alone would not establish
physical planning utility. Same K does not mean equal total memory/compute.

`audit.json` reconstructs saved-vector arithmetic, probe choices, first information
loss/survival, irreversible traces and pair CIs. Large prediction vectors and full
traces remain under the manifest-bound ignored build directory. Data features/RGB
are not included here; `data_manifest.json` identifies all source bytes.

`temporal_audit.json` and the original `submission_receipt.json` additionally verify
that protocol lock preceded submission, which preceded encoding. This post-export
audit does not independently rerun the reader; its arithmetic/trace scope is
explicit in `audit.json`. Submission-time verification flags remain historical.
