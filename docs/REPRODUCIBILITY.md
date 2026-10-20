# Reproducibility

## Published snapshot

Use Python 3.10–3.12. The repository contains frozen source, compact development evidence, final confirmation summaries and the report. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python tools/verify_independent_confirmation.py
python tools/verify_results.py
python tools/verify_phase1_latest.py
python -m pytest -q
```

These checks validate file integrity, counts and software. They do not rerun model inference. The fresh HPC independent audit replayed 310 decision checks, 248,832 prediction rows, 248,832 deletion rows and 18,432 paired deletion summaries. Its reported maximum original-statistic difference is zero. The local final JSON is a compact transcription of audited outcomes; full fresh pair-level arrays remain external, so this checkout alone cannot independently recompute fresh confidence intervals from raw rows.

## Current report

```bash
python -m pip install -r requirements-report.txt
python tools/build_independent_confirmation_report.py
```

The builder reads `results/phase1_independent_confirmation/local_verified_results_20261020.json` and the retained explanatory figures. It writes the current Word report and plots. PDF export requires LibreOffice or Word. After editing, render and inspect every page; refresh the publication manifest after approved changes.

## Scientific source and external assets

The canonical final execution is `results/phase1_independent_confirmation/execution_v10_checkpoint_resume/source/`, with registered source lock `4133e4335f87f0b719376f45c71ac9adb52e05b3e2c45550bcb22863472727d9`. Execution-only repairs and interruption evidence are retained separately. Earlier snapshots and negative results are dependencies and provenance, not runnable live submissions. Do not blindly resubmit historical job scripts.

The complete study used new seed 9137 data; 512 training, 96 validation and 256 pairs per test condition per task; 12 new reference/candidate models at 120 epochs and 7,680 steps; two new maps and train-only transforms. Test generation followed locking all model assets. Training/validation encoding comprised 62,418 frames. Final numerical evidence records its original HPC locations under the v9 `fresh` directory, while resumed execution source is v10. These distinct paths reflect reuse, not different test results.

To repeat model inference or raw auditing, obtain the matching I-JEPA checkpoint/processor, feature caches, all 12 locked model assets, maps, normalization/PCA/readouts, original paired manifests and raw rows from the recorded HPC asset store. Weights and large arrays are deliberately excluded from GitHub. Their availability is not guaranteed by a local JSON path. Historical SSH aliases must not be reused.

## Attribution and scope

I-JEPA: `facebook/ijepa_vith14_1k`, revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e`. The predictor adapts last-observation centering from NLinear and learned-query pooling from Set Transformer. The original Set Transformer license is retained in `experiments/predictor/control/SET_TRANSFORMER_LICENSE.txt`. External checkpoint licenses apply. This is a synthetic I-JEPA study, not V-JEPA or real-video validation; position readouts use training supervision and the guard uses observed RGB.
