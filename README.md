# Bounded Visual Memory: Gate Retention Study

This project studies how a model with limited visual memory can preserve observations that matter for predicting the future. The current stage uses a controlled, synthetic **Gate** task to examine two questions:

1. Does retaining a key visual cue help the frozen model predict what happens next?
2. How do online memory-retention policies compare on the same task and memory budget?

The current study is an exploratory comparison on a simple task. Its results do not depend on passing the earlier Phase 1 gate, and they do not change the historical evaluation.

## The Gate task and data

An episode contains a direction cue represented by pale-gray arrows, a sequence of frames with randomly distributed vertical-line distractions, and a future route or state to predict. The cue can occur at different positions in the history. Input frames contain the distractions; future prediction targets are rendered without them. Paired scenes share actions and nuisance patterns while varying the cue and corresponding future.

The latest comparison reuses the existing encoded train/validation data: **1,024 training episodes and 192 validation episodes**. Each policy has two history slots (B = 2) and receives the same four recent frames, with the same frozen encoder and predictor. On arrival, a policy can retain only two of its two stored history frames and the new frame; a discarded frame cannot return.

![An example Gate sequence](results/gate_retention_comparison_20261029/data_preview.png)

[Data generation and configuration](docs/GATE_RETENTION_COMPARISON_LATEST.md#data-design) · [Dataset preflight](results/gate_retention_comparison_20261029/data_preflight.json)

## Current validation comparison

The table aggregates the 192 validation episodes across three predictor seeds and two horizons. These repeated measurements are not independent examples. Cue hit means at least one retained history frame is in the cue interval; it is separate from route/state prediction accuracy.

| Retention method | Route/state accuracy | Cue hit | Normalized latent MSE ↓ |
|---|---:|---:|---:|
| Manual keyframe reference* | 95.57% | 100.00% | 0.09783 |
| Deletion-cost scorer | 94.53% | 100.00% | 0.09934 |
| Visual diversity | 71.79% | 47.40% | 0.11385 |
| Prediction surprise | 66.15% | 34.55% | 0.11765 |
| Reservoir | 57.38% | 16.84% | 0.12362 |
| Temporal buckets | 56.60% | 15.62% | 0.12256 |
| FIFO | 56.51% | 13.54% | 0.12423 |
| Feature-attention-proxy scorer | 56.08% | 13.72% | 0.12332 |
| Recent frames only | 50.00% | 0.00% | 0.12497 |

\* Manual keyframe selection is a privileged diagnostic reference: it knows the cue position. Its advantage over the recent-only control supports the usefulness of cue-containing history **in this setting**, but is not by itself a general causal or independent-test guarantee. The two learned scorers have the same architecture and student training budget, but their teacher signals and preparation costs differ. The deletion-cost teacher uses the frozen predictor and future targets during training; it is therefore predictor-aligned. No future information or cue labels are used by the learned students during online selection.

The retired task-trained attention method is excluded from the current table. Historical code and records remain available for provenance.

[Full result and fairness details](docs/GATE_RETENTION_COMPARISON_LATEST.md) · [Chinese result summary](results/gate_retention_comparison_20261029/RESULTS_ZH.md) · [Machine-readable results](results/gate_retention_comparison_20261029/summary.json)

## Code map: reproduce the Gate study

### Team model and data downloads

- [Private Gate model bundle](https://huggingface.co/cv-project-team5guys/gate-retention-models-v1) — three frozen predictors, both new scorers for seeds 17/29/43, transforms, probes, checksums and portable loading examples.
- [Private Gate train/validation dataset](https://huggingface.co/datasets/cv-project-team5guys/gate-clean-future-trainval-v1) — the 1,216 episodes, matching feature cache, original validation output and per-file checksums.

An organization member must grant your Hugging Face account access before either private link works. The bundles were downloaded again and their checksums verified; a real validation-episode loading and inference example ran successfully. These are validation artifacts, not a fresh independent test. Official I-JEPA pretrained encoder weights are not redistributed.

The links below follow the experiment flow. The original course-HPC batch launchers retain their historical paths and are **not portable launchers**; use the download and loading instructions in the model and dataset repositories. `/scratch/xl4358/...` paths in historical records are provenance, not required teammate paths.

### 1. Generate and check the data

- [Clean-future dataset generator](experiments/gate_clean_future/materialize.py): builds paired Gate episodes with noisy input frames and clean future targets.
- [Structured scene and distraction design](experiments/retention_structured_noise_audit/structured_design.py) · [cue-position/noise utilities](experiments/retention_position_noise_audit/design.py): scene and augmentation rules used by the generator.
- [Dataset configuration template](results/gate_retention_comparison_20261029/dataset_config_template.json) · [train/validation protocol](experiments/gate_clean_future/protocol.json) · [preflight checks](experiments/retention_structured_noise_audit/clean_future_preflight.py) · [recorded preflight results](results/gate_retention_comparison_20261029/data_preflight.json). The configuration is a readable path-adjusted template, not a byte-identical copy of the HPC resolved configuration.

### 2. Encode frames and train the future predictor

- [Feature encoding and train-only transforms](experiments/gate_clean_future/fit_features.py) · [encoder cache implementation](experiments/predictor/control/src/phase1/cache.py).
- [Predictor architecture and projection](experiments/predictor/control/src/phase1/model.py) · [training and checkpointing](experiments/predictor/control/src/phase1/training.py) · [three-seed training entry point](experiments/retention_structured_noise_audit/train_reference.py).
- [Original train/validation batch launcher](experiments/gate_clean_future/run_trainval.sbatch) · [predictor validation](experiments/gate_clean_future/evaluate_validation.py). The recorded validation used this [per-sibling bookkeeping repair](results/retention_structured_noise_audit/clean_future_evaluation_repair_20261022/evaluate_each_sibling.py); preserve its behavior when reproducing that validation. It is an HPC-specific execution snapshot.
- [Frozen clean-future source checksums](experiments/retention_structured_noise_audit/clean_future_source_lock.json) · [clean-future protocol](experiments/gate_clean_future/protocol.json).

### 3. Train the two new retention scorers

- [Scorer architecture and matched initialization](experiments/gate_supervision_comparison_design_v2/core.py).
- [Teacher definitions](experiments/gate_supervision_comparison_design_v2/teachers.py) · [teacher-label preparation](experiments/gate_supervision_comparison_design_v2/teacher_stage.py) · [matched checkpointed training](experiments/gate_supervision_comparison_design_v2/checkpoint_fit.py).
- [Train/validation runner](experiments/gate_supervision_comparison_design_v2/run_trainval.py) · [original training batch script](experiments/gate_supervision_comparison_design_v2/train.sbatch) · [experiment protocol](experiments/gate_supervision_comparison_design_v2/PROTOCOL.md) · [execution amendment](experiments/gate_supervision_comparison_design_v2/EXECUTION_AMENDMENT_20261029.md). The protocol contains its original pre-run wording; the [package README](experiments/gate_supervision_comparison_design_v2/README.md) explains the completed run.

### 4. Run and inspect the policy comparison

- [Online B=2 update rules](experiments/gate_retention_comparison_v1/online.py) · [baseline policy implementations](experiments/retention_online_matrix/policies.py).
- [Policy validation and trace checks](experiments/gate_supervision_comparison_design_v2/validation.py) · [original validation batch script](experiments/gate_supervision_comparison_design_v2/validate.sbatch) · [independent validation-audit snapshot](results/gate_retention_comparison_20261029/audit_validation_hpc_snapshot.py).
- [Results and fairness details](docs/GATE_RETENTION_COMPARISON_LATEST.md) · [Chinese result summary](results/gate_retention_comparison_20261029/RESULTS_ZH.md) · [machine-readable summary](results/gate_retention_comparison_20261029/summary.json).
- [Scorer software tests](experiments/gate_supervision_comparison_design_v2/test_v2.py) · [data-protocol tests](experiments/gate_clean_future/test_protocol.py). These are software checks, not new scientific results.

The historical validation code also computed the retired task-trained attention comparator. It is preserved for provenance and omitted from the latest results table. The reported comparison is on the existing validation set, not a new independent test.

## Next stage

The next research stage can extend this comparison to **Motion and more complex simulated scenes**, where relevant information may unfold over time and require multiple observations. This is a future direction, not a completed result of the current Gate study.
