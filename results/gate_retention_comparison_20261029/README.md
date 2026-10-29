# Validation artifacts

`summary.json` is the current comparison view, excluding the retired task-trained-score-plus-novelty method; the original HPC raw result is not modified. `RESULTS_ZH.md` explains the same numbers in Chinese. `dataset_config_template.json` is the recorded clean-future configuration with a placeholder output path; it is not the exact HPC file. `data_preflight.json` and `data_preview.png` document the train/validation data design.

This is a validation result for 192 episodes, not a new independent test. The raw result and feature cache are in the [private team dataset](https://huggingface.co/datasets/cv-project-team5guys/gate-clean-future-trainval-v1); student and predictor weights are in the [private team model bundle](https://huggingface.co/cv-project-team5guys/gate-retention-models-v1). Organization access is required. The original experiment's SHA-256 is recorded in `summary.json`.
