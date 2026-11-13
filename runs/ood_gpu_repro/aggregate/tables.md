# Run ood_gpu_repro: aggregate tables

Replayed artifacts and retrained models are separate tables. `diff` is this run minus the reference value.

## Synergy, retrain_gpu, multihead reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| distilled_sensitivity_student_set | ablation | 3/3 | 11.40% | +0.0046 | 8.81% | +0.0046 | differs by more than 1 point |
| deletion_cost_student_set | ours | 3/3 | 17.27% | +0.0115 | 4.59% | +0.0036 | differs by more than 1 point |
| future_attention_student_independent | ablation | 3/3 | 55.71% | +0.0427 | 54.83% | +0.0420 | differs by more than 1 point |
| distilled_sensitivity_student_independent | ablation | 3/3 | 60.01% | +0.0435 | 59.14% | +0.0435 | within 1 point |
| appearance_only_student | ablation | 3/3 | 62.27% | +0.0442 | 62.27% | +0.0442 | within 1 point |
| deletion_cost_student_independent | ablation | 3/3 | 61.81% | +0.0449 | 62.80% | +0.0449 | differs by more than 1 point |
| shuffled_label_student_independent | ablation | 3/3 | 63.22% | +0.0476 | 63.22% | +0.0476 | within 1 point |
| shuffled_label_student_set | ablation | 3/3 | 64.37% | +0.0489 | 64.37% | +0.0489 | within 1 point |
| future_attention_student_set | ablation | 3/3 | 66.42% | +0.0503 | 67.38% | +0.0508 | within 1 point |

## Synergy, retrain_gpu, item_attention reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| deletion_cost_student_set | ours | 3/3 | 0.12% | +0.0020 | 0.65% | +0.0020 | differs by more than 1 point |
| distilled_sensitivity_student_set | ablation | 3/3 | 19.75% | +0.0103 | 17.45% | +0.0096 | differs by more than 1 point |
| appearance_only_student | ablation | 3/3 | 57.29% | +0.0427 | 57.99% | +0.0433 | within 1 point |
| future_attention_student_set | ablation | 3/3 | 56.15% | +0.0435 | 56.29% | +0.0438 | within 1 point |
| future_attention_student_independent | ablation | 3/3 | 55.93% | +0.0441 | 56.52% | +0.0443 | differs by more than 1 point |
| distilled_sensitivity_student_independent | ablation | 3/3 | 60.85% | +0.0457 | 60.43% | +0.0468 | within 1 point |
| deletion_cost_student_independent | ablation | 3/3 | 61.47% | +0.0466 | 61.68% | +0.0470 | differs by more than 1 point |
| shuffled_label_student_independent | ablation | 3/3 | 63.22% | +0.0496 | 63.22% | +0.0496 | within 1 point |
| shuffled_label_student_set | ablation | 3/3 | 64.37% | +0.0501 | 64.37% | +0.0501 | within 1 point |

## Not reproducible in this run

- gate / fm_ijepa_published (replay): not_run - optional variant on the project's I-JEPA features; never run
- gate / fm_ijepa_online (replay): not_run - optional variant on the project's I-JEPA features; never run
- motion_v2 / reference_arms (retrain): extracted_only - reader training not attempted; see environments.motion_v2.from_scratch
- motion_v2 / online_validation (retrain): assets_missing - the 54 students are refitted from the tracked labels (motion_v2.retrain.students), but scoring them online needs the frozen readers and features
- motion_v2 / literature_baselines (replay): assets_missing - PaFu-KV, Compress to Remember and FrameMorrow were never run on Motion v2
- motion_v2 / literature_baselines (retrain): not_run - PaFu-KV, Compress to Remember and FrameMorrow were never run on Motion v2
- pusht / v5_marker_corpus (replay): not_run
- pusht / v4_natural_corpus (replay): not_run
- pusht / v4_diagnostics (replay): assets_missing - v4 I-JEPA features are on a teammate's project storage
- pusht / v4_diagnostics (retrain): not_run
- pusht / predictor (replay): not_run - no PushT predictor has been trained
- pusht / predictor (retrain): not_run - no PushT predictor has been trained
- pusht / retention (replay): not_run - no retention result exists for PushT
- pusht / retention (retrain): not_run - no retention result exists for PushT
