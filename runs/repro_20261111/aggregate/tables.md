# Run repro_20261111: aggregate tables

Replayed artifacts and retrained models are separate tables. `diff` is this run minus the reference value.

## Gate, replay (route accuracy, 192 validation episodes)

| Policy | Group | Memory | Seeds | Per seed | Pooled | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|---|---|
| oracle_keyframe | reference | privileged | 3/3 | 95.05 / 95.57 / 96.09 | 95.57% | 95.57% | +0.00 | identical |
| deletion_cost_student | ours | bounded | 3/3 | 93.75 / 94.79 / 95.05 | 94.53% | 94.53% | +0.00 | identical |
| full_history | reference | full_history | 3/3 | 95.83 / 93.49 / 92.19 | 93.84% | 93.84% | +0.00 | identical |
| pafu_kv | literature | bounded | 3/3 | 90.10 / 90.62 / 90.10 | 90.28% | 90.28% | +0.00 | within 1 point |
| pafu_teacher_top2 | literature_teacher | privileged | 3/3 | 89.06 / 92.19 / 89.06 | 90.10% | 90.10% | +0.00 | identical |
| attention | project_baseline | bounded | 3/3 | 88.80 / 90.89 / 88.02 | 89.24% | 89.24% | +0.00 | identical |
| pafu_kv_repo | literature | bounded | 3/3 | 88.54 / 90.10 / 88.02 | 88.89% | 88.89% | +0.00 | within 1 point |
| visual_diversity | project_baseline | bounded | 3/3 | 73.18 / 70.31 / 71.88 | 71.79% | 71.79% | +0.00 | identical |
| prediction_surprise | project_baseline | bounded | 3/3 | 66.93 / 65.62 / 65.89 | 66.15% | 66.15% | +0.00 | identical |
| reservoir | project_baseline | bounded | 3/3 | 59.90 / 55.21 / 57.03 | 57.38% | 57.38% | +0.00 | identical |
| temporal_buckets | project_baseline | bounded | 3/3 | 56.77 / 56.51 / 56.51 | 56.60% | 56.60% | +0.00 | identical |
| fifo | project_baseline | bounded | 3/3 | 56.77 / 56.51 / 56.25 | 56.51% | 56.51% | +0.00 | identical |
| attention_proxy_student | ablation | bounded | 3/3 | 55.73 / 57.81 / 54.69 | 56.08% | 56.08% | +0.00 | identical |
| recent_only | reference | none | 3/3 | 50.00 / 50.00 / 50.00 | 50.00% | 50.00% | +0.00 | identical |
| fm_dinov2_online | literature | bounded | 0/3 | | | | | incomplete |
| fm_dinov2_published | literature | full_history | 0/3 | | | | | incomplete |
| fm_dinov2_teacher_top2 | literature_teacher | privileged | 0/3 | | | | | incomplete |
| pacc_full | literature | full_history | 0/3 | | | | | incomplete |
| pacc_recurrent | literature | bounded | 0/3 | | | | | incomplete |

## Gate, retrain (route accuracy, 192 validation episodes)

| Policy | Group | Memory | Seeds | Per seed | Pooled | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|---|---|

## Gate, retrain_scorer (route accuracy, 192 validation episodes)

| Policy | Group | Memory | Seeds | Per seed | Pooled | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|---|---|
| deletion_cost_student | ours | bounded | 3/3 | 93.75 / 94.79 / 95.05 | 94.53% | 94.53% | +0.00 | within 1 point |
| attention_proxy_student | ablation | bounded | 3/3 | 55.73 / 57.81 / 54.69 | 56.08% | 56.08% | +0.00 | within 1 point |

## Synergy, replay, multihead reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| pacc | literature | 3/3 | n/a | -0.0046 | n/a | -0.0046 | within 1 point |
| oracle | reference | 1/1 | 22.92% | +0.0000 | 22.92% | +0.0000 | identical |
| three_cards | reference | 1/1 | n/a | +0.0036 | n/a | +0.0036 | identical |
| deletion_cost_student_set | ours | 3/3 | 4.59% | +0.0036 | 4.59% | +0.0036 | identical |
| distilled_sensitivity_student_set | ablation | 3/3 | 8.81% | +0.0046 | 8.81% | +0.0046 | identical |
| full_set_sensitivity | project_baseline | 1/1 | 35.89% | +0.0077 | 35.89% | +0.0077 | identical |
| future_attention_student_independent | ablation | 3/3 | 54.83% | +0.0420 | 54.83% | +0.0420 | identical |
| distilled_sensitivity_student_independent | ablation | 3/3 | 59.14% | +0.0435 | 59.14% | +0.0435 | identical |
| appearance_only_student | ablation | 3/3 | 62.27% | +0.0442 | 62.27% | +0.0442 | within 1 point |
| deletion_cost_student_independent | ablation | 3/3 | 62.80% | +0.0449 | 62.80% | +0.0449 | identical |
| pafu_kv | literature | 3/3 | 60.30% | +0.0454 | 60.30% | +0.0454 | within 1 point |
| fm_teacher_top2 | literature_teacher | 1/1 | 62.28% | +0.0475 | 62.28% | +0.0475 | identical |
| shuffled_label_student_independent | ablation | 3/3 | 63.22% | +0.0476 | 63.22% | +0.0476 | identical |
| fifo | reference | 1/1 | 66.67% | +0.0476 | 66.67% | +0.0476 | identical |
| fm_published | literature | 3/3 | 62.62% | +0.0477 | 62.62% | +0.0477 | within 1 point |
| shuffled_label_student_set | ablation | 3/3 | 64.37% | +0.0489 | 64.37% | +0.0489 | identical |
| future_attention_student_set | ablation | 3/3 | 67.38% | +0.0508 | 67.38% | +0.0508 | identical |
| pafu_teacher_top2 | literature_teacher | 1/1 | 67.62% | +0.0511 | 67.62% | +0.0511 | identical |
| attention | project_baseline | 1/1 | 70.27% | +0.0545 | 70.27% | +0.0545 | identical |
| medoid | project_baseline | 1/1 | 97.53% | +0.0649 |  |  | no reference |

## Synergy, replay, item_attention reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| oracle | reference | 1/1 | 10.20% | +0.0000 | 10.20% | +0.0000 | identical |
| deletion_cost_student_set | ours | 3/3 | 0.65% | +0.0020 | 0.65% | +0.0020 | identical |
| distilled_sensitivity_student_set | ablation | 3/3 | 17.45% | +0.0096 | 17.45% | +0.0096 | identical |
| three_cards | reference | 1/1 | n/a | +0.0109 | n/a | +0.0109 | differs |
| full_set_sensitivity | project_baseline | 1/1 | 37.80% | +0.0179 | 37.80% | +0.0179 | identical |
| appearance_only_student | ablation | 3/3 | 57.99% | +0.0433 | 57.99% | +0.0433 | within 1 point |
| future_attention_student_set | ablation | 3/3 | 56.29% | +0.0438 | 56.29% | +0.0438 | identical |
| pafu_teacher_top2 | literature_teacher | 1/1 | 55.21% | +0.0439 | 55.21% | +0.0439 | identical |
| attention | project_baseline | 1/1 | 55.12% | +0.0439 | 55.12% | +0.0439 | identical |
| future_attention_student_independent | ablation | 3/3 | 56.52% | +0.0443 | 56.52% | +0.0443 | identical |
| pafu_kv | literature | 3/3 | 56.32% | +0.0446 | 56.32% | +0.0446 | within 1 point |
| distilled_sensitivity_student_independent | ablation | 3/3 | 60.43% | +0.0468 | 60.43% | +0.0468 | identical |
| deletion_cost_student_independent | ablation | 3/3 | 61.68% | +0.0470 | 61.68% | +0.0470 | identical |
| shuffled_label_student_independent | ablation | 3/3 | 63.22% | +0.0496 | 63.22% | +0.0496 | identical |
| shuffled_label_student_set | ablation | 3/3 | 64.37% | +0.0501 | 64.37% | +0.0501 | identical |
| fifo | reference | 1/1 | 66.67% | +0.0516 | 66.67% | +0.0516 | identical |
| medoid | project_baseline | 1/1 | 96.05% | +0.0613 |  |  | no reference |
| fm_published | literature | 0/3 | | | | | incomplete |
| fm_teacher_top2 | literature_teacher | 0/1 | | | | | incomplete |
| pacc | literature | 0/3 | | | | | incomplete |

## Synergy, retrain, multihead reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| deletion_cost_student_set | ours | 3/3 | 0.48% | +0.0038 | 4.59% | +0.0036 | differs by more than 1 point |
| distilled_sensitivity_student_set | ablation | 3/3 | 8.83% | +0.0042 | 8.81% | +0.0046 | differs by more than 1 point |
| appearance_only_student | ablation | 3/3 | 59.03% | +0.0415 | 62.27% | +0.0442 | differs by more than 1 point |
| future_attention_student_independent | ablation | 3/3 | 55.74% | +0.0428 | 54.83% | +0.0420 | differs by more than 1 point |
| distilled_sensitivity_student_independent | ablation | 3/3 | 58.65% | +0.0428 | 59.14% | +0.0435 | differs by more than 1 point |
| deletion_cost_student_independent | ablation | 3/3 | 63.18% | +0.0456 | 62.80% | +0.0449 | within 1 point |
| shuffled_label_student_independent | ablation | 3/3 | 63.22% | +0.0476 | 63.22% | +0.0476 | within 1 point |
| shuffled_label_student_set | ablation | 3/3 | 64.38% | +0.0489 | 64.37% | +0.0489 | within 1 point |
| future_attention_student_set | ablation | 3/3 | 67.85% | +0.0513 | 67.38% | +0.0508 | differs by more than 1 point |

## Synergy, retrain, item_attention reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| deletion_cost_student_set | ours | 3/3 | 0.09% | +0.0021 | 0.65% | +0.0020 | differs by more than 1 point |
| distilled_sensitivity_student_set | ablation | 3/3 | 21.67% | +0.0114 | 17.45% | +0.0096 | differs by more than 1 point |
| appearance_only_student | ablation | 3/3 | 58.10% | +0.0433 | 57.99% | +0.0433 | differs by more than 1 point |
| future_attention_student_independent | ablation | 3/3 | 56.08% | +0.0440 | 56.52% | +0.0443 | differs by more than 1 point |
| future_attention_student_set | ablation | 3/3 | 57.07% | +0.0443 | 56.29% | +0.0438 | within 1 point |
| deletion_cost_student_independent | ablation | 3/3 | 61.33% | +0.0465 | 61.68% | +0.0470 | within 1 point |
| distilled_sensitivity_student_independent | ablation | 3/3 | 60.71% | +0.0465 | 60.43% | +0.0468 | within 1 point |
| shuffled_label_student_independent | ablation | 3/3 | 63.22% | +0.0496 | 63.22% | +0.0496 | within 1 point |
| shuffled_label_student_set | ablation | 3/3 | 64.38% | +0.0501 | 64.37% | +0.0501 | within 1 point |

## Synergy, retrain_readers_old_code, multihead reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| oracle | reference | 1/1 | 23.87% | +0.0000 | 22.92% | +0.0000 | differs by more than 1 point |
| three_cards | reference | 1/1 | n/a | +0.0012 | n/a | +0.0036 | within 1 point |
| full_set_sensitivity | project_baseline | 1/1 | 33.12% | +0.0050 | 35.89% | +0.0077 | differs by more than 1 point |
| fifo | reference | 1/1 | 66.67% | +0.0496 | 66.67% | +0.0476 | within 1 point |
| attention | project_baseline | 1/1 | 72.27% | +0.0579 | 70.27% | +0.0545 | differs by more than 1 point |
| medoid | project_baseline | 1/1 | 97.35% | +0.0678 |  |  | no reference |

## Synergy, retrain_readers_old_code, item_attention reader (S task: needed card deleted, regret)

| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |
|---|---|---|---|---|---|---|---|
| oracle | reference | 1/1 | 20.66% | +0.0000 | 10.20% | +0.0000 | differs by more than 1 point |
| three_cards | reference | 1/1 | n/a | +0.0042 | n/a | +0.0109 | within 1 point |
| full_set_sensitivity | project_baseline | 1/1 | 32.68% | +0.0093 | 37.80% | +0.0179 | differs by more than 1 point |
| attention | project_baseline | 1/1 | 50.00% | +0.0402 | 55.12% | +0.0439 | differs by more than 1 point |
| fifo | reference | 1/1 | 66.67% | +0.0518 | 66.67% | +0.0516 | within 1 point |
| medoid | project_baseline | 1/1 | 98.09% | +0.0663 |  |  | no reference |

## Motion v2, replay from tracked loss arrays (position MSE at +3, ordinary interval)

| Method | Position MSE +3 | Both cue frames kept |
|---|---|---|
| reference/posthoc_best | 0.0025 | 52.6% |
| reference/b | 0.0061 | 100.0% |
| baseline/attention | 0.0091 | 100.0% |
| student/independent_future_attention | 0.0096 | 97.4% |
| student/set_future_attention | 0.0098 | 100.0% |
| reference/last_visible | 0.0111 | 100.0% |
| baseline/diversity | 0.0167 | 9.4% |
| baseline/reservoir | 0.0208 | 4.5% |
| student/set_local_feature | 0.0214 | 5.9% |
| student/set_local_position | 0.0214 | 5.9% |
| student/set_shuffle_local_feature | 0.0214 | 5.9% |
| student/set_shuffle_local_position | 0.0214 | 5.9% |
| student/set_shuffle_use_time_feature | 0.0214 | 5.9% |
| student/set_shuffle_use_time_position | 0.0214 | 5.9% |
| student/set_use_time_feature | 0.0214 | 5.9% |
| student/set_use_time_position | 0.0214 | 5.9% |
| student/independent_local_feature | 0.0221 | 5.0% |
| student/independent_local_position | 0.0221 | 5.0% |
| student/independent_shuffle_local_feature | 0.0221 | 5.0% |
| student/independent_shuffle_local_position | 0.0221 | 5.0% |
| student/independent_shuffle_use_time_feature | 0.0221 | 5.0% |
| student/independent_shuffle_use_time_position | 0.0221 | 5.0% |
| student/independent_use_time_feature | 0.0221 | 5.0% |
| student/independent_use_time_position | 0.0221 | 5.0% |
| baseline/temporal_buckets | 0.0236 | 0.0% |
| baseline/fifo | 0.0238 | 0.0% |
| baseline/surprise | 0.0248 | 0.0% |
| baseline/keep_first | 0.0266 | 0.0% |

120 published cells compared, largest difference 5.0e-09: identical to the published precision.
Student arms that kept their initial weights on every seed: 16 of 18.

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
