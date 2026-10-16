# Frozen motion checkpoint × frame-policy diagnostic

This is a development diagnostic, not a Phase 1 GO result. No component was fitted or selected.

Each row uses one unchanged checkpoint. Old = original start/end; wide = frame 0 / the same end.
All recent/full/random input tensors and prediction bytes match across the two policies.
Wrong evidence swaps the actual selected frames from the matched sibling.

| Checkpoint arm | Seed | Split | Horizon | Old privileged MSE | Wide privileged MSE | Absolute reduction (95% pair CI) |
|---|---|---|---:|---:|---:|---|
| reference | 17 | validation | 3 | 0.085206 | 0.077362 | 0.007845 [0.005545, 0.010381] |
| reference | 29 | validation | 3 | 0.082333 | 0.075249 | 0.007084 [0.004541, 0.009707] |
| reference | 43 | validation | 3 | 0.085110 | 0.076951 | 0.008159 [0.006174, 0.010310] |
| reference | all_seeds_mean | validation | 3 | 0.084216 | 0.076520 | 0.007696 [0.005883, 0.009529] |
| reference | 17 | validation | 6 | 0.086092 | 0.076386 | 0.009706 [0.006927, 0.012830] |
| reference | 29 | validation | 6 | 0.078959 | 0.074478 | 0.004481 [0.002258, 0.006711] |
| reference | 43 | validation | 6 | 0.082733 | 0.076502 | 0.006231 [0.004044, 0.008487] |
| reference | all_seeds_mean | validation | 6 | 0.082595 | 0.075789 | 0.006806 [0.005088, 0.008552] |
| reference | 17 | test_id | 3 | 0.083739 | 0.076342 | 0.007397 [0.005223, 0.009461] |
| reference | 29 | test_id | 3 | 0.084399 | 0.077446 | 0.006953 [0.004632, 0.009334] |
| reference | 43 | test_id | 3 | 0.082132 | 0.075407 | 0.006724 [0.004834, 0.008680] |
| reference | all_seeds_mean | test_id | 3 | 0.083423 | 0.076398 | 0.007025 [0.005301, 0.008744] |
| reference | 17 | test_id | 6 | 0.081844 | 0.077032 | 0.004812 [0.002772, 0.006871] |
| reference | 29 | test_id | 6 | 0.081036 | 0.077285 | 0.003750 [0.001617, 0.005872] |
| reference | 43 | test_id | 6 | 0.082350 | 0.076751 | 0.005599 [0.003667, 0.007515] |
| reference | all_seeds_mean | test_id | 6 | 0.081743 | 0.077023 | 0.004720 [0.003181, 0.006297] |
| reference | 17 | test_long_gap | 3 | 0.143107 | 0.137563 | 0.005544 [0.001595, 0.009360] |
| reference | 29 | test_long_gap | 3 | 0.137652 | 0.133725 | 0.003927 [0.000411, 0.007713] |
| reference | 43 | test_long_gap | 3 | 0.147620 | 0.140076 | 0.007544 [0.003809, 0.011664] |
| reference | all_seeds_mean | test_long_gap | 3 | 0.142793 | 0.137121 | 0.005672 [0.002749, 0.008827] |
| reference | 17 | test_long_gap | 6 | 0.162028 | 0.153668 | 0.008360 [0.004509, 0.012133] |
| reference | 29 | test_long_gap | 6 | 0.156356 | 0.149959 | 0.006398 [0.002520, 0.010746] |
| reference | 43 | test_long_gap | 6 | 0.167842 | 0.157995 | 0.009848 [0.005609, 0.014451] |
| reference | all_seeds_mean | test_long_gap | 6 | 0.162075 | 0.153874 | 0.008202 [0.005179, 0.011317] |
| reference | 17 | test_distractor_shift | 3 | 0.286200 | 0.280039 | 0.006160 [0.001470, 0.011005] |
| reference | 29 | test_distractor_shift | 3 | 0.231894 | 0.225560 | 0.006334 [0.003787, 0.008856] |
| reference | 43 | test_distractor_shift | 3 | 0.244608 | 0.237732 | 0.006875 [0.004423, 0.009434] |
| reference | all_seeds_mean | test_distractor_shift | 3 | 0.254234 | 0.247777 | 0.006456 [0.004121, 0.008763] |
| reference | 17 | test_distractor_shift | 6 | 0.290670 | 0.284495 | 0.006175 [0.001814, 0.010503] |
| reference | 29 | test_distractor_shift | 6 | 0.237048 | 0.232112 | 0.004935 [0.002595, 0.007424] |
| reference | 43 | test_distractor_shift | 6 | 0.243057 | 0.238588 | 0.004469 [0.001635, 0.007316] |
| reference | all_seeds_mean | test_distractor_shift | 6 | 0.256925 | 0.251732 | 0.005193 [0.002861, 0.007615] |
| motion_wide_span | 17 | validation | 3 | 0.088266 | 0.076632 | 0.011634 [0.008968, 0.014576] |
| motion_wide_span | 29 | validation | 3 | 0.084457 | 0.075534 | 0.008923 [0.006554, 0.011307] |
| motion_wide_span | 43 | validation | 3 | 0.089442 | 0.078521 | 0.010920 [0.008273, 0.013695] |
| motion_wide_span | all_seeds_mean | validation | 3 | 0.087388 | 0.076896 | 0.010492 [0.008571, 0.012503] |
| motion_wide_span | 17 | validation | 6 | 0.086487 | 0.076402 | 0.010086 [0.007395, 0.013082] |
| motion_wide_span | 29 | validation | 6 | 0.081992 | 0.074805 | 0.007187 [0.004646, 0.009962] |
| motion_wide_span | 43 | validation | 6 | 0.088023 | 0.078735 | 0.009288 [0.006318, 0.012422] |
| motion_wide_span | all_seeds_mean | validation | 6 | 0.085500 | 0.076647 | 0.008853 [0.006833, 0.011110] |
| motion_wide_span | 17 | test_id | 3 | 0.089376 | 0.077467 | 0.011909 [0.009176, 0.014900] |
| motion_wide_span | 29 | test_id | 3 | 0.086093 | 0.077830 | 0.008263 [0.005718, 0.010705] |
| motion_wide_span | 43 | test_id | 3 | 0.088352 | 0.078930 | 0.009422 [0.007219, 0.011629] |
| motion_wide_span | all_seeds_mean | test_id | 3 | 0.087940 | 0.078076 | 0.009864 [0.007917, 0.011851] |
| motion_wide_span | 17 | test_id | 6 | 0.087560 | 0.078648 | 0.008912 [0.006142, 0.011761] |
| motion_wide_span | 29 | test_id | 6 | 0.086347 | 0.078876 | 0.007472 [0.005011, 0.010018] |
| motion_wide_span | 43 | test_id | 6 | 0.087154 | 0.079629 | 0.007525 [0.005232, 0.009984] |
| motion_wide_span | all_seeds_mean | test_id | 6 | 0.087021 | 0.079051 | 0.007969 [0.005972, 0.010022] |
| motion_wide_span | 17 | test_long_gap | 3 | 0.146824 | 0.139533 | 0.007290 [0.003457, 0.011412] |
| motion_wide_span | 29 | test_long_gap | 3 | 0.140227 | 0.132820 | 0.007407 [0.004541, 0.010464] |
| motion_wide_span | 43 | test_long_gap | 3 | 0.131929 | 0.126872 | 0.005057 [0.002082, 0.008043] |
| motion_wide_span | all_seeds_mean | test_long_gap | 3 | 0.139660 | 0.133075 | 0.006585 [0.004278, 0.009185] |
| motion_wide_span | 17 | test_long_gap | 6 | 0.164976 | 0.154910 | 0.010066 [0.005818, 0.014543] |
| motion_wide_span | 29 | test_long_gap | 6 | 0.160444 | 0.149949 | 0.010495 [0.006719, 0.014621] |
| motion_wide_span | 43 | test_long_gap | 6 | 0.155119 | 0.145784 | 0.009335 [0.005426, 0.013387] |
| motion_wide_span | all_seeds_mean | test_long_gap | 6 | 0.160180 | 0.150214 | 0.009965 [0.006888, 0.013215] |
| motion_wide_span | 17 | test_distractor_shift | 3 | 0.221672 | 0.215739 | 0.005932 [0.003456, 0.008335] |
| motion_wide_span | 29 | test_distractor_shift | 3 | 0.266574 | 0.257570 | 0.009003 [0.005946, 0.012237] |
| motion_wide_span | 43 | test_distractor_shift | 3 | 0.286401 | 0.279836 | 0.006566 [0.003377, 0.009904] |
| motion_wide_span | all_seeds_mean | test_distractor_shift | 3 | 0.258216 | 0.251049 | 0.007167 [0.005242, 0.009294] |
| motion_wide_span | 17 | test_distractor_shift | 6 | 0.221964 | 0.217660 | 0.004304 [0.001399, 0.007223] |
| motion_wide_span | 29 | test_distractor_shift | 6 | 0.271321 | 0.262181 | 0.009140 [0.006053, 0.012219] |
| motion_wide_span | 43 | test_distractor_shift | 6 | 0.285604 | 0.280983 | 0.004621 [0.000864, 0.008834] |
| motion_wide_span | all_seeds_mean | test_distractor_shift | 6 | 0.259630 | 0.253608 | 0.006022 [0.003846, 0.008287] |

Positive reduction means wide is better under that fixed checkpoint.
Pooled intervals average the measured seeds per episode before bootstrapping base sibling pairs; they do not estimate uncertainty over new training seeds.
All per-seed conditions, horizons, slices, wrong-evidence shifts, original-scale errors and pair decompositions are retained in JSON/CSV.
Cross-policy inputs may be outside a checkpoint's training input distribution. These comparisons isolate inference policy at fixed weights; they do not replace retrained-arm or fresh-confirmation tests.
No thresholds or checkpoints were changed and no new confirmation data were opened.
