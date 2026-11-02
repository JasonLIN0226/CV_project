# Motion v2 第3/4步 17925：完整逐seed及配对比较附表

主结论、口径和运行证据见[主报告](MOTION_V2_STEP3_4_RECOVERY1_REPORT_ZH.md)。本文件仅格式化保存的JSON，不重新运行模型、解码科学数组或计算统计。INIT为未训练初始化；所有数值均为均值 [配对95% CI]。success采用百分比，success差采用百分点，MSE保留8位小数；完整精度在原JSON。正的MSE改善=对手−候选，正的success差=候选−对手。每seed表只描述固定seed，不新增门槛。普通和long gap不能跨split配对。

## 1. 全部54臂选点与训练记录

每臂执行1000次更新，每100步及step0共11次validation检查；按自身validation标签CE严格最小、tie选早。记录文件包含每一步训练loss/clip前gradient norm和每次validation CE，checkpoint SHA在students/summary.json及交付核验表中。

| 架构 | 标签 | seed | 选中step | 模型身份 | 初始val CE | 选中val CE | step1000 val CE | 完整训练记录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| independent | future_attention | 17 | 100 | 已训练 | 1.100234441 | 0.944061065 | 0.950958029 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_future_attention_seed17_history.json) |
| independent | future_attention | 29 | 100 | 已训练 | 1.097612549 | 0.945158433 | 0.955200582 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_future_attention_seed29_history.json) |
| independent | future_attention | 43 | 100 | 已训练 | 1.099565700 | 0.941275751 | 0.956962492 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_future_attention_seed43_history.json) |
| independent | local_feature | 17 | 0 | **INIT** | 1.099270291 | 1.099270291 | 1.147712841 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_local_feature_seed17_history.json) |
| independent | local_feature | 29 | 0 | **INIT** | 1.098799102 | 1.098799102 | 1.152249058 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_local_feature_seed29_history.json) |
| independent | local_feature | 43 | 0 | **INIT** | 1.099273427 | 1.099273427 | 1.146022929 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_local_feature_seed43_history.json) |
| independent | local_position | 17 | 0 | **INIT** | 1.099021801 | 1.099021801 | 1.132340487 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_local_position_seed17_history.json) |
| independent | local_position | 29 | 0 | **INIT** | 1.098677623 | 1.098677623 | 1.131609936 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_local_position_seed29_history.json) |
| independent | local_position | 43 | 0 | **INIT** | 1.099113578 | 1.099113578 | 1.134764807 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_local_position_seed43_history.json) |
| independent | shuffle_local_feature | 17 | 0 | **INIT** | 1.099462319 | 1.099462319 | 1.119923243 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_local_feature_seed17_history.json) |
| independent | shuffle_local_feature | 29 | 0 | **INIT** | 1.098852487 | 1.098852487 | 1.125471274 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_local_feature_seed29_history.json) |
| independent | shuffle_local_feature | 43 | 0 | **INIT** | 1.099510262 | 1.099510262 | 1.123707370 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_local_feature_seed43_history.json) |
| independent | shuffle_local_position | 17 | 0 | **INIT** | 1.099773301 | 1.099773301 | 1.121581664 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_local_position_seed17_history.json) |
| independent | shuffle_local_position | 29 | 0 | **INIT** | 1.098627841 | 1.098627841 | 1.118975233 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_local_position_seed29_history.json) |
| independent | shuffle_local_position | 43 | 0 | **INIT** | 1.099476905 | 1.099476905 | 1.122346005 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_local_position_seed43_history.json) |
| independent | shuffle_use_time_feature | 17 | 0 | **INIT** | 1.099257419 | 1.099257419 | 1.124195669 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_use_time_feature_seed17_history.json) |
| independent | shuffle_use_time_feature | 29 | 0 | **INIT** | 1.099305270 | 1.099305270 | 1.123531844 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_use_time_feature_seed29_history.json) |
| independent | shuffle_use_time_feature | 43 | 0 | **INIT** | 1.099249299 | 1.099249299 | 1.125473959 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_use_time_feature_seed43_history.json) |
| independent | shuffle_use_time_position | 17 | 0 | **INIT** | 1.099291984 | 1.099291984 | 1.122219208 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_use_time_position_seed17_history.json) |
| independent | shuffle_use_time_position | 29 | 0 | **INIT** | 1.099013353 | 1.099013353 | 1.122580169 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_use_time_position_seed29_history.json) |
| independent | shuffle_use_time_position | 43 | 0 | **INIT** | 1.099296775 | 1.099296775 | 1.118431664 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_shuffle_use_time_position_seed43_history.json) |
| independent | use_time_feature | 17 | 0 | **INIT** | 1.099325732 | 1.099325732 | 1.144373388 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_use_time_feature_seed17_history.json) |
| independent | use_time_feature | 29 | 0 | **INIT** | 1.099240358 | 1.099240358 | 1.146621904 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_use_time_feature_seed29_history.json) |
| independent | use_time_feature | 43 | 0 | **INIT** | 1.099149974 | 1.099149974 | 1.149707186 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_use_time_feature_seed43_history.json) |
| independent | use_time_position | 17 | 0 | **INIT** | 1.098975987 | 1.098975987 | 1.125817933 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_use_time_position_seed17_history.json) |
| independent | use_time_position | 29 | 0 | **INIT** | 1.099029911 | 1.099029911 | 1.118454852 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_use_time_position_seed29_history.json) |
| independent | use_time_position | 43 | 0 | **INIT** | 1.099072213 | 1.099072213 | 1.125889251 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/independent_use_time_position_seed43_history.json) |
| set | future_attention | 17 | 1000 | 已训练 | 1.097059177 | 0.901172124 | 0.901172124 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_future_attention_seed17_history.json) |
| set | future_attention | 29 | 800 | 已训练 | 1.098968283 | 0.902565153 | 0.902949225 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_future_attention_seed29_history.json) |
| set | future_attention | 43 | 700 | 已训练 | 1.097772813 | 0.900973228 | 0.901608534 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_future_attention_seed43_history.json) |
| set | local_feature | 17 | 0 | **INIT** | 1.099492725 | 1.099492725 | 1.155231461 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_local_feature_seed17_history.json) |
| set | local_feature | 29 | 0 | **INIT** | 1.098721747 | 1.098721747 | 1.150024663 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_local_feature_seed29_history.json) |
| set | local_feature | 43 | 0 | **INIT** | 1.098528283 | 1.098528283 | 1.150700376 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_local_feature_seed43_history.json) |
| set | local_position | 17 | 0 | **INIT** | 1.099421855 | 1.099421855 | 1.130727228 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_local_position_seed17_history.json) |
| set | local_position | 29 | 0 | **INIT** | 1.098135047 | 1.098135047 | 1.132851291 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_local_position_seed29_history.json) |
| set | local_position | 43 | 0 | **INIT** | 1.098621844 | 1.098621844 | 1.138727616 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_local_position_seed43_history.json) |
| set | shuffle_local_feature | 17 | 0 | **INIT** | 1.098730801 | 1.098730801 | 1.127317280 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_local_feature_seed17_history.json) |
| set | shuffle_local_feature | 29 | 0 | **INIT** | 1.098895847 | 1.098895847 | 1.123704134 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_local_feature_seed29_history.json) |
| set | shuffle_local_feature | 43 | 0 | **INIT** | 1.098717816 | 1.098717816 | 1.126705972 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_local_feature_seed43_history.json) |
| set | shuffle_local_position | 17 | 0 | **INIT** | 1.098795084 | 1.098795084 | 1.118312929 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_local_position_seed17_history.json) |
| set | shuffle_local_position | 29 | 0 | **INIT** | 1.098895913 | 1.098895913 | 1.122818536 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_local_position_seed29_history.json) |
| set | shuffle_local_position | 43 | 0 | **INIT** | 1.098639454 | 1.098639454 | 1.123569077 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_local_position_seed43_history.json) |
| set | shuffle_use_time_feature | 17 | 0 | **INIT** | 1.098959601 | 1.098959601 | 1.120556726 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_use_time_feature_seed17_history.json) |
| set | shuffle_use_time_feature | 29 | 0 | **INIT** | 1.099008226 | 1.099008226 | 1.122631295 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_use_time_feature_seed29_history.json) |
| set | shuffle_use_time_feature | 43 | 0 | **INIT** | 1.098653155 | 1.098653155 | 1.123439557 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_use_time_feature_seed43_history.json) |
| set | shuffle_use_time_position | 17 | 0 | **INIT** | 1.098950938 | 1.098950938 | 1.119338618 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_use_time_position_seed17_history.json) |
| set | shuffle_use_time_position | 29 | 0 | **INIT** | 1.098964266 | 1.098964266 | 1.119367869 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_use_time_position_seed29_history.json) |
| set | shuffle_use_time_position | 43 | 0 | **INIT** | 1.098575456 | 1.098575456 | 1.121566283 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_shuffle_use_time_position_seed43_history.json) |
| set | use_time_feature | 17 | 0 | **INIT** | 1.099363316 | 1.099363316 | 1.139543956 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_use_time_feature_seed17_history.json) |
| set | use_time_feature | 29 | 0 | **INIT** | 1.098635880 | 1.098635880 | 1.148435257 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_use_time_feature_seed29_history.json) |
| set | use_time_feature | 43 | 0 | **INIT** | 1.098778377 | 1.098778377 | 1.150522974 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_use_time_feature_seed43_history.json) |
| set | use_time_position | 17 | 0 | **INIT** | 1.099234566 | 1.099234566 | 1.121111795 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_use_time_position_seed17_history.json) |
| set | use_time_position | 29 | 0 | **INIT** | 1.098267880 | 1.098267880 | 1.122518578 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_use_time_position_seed29_history.json) |
| set | use_time_position | 43 | 0 | **INIT** | 1.098628340 | 1.098628340 | 1.125308610 | [history](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/set_use_time_position_seed43_history.json) |

## 2. 全部标签的聚合配对比较
### ordinary：四种集合删除代价 × 全部8个对手

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 5.903 [3.819, 8.333] | 0.00239137 [0.00086175, 0.00416733] | 0.00389113 [-0.00010994, 0.00818087] | 是 |
| local_feature | fifo | 6 | 5.903 [3.819, 8.333] | 0.00205010 [0.00047693, 0.00377644] | 0.00362521 [-0.00032004, 0.00768395] | 是 |
| local_feature | reservoir | 3 | 1.389 [-1.910, 4.861] | -0.00055941 [-0.00190570, 0.00085895] | -0.00135033 [-0.00495068, 0.00276648] | 否 |
| local_feature | reservoir | 6 | 1.389 [-1.910, 4.861] | -0.00101163 [-0.00234286, 0.00045220] | -0.00282009 [-0.00651386, 0.00132528] | 否 |
| local_feature | temporal_buckets | 3 | 5.903 [3.819, 8.333] | 0.00219708 [0.00088672, 0.00373226] | 0.00068463 [-0.00329650, 0.00521495] | 是 |
| local_feature | temporal_buckets | 6 | 5.903 [3.819, 8.333] | 0.00176112 [0.00052675, 0.00320087] | -0.00036623 [-0.00367211, 0.00313502] | 是 |
| local_feature | diversity | 3 | -3.472 [-9.028, 2.083] | -0.00473919 [-0.00652662, -0.00303161] | -0.01053408 [-0.01634224, -0.00451977] | 否 |
| local_feature | diversity | 6 | -3.472 [-9.028, 2.083] | -0.00471082 [-0.00655505, -0.00288882] | -0.01264537 [-0.01938553, -0.00589546] | 否 |
| local_feature | surprise | 3 | 5.903 [3.819, 8.333] | 0.00343050 [0.00157069, 0.00564920] | 0.00995653 [0.00431966, 0.01606860] | 是 |
| local_feature | surprise | 6 | 5.903 [3.819, 8.333] | 0.00346751 [0.00158280, 0.00576021] | 0.01359142 [0.00863956, 0.01900100] | 是 |
| local_feature | attention | 3 | -94.097 [-96.181, -91.667] | -0.01225465 [-0.01455777, -0.01007724] | -0.03901084 [-0.04664150, -0.03092612] | 否 |
| local_feature | attention | 6 | -94.097 [-96.181, -91.667] | -0.01415738 [-0.01632589, -0.01206640] | -0.03738140 [-0.04649321, -0.02771790] | 否 |
| local_feature | keep_first | 3 | 5.903 [3.819, 8.333] | 0.00523970 [0.00316140, 0.00746197] | 0.02239897 [0.01551253, 0.02978129] | 是 |
| local_feature | keep_first | 6 | 5.903 [3.819, 8.333] | 0.00491782 [0.00292056, 0.00711871] | 0.02263609 [0.01679350, 0.02911045] | 是 |
| local_feature | set/future_attention | 3 | -94.097 [-96.181, -91.667] | -0.01164500 [-0.01363568, -0.00980478] | -0.03748243 [-0.04546002, -0.02912963] | 否 |
| local_feature | set/future_attention | 6 | -94.097 [-96.181, -91.667] | -0.01355392 [-0.01552995, -0.01155612] | -0.03494539 [-0.04272691, -0.02687234] | 否 |
| local_position | fifo | 3 | 5.903 [3.819, 8.333] | 0.00239137 [0.00086175, 0.00416733] | 0.00389113 [-0.00010994, 0.00818087] | 是 |
| local_position | fifo | 6 | 5.903 [3.819, 8.333] | 0.00205010 [0.00047693, 0.00377644] | 0.00362521 [-0.00032004, 0.00768395] | 是 |
| local_position | reservoir | 3 | 1.389 [-1.910, 4.861] | -0.00055941 [-0.00190570, 0.00085895] | -0.00135033 [-0.00495068, 0.00276648] | 否 |
| local_position | reservoir | 6 | 1.389 [-1.910, 4.861] | -0.00101163 [-0.00234286, 0.00045220] | -0.00282009 [-0.00651386, 0.00132528] | 否 |
| local_position | temporal_buckets | 3 | 5.903 [3.819, 8.333] | 0.00219708 [0.00088672, 0.00373226] | 0.00068463 [-0.00329650, 0.00521495] | 是 |
| local_position | temporal_buckets | 6 | 5.903 [3.819, 8.333] | 0.00176112 [0.00052675, 0.00320087] | -0.00036623 [-0.00367211, 0.00313502] | 是 |
| local_position | diversity | 3 | -3.472 [-9.028, 2.083] | -0.00473919 [-0.00652662, -0.00303161] | -0.01053408 [-0.01634224, -0.00451977] | 否 |
| local_position | diversity | 6 | -3.472 [-9.028, 2.083] | -0.00471082 [-0.00655505, -0.00288882] | -0.01264537 [-0.01938553, -0.00589546] | 否 |
| local_position | surprise | 3 | 5.903 [3.819, 8.333] | 0.00343050 [0.00157069, 0.00564920] | 0.00995653 [0.00431966, 0.01606860] | 是 |
| local_position | surprise | 6 | 5.903 [3.819, 8.333] | 0.00346751 [0.00158280, 0.00576021] | 0.01359142 [0.00863956, 0.01900100] | 是 |
| local_position | attention | 3 | -94.097 [-96.181, -91.667] | -0.01225465 [-0.01455777, -0.01007724] | -0.03901084 [-0.04664150, -0.03092612] | 否 |
| local_position | attention | 6 | -94.097 [-96.181, -91.667] | -0.01415738 [-0.01632589, -0.01206640] | -0.03738140 [-0.04649321, -0.02771790] | 否 |
| local_position | keep_first | 3 | 5.903 [3.819, 8.333] | 0.00523970 [0.00316140, 0.00746197] | 0.02239897 [0.01551253, 0.02978129] | 是 |
| local_position | keep_first | 6 | 5.903 [3.819, 8.333] | 0.00491782 [0.00292056, 0.00711871] | 0.02263609 [0.01679350, 0.02911045] | 是 |
| local_position | set/future_attention | 3 | -94.097 [-96.181, -91.667] | -0.01164500 [-0.01363568, -0.00980478] | -0.03748243 [-0.04546002, -0.02912963] | 否 |
| local_position | set/future_attention | 6 | -94.097 [-96.181, -91.667] | -0.01355392 [-0.01552995, -0.01155612] | -0.03494539 [-0.04272691, -0.02687234] | 否 |
| use_time_feature | fifo | 3 | 5.903 [3.819, 8.333] | 0.00239137 [0.00086175, 0.00416733] | 0.00389113 [-0.00010994, 0.00818087] | 是 |
| use_time_feature | fifo | 6 | 5.903 [3.819, 8.333] | 0.00205010 [0.00047693, 0.00377644] | 0.00362521 [-0.00032004, 0.00768395] | 是 |
| use_time_feature | reservoir | 3 | 1.389 [-1.910, 4.861] | -0.00055941 [-0.00190570, 0.00085895] | -0.00135033 [-0.00495068, 0.00276648] | 否 |
| use_time_feature | reservoir | 6 | 1.389 [-1.910, 4.861] | -0.00101163 [-0.00234286, 0.00045220] | -0.00282009 [-0.00651386, 0.00132528] | 否 |
| use_time_feature | temporal_buckets | 3 | 5.903 [3.819, 8.333] | 0.00219708 [0.00088672, 0.00373226] | 0.00068463 [-0.00329650, 0.00521495] | 是 |
| use_time_feature | temporal_buckets | 6 | 5.903 [3.819, 8.333] | 0.00176112 [0.00052675, 0.00320087] | -0.00036623 [-0.00367211, 0.00313502] | 是 |
| use_time_feature | diversity | 3 | -3.472 [-9.028, 2.083] | -0.00473919 [-0.00652662, -0.00303161] | -0.01053408 [-0.01634224, -0.00451977] | 否 |
| use_time_feature | diversity | 6 | -3.472 [-9.028, 2.083] | -0.00471082 [-0.00655505, -0.00288882] | -0.01264537 [-0.01938553, -0.00589546] | 否 |
| use_time_feature | surprise | 3 | 5.903 [3.819, 8.333] | 0.00343050 [0.00157069, 0.00564920] | 0.00995653 [0.00431966, 0.01606860] | 是 |
| use_time_feature | surprise | 6 | 5.903 [3.819, 8.333] | 0.00346751 [0.00158280, 0.00576021] | 0.01359142 [0.00863956, 0.01900100] | 是 |
| use_time_feature | attention | 3 | -94.097 [-96.181, -91.667] | -0.01225465 [-0.01455777, -0.01007724] | -0.03901084 [-0.04664150, -0.03092612] | 否 |
| use_time_feature | attention | 6 | -94.097 [-96.181, -91.667] | -0.01415738 [-0.01632589, -0.01206640] | -0.03738140 [-0.04649321, -0.02771790] | 否 |
| use_time_feature | keep_first | 3 | 5.903 [3.819, 8.333] | 0.00523970 [0.00316140, 0.00746197] | 0.02239897 [0.01551253, 0.02978129] | 是 |
| use_time_feature | keep_first | 6 | 5.903 [3.819, 8.333] | 0.00491782 [0.00292056, 0.00711871] | 0.02263609 [0.01679350, 0.02911045] | 是 |
| use_time_feature | set/future_attention | 3 | -94.097 [-96.181, -91.667] | -0.01164500 [-0.01363568, -0.00980478] | -0.03748243 [-0.04546002, -0.02912963] | 否 |
| use_time_feature | set/future_attention | 6 | -94.097 [-96.181, -91.667] | -0.01355392 [-0.01552995, -0.01155612] | -0.03494539 [-0.04272691, -0.02687234] | 否 |
| use_time_position | fifo | 3 | 5.903 [3.819, 8.333] | 0.00239137 [0.00086175, 0.00416733] | 0.00389113 [-0.00010994, 0.00818087] | 是 |
| use_time_position | fifo | 6 | 5.903 [3.819, 8.333] | 0.00205010 [0.00047693, 0.00377644] | 0.00362521 [-0.00032004, 0.00768395] | 是 |
| use_time_position | reservoir | 3 | 1.389 [-1.910, 4.861] | -0.00055941 [-0.00190570, 0.00085895] | -0.00135033 [-0.00495068, 0.00276648] | 否 |
| use_time_position | reservoir | 6 | 1.389 [-1.910, 4.861] | -0.00101163 [-0.00234286, 0.00045220] | -0.00282009 [-0.00651386, 0.00132528] | 否 |
| use_time_position | temporal_buckets | 3 | 5.903 [3.819, 8.333] | 0.00219708 [0.00088672, 0.00373226] | 0.00068463 [-0.00329650, 0.00521495] | 是 |
| use_time_position | temporal_buckets | 6 | 5.903 [3.819, 8.333] | 0.00176112 [0.00052675, 0.00320087] | -0.00036623 [-0.00367211, 0.00313502] | 是 |
| use_time_position | diversity | 3 | -3.472 [-9.028, 2.083] | -0.00473919 [-0.00652662, -0.00303161] | -0.01053408 [-0.01634224, -0.00451977] | 否 |
| use_time_position | diversity | 6 | -3.472 [-9.028, 2.083] | -0.00471082 [-0.00655505, -0.00288882] | -0.01264537 [-0.01938553, -0.00589546] | 否 |
| use_time_position | surprise | 3 | 5.903 [3.819, 8.333] | 0.00343050 [0.00157069, 0.00564920] | 0.00995653 [0.00431966, 0.01606860] | 是 |
| use_time_position | surprise | 6 | 5.903 [3.819, 8.333] | 0.00346751 [0.00158280, 0.00576021] | 0.01359142 [0.00863956, 0.01900100] | 是 |
| use_time_position | attention | 3 | -94.097 [-96.181, -91.667] | -0.01225465 [-0.01455777, -0.01007724] | -0.03901084 [-0.04664150, -0.03092612] | 否 |
| use_time_position | attention | 6 | -94.097 [-96.181, -91.667] | -0.01415738 [-0.01632589, -0.01206640] | -0.03738140 [-0.04649321, -0.02771790] | 否 |
| use_time_position | keep_first | 3 | 5.903 [3.819, 8.333] | 0.00523970 [0.00316140, 0.00746197] | 0.02239897 [0.01551253, 0.02978129] | 是 |
| use_time_position | keep_first | 6 | 5.903 [3.819, 8.333] | 0.00491782 [0.00292056, 0.00711871] | 0.02263609 [0.01679350, 0.02911045] | 是 |
| use_time_position | set/future_attention | 3 | -94.097 [-96.181, -91.667] | -0.01164500 [-0.01363568, -0.00980478] | -0.03748243 [-0.04546002, -0.02912963] | 否 |
| use_time_position | set/future_attention | 6 | -94.097 [-96.181, -91.667] | -0.01355392 [-0.01552995, -0.01155612] | -0.03494539 [-0.04272691, -0.02687234] | 否 |

### ordinary：四种同标签集合/独立比较

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | 0.868 [-2.257, 3.993] | 0.00066321 [-0.00052329, 0.00180378] | 0.00014570 [-0.00323467, 0.00357075] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | 0.868 [-2.257, 3.993] | 0.00039854 [-0.00077887, 0.00157679] | -0.00114078 [-0.00455442, 0.00221714] | 否 |
| local_position | independent/local_position **INIT** | 3 | 0.868 [-2.257, 3.993] | 0.00066321 [-0.00052329, 0.00180378] | 0.00014570 [-0.00323467, 0.00357075] | 否 |
| local_position | independent/local_position **INIT** | 6 | 0.868 [-2.257, 3.993] | 0.00039854 [-0.00077887, 0.00157679] | -0.00114078 [-0.00455442, 0.00221714] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | 0.868 [-2.257, 3.993] | 0.00066321 [-0.00052329, 0.00180378] | 0.00014570 [-0.00323467, 0.00357075] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | 0.868 [-2.257, 3.993] | 0.00039854 [-0.00077887, 0.00157679] | -0.00114078 [-0.00455442, 0.00221714] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | 0.868 [-2.257, 3.993] | 0.00066321 [-0.00052329, 0.00180378] | 0.00014570 [-0.00323467, 0.00357075] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | 0.868 [-2.257, 3.993] | 0.00039854 [-0.00077887, 0.00157679] | -0.00114078 [-0.00455442, 0.00221714] | 否 |

### long_gap：四种集合删除代价 × 全部8个对手

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 1.562 [0.521, 2.778] | 0.00060751 [-0.00039285, 0.00163794] | -0.00294359 [-0.00734782, 0.00123529] | 否 |
| local_feature | fifo | 6 | 1.562 [0.521, 2.778] | 0.00016751 [-0.00079566, 0.00108879] | -0.00630262 [-0.01113557, -0.00137038] | 否 |
| local_feature | reservoir | 3 | 0.521 [-1.215, 2.257] | 0.00024940 [-0.00075285, 0.00126786] | 0.00685425 [0.00270527, 0.01072522] | 否 |
| local_feature | reservoir | 6 | 0.521 [-1.215, 2.257] | 0.00018321 [-0.00081215, 0.00121261] | 0.00638415 [0.00155789, 0.01124094] | 否 |
| local_feature | temporal_buckets | 3 | 1.562 [0.521, 2.778] | -0.00017351 [-0.00102359, 0.00067813] | -0.00614850 [-0.00990590, -0.00269097] | 否 |
| local_feature | temporal_buckets | 6 | 1.562 [0.521, 2.778] | -0.00059696 [-0.00147519, 0.00023598] | -0.01025267 [-0.01508086, -0.00560838] | 否 |
| local_feature | diversity | 3 | -2.083 [-5.556, 0.694] | -0.00002808 [-0.00255374, 0.00248636] | 0.02314932 [0.01612694, 0.02995139] | 否 |
| local_feature | diversity | 6 | -2.083 [-5.556, 0.694] | 0.00187343 [-0.00072642, 0.00443867] | 0.02592137 [0.01865397, 0.03338292] | 否 |
| local_feature | surprise | 3 | 1.562 [0.521, 2.778] | 0.00233117 [0.00075684, 0.00405822] | 0.01808657 [0.01184391, 0.02482075] | 是 |
| local_feature | surprise | 6 | 1.562 [0.521, 2.778] | 0.00223384 [0.00058820, 0.00403088] | 0.01930319 [0.01134287, 0.02769451] | 是 |
| local_feature | attention | 3 | -98.438 [-99.479, -97.222] | -0.01562908 [-0.01950486, -0.01174498] | 0.07935918 [0.06468510, 0.09462173] | 否 |
| local_feature | attention | 6 | -98.438 [-99.479, -97.222] | -0.01391793 [-0.01840219, -0.00958891] | 0.10093026 [0.08293153, 0.12053033] | 否 |
| local_feature | keep_first | 3 | 1.562 [0.521, 2.778] | 0.00312118 [0.00164774, 0.00457799] | 0.03733646 [0.02895222, 0.04507544] | 是 |
| local_feature | keep_first | 6 | 1.562 [0.521, 2.778] | 0.00342778 [0.00181337, 0.00505673] | 0.03923518 [0.02963392, 0.04918741] | 是 |
| local_feature | set/future_attention | 3 | -98.438 [-99.479, -97.222] | -0.01615634 [-0.02010107, -0.01218810] | 0.08097049 [0.06666184, 0.09607993] | 否 |
| local_feature | set/future_attention | 6 | -98.438 [-99.479, -97.222] | -0.01396682 [-0.01856220, -0.00951630] | 0.10348421 [0.08620745, 0.12213963] | 否 |
| local_position | fifo | 3 | 1.562 [0.521, 2.778] | 0.00060751 [-0.00039285, 0.00163794] | -0.00294359 [-0.00734782, 0.00123529] | 否 |
| local_position | fifo | 6 | 1.562 [0.521, 2.778] | 0.00016751 [-0.00079566, 0.00108879] | -0.00630262 [-0.01113557, -0.00137038] | 否 |
| local_position | reservoir | 3 | 0.521 [-1.215, 2.257] | 0.00024940 [-0.00075285, 0.00126786] | 0.00685425 [0.00270527, 0.01072522] | 否 |
| local_position | reservoir | 6 | 0.521 [-1.215, 2.257] | 0.00018321 [-0.00081215, 0.00121261] | 0.00638415 [0.00155789, 0.01124094] | 否 |
| local_position | temporal_buckets | 3 | 1.562 [0.521, 2.778] | -0.00017351 [-0.00102359, 0.00067813] | -0.00614850 [-0.00990590, -0.00269097] | 否 |
| local_position | temporal_buckets | 6 | 1.562 [0.521, 2.778] | -0.00059696 [-0.00147519, 0.00023598] | -0.01025267 [-0.01508086, -0.00560838] | 否 |
| local_position | diversity | 3 | -2.083 [-5.556, 0.694] | -0.00002808 [-0.00255374, 0.00248636] | 0.02314932 [0.01612694, 0.02995139] | 否 |
| local_position | diversity | 6 | -2.083 [-5.556, 0.694] | 0.00187343 [-0.00072642, 0.00443867] | 0.02592137 [0.01865397, 0.03338292] | 否 |
| local_position | surprise | 3 | 1.562 [0.521, 2.778] | 0.00233117 [0.00075684, 0.00405822] | 0.01808657 [0.01184391, 0.02482075] | 是 |
| local_position | surprise | 6 | 1.562 [0.521, 2.778] | 0.00223384 [0.00058820, 0.00403088] | 0.01930319 [0.01134287, 0.02769451] | 是 |
| local_position | attention | 3 | -98.438 [-99.479, -97.222] | -0.01562908 [-0.01950486, -0.01174498] | 0.07935918 [0.06468510, 0.09462173] | 否 |
| local_position | attention | 6 | -98.438 [-99.479, -97.222] | -0.01391793 [-0.01840219, -0.00958891] | 0.10093026 [0.08293153, 0.12053033] | 否 |
| local_position | keep_first | 3 | 1.562 [0.521, 2.778] | 0.00312118 [0.00164774, 0.00457799] | 0.03733646 [0.02895222, 0.04507544] | 是 |
| local_position | keep_first | 6 | 1.562 [0.521, 2.778] | 0.00342778 [0.00181337, 0.00505673] | 0.03923518 [0.02963392, 0.04918741] | 是 |
| local_position | set/future_attention | 3 | -98.438 [-99.479, -97.222] | -0.01615634 [-0.02010107, -0.01218810] | 0.08097049 [0.06666184, 0.09607993] | 否 |
| local_position | set/future_attention | 6 | -98.438 [-99.479, -97.222] | -0.01396682 [-0.01856220, -0.00951630] | 0.10348421 [0.08620745, 0.12213963] | 否 |
| use_time_feature | fifo | 3 | 1.562 [0.521, 2.778] | 0.00060751 [-0.00039285, 0.00163794] | -0.00294359 [-0.00734782, 0.00123529] | 否 |
| use_time_feature | fifo | 6 | 1.562 [0.521, 2.778] | 0.00016751 [-0.00079566, 0.00108879] | -0.00630262 [-0.01113557, -0.00137038] | 否 |
| use_time_feature | reservoir | 3 | 0.521 [-1.215, 2.257] | 0.00024940 [-0.00075285, 0.00126786] | 0.00685425 [0.00270527, 0.01072522] | 否 |
| use_time_feature | reservoir | 6 | 0.521 [-1.215, 2.257] | 0.00018321 [-0.00081215, 0.00121261] | 0.00638415 [0.00155789, 0.01124094] | 否 |
| use_time_feature | temporal_buckets | 3 | 1.562 [0.521, 2.778] | -0.00017351 [-0.00102359, 0.00067813] | -0.00614850 [-0.00990590, -0.00269097] | 否 |
| use_time_feature | temporal_buckets | 6 | 1.562 [0.521, 2.778] | -0.00059696 [-0.00147519, 0.00023598] | -0.01025267 [-0.01508086, -0.00560838] | 否 |
| use_time_feature | diversity | 3 | -2.083 [-5.556, 0.694] | -0.00002808 [-0.00255374, 0.00248636] | 0.02314932 [0.01612694, 0.02995139] | 否 |
| use_time_feature | diversity | 6 | -2.083 [-5.556, 0.694] | 0.00187343 [-0.00072642, 0.00443867] | 0.02592137 [0.01865397, 0.03338292] | 否 |
| use_time_feature | surprise | 3 | 1.562 [0.521, 2.778] | 0.00233117 [0.00075684, 0.00405822] | 0.01808657 [0.01184391, 0.02482075] | 是 |
| use_time_feature | surprise | 6 | 1.562 [0.521, 2.778] | 0.00223384 [0.00058820, 0.00403088] | 0.01930319 [0.01134287, 0.02769451] | 是 |
| use_time_feature | attention | 3 | -98.438 [-99.479, -97.222] | -0.01562908 [-0.01950486, -0.01174498] | 0.07935918 [0.06468510, 0.09462173] | 否 |
| use_time_feature | attention | 6 | -98.438 [-99.479, -97.222] | -0.01391793 [-0.01840219, -0.00958891] | 0.10093026 [0.08293153, 0.12053033] | 否 |
| use_time_feature | keep_first | 3 | 1.562 [0.521, 2.778] | 0.00312118 [0.00164774, 0.00457799] | 0.03733646 [0.02895222, 0.04507544] | 是 |
| use_time_feature | keep_first | 6 | 1.562 [0.521, 2.778] | 0.00342778 [0.00181337, 0.00505673] | 0.03923518 [0.02963392, 0.04918741] | 是 |
| use_time_feature | set/future_attention | 3 | -98.438 [-99.479, -97.222] | -0.01615634 [-0.02010107, -0.01218810] | 0.08097049 [0.06666184, 0.09607993] | 否 |
| use_time_feature | set/future_attention | 6 | -98.438 [-99.479, -97.222] | -0.01396682 [-0.01856220, -0.00951630] | 0.10348421 [0.08620745, 0.12213963] | 否 |
| use_time_position | fifo | 3 | 1.562 [0.521, 2.778] | 0.00060751 [-0.00039285, 0.00163794] | -0.00294359 [-0.00734782, 0.00123529] | 否 |
| use_time_position | fifo | 6 | 1.562 [0.521, 2.778] | 0.00016751 [-0.00079566, 0.00108879] | -0.00630262 [-0.01113557, -0.00137038] | 否 |
| use_time_position | reservoir | 3 | 0.521 [-1.215, 2.257] | 0.00024940 [-0.00075285, 0.00126786] | 0.00685425 [0.00270527, 0.01072522] | 否 |
| use_time_position | reservoir | 6 | 0.521 [-1.215, 2.257] | 0.00018321 [-0.00081215, 0.00121261] | 0.00638415 [0.00155789, 0.01124094] | 否 |
| use_time_position | temporal_buckets | 3 | 1.562 [0.521, 2.778] | -0.00017351 [-0.00102359, 0.00067813] | -0.00614850 [-0.00990590, -0.00269097] | 否 |
| use_time_position | temporal_buckets | 6 | 1.562 [0.521, 2.778] | -0.00059696 [-0.00147519, 0.00023598] | -0.01025267 [-0.01508086, -0.00560838] | 否 |
| use_time_position | diversity | 3 | -2.083 [-5.556, 0.694] | -0.00002808 [-0.00255374, 0.00248636] | 0.02314932 [0.01612694, 0.02995139] | 否 |
| use_time_position | diversity | 6 | -2.083 [-5.556, 0.694] | 0.00187343 [-0.00072642, 0.00443867] | 0.02592137 [0.01865397, 0.03338292] | 否 |
| use_time_position | surprise | 3 | 1.562 [0.521, 2.778] | 0.00233117 [0.00075684, 0.00405822] | 0.01808657 [0.01184391, 0.02482075] | 是 |
| use_time_position | surprise | 6 | 1.562 [0.521, 2.778] | 0.00223384 [0.00058820, 0.00403088] | 0.01930319 [0.01134287, 0.02769451] | 是 |
| use_time_position | attention | 3 | -98.438 [-99.479, -97.222] | -0.01562908 [-0.01950486, -0.01174498] | 0.07935918 [0.06468510, 0.09462173] | 否 |
| use_time_position | attention | 6 | -98.438 [-99.479, -97.222] | -0.01391793 [-0.01840219, -0.00958891] | 0.10093026 [0.08293153, 0.12053033] | 否 |
| use_time_position | keep_first | 3 | 1.562 [0.521, 2.778] | 0.00312118 [0.00164774, 0.00457799] | 0.03733646 [0.02895222, 0.04507544] | 是 |
| use_time_position | keep_first | 6 | 1.562 [0.521, 2.778] | 0.00342778 [0.00181337, 0.00505673] | 0.03923518 [0.02963392, 0.04918741] | 是 |
| use_time_position | set/future_attention | 3 | -98.438 [-99.479, -97.222] | -0.01615634 [-0.02010107, -0.01218810] | 0.08097049 [0.06666184, 0.09607993] | 否 |
| use_time_position | set/future_attention | 6 | -98.438 [-99.479, -97.222] | -0.01396682 [-0.01856220, -0.00951630] | 0.10348421 [0.08620745, 0.12213963] | 否 |

### long_gap：四种同标签集合/独立比较

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | 0.521 [-1.042, 2.083] | 0.00021587 [-0.00077626, 0.00121421] | 0.00364783 [-0.00045108, 0.00787560] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | 0.521 [-1.042, 2.083] | 0.00012609 [-0.00082459, 0.00108909] | 0.00137843 [-0.00322447, 0.00564780] | 否 |
| local_position | independent/local_position **INIT** | 3 | 0.521 [-1.042, 2.083] | 0.00021587 [-0.00077626, 0.00121421] | 0.00364783 [-0.00045108, 0.00787560] | 否 |
| local_position | independent/local_position **INIT** | 6 | 0.521 [-1.042, 2.083] | 0.00012609 [-0.00082459, 0.00108909] | 0.00137843 [-0.00322447, 0.00564780] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | 0.521 [-1.042, 2.083] | 0.00021587 [-0.00077626, 0.00121421] | 0.00364783 [-0.00045108, 0.00787560] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | 0.521 [-1.042, 2.083] | 0.00012609 [-0.00082459, 0.00108909] | 0.00137843 [-0.00322447, 0.00564780] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | 0.521 [-1.042, 2.083] | 0.00021587 [-0.00077626, 0.00121421] | 0.00364783 [-0.00045108, 0.00787560] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | 0.521 [-1.042, 2.083] | 0.00012609 [-0.00082459, 0.00108909] | 0.00137843 [-0.00322447, 0.00564780] | 否 |

## 3. 预测器逐seed诊断
### ordinary / predictor seed 17

| memory 条件 | h | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- |
| b | 3 | 0.00591595 [0.00420802, 0.00792924] | 0.18453655 [0.17301906, 0.19673446] |
| b | 6 | 0.00716695 [0.00546989, 0.00901420] | 0.19722937 [0.18608108, 0.20786069] |
| d | 3 | 0.01618040 [0.01433357, 0.01809723] | 0.24148145 [0.23252918, 0.25116152] |
| d | 6 | 0.02011633 [0.01853882, 0.02173682] | 0.24897846 [0.23936718, 0.25851689] |
| b_random | 3 | 0.01005229 [0.00781258, 0.01249563] | 0.20757887 [0.19581620, 0.22010385] |
| b_random | 6 | 0.01222668 [0.00989090, 0.01479173] | 0.22545060 [0.21425159, 0.23648341] |


| h | B [95% CI] | 真实 target feature 的 decoder position MSE [95% CI] |
| --- | --- | --- |
| 3 | 0.01050308 [0.00981344, 0.01118945] | 0.00001694 [0.00001284, 0.00002106] |
| 6 | 0.01463925 [0.01366895, 0.01554615] | 0.00001514 [0.00001198, 0.00001848] |


| 差值 | h | position 差值 [95% CI] | feature 差值 [95% CI] |
| --- | --- | --- | --- |
| B_minus_b | 3 | 0.00458713 [0.00238812, 0.00654639] | — |
| d_minus_B | 3 | 0.00567732 [0.00393472, 0.00766439] | — |
| d_minus_b | 3 | 0.01026446 [0.00815170, 0.01228292] | 0.05694489 [0.04662360, 0.06637407] |
| B_minus_b_random | 3 | 0.00045079 [-0.00198727, 0.00263627] | — |
| d_minus_b_random | 3 | 0.00612811 [0.00336003, 0.00884544] | 0.03390257 [0.02345584, 0.04412217] |
| b_random_minus_b | 3 | 0.00413635 [0.00213096, 0.00627172] | 0.02304232 [0.01637106, 0.03062040] |
| B_minus_b | 6 | 0.00747230 [0.00522029, 0.00958898] | — |
| d_minus_B | 6 | 0.00547708 [0.00389038, 0.00725647] | — |
| d_minus_b | 6 | 0.01294938 [0.01085544, 0.01493287] | 0.05174909 [0.04133534, 0.06177860] |
| B_minus_b_random | 6 | 0.00241257 [-0.00023020, 0.00477663] | — |
| d_minus_b_random | 6 | 0.00788965 [0.00521333, 0.01047510] | 0.02352786 [0.01270964, 0.03340713] |
| b_random_minus_b | 6 | 0.00505973 [0.00280726, 0.00734330] | 0.02822123 [0.01927780, 0.03761481] |

### ordinary / predictor seed 29

| memory 条件 | h | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- |
| b | 3 | 0.00634484 [0.00497803, 0.00784923] | 0.18854763 [0.17801691, 0.19999264] |
| b | 6 | 0.00722309 [0.00590504, 0.00864964] | 0.19846950 [0.18779647, 0.20881405] |
| d | 3 | 0.01567384 [0.01408785, 0.01739059] | 0.24054037 [0.23074687, 0.24986089] |
| d | 6 | 0.01952815 [0.01813632, 0.02098952] | 0.24972556 [0.24010905, 0.25963867] |
| b_random | 3 | 0.00946840 [0.00750706, 0.01145054] | 0.20701172 [0.19519940, 0.21986897] |
| b_random | 6 | 0.01162685 [0.00936652, 0.01420104] | 0.21993140 [0.20907927, 0.23046795] |


| h | B [95% CI] | 真实 target feature 的 decoder position MSE [95% CI] |
| --- | --- | --- |
| 3 | 0.01050308 [0.00981344, 0.01118945] | 0.00001694 [0.00001284, 0.00002106] |
| 6 | 0.01463925 [0.01366895, 0.01554615] | 0.00001514 [0.00001198, 0.00001848] |


| 差值 | h | position 差值 [95% CI] | feature 差值 [95% CI] |
| --- | --- | --- | --- |
| B_minus_b | 3 | 0.00415824 [0.00242298, 0.00576972] | — |
| d_minus_B | 3 | 0.00517076 [0.00362946, 0.00688263] | — |
| d_minus_b | 3 | 0.00932900 [0.00773039, 0.01090937] | 0.05199274 [0.04244048, 0.06097369] |
| B_minus_b_random | 3 | 0.00103467 [-0.00116054, 0.00307025] | — |
| d_minus_b_random | 3 | 0.00620543 [0.00408554, 0.00824484] | 0.03352864 [0.02385071, 0.04287280] |
| b_random_minus_b | 3 | 0.00312357 [0.00141095, 0.00492735] | 0.01846409 [0.01222362, 0.02543818] |
| B_minus_b | 6 | 0.00741616 [0.00561447, 0.00912544] | — |
| d_minus_B | 6 | 0.00488890 [0.00354986, 0.00637537] | — |
| d_minus_b | 6 | 0.01230506 [0.01061207, 0.01399425] | 0.05125606 [0.04172052, 0.06031389] |
| B_minus_b_random | 6 | 0.00301241 [0.00042589, 0.00549531] | — |
| d_minus_b_random | 6 | 0.00790130 [0.00535403, 0.01030998] | 0.02979416 [0.02092110, 0.03934482] |
| b_random_minus_b | 6 | 0.00440376 [0.00234381, 0.00674106] | 0.02146190 [0.01361124, 0.02934528] |

### ordinary / predictor seed 43

| memory 条件 | h | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- |
| b | 3 | 0.00589497 [0.00460376, 0.00734997] | 0.18734906 [0.17573201, 0.19992134] |
| b | 6 | 0.00618953 [0.00488927, 0.00772009] | 0.20096324 [0.19006479, 0.21229959] |
| d | 3 | 0.01544338 [0.01403069, 0.01688838] | 0.23923241 [0.22950141, 0.24922894] |
| d | 6 | 0.01898153 [0.01770701, 0.02030465] | 0.24945660 [0.23959332, 0.25977954] |
| b_random | 3 | 0.00825126 [0.00663756, 0.00987959] | 0.20671972 [0.19400183, 0.21892323] |
| b_random | 6 | 0.00995874 [0.00830334, 0.01168132] | 0.22167401 [0.21045510, 0.23320697] |


| h | B [95% CI] | 真实 target feature 的 decoder position MSE [95% CI] |
| --- | --- | --- |
| 3 | 0.01050308 [0.00981344, 0.01118945] | 0.00001694 [0.00001284, 0.00002106] |
| 6 | 0.01463925 [0.01366895, 0.01554615] | 0.00001514 [0.00001198, 0.00001848] |


| 差值 | h | position 差值 [95% CI] | feature 差值 [95% CI] |
| --- | --- | --- | --- |
| B_minus_b | 3 | 0.00460811 [0.00290814, 0.00611665] | — |
| d_minus_B | 3 | 0.00494030 [0.00361425, 0.00639501] | — |
| d_minus_b | 3 | 0.00954841 [0.00782047, 0.01114386] | 0.05188335 [0.04197782, 0.06152637] |
| B_minus_b_random | 3 | 0.00225182 [0.00056262, 0.00394953] | — |
| d_minus_b_random | 3 | 0.00719212 [0.00538476, 0.00911101] | 0.03251269 [0.02263737, 0.04299865] |
| b_random_minus_b | 3 | 0.00235629 [0.00107418, 0.00357555] | 0.01937066 [0.01308576, 0.02538336] |
| B_minus_b | 6 | 0.00844972 [0.00671156, 0.01003886] | — |
| d_minus_B | 6 | 0.00434228 [0.00323808, 0.00555039] | — |
| d_minus_b | 6 | 0.01279200 [0.01108067, 0.01440176] | 0.04849336 [0.03974471, 0.05686945] |
| B_minus_b_random | 6 | 0.00468051 [0.00290254, 0.00646825] | — |
| d_minus_b_random | 6 | 0.00902279 [0.00714448, 0.01101194] | 0.02778259 [0.01931049, 0.03670862] |
| b_random_minus_b | 6 | 0.00376921 [0.00230364, 0.00526264] | 0.02071077 [0.01393113, 0.02710407] |

### long_gap / predictor seed 17

| memory 条件 | h | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- |
| a | 3 | 0.04113557 [0.03768903, 0.04491759] | 0.25665380 [0.24799371, 0.26583237] |
| a | 6 | 0.04312104 [0.03919909, 0.04743824] | 0.26740761 [0.25685713, 0.27962615] |
| b | 3 | 0.01676096 [0.01419114, 0.01942363] | 0.33275124 [0.31217553, 0.35355125] |
| b | 6 | 0.02096967 [0.01785081, 0.02421018] | 0.37102139 [0.34699680, 0.39741240] |
| d | 3 | 0.04050974 [0.03699695, 0.04429209] | 0.28408278 [0.27180030, 0.29706819] |
| d | 6 | 0.04467108 [0.04075625, 0.04879151] | 0.30118219 [0.28638244, 0.31743717] |


| h | B [95% CI] | 真实 target feature 的 decoder position MSE [95% CI] |
| --- | --- | --- |
| 3 | 0.03706187 [0.03360866, 0.04088258] | 0.00001950 [0.00001600, 0.00002305] |
| 6 | 0.04016588 [0.03629059, 0.04448014] | 0.00001848 [0.00001497, 0.00002204] |


| 差值 | h | position 差值 [95% CI] | feature 差值 [95% CI] |
| --- | --- | --- | --- |
| B_minus_b | 3 | 0.02030090 [0.01601045, 0.02467136] | — |
| d_minus_B | 3 | 0.00344788 [0.00264204, 0.00436600] | — |
| d_minus_b | 3 | 0.02374878 [0.01946634, 0.02794417] | -0.04866846 [-0.06600355, -0.03100818] |
| B_minus_b | 6 | 0.01919621 [0.01404762, 0.02459610] | — |
| d_minus_B | 6 | 0.00450520 [0.00343054, 0.00570405] | — |
| d_minus_b | 6 | 0.02370141 [0.01878174, 0.02878560] | -0.06983920 [-0.08815625, -0.05180508] |

### long_gap / predictor seed 29

| memory 条件 | h | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- |
| a | 3 | 0.04108121 [0.03773796, 0.04479754] | 0.25678377 [0.24858847, 0.26592418] |
| a | 6 | 0.04309154 [0.03923524, 0.04740795] | 0.26936965 [0.25898125, 0.28172379] |
| b | 3 | 0.01670842 [0.01347340, 0.02011625] | 0.33578532 [0.31237585, 0.36088739] |
| b | 6 | 0.02076926 [0.01704644, 0.02480483] | 0.37718782 [0.34724628, 0.41187926] |
| d | 3 | 0.04138127 [0.03765253, 0.04518445] | 0.28270800 [0.27190992, 0.29468567] |
| d | 6 | 0.04586911 [0.04176778, 0.05007828] | 0.30139425 [0.28858778, 0.31593353] |


| h | B [95% CI] | 真实 target feature 的 decoder position MSE [95% CI] |
| --- | --- | --- |
| 3 | 0.03706187 [0.03360866, 0.04088258] | 0.00001950 [0.00001600, 0.00002305] |
| 6 | 0.04016588 [0.03629059, 0.04448014] | 0.00001848 [0.00001497, 0.00002204] |


| 差值 | h | position 差值 [95% CI] | feature 差值 [95% CI] |
| --- | --- | --- | --- |
| B_minus_b | 3 | 0.02035344 [0.01578069, 0.02485390] | — |
| d_minus_B | 3 | 0.00431940 [0.00332387, 0.00545539] | — |
| d_minus_b | 3 | 0.02467285 [0.02021728, 0.02895581] | -0.05307732 [-0.07512615, -0.03402682] |
| B_minus_b | 6 | 0.01939662 [0.01377166, 0.02486132] | — |
| d_minus_B | 6 | 0.00570324 [0.00432905, 0.00723548] | — |
| d_minus_b | 6 | 0.02509986 [0.01973435, 0.03026310] | -0.07579357 [-0.10432216, -0.05207072] |

### long_gap / predictor seed 43

| memory 条件 | h | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- |
| a | 3 | 0.04201579 [0.03861955, 0.04579612] | 0.25812883 [0.25014091, 0.26691357] |
| a | 6 | 0.04373812 [0.03981219, 0.04792932] | 0.26788054 [0.25750908, 0.27981484] |
| b | 3 | 0.01944779 [0.01593560, 0.02315747] | 0.33694701 [0.31655530, 0.35724174] |
| b | 6 | 0.02370465 [0.01945598, 0.02777444] | 0.36917409 [0.34612156, 0.39369156] |
| d | 3 | 0.04030731 [0.03660824, 0.04431297] | 0.28491815 [0.27429824, 0.29646416] |
| d | 6 | 0.04449323 [0.04044220, 0.04875099] | 0.30030152 [0.28742114, 0.31451903] |


| h | B [95% CI] | 真实 target feature 的 decoder position MSE [95% CI] |
| --- | --- | --- |
| 3 | 0.03706187 [0.03360866, 0.04088258] | 0.00001950 [0.00001600, 0.00002305] |
| 6 | 0.04016588 [0.03629059, 0.04448014] | 0.00001848 [0.00001497, 0.00002204] |


| 差值 | h | position 差值 [95% CI] | feature 差值 [95% CI] |
| --- | --- | --- | --- |
| B_minus_b | 3 | 0.01761407 [0.01257758, 0.02262534] | — |
| d_minus_B | 3 | 0.00324545 [0.00233791, 0.00427917] | — |
| d_minus_b | 3 | 0.02085952 [0.01599245, 0.02587230] | -0.05202886 [-0.06842403, -0.03556500] |
| B_minus_b | 6 | 0.01646122 [0.01020698, 0.02257400] | — |
| d_minus_B | 6 | 0.00432736 [0.00309208, 0.00567496] | — |
| d_minus_b | 6 | 0.02078858 [0.01498376, 0.02648635] | -0.06887256 [-0.08665053, -0.05078968] |

## 4. 学生/RNG逐seed在线表与全部配对比较

每张表已平均三个冻结predictor。确定性baseline与参照的重复seed行只用于对齐，不是新独立重复。
### ordinary / student或RNG seed 17

| 方法 | h | 成功率 % [95% CI] | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- | --- |
| fifo | 3 | 0.000 [0.000, 0.000] | 0.02379281 [0.02101828, 0.02673472] | 0.24782456 [0.23775571, 0.25816802] |
| fifo | 6 | 0.000 [0.000, 0.000] | 0.02735527 [0.02493662, 0.02992922] | 0.26252034 [0.25342580, 0.27179227] |
| reservoir | 3 | 5.208 [1.042, 9.375] | 0.02032435 [0.01807598, 0.02263637] | 0.24104878 [0.23147176, 0.25056813] |
| reservoir | 6 | 5.208 [1.042, 9.375] | 0.02371332 [0.02166662, 0.02585010] | 0.25573446 [0.24626460, 0.26503680] |
| temporal_buckets | 3 | 0.000 [0.000, 0.000] | 0.02359853 [0.02106712, 0.02636866] | 0.24461806 [0.23475158, 0.25471799] |
| temporal_buckets | 6 | 0.000 [0.000, 0.000] | 0.02706629 [0.02487122, 0.02942262] | 0.25852891 [0.24939653, 0.26793686] |
| diversity | 3 | 9.375 [4.688, 14.583] | 0.01666226 [0.01505911, 0.01827971] | 0.23339935 [0.22340898, 0.24354676] |
| diversity | 6 | 9.375 [4.688, 14.583] | 0.02059435 [0.01904824, 0.02216771] | 0.24624977 [0.23605378, 0.25614476] |
| surprise | 3 | 0.000 [0.000, 0.000] | 0.02483195 [0.02197593, 0.02797348] | 0.25388996 [0.24325622, 0.26511745] |
| surprise | 6 | 0.000 [0.000, 0.000] | 0.02877268 [0.02608214, 0.03166681] | 0.27248656 [0.26282511, 0.28274330] |
| attention | 3 | 100.000 [100.000, 100.000] | 0.00914680 [0.00749575, 0.01088691] | 0.20492259 [0.19395310, 0.21653455] |
| attention | 6 | 100.000 [100.000, 100.000] | 0.01114779 [0.00928566, 0.01308783] | 0.22151373 [0.21047667, 0.23266554] |
| keep_first | 3 | 0.000 [0.000, 0.000] | 0.02664115 [0.02370149, 0.02978971] | 0.26633239 [0.25533553, 0.27805961] |
| keep_first | 6 | 0.000 [0.000, 0.000] | 0.03022299 [0.02766103, 0.03294746] | 0.28153123 [0.27165338, 0.29235949] |
| independent/local_feature **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/local_feature **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| independent/local_position **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/local_position **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| independent/use_time_feature **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/use_time_feature **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| independent/use_time_position **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/use_time_position **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| independent/future_attention | 3 | 97.917 [95.312, 100.000] | 0.00938604 [0.00773124, 0.01120229] | 0.20581470 [0.19490560, 0.21711893] |
| independent/future_attention | 6 | 97.917 [95.312, 100.000] | 0.01148812 [0.00972320, 0.01339060] | 0.22287564 [0.21301600, 0.23320103] |
| independent/shuffle_local_feature **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/shuffle_local_feature **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| independent/shuffle_local_position **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/shuffle_local_position **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| independent/shuffle_use_time_feature **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/shuffle_use_time_feature **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| independent/shuffle_use_time_position **INIT** | 3 | 4.688 [1.562, 8.333] | 0.02208223 [0.01968411, 0.02467463] | 0.24440094 [0.23458297, 0.25462181] |
| independent/shuffle_use_time_position **INIT** | 6 | 4.688 [1.562, 8.333] | 0.02544834 [0.02318741, 0.02770580] | 0.25716528 [0.24763115, 0.26711607] |
| set/local_feature **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/local_feature **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| set/local_position **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/local_position **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| set/use_time_feature **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/use_time_feature **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| set/use_time_position **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/use_time_position **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| set/future_attention | 3 | 100.000 [100.000, 100.000] | 0.01002280 [0.00820500, 0.01212692] | 0.20720932 [0.19640896, 0.21855370] |
| set/future_attention | 6 | 100.000 [100.000, 100.000] | 0.01192450 [0.01011141, 0.01394966] | 0.22573774 [0.21541526, 0.23654682] |
| set/shuffle_local_feature **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/shuffle_local_feature **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| set/shuffle_local_position **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/shuffle_local_position **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| set/shuffle_use_time_feature **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/shuffle_use_time_feature **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| set/shuffle_use_time_position **INIT** | 3 | 6.250 [2.604, 10.417] | 0.02161000 [0.01899305, 0.02456596] | 0.24540792 [0.23555544, 0.25565501] |
| set/shuffle_use_time_position **INIT** | 6 | 6.250 [2.604, 10.417] | 0.02572426 [0.02332032, 0.02839397] | 0.25951266 [0.24967504, 0.26973918] |
| 参照/last_visible | 3 | 100.000 [100.000, 100.000] | 0.01112028 [0.00942257, 0.01314966] | 0.22187655 [0.21128329, 0.23302433] |
| 参照/last_visible | 6 | 100.000 [100.000, 100.000] | 0.01336026 [0.01162503, 0.01532875] | 0.23312347 [0.22200088, 0.24453439] |
| 参照/b | 3 | 100.000 [100.000, 100.000] | 0.00605192 [0.00476684, 0.00758422] | 0.18681108 [0.17628059, 0.19850645] |
| 参照/b | 6 | 100.000 [100.000, 100.000] | 0.00685986 [0.00563873, 0.00820422] | 0.19888737 [0.18907097, 0.20860782] |
| 参照/posthoc_best | 3 | 52.604 [46.875, 58.333] | 0.00253535 [0.00188422, 0.00331072] | 0.19745953 [0.18882705, 0.20619257] |
| 参照/posthoc_best | 6 | 52.604 [46.875, 58.333] | 0.00313890 [0.00242108, 0.00398104] | 0.21405261 [0.20471316, 0.22317230] |

四标签对全部对手：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 6.250 [2.604, 10.417] | 0.00218281 [0.00042294, 0.00398624] | 0.00241664 [-0.00274024, 0.00763349] | 是 |
| local_feature | fifo | 6 | 6.250 [2.604, 10.417] | 0.00163101 [-0.00032747, 0.00364351] | 0.00300768 [-0.00249074, 0.00830442] | 否 |
| local_feature | reservoir | 3 | 1.042 [-5.208, 7.305] | -0.00128566 [-0.00364378, 0.00088376] | -0.00435914 [-0.01115675, 0.00231936] | 否 |
| local_feature | reservoir | 6 | 1.042 [-5.208, 7.305] | -0.00201094 [-0.00448167, 0.00024784] | -0.00377821 [-0.01048967, 0.00300161] | 否 |
| local_feature | temporal_buckets | 3 | 6.250 [2.604, 10.417] | 0.00198852 [0.00036777, 0.00363554] | -0.00078986 [-0.00575386, 0.00429049] | 是 |
| local_feature | temporal_buckets | 6 | 6.250 [2.604, 10.417] | 0.00134203 [-0.00049119, 0.00318038] | -0.00098376 [-0.00621445, 0.00395879] | 否 |
| local_feature | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00494775 [-0.00732321, -0.00277312] | -0.01200857 [-0.01895408, -0.00486269] | 否 |
| local_feature | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00512991 [-0.00766154, -0.00278490] | -0.01326290 [-0.02168183, -0.00508389] | 否 |
| local_feature | surprise | 3 | 6.250 [2.604, 10.417] | 0.00322194 [0.00120766, 0.00545688] | 0.00848204 [0.00256230, 0.01519115] | 是 |
| local_feature | surprise | 6 | 6.250 [2.604, 10.417] | 0.00304842 [0.00073247, 0.00551890] | 0.01297389 [0.00678709, 0.01960449] | 是 |
| local_feature | attention | 3 | -93.750 [-97.396, -89.583] | -0.01246321 [-0.01545801, -0.00983480] | -0.04048533 [-0.04948178, -0.03139584] | 否 |
| local_feature | attention | 6 | -93.750 [-97.396, -89.583] | -0.01457647 [-0.01750888, -0.01196182] | -0.03799893 [-0.04775555, -0.02781426] | 否 |
| local_feature | keep_first | 3 | 6.250 [2.604, 10.417] | 0.00503114 [0.00262436, 0.00732060] | 0.02092447 [0.01381912, 0.02845838] | 是 |
| local_feature | keep_first | 6 | 6.250 [2.604, 10.417] | 0.00449873 [0.00197373, 0.00705119] | 0.02201856 [0.01529720, 0.02862870] | 是 |
| local_feature | set/future_attention | 3 | -93.750 [-97.396, -89.583] | -0.01158721 [-0.01423195, -0.00906207] | -0.03819860 [-0.04764369, -0.02873768] | 否 |
| local_feature | set/future_attention | 6 | -93.750 [-97.396, -89.583] | -0.01379976 [-0.01659548, -0.01113028] | -0.03377492 [-0.04293457, -0.02424482] | 否 |
| local_position | fifo | 3 | 6.250 [2.604, 10.417] | 0.00218281 [0.00042294, 0.00398624] | 0.00241664 [-0.00274024, 0.00763349] | 是 |
| local_position | fifo | 6 | 6.250 [2.604, 10.417] | 0.00163101 [-0.00032747, 0.00364351] | 0.00300768 [-0.00249074, 0.00830442] | 否 |
| local_position | reservoir | 3 | 1.042 [-5.208, 7.305] | -0.00128566 [-0.00364378, 0.00088376] | -0.00435914 [-0.01115675, 0.00231936] | 否 |
| local_position | reservoir | 6 | 1.042 [-5.208, 7.305] | -0.00201094 [-0.00448167, 0.00024784] | -0.00377821 [-0.01048967, 0.00300161] | 否 |
| local_position | temporal_buckets | 3 | 6.250 [2.604, 10.417] | 0.00198852 [0.00036777, 0.00363554] | -0.00078986 [-0.00575386, 0.00429049] | 是 |
| local_position | temporal_buckets | 6 | 6.250 [2.604, 10.417] | 0.00134203 [-0.00049119, 0.00318038] | -0.00098376 [-0.00621445, 0.00395879] | 否 |
| local_position | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00494775 [-0.00732321, -0.00277312] | -0.01200857 [-0.01895408, -0.00486269] | 否 |
| local_position | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00512991 [-0.00766154, -0.00278490] | -0.01326290 [-0.02168183, -0.00508389] | 否 |
| local_position | surprise | 3 | 6.250 [2.604, 10.417] | 0.00322194 [0.00120766, 0.00545688] | 0.00848204 [0.00256230, 0.01519115] | 是 |
| local_position | surprise | 6 | 6.250 [2.604, 10.417] | 0.00304842 [0.00073247, 0.00551890] | 0.01297389 [0.00678709, 0.01960449] | 是 |
| local_position | attention | 3 | -93.750 [-97.396, -89.583] | -0.01246321 [-0.01545801, -0.00983480] | -0.04048533 [-0.04948178, -0.03139584] | 否 |
| local_position | attention | 6 | -93.750 [-97.396, -89.583] | -0.01457647 [-0.01750888, -0.01196182] | -0.03799893 [-0.04775555, -0.02781426] | 否 |
| local_position | keep_first | 3 | 6.250 [2.604, 10.417] | 0.00503114 [0.00262436, 0.00732060] | 0.02092447 [0.01381912, 0.02845838] | 是 |
| local_position | keep_first | 6 | 6.250 [2.604, 10.417] | 0.00449873 [0.00197373, 0.00705119] | 0.02201856 [0.01529720, 0.02862870] | 是 |
| local_position | set/future_attention | 3 | -93.750 [-97.396, -89.583] | -0.01158721 [-0.01423195, -0.00906207] | -0.03819860 [-0.04764369, -0.02873768] | 否 |
| local_position | set/future_attention | 6 | -93.750 [-97.396, -89.583] | -0.01379976 [-0.01659548, -0.01113028] | -0.03377492 [-0.04293457, -0.02424482] | 否 |
| use_time_feature | fifo | 3 | 6.250 [2.604, 10.417] | 0.00218281 [0.00042294, 0.00398624] | 0.00241664 [-0.00274024, 0.00763349] | 是 |
| use_time_feature | fifo | 6 | 6.250 [2.604, 10.417] | 0.00163101 [-0.00032747, 0.00364351] | 0.00300768 [-0.00249074, 0.00830442] | 否 |
| use_time_feature | reservoir | 3 | 1.042 [-5.208, 7.305] | -0.00128566 [-0.00364378, 0.00088376] | -0.00435914 [-0.01115675, 0.00231936] | 否 |
| use_time_feature | reservoir | 6 | 1.042 [-5.208, 7.305] | -0.00201094 [-0.00448167, 0.00024784] | -0.00377821 [-0.01048967, 0.00300161] | 否 |
| use_time_feature | temporal_buckets | 3 | 6.250 [2.604, 10.417] | 0.00198852 [0.00036777, 0.00363554] | -0.00078986 [-0.00575386, 0.00429049] | 是 |
| use_time_feature | temporal_buckets | 6 | 6.250 [2.604, 10.417] | 0.00134203 [-0.00049119, 0.00318038] | -0.00098376 [-0.00621445, 0.00395879] | 否 |
| use_time_feature | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00494775 [-0.00732321, -0.00277312] | -0.01200857 [-0.01895408, -0.00486269] | 否 |
| use_time_feature | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00512991 [-0.00766154, -0.00278490] | -0.01326290 [-0.02168183, -0.00508389] | 否 |
| use_time_feature | surprise | 3 | 6.250 [2.604, 10.417] | 0.00322194 [0.00120766, 0.00545688] | 0.00848204 [0.00256230, 0.01519115] | 是 |
| use_time_feature | surprise | 6 | 6.250 [2.604, 10.417] | 0.00304842 [0.00073247, 0.00551890] | 0.01297389 [0.00678709, 0.01960449] | 是 |
| use_time_feature | attention | 3 | -93.750 [-97.396, -89.583] | -0.01246321 [-0.01545801, -0.00983480] | -0.04048533 [-0.04948178, -0.03139584] | 否 |
| use_time_feature | attention | 6 | -93.750 [-97.396, -89.583] | -0.01457647 [-0.01750888, -0.01196182] | -0.03799893 [-0.04775555, -0.02781426] | 否 |
| use_time_feature | keep_first | 3 | 6.250 [2.604, 10.417] | 0.00503114 [0.00262436, 0.00732060] | 0.02092447 [0.01381912, 0.02845838] | 是 |
| use_time_feature | keep_first | 6 | 6.250 [2.604, 10.417] | 0.00449873 [0.00197373, 0.00705119] | 0.02201856 [0.01529720, 0.02862870] | 是 |
| use_time_feature | set/future_attention | 3 | -93.750 [-97.396, -89.583] | -0.01158721 [-0.01423195, -0.00906207] | -0.03819860 [-0.04764369, -0.02873768] | 否 |
| use_time_feature | set/future_attention | 6 | -93.750 [-97.396, -89.583] | -0.01379976 [-0.01659548, -0.01113028] | -0.03377492 [-0.04293457, -0.02424482] | 否 |
| use_time_position | fifo | 3 | 6.250 [2.604, 10.417] | 0.00218281 [0.00042294, 0.00398624] | 0.00241664 [-0.00274024, 0.00763349] | 是 |
| use_time_position | fifo | 6 | 6.250 [2.604, 10.417] | 0.00163101 [-0.00032747, 0.00364351] | 0.00300768 [-0.00249074, 0.00830442] | 否 |
| use_time_position | reservoir | 3 | 1.042 [-5.208, 7.305] | -0.00128566 [-0.00364378, 0.00088376] | -0.00435914 [-0.01115675, 0.00231936] | 否 |
| use_time_position | reservoir | 6 | 1.042 [-5.208, 7.305] | -0.00201094 [-0.00448167, 0.00024784] | -0.00377821 [-0.01048967, 0.00300161] | 否 |
| use_time_position | temporal_buckets | 3 | 6.250 [2.604, 10.417] | 0.00198852 [0.00036777, 0.00363554] | -0.00078986 [-0.00575386, 0.00429049] | 是 |
| use_time_position | temporal_buckets | 6 | 6.250 [2.604, 10.417] | 0.00134203 [-0.00049119, 0.00318038] | -0.00098376 [-0.00621445, 0.00395879] | 否 |
| use_time_position | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00494775 [-0.00732321, -0.00277312] | -0.01200857 [-0.01895408, -0.00486269] | 否 |
| use_time_position | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00512991 [-0.00766154, -0.00278490] | -0.01326290 [-0.02168183, -0.00508389] | 否 |
| use_time_position | surprise | 3 | 6.250 [2.604, 10.417] | 0.00322194 [0.00120766, 0.00545688] | 0.00848204 [0.00256230, 0.01519115] | 是 |
| use_time_position | surprise | 6 | 6.250 [2.604, 10.417] | 0.00304842 [0.00073247, 0.00551890] | 0.01297389 [0.00678709, 0.01960449] | 是 |
| use_time_position | attention | 3 | -93.750 [-97.396, -89.583] | -0.01246321 [-0.01545801, -0.00983480] | -0.04048533 [-0.04948178, -0.03139584] | 否 |
| use_time_position | attention | 6 | -93.750 [-97.396, -89.583] | -0.01457647 [-0.01750888, -0.01196182] | -0.03799893 [-0.04775555, -0.02781426] | 否 |
| use_time_position | keep_first | 3 | 6.250 [2.604, 10.417] | 0.00503114 [0.00262436, 0.00732060] | 0.02092447 [0.01381912, 0.02845838] | 是 |
| use_time_position | keep_first | 6 | 6.250 [2.604, 10.417] | 0.00449873 [0.00197373, 0.00705119] | 0.02201856 [0.01529720, 0.02862870] | 是 |
| use_time_position | set/future_attention | 3 | -93.750 [-97.396, -89.583] | -0.01158721 [-0.01423195, -0.00906207] | -0.03819860 [-0.04764369, -0.02873768] | 否 |
| use_time_position | set/future_attention | 6 | -93.750 [-97.396, -89.583] | -0.01379976 [-0.01659548, -0.01113028] | -0.03377492 [-0.04293457, -0.02424482] | 否 |

同标签集合/独立：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | 1.562 [-4.167, 6.771] | 0.00047222 [-0.00153470, 0.00238518] | -0.00100698 [-0.00703244, 0.00467655] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | 1.562 [-4.167, 6.771] | -0.00027592 [-0.00260923, 0.00201637] | -0.00234738 [-0.00861844, 0.00367080] | 否 |
| local_position | independent/local_position **INIT** | 3 | 1.562 [-4.167, 6.771] | 0.00047222 [-0.00153470, 0.00238518] | -0.00100698 [-0.00703244, 0.00467655] | 否 |
| local_position | independent/local_position **INIT** | 6 | 1.562 [-4.167, 6.771] | -0.00027592 [-0.00260923, 0.00201637] | -0.00234738 [-0.00861844, 0.00367080] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | 1.562 [-4.167, 6.771] | 0.00047222 [-0.00153470, 0.00238518] | -0.00100698 [-0.00703244, 0.00467655] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | 1.562 [-4.167, 6.771] | -0.00027592 [-0.00260923, 0.00201637] | -0.00234738 [-0.00861844, 0.00367080] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | 1.562 [-4.167, 6.771] | 0.00047222 [-0.00153470, 0.00238518] | -0.00100698 [-0.00703244, 0.00467655] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | 1.562 [-4.167, 6.771] | -0.00027592 [-0.00260923, 0.00201637] | -0.00234738 [-0.00861844, 0.00367080] | 否 |

### ordinary / student或RNG seed 29

| 方法 | h | 成功率 % [95% CI] | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- | --- |
| fifo | 3 | 0.000 [0.000, 0.000] | 0.02379281 [0.02101828, 0.02673472] | 0.24782456 [0.23775571, 0.25816802] |
| fifo | 6 | 0.000 [0.000, 0.000] | 0.02735527 [0.02493662, 0.02992922] | 0.26252034 [0.25342580, 0.27179227] |
| reservoir | 3 | 5.208 [1.042, 10.417] | 0.02091246 [0.01848687, 0.02358940] | 0.24130952 [0.23039203, 0.25360424] |
| reservoir | 6 | 5.208 [1.042, 10.417] | 0.02443666 [0.02207989, 0.02693941] | 0.25750064 [0.24673235, 0.26814616] |
| temporal_buckets | 3 | 0.000 [0.000, 0.000] | 0.02359853 [0.02106712, 0.02636866] | 0.24461806 [0.23475158, 0.25471799] |
| temporal_buckets | 6 | 0.000 [0.000, 0.000] | 0.02706629 [0.02487122, 0.02942262] | 0.25852891 [0.24939653, 0.26793686] |
| diversity | 3 | 9.375 [4.688, 14.583] | 0.01666226 [0.01505911, 0.01827971] | 0.23339935 [0.22340898, 0.24354676] |
| diversity | 6 | 9.375 [4.688, 14.583] | 0.02059435 [0.01904824, 0.02216771] | 0.24624977 [0.23605378, 0.25614476] |
| surprise | 3 | 0.000 [0.000, 0.000] | 0.02483195 [0.02197593, 0.02797348] | 0.25388996 [0.24325622, 0.26511745] |
| surprise | 6 | 0.000 [0.000, 0.000] | 0.02877268 [0.02608214, 0.03166681] | 0.27248656 [0.26282511, 0.28274330] |
| attention | 3 | 100.000 [100.000, 100.000] | 0.00914680 [0.00749575, 0.01088691] | 0.20492259 [0.19395310, 0.21653455] |
| attention | 6 | 100.000 [100.000, 100.000] | 0.01114779 [0.00928566, 0.01308783] | 0.22151373 [0.21047667, 0.23266554] |
| keep_first | 3 | 0.000 [0.000, 0.000] | 0.02664115 [0.02370149, 0.02978971] | 0.26633239 [0.25533553, 0.27805961] |
| keep_first | 6 | 0.000 [0.000, 0.000] | 0.03022299 [0.02766103, 0.03294746] | 0.28153123 [0.27165338, 0.29235949] |
| independent/local_feature **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/local_feature **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| independent/local_position **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/local_position **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| independent/use_time_feature **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/use_time_feature **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| independent/use_time_position **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/use_time_position **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| independent/future_attention | 3 | 98.438 [96.354, 100.000] | 0.00985681 [0.00814178, 0.01170614] | 0.21029857 [0.19857468, 0.22231332] |
| independent/future_attention | 6 | 98.438 [96.354, 100.000] | 0.01219645 [0.01040463, 0.01413610] | 0.22905263 [0.21788413, 0.24048884] |
| independent/shuffle_local_feature **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/shuffle_local_feature **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| independent/shuffle_local_position **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/shuffle_local_position **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| independent/shuffle_use_time_feature **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/shuffle_use_time_feature **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| independent/shuffle_use_time_position **INIT** | 3 | 6.771 [3.112, 11.458] | 0.02146730 [0.01884841, 0.02430635] | 0.24180350 [0.23223747, 0.25168230] |
| independent/shuffle_use_time_position **INIT** | 6 | 6.771 [3.112, 11.458] | 0.02517019 [0.02258832, 0.02778702] | 0.25581867 [0.24565172, 0.26632813] |
| set/local_feature **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/local_feature **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| set/local_position **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/local_position **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| set/use_time_feature **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/use_time_feature **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| set/use_time_position **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/use_time_position **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| set/future_attention | 3 | 100.000 [100.000, 100.000] | 0.01017767 [0.00835015, 0.01220078] | 0.20621256 [0.19555271, 0.21687137] |
| set/future_attention | 6 | 100.000 [100.000, 100.000] | 0.01213362 [0.01027045, 0.01417181] | 0.22456966 [0.21415112, 0.23494226] |
| set/shuffle_local_feature **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/shuffle_local_feature **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| set/shuffle_local_position **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/shuffle_local_position **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| set/shuffle_use_time_feature **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/shuffle_use_time_feature **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| set/shuffle_use_time_position **INIT** | 3 | 5.208 [1.562, 9.896] | 0.02096171 [0.01883229, 0.02309204] | 0.24357688 [0.23382014, 0.25338247] |
| set/shuffle_use_time_position **INIT** | 6 | 5.208 [1.562, 9.896] | 0.02475724 [0.02273933, 0.02666088] | 0.26216309 [0.25237045, 0.27227952] |
| 参照/last_visible | 3 | 100.000 [100.000, 100.000] | 0.01112028 [0.00942257, 0.01314966] | 0.22187655 [0.21128329, 0.23302433] |
| 参照/last_visible | 6 | 100.000 [100.000, 100.000] | 0.01336026 [0.01162503, 0.01532875] | 0.23312347 [0.22200088, 0.24453439] |
| 参照/b | 3 | 100.000 [100.000, 100.000] | 0.00605192 [0.00476684, 0.00758422] | 0.18681108 [0.17628059, 0.19850645] |
| 参照/b | 6 | 100.000 [100.000, 100.000] | 0.00685986 [0.00563873, 0.00820422] | 0.19888737 [0.18907097, 0.20860782] |
| 参照/posthoc_best | 3 | 52.604 [46.875, 58.333] | 0.00253535 [0.00188422, 0.00331072] | 0.19745953 [0.18882705, 0.20619257] |
| 参照/posthoc_best | 6 | 52.604 [46.875, 58.333] | 0.00313890 [0.00242108, 0.00398104] | 0.21405261 [0.20471316, 0.22317230] |

四标签对全部对手：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 5.208 [1.562, 9.896] | 0.00283111 [0.00063758, 0.00528572] | 0.00424768 [-0.00177718, 0.01107815] | 是 |
| local_feature | fifo | 6 | 5.208 [1.562, 9.896] | 0.00259803 [0.00042896, 0.00493589] | 0.00035725 [-0.00589537, 0.00683096] | 是 |
| local_feature | reservoir | 3 | 0.000 [-6.771, 5.729] | -0.00004925 [-0.00277371, 0.00296124] | -0.00226736 [-0.01124524, 0.00809049] | 否 |
| local_feature | reservoir | 6 | 0.000 [-6.771, 5.729] | -0.00032058 [-0.00299253, 0.00259246] | -0.00466245 [-0.01245699, 0.00397151] | 否 |
| local_feature | temporal_buckets | 3 | 5.208 [1.562, 9.896] | 0.00263682 [0.00055842, 0.00506707] | 0.00104118 [-0.00474650, 0.00815444] | 是 |
| local_feature | temporal_buckets | 6 | 5.208 [1.562, 9.896] | 0.00230906 [0.00032752, 0.00456572] | -0.00363418 [-0.00952287, 0.00252849] | 是 |
| local_feature | diversity | 3 | -4.167 [-10.417, 2.083] | -0.00429945 [-0.00636252, -0.00218293] | -0.01017753 [-0.01686045, -0.00249766] | 否 |
| local_feature | diversity | 6 | -4.167 [-10.417, 2.083] | -0.00416289 [-0.00622795, -0.00197591] | -0.01591332 [-0.02363519, -0.00836690] | 否 |
| local_feature | surprise | 3 | 5.208 [1.562, 9.896] | 0.00387024 [0.00125302, 0.00681376] | 0.01031308 [0.00279528, 0.01830068] | 是 |
| local_feature | surprise | 6 | 5.208 [1.562, 9.896] | 0.00401544 [0.00143968, 0.00693937] | 0.01032347 [0.00333451, 0.01759005] | 是 |
| local_feature | attention | 3 | -94.792 [-98.438, -90.104] | -0.01181491 [-0.01426660, -0.00947361] | -0.03865429 [-0.04770751, -0.02892738] | 否 |
| local_feature | attention | 6 | -94.792 [-98.438, -90.104] | -0.01360945 [-0.01590120, -0.01124246] | -0.04064935 [-0.05110220, -0.02975946] | 否 |
| local_feature | keep_first | 3 | 5.208 [1.562, 9.896] | 0.00567944 [0.00308337, 0.00859595] | 0.02275552 [0.01376408, 0.03231941] | 是 |
| local_feature | keep_first | 6 | 5.208 [1.562, 9.896] | 0.00546576 [0.00285519, 0.00831831] | 0.01936814 [0.01104770, 0.02834519] | 是 |
| local_feature | set/future_attention | 3 | -94.792 [-98.438, -90.104] | -0.01078404 [-0.01300572, -0.00834382] | -0.03736432 [-0.04689485, -0.02732050] | 否 |
| local_feature | set/future_attention | 6 | -94.792 [-98.438, -90.104] | -0.01262362 [-0.01501255, -0.00992872] | -0.03759343 [-0.04720038, -0.02794084] | 否 |
| local_position | fifo | 3 | 5.208 [1.562, 9.896] | 0.00283111 [0.00063758, 0.00528572] | 0.00424768 [-0.00177718, 0.01107815] | 是 |
| local_position | fifo | 6 | 5.208 [1.562, 9.896] | 0.00259803 [0.00042896, 0.00493589] | 0.00035725 [-0.00589537, 0.00683096] | 是 |
| local_position | reservoir | 3 | 0.000 [-6.771, 5.729] | -0.00004925 [-0.00277371, 0.00296124] | -0.00226736 [-0.01124524, 0.00809049] | 否 |
| local_position | reservoir | 6 | 0.000 [-6.771, 5.729] | -0.00032058 [-0.00299253, 0.00259246] | -0.00466245 [-0.01245699, 0.00397151] | 否 |
| local_position | temporal_buckets | 3 | 5.208 [1.562, 9.896] | 0.00263682 [0.00055842, 0.00506707] | 0.00104118 [-0.00474650, 0.00815444] | 是 |
| local_position | temporal_buckets | 6 | 5.208 [1.562, 9.896] | 0.00230906 [0.00032752, 0.00456572] | -0.00363418 [-0.00952287, 0.00252849] | 是 |
| local_position | diversity | 3 | -4.167 [-10.417, 2.083] | -0.00429945 [-0.00636252, -0.00218293] | -0.01017753 [-0.01686045, -0.00249766] | 否 |
| local_position | diversity | 6 | -4.167 [-10.417, 2.083] | -0.00416289 [-0.00622795, -0.00197591] | -0.01591332 [-0.02363519, -0.00836690] | 否 |
| local_position | surprise | 3 | 5.208 [1.562, 9.896] | 0.00387024 [0.00125302, 0.00681376] | 0.01031308 [0.00279528, 0.01830068] | 是 |
| local_position | surprise | 6 | 5.208 [1.562, 9.896] | 0.00401544 [0.00143968, 0.00693937] | 0.01032347 [0.00333451, 0.01759005] | 是 |
| local_position | attention | 3 | -94.792 [-98.438, -90.104] | -0.01181491 [-0.01426660, -0.00947361] | -0.03865429 [-0.04770751, -0.02892738] | 否 |
| local_position | attention | 6 | -94.792 [-98.438, -90.104] | -0.01360945 [-0.01590120, -0.01124246] | -0.04064935 [-0.05110220, -0.02975946] | 否 |
| local_position | keep_first | 3 | 5.208 [1.562, 9.896] | 0.00567944 [0.00308337, 0.00859595] | 0.02275552 [0.01376408, 0.03231941] | 是 |
| local_position | keep_first | 6 | 5.208 [1.562, 9.896] | 0.00546576 [0.00285519, 0.00831831] | 0.01936814 [0.01104770, 0.02834519] | 是 |
| local_position | set/future_attention | 3 | -94.792 [-98.438, -90.104] | -0.01078404 [-0.01300572, -0.00834382] | -0.03736432 [-0.04689485, -0.02732050] | 否 |
| local_position | set/future_attention | 6 | -94.792 [-98.438, -90.104] | -0.01262362 [-0.01501255, -0.00992872] | -0.03759343 [-0.04720038, -0.02794084] | 否 |
| use_time_feature | fifo | 3 | 5.208 [1.562, 9.896] | 0.00283111 [0.00063758, 0.00528572] | 0.00424768 [-0.00177718, 0.01107815] | 是 |
| use_time_feature | fifo | 6 | 5.208 [1.562, 9.896] | 0.00259803 [0.00042896, 0.00493589] | 0.00035725 [-0.00589537, 0.00683096] | 是 |
| use_time_feature | reservoir | 3 | 0.000 [-6.771, 5.729] | -0.00004925 [-0.00277371, 0.00296124] | -0.00226736 [-0.01124524, 0.00809049] | 否 |
| use_time_feature | reservoir | 6 | 0.000 [-6.771, 5.729] | -0.00032058 [-0.00299253, 0.00259246] | -0.00466245 [-0.01245699, 0.00397151] | 否 |
| use_time_feature | temporal_buckets | 3 | 5.208 [1.562, 9.896] | 0.00263682 [0.00055842, 0.00506707] | 0.00104118 [-0.00474650, 0.00815444] | 是 |
| use_time_feature | temporal_buckets | 6 | 5.208 [1.562, 9.896] | 0.00230906 [0.00032752, 0.00456572] | -0.00363418 [-0.00952287, 0.00252849] | 是 |
| use_time_feature | diversity | 3 | -4.167 [-10.417, 2.083] | -0.00429945 [-0.00636252, -0.00218293] | -0.01017753 [-0.01686045, -0.00249766] | 否 |
| use_time_feature | diversity | 6 | -4.167 [-10.417, 2.083] | -0.00416289 [-0.00622795, -0.00197591] | -0.01591332 [-0.02363519, -0.00836690] | 否 |
| use_time_feature | surprise | 3 | 5.208 [1.562, 9.896] | 0.00387024 [0.00125302, 0.00681376] | 0.01031308 [0.00279528, 0.01830068] | 是 |
| use_time_feature | surprise | 6 | 5.208 [1.562, 9.896] | 0.00401544 [0.00143968, 0.00693937] | 0.01032347 [0.00333451, 0.01759005] | 是 |
| use_time_feature | attention | 3 | -94.792 [-98.438, -90.104] | -0.01181491 [-0.01426660, -0.00947361] | -0.03865429 [-0.04770751, -0.02892738] | 否 |
| use_time_feature | attention | 6 | -94.792 [-98.438, -90.104] | -0.01360945 [-0.01590120, -0.01124246] | -0.04064935 [-0.05110220, -0.02975946] | 否 |
| use_time_feature | keep_first | 3 | 5.208 [1.562, 9.896] | 0.00567944 [0.00308337, 0.00859595] | 0.02275552 [0.01376408, 0.03231941] | 是 |
| use_time_feature | keep_first | 6 | 5.208 [1.562, 9.896] | 0.00546576 [0.00285519, 0.00831831] | 0.01936814 [0.01104770, 0.02834519] | 是 |
| use_time_feature | set/future_attention | 3 | -94.792 [-98.438, -90.104] | -0.01078404 [-0.01300572, -0.00834382] | -0.03736432 [-0.04689485, -0.02732050] | 否 |
| use_time_feature | set/future_attention | 6 | -94.792 [-98.438, -90.104] | -0.01262362 [-0.01501255, -0.00992872] | -0.03759343 [-0.04720038, -0.02794084] | 否 |
| use_time_position | fifo | 3 | 5.208 [1.562, 9.896] | 0.00283111 [0.00063758, 0.00528572] | 0.00424768 [-0.00177718, 0.01107815] | 是 |
| use_time_position | fifo | 6 | 5.208 [1.562, 9.896] | 0.00259803 [0.00042896, 0.00493589] | 0.00035725 [-0.00589537, 0.00683096] | 是 |
| use_time_position | reservoir | 3 | 0.000 [-6.771, 5.729] | -0.00004925 [-0.00277371, 0.00296124] | -0.00226736 [-0.01124524, 0.00809049] | 否 |
| use_time_position | reservoir | 6 | 0.000 [-6.771, 5.729] | -0.00032058 [-0.00299253, 0.00259246] | -0.00466245 [-0.01245699, 0.00397151] | 否 |
| use_time_position | temporal_buckets | 3 | 5.208 [1.562, 9.896] | 0.00263682 [0.00055842, 0.00506707] | 0.00104118 [-0.00474650, 0.00815444] | 是 |
| use_time_position | temporal_buckets | 6 | 5.208 [1.562, 9.896] | 0.00230906 [0.00032752, 0.00456572] | -0.00363418 [-0.00952287, 0.00252849] | 是 |
| use_time_position | diversity | 3 | -4.167 [-10.417, 2.083] | -0.00429945 [-0.00636252, -0.00218293] | -0.01017753 [-0.01686045, -0.00249766] | 否 |
| use_time_position | diversity | 6 | -4.167 [-10.417, 2.083] | -0.00416289 [-0.00622795, -0.00197591] | -0.01591332 [-0.02363519, -0.00836690] | 否 |
| use_time_position | surprise | 3 | 5.208 [1.562, 9.896] | 0.00387024 [0.00125302, 0.00681376] | 0.01031308 [0.00279528, 0.01830068] | 是 |
| use_time_position | surprise | 6 | 5.208 [1.562, 9.896] | 0.00401544 [0.00143968, 0.00693937] | 0.01032347 [0.00333451, 0.01759005] | 是 |
| use_time_position | attention | 3 | -94.792 [-98.438, -90.104] | -0.01181491 [-0.01426660, -0.00947361] | -0.03865429 [-0.04770751, -0.02892738] | 否 |
| use_time_position | attention | 6 | -94.792 [-98.438, -90.104] | -0.01360945 [-0.01590120, -0.01124246] | -0.04064935 [-0.05110220, -0.02975946] | 否 |
| use_time_position | keep_first | 3 | 5.208 [1.562, 9.896] | 0.00567944 [0.00308337, 0.00859595] | 0.02275552 [0.01376408, 0.03231941] | 是 |
| use_time_position | keep_first | 6 | 5.208 [1.562, 9.896] | 0.00546576 [0.00285519, 0.00831831] | 0.01936814 [0.01104770, 0.02834519] | 是 |
| use_time_position | set/future_attention | 3 | -94.792 [-98.438, -90.104] | -0.01078404 [-0.01300572, -0.00834382] | -0.03736432 [-0.04689485, -0.02732050] | 否 |
| use_time_position | set/future_attention | 6 | -94.792 [-98.438, -90.104] | -0.01262362 [-0.01501255, -0.00992872] | -0.03759343 [-0.04720038, -0.02794084] | 否 |

同标签集合/独立：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | -1.562 [-7.292, 4.688] | 0.00050560 [-0.00192168, 0.00308713] | -0.00177338 [-0.00965775, 0.00655020] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | -1.562 [-7.292, 4.688] | 0.00041295 [-0.00208276, 0.00304846] | -0.00634442 [-0.01487054, 0.00230995] | 否 |
| local_position | independent/local_position **INIT** | 3 | -1.562 [-7.292, 4.688] | 0.00050560 [-0.00192168, 0.00308713] | -0.00177338 [-0.00965775, 0.00655020] | 否 |
| local_position | independent/local_position **INIT** | 6 | -1.562 [-7.292, 4.688] | 0.00041295 [-0.00208276, 0.00304846] | -0.00634442 [-0.01487054, 0.00230995] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | -1.562 [-7.292, 4.688] | 0.00050560 [-0.00192168, 0.00308713] | -0.00177338 [-0.00965775, 0.00655020] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | -1.562 [-7.292, 4.688] | 0.00041295 [-0.00208276, 0.00304846] | -0.00634442 [-0.01487054, 0.00230995] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | -1.562 [-7.292, 4.688] | 0.00050560 [-0.00192168, 0.00308713] | -0.00177338 [-0.00965775, 0.00655020] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | -1.562 [-7.292, 4.688] | 0.00041295 [-0.00208276, 0.00304846] | -0.00634442 [-0.01487054, 0.00230995] | 否 |

### ordinary / student或RNG seed 43

| 方法 | h | 成功率 % [95% CI] | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- | --- |
| fifo | 3 | 0.000 [0.000, 0.000] | 0.02379281 [0.02101828, 0.02673472] | 0.24782456 [0.23775571, 0.25816802] |
| fifo | 6 | 0.000 [0.000, 0.000] | 0.02735527 [0.02493662, 0.02992922] | 0.26252034 [0.25342580, 0.27179227] |
| reservoir | 3 | 3.125 [0.000, 7.292] | 0.02128930 [0.01894887, 0.02406588] | 0.24539099 [0.23435488, 0.25673172] |
| reservoir | 6 | 3.125 [0.000, 7.292] | 0.02473065 [0.02259285, 0.02707444] | 0.25499005 [0.24463443, 0.26509282] |
| temporal_buckets | 3 | 0.000 [0.000, 0.000] | 0.02359853 [0.02106712, 0.02636866] | 0.24461806 [0.23475158, 0.25471799] |
| temporal_buckets | 6 | 0.000 [0.000, 0.000] | 0.02706629 [0.02487122, 0.02942262] | 0.25852891 [0.24939653, 0.26793686] |
| diversity | 3 | 9.375 [4.688, 14.583] | 0.01666226 [0.01505911, 0.01827971] | 0.23339935 [0.22340898, 0.24354676] |
| diversity | 6 | 9.375 [4.688, 14.583] | 0.02059435 [0.01904824, 0.02216771] | 0.24624977 [0.23605378, 0.25614476] |
| surprise | 3 | 0.000 [0.000, 0.000] | 0.02483195 [0.02197593, 0.02797348] | 0.25388996 [0.24325622, 0.26511745] |
| surprise | 6 | 0.000 [0.000, 0.000] | 0.02877268 [0.02608214, 0.03166681] | 0.27248656 [0.26282511, 0.28274330] |
| attention | 3 | 100.000 [100.000, 100.000] | 0.00914680 [0.00749575, 0.01088691] | 0.20492259 [0.19395310, 0.21653455] |
| attention | 6 | 100.000 [100.000, 100.000] | 0.01114779 [0.00928566, 0.01308783] | 0.22151373 [0.21047667, 0.23266554] |
| keep_first | 3 | 0.000 [0.000, 0.000] | 0.02664115 [0.02370149, 0.02978971] | 0.26633239 [0.25533553, 0.27805961] |
| keep_first | 6 | 0.000 [0.000, 0.000] | 0.03022299 [0.02766103, 0.03294746] | 0.28153123 [0.27165338, 0.29235949] |
| independent/local_feature **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/local_feature **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| independent/local_position **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/local_position **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| independent/use_time_feature **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/use_time_feature **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| independent/use_time_position **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/use_time_position **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| independent/future_attention | 3 | 95.833 [91.667, 98.958] | 0.00941231 [0.00774707, 0.01130999] | 0.20696965 [0.19644307, 0.21842570] |
| independent/future_attention | 6 | 95.833 [91.667, 98.958] | 0.01146399 [0.00965372, 0.01341472] | 0.22544175 [0.21493921, 0.23628478] |
| independent/shuffle_local_feature **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/shuffle_local_feature **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| independent/shuffle_local_position **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/shuffle_local_position **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| independent/shuffle_use_time_feature **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/shuffle_use_time_feature **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| independent/shuffle_use_time_position **INIT** | 3 | 3.646 [1.042, 7.292] | 0.02264442 [0.02021216, 0.02521053] | 0.24603296 [0.23526852, 0.25643485] |
| independent/shuffle_use_time_position **INIT** | 6 | 3.646 [1.042, 7.292] | 0.02649261 [0.02411400, 0.02900167] | 0.26027912 [0.25035951, 0.27070435] |
| set/local_feature **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/local_feature **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| set/local_position **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/local_position **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| set/use_time_feature **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/use_time_feature **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| set/use_time_position **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/use_time_position **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| set/future_attention | 3 | 100.000 [100.000, 100.000] | 0.00906886 [0.00757210, 0.01067853] | 0.20593111 [0.19531046, 0.21679049] |
| set/future_attention | 6 | 100.000 [100.000, 100.000] | 0.01119565 [0.00957708, 0.01303689] | 0.22154183 [0.21126710, 0.23175315] |
| set/shuffle_local_feature **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/shuffle_local_feature **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| set/shuffle_local_position **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/shuffle_local_position **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| set/shuffle_use_time_feature **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/shuffle_use_time_feature **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| set/shuffle_use_time_position **INIT** | 3 | 6.250 [2.604, 10.938] | 0.02163262 [0.01919740, 0.02431769] | 0.24281549 [0.23124226, 0.25400335] |
| set/shuffle_use_time_position **INIT** | 6 | 6.250 [2.604, 10.938] | 0.02543401 [0.02307200, 0.02790035] | 0.25500965 [0.24524380, 0.26512143] |
| 参照/last_visible | 3 | 100.000 [100.000, 100.000] | 0.01112028 [0.00942257, 0.01314966] | 0.22187655 [0.21128329, 0.23302433] |
| 参照/last_visible | 6 | 100.000 [100.000, 100.000] | 0.01336026 [0.01162503, 0.01532875] | 0.23312347 [0.22200088, 0.24453439] |
| 参照/b | 3 | 100.000 [100.000, 100.000] | 0.00605192 [0.00476684, 0.00758422] | 0.18681108 [0.17628059, 0.19850645] |
| 参照/b | 6 | 100.000 [100.000, 100.000] | 0.00685986 [0.00563873, 0.00820422] | 0.19888737 [0.18907097, 0.20860782] |
| 参照/posthoc_best | 3 | 52.604 [46.875, 58.333] | 0.00253535 [0.00188422, 0.00331072] | 0.19745953 [0.18882705, 0.20619257] |
| 参照/posthoc_best | 6 | 52.604 [46.875, 58.333] | 0.00313890 [0.00242108, 0.00398104] | 0.21405261 [0.20471316, 0.22317230] |

四标签对全部对手：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 6.250 [2.604, 10.938] | 0.00216019 [0.00014801, 0.00448789] | 0.00500907 [-0.00106384, 0.01132384] | 是 |
| local_feature | fifo | 6 | 6.250 [2.604, 10.938] | 0.00192125 [-0.00021709, 0.00431320] | 0.00751069 [0.00155125, 0.01379054] | 否 |
| local_feature | reservoir | 3 | 3.125 [-2.083, 8.333] | -0.00034333 [-0.00239118, 0.00199710] | 0.00257550 [-0.00496449, 0.01051541] | 否 |
| local_feature | reservoir | 6 | 3.125 [-2.083, 8.333] | -0.00070336 [-0.00280174, 0.00175052] | -0.00001961 [-0.00759032, 0.00836055] | 否 |
| local_feature | temporal_buckets | 3 | 6.250 [2.604, 10.938] | 0.00196590 [0.00026323, 0.00399128] | 0.00180257 [-0.00448947, 0.00837855] | 是 |
| local_feature | temporal_buckets | 6 | 6.250 [2.604, 10.938] | 0.00163228 [-0.00008279, 0.00364343] | 0.00351926 [-0.00191521, 0.00936108] | 否 |
| local_feature | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00497037 [-0.00723750, -0.00278871] | -0.00941614 [-0.01678635, -0.00199791] | 否 |
| local_feature | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00483967 [-0.00724776, -0.00254021] | -0.00875988 [-0.01682518, -0.00047523] | 否 |
| local_feature | surprise | 3 | 6.250 [2.604, 10.938] | 0.00319932 [0.00096085, 0.00577321] | 0.01107447 [0.00410794, 0.01907510] | 是 |
| local_feature | surprise | 6 | 6.250 [2.604, 10.938] | 0.00333867 [0.00113089, 0.00598236] | 0.01747691 [0.01100919, 0.02450297] | 是 |
| local_feature | attention | 3 | -93.750 [-97.396, -89.062] | -0.01248583 [-0.01510502, -0.01006373] | -0.03789290 [-0.04652171, -0.02910197] | 否 |
| local_feature | attention | 6 | -93.750 [-97.396, -89.062] | -0.01428623 [-0.01692028, -0.01185006] | -0.03349592 [-0.04352405, -0.02294009] | 否 |
| local_feature | keep_first | 3 | 6.250 [2.604, 10.938] | 0.00500852 [0.00270740, 0.00743614] | 0.02351691 [0.01462121, 0.03279532] | 是 |
| local_feature | keep_first | 6 | 6.250 [2.604, 10.938] | 0.00478898 [0.00249688, 0.00728903] | 0.02652158 [0.01938267, 0.03421549] | 是 |
| local_feature | set/future_attention | 3 | -93.750 [-97.396, -89.062] | -0.01256377 [-0.01507130, -0.01019662] | -0.03688438 [-0.04637504, -0.02745994] | 否 |
| local_feature | set/future_attention | 6 | -93.750 [-97.396, -89.062] | -0.01423837 [-0.01677181, -0.01170021] | -0.03346783 [-0.04310351, -0.02381831] | 否 |
| local_position | fifo | 3 | 6.250 [2.604, 10.938] | 0.00216019 [0.00014801, 0.00448789] | 0.00500907 [-0.00106384, 0.01132384] | 是 |
| local_position | fifo | 6 | 6.250 [2.604, 10.938] | 0.00192125 [-0.00021709, 0.00431320] | 0.00751069 [0.00155125, 0.01379054] | 否 |
| local_position | reservoir | 3 | 3.125 [-2.083, 8.333] | -0.00034333 [-0.00239118, 0.00199710] | 0.00257550 [-0.00496449, 0.01051541] | 否 |
| local_position | reservoir | 6 | 3.125 [-2.083, 8.333] | -0.00070336 [-0.00280174, 0.00175052] | -0.00001961 [-0.00759032, 0.00836055] | 否 |
| local_position | temporal_buckets | 3 | 6.250 [2.604, 10.938] | 0.00196590 [0.00026323, 0.00399128] | 0.00180257 [-0.00448947, 0.00837855] | 是 |
| local_position | temporal_buckets | 6 | 6.250 [2.604, 10.938] | 0.00163228 [-0.00008279, 0.00364343] | 0.00351926 [-0.00191521, 0.00936108] | 否 |
| local_position | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00497037 [-0.00723750, -0.00278871] | -0.00941614 [-0.01678635, -0.00199791] | 否 |
| local_position | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00483967 [-0.00724776, -0.00254021] | -0.00875988 [-0.01682518, -0.00047523] | 否 |
| local_position | surprise | 3 | 6.250 [2.604, 10.938] | 0.00319932 [0.00096085, 0.00577321] | 0.01107447 [0.00410794, 0.01907510] | 是 |
| local_position | surprise | 6 | 6.250 [2.604, 10.938] | 0.00333867 [0.00113089, 0.00598236] | 0.01747691 [0.01100919, 0.02450297] | 是 |
| local_position | attention | 3 | -93.750 [-97.396, -89.062] | -0.01248583 [-0.01510502, -0.01006373] | -0.03789290 [-0.04652171, -0.02910197] | 否 |
| local_position | attention | 6 | -93.750 [-97.396, -89.062] | -0.01428623 [-0.01692028, -0.01185006] | -0.03349592 [-0.04352405, -0.02294009] | 否 |
| local_position | keep_first | 3 | 6.250 [2.604, 10.938] | 0.00500852 [0.00270740, 0.00743614] | 0.02351691 [0.01462121, 0.03279532] | 是 |
| local_position | keep_first | 6 | 6.250 [2.604, 10.938] | 0.00478898 [0.00249688, 0.00728903] | 0.02652158 [0.01938267, 0.03421549] | 是 |
| local_position | set/future_attention | 3 | -93.750 [-97.396, -89.062] | -0.01256377 [-0.01507130, -0.01019662] | -0.03688438 [-0.04637504, -0.02745994] | 否 |
| local_position | set/future_attention | 6 | -93.750 [-97.396, -89.062] | -0.01423837 [-0.01677181, -0.01170021] | -0.03346783 [-0.04310351, -0.02381831] | 否 |
| use_time_feature | fifo | 3 | 6.250 [2.604, 10.938] | 0.00216019 [0.00014801, 0.00448789] | 0.00500907 [-0.00106384, 0.01132384] | 是 |
| use_time_feature | fifo | 6 | 6.250 [2.604, 10.938] | 0.00192125 [-0.00021709, 0.00431320] | 0.00751069 [0.00155125, 0.01379054] | 否 |
| use_time_feature | reservoir | 3 | 3.125 [-2.083, 8.333] | -0.00034333 [-0.00239118, 0.00199710] | 0.00257550 [-0.00496449, 0.01051541] | 否 |
| use_time_feature | reservoir | 6 | 3.125 [-2.083, 8.333] | -0.00070336 [-0.00280174, 0.00175052] | -0.00001961 [-0.00759032, 0.00836055] | 否 |
| use_time_feature | temporal_buckets | 3 | 6.250 [2.604, 10.938] | 0.00196590 [0.00026323, 0.00399128] | 0.00180257 [-0.00448947, 0.00837855] | 是 |
| use_time_feature | temporal_buckets | 6 | 6.250 [2.604, 10.938] | 0.00163228 [-0.00008279, 0.00364343] | 0.00351926 [-0.00191521, 0.00936108] | 否 |
| use_time_feature | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00497037 [-0.00723750, -0.00278871] | -0.00941614 [-0.01678635, -0.00199791] | 否 |
| use_time_feature | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00483967 [-0.00724776, -0.00254021] | -0.00875988 [-0.01682518, -0.00047523] | 否 |
| use_time_feature | surprise | 3 | 6.250 [2.604, 10.938] | 0.00319932 [0.00096085, 0.00577321] | 0.01107447 [0.00410794, 0.01907510] | 是 |
| use_time_feature | surprise | 6 | 6.250 [2.604, 10.938] | 0.00333867 [0.00113089, 0.00598236] | 0.01747691 [0.01100919, 0.02450297] | 是 |
| use_time_feature | attention | 3 | -93.750 [-97.396, -89.062] | -0.01248583 [-0.01510502, -0.01006373] | -0.03789290 [-0.04652171, -0.02910197] | 否 |
| use_time_feature | attention | 6 | -93.750 [-97.396, -89.062] | -0.01428623 [-0.01692028, -0.01185006] | -0.03349592 [-0.04352405, -0.02294009] | 否 |
| use_time_feature | keep_first | 3 | 6.250 [2.604, 10.938] | 0.00500852 [0.00270740, 0.00743614] | 0.02351691 [0.01462121, 0.03279532] | 是 |
| use_time_feature | keep_first | 6 | 6.250 [2.604, 10.938] | 0.00478898 [0.00249688, 0.00728903] | 0.02652158 [0.01938267, 0.03421549] | 是 |
| use_time_feature | set/future_attention | 3 | -93.750 [-97.396, -89.062] | -0.01256377 [-0.01507130, -0.01019662] | -0.03688438 [-0.04637504, -0.02745994] | 否 |
| use_time_feature | set/future_attention | 6 | -93.750 [-97.396, -89.062] | -0.01423837 [-0.01677181, -0.01170021] | -0.03346783 [-0.04310351, -0.02381831] | 否 |
| use_time_position | fifo | 3 | 6.250 [2.604, 10.938] | 0.00216019 [0.00014801, 0.00448789] | 0.00500907 [-0.00106384, 0.01132384] | 是 |
| use_time_position | fifo | 6 | 6.250 [2.604, 10.938] | 0.00192125 [-0.00021709, 0.00431320] | 0.00751069 [0.00155125, 0.01379054] | 否 |
| use_time_position | reservoir | 3 | 3.125 [-2.083, 8.333] | -0.00034333 [-0.00239118, 0.00199710] | 0.00257550 [-0.00496449, 0.01051541] | 否 |
| use_time_position | reservoir | 6 | 3.125 [-2.083, 8.333] | -0.00070336 [-0.00280174, 0.00175052] | -0.00001961 [-0.00759032, 0.00836055] | 否 |
| use_time_position | temporal_buckets | 3 | 6.250 [2.604, 10.938] | 0.00196590 [0.00026323, 0.00399128] | 0.00180257 [-0.00448947, 0.00837855] | 是 |
| use_time_position | temporal_buckets | 6 | 6.250 [2.604, 10.938] | 0.00163228 [-0.00008279, 0.00364343] | 0.00351926 [-0.00191521, 0.00936108] | 否 |
| use_time_position | diversity | 3 | -3.125 [-9.375, 3.646] | -0.00497037 [-0.00723750, -0.00278871] | -0.00941614 [-0.01678635, -0.00199791] | 否 |
| use_time_position | diversity | 6 | -3.125 [-9.375, 3.646] | -0.00483967 [-0.00724776, -0.00254021] | -0.00875988 [-0.01682518, -0.00047523] | 否 |
| use_time_position | surprise | 3 | 6.250 [2.604, 10.938] | 0.00319932 [0.00096085, 0.00577321] | 0.01107447 [0.00410794, 0.01907510] | 是 |
| use_time_position | surprise | 6 | 6.250 [2.604, 10.938] | 0.00333867 [0.00113089, 0.00598236] | 0.01747691 [0.01100919, 0.02450297] | 是 |
| use_time_position | attention | 3 | -93.750 [-97.396, -89.062] | -0.01248583 [-0.01510502, -0.01006373] | -0.03789290 [-0.04652171, -0.02910197] | 否 |
| use_time_position | attention | 6 | -93.750 [-97.396, -89.062] | -0.01428623 [-0.01692028, -0.01185006] | -0.03349592 [-0.04352405, -0.02294009] | 否 |
| use_time_position | keep_first | 3 | 6.250 [2.604, 10.938] | 0.00500852 [0.00270740, 0.00743614] | 0.02351691 [0.01462121, 0.03279532] | 是 |
| use_time_position | keep_first | 6 | 6.250 [2.604, 10.938] | 0.00478898 [0.00249688, 0.00728903] | 0.02652158 [0.01938267, 0.03421549] | 是 |
| use_time_position | set/future_attention | 3 | -93.750 [-97.396, -89.062] | -0.01256377 [-0.01507130, -0.01019662] | -0.03688438 [-0.04637504, -0.02745994] | 否 |
| use_time_position | set/future_attention | 6 | -93.750 [-97.396, -89.062] | -0.01423837 [-0.01677181, -0.01170021] | -0.03346783 [-0.04310351, -0.02381831] | 否 |

同标签集合/独立：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | 2.604 [-2.604, 7.812] | 0.00101180 [-0.00091342, 0.00304036] | 0.00321747 [-0.00421957, 0.01035298] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | 2.604 [-2.604, 7.812] | 0.00105860 [-0.00094081, 0.00316233] | 0.00526947 [-0.00154120, 0.01229457] | 否 |
| local_position | independent/local_position **INIT** | 3 | 2.604 [-2.604, 7.812] | 0.00101180 [-0.00091342, 0.00304036] | 0.00321747 [-0.00421957, 0.01035298] | 否 |
| local_position | independent/local_position **INIT** | 6 | 2.604 [-2.604, 7.812] | 0.00105860 [-0.00094081, 0.00316233] | 0.00526947 [-0.00154120, 0.01229457] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | 2.604 [-2.604, 7.812] | 0.00101180 [-0.00091342, 0.00304036] | 0.00321747 [-0.00421957, 0.01035298] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | 2.604 [-2.604, 7.812] | 0.00105860 [-0.00094081, 0.00316233] | 0.00526947 [-0.00154120, 0.01229457] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | 2.604 [-2.604, 7.812] | 0.00101180 [-0.00091342, 0.00304036] | 0.00321747 [-0.00421957, 0.01035298] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | 2.604 [-2.604, 7.812] | 0.00105860 [-0.00094081, 0.00316233] | 0.00526947 [-0.00154120, 0.01229457] | 否 |

### long_gap / student或RNG seed 17

| 方法 | h | 成功率 % [95% CI] | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- | --- |
| fifo | 3 | 0.000 [0.000, 0.000] | 0.04214762 [0.03873740, 0.04587208] | 0.26536781 [0.25663814, 0.27494357] |
| fifo | 6 | 0.000 [0.000, 0.000] | 0.04430579 [0.04041927, 0.04849210] | 0.27844352 [0.26757627, 0.29110831] |
| reservoir | 3 | 1.042 [0.000, 3.125] | 0.04181428 [0.03804538, 0.04568595] | 0.27511546 [0.26517039, 0.28594706] |
| reservoir | 6 | 1.042 [0.000, 3.125] | 0.04439522 [0.04028905, 0.04880284] | 0.29095796 [0.27706772, 0.30633012] |
| temporal_buckets | 3 | 0.000 [0.000, 0.000] | 0.04136660 [0.03798401, 0.04512719] | 0.26216290 [0.25359664, 0.27205960] |
| temporal_buckets | 6 | 0.000 [0.000, 0.000] | 0.04354132 [0.03961855, 0.04781825] | 0.27449347 [0.26374027, 0.28700720] |
| diversity | 3 | 3.646 [1.042, 6.771] | 0.04151203 [0.03691887, 0.04635456] | 0.29146073 [0.27956205, 0.30409375] |
| diversity | 6 | 3.646 [1.042, 6.771] | 0.04601170 [0.04107869, 0.05135678] | 0.31066750 [0.29621519, 0.32545413] |
| surprise | 3 | 0.000 [0.000, 0.000] | 0.04387128 [0.04012465, 0.04776287] | 0.28639797 [0.27602966, 0.29725240] |
| surprise | 6 | 0.000 [0.000, 0.000] | 0.04637211 [0.04217884, 0.05076471] | 0.30404933 [0.29084016, 0.31769186] |
| attention | 3 | 100.000 [100.000, 100.000] | 0.02591103 [0.02224589, 0.02991959] | 0.34767058 [0.32975061, 0.36634482] |
| attention | 6 | 100.000 [100.000, 100.000] | 0.03022034 [0.02640182, 0.03436905] | 0.38567640 [0.36234696, 0.41214033] |
| keep_first | 3 | 0.000 [0.000, 0.000] | 0.04466129 [0.04106630, 0.04848013] | 0.30564786 [0.29303160, 0.31840464] |
| keep_first | 6 | 0.000 [0.000, 0.000] | 0.04756606 [0.04336411, 0.05207880] | 0.32398132 [0.30914340, 0.33989151] |
| independent/local_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/local_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| independent/local_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/local_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| independent/use_time_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/use_time_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| independent/use_time_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/use_time_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| independent/future_attention | 3 | 93.229 [89.062, 96.875] | 0.02545267 [0.02205380, 0.02917028] | 0.34274094 [0.32490355, 0.36170082] |
| independent/future_attention | 6 | 93.229 [89.062, 96.875] | 0.03018359 [0.02644913, 0.03436474] | 0.38233111 [0.35846235, 0.40769589] |
| independent/shuffle_local_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/shuffle_local_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| independent/shuffle_local_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/shuffle_local_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| independent/shuffle_use_time_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/shuffle_use_time_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| independent/shuffle_use_time_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04132794 [0.03769077, 0.04529826] | 0.26929579 [0.25923697, 0.28084971] |
| independent/shuffle_use_time_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04400045 [0.03984797, 0.04835800] | 0.28535147 [0.27406815, 0.29825615] |
| set/local_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/local_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| set/local_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/local_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| set/use_time_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/use_time_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| set/use_time_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/use_time_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| set/future_attention | 3 | 100.000 [100.000, 100.000] | 0.02514704 [0.02188513, 0.02889371] | 0.34816426 [0.32980584, 0.36828037] |
| set/future_attention | 6 | 100.000 [100.000, 100.000] | 0.02980009 [0.02621295, 0.03386094] | 0.38516063 [0.36131862, 0.41232987] |
| set/shuffle_local_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/shuffle_local_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| set/shuffle_local_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/shuffle_local_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| set/shuffle_use_time_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/shuffle_use_time_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| set/shuffle_use_time_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04084413 [0.03727685, 0.04500168] | 0.26667227 [0.25753885, 0.27724481] |
| set/shuffle_use_time_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04352628 [0.03937438, 0.04817317] | 0.28134682 [0.27049334, 0.29409081] |
| 参照/last_visible | 3 | 100.000 [100.000, 100.000] | 0.02921170 [0.02604360, 0.03277738] | 0.33395575 [0.31772962, 0.35107652] |
| 参照/last_visible | 6 | 100.000 [100.000, 100.000] | 0.03382171 [0.03032666, 0.03760882] | 0.37014249 [0.34969096, 0.39280120] |
| 参照/b | 3 | 100.000 [100.000, 100.000] | 0.01763906 [0.01524359, 0.02014464] | 0.33516119 [0.31541262, 0.35505115] |
| 参照/b | 6 | 100.000 [100.000, 100.000] | 0.02181453 [0.01897905, 0.02471951] | 0.37246110 [0.34915977, 0.39784230] |
| 参照/posthoc_best | 3 | 52.604 [45.833, 59.896] | 0.00450824 [0.00370655, 0.00548251] | 0.27852603 [0.26302341, 0.29485930] |
| 参照/posthoc_best | 6 | 52.604 [45.833, 59.896] | 0.00506205 [0.00425110, 0.00597218] | 0.30553522 [0.28634330, 0.32634326] |

四标签对全部对手：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 0.521 [0.000, 1.562] | 0.00130350 [-0.00014266, 0.00278111] | -0.00130446 [-0.00547163, 0.00257852] | 否 |
| local_feature | fifo | 6 | 0.521 [0.000, 1.562] | 0.00077951 [-0.00066926, 0.00217814] | -0.00290330 [-0.00761770, 0.00171473] | 否 |
| local_feature | reservoir | 3 | -0.521 [-3.125, 1.562] | 0.00097015 [-0.00115924, 0.00294390] | 0.00844318 [0.00277468, 0.01386767] | 否 |
| local_feature | reservoir | 6 | -0.521 [-3.125, 1.562] | 0.00086894 [-0.00141621, 0.00303071] | 0.00961115 [0.00310526, 0.01658984] | 否 |
| local_feature | temporal_buckets | 3 | 0.521 [0.000, 1.562] | 0.00052248 [-0.00086467, 0.00191356] | -0.00450937 [-0.00819884, -0.00084707] | 否 |
| local_feature | temporal_buckets | 6 | 0.521 [0.000, 1.562] | 0.00001504 [-0.00135545, 0.00137003] | -0.00685335 [-0.01141379, -0.00257562] | 否 |
| local_feature | diversity | 3 | -3.125 [-6.250, -0.521] | 0.00066791 [-0.00192430, 0.00322072] | 0.02478845 [0.01732618, 0.03225971] | 否 |
| local_feature | diversity | 6 | -3.125 [-6.250, -0.521] | 0.00248542 [-0.00019415, 0.00516048] | 0.02932069 [0.02167684, 0.03715837] | 否 |
| local_feature | surprise | 3 | 0.521 [0.000, 1.562] | 0.00302715 [0.00093378, 0.00519680] | 0.01972570 [0.01321717, 0.02643916] | 否 |
| local_feature | surprise | 6 | 0.521 [0.000, 1.562] | 0.00284584 [0.00065597, 0.00505642] | 0.02270251 [0.01487769, 0.03069303] | 否 |
| local_feature | attention | 3 | -99.479 [-100.000, -98.438] | -0.01493309 [-0.01898191, -0.01099201] | 0.08099831 [0.06483258, 0.09769105] | 否 |
| local_feature | attention | 6 | -99.479 [-100.000, -98.438] | -0.01330594 [-0.01795276, -0.00893396] | 0.10432958 [0.08532687, 0.12458634] | 否 |
| local_feature | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00381716 [0.00189259, 0.00563010] | 0.03897559 [0.03026495, 0.04692332] | 否 |
| local_feature | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00403978 [0.00204631, 0.00591780] | 0.04263450 [0.03306861, 0.05219883] | 否 |
| local_feature | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01569708 [-0.02045324, -0.01123640] | 0.08149199 [0.06549102, 0.09880244] | 否 |
| local_feature | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01372618 [-0.01927609, -0.00856469] | 0.10381381 [0.08437505, 0.12527063] | 否 |
| local_position | fifo | 3 | 0.521 [0.000, 1.562] | 0.00130350 [-0.00014266, 0.00278111] | -0.00130446 [-0.00547163, 0.00257852] | 否 |
| local_position | fifo | 6 | 0.521 [0.000, 1.562] | 0.00077951 [-0.00066926, 0.00217814] | -0.00290330 [-0.00761770, 0.00171473] | 否 |
| local_position | reservoir | 3 | -0.521 [-3.125, 1.562] | 0.00097015 [-0.00115924, 0.00294390] | 0.00844318 [0.00277468, 0.01386767] | 否 |
| local_position | reservoir | 6 | -0.521 [-3.125, 1.562] | 0.00086894 [-0.00141621, 0.00303071] | 0.00961115 [0.00310526, 0.01658984] | 否 |
| local_position | temporal_buckets | 3 | 0.521 [0.000, 1.562] | 0.00052248 [-0.00086467, 0.00191356] | -0.00450937 [-0.00819884, -0.00084707] | 否 |
| local_position | temporal_buckets | 6 | 0.521 [0.000, 1.562] | 0.00001504 [-0.00135545, 0.00137003] | -0.00685335 [-0.01141379, -0.00257562] | 否 |
| local_position | diversity | 3 | -3.125 [-6.250, -0.521] | 0.00066791 [-0.00192430, 0.00322072] | 0.02478845 [0.01732618, 0.03225971] | 否 |
| local_position | diversity | 6 | -3.125 [-6.250, -0.521] | 0.00248542 [-0.00019415, 0.00516048] | 0.02932069 [0.02167684, 0.03715837] | 否 |
| local_position | surprise | 3 | 0.521 [0.000, 1.562] | 0.00302715 [0.00093378, 0.00519680] | 0.01972570 [0.01321717, 0.02643916] | 否 |
| local_position | surprise | 6 | 0.521 [0.000, 1.562] | 0.00284584 [0.00065597, 0.00505642] | 0.02270251 [0.01487769, 0.03069303] | 否 |
| local_position | attention | 3 | -99.479 [-100.000, -98.438] | -0.01493309 [-0.01898191, -0.01099201] | 0.08099831 [0.06483258, 0.09769105] | 否 |
| local_position | attention | 6 | -99.479 [-100.000, -98.438] | -0.01330594 [-0.01795276, -0.00893396] | 0.10432958 [0.08532687, 0.12458634] | 否 |
| local_position | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00381716 [0.00189259, 0.00563010] | 0.03897559 [0.03026495, 0.04692332] | 否 |
| local_position | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00403978 [0.00204631, 0.00591780] | 0.04263450 [0.03306861, 0.05219883] | 否 |
| local_position | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01569708 [-0.02045324, -0.01123640] | 0.08149199 [0.06549102, 0.09880244] | 否 |
| local_position | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01372618 [-0.01927609, -0.00856469] | 0.10381381 [0.08437505, 0.12527063] | 否 |
| use_time_feature | fifo | 3 | 0.521 [0.000, 1.562] | 0.00130350 [-0.00014266, 0.00278111] | -0.00130446 [-0.00547163, 0.00257852] | 否 |
| use_time_feature | fifo | 6 | 0.521 [0.000, 1.562] | 0.00077951 [-0.00066926, 0.00217814] | -0.00290330 [-0.00761770, 0.00171473] | 否 |
| use_time_feature | reservoir | 3 | -0.521 [-3.125, 1.562] | 0.00097015 [-0.00115924, 0.00294390] | 0.00844318 [0.00277468, 0.01386767] | 否 |
| use_time_feature | reservoir | 6 | -0.521 [-3.125, 1.562] | 0.00086894 [-0.00141621, 0.00303071] | 0.00961115 [0.00310526, 0.01658984] | 否 |
| use_time_feature | temporal_buckets | 3 | 0.521 [0.000, 1.562] | 0.00052248 [-0.00086467, 0.00191356] | -0.00450937 [-0.00819884, -0.00084707] | 否 |
| use_time_feature | temporal_buckets | 6 | 0.521 [0.000, 1.562] | 0.00001504 [-0.00135545, 0.00137003] | -0.00685335 [-0.01141379, -0.00257562] | 否 |
| use_time_feature | diversity | 3 | -3.125 [-6.250, -0.521] | 0.00066791 [-0.00192430, 0.00322072] | 0.02478845 [0.01732618, 0.03225971] | 否 |
| use_time_feature | diversity | 6 | -3.125 [-6.250, -0.521] | 0.00248542 [-0.00019415, 0.00516048] | 0.02932069 [0.02167684, 0.03715837] | 否 |
| use_time_feature | surprise | 3 | 0.521 [0.000, 1.562] | 0.00302715 [0.00093378, 0.00519680] | 0.01972570 [0.01321717, 0.02643916] | 否 |
| use_time_feature | surprise | 6 | 0.521 [0.000, 1.562] | 0.00284584 [0.00065597, 0.00505642] | 0.02270251 [0.01487769, 0.03069303] | 否 |
| use_time_feature | attention | 3 | -99.479 [-100.000, -98.438] | -0.01493309 [-0.01898191, -0.01099201] | 0.08099831 [0.06483258, 0.09769105] | 否 |
| use_time_feature | attention | 6 | -99.479 [-100.000, -98.438] | -0.01330594 [-0.01795276, -0.00893396] | 0.10432958 [0.08532687, 0.12458634] | 否 |
| use_time_feature | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00381716 [0.00189259, 0.00563010] | 0.03897559 [0.03026495, 0.04692332] | 否 |
| use_time_feature | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00403978 [0.00204631, 0.00591780] | 0.04263450 [0.03306861, 0.05219883] | 否 |
| use_time_feature | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01569708 [-0.02045324, -0.01123640] | 0.08149199 [0.06549102, 0.09880244] | 否 |
| use_time_feature | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01372618 [-0.01927609, -0.00856469] | 0.10381381 [0.08437505, 0.12527063] | 否 |
| use_time_position | fifo | 3 | 0.521 [0.000, 1.562] | 0.00130350 [-0.00014266, 0.00278111] | -0.00130446 [-0.00547163, 0.00257852] | 否 |
| use_time_position | fifo | 6 | 0.521 [0.000, 1.562] | 0.00077951 [-0.00066926, 0.00217814] | -0.00290330 [-0.00761770, 0.00171473] | 否 |
| use_time_position | reservoir | 3 | -0.521 [-3.125, 1.562] | 0.00097015 [-0.00115924, 0.00294390] | 0.00844318 [0.00277468, 0.01386767] | 否 |
| use_time_position | reservoir | 6 | -0.521 [-3.125, 1.562] | 0.00086894 [-0.00141621, 0.00303071] | 0.00961115 [0.00310526, 0.01658984] | 否 |
| use_time_position | temporal_buckets | 3 | 0.521 [0.000, 1.562] | 0.00052248 [-0.00086467, 0.00191356] | -0.00450937 [-0.00819884, -0.00084707] | 否 |
| use_time_position | temporal_buckets | 6 | 0.521 [0.000, 1.562] | 0.00001504 [-0.00135545, 0.00137003] | -0.00685335 [-0.01141379, -0.00257562] | 否 |
| use_time_position | diversity | 3 | -3.125 [-6.250, -0.521] | 0.00066791 [-0.00192430, 0.00322072] | 0.02478845 [0.01732618, 0.03225971] | 否 |
| use_time_position | diversity | 6 | -3.125 [-6.250, -0.521] | 0.00248542 [-0.00019415, 0.00516048] | 0.02932069 [0.02167684, 0.03715837] | 否 |
| use_time_position | surprise | 3 | 0.521 [0.000, 1.562] | 0.00302715 [0.00093378, 0.00519680] | 0.01972570 [0.01321717, 0.02643916] | 否 |
| use_time_position | surprise | 6 | 0.521 [0.000, 1.562] | 0.00284584 [0.00065597, 0.00505642] | 0.02270251 [0.01487769, 0.03069303] | 否 |
| use_time_position | attention | 3 | -99.479 [-100.000, -98.438] | -0.01493309 [-0.01898191, -0.01099201] | 0.08099831 [0.06483258, 0.09769105] | 否 |
| use_time_position | attention | 6 | -99.479 [-100.000, -98.438] | -0.01330594 [-0.01795276, -0.00893396] | 0.10432958 [0.08532687, 0.12458634] | 否 |
| use_time_position | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00381716 [0.00189259, 0.00563010] | 0.03897559 [0.03026495, 0.04692332] | 否 |
| use_time_position | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00403978 [0.00204631, 0.00591780] | 0.04263450 [0.03306861, 0.05219883] | 否 |
| use_time_position | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01569708 [-0.02045324, -0.01123640] | 0.08149199 [0.06549102, 0.09880244] | 否 |
| use_time_position | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01372618 [-0.01927609, -0.00856469] | 0.10381381 [0.08437505, 0.12527063] | 否 |

同标签集合/独立：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | -1.042 [-4.167, 1.042] | 0.00048381 [-0.00155996, 0.00243409] | 0.00262351 [-0.00231663, 0.00784449] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | -1.042 [-4.167, 1.042] | 0.00047417 [-0.00142003, 0.00228904] | 0.00400465 [-0.00102587, 0.00912713] | 否 |
| local_position | independent/local_position **INIT** | 3 | -1.042 [-4.167, 1.042] | 0.00048381 [-0.00155996, 0.00243409] | 0.00262351 [-0.00231663, 0.00784449] | 否 |
| local_position | independent/local_position **INIT** | 6 | -1.042 [-4.167, 1.042] | 0.00047417 [-0.00142003, 0.00228904] | 0.00400465 [-0.00102587, 0.00912713] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | -1.042 [-4.167, 1.042] | 0.00048381 [-0.00155996, 0.00243409] | 0.00262351 [-0.00231663, 0.00784449] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | -1.042 [-4.167, 1.042] | 0.00047417 [-0.00142003, 0.00228904] | 0.00400465 [-0.00102587, 0.00912713] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | -1.042 [-4.167, 1.042] | 0.00048381 [-0.00155996, 0.00243409] | 0.00262351 [-0.00231663, 0.00784449] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | -1.042 [-4.167, 1.042] | 0.00047417 [-0.00142003, 0.00228904] | 0.00400465 [-0.00102587, 0.00912713] | 否 |

### long_gap / student或RNG seed 29

| 方法 | h | 成功率 % [95% CI] | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- | --- |
| fifo | 3 | 0.000 [0.000, 0.000] | 0.04214762 [0.03873740, 0.04587208] | 0.26536781 [0.25663814, 0.27494357] |
| fifo | 6 | 0.000 [0.000, 0.000] | 0.04430579 [0.04041927, 0.04849210] | 0.27844352 [0.26757627, 0.29110831] |
| reservoir | 3 | 2.083 [0.000, 5.208] | 0.04132283 [0.03780625, 0.04506234] | 0.27891022 [0.26908184, 0.28988285] |
| reservoir | 6 | 2.083 [0.000, 5.208] | 0.04380562 [0.03979151, 0.04827045] | 0.29400341 [0.28161150, 0.30711582] |
| temporal_buckets | 3 | 0.000 [0.000, 0.000] | 0.04136660 [0.03798401, 0.04512719] | 0.26216290 [0.25359664, 0.27205960] |
| temporal_buckets | 6 | 0.000 [0.000, 0.000] | 0.04354132 [0.03961855, 0.04781825] | 0.27449347 [0.26374027, 0.28700720] |
| diversity | 3 | 3.646 [1.042, 6.771] | 0.04151203 [0.03691887, 0.04635456] | 0.29146073 [0.27956205, 0.30409375] |
| diversity | 6 | 3.646 [1.042, 6.771] | 0.04601170 [0.04107869, 0.05135678] | 0.31066750 [0.29621519, 0.32545413] |
| surprise | 3 | 0.000 [0.000, 0.000] | 0.04387128 [0.04012465, 0.04776287] | 0.28639797 [0.27602966, 0.29725240] |
| surprise | 6 | 0.000 [0.000, 0.000] | 0.04637211 [0.04217884, 0.05076471] | 0.30404933 [0.29084016, 0.31769186] |
| attention | 3 | 100.000 [100.000, 100.000] | 0.02591103 [0.02224589, 0.02991959] | 0.34767058 [0.32975061, 0.36634482] |
| attention | 6 | 100.000 [100.000, 100.000] | 0.03022034 [0.02640182, 0.03436905] | 0.38567640 [0.36234696, 0.41214033] |
| keep_first | 3 | 0.000 [0.000, 0.000] | 0.04466129 [0.04106630, 0.04848013] | 0.30564786 [0.29303160, 0.31840464] |
| keep_first | 6 | 0.000 [0.000, 0.000] | 0.04756606 [0.04336411, 0.05207880] | 0.32398132 [0.30914340, 0.33989151] |
| independent/local_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/local_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| independent/local_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/local_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| independent/use_time_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/use_time_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| independent/use_time_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/use_time_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| independent/future_attention | 3 | 93.229 [88.542, 97.396] | 0.02686359 [0.02367471, 0.03028133] | 0.33935281 [0.32087498, 0.35900681] |
| independent/future_attention | 6 | 93.229 [88.542, 97.396] | 0.03120772 [0.02773089, 0.03492667] | 0.37367967 [0.35049097, 0.40052602] |
| independent/shuffle_local_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/shuffle_local_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| independent/shuffle_local_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/shuffle_local_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| independent/shuffle_use_time_feature **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/shuffle_use_time_feature **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| independent/shuffle_use_time_position **INIT** | 3 | 1.562 [0.000, 4.167] | 0.04203234 [0.03835610, 0.04620611] | 0.27598787 [0.26614895, 0.28845145] |
| independent/shuffle_use_time_position **INIT** | 6 | 1.562 [0.000, 4.167] | 0.04454348 [0.04035288, 0.04915922] | 0.28941273 [0.27656315, 0.30358153] |
| set/local_feature **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/local_feature **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| set/local_position **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/local_position **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| set/use_time_feature **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/use_time_feature **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| set/use_time_position **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/use_time_position **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| set/future_attention | 3 | 100.000 [100.000, 100.000] | 0.02515051 [0.02187180, 0.02887717] | 0.35283024 [0.33383134, 0.37290507] |
| set/future_attention | 6 | 100.000 [100.000, 100.000] | 0.02968576 [0.02597640, 0.03389813] | 0.39235094 [0.36805588, 0.41967315] |
| set/shuffle_local_feature **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/shuffle_local_feature **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| set/shuffle_local_position **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/shuffle_local_position **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| set/shuffle_use_time_feature **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/shuffle_use_time_feature **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| set/shuffle_use_time_position **INIT** | 3 | 3.646 [1.029, 7.292] | 0.04238369 [0.03878172, 0.04620464] | 0.27215354 [0.26080531, 0.28404283] |
| set/shuffle_use_time_position **INIT** | 6 | 3.646 [1.029, 7.292] | 0.04490243 [0.04075758, 0.04937451] | 0.28994681 [0.27520578, 0.30584736] |
| 参照/last_visible | 3 | 100.000 [100.000, 100.000] | 0.02921170 [0.02604360, 0.03277738] | 0.33395575 [0.31772962, 0.35107652] |
| 参照/last_visible | 6 | 100.000 [100.000, 100.000] | 0.03382171 [0.03032666, 0.03760882] | 0.37014249 [0.34969096, 0.39280120] |
| 参照/b | 3 | 100.000 [100.000, 100.000] | 0.01763906 [0.01524359, 0.02014464] | 0.33516119 [0.31541262, 0.35505115] |
| 参照/b | 6 | 100.000 [100.000, 100.000] | 0.02181453 [0.01897905, 0.02471951] | 0.37246110 [0.34915977, 0.39784230] |
| 参照/posthoc_best | 3 | 52.604 [45.833, 59.896] | 0.00450824 [0.00370655, 0.00548251] | 0.27852603 [0.26302341, 0.29485930] |
| 参照/posthoc_best | 6 | 52.604 [45.833, 59.896] | 0.00506205 [0.00425110, 0.00597218] | 0.30553522 [0.28634330, 0.32634326] |

四标签对全部对手：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 3.646 [1.029, 7.292] | -0.00023606 [-0.00162580, 0.00118651] | -0.00678573 [-0.01596967, 0.00083704] | 否 |
| local_feature | fifo | 6 | 3.646 [1.029, 7.292] | -0.00059665 [-0.00202360, 0.00074880] | -0.01150329 [-0.02157299, -0.00277926] | 否 |
| local_feature | reservoir | 3 | 1.562 [-3.125, 5.742] | -0.00106085 [-0.00259864, 0.00052500] | 0.00675668 [-0.00280996, 0.01553685] | 否 |
| local_feature | reservoir | 6 | 1.562 [-3.125, 5.742] | -0.00109681 [-0.00279802, 0.00055438] | 0.00405660 [-0.00719344, 0.01446510] | 否 |
| local_feature | temporal_buckets | 3 | 3.646 [1.029, 7.292] | -0.00101708 [-0.00226758, 0.00022345] | -0.00999064 [-0.01864091, -0.00270196] | 否 |
| local_feature | temporal_buckets | 6 | 3.646 [1.029, 7.292] | -0.00136112 [-0.00283698, -0.00005551] | -0.01545334 [-0.02588404, -0.00628119] | 否 |
| local_feature | diversity | 3 | 0.000 [-4.167, 4.688] | -0.00087165 [-0.00371989, 0.00199124] | 0.01930719 [0.00937092, 0.02852060] | 否 |
| local_feature | diversity | 6 | 0.000 [-4.167, 4.688] | 0.00110927 [-0.00186256, 0.00399132] | 0.02072070 [0.01058974, 0.03060506] | 否 |
| local_feature | surprise | 3 | 3.646 [1.029, 7.292] | 0.00148759 [-0.00036645, 0.00336097] | 0.01424443 [0.00392716, 0.02382532] | 否 |
| local_feature | surprise | 6 | 3.646 [1.029, 7.292] | 0.00146968 [-0.00051846, 0.00343567] | 0.01410252 [0.00163401, 0.02583793] | 否 |
| local_feature | attention | 3 | -96.354 [-98.971, -92.708] | -0.01647265 [-0.02047053, -0.01238247] | 0.07551704 [0.05964704, 0.09191228] | 否 |
| local_feature | attention | 6 | -96.354 [-98.971, -92.708] | -0.01468209 [-0.01910320, -0.01016154] | 0.09572959 [0.07642107, 0.11621559] | 否 |
| local_feature | keep_first | 3 | 3.646 [1.029, 7.292] | 0.00227760 [0.00044884, 0.00415777] | 0.03349432 [0.02248238, 0.04338911] | 是 |
| local_feature | keep_first | 6 | 3.646 [1.029, 7.292] | 0.00266362 [0.00068333, 0.00461825] | 0.03403451 [0.02054286, 0.04652233] | 是 |
| local_feature | set/future_attention | 3 | -96.354 [-98.971, -92.708] | -0.01723317 [-0.02133056, -0.01301076] | 0.08067670 [0.06469722, 0.09755062] | 否 |
| local_feature | set/future_attention | 6 | -96.354 [-98.971, -92.708] | -0.01521667 [-0.01980811, -0.01051310] | 0.10240413 [0.08353060, 0.12237181] | 否 |
| local_position | fifo | 3 | 3.646 [1.029, 7.292] | -0.00023606 [-0.00162580, 0.00118651] | -0.00678573 [-0.01596967, 0.00083704] | 否 |
| local_position | fifo | 6 | 3.646 [1.029, 7.292] | -0.00059665 [-0.00202360, 0.00074880] | -0.01150329 [-0.02157299, -0.00277926] | 否 |
| local_position | reservoir | 3 | 1.562 [-3.125, 5.742] | -0.00106085 [-0.00259864, 0.00052500] | 0.00675668 [-0.00280996, 0.01553685] | 否 |
| local_position | reservoir | 6 | 1.562 [-3.125, 5.742] | -0.00109681 [-0.00279802, 0.00055438] | 0.00405660 [-0.00719344, 0.01446510] | 否 |
| local_position | temporal_buckets | 3 | 3.646 [1.029, 7.292] | -0.00101708 [-0.00226758, 0.00022345] | -0.00999064 [-0.01864091, -0.00270196] | 否 |
| local_position | temporal_buckets | 6 | 3.646 [1.029, 7.292] | -0.00136112 [-0.00283698, -0.00005551] | -0.01545334 [-0.02588404, -0.00628119] | 否 |
| local_position | diversity | 3 | 0.000 [-4.167, 4.688] | -0.00087165 [-0.00371989, 0.00199124] | 0.01930719 [0.00937092, 0.02852060] | 否 |
| local_position | diversity | 6 | 0.000 [-4.167, 4.688] | 0.00110927 [-0.00186256, 0.00399132] | 0.02072070 [0.01058974, 0.03060506] | 否 |
| local_position | surprise | 3 | 3.646 [1.029, 7.292] | 0.00148759 [-0.00036645, 0.00336097] | 0.01424443 [0.00392716, 0.02382532] | 否 |
| local_position | surprise | 6 | 3.646 [1.029, 7.292] | 0.00146968 [-0.00051846, 0.00343567] | 0.01410252 [0.00163401, 0.02583793] | 否 |
| local_position | attention | 3 | -96.354 [-98.971, -92.708] | -0.01647265 [-0.02047053, -0.01238247] | 0.07551704 [0.05964704, 0.09191228] | 否 |
| local_position | attention | 6 | -96.354 [-98.971, -92.708] | -0.01468209 [-0.01910320, -0.01016154] | 0.09572959 [0.07642107, 0.11621559] | 否 |
| local_position | keep_first | 3 | 3.646 [1.029, 7.292] | 0.00227760 [0.00044884, 0.00415777] | 0.03349432 [0.02248238, 0.04338911] | 是 |
| local_position | keep_first | 6 | 3.646 [1.029, 7.292] | 0.00266362 [0.00068333, 0.00461825] | 0.03403451 [0.02054286, 0.04652233] | 是 |
| local_position | set/future_attention | 3 | -96.354 [-98.971, -92.708] | -0.01723317 [-0.02133056, -0.01301076] | 0.08067670 [0.06469722, 0.09755062] | 否 |
| local_position | set/future_attention | 6 | -96.354 [-98.971, -92.708] | -0.01521667 [-0.01980811, -0.01051310] | 0.10240413 [0.08353060, 0.12237181] | 否 |
| use_time_feature | fifo | 3 | 3.646 [1.029, 7.292] | -0.00023606 [-0.00162580, 0.00118651] | -0.00678573 [-0.01596967, 0.00083704] | 否 |
| use_time_feature | fifo | 6 | 3.646 [1.029, 7.292] | -0.00059665 [-0.00202360, 0.00074880] | -0.01150329 [-0.02157299, -0.00277926] | 否 |
| use_time_feature | reservoir | 3 | 1.562 [-3.125, 5.742] | -0.00106085 [-0.00259864, 0.00052500] | 0.00675668 [-0.00280996, 0.01553685] | 否 |
| use_time_feature | reservoir | 6 | 1.562 [-3.125, 5.742] | -0.00109681 [-0.00279802, 0.00055438] | 0.00405660 [-0.00719344, 0.01446510] | 否 |
| use_time_feature | temporal_buckets | 3 | 3.646 [1.029, 7.292] | -0.00101708 [-0.00226758, 0.00022345] | -0.00999064 [-0.01864091, -0.00270196] | 否 |
| use_time_feature | temporal_buckets | 6 | 3.646 [1.029, 7.292] | -0.00136112 [-0.00283698, -0.00005551] | -0.01545334 [-0.02588404, -0.00628119] | 否 |
| use_time_feature | diversity | 3 | 0.000 [-4.167, 4.688] | -0.00087165 [-0.00371989, 0.00199124] | 0.01930719 [0.00937092, 0.02852060] | 否 |
| use_time_feature | diversity | 6 | 0.000 [-4.167, 4.688] | 0.00110927 [-0.00186256, 0.00399132] | 0.02072070 [0.01058974, 0.03060506] | 否 |
| use_time_feature | surprise | 3 | 3.646 [1.029, 7.292] | 0.00148759 [-0.00036645, 0.00336097] | 0.01424443 [0.00392716, 0.02382532] | 否 |
| use_time_feature | surprise | 6 | 3.646 [1.029, 7.292] | 0.00146968 [-0.00051846, 0.00343567] | 0.01410252 [0.00163401, 0.02583793] | 否 |
| use_time_feature | attention | 3 | -96.354 [-98.971, -92.708] | -0.01647265 [-0.02047053, -0.01238247] | 0.07551704 [0.05964704, 0.09191228] | 否 |
| use_time_feature | attention | 6 | -96.354 [-98.971, -92.708] | -0.01468209 [-0.01910320, -0.01016154] | 0.09572959 [0.07642107, 0.11621559] | 否 |
| use_time_feature | keep_first | 3 | 3.646 [1.029, 7.292] | 0.00227760 [0.00044884, 0.00415777] | 0.03349432 [0.02248238, 0.04338911] | 是 |
| use_time_feature | keep_first | 6 | 3.646 [1.029, 7.292] | 0.00266362 [0.00068333, 0.00461825] | 0.03403451 [0.02054286, 0.04652233] | 是 |
| use_time_feature | set/future_attention | 3 | -96.354 [-98.971, -92.708] | -0.01723317 [-0.02133056, -0.01301076] | 0.08067670 [0.06469722, 0.09755062] | 否 |
| use_time_feature | set/future_attention | 6 | -96.354 [-98.971, -92.708] | -0.01521667 [-0.01980811, -0.01051310] | 0.10240413 [0.08353060, 0.12237181] | 否 |
| use_time_position | fifo | 3 | 3.646 [1.029, 7.292] | -0.00023606 [-0.00162580, 0.00118651] | -0.00678573 [-0.01596967, 0.00083704] | 否 |
| use_time_position | fifo | 6 | 3.646 [1.029, 7.292] | -0.00059665 [-0.00202360, 0.00074880] | -0.01150329 [-0.02157299, -0.00277926] | 否 |
| use_time_position | reservoir | 3 | 1.562 [-3.125, 5.742] | -0.00106085 [-0.00259864, 0.00052500] | 0.00675668 [-0.00280996, 0.01553685] | 否 |
| use_time_position | reservoir | 6 | 1.562 [-3.125, 5.742] | -0.00109681 [-0.00279802, 0.00055438] | 0.00405660 [-0.00719344, 0.01446510] | 否 |
| use_time_position | temporal_buckets | 3 | 3.646 [1.029, 7.292] | -0.00101708 [-0.00226758, 0.00022345] | -0.00999064 [-0.01864091, -0.00270196] | 否 |
| use_time_position | temporal_buckets | 6 | 3.646 [1.029, 7.292] | -0.00136112 [-0.00283698, -0.00005551] | -0.01545334 [-0.02588404, -0.00628119] | 否 |
| use_time_position | diversity | 3 | 0.000 [-4.167, 4.688] | -0.00087165 [-0.00371989, 0.00199124] | 0.01930719 [0.00937092, 0.02852060] | 否 |
| use_time_position | diversity | 6 | 0.000 [-4.167, 4.688] | 0.00110927 [-0.00186256, 0.00399132] | 0.02072070 [0.01058974, 0.03060506] | 否 |
| use_time_position | surprise | 3 | 3.646 [1.029, 7.292] | 0.00148759 [-0.00036645, 0.00336097] | 0.01424443 [0.00392716, 0.02382532] | 否 |
| use_time_position | surprise | 6 | 3.646 [1.029, 7.292] | 0.00146968 [-0.00051846, 0.00343567] | 0.01410252 [0.00163401, 0.02583793] | 否 |
| use_time_position | attention | 3 | -96.354 [-98.971, -92.708] | -0.01647265 [-0.02047053, -0.01238247] | 0.07551704 [0.05964704, 0.09191228] | 否 |
| use_time_position | attention | 6 | -96.354 [-98.971, -92.708] | -0.01468209 [-0.01910320, -0.01016154] | 0.09572959 [0.07642107, 0.11621559] | 否 |
| use_time_position | keep_first | 3 | 3.646 [1.029, 7.292] | 0.00227760 [0.00044884, 0.00415777] | 0.03349432 [0.02248238, 0.04338911] | 是 |
| use_time_position | keep_first | 6 | 3.646 [1.029, 7.292] | 0.00266362 [0.00068333, 0.00461825] | 0.03403451 [0.02054286, 0.04652233] | 是 |
| use_time_position | set/future_attention | 3 | -96.354 [-98.971, -92.708] | -0.01723317 [-0.02133056, -0.01301076] | 0.08067670 [0.06469722, 0.09755062] | 否 |
| use_time_position | set/future_attention | 6 | -96.354 [-98.971, -92.708] | -0.01521667 [-0.01980811, -0.01051310] | 0.10240413 [0.08353060, 0.12237181] | 否 |

同标签集合/独立：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | 2.083 [-2.083, 6.250] | -0.00035134 [-0.00229169, 0.00152005] | 0.00383433 [-0.00591649, 0.01405037] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | 2.083 [-2.083, 6.250] | -0.00035896 [-0.00221555, 0.00152626] | -0.00053408 [-0.01164451, 0.01026091] | 否 |
| local_position | independent/local_position **INIT** | 3 | 2.083 [-2.083, 6.250] | -0.00035134 [-0.00229169, 0.00152005] | 0.00383433 [-0.00591649, 0.01405037] | 否 |
| local_position | independent/local_position **INIT** | 6 | 2.083 [-2.083, 6.250] | -0.00035896 [-0.00221555, 0.00152626] | -0.00053408 [-0.01164451, 0.01026091] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | 2.083 [-2.083, 6.250] | -0.00035134 [-0.00229169, 0.00152005] | 0.00383433 [-0.00591649, 0.01405037] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | 2.083 [-2.083, 6.250] | -0.00035896 [-0.00221555, 0.00152626] | -0.00053408 [-0.01164451, 0.01026091] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | 2.083 [-2.083, 6.250] | -0.00035134 [-0.00229169, 0.00152005] | 0.00383433 [-0.00591649, 0.01405037] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | 2.083 [-2.083, 6.250] | -0.00035896 [-0.00221555, 0.00152626] | -0.00053408 [-0.01164451, 0.01026091] | 否 |

### long_gap / student或RNG seed 43

| 方法 | h | 成功率 % [95% CI] | position MSE [95% CI] | feature MSE [95% CI] |
| --- | --- | --- | --- | --- |
| fifo | 3 | 0.000 [0.000, 0.000] | 0.04214762 [0.03873740, 0.04587208] | 0.26536781 [0.25663814, 0.27494357] |
| fifo | 6 | 0.000 [0.000, 0.000] | 0.04430579 [0.04041927, 0.04849210] | 0.27844352 [0.26757627, 0.29110831] |
| reservoir | 3 | 0.000 [0.000, 0.000] | 0.04223143 [0.03872132, 0.04603870] | 0.27147128 [0.26179524, 0.28150283] |
| reservoir | 6 | 0.000 [0.000, 0.000] | 0.04476361 [0.04080367, 0.04905146] | 0.28842951 [0.27585549, 0.30217027] |
| temporal_buckets | 3 | 0.000 [0.000, 0.000] | 0.04136660 [0.03798401, 0.04512719] | 0.26216290 [0.25359664, 0.27205960] |
| temporal_buckets | 6 | 0.000 [0.000, 0.000] | 0.04354132 [0.03961855, 0.04781825] | 0.27449347 [0.26374027, 0.28700720] |
| diversity | 3 | 3.646 [1.042, 6.771] | 0.04151203 [0.03691887, 0.04635456] | 0.29146073 [0.27956205, 0.30409375] |
| diversity | 6 | 3.646 [1.042, 6.771] | 0.04601170 [0.04107869, 0.05135678] | 0.31066750 [0.29621519, 0.32545413] |
| surprise | 3 | 0.000 [0.000, 0.000] | 0.04387128 [0.04012465, 0.04776287] | 0.28639797 [0.27602966, 0.29725240] |
| surprise | 6 | 0.000 [0.000, 0.000] | 0.04637211 [0.04217884, 0.05076471] | 0.30404933 [0.29084016, 0.31769186] |
| attention | 3 | 100.000 [100.000, 100.000] | 0.02591103 [0.02224589, 0.02991959] | 0.34767058 [0.32975061, 0.36634482] |
| attention | 6 | 100.000 [100.000, 100.000] | 0.03022034 [0.02640182, 0.03436905] | 0.38567640 [0.36234696, 0.41214033] |
| keep_first | 3 | 0.000 [0.000, 0.000] | 0.04466129 [0.04106630, 0.04848013] | 0.30564786 [0.29303160, 0.31840464] |
| keep_first | 6 | 0.000 [0.000, 0.000] | 0.04756606 [0.04336411, 0.05207880] | 0.32398132 [0.30914340, 0.33989151] |
| independent/local_feature **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/local_feature **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| independent/local_position **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/local_position **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| independent/use_time_feature **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/use_time_feature **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| independent/use_time_position **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/use_time_position **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| independent/future_attention | 3 | 93.229 [88.021, 97.396] | 0.02764372 [0.02414736, 0.03129401] | 0.34609877 [0.32657561, 0.36724608] |
| independent/future_attention | 6 | 93.229 [88.021, 97.396] | 0.03241814 [0.02855612, 0.03645801] | 0.38248044 [0.35817871, 0.41017417] |
| independent/shuffle_local_feature **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/shuffle_local_feature **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| independent/shuffle_local_position **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/shuffle_local_position **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| independent/shuffle_use_time_feature **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/shuffle_use_time_feature **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| independent/shuffle_use_time_position **INIT** | 3 | 0.000 [0.000, 0.000] | 0.04190766 [0.03835583, 0.04577201] | 0.27059405 [0.26176203, 0.28056192] |
| independent/shuffle_use_time_position **INIT** | 6 | 0.000 [0.000, 0.000] | 0.04424918 [0.04022470, 0.04868349] | 0.28360950 [0.27192577, 0.29710032] |
| set/local_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/local_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| set/local_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/local_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| set/use_time_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/use_time_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| set/use_time_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/use_time_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| set/future_attention | 3 | 100.000 [100.000, 100.000] | 0.02585377 [0.02268676, 0.02944421] | 0.34685118 [0.32821434, 0.36630436] |
| set/future_attention | 6 | 100.000 [100.000, 100.000] | 0.03102850 [0.02741882, 0.03509687] | 0.38717949 [0.36410631, 0.41283209] |
| set/shuffle_local_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/shuffle_local_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| set/shuffle_local_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/shuffle_local_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| set/shuffle_use_time_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/shuffle_use_time_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| set/shuffle_use_time_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.04139252 [0.03798386, 0.04520868] | 0.26610840 [0.25714011, 0.27601649] |
| set/shuffle_use_time_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.04398611 [0.04009784, 0.04827699] | 0.28294479 [0.27137635, 0.29629628] |
| 参照/last_visible | 3 | 100.000 [100.000, 100.000] | 0.02921170 [0.02604360, 0.03277738] | 0.33395575 [0.31772962, 0.35107652] |
| 参照/last_visible | 6 | 100.000 [100.000, 100.000] | 0.03382171 [0.03032666, 0.03760882] | 0.37014249 [0.34969096, 0.39280120] |
| 参照/b | 3 | 100.000 [100.000, 100.000] | 0.01763906 [0.01524359, 0.02014464] | 0.33516119 [0.31541262, 0.35505115] |
| 参照/b | 6 | 100.000 [100.000, 100.000] | 0.02181453 [0.01897905, 0.02471951] | 0.37246110 [0.34915977, 0.39784230] |
| 参照/posthoc_best | 3 | 52.604 [45.833, 59.896] | 0.00450824 [0.00370655, 0.00548251] | 0.27852603 [0.26302341, 0.29485930] |
| 参照/posthoc_best | 6 | 52.604 [45.833, 59.896] | 0.00506205 [0.00425110, 0.00597218] | 0.30553522 [0.28634330, 0.32634326] |

四标签对全部对手：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | fifo | 3 | 0.521 [0.000, 1.562] | 0.00075510 [-0.00046706, 0.00194757] | -0.00074059 [-0.00513218, 0.00335413] | 否 |
| local_feature | fifo | 6 | 0.521 [0.000, 1.562] | 0.00031968 [-0.00100638, 0.00151420] | -0.00450127 [-0.01004958, 0.00071380] | 否 |
| local_feature | reservoir | 3 | 0.521 [0.000, 1.562] | 0.00083891 [-0.00029135, 0.00205744] | 0.00536288 [0.00113957, 0.00990057] | 否 |
| local_feature | reservoir | 6 | 0.521 [0.000, 1.562] | 0.00077750 [-0.00038590, 0.00192630] | 0.00548471 [0.00044475, 0.01059396] | 否 |
| local_feature | temporal_buckets | 3 | 0.521 [0.000, 1.562] | -0.00002592 [-0.00102221, 0.00100349] | -0.00394550 [-0.00777790, -0.00034953] | 否 |
| local_feature | temporal_buckets | 6 | 0.521 [0.000, 1.562] | -0.00044479 [-0.00153920, 0.00059326] | -0.00845132 [-0.01299855, -0.00404270] | 否 |
| local_feature | diversity | 3 | -3.125 [-6.771, -0.521] | 0.00011951 [-0.00238688, 0.00260251] | 0.02535233 [0.01870023, 0.03192254] | 否 |
| local_feature | diversity | 6 | -3.125 [-6.771, -0.521] | 0.00202559 [-0.00056325, 0.00460060] | 0.02772271 [0.02001640, 0.03555562] | 否 |
| local_feature | surprise | 3 | 0.521 [0.000, 1.562] | 0.00247875 [0.00087532, 0.00416264] | 0.02028958 [0.01398121, 0.02644801] | 否 |
| local_feature | surprise | 6 | 0.521 [0.000, 1.562] | 0.00238601 [0.00064226, 0.00415406] | 0.02110454 [0.01349513, 0.02921472] | 否 |
| local_feature | attention | 3 | -99.479 [-100.000, -98.438] | -0.01548149 [-0.01947869, -0.01143132] | 0.08156218 [0.06635510, 0.09637490] | 否 |
| local_feature | attention | 6 | -99.479 [-100.000, -98.438] | -0.01376576 [-0.01830018, -0.00922336] | 0.10273161 [0.08517116, 0.12216759] | 否 |
| local_feature | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00326876 [0.00185108, 0.00472139] | 0.03953946 [0.03153928, 0.04751879] | 否 |
| local_feature | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00357995 [0.00191724, 0.00526949] | 0.04103653 [0.03121329, 0.05081508] | 否 |
| local_feature | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01553875 [-0.02003808, -0.01092389] | 0.08074278 [0.06505843, 0.09664997] | 否 |
| local_feature | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01295760 [-0.01808173, -0.00774904] | 0.10423470 [0.08532045, 0.12368549] | 否 |
| local_position | fifo | 3 | 0.521 [0.000, 1.562] | 0.00075510 [-0.00046706, 0.00194757] | -0.00074059 [-0.00513218, 0.00335413] | 否 |
| local_position | fifo | 6 | 0.521 [0.000, 1.562] | 0.00031968 [-0.00100638, 0.00151420] | -0.00450127 [-0.01004958, 0.00071380] | 否 |
| local_position | reservoir | 3 | 0.521 [0.000, 1.562] | 0.00083891 [-0.00029135, 0.00205744] | 0.00536288 [0.00113957, 0.00990057] | 否 |
| local_position | reservoir | 6 | 0.521 [0.000, 1.562] | 0.00077750 [-0.00038590, 0.00192630] | 0.00548471 [0.00044475, 0.01059396] | 否 |
| local_position | temporal_buckets | 3 | 0.521 [0.000, 1.562] | -0.00002592 [-0.00102221, 0.00100349] | -0.00394550 [-0.00777790, -0.00034953] | 否 |
| local_position | temporal_buckets | 6 | 0.521 [0.000, 1.562] | -0.00044479 [-0.00153920, 0.00059326] | -0.00845132 [-0.01299855, -0.00404270] | 否 |
| local_position | diversity | 3 | -3.125 [-6.771, -0.521] | 0.00011951 [-0.00238688, 0.00260251] | 0.02535233 [0.01870023, 0.03192254] | 否 |
| local_position | diversity | 6 | -3.125 [-6.771, -0.521] | 0.00202559 [-0.00056325, 0.00460060] | 0.02772271 [0.02001640, 0.03555562] | 否 |
| local_position | surprise | 3 | 0.521 [0.000, 1.562] | 0.00247875 [0.00087532, 0.00416264] | 0.02028958 [0.01398121, 0.02644801] | 否 |
| local_position | surprise | 6 | 0.521 [0.000, 1.562] | 0.00238601 [0.00064226, 0.00415406] | 0.02110454 [0.01349513, 0.02921472] | 否 |
| local_position | attention | 3 | -99.479 [-100.000, -98.438] | -0.01548149 [-0.01947869, -0.01143132] | 0.08156218 [0.06635510, 0.09637490] | 否 |
| local_position | attention | 6 | -99.479 [-100.000, -98.438] | -0.01376576 [-0.01830018, -0.00922336] | 0.10273161 [0.08517116, 0.12216759] | 否 |
| local_position | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00326876 [0.00185108, 0.00472139] | 0.03953946 [0.03153928, 0.04751879] | 否 |
| local_position | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00357995 [0.00191724, 0.00526949] | 0.04103653 [0.03121329, 0.05081508] | 否 |
| local_position | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01553875 [-0.02003808, -0.01092389] | 0.08074278 [0.06505843, 0.09664997] | 否 |
| local_position | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01295760 [-0.01808173, -0.00774904] | 0.10423470 [0.08532045, 0.12368549] | 否 |
| use_time_feature | fifo | 3 | 0.521 [0.000, 1.562] | 0.00075510 [-0.00046706, 0.00194757] | -0.00074059 [-0.00513218, 0.00335413] | 否 |
| use_time_feature | fifo | 6 | 0.521 [0.000, 1.562] | 0.00031968 [-0.00100638, 0.00151420] | -0.00450127 [-0.01004958, 0.00071380] | 否 |
| use_time_feature | reservoir | 3 | 0.521 [0.000, 1.562] | 0.00083891 [-0.00029135, 0.00205744] | 0.00536288 [0.00113957, 0.00990057] | 否 |
| use_time_feature | reservoir | 6 | 0.521 [0.000, 1.562] | 0.00077750 [-0.00038590, 0.00192630] | 0.00548471 [0.00044475, 0.01059396] | 否 |
| use_time_feature | temporal_buckets | 3 | 0.521 [0.000, 1.562] | -0.00002592 [-0.00102221, 0.00100349] | -0.00394550 [-0.00777790, -0.00034953] | 否 |
| use_time_feature | temporal_buckets | 6 | 0.521 [0.000, 1.562] | -0.00044479 [-0.00153920, 0.00059326] | -0.00845132 [-0.01299855, -0.00404270] | 否 |
| use_time_feature | diversity | 3 | -3.125 [-6.771, -0.521] | 0.00011951 [-0.00238688, 0.00260251] | 0.02535233 [0.01870023, 0.03192254] | 否 |
| use_time_feature | diversity | 6 | -3.125 [-6.771, -0.521] | 0.00202559 [-0.00056325, 0.00460060] | 0.02772271 [0.02001640, 0.03555562] | 否 |
| use_time_feature | surprise | 3 | 0.521 [0.000, 1.562] | 0.00247875 [0.00087532, 0.00416264] | 0.02028958 [0.01398121, 0.02644801] | 否 |
| use_time_feature | surprise | 6 | 0.521 [0.000, 1.562] | 0.00238601 [0.00064226, 0.00415406] | 0.02110454 [0.01349513, 0.02921472] | 否 |
| use_time_feature | attention | 3 | -99.479 [-100.000, -98.438] | -0.01548149 [-0.01947869, -0.01143132] | 0.08156218 [0.06635510, 0.09637490] | 否 |
| use_time_feature | attention | 6 | -99.479 [-100.000, -98.438] | -0.01376576 [-0.01830018, -0.00922336] | 0.10273161 [0.08517116, 0.12216759] | 否 |
| use_time_feature | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00326876 [0.00185108, 0.00472139] | 0.03953946 [0.03153928, 0.04751879] | 否 |
| use_time_feature | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00357995 [0.00191724, 0.00526949] | 0.04103653 [0.03121329, 0.05081508] | 否 |
| use_time_feature | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01553875 [-0.02003808, -0.01092389] | 0.08074278 [0.06505843, 0.09664997] | 否 |
| use_time_feature | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01295760 [-0.01808173, -0.00774904] | 0.10423470 [0.08532045, 0.12368549] | 否 |
| use_time_position | fifo | 3 | 0.521 [0.000, 1.562] | 0.00075510 [-0.00046706, 0.00194757] | -0.00074059 [-0.00513218, 0.00335413] | 否 |
| use_time_position | fifo | 6 | 0.521 [0.000, 1.562] | 0.00031968 [-0.00100638, 0.00151420] | -0.00450127 [-0.01004958, 0.00071380] | 否 |
| use_time_position | reservoir | 3 | 0.521 [0.000, 1.562] | 0.00083891 [-0.00029135, 0.00205744] | 0.00536288 [0.00113957, 0.00990057] | 否 |
| use_time_position | reservoir | 6 | 0.521 [0.000, 1.562] | 0.00077750 [-0.00038590, 0.00192630] | 0.00548471 [0.00044475, 0.01059396] | 否 |
| use_time_position | temporal_buckets | 3 | 0.521 [0.000, 1.562] | -0.00002592 [-0.00102221, 0.00100349] | -0.00394550 [-0.00777790, -0.00034953] | 否 |
| use_time_position | temporal_buckets | 6 | 0.521 [0.000, 1.562] | -0.00044479 [-0.00153920, 0.00059326] | -0.00845132 [-0.01299855, -0.00404270] | 否 |
| use_time_position | diversity | 3 | -3.125 [-6.771, -0.521] | 0.00011951 [-0.00238688, 0.00260251] | 0.02535233 [0.01870023, 0.03192254] | 否 |
| use_time_position | diversity | 6 | -3.125 [-6.771, -0.521] | 0.00202559 [-0.00056325, 0.00460060] | 0.02772271 [0.02001640, 0.03555562] | 否 |
| use_time_position | surprise | 3 | 0.521 [0.000, 1.562] | 0.00247875 [0.00087532, 0.00416264] | 0.02028958 [0.01398121, 0.02644801] | 否 |
| use_time_position | surprise | 6 | 0.521 [0.000, 1.562] | 0.00238601 [0.00064226, 0.00415406] | 0.02110454 [0.01349513, 0.02921472] | 否 |
| use_time_position | attention | 3 | -99.479 [-100.000, -98.438] | -0.01548149 [-0.01947869, -0.01143132] | 0.08156218 [0.06635510, 0.09637490] | 否 |
| use_time_position | attention | 6 | -99.479 [-100.000, -98.438] | -0.01376576 [-0.01830018, -0.00922336] | 0.10273161 [0.08517116, 0.12216759] | 否 |
| use_time_position | keep_first | 3 | 0.521 [0.000, 1.562] | 0.00326876 [0.00185108, 0.00472139] | 0.03953946 [0.03153928, 0.04751879] | 否 |
| use_time_position | keep_first | 6 | 0.521 [0.000, 1.562] | 0.00357995 [0.00191724, 0.00526949] | 0.04103653 [0.03121329, 0.05081508] | 否 |
| use_time_position | set/future_attention | 3 | -99.479 [-100.000, -98.438] | -0.01553875 [-0.02003808, -0.01092389] | 0.08074278 [0.06505843, 0.09664997] | 否 |
| use_time_position | set/future_attention | 6 | -99.479 [-100.000, -98.438] | -0.01295760 [-0.01808173, -0.00774904] | 0.10423470 [0.08532045, 0.12368549] | 否 |

同标签集合/独立：

| 集合候选标签 | 对手 | h | 成功率差 pp [95% CI] | position 改善 [95% CI] | feature 改善 [95% CI] | 该 h 数值联合下界>0 |
| --- | --- | --- | --- | --- | --- | --- |
| local_feature | independent/local_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.00051513 [-0.00070995, 0.00174466] | 0.00448565 [-0.00001511, 0.00926288] | 否 |
| local_feature | independent/local_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.00026307 [-0.00105361, 0.00158370] | 0.00066471 [-0.00454380, 0.00574664] | 否 |
| local_position | independent/local_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.00051513 [-0.00070995, 0.00174466] | 0.00448565 [-0.00001511, 0.00926288] | 否 |
| local_position | independent/local_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.00026307 [-0.00105361, 0.00158370] | 0.00066471 [-0.00454380, 0.00574664] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 3 | 0.521 [0.000, 1.562] | 0.00051513 [-0.00070995, 0.00174466] | 0.00448565 [-0.00001511, 0.00926288] | 否 |
| use_time_feature | independent/use_time_feature **INIT** | 6 | 0.521 [0.000, 1.562] | 0.00026307 [-0.00105361, 0.00158370] | 0.00066471 [-0.00454380, 0.00574664] | 否 |
| use_time_position | independent/use_time_position **INIT** | 3 | 0.521 [0.000, 1.562] | 0.00051513 [-0.00070995, 0.00174466] | 0.00448565 [-0.00001511, 0.00926288] | 否 |
| use_time_position | independent/use_time_position **INIT** | 6 | 0.521 [0.000, 1.562] | 0.00026307 [-0.00105361, 0.00158370] | 0.00066471 [-0.00454380, 0.00574664] | 否 |

## 5. 原始机器可读入口
- [diagnostics/ordinary完整summary](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/diagnostics/ordinary/summary.json)
- [online/ordinary完整summary](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/online/ordinary/summary.json)
- [diagnostics/long_gap完整summary](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/diagnostics/long_gap/summary.json)
- [online/long_gap完整summary](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/online/long_gap/summary.json)
- [54臂summary及checkpoint/hash](../results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu/students/summary.json)
