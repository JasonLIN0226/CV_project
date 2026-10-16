# V5 Gate 原规则开发评估

**REJECTED**；这是已曝光开发集上的候选采用判定，Phase1未通过。

三个当前种子均完成120轮，独立核验best/latest、RNG、优化器、调度、原train-only基底与实际源码后才运行评估。
新版因果Transformer加载器明确使用新类和真实新签名；未冒用旧DynamicsPredictor来源。
原任务规则：DEVELOPMENT_REVISE。固定参考recent检查通过 18/24；关键绝对MSE检查通过 8/8。

| seed | best epoch | ID h3/h6 privileged MSE | distractor h3/h6 privileged MSE |
|---|---:|---|---|
| 17 | 57 | 0.03915353/0.04555183 | 0.09472044/0.09600430 |
| 29 | 4 | 0.10371822/0.10408411 | 0.11821623/0.11816719 |
| 43 | 66 | 0.03943546/0.04664188 | 0.06982001/0.06788797 |

全部逐种子/pooled、OOD、错误证据方向、任务读出与配对区间保存在evaluation/metrics.json及development_comparison.json。
训练误差或验证误差下降不替代采用门槛；Motion、真实删除可行性与独立确认仍需分别完成。

未通过项目：
- gate/test_id/h3/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- gate/test_id/h3/seed29 内容替换干预未可靠支持历史使用。
- gate/test_id/h6/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- gate/test_id/h6/seed29 内容替换干预未可靠支持历史使用。
- gate/test_long_gap/h3/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- gate/test_long_gap/h3/seed29 内容替换干预未可靠支持历史使用。
- gate/test_long_gap/h6/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- gate/test_long_gap/h6/seed29 内容替换干预未可靠支持历史使用。
- gate/test_distractor_shift/h3/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- gate/test_distractor_shift/h3/seed29 内容替换干预未可靠支持历史使用。
- gate/test_distractor_shift/h6/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- gate/test_distractor_shift/h6/seed29 内容替换干预未可靠支持历史使用。
- gate/test_id/h3/seed29 candidate_vs_reference_recent
- gate/test_id/h6/seed29 candidate_vs_reference_recent
- gate/test_long_gap/h3/seed29 candidate_vs_reference_recent
- gate/test_long_gap/h6/seed29 candidate_vs_reference_recent
- gate/test_distractor_shift/h3/seed29 candidate_vs_reference_recent
- gate/test_distractor_shift/h6/seed29 candidate_vs_reference_recent
