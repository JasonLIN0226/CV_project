# V5 Motion 原规则开发评估

**REJECTED**；这是已曝光开发集上的候选采用判定，Phase1未通过。

三个当前种子均完成120轮，独立核验best/latest、RNG、优化器、调度、原train-only基底与实际源码后才运行评估。
新版因果Transformer加载器明确使用新类和真实新签名；未冒用旧DynamicsPredictor来源。
原任务规则：DEVELOPMENT_REVISE。固定参考recent检查通过 11/24；关键绝对MSE检查通过 0/8。

| seed | best epoch | ID h3/h6 privileged MSE | long-gap h3/h6 privileged MSE |
|---|---:|---|---|
| 17 | 79 | 0.11238839/0.11776017 | 0.19097500/0.19688656 |
| 29 | 119 | 0.07887772/0.08048296 | 0.16129724/0.16439634 |
| 43 | 99 | 0.13030218/0.13744536 | 0.21122417/0.21745287 |

全部逐种子/pooled、OOD、错误证据方向、任务读出与配对区间保存在evaluation/metrics.json及development_comparison.json。
训练误差或验证误差下降不替代采用门槛；Gate、真实删除可行性与独立确认仍需分别完成。

未通过项目：
- motion/test_long_gap/h3/seed17 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_long_gap/h3/seed43 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_long_gap/h3/seed43 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
- motion/test_long_gap/h3/seedall_seeds_mean 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_long_gap/h6/seed17 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_long_gap/h6/seed43 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_long_gap/h6/seed43 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
- motion/test_long_gap/h6/seedall_seeds_mean 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_distractor_shift/h3/seed17 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_distractor_shift/h3/seed17 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
- motion/test_distractor_shift/h3/seed43 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_distractor_shift/h3/seedall_seeds_mean 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_distractor_shift/h6/seed17 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_distractor_shift/h6/seed17 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
- motion/test_distractor_shift/h6/seed43 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_distractor_shift/h6/seedall_seeds_mean 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
- motion/test_long_gap/h3/seed29 区间跨过推进阈值，增加样本／训练种子前不能定论。
- motion/test_long_gap/h6/seed29 区间跨过推进阈值，增加样本／训练种子前不能定论。
- motion/test_id/h6/seed43 candidate_vs_reference_recent
- motion/test_long_gap/h3/seed17 candidate_vs_reference_recent
- motion/test_long_gap/h3/seed17 critical_absolute_mse_reduction
- motion/test_long_gap/h3/seed29 critical_absolute_mse_reduction
- motion/test_long_gap/h3/seed43 candidate_vs_reference_recent
- motion/test_long_gap/h3/seed43 critical_absolute_mse_reduction
- motion/test_long_gap/h3/seedall_seeds_mean candidate_vs_reference_recent
- motion/test_long_gap/h3/seedall_seeds_mean critical_absolute_mse_reduction
- motion/test_long_gap/h6/seed17 candidate_vs_reference_recent
- motion/test_long_gap/h6/seed17 critical_absolute_mse_reduction
- motion/test_long_gap/h6/seed29 critical_absolute_mse_reduction
- motion/test_long_gap/h6/seed43 candidate_vs_reference_recent
- motion/test_long_gap/h6/seed43 critical_absolute_mse_reduction
- motion/test_long_gap/h6/seedall_seeds_mean candidate_vs_reference_recent
- motion/test_long_gap/h6/seedall_seeds_mean critical_absolute_mse_reduction
- motion/test_distractor_shift/h3/seed17 candidate_vs_reference_recent
- motion/test_distractor_shift/h3/seed43 candidate_vs_reference_recent
- motion/test_distractor_shift/h3/seedall_seeds_mean candidate_vs_reference_recent
- motion/test_distractor_shift/h6/seed17 candidate_vs_reference_recent
- motion/test_distractor_shift/h6/seed43 candidate_vs_reference_recent
- motion/test_distractor_shift/h6/seedall_seeds_mean candidate_vs_reference_recent
