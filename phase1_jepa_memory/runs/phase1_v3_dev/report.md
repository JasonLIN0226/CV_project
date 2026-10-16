# 第一阶段开发对照

**DEVELOPMENT_REVISE** · 开发证据，不是独立 Phase 1 GO。

Gate 只删除 ranking；motion 只改变 privileged B=2 时间选择。对照也按共同 validation MSE 重新训练。

- gate: REJECTED
  - Candidate original task rule: gate/test_distractor_shift/h3/seed17 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h3/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h3/seed29 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
  - Candidate original task rule: gate/test_distractor_shift/h3/seed43 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h3/seed43 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
  - Candidate original task rule: gate/test_distractor_shift/h3/seedall_seeds_mean 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h3/seedall_seeds_mean full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
  - Candidate original task rule: gate/test_distractor_shift/h6/seed17 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h6/seed17 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
  - Candidate original task rule: gate/test_distractor_shift/h6/seed29 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h6/seed29 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
  - Candidate original task rule: gate/test_distractor_shift/h6/seed43 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h6/seed43 full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
  - Candidate original task rule: gate/test_distractor_shift/h6/seedall_seeds_mean 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。
  - Candidate original task rule: gate/test_distractor_shift/h6/seedall_seeds_mean full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。
  - gate/test_distractor_shift/h3/seed17 Candidate did not meet original latent-benefit rule against reference recent predictions
  - gate/test_distractor_shift/h3/seed29 Candidate privileged absolute MSE did not reliably improve over matched reference
  - gate/test_distractor_shift/h3/seed29 Candidate did not meet original latent-benefit rule against reference recent predictions
  - gate/test_distractor_shift/h3/seed43 Candidate privileged absolute MSE did not reliably improve over matched reference
  - gate/test_distractor_shift/h3/seed43 Candidate did not meet original latent-benefit rule against reference recent predictions
  - gate/test_distractor_shift/h3/seedall_seeds_mean Candidate privileged absolute MSE did not reliably improve over matched reference
  - gate/test_distractor_shift/h3/seedall_seeds_mean Candidate did not meet original latent-benefit rule against reference recent predictions
  - gate/test_distractor_shift/h6/seed17 Candidate did not meet original latent-benefit rule against reference recent predictions
  - gate/test_distractor_shift/h6/seed29 Candidate privileged absolute MSE did not reliably improve over matched reference
  - gate/test_distractor_shift/h6/seed29 Candidate did not meet original latent-benefit rule against reference recent predictions
  - gate/test_distractor_shift/h6/seed43 Candidate privileged absolute MSE did not reliably improve over matched reference
  - gate/test_distractor_shift/h6/seed43 Candidate did not meet original latent-benefit rule against reference recent predictions
  - gate/test_distractor_shift/h6/seedall_seeds_mean Candidate privileged absolute MSE did not reliably improve over matched reference
  - gate/test_distractor_shift/h6/seedall_seeds_mean Candidate did not meet original latent-benefit rule against reference recent predictions
- motion: REJECTED
  - motion/test_long_gap/h3/seed17 Candidate privileged absolute MSE did not reliably improve over matched reference
  - motion/test_distractor_shift/h3/seed29 Candidate did not meet original latent-benefit rule against reference recent predictions
  - motion/test_distractor_shift/h3/seed43 Candidate did not meet original latent-benefit rule against reference recent predictions
  - motion/test_distractor_shift/h6/seed29 Candidate did not meet original latent-benefit rule against reference recent predictions
  - motion/test_distractor_shift/h6/seed43 Candidate did not meet original latent-benefit rule against reference recent predictions

只有两任务均 SUPPORTED，才允许 seed9137 新轨迹确认；不调阈值、种子或额外模块。
开发结果来自已揭示的 v2 数据，仅支持候选筛选。各任务实际曲线、原始评估与对照见子目录。
