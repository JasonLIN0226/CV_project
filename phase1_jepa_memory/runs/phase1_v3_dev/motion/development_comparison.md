# motion 单项开发比较

**REJECTED** — 仅决定候选是否进入独立确认，不构成 Phase 1 GO。

所有检查使用三种子各 120 轮训练、共同 validation MSE 选模、完整 pair 配对统计；逐种子与三种子平均均保留。

- motion/test_long_gap/h3/seed17 Candidate privileged absolute MSE did not reliably improve over matched reference
- motion/test_distractor_shift/h3/seed29 Candidate did not meet original latent-benefit rule against reference recent predictions
- motion/test_distractor_shift/h3/seed43 Candidate did not meet original latent-benefit rule against reference recent predictions
- motion/test_distractor_shift/h6/seed29 Candidate did not meet original latent-benefit rule against reference recent predictions
- motion/test_distractor_shift/h6/seed43 Candidate did not meet original latent-benefit rule against reference recent predictions

排序损失引起放大、或时间跨度改善速度精度，均为待检验的机制解释；性能比较本身不证明根因。

完整切片、置信区间和原始文件校验见 development_comparison.json。
