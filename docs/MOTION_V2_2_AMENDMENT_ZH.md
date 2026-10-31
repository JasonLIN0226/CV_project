# Motion v2.2 修订登记（2026-10-31）

本修订在任何本轮重判或实验计算之前提交并推送。沿用 PR #3、分支 `experiments/motion-v2-20261030`；保留原停止记录、原 protocol、原 JSON 和原脚本，不删除本地或集群文件，不合并 main。

## 1. 修订依据与证据来源

v2.1 第 0′ 步的停止来自逐数值完全相同的比较规则过严，不应将该停止直接解释为科学复现失败。以下为用户提供的 Claude 外部复算记录，作为此次修订依据；Codex 未独立执行这些外部复算，也不为本修订重跑它们：

- Linux x86_64、Python 3、scikit-learn 1.9.1、NumPy 2.5.3，使用未改动的 `motion_v2_1_checks.py`，得到与参考 JSON 完全一致的 leakage：v1 单帧 0.624，v2.1 单帧 0.538。
- macOS arm64、scikit-learn 1.7.2 的已保存本地结果为 0.620 / 0.545。配对审计、在线策略和两帧准确率两边逐位相同。用户据此将差异归于分类器实现的版本或架构差异，而不是场景生成变化；这里记录该外部解释，不将其包装为本次另做的因果实验。
- 同一分类器只改变数据 seed：v1 单帧 0.607 ± 0.010（seed 8–17）；v2.1 单帧 0.541 ± 0.011（seed 8–13，最大 0.561）。原来的 0.4 / 0.7 个百分点差异在一个标准差以内。
- `end_only` 与 v2.1 `stale_plus_end` 恰为 0.5 是结构性结果：两个 sibling 的输入完全相同，且标签成对平衡。v1 没有 stale 段，原值为 null。

## 2. 第 0′ 步新判据

仅对已保存的 `results/motion_v2/step0_v2_1/v2_1_reference/stdout.json` 重判，不重跑 `motion_v2_1_checks.py`。

1. 四组配对审计、所有在线策略数字、`end_only`、`stale_plus_end` 必须与参考完全一致（包括 v1 不适用项的 null）。
2. 分类器估计项 `one_post_non_end`、`two_post`：与参考的绝对差不超过 0.02，且分别与参考处于 0.58 / 0.95 阈值的同侧。v2.1 必须满足单帧 ≤ 0.58、两帧 ≥ 0.95。v1 是对照，其单帧参考本来高于 0.58，因此“同侧”不要求 v1 变成低于 0.58。
3. 所有新判定、输入哈希和状态另存，不改写旧 comparison/status/JSON。新结果不能伪装成重新采样或重新训练的测量。
4. 新判定通过后，按原顺序执行一次未运行的 `motion_headroom.py`。该脚本没有随机分类器，其解析后 JSON 仍必须完全等于 `motion_v1_pool_headroom.json`；不同即停，不改参数或重跑。
5. 执行脚本与源码哈希另以本轮 step protocol 登记并提交后运行，旧 runner 的停止规则不修改。

## 3. 原 manifest 为非阻塞的回溯核查

第 1/2 步使用新的 v2.1 数据，不依赖 Phase 1 manifest。取得原 manifest 前，原协议第 1 节的 60.9% / 18.4% / 62.6% 等比例均应解释为：**按公开代码重建，未与原 manifest 逐对核对**。为保留原协议字节不修改原文件；本修订及新报告明确附上该限定。

不再尝试登录 OOD/SSH，也不尝试访问另一个账号的目录。需要账号 **xl4358** 的持有人将以下三个文件复制出来后，才可完成回溯核对；Codex 不代发请求：

- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/dataset/manifest.json`
- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/locked_test_generation/dataset/manifest.json`
- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/confirmation_test_generation_complete.json`

## 4. 第 1 步：真实 v2.1 生成器与 CPU 审计

新代码仅位于 `experiments/motion_v2/`；实现独立的 `motion_v2`，不修改 Phase 1 envs.py。复用画面风格、96×96、动作接口、R=4、B=2、h=3/6 和配对 ID 方式。运动按 v2.1：旧段在 y=0.40、两样本逐帧一致；t_c 踢到 y=0.62；新段以共享 x_end 锚定、最终速度方向相反，终点位置一致。保持登记的时间范围、遮挡保证及逐帧移动的场外干扰物。

仅生成新 seed 20261030 的 train 512 对、validation 96 对；不生成或打开 test。位置、速度、可见性、t_c 和充分帧标签只写 evaluation_only，不能作为模型输入。独立入口必须拒绝 test split。

先完成并提交包含源码哈希的第 1 步 protocol，再运行生成与 pytest。pytest 至少检查配对旧帧、最近四帧、动作及 end 位置一致，差异仅出现在允许的历史可见新段，所有可见帧约束与在线充分帧上限成立。对真实 train 数据做按场景分组的隐藏状态分类检查：

- 配对审计全部通过；
- `end_only`、`stale_plus_end` 均为 0.50 ± 0.03；
- 单张非终点新段帧准确率 ≤ 0.58；
- 两张新段帧准确率 ≥ 0.95。

按阈值判定，不要求与参考数逐位相同。记录环境版本；任何一项不过即停，不进入第 2 步，不看结果后调整阈值或重跑。

## 5. 第 2 步：加入位置已知而方向未知的对照

第 1 步通过后，先准备完整第 2 步 protocol 与 GPU 估时，提交并推送后才能开始任何第 2 步训练。冻结官方 I-JEPA、四格池化 5120 维、train-only 归一化和 basis；沿用 Phase 1 预测器架构与训练预算，seeds 17/29/43，含 memory dropout 和多样历史子集。checkpoint 仅由 validation 选择。位置 decoder 仅在 train 上拟合并冻结。

validation 的四组历史为：

| 组 | 历史 | 可获得的信息 |
|---|---|---|
| a | recent-only，无 memory | 最近四帧均被挡 |
| b | 两张 t_c 之后的可见帧 | 新段位置与运动方向 |
| c | 两张旧段可见帧 | 两 sibling 相同的无用历史 |
| d | end 帧 + 一张旧段可见帧 | 有终点位置，没有方向；两 sibling 输入完全相同 |

主判据：以 paired bootstrap 给出 `loss(d) − loss(b)` 的 95% CI，**h=3 和 h=6 的解码位置误差改善下界分别都必须 > 0**。同时报告特征 MSE 及 CI。次要报告 b 对 a、b 对 c，仍要求 b 优于 c。确切误差定义、bootstrap 单位、seed 汇总和 c 比较的执行规则在第 2 步 protocol 中于训练前写明，不事后选择。

总预算上限 4 A100-GPU 小时；运行前估时必须 ≤4 小时，实际超出登记估计即停。需要 HPC 时若遇登录或认证，由用户本人处理；当前不得自行尝试 OOD/SSH 登录。第 2 步无论通过或失败都停，不训练任何保留策略，不进入第 3 步，不打开或生成 test。

## 6. 记录与交付

每步先提交 protocol 再运行；已有记录和锁不改写。新状态、重判、计算结果和 SHA256 另存于 `results/motion_v2/v2_2/`；新报告另存于 `docs/MOTION_V2_2_REPORT_ZH.md`，PR #3 链接指向最新报告，原停止报告保留。

交付包括本修订、重判、Phase 1 headroom 比较、真实生成器 pytest/审计、第 2 步 protocol 与 a/b/c/d × h=3/6 的结果表（如触发停止条件则明确未运行），以及每个门槛的状态和实际 GPU 用时。
