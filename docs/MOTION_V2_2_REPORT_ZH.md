# Motion v2.2 进展报告（2026-10-31）

**当前结果：第 0′ 步重判、Phase 1 headroom 复现、第 1 步真实生成器审计均通过；第 2 步协议已提交推送，9 项统计软件检查通过，GPU 执行停在新的 HPC 认证请求，尚无预测器结论。**

工作继续在 PR #3、分支 `experiments/motion-v2-20261030`。本报告是 v2.2 的最新进展入口；[v2.2 修订](MOTION_V2_2_AMENDMENT_ZH.md)说明新的判定规则和执行范围。原 v2 停止报告、v2.1 停止记录、原协议、原 JSON、脚本和哈希锁全部保留。旧 `status.json` 不改写；本轮结果另存于 `results/motion_v2/v2_2/`。

## 第 0′ 步：对已保存结果重判

依据先行提交的 v2.2 修订和 [step0 protocol](../experiments/motion_v2/protocol_step0_v2_2.json)，仅重新判断此前保存的 v2.1 输出，没有重跑参考分类器、重新采样场景或执行用户提供的外部复算。

| 分类器估计项 | 附件参考 | 已保存本地结果 | 本轮判定 |
|---|---:|---:|---|
| v1：单张非终点新段帧 | 0.624 | 0.620 | 差 0.004，且都在 0.58 以上，通过 |
| v2.1：单张非终点新段帧 | 0.538 | 0.545 | 差 0.007，且都在 0.58 以下，通过 |
| v1：两张新段帧 | 0.996 | 0.996 | 相同，且 ≥0.95，通过 |
| v2.1：两张新段帧 | 0.993 | 0.993 | 相同，且 ≥0.95，通过 |

四组配对审计均为 500/500，全部在线策略数字与附件参考完全一致。`end_only` 完全一致；v2.1 的 `stale_plus_end` 为 0.5，v1 的不适用项为 `null`，也完全一致。分类器估计项满足预先修订的“绝对差 ≤0.02 且与参考位于阈值同侧”规则。

这是对**同一份已保存测量**的重判，不是一次新的科学复现。v2.1 原先因逐数值不一致而停止的记录仍然存在。用户提供的 Linux 版本复算及多 seed 波动说明已在修订文档中标明外部来源，本报告不把它们列为 Codex 新运行的证据。

证据：[decision.json](../results/motion_v2/v2_2/step0/rejudge/decision.json)、[status.json](../results/motion_v2/v2_2/step0/rejudge/status.json)、[SHA256SUMS](../results/motion_v2/v2_2/step0/rejudge/SHA256SUMS)。执行前提交为 `908c6dfec143f82b69daf4dcd8ba2d8c9b1196cb`。

## Phase 1 headroom：公开代码重建一致，原 manifest 尚未逐对核查

随后只执行了一次此前尚未运行的 `motion_headroom.py`；解析后的完整 JSON 与附件 `motion_v1_pool_headroom.json` **完全一致**。

| Phase 1 split | 三帧池数量 | reject-newest 达到信息最优 | 最优保留含至少两张可见帧 |
|---|---:|---:|---:|
| test_id | 1,536 | 100% | 60.87% |
| test_long_gap | 1,536 | 100% | 18.42% |
| test_distractor_shift | 1,536 | 100% | 62.63% |

**这些比例按公开代码重建，未与原 manifest 逐对核对。** 本次重建的是原时间结构和候选池统计，不是读取原 test 图像、编码特征或生成新 test 数据集。不能将 JSON 一致扩大解释为原 seed 9137 实际执行记录已核实。

原 manifest 回溯核查按 v2.2 修订改为非阻塞事项。需要账号 `xl4358` 的持有人复制以下三个文件后，才能核对实际 pair ID、query 和 gap：

- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/dataset/manifest.json`
- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/locked_test_generation/dataset/manifest.json`
- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/confirmation_test_generation_complete.json`

本轮没有为取得这些文件提交新的认证或发起 SSH，没有访问另一账号目录，也没有代发文件请求。对当前用户既有门户会话的只读检查见第 2 步状态。

证据：[comparison.json](../results/motion_v2/v2_2/step0/phase1_headroom/comparison.json)、[stdout.json](../results/motion_v2/v2_2/step0/phase1_headroom/stdout.json)、[status.json](../results/motion_v2/v2_2/step0/phase1_headroom/status.json)、[SHA256SUMS](../results/motion_v2/v2_2/step0/phase1_headroom/SHA256SUMS)。

## 第 1 步：真实 v2.1 生成器与 CPU 审计通过

在 [step1 protocol](../experiments/motion_v2/protocol_step1_v2_2.json) 和源码哈希提交后，仅生成一次 seed **20261030** 的 train **512 对**、validation **96 对**，共 1,216 episode、39,398 帧。没有生成或打开 test。

新生成器使用共同旧段、上方车道 y=0.40；在 `t_c` 踢到下方车道 y=0.62；新段围绕共享 `x_end` 反向运动。隐藏状态、可见性及时间标记隔离在 `evaluation_only`。原 Phase 1、Gate 源码和既有结果未修改。

配对结构审计 **608/608 通过**，检查了共同旧段、最近四帧、动作、终点位置、允许的帧差异范围及可见帧约束。登记的 pytest **7 项全部通过**，生成和审计各执行一次。

真实 train 数据的隐藏状态分类检查使用 512 个场景、按 sibling pair 分组的 5 折交叉验证。下表是已保存的折均值，没有用结果重新调整阈值：

| 输入帧 | 实际准确率 | 登记门槛 | 判定 |
|---|---:|---|---|
| 只有终点帧 | 0.5000000000 | 0.47–0.53 | 通过 |
| 一张非终点新段帧 | 0.5253188654 | ≤0.58 | 通过 |
| 一张旧段帧 + 终点帧 | 0.5000000000 | 0.47–0.53 | 通过 |
| 两张新段帧 | 0.9756520084 | ≥0.95 | 通过 |

在线 B=2 的充分帧保留检查在 train、validation 分别得到相同结果：

| 策略／参照 | train（1,024 episode） | validation（192 episode） |
|---|---:|---:|
| FIFO | 0% | 0% |
| keep-first | 0% | 0% |
| 事后充分帧上限 | 100% | 100% |
| 最新两张可见帧结构参照 | 100% | 100% |

这里的在线成功仍定义为“最终两张历史帧均属于充分帧”，不是预测器精度。真实 train 审计与参考脚本的场景数量和采样条件不同，按登记阈值判定，不要求准确率逐位相等。

证据：[audit.json](../results/motion_v2/v2_2/step1/audit.json)、[pytest.stdout.log](../results/motion_v2/v2_2/step1/pytest.stdout.log)、[status.json](../results/motion_v2/v2_2/step1/status.json)、[SHA256SUMS](../results/motion_v2/v2_2/step1/SHA256SUMS)。执行前提交为 `e9115c5abd37fd2a3cc60d9462a9e3e8ac2b0e96`。数据 manifest SHA256 为 `9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a`。

## 第 2 步：协议已冻结，GPU 执行停在认证请求

[第 2 步 protocol](MOTION_V2_2_STEP2_PROTOCOL_ZH.md)、[机器可读 protocol](../experiments/motion_v2/protocol_step2_v2_2.json)、[可执行参数](../experiments/motion_v2/config_step2_v2_2.json)及源码、输入哈希已于提交 `a36db1ec2c3ae10e2ed44a1c147f0e186886c1d9` 冻结并推送。

登记的软件预检 **9 项通过（pytest 报告 0.25 秒）**。这些检查只验证纯数组统计的配对聚合、差值方向、独立 horizon 门槛及非法输入拒绝，没有拟合预测器或 decoder，没有编码，也没有生成或读取 test episode。证据：[stdout.log](../results/motion_v2/v2_2/step2_software/stdout.log)、[status.json](../results/motion_v2/v2_2/step2_software/status.json)、[SHA256SUMS](../results/motion_v2/v2_2/step2_software/SHA256SUMS)。

执行方案冻结官方 I-JEPA、四格池化 5120 维、train-only 归一化及 basis；沿用 Phase 1 `DynamicsPredictor`，seeds 17/29/43、各 120 epochs，共 23,040 optimizer steps。保留四类训练历史，并登记训练时非空 memory 的整段 dropout 概率 0.25。checkpoint 只用 validation 的固定 feature MSE 选择。位置 decoder 的拟合和 ridge 分组交叉验证都仅使用 train。

validation 的四组历史为：a 无 memory；b 为第一张新段可见帧加 end；c 为第一张与最后一张旧段可见帧；d 为最后一张旧段可见帧加 end。d 在 sibling 间完全相同，提供终点位置而不提供方向；它是本轮的主要对照。

主判据是 `position MSE(d) − position MSE(b)`：h=3、h=6 的配对 95% CI 下界都必须严格大于 0。次要必需判据是 `position MSE(c) − position MSE(b)` 的两个 horizon 下界也都严格大于 0。b 对 a 及 feature MSE 改善只报告。先在 pair 内平均两个 sibling，再平均三个固定 seed 的 loss，对 96 个 pair 做 2,000 次 bootstrap；不把 sibling 或模型 seed 视为独立场景。

第 2 步实验尚未启动。只读查看既有 Chrome 无痕窗口时，OOD Dashboard 初始显示 `Logged in as xy3257`；点击 `My Interactive Sessions` 进行正常导航后，页面重定向到 `device.login.microsoftonline.com` 认证页。此时已立即停止浏览器操作，未填写或提交认证、未发起 SSH、未写入集群、未提交作业，并已请用户本人完成认证。初始门户登录提示不能据此证明已有可用 A100 会话；目前尚未确认 GPU 可用性。

下表明确保留未运行状态。这是执行环境的认证阻塞，不是预测器统计门槛失败，也不存在可报告的统计通过结论。

| 条件 | horizon | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---|---:|---|---|---|---|
| a：recent-only | 3 | 未运行 | 未运行 | 未运行 | 未运行 |
| a：recent-only | 6 | 未运行 | 未运行 | 未运行 | 未运行 |
| b：两张新段帧 | 3 | 未运行 | 未运行 | 未运行 | 未运行 |
| b：两张新段帧 | 6 | 未运行 | 未运行 | 未运行 | 未运行 |
| c：两张旧段帧 | 3 | 未运行 | 未运行 | 未运行 | 未运行 |
| c：两张旧段帧 | 6 | 未运行 | 未运行 | 未运行 | 未运行 |
| d：end + 旧段帧 | 3 | 未运行 | 未运行 | 未运行 | 未运行 |
| d：end + 旧段帧 | 6 | 未运行 | 未运行 | 未运行 | 未运行 |

d−b、c−b 的两个 horizon 门槛目前均为**未评估**。第 2 步将无论通过或失败都停止，不改阈值重跑，不训练保留策略，不进入第 3 步。

## GPU 用时与证据边界

第 2 步登记估时上限为 **2 A100-GPU 小时（7,200 秒）**，低于用户授权总上限 4 小时。按已生成 39,398 帧及仓库历史速率，编码约 1,451 秒、360 epochs 约 2,062 秒，其他工作预留 3,600 秒，合计约 7,113 秒。以上是历史记录支持的估计，不是本次 GPU 实测。实际分配用时或剩余工作预测超出 7,200 秒即停；GPU 分配期间的初始化、CPU 工作及失败时间也计入。

**本次实际 GPU 用时：0 A100-GPU 秒。** 已完成的重判、Phase 1 headroom 复现和第 1 步均为 CPU 工作。

最大的研究风险仍是：**隐藏状态下两帧能恢复方向，不代表冻结 I-JEPA 特征保留了足够信息，更不代表指定预算的预测器能够利用它。** 分组分类准确率和充分帧保留率支持继续做第 2 步，但不能替代 b 对 d 的位置误差实验。单帧分类检查也是给定分类器和场景分布下的经验结果，不证明所有模型都无法利用任何单帧线索。

另外，原 seed 9137 manifest 尚未逐对核查；Phase 1 headroom 比例保留公开代码重建的限定。第 2 步未来的 bootstrap 区间只覆盖三个固定训练 seed 平均性能的场景抽样不确定性，不宣称覆盖所有训练随机性。
