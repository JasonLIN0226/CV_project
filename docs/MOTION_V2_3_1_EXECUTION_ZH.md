# Motion v2.3.1 最终执行报告

**本次第 2 步没有产生可评估的科学结果，现已停止。** 唯一 A100 作业 **16550** 的终态为 `CANCELLED by 0`，实际分配 **769 A100 秒（0.213611 小时）**。科学输出目录不存在，wrapper 调用入口前应保存的三份文件及该作业 stdout/stderr 均缺失；没有执行到科学入口的证据，也不能判断预测器通过或未通过。取消记录标注 UID 0，具体原因未知。

原生 `ExitCode=0:0` 不构成科学成功证据；结算 `status=passed` 仅表示原生计费核验完成。下文的 N/A 表示未运行／没有可验证的测量，不能替换为 0，也不是统计判据失败。

## 已完成的登记、软件检查和部署

作业源码固定于预先推送的提交 `6dc5ed253dfa305b26be8f19d40edcd6a5878ea7`。本次沿用 [v2.3.1 修订](MOTION_V2_3_1_AMENDMENT_ZH.md)及[第 2 步协议](MOTION_V2_3_1_STEP2_PROTOCOL_ZH.md)：科学参数、数据、I-JEPA revision、三组 seed、训练步数、decoder、bootstrap 和六项必需判据均未改变；登记分配上限为 **10800 秒**，用户总上限仍为 **14400 秒**。结果文档提交不改变远端作业源码。

真实原生记录及人工边界的回放 **98/98 通过**。`NumNodes=1` 与 `1-1`、启动期有界等待、终态拒绝、Restarts、GPU TRES 及缺失／重复字段按登记测试；实际 CPU 多于请求不构成 GPU 判断失败。7 个科学函数及整个 BudgetGuard 类的 AST 与修订基线保持一致。见[回放报告](MOTION_V2_3_1_REPLAY_ZH.md)、[软件 status](../results/motion_v2/v2_3_1/software/attempt1/status.json)和[实现不变性记录](../results/motion_v2/v2_3_1/implementation_invariance.json)。这些结果只证明运行层检查通过，不证明模型完成了编码或训练。

新现场为 `/scratch/xy3257/motion-v2-20261031-deployment3`。预检核对完整 HEAD，既有 venv 的 import／版本检查通过；数据包及全部 **2432 个 NPZ** 的 SHA256 核验通过。只读复用了旧环境和已保存的 CPU B 工件，没有重装依赖或重新计算 B。见[预检回执](../results/motion_v2/v2_3_1/deployment3/remote_evidence/preflight/receipt.json)、[预检状态](../results/motion_v2/v2_3_1/deployment3/remote_evidence/preflight/status.json)和[解包回执](../results/motion_v2/v2_3_1/deployment3/remote_evidence/unpack_receipt.json)。

## 原生终态和实际 GPU 用时

| 原生字段 | 最终证据 |
|---|---|
| Job ID | 16550 |
| State | `CANCELLED by 0` |
| Start | `2026-10-31T22:07:12` |
| End | `2026-10-31T22:20:01` |
| ElapsedRaw | 769 秒 |
| GPU 分配 | 1 张；A100 分区 `c12m85-a100-1`，原生 `TresPerNode=gres/gpu:a100:1` |
| Restarts / Requeue | 0 / 1；Requeue 仅记录 |
| TimeLimit | `03:00:00` |
| ExitCode | `0:0`；不据此推断科学成功 |

Start/End 按原生 Slurm 字段原样报告。`1 × 769 = 769` A100 秒，即 **0.213611 小时**；此前科学作业实际分配为 0 秒，因此截至本报告累计也为 769 秒。该时间包括分配后的全部占用，不能记为模型训练用时。它未超过本轮 10800 秒登记上限或 14400 秒总上限。

证据：[最终 sacct](../results/motion_v2/v2_3_1/deployment3/remote_evidence/settlement/final_sacct.stdout.txt)、[最终 scontrol](../results/motion_v2/v2_3_1/deployment3/remote_evidence/settlement/final_scontrol.stdout.txt)、[结算回执](../results/motion_v2/v2_3_1/deployment3/remote_evidence/settlement/accounting.json)和[结算状态](../results/motion_v2/v2_3_1/deployment3/remote_evidence/settlement/status.json)。

[科学文件盘点](../results/motion_v2/v2_3_1/deployment3/remote_evidence/scientific_file_inventory.json)记录科学输出目录不存在，文件数为 0。[导出 manifest](../results/motion_v2/v2_3_1/deployment3/remote_evidence/export_manifest.json)列出缺失的 `scientific_job_started/scontrol_start.txt`、`started_at_utc.txt`、`git_head.txt`，以及 stdout/stderr、编码产物、checkpoint、decoder、metrics 和 runner status。这三份文件在登记 wrapper 中位于科学入口之前；缺失文件没有重建或补写。导出记录不能证明 shell 从未启动、启动标记文件夹在终态不存在，也不能定位具体停在哪一行。现有证据无法确定取消的具体原因；本轮不自动重提作业。

## 第 2 步结果表

| 条件 | horizon | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---|---:|---|---|---|---|
| a：recent-only | 3 | N/A | N/A | N/A | N/A |
| a：recent-only | 6 | N/A | N/A | N/A | N/A |
| b：两张新段可见帧 | 3 | N/A | N/A | N/A | N/A |
| b：两张新段可见帧 | 6 | N/A | N/A | N/A | N/A |
| c：两张旧段可见帧 | 3 | N/A | N/A | N/A | N/A |
| c：两张旧段可见帧 | 6 | N/A | N/A | N/A | N/A |
| d：end + 旧段帧 | 3 | N/A | N/A | N/A | N/A |
| d：end + 旧段帧 | 6 | N/A | N/A | N/A | N/A |

下表只引用 v2.3 已保存的 CPU B 均值，来自 96 个 validation pair 的 `evaluation_only` 位置；本次没有重算或新增区间。B 不是预测器测量，也不占 A100 时间。原记录见 [CPU B status](../results/motion_v2/v2_3/direction_bound/status.json)，完成时间为 `2026-10-31T21:37:24.213943+00:00`。

| horizon | 已保存 CPU B 均值 | B 95% CI | 冻结 decoder 在真实 target features 上的 position MSE / 95% CI |
|---|---:|---|---|
| 3 | 0.010503079706867778 | N/A；本次未计算 | N/A / N/A |
| 6 | 0.014639254826637952 | N/A；本次未计算 | N/A / N/A |

| 位置误差对比 | h=3 差值 / 95% CI | h=6 差值 / 95% CI | 判定 |
|---|---|---|---|
| d−b | N/A / N/A | N/A / N/A | 两项必需门槛均未评估 |
| c−b | N/A / N/A | N/A / N/A | 两项必需门槛均未评估 |
| B−b | N/A / N/A | N/A / N/A | 两项必需门槛均未评估 |
| d−B | N/A / N/A | N/A / N/A | 诊断未评估 |
| a−b | N/A / N/A | N/A / N/A | 报告项未评估 |

| 训练 seed | a/b/c/d × h=3/6 的 8 行 feature/position MSE 与 CI | checkpoint / decoder 结果 |
|---:|---|---|
| 17 | 全部 N/A | N/A |
| 29 | 全部 N/A | N/A |
| 43 | 全部 N/A | N/A |

机器可读的 8 行总表、24 行逐 seed 表、对比和六项门槛均以 `null` 保留在 [step2_results_not_run.json](../results/motion_v2/v2_3_1/step2_results_not_run.json)。**第 2 步结论为未评估，不是“预测器还不能用对的帧”，也不是通过。** 没有结果可用于修改阈值或选择新实验。

## 与已有步骤的衔接、证据和风险

此前第 0′ 步已保存结果按 v2.2 修订重判通过；Phase 1 headroom 的完整 JSON 与附件一致，但原 seed 9137 manifest 尚未逐 pair 核查。第 1 步实际生成 train 512 对、validation 96 对，配对审计 **608/608** 和 **7 项 pytest** 通过；在线 FIFO/keep-first 为 0%，充分帧上限为 100%。这些已保存结论没有重跑，详见 [v2.2 报告](MOTION_V2_2_REPORT_ZH.md)。

导出的 5 个原文件保留于 [export/](../results/motion_v2/v2_3_1/deployment3/export/)，包括原 ZIP 及可直接核验的原 SHA256SUMS；解包的 30 个文件保留于 [remote_evidence/](../results/motion_v2/v2_3_1/deployment3/remote_evidence/)。本地接收核验通过：ZIP 为 **47357 字节**，SHA256 为 `cfe71792a0a674910aaedadaefecc029af107ec61b857baa8afd89e42c2dfaa3`，全部 28 个源文件及 15 项嵌套回执校验通过；详见 [download_verification.json](../results/motion_v2/v2_3_1/deployment3/download_verification.json)。原作业 16542、三个 deployment 现场、旧报告、旧状态、协议与哈希锁保留；本次最终状态与哈希另列于 [v2.3.1 结果目录](../results/motion_v2/v2_3_1/)。

最大的研究风险仍是：隐藏状态中两张正确帧能够恢复方向，不代表冻结 I-JEPA 特征与既定预测器能利用这些信息。本轮没有模型结果来消除这个风险；作业取消原因也仍待外部证据解释。未打开或生成 test，未训练保留策略，未进入第 3 步。

证据接收完成后，释放了本次零 GPU 辅助会话 16545，OOD 页面已显示 Completed；未删除会话记录或任何文件。该操作发生在 16550 的原生终态之后，未取消或重新提交科学作业。见[辅助会话释放记录](../results/motion_v2/v2_3_1/deployment3/auxiliary_release_observation.json)。
