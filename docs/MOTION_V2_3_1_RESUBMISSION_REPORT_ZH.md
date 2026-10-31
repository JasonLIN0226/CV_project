# Motion v2.3.1 基础设施核查与有限重提进展

本页为 2026-10-31 的新增证据，保留旧的 [16550 交付记录](MOTION_V2_3_1_EXECUTION_ZH.md) 和全部原始文件。第 0′ 步、Phase 1 headroom 精确比较及第 1 步均已有通过记录；本轮不重跑这些测量。第 2 步正在跟进作业 **16614**，尚无预测器指标或科学门槛结论。

## 16550 原因核查

23:48:43–23:48:50 UTC 只读采集 `sacct --duplicates` 的 **120 个字段及全部作业步**，未加 `-X`。顶层为 `CANCELLED`，作业步列表为空。

| 入口前失败条件 | 原生或文件系统证据 | 结论 |
|---|---|---|
| 无 `.batch` 步 | 完整查询中的作业步为 `[]` | proven |
| stdout/stderr 均不存在 | 两个带 16550 job ID 的日志路径均不存在 | proven |
| 启动标记目录不存在 | `scientific_job_started` 目录不存在 | proven |
| 科学输出目录不存在 | `step2_gpu_attempt1` 目录不存在 | proven |

节点 `b-19-12` 为 `DOWN+CLOUD+POWERED_DOWN`，Reason 原文为 `16550 finishing : ResumeTimeout reached [slurm@2026-10-31T23:47:14]`。当次配置快照显示 `ResumeTimeout=6000 sec`、`ResumeProgram=/bin/true`，并保存了整个 A100 分区的节点状态快照。这些证据支持“科学入口前的基础设施失败”，并明确关联到节点恢复超时；尚不能确定更底层的云资源或启动故障。

唯一查询错误是旧 job 的 `scontrol` 已被控制器清理（`Invalid job id specified`）。重提前重新读取的 sacct 与已保存终态记录逐字段一致，经过 SHA256 校验后使用原 scontrol 的 `Restarts=0` 证据，符合登记的受限回退规则。此错误不影响四项肯定证据。16550 实际 A100 时间仍为 **769 秒**，不把 6000 秒配置值当作实际用时。

完整证据：[诊断回执](../results/motion_v2/v2_3_1_resubmission/submission2_evidence/diagnosis_16550/receipt.json)、[状态](../results/motion_v2/v2_3_1_resubmission/submission2_evidence/diagnosis_16550/status.json)、[原生输出与 SHA256](../results/motion_v2/v2_3_1_resubmission/submission2_evidence/diagnosis_16550/SHA256SUMS)。回执 SHA256 为 `02907f56c354a2042ddd2896cb0ce2e84defe72ad0120af4afa58a17fc17bd06`。

## 冻结检查与 attempt2

科学 checkout 仍为 `6dc5ed253dfa305b26be8f19d40edcd6a5878ea7`；科学源码、protocol、config、数据、seed、判据、BudgetGuard 和 wrapper 均未改变。新增运行层代码、测试和最终协议均先提交推送，再执行。新增软件测试 **158/158 通过**；原 **98/98** 回放记录和对应源码保留。

本次重新核对全部 **2432 个 NPZ** 和原 ZIP 的 SHA256，原 venv 的 import 与关键版本检查通过，未安装或更换依赖。累计预算准入为 `769 + 10800 = 11569 ≤ 14400` 秒。

| 尝试 | Job ID | 已观察状态 | 节点 | 实际 A100 秒 |
|---|---|---|---|---:|
| attempt1 | 16550 | CANCELLED；四项入口前失败已证明 | b-19-12 | 769（已结算） |
| attempt2 | 16614 | 提交时 PENDING，Restarts=0 | 提交时尚未分配 | 待原生终态结算 |
| attempt3 | 未提交 | 仅在全部登记条件成立时允许 | — | — |

16614 于 **2026-10-31 23:50:18 UTC** 唯一提交一次，立即保存 scontrol：单节点、单张 A100、03:00:00 wall-time，Requeue=1 记录但不阻塞，Restarts=0。回执 SHA256 为 `6f152d0c4ed826ab92f69fe46e3d0912c94d9d42f69fdc303c1e81370bca59dd`。当前状态是提交时快照，不代表已经运行。

证据：[提交回执](../results/motion_v2/v2_3_1_resubmission/submission2_evidence/attempt2/submission/submission.json)、[立即 scontrol](../results/motion_v2/v2_3_1_resubmission/submission2_evidence/attempt2/submission/immediate_scontrol.stdout.txt)、[环境复核](../results/motion_v2/v2_3_1_resubmission/submission2_evidence/attempt2/preflight_environment/receipt.json)、[数据复核](../results/motion_v2/v2_3_1_resubmission/submission2_evidence/attempt2/submission/data_reverification.json)。新证据包共 32 个源文件、34150 字节，下载后包哈希及全部内部 SHA256 校验通过；包 SHA256 为 `c49ad84afdd9066fee960d3377e5c81afd7830fadec52603246260233a6c9573`。

## 科学结论与后续边界

a/b/c/d × h=3/6 的 feature/position MSE、配对区间、decoder 诊断和逐 seed 表尚未生成，六项必需门槛均为 **未评估**。已冻结 CPU 方向下界 B 为 h=3：`0.010503079706867778`，h=6：`0.014639254826637952`；这些数值不代表模型表现。

继续每 10 分钟读取原生状态并保留跟进。只有 16614 再次符合全部四项入口前失败条件、Restarts=0、无科学错误，且累计实际秒数加 10800 不超过 14400，才能进行最后一次 attempt3。启动目录存在即禁止重提。终态后先核对 sacct/scontrol 的状态和 GPU 数一致，再执行一次冻结结算并补齐结果。第 2 步无论通过或失败均停止；不打开或生成 test，不训练保留策略，不删除现场，不合并 PR。

当前最大的科学风险仍是冻结 I-JEPA 特征和既定预测器未必能利用方向信息；基础设施和软件检查通过不能回答这一点。节点恢复超时仍可能再次影响执行。

机器状态：[status.json](../results/motion_v2/v2_3_1_resubmission/status.json)。协议：[有限重提登记](MOTION_V2_3_1_RESUBMISSION_ZH.md)、[运行层 protocol](../experiments/motion_v2/protocol_resubmission_runtime_v2_3_1.json)。
