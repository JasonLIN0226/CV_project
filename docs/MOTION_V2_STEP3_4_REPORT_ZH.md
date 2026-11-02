# Motion v2 第 3/4 步当前进展报告

本报告记录 CPU 审计、远端传输及预检完成，GPU 作业 **17761 已唯一提交并验证、即时状态为 PENDING（Resources）** 时的进展。新 long-gap validation 的 **96/96 对结构审计、四项泄漏检查和无模型在线检查全部通过**；新实现的 334 项软件测试、在独立进程中正确调用的原 98 项 Slurm 回放测试均通过。另行登记的 18 项纯传输检查也已通过。预测器诊断、删除代价标签、54 个学生及在线比较尚无科学结果，是否值得开启 CONF 尚未评估。

CPU 回执完成时间为 2026-11-02 01:01:45 UTC（纽约时间 2026-11-01）。本文件为第 3/4 步的新报告，不替换此前 Step2 报告、停止记录或原始结果。CPU/软件数字抄录已保存 JSON/JUnit；最新远端部署进展来自终端 UI 观测及其本地记录，完整远端原生回执将在终态后导出。撰写报告未读取 NPZ 数组或重新计算指标。

## 登记与软件检查

科学协议先于执行提交并推送于 `9f8600a`。其 [中文协议](MOTION_V2_STEP3_4_PROTOCOL_ZH.md)、[机器可读登记](../experiments/motion_v2/protocol_step34.json)、[固定配置](../experiments/motion_v2/config_step34.json) 与 [复用资产清单](../experiments/motion_v2/step34_assets.json) 共同限定本次范围。机器可读登记 SHA256 为 `3b4fdd6b7888ef0d052d74b2be3a07b4680debccb59406487d054308be3bd91d`，锁定 72 项执行源码；其后两次软件调用修正没有改变这些科学源码、参数或判据。

| 软件检查记录 | 保存的结果 | 解释与证据 |
|---|---|---|
| preflight1 | 430 通过、2 失败；原状态保留失败 | 将新模型单测与原生回放放入同一进程，违反原生测试要求的无 torch/NumPy 预导入及输出环境变量契约。[状态](../results/motion_v2/step3_4/software_preflight1/status.json)、[JUnit](../results/motion_v2/step3_4/software_preflight1/junit.xml)、[SHA256](../results/motion_v2/step3_4/software_preflight1/SHA256SUMS) |
| preflight2：新实现组 | **334/334 通过** | 以独立进程执行新数据、核心流程、统计、预算、学生、提交、导出及入口检查。[JUnit](../results/motion_v2/step3_4/software_preflight2/step34_synthetic.xml) |
| preflight2：原生回放组 | 97 通过、1 失败；该次整体原状态仍为失败 | `MOTION_SLURM_REPLAY_OUTPUT` 误传为 JSON 文件路径；原测试实际要求已存在的目录，以写入 `fixture_replay.json`。[状态](../results/motion_v2/step3_4/software_preflight2/status.json)、[JUnit](../results/motion_v2/step3_4/software_preflight2/native_replay.xml)、[SHA256](../results/motion_v2/step3_4/software_preflight2/SHA256SUMS) |
| preflight3：原生回放组 | **98/98 通过** | 使用独立新进程及正确输出目录；没有重复已经通过的新 334 项。[状态](../results/motion_v2/step3_4/software_preflight3/status.json)、[JUnit](../results/motion_v2/step3_4/software_preflight3/junit.xml)、[逐字段回放](../results/motion_v2/step3_4/software_preflight3/fixture_replay.json)、[SHA256](../results/motion_v2/step3_4/software_preflight3/SHA256SUMS) |

调用修正分别先登记、提交推送于 `50077d1` 和 `8c226dd`，见 [进程隔离登记](../experiments/motion_v2/protocol_step34_software_execution.json) 与 [输出目录登记](../experiments/motion_v2/protocol_step34_native_replay_execution.json)。前两次失败证据完整保留；这些失败发生在软件检查的调用契约上，没有启动真实数据科学计算，也不代表预测器或策略的统计失败。新 long-gap 数据在上述检查通过后才唯一生成并审计。

## 数据迁移与执行绑定

在 `bffd086` 中先提交推送 [纯传输登记](../experiments/motion_v2/protocol_step34_transport.json)，再运行人工内存 ZIP 测试，**18/18 通过**。其 [JUnit](../results/motion_v2/step3_4/transport_software_preflight1/junit.xml)、[状态](../results/motion_v2/step3_4/transport_software_preflight1/status.json) 和 [SHA256](../results/motion_v2/step3_4/transport_software_preflight1/SHA256SUMS) 已保留；检查不解码真实 NPZ、不调用科学入口。科学协议中的 72 项源码未更改。

[GPU 执行登记](../experiments/motion_v2/protocol_step34_execution.json) 绑定已通过的 432+18 项软件检查、CPU 391 文件清单和传输工具哈希。新 validation 的 384 个 NPZ 已通过一次生成的 ZIP 迁移：1371657 字节，SHA256 `e52b8605dd29607ed1982620ebc526e1f7c9fe018fb41d992688c813360c2070`。传输 helper 唯一执行并通过：验证整个包及每个成员字节，要求所有目标 NPZ 不存在，exclusive 写入后复核全部 391 文件；7 个已有元数据文件只读核对，`metadata_overwritten=false`、`arrays_decoded=false`。传输回执 SHA256 为 `fa1e555e7deae2fe3c3ebfa96d3e7fd5803fa73de933f18514c21a9e719330db`。HPC 没有重新生成或审计该数据。

独立的 [只读回执核对](../results/motion_v2/step3_4/cpu_independent_receipt_check.json) **20/20 通过**，验证文件/源码哈希与记录一致、无 ID/pair/场景 seed 重叠；未解码 NPZ 数组或重算科学指标。远端冻结 HEAD 仍为 `bfe66f998aa057598d6576fcd9c4deaccb5d5543`。部署预检唯一执行并通过，覆盖 2432 个旧 NPZ、1239 项冻结资产、72 项源码及环境；该 CPU 预检记录 `cuda_initialized=false`、GPU 用时 0，不代表后续科学作业的 GPU 核验结果。预检回执 SHA256 为 `a8e7eca08db80941509eb908678de665d513bae63e693ed03594a128e046f5fb`。随后唯一提交返回 `submitted_verified`，科学 job ID 为 17761；原生即时字段见下文。

## 新 long-gap validation 的 CPU 结果

数据 seed 为 `20261101`，只生成 validation 96 对、192 个 episode，共 9922 帧，gap 为 20–32。生成器的其他场景语义沿用冻结版本；没有生成或打开 test/CONF。原 ordinary 数据仅使用 train512 对拟合泄漏探针，新 validation 用于一次评分，不用于拟合或选超参数。

| 检查 | 结果 |
|---|---|
| 配对、时间结构、可见约束、动作、隐藏状态隔离及文件一致性 | **96/96 对通过** |
| 审计总状态 | `passed` |
| 新数据 manifest SHA256 | `18185ffb796ff954d6d577f0bf134aea22febc54899400411a98e6fe36334925` |
| 原 ordinary manifest SHA256 | `9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a` |
| CPU 审计计时 | 2.3348340839 秒；新增 GPU 用时 0 |

泄漏探针为预登记的 `HistGradientBoostingClassifier(max_iter=200, random_state=0)`，选帧 seed 为 7；1024 个原 train sibling 样本拟合，192 个新 long-gap sibling 样本评分。以下准确率按未四舍五入值判断，阈值未调整。

| 输入 | 正确数 / 192 | 准确率 | 登记范围 | 结果 |
|---|---:|---:|---|---|
| end_only | 96 | 0.5000000000 | [0.47, 0.53] | 通过 |
| one_post_non_end | 107 | 0.5572916667 | ≤ 0.58 | 通过 |
| stale_plus_end | 96 | 0.5000000000 | [0.47, 0.53] | 通过 |
| two_post | 187 | 0.9739583333 | ≥ 0.95 | 通过 |

该检查按本次登记采用 train→新 validation 的划分，区别于原 Step1 在 train 上的五折审计。两个结果不能作为相同采样方式下的提升或下降比较。

无模型在线检查以“最终保留两帧均是 evaluator 标记的充分帧”为成功，不使用学习预测器：

| 策略或特权参照 | 192 个 episode 成功率 |
|---|---:|
| FIFO | 0% |
| keep_first | 0% |
| hindsight_oracle | 100% |
| latest_visible_reference | 100% |

原始证据为 [audit.json](../results/motion_v2/step3_4/cpu/audit.json)、[status.json](../results/motion_v2/step3_4/cpu/status.json)、[产物哈希清单](../results/motion_v2/step3_4/cpu/artifact_sha256.json) 和 [SHA256SUMS](../results/motion_v2/step3_4/cpu/SHA256SUMS)。`audit.json` 的 SHA256 为 `71a452aa7717e744d98d4c69383bc4058de1491897d7d655c60ba9c32d6c6195`。回执保留拟合/评估 ID、选帧、输入/标签/预测哈希及分类器参数。

## GPU 阶段与科学结论的边界

本次继续复用已结算作业 16614、科学提交 `6dc5ed253dfa305b26be8f19d40edcd6a5878ea7` 的原编码特征、train-only normalization、basis、decoder 和 seed17/29/43 预测器选点。旧预测器结果见 [Step2 最终报告](MOTION_V2_3_1_RESUBMISSION_REPORT_ZH.md)。本报告没有新增这些模型的结果。

| 登记工作 | 范围 | 当前结果 |
|---|---|---|
| ordinary 预测器诊断 | 随机两张新段帧、b、d，与 B/真实特征 decoder 误差；h3/6 | 未运行；feature/position MSE、95% CI 与差值均 N/A |
| long-gap 预测器诊断 | a/b/d、B、真实特征 decoder 误差；h3/6 | 未运行；B−b、d−b、d−B 与各区间均 N/A |
| 四种删除代价 | local_feature、local_position、use_time_feature、use_time_position | 未生成标签或评价标签 |
| 学生及对照 | 4 种代价 + future_attention + 4 种 shuffle，2 种架构 × 3 seeds，共 54 臂 | 未训练、未选 checkpoint；各 seed 结果均 N/A |
| ordinary 在线评价 | 54 学生、7 个通用 baseline 及 3 个特权/事后参照 | 未评估；成功率、h3/6 损失与全部配对差值均 N/A |
| long-gap 在线评价 | 仅在冻结 reader 的 long-gap 诊断获得资格后评价 | 资格尚未判断；结果均 N/A |
| 是否建议开启 CONF | 主要候选 set/use_time_position 的预登记联合条件 | **当前证据不足，不能建议开启** |

long-gap 的资格要求 h3、h6 各自的 `B − position MSE(b)` 配对 95% 下界严格大于 0。未评估不等于通过或失败；若不满足，该间隔的主张范围受限，不能用普通间隔的结果补足。在线成功率及损失、集合与独立学生的差值、所有 future_attention/shuffle 臂都需要完整报告；不能凭本次 CPU 上限或旧 Step2 结果选择性省略。

正式统计仍按冻结协议使用 96 个 pair、2000 次 pair bootstrap、seed901，先平均 siblings 及固定 reader/student seeds；CI 不把这些 seeds 当作独立样本，也不外推训练 seed 总体。主要候选及 future_attention 对手若任何 seed 选中 step0，只能如实报告初始化参照，不能据此作“训练标签优于训练标签”的联合主张。

## 用时与当前运行状态

| 项目 | A100 秒 |
|---|---:|
| 已结算旧作业 16550 | 769 |
| 已结算旧作业 16614 | 2936 |
| 当前已结算累计 | **3705** |
| 新作业 17761 实际 | **待原生终态结算** |
| 总预算 | 14400 |
| 本次提交前剩余 | **10695** |
| 本次登记估计 | 8500 |
| 本次单 A100 作业上限 | 10500 |

8500 秒为运行前估算，不是实测；登记上限与旧累计之和为 14205 秒，仍须按原生 Slurm 计时核验及动态预算守卫停止。CPU 数据审计、软件测试和部署预检没有消耗 A100 时间。17761 提交后即刻尚未分配 GPU，不能把这一 PENDING 快照的零分配当作该作业最终用时为 0；最终累计须加上其终态结算。

部署历史：用户回复后只读确认 Microsoft 跳转已经完成，My Interactive Sessions 显示 `xy3257` 已登录；此前对暂时跳转的阻塞判断过早，原记录保留。零 GPU 辅助会话 `17718` 在 `b-10-167` 运行（Start `2026-11-02T01:12:17Z`、Restarts=0、Requeue=1、请求4 CPU/实际24 CPU）；这些辅助调度差异按 v2.3 分级规则记录后继续。新目录独立浅克隆完成，完整 HEAD 与部署登记匹配，上传了唯一 CPU ZIP。随后其他 Chrome 窗口切换，工具两次中止操作，当时已请求用户短暂保留 HPC 前台；当时保存辅助原生记录的命令完成状态尚未确认，因此没有盲目重发。该阶段原记录保留在 [deployment_progress1](../results/motion_v2/step3_4/deployment_progress1/deployment_state.json)。

用户释放 HPC 窗口后，只读查明原辅助原生记录命令已经完成，两个输出分别为 1486、128 字节。之后传输 helper、部署预检各唯一执行并通过，submit helper 也只调用一次，返回 `submitted_verified`。作业 17761 的即时原生观测如下：

| 字段 | 提交后即时观测 |
|---|---|
| SubmitTime | `2026-11-02T01:33:01 UTC` |
| JobState / Reason | `PENDING` / `Resources` |
| Restarts / Requeue | `0` / `1` |
| NumNodes | `1-1` |
| 请求 GPU | 1 张 A100 |
| AllocTRES | `(null)`，尚未获得分配 |
| TimeLimit | `02:55:00` |
| submission 回执 SHA256 | `5257ba414fe0f2fb4db82aa7c6aafd63ce278d407ca9e3af7fbefc6562cb0091` |

`Requeue=1` 按已登记规则记录；任何实际 `Restarts>0` 仍是停止条件。已建立 `motion-v2-3-4-17761` 的 ACTIVE heartbeat，每 10 分钟检查；这不代表已取得科学结果。最新本地观测见 [deployment_progress2/deployment_state.json](../results/motion_v2/step3_4/deployment_progress2/deployment_state.json) 及其 [SHA256SUMS](../results/motion_v2/step3_4/deployment_progress2/SHA256SUMS)，当前记录来源是终端 UI；完整远端原生回执及科学输出须在终态后导出核验。科学结果继续标 N/A。

当前最大的研究风险是：**CPU 隐藏状态审计证明两张充分帧包含信息，不能证明冻结 I-JEPA/预测器能在新 long-gap 上读取这些信息，也不能证明学生会在线保留它们。** 本次尚缺这两层实证。第 4 步结束后仍须停止，不能因 validation 表现理想而自行生成或打开 CONF/test。

校验清单的版本边界：`SHA256SUMS_CPU_DELIVERY` 固定描述 `bfe66f9` 的交付快照，包含当时的报告和根 status；`SHA256SUMS_DEPLOYMENT_PROGRESS1` 保存前次部署进展快照。本次报告、根 status 与新增部署观测的当前版本由 `SHA256SUMS_DEPLOYMENT_PROGRESS2` 记录。CPU 子目录的原始文件和哈希清单保持不变。
