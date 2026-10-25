# Reader 聚合诊断：A100 已完成结果（2026-10-25）

**四头读取 `multihead` 和先做 item self-attention 的 `item_attention`，在旧 DEV 与新诊断集上均通过了 R/D/S 的全部 K2、K3 资格门槛。** 原版、仅增加均匀通道、仅加宽输出 MLP、以及原版的无训练 uniform 干预，没有全部通过。这说明当前读取问题可以在固定数据与训练预算下改善；尚未证明任何保留策略或学习型 scorer 的收益。

任务 11544 已完成，Slurm 为 `COMPLETED / 0:0`，单张 A100 分配时间 2010 秒（33 分 30 秒，**0.55833 GPUh**）。这是 **一个配对训练 seed17 的架构诊断**。新诊断集不是原 proposal 的最终确认集；本轮没有训练保留学生，没有重跑新 reader 下的 baseline/medoid/full-reference sensitivity，原 CONF 仍未打开。状态以本报告及[执行记录](../results/literature_diagnosis/reader-aggregation-20261025/gpu_study_status.json)为准；[冻结协议](READER_AGGREGATION_PREREGISTRATION_20261025.md)保留运行前的状态文字。

## 1. 本轮固定了什么

源码冻结为 `82e31a0ecc555d9a9a7ffa6b13282dad00b9bef5`。五个训练臂共用冻结的 I-JEPA 特征、TRAIN 拟合的归一化与 rank128 输入/输出基底、动作接口、h3/h6 目标、纯 feature MSE、配对初始化 seed17 和完全相同的 1024 步样本日程。每个臂是一个覆盖三种任务的 reader，均仅按 VAL 选择 checkpoint；五个臂最终都选中 step1024。`uniform_intervention` 复用原版训练权重，只强制合法历史项均匀加权并重算全部时间/动作摘要，不是第六个训练臂。

| 数据 | seed | 基础场景 family 数 | 用途 |
|---|---:|---:|---|
| TRAIN | 20261101 | 48 | reader / probe 训练 |
| VAL | 20261102 | 24 | checkpoint 与 probe 正则选择 |
| 旧 DEV | 20261103 | 24 | 已观察的诊断数据 |
| 新诊断集 | 20261120 | 24 | 所有 checkpoint / probe 锁定后才生成、编码、评价 |
| 原 CONF | 20261104 | 48 | **未打开** |

每个评价 split 包含 R（冗余）288、D（诱饵）288、S（协同）1152 条记录，共 1728 条，但独立分组单位仍是 **24 个基础场景**，不是 1728 次独立重复。R/D 每种外观条件覆盖 24 个 family，S 每种外观条件覆盖 6 个。不得把 S 较大的记录数直接用于三个任务的加权总分。

## 2. 完整分类结果

以下为百分比。K2 是保留语义完整的两项证据时，native 时间与 fixed 时间条件合并后的准确率；K3 是完整三候选输入。分类依据是预测特征与真实未来原型的距离，R/D 为四类，S 为两类。**它不是规划成功率，也不代表 reader 已达到条件均值或完全校准。**

### 旧 DEV

| Reader | R K2 | R K3 | D K2 | D K3 | S K2 | S K3 |
|---|---:|---:|---:|---:|---:|---:|
| `original` | 75.61 | 73.26 | 85.24 | 84.38 | 93.75 | 72.48 |
| `attention_mean` | 100.00 | 99.31 | 100.00 | 94.79 | 94.79 | 77.60 |
| `multihead` | 99.83 | 97.57 | 99.65 | 98.61 | 94.44 | 96.27 |
| `item_attention` | 100.00 | 100.00 | 99.31 | 99.65 | 95.23 | 89.32 |
| `wide_control` | 88.45 | 84.38 | 92.53 | 92.01 | 94.18 | 72.48 |
| `uniform_intervention` | 97.40 | 94.10 | 98.26 | 68.75 | 93.75 | 65.10 |

### 新诊断集

| Reader | R K2 | R K3 | D K2 | D K3 | S K2 | S K3 |
|---|---:|---:|---:|---:|---:|---:|
| `original` | 76.39 | 75.35 | 86.11 | 84.72 | 96.96 | 73.44 |
| `attention_mean` | 100.00 | 99.65 | 100.00 | 96.53 | 97.31 | 79.69 |
| `multihead` | 100.00 | 97.22 | 100.00 | 98.61 | 97.22 | 96.96 |
| `item_attention` | 100.00 | 100.00 | 99.48 | 100.00 | 96.61 | 93.32 |
| `wide_control` | 88.89 | 83.68 | 93.92 | 93.75 | 94.70 | 72.57 |
| `uniform_intervention` | 97.05 | 93.75 | 97.40 | 69.79 | 97.05 | 65.10 |

新诊断集的 S K3：原版 **73.44%**，四头 **96.96%**（+23.52 个百分点），item self-attention **93.32%**（+19.88 个百分点）。`attention_mean` 为 **79.69%**，低于预先固定的 80% 门槛，不能四舍五入为通过。`item_attention` 的 VAL MSE 最低，四头的 S K3 分类率更高；本轮不能据此声称存在一个跨指标都最好的架构。

### K2 的 native / fixed 分开看

每格为 **native / fixed** 准确率（%）。native 保留实际时间与动作摘要；fixed 使用预先固定的槽位时间及动作摘要，以隔离内容。

| Reader | 旧 R | 新 R | 旧 D | 新 D | 旧 S | 新 S |
|---|---:|---:|---:|---:|---:|---:|
| `original` | 74.83 / 76.39 | 76.22 / 76.56 | 84.38 / 86.11 | 84.72 / 87.50 | 93.75 / 93.75 | 96.88 / 97.05 |
| `attention_mean` | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 94.79 / 94.79 | 97.22 / 97.40 |
| `multihead` | 99.83 / 99.83 | 100.00 / 100.00 | 99.65 / 99.65 | 100.00 / 100.00 | 94.44 / 94.44 | 97.22 / 97.22 |
| `item_attention` | 100.00 / 100.00 | 100.00 / 100.00 | 99.31 / 99.31 | 99.65 / 99.31 | 94.10 / 96.35 | 96.70 / 96.53 |
| `wide_control` | 88.02 / 88.89 | 88.37 / 89.41 | 92.01 / 93.06 | 93.75 / 94.10 | 93.92 / 94.44 | 94.27 / 95.14 |
| `uniform_intervention` | 97.40 / 97.40 | 97.22 / 96.88 | 98.26 / 98.26 | 97.22 / 97.57 | 93.75 / 93.75 | 97.05 / 97.05 |

## 3. 每个条件都必须过门槛

沿用冻结规则，没有根据本轮结果降低门槛：

- **K2：**任务整体完整集合准确率、内容替换（flip）准确率均至少 80%；fixed 完整集合 MSE 比 missing 集合低，差值大于 `1e-6`。每个外观条件还要求 native 完整集合、每一种语义有效删除的 fixed 完整集合、以及内容替换准确率分别达到 80%；每种有效删除的 MSE 必须低于每一种缺必要证据的删除条件均值。
- **K3：**任务整体及每个外观条件的完整三项分类准确率都至少 80%。
- 使用 full-reference sensitivity 的任务需要 **K2 和 K3 都合格**。平均准确率超过 80% 不足以代替逐条件资格。

下表“双方通过”指 K2、K3 都过；“仅 K2”“仅 K3”“双方失败”同理。除箭头格外，新旧两套数据的资格相同；箭头表示 **旧 DEV → 新诊断集**。

| 条件 | original | attention_mean | multihead | item_attention | wide_control | uniform |
|---|---|---|---|---|---|---|
| `R/neutral` | 双方通过 | 双方通过 | 双方通过 | 双方通过 | 双方通过 | 双方通过 |
| `R/salient_redundant` | 双方失败 | 双方通过 | 双方通过 | 双方通过 | 双方通过 | 双方通过 |
| `R/salient_unique` | 双方失败 | 双方通过 | 双方通过 | 双方通过 | 双方失败 | 仅 K3 |
| `D/neutral` | 双方通过 | 双方通过 | 双方通过 | 双方通过 | 双方通过 | 仅 K2 |
| `D/salient_decoy` | 双方通过 | 双方通过 | 双方通过 | 双方通过 | 双方通过 | 仅 K2 |
| `D/salient_useful` | 双方失败 | 双方通过 | 双方通过 | 双方通过 | 双方失败 → 双方通过 | 仅 K2 |
| `S/neutral` | 仅 K2 | 仅 K2 | 双方通过 | 双方通过 | 仅 K2 | 仅 K2 |
| `S/salient_slot_0` | 仅 K2 | 仅 K2 | 双方通过 | 双方通过 | 仅 K2 | 仅 K2 |
| `S/salient_slot_1` | 仅 K2 | 双方通过 | 双方通过 | 双方通过 | 仅 K2 | 仅 K2 |
| `S/salient_slot_2` | 仅 K2 | 双方通过 | 双方通过 | 双方通过 | 仅 K2 | 仅 K2 |

`wide_control` 在新诊断集的 D 已通过，但 R 的 `salient_unique` 和全部 S K3 条件仍失败。uniform 的 R 总体分类很高，却在 `salient_unique` 的完整/缺失证据 MSE 比较上失败；D、S 则是 K3 失败。因此不能把“均匀加权提高准确率”写成“uniform 已修复 reader”。

## 4. MSE 保留了不同的取舍

每格为 **旧 DEV / 新诊断集** 的 feature MSE，越低越好。K2 仅使用 fixed 时间的语义完整集合；K3 使用完整三项。h3/h6 与特征维度的聚合口径沿用原评价，不以分类正确率替代连续预测损失。

| Reader | R fixed K2 | R K3 | D fixed K2 | D K3 | S fixed K2 | S K3 |
|---|---:|---:|---:|---:|---:|---:|
| `original` | 0.05936 / 0.05678 | 0.06139 / 0.05888 | 0.06525 / 0.06122 | 0.06476 / 0.06080 | 0.08571 / 0.07954 | 0.10928 / 0.10541 |
| `attention_mean` | 0.04761 / 0.04102 | 0.04951 / 0.04343 | 0.05781 / 0.05139 | 0.06359 / 0.05982 | 0.07915 / 0.07186 | 0.10484 / 0.09919 |
| `multihead` | 0.04502 / 0.04197 | 0.04681 / 0.04384 | 0.05411 / 0.05002 | 0.05472 / 0.05428 | 0.08409 / 0.07787 | 0.07549 / 0.07318 |
| `item_attention` | 0.04230 / 0.03758 | 0.04306 / 0.03918 | 0.05227 / 0.04637 | 0.05173 / 0.04539 | 0.07175 / 0.06370 | 0.08407 / 0.07555 |
| `wide_control` | 0.05255 / 0.04904 | 0.05534 / 0.05321 | 0.05948 / 0.05435 | 0.05806 / 0.05285 | 0.08006 / 0.07813 | 0.10817 / 0.10624 |
| `uniform_intervention` | 0.08522 / 0.07743 | 0.09124 / 0.08299 | 0.08959 / 0.08019 | 0.15298 / 0.14776 | 0.08552 / 0.07880 | 0.17168 / 0.16702 |

uniform 在新诊断集 S K3 的 MSE 从原版 **0.10541** 升到 **0.16702**。强制权重变化对固定下游网络产生了分布偏移，因此这是固定网络上的计算干预，不是“uniform 聚合训练后也会失败”的结论。相反，训练过的多通道 / item self-attention 的改善同时涉及读取机制与学习过程。

## 5. 参数与实测成本

参数量为活动可训练参数，未把共享基底缓冲区计入。训练阶段秒数包括该臂训练阶段的准备、VAL 与 checkpoint 操作；它不同于全部作业的 GPU 分配时长。K2/K3 延迟使用新诊断集固定的前 16 个 parent，1 次 warmup、5 次计时，取批次耗时中位数再除以 16；每个 parent 预测两个 horizon，包含组批和输出重建，排除编码器。**这是批处理摊销耗时，不是单请求延迟，更不是完整选择器的 48/64 次预测成本。**

| Reader | 可训练参数 | 训练阶段（秒） | 选中 VAL MSE | K2（ms/parent） | K3（ms/parent） |
|---|---:|---:|---:|---:|---:|
| `original` | 215,297 | 188.59 | 0.10706 | 0.932 | 0.927 |
| `attention_mean` | 313,857 | 200.73 | 0.10594 | 0.953 | 0.971 |
| `multihead` | 511,364 | 230.82 | 0.10289 | 1.031 | 1.036 |
| `item_attention` | 347,905 | 227.26 | 0.10156 | 1.055 | 1.056 |
| `wide_control` | 510,927 | 175.62 | 0.10347 | 0.923 | 0.924 |
| `uniform_intervention` | 215,297 | 无新增训练 | 复用 original | 0.891 | 0.890 |

float32 参数存储分别为 861,188 / 1,255,428 / 2,045,456 / 1,391,620 / 2,043,708 字节（按上表五个训练臂顺序），Adam 两个 moment 为对应值的两倍；这些不含梯度、激活、共享缓冲区或 CUDA 工作空间，不能当作显存峰值。uniform 复用原版权重。

`wide_control` 的参数量只比四头少 **0.0855%**，但读取门槛仍失败。这削弱了本实验中“只是参数增加”的解释；容量控制没有同时控制计算量、优化难度和结构，因此不能推广成增加容量永远无效。GPU 全作业 0.55833 GPUh 包含五臂训练、资格评价、新数据编码和产物导出，不是单个 reader 的训练预算。

## 6. 现在可以说什么

1. **证据信息存在，当前读取方式能改进。** CPU 的真实 payload/tag 线性探针与阴性对照，和旧 checkpoint 的无训练干预见[CPU 审计报告](READER_AGGREGATION_AUDIT_ZH.md)。真实 raw / projected 探针可解码，不等于 reader 必然使用成功；打乱标签对照也没有被写成全部恰好 50%。本轮在数据、基底、目标和更新次数固定时，替代聚合器通过资格，为“先修好 reader 再解释淘汰”提供了直接实证。
2. **显式 item-item self-attention 并非这组任务的必要条件。** 四头臂使用四个分别归一化的 itemwise attention 通道加下游非线性读出，没有显式 item-item self-attention，也能通过。但这不证明任务没有关系，也不证明所有关系任务都不需要关系计算。原版单 softmax 在表达能力上也不是被理论证明不可能，结果只能说明本实现与本训练设置下的表现差异。
3. **读取失败不能吞掉原有保留失败。** 旧研究的 S K2 原本合格，medoid 在该条件下的失败仍是有效的旧 reader 结果；本轮没有把它撤销，也没有用新 reader 重跑后得到新的失败率。K3 合格现在使 full-reference sensitivity 具备继续测试的前提，但分类资格不等于 Bayes 校准，更不保证该规则会获胜。
4. **原版重现成功。** 新旧 original checkpoint 的 19 个 `state_dict` 张量逐值完全相同，稳定状态哈希均为 `b9e22d3d4c1450fc3976e5fca32590e612e689af3ab35fc69e4d3d4d066b01d7`；旧 DEV 结果也重现。该哈希按排序后的 tensor key、dtype、shape 与连续 CPU tensor 字节计算。两个 checkpoint 文件的容器/元数据 SHA 不同，不能说整个文件相同。协议事前保留了“可能不逐位重现”的不确定性，本轮实际验证已消除此项不确定性。

这些是单个配对 seed17 的诊断结论。新诊断 seed20261120 已经用于比较架构，之后不能再把它称作未观察的确认集。发布数据提供以 family 为单位的 2000 次配对 bootstrap 描述区间；没有把许多变体当作独立训练重复，也没有从多个终点挑选显著性结论。S 外观分层只有 6 个 family，后续仍需要多 seed 与独立确认。

## 7. 下一步与证据入口

下一步应另行冻结协议：先固定一个合格 reader 和选择规则，再在同一 reader、同一预算与输入约束下比较强 baseline、medoid、动作平均的 full-reference sensitivity 和局部删除 oracle；随后才比较独立学生、集合条件学生、future-attention 与打乱标签对照。新方法比较应使用至少三个预先计划的训练 seed，以及明确锁定、尚未观察的确认数据。这里列的是后续工作，**本轮尚未运行这些选择器比较或保留学生**。

与 PM 的联系仍只限于 **readout adequacy**：世界模型的输出用于记忆决策之前，要验证它确实能利用相关证据。本轮没有新增 PM 的 S0–S2、CHANGE 或 BF-Delta 实验结果。

- [公开紧凑结果目录](https://github.com/JasonLIN0226/CV_project/tree/phase2/independent-interfaces-20261020/results/literature_diagnosis/reader-aggregation-20261025)；若分支后续更名，可从 [CV PR #1](https://github.com/JasonLIN0226/CV_project/pull/1) 的文件页查找同一路径。
- [完整汇总与分层资格](../results/literature_diagnosis/reader-aggregation-20261025/gpu_summary.json)、[全部模型锁定记录](../results/literature_diagnosis/reader-aggregation-20261025/gpu_all_models_locked.json)、[新诊断集解锁记录](../results/literature_diagnosis/reader-aggregation-20261025/gpu_diagnostic_opened.json)、[Slurm 最终记录](../results/literature_diagnosis/reader-aggregation-20261025/gpu_sacct-final.txt)。
- [独立只读复核](../results/literature_diagnosis/reader-aggregation-20261025/gpu_independent_audit.json)：12 套评价各 21,312 行的统计/资格重算一致；分层抽查 2,340 行预测系数重建，最大距离误差 `2.033e-8`，分类最优集合无不一致；配对样本日程、VAL 选点及新 split 锁定顺序通过。
- [GitHub release](https://github.com/JasonLIN0226/CV_project/releases/tag/reader-aggregation-20261025)：[完整 GPU 包](https://github.com/JasonLIN0226/CV_project/releases/download/reader-aggregation-20261025/reader-aggregation-results.zip)（160,782,411 字节，SHA256 `3bd921202d6b2275513d021362448e37d0079b95b03c1ed80c1847feeb8e9906`）；[重建/后处理包](https://github.com/JasonLIN0226/CV_project/releases/download/reader-aggregation-20261025/reader-aggregation-postprocessing.zip)（74,127,336 字节，SHA256 `6420cd69288ea5ec2b85da3c4a25d310484f229148e32413c5b652a1e8d734c8`）。GPU 包保存原始逐样本评价、checkpoint、学习曲线和架构/耗时记录；大体积派生 RGB / feature 文件的省略项及哈希记录在 manifest，远端原件保留。最终 Slurm 记录在后处理包及紧凑目录中，不能把作业结束前打包的日志误当最终调度状态。
