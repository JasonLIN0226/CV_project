# Gate：oracle 删掉 cue 的预测器与指标诊断

2026-10-22。复用已有192个开发 episode；共享 epoch-35 Gate predictor、旧归一化、
R4/K2 和 `L=(L_h3+L_h6)/2` 固定。没有打开正式 Phase1 测试集。

**结论：多数已选坏例中，保 cue 子集的输出确实更偏向错误未来；仅用“feature MSE 与任务不一致”
解释不够。同时，全历史 oracle 还利用未来选择时间索引，不能把其全部优势当成可部署
保留策略的收益。** 这些结果定位了问题，尚未唯一确定模型内部的原因。

## 从误差分解检查指标

对相同场景、两个未来分支的目标 `y`、`y_sibling`，定义 `d=(y-y_sibling)/2`。
将预测误差正交分解为沿 `d` 的部分及其正交余项；两者之和严格等于原 feature MSE。
这是使用未来真值的评估分析，不是在线策略输入；正交余项也不等于无关信息。

同时报告旧归一化还原后的原始 feature MSE、旧 TRAIN 独立拟合的位置 ridge readout。
位置 readout 固定正则系数1，新数据不参与拟合或选参。其 GT feature 位置误差在
ID/long-gap/shift 为0.000772/0.001125/0.005449，所以不能把它当物理真值或规划成功率。

| 比较范围 | 原 feature MSE 严格偏好无 cue | 平均保 cue 代价 | 沿未来分支差异方向 | 正交余项 |
|---|---:|---:|---:|---:|
| 全历史各自最优 K2 | 46/192 | 0.054760 | 0.045441（约83%） | 0.009318 |
| 真实 attention K+1 parent | 27/72可比较 | 0.060659 | 0.052706（约87%） | 0.007954 |

表中均值只针对原指标选择出的失败子集，不是全体改善。在46个全历史坏例中，
删 cue 同时改善原始 feature/分支方向/位置 readout 的数量为45/41/43；在27个真实
parent 坏例中分别为26/25/25。评价指标并未普遍把“更正确的分支”评得更差。

真实 parent 比较还固定了共同保留项，避免把候选档案不同当成因果解释。192个
attention parent 中另外120个有两个等价 cue 副本，任何单次删除仍保留 cue，
因此无法作“有 cue 对完全无 cue”的比较；这些 parent 完整保留在归档中，没有计为零差距。
但被交换的 cue 与非 cue 项仍有不同时间戳及已执行动作摘要，因此局部比较不能唯一
归因于 cue 内容本身；内容、时间和动作的共同影响尚未完全分离。

## 读出路径干预没有给出通用修复

固定原先选定的历史子集，在同一冻结模型上置零 transport、velocity、past-action 项，
或改成 uniform attention。没有重新挑子集或训练。原始路径与冻结模型的输出逐位一致。

| 路径干预 | 全历史原46坏例的平均 cue 代价 | 真实 parent 原27坏例的平均 cue 代价 |
|---|---:|---:|
| 原模型 | 0.054760 | 0.060659 |
| 无 transport | 0.057322 | 0.062073 |
| 无 velocity | 0.054455 | 0.053212 |
| 无 past-action | 0.043076 | 0.051913 |
| Uniform attention | 0.019394 | 0.032508 |

均匀 attention 缓解部分已选失败，却把全体192个 episode 的平均 cue 优势从
−0.013792改为+0.000225，不能作为整体修复。去速度项也未消除现象。
这些是可能离开训练分布的组件干预，只能帮助定位，不能据此宣布新 reader 或保留方法胜出。

## 全历史 oracle 的时间选择混淆

全部192个 episode 内，非 cue 历史的 feature 向量相同；但时间和执行动作上下文不同，
同内容的不同时间子集会改变预测，损失范围均值0.038673、最大0.145742。
给 sibling 完全相同的无 cue 子集，输出逐位相同；事后 oracle 却在全部96对 sibling
中选择不同的最佳无 cue 时间子集。因此其无 cue 分支准确率76.1%包含未来选择优势，
不能作为过去可用策略的预测能力。

这与真实 parent 的27个失败同时成立：保 cue 子集在冻结 reader 上预测更差，而更宽的历史
oracle 又有额外未来信息优势。需要分别量化，不能将其中一个解释扩展为全部原因。

## Probe 的支持域与证据限制

旧32个 TRAIN episode 的 cue 分类 probe 在新场景接近随机，但它把训练中近乎常量
的特征除以极小尺度，产生极大的新场景分数。**不能用这次 probe 失败证明 encoder
没有编码 cue。** 固定 TRAIN 支持域屏蔽后，cue probe 在 ID/long-gap/shift 的准确率
为100%/100%/96.875%，支持 cue 信息仍可从特征读取。相同支持域控制在共享预测器
上没有救回任何27个局部坏例，故近零训练支持不是这些预测错误的充分解释。
模型输入的训练支持域、固定系数屏蔽控制见
[输入支持域诊断](GATE_INPUT_SUPPORT_DIAGNOSIS.md)；oracle-label probe 的相同问题及
打乱标签对照见[可学性诊断](ORACLE_LEARNABILITY_DIAGNOSIS.md)。

## 复现、核验与范围

- [完整归档与图](../results/literature_diagnosis/metric-reader-20261022/README.md)
- [全历史 runner](../experiments/phase2/diagnose_gate_oracle_metric.py)、
  [真实 parent runner](../experiments/phase2/diagnose_gate_reader_local.py)、
  [不调用模型的导出复算](../experiments/phase2/export_gate_reader_diagnosis.py)
- [全部 parent](../results/literature_diagnosis/metric-reader-20261022/local/parents.json)、
  [时间选择审计](../results/literature_diagnosis/metric-reader-20261022/local/noncue_timing_audit.json)
- [独立模型复算](../results/literature_diagnosis/metric-reader-20261022/global/independent_review.json)、
  [2640行保存预测的独立算术核验](../results/literature_diagnosis/metric-reader-20261022/audit.json)

原始失败尝试保留，完整结果只来自两个 v2 run；大预测数组和训练 readout 保留在
manifest 指向的本地 build 路径，不放入 Git。CPU运行，没有新增编码或 GPU训练。
所有结果是重复使用开发集后的探索诊断，不是确认性检验。完整跨项目结论见
[综合报告](BASELINE_ORACLE_DIAGNOSIS_20261022.md)。
