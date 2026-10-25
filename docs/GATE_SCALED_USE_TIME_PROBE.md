# 扩大 reader 与 scorer 数据后的 use-time 实验

**标签修复现在可以被因果学生学到，但没有超过强 baseline。** 在同一个冻结的 512 对训练 reader 下，use-time 与 future-attention ridge 都保住了全部 96 条评价轨迹的 cue。use-time 显著优于 local-cost 和打乱标签，却没有优于 reader attention，且 MSE 高于 visual diversity。因此本轮支持修复实验管线、继续明确监督目标，不支持宣布新的保留方法优势。

## 实际执行与比较

- 前置 reader 已在 A100 上完成；本轮加载其固定 `pairs-512/selected.pt`，不再更新 reader、编码器或特征变换。
- 512 对 TRAIN 场景、1,024 条 episode。FIFO、reservoir、diversity、attention、WR 五种固定策略产生真实 K+1 parent；按 episode、时刻、三个候选时间去重后，共 **36,099 个 parent**，没有挑选有利样本。
- 四个学生共用 903 维因果输入、parent-centered ridge、λ=1、相同 TRAIN 支持规则和参数字节数：local-cost、use-time-cost、future-attention-label、parent 内打乱的 use-time-cost。
- use-time 标签：每个 parent 枚举一次删除，然后统一用冻结 attention 不可逆地继续到 query，在同一个最终集合上评价 q+3、q+6。它是参考策略条件下的离线 teacher，不是最优后续策略，也不是可部署 oracle。
- 部署学生不接收 query 时间、未来动作、未来图像或 cue 注释；仅接收当前实际存在的候选、recent 和已执行动作摘要。所有在线策略共用 R=4、K=2 和同一个 reader。
- 评价为 seed-20261027 的 48 对、96 条场景，ID、long-gap、distractor-shift 各 16 对。该集合已经看过 reader 对照结果，**本轮是 pilot，不是独立确认**。
- CPU 实验用时 **759.74 秒**，其中标签阶段约 725.31 秒、四个 ridge 拟合约 14.27 秒；没有新增 GPU 时间。未运行 TD 或新诱饵任务。

## 在线结果：必须同时看强对照和打乱对照

| 策略 | cue 保留 | 平均特征 MSE | 特征分支方向比例 |
|---|---:|---:|---:|
| local-cost ridge | **0/96** | 0.113421 | 50.00% |
| use-time-cost ridge | **96/96** | **0.058482** | 96.875% |
| future-attention ridge | 96/96 | 0.058471 | 96.875% |
| 打乱 use-time 标签 ridge | 27/96 | 0.096232 | 64.063% |
| reader attention | 96/96 | 0.058295 | 96.354% |
| visual diversity | 96/96 | **0.056376** | **97.917%** |

“特征分支方向”是预测沿配对未来特征轴的 alpha 是否大于零，在两个 horizon 上平均；不是物理轨道解码准确率，也不是任务成功率。future-attention 在这里是同架构 ridge 标签对照，不是原 PaFu 适配 MLP，更不是原论文系统复现。

按场景对重采样、固定种子、2,000 次 bootstrap 的描述性差值：

| use-time 减去对照 | 平均 MSE 差 | 95% 区间 |
|---|---:|---|
| local-cost | −0.054939 | [−0.059738, −0.048445] |
| 打乱 use-time | −0.037749 | [−0.046107, −0.029094] |
| future-attention ridge | +0.000011 | [−0.000092, +0.000155] |
| reader attention | +0.000187 | [−0.000582, +0.001209] |
| visual diversity | **+0.002106** | **[+0.000694, +0.004180]** |

这些是同一个已检查 pilot 上的描述性区间，未做多重比较校正。不能只报告前两行而忽略后面三行。

use-time 在三个条件均为 32/32 保 cue，MSE 分别为 ID 0.055695、long-gap 0.052744、distractor-shift 0.067007。local-cost 在三个条件均为 0/32。use-time 与 future-attention 的最终两个时间戳在 86/96 条轨迹上一致；这只是行为相似性，不是机制等价证明。

## local-cost 的实际失败发生在长期槽入口

完整重放 trace 显示：

- 96/96 条轨迹最终一直保留最早的 `[0,1]` 两帧。
- **全部 1,708 次 overflow 都拒绝刚离开 R4 的 incoming 帧。**
- 96/96 条轨迹中，任何 cue 副本都没有进入过长期历史槽。
- 第一次彻底丢失 cue 的时刻全部等于 **最后一份 cue 的时刻 +4**，即最后副本离开 recent 的时刻。

因此这个学生并非先把线索存好，之后在很晚的竞争中删错。它学出了持续拒绝新历史项、固定保留最早两个槽的行为。它与 WR 的冷启动失败也不同：这里没有“统计量尚未建立”的必要条件，失败来自已训练 scorer 的选择。

这个观察不能单独证明学生为什么学出该规则，但它限定了下一步解释必须覆盖的现象。

## 新 reader 下，local 标签仍强烈要求毁掉 cue

使用固定生成器重新计算了全部 1,184 条记录的 cue 时间，并严格核对 seed、query、长度、pair/id 和 RGB bank 索引。旧 32 条 TRAIN 还核对了原注释和特征等价副本。以下 cue 审计没有通过 salience 猜测 cue，也没有用于训练学生。

只看“候选中有且只有一份 cue、recent 中已没有 cue，删这一项会彻底丢线索”的 parent。表中计数使用容差 1e−8，要求**所有最优删除动作都会毁 cue**，不是随意打破并列。

| 目标时段 | 风险 parent 数 | local 标签要求毁 cue | use-time 要求毁 cue | attention 标签要求毁 cue |
|---|---:|---:|---:|---:|
| 两目标均在 query 或之前 | 3,480 | **3,368（96.78%）** | 11（0.32%） | 0 |
| 跨越 query 后的结果出现时刻 | 5,021 | 3,772 | 23 | 0 |
| 两目标均在 query 之后 | 5,351 | 1,748 | 24 | 0 |

完整 36,099 个 parent 的三个时段数分别为 13,142、10,734、12,223。表格仅限上述最后副本风险子集，分母不可混用。

use-time 标签并非每个场景都正确保护任务证据；仍有 58 个风险 parent 要求丢 cue，尚未逐一解释。它也有大量合法并列：全部 parent 中 24,344 个最优动作集合有并列，固定 continuation 可能使不同删除汇合到同一个最终记忆。训练最优动作命中数为 27,592/36,099，平均标签 regret 为 0.000446；不能脱离并列和 regret 单看命中率。

## 这次究竟证明了什么

本轮同 reader、同训练 parent、同输入和 ridge 容量的对照支持：**按使用时刻定义的标签包含了因果学生能学习并迁移到新场景的保留信息**。打乱标签显著退化；所以本轮收益不是只增加了一个任意打分器。

但它没有证明 use-time 优于 future-attention 或 diversity。Gate 中 cue 是独特、明显的历史信息，强 baseline 已经全部保住；现有实验还没有让“显眼或高 attention”与“真正有用”充分分离。

跨越前一轮 16 对训练实验，也不能把转正全归因于某一个因素：这次同时扩大了 reader 与 scorer 的训练数据，换了评价场景，TRAIN 支持维度从 193 增至 515；新实验的因果输入以 float32 存储、ridge 再转回 float64，旧实验直接保留 float64。四个本轮学生之间仍严格共用同一数值输入。

最关键的解释限制仍然存在：**reader 只在 `rec.query` 训练，local-cost 却在任意 overflow 时刻调用它。** use-time 同时把目标移到证据生效之后，并把 reader 调用起点搬回它训练过的 query 分布。当前结果不能独立证明“延迟信用机制”是全部原因。更强 query-reader 也没有自动使 query 前的 local 删除标签有效。

下一步最有区分力的检查，应先固定训练覆盖，区分“reader 对预测起点的失配”和“短目标时距本身没有证据价值信号”。之后若构建诱饵任务，需要验证 strong baseline 实际失败、reader 能使用真正证据、以及过去信息足以识别用途；不能预设 surprise/diversity/attention 一定会被诱饵骗过。

## 审计、成本与可复查材料

- 独立 verifier 检查了完整包 30 个成员的 SHA、576 条在线结果和不可逆轨迹、960 个配对统计/区间字段；bootstrap 最大差异为 **0**。
- 四个学生都保留 903 个权重、903 个尺度和一个 label scale，逻辑参数存储均为 **14,456 bytes**。零权重也未假装不占空间。共同冻结 reader 另计；同 K 不代表与启发式策略具有完全相同的总字节或算力。
- 此审计重算了保存结果的聚合、配对差值、cue 和轨迹语义，**没有重新执行 scaled reader 预测**。前置 A100 reader 的 1,920 条保存预测已另做 target-level 算术核验，二者不要混淆。
- CPU 早期启动/控制会话失败日志保留在完整包中。最终报告对应完整结束的执行，不把中断部分当作结果，也没有因中断改变策略或调参。

入口为 `experiments/phase2/run_scaled_use_time_probe.py`，独立审计为 `experiments/phase2/verify_scaled_probe.py`。便携材料在 `results/literature_diagnosis/scaled-use-time-probe-20261024/`：保存完整标签、参数、逐 episode 指标、压缩的全部训练 metadata 与在线 trace，以及 `analysis/verification.json` 和 `analysis/admission_diagnosis.json`。

完整包中还含 **391,168,892 bytes** 的 `training_inputs.npy`、精确 reader checkpoint 和固定变换；为了避免把大训练数组塞进 Git，便携目录保留原完整包 manifest 和哈希。完整 RGB/逐帧特征仍在 reader run，重新做全部在线模型调用需要它们；仅审查标签、训练拟合和已保存在线统计不需要再次编码。
