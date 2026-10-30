# 配着案例读论文：先定位失败环节，再谈新方法

2026-10-24。三篇指定版本的作者、章节和公式已按 arXiv 原文重新核对。本轮只整理已有证据和阅读路线，没有新增训练或实验。

先打开[新版案例图册](../results/literature_diagnosis/visual-casebook-20261024/index.html)。**内容失败与成功对照是一组；时间戳是这组的辅助消融；WR 启动是另一种、更早发生的失败。** 它们沿同一条证据链定位，但不能当成三种独立场景的泛化证明：

> 观察到线索 → 被接纳为事件 → 在预算内留住 → reader 正确利用 → 任务改善。

| 阅读顺序 | 必读范围 | 配着看的证据 | 读完要能回答 |
|---|---|---|---|
| 1. Worth Remembering | §3.2 式1–3、§5；略读 §3.3 | 图册 WR 启动；[完整启动诊断](WR_STARTUP_DIAGNOSIS.md) | 线索是没形成事件，还是形成后又被挤掉？ |
| 2. PaFu-KV | 图2、§3.2 式4–5、§3.3、§3.4 式6；附录C选读 | 图册内容失败＋成功；时间戳辅助面板 | attention 命中后，信息有没有带来正确的预测方向？ |
| 3. Gated DeltaNet | §2.3、§3.1 式8、表1；只做 CV 时选读，结合 PM 时必读 | [PM 卡片1–4](https://github.com/XintongYang1010/predictive-memory-lab/blob/experiments/bank-reference-20261022/docs/PM_CASE_GALLERY_ZH.md) | 改写旧状态时，应保护什么，收益量级够不够？ |

## 1. Worth Remembering：先检查事件接纳

[Worth Remembering: Surprise-Gated Robot Episodic Memory，v3](https://arxiv.org/html/2606.03787v3)；Nicolas Gorlo、Derek K. Wise、Alberto Speranzon、Luca Carlone。

**原机制。** §3.2 式1用局部 Gaussian 均值、标准差归一化每维绝对偏差再求平均；式3用 median/MAD 阈值，随后局部峰值与 NMS 触发事件。它不是普通预测 MSE，也不是直接精确计算 KL。§3.3 把事件片段提供给后续检索。[方法原文](https://arxiv.org/html/2606.03787v3#S3.SS2)

**我们的适配。** 这是有界历史项上的 WR-inspired 接纳／保留；固定 K 的容量处理、共享冻结 reader 和短 Gate 序列属于我们的协议，不能把启动失败直接归为原论文系统的失败。附录A可选读，用来核查 Gaussian 近似；暂时不需要细读整套机器人 QA 实验。

配着启动面板，回答三个问题：

1. cue 到达时 Gaussian 历史和 MAD 阈值是否已就绪？trace 中第一次失败发生在“事件没被确认”，还是“确认后被淘汰”？先沿逐步状态找断点，再看最终记忆。
2. 所有策略收到同样四帧静止前缀后，改变的是启动统计、任务信息还是 reader 的时间输入？用原始／前缀 trace 与896次固定子集时间平移检查区分它们。
3. 62个早期 cue 都被救回，但41个 MSE 改善、21个变差：为什么接纳成功不能单独证明保留策略或下游预测更好？每条轨迹额外执行4步；数据集最大逻辑状态增加32字节，本展示例自身峰值增加40字节，又应怎样记账？

**读完应能说清：这批数据支持短序列启动导致接纳失败；不支持“低惊讶关键线索被高惊讶干扰挤掉”已经得到验证。** 后一个假说需要事件已正常形成后的竞争淘汰证据。[完整分母、控制与成本](WR_STARTUP_DIAGNOSIS.md)

## 2. PaFu-KV：将“会注意它”与“用它有帮助”分开

[Past- and Future-Informed KV Cache Policy with Salience Estimation in Autoregressive Video Diffusion，v3](https://arxiv.org/html/2601.21896v3)；Hanmo Chen、Chenghao Xu、Xu Yang、Xuan Chen、Cheng Deng。

**原机制。** §3.2 式4–5按过去／当前／未来分区聚合 teacher attention；§3.3 的学生 head 输入 Q/K/V，§3.4 式6用 SmoothL1 回归该显著性标签。推理时按预测分数管理有界 KV。标签是 attention，不是删除后任务损失。[方法原文](https://arxiv.org/html/2601.21896v3#S3)

**我们的适配。** 帧级历史项与 PM residual bank 只借用监督机制；共享冻结 reader 既不是原论文的视频扩散生成器，也没有复现其完整训练。CV 内容失败组直接诊断的是**我们这个 reader 的输入利用**，不是 PaFu 原系统的失败证据。附录C可选读，用来核对训练和推理能访问的信息。

把内容失败和成功同时摆在面前，回答三个问题：

1. 只换内容、固定时间与动作后，失败例 MSE 从0.127822升到0.324210，成功例则从0.120658降到0.066891：相同输入路径为什么产生相反效果？检查各 horizon 的预测向量与 sibling 方向，不能只看真实未来视频。
2. 失败例实际正确 cue 的 attention 为0.999771，预测方向却不对。attention 标签可以告诉学生“它会被读取”，但是否知道 reader 的贡献方向、替代项或两项互补？下一步应在哪个相同 K+1 候选集合上比较 attention 排序与实际删除代价？
3. 固定内容后单独换时间戳的作用，是否与内容有交互？这张辅助面板只能说明输入敏感性，不能单独证明真实时间因果。若 oracle gap 只反映 reader 偏差、合成混搭输入或未来随机性，因果学生还能学到什么？

**读完应能说清：先证明 reader 能可靠利用证据，再把 attention 与边际任务贡献作公平对照。** 不能把删除损失换进 PaFu 标签后继续叫 future-attention baseline。完整72个可比案例中，正确 cue 相对匹配错误 cue 有48个更好、24个更差；原失败子集是21/27更差，不能据单例宣布模型统一学反。[因子诊断与混搭输入限制](GATE_CUE_FACTOR_DIAGNOSIS.md)

独立新场景的冻结确认也必须同时看：cost scorer 尚未确认优于 shuffled-cost，且比强 attention/diversity 更差；当前案例解释不等于方法优势。[独立确认结果](GATE_CONFIRMATION_AND_CASEBOOK.md)

## 3. Gated DeltaNet：结合 PM，区分改写与保护

[Gated Delta Networks: Improving Mamba2 with Delta Rule，v1](https://arxiv.org/html/2412.06464v1)；Songlin Yang、Jan Kautz、Ali Hatamizadeh。

**原机制。** §3.1 式8将状态衰减门与定向 delta 更新结合：α 控制旧状态整体衰减，β 控制当前 key 的修正。对照 §2.3 与表1理解这两个动作即可；§3.2 的并行实现暂时选读。[公式原文](https://arxiv.org/html/2412.06464v1#S3.SS1)

**与我们工作的关系。** PM 的 A2 α/β 更新属于已有机制参照，不能重新包装成创新；矩阵状态与离散历史 bank 的表示和读取不同，相同字节也不保证只比较了淘汰规则。这里不据此新增 CV baseline。

配着 [PM 四张案例卡](https://github.com/XintongYang1010/predictive-memory-lab/blob/experiments/bank-reference-20261022/docs/PM_CASE_GALLERY_ZH.md)，回答三个问题：

1. 卡片1删旧值后误差下降，但覆盖也消失，且不如零残差参照：这是“恢复正确新状态”，还是“避免读错”？何种新证据才足以支持改写？
2. 卡片2总损失下降而 signal 变差，卡片3同一 source 却有真实收益：能否在过去输入中区分状态变化、噪声与仍有效旧信息？应同时报告哪些保护和改写指标？
3. 卡片4旧 A2 局部修复的账面收益只有合并 MPE 的0.0125%：是否足以支撑主线？若找不到更大、可重复的预算匹配收益，应保留什么负结论？

**读完应能说清：何时需要忘、何时应保护，是已有门控机制上的任务问题；当前旧 A2 小效应并没有证明结构表达能力不足。** PM 保持 oracle L0 输入，不把本轮阅读扩展成新的感知排查。

## 阅读后的一页分析模板

每组只写以下六行，失败与成功都填：

1. **观察与比较：** 哪条证据、什么 query、同一个 reader、哪些输入保持不变？
2. **首个失败环节：** 接纳／保留／利用／目标，哪条 trace 或干预支持？
3. **指标：** 固定共同多 horizon 目标；再核查方向、signal 或任务指标，避免只追总 MSE。
4. **oracle 能力：** 同一候选集局部删除，还是额外读过历史档案／实际未来的事后选择？
5. **过去可学性：** 若过去完全相同而未来随机不同，oracle 的不同选择不属于学生可学收益。
6. **结论边界：** 单例与完整分母、开发与独立确认、固定 K 与总字节／时间成本分别是什么？

先读这三篇就够。当前最需要的是把这六行写清，而不是继续扩充相关工作列表。
