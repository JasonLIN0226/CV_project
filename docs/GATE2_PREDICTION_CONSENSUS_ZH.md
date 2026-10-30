# Gate2：先检验预测一致性，再判断是否需要学习保留器

2026-10-25。本文补充既有Gate2开发实验，不替换其运行前协议。**不训练scorer、也不读取实际未来动作的预测一致性规则，将丢唯一证据从attention的101/192、diversity的48/192降到2/192，平均regret降至0.006753。** 它用同一个冻结reader，对16个TRAIN参考动作计划分别比较三种删除的预测。这个结果削弱了“必须学习集合条件scorer”的必要性；它尚未通过打破外观捷径的新任务或独立确认。

## 对上一轮判读的修正

- Attention 的失败比只说“存在差距”更明确：101次丢唯一证据中，这条证据全部取得严格唯一最低的读取权重。重复物理a轴时，两种外观分别失败37/48；重复b轴时为24/48和3/48。它在本DEV上比终局均匀删除的56/192和FIFO的64/192丢得更多。这里的attention是冻结reader的读取分数，仍不是PaFu学生。
- Uniform的192条轨迹不是192次独立抽签。一个family的八条轨迹共用一次选择，共24次独立family选择；实际7个family失败，均匀删除的理论期望为8个，即64/192。不能把56当作随机方法的理论水平，也不能按192个独立样本算显著性。
- Diversity在“特殊副本外观＋重复b轴”48/48失败、其余0/144，是预先测量的特征距离结构所预测的机制：当两份语义重复证据距离最远，最大距离规则保留它们。应突出这一机制，同时保留它尚未跨renderer/encoder验证的范围。
- 后续主终点采用必要证据丢失率和联合h3/h6 regret。A与A′的精确选择作为次要诊断。29/48 quartet共享精确最优副本不是终局研究的停止条件；它也不是不可学习的证明，因为不同sibling的过去内容本来不同。
- 早期静止reader检查失败限制的是早期信用标签；已经通过的query内容检查足以支持本次终局比较。两条问题脱钩，不再串行等待早期reader修复。

## 不训练的规则到底算什么

在同一个三候选parent中，分别删除第0、1、2项，使用同一冻结reader得到三个预测向量，每个都覆盖h3和h6。令

\[
D_{ij}=\operatorname{mean}_{h,d}(p_i[h,d]-p_j[h,d])^2,
\qquad i^*=\arg\min_i\sum_jD_{ij}.
\]

执行删除动作i*。它选的是三个反事实预测中的medoid，即与其余预测总距离最小的那个；不是选择最接近真实未来的预测，也不是把不同horizon的不同删除动作拼在一起。exact float64 argmin按原生时间顺序打破精确平局，1e−6容差集合仅报告，不改变动作。

纯选择函数只接收三个预测向量。真实目标、cue语义、sibling编号和实际删除损失不进入选择。规则和输入哈希先保存到[rule_lock.json](../results/literature_diagnosis/gate2-prediction-consensus-20261025/rule_lock.json)，再计算结果；不过原DEV早已被查看，所以这只是**既有开发集上的新诊断规则锁定，不是独立预注册确认**。

## 完整向量检查：已经完成

| 终局规则 | 丢唯一证据 | 平均regret |
|---|---:|---:|
| FIFO | 64/192 | 0.026728 |
| 均匀删除，本次固定抽样 | 56/192 | 0.032342 |
| Diversity | 48/192 | 0.038896 |
| Reader attention | 101/192 | 0.037452 |
| 完整预测向量medoid，实际计划动作 | **2/192** | **0.006423** |
| TRAIN动作平均medoid，选择器不读实际未来动作 | **2/192** | **0.006753** |
| 四个真实未来距离坐标medoid，特权代理 | 12/192 | 0.007647 |

完整向量规则MSE=0.106966；同parent hindsight oracle=0.100543。24个family、2,000次重采样的描述性95%regret区间为[0.003885,0.009267]。相对diversity的平均MSE差为−0.032472，区间[−0.043280,−0.021726]；相对attention为−0.031029，区间[−0.039789,−0.022670]。不将相关的192条轨迹当独立重复，不作确认性显著性主张。

两个medoid版本都没有精确或1e−6容差平局。完整向量的两次丢唯一证据都在特殊外观、重复a轴、顺序[A′,A,B]，来自两个不同family，选择分数间隔分别0.02876和0.02013，不能归因于数值平局。尽管两次恰好仍被四原型分类判对，必要信息已经丢失，因此192/192的该分类准确率不能替代保留终点。

评审提出的“未来原型坐标代理3/192”没有在本次明确公式下复现：以归档的四个联合未来MSE距离为坐标，再取平方距离medoid，得到12/192。**评审随后澄清其代理使用开平方后的欧氏距离，二者定义不同。** 未搜索变体去匹配3，也不再追该代理数字。完整向量和本地MSE坐标代理有161/192次选择相同。代理本身使用真实未来定义坐标，不作为可部署对照；完整向量和TRAIN动作库结果不受此澄清影响。

复算代码为[audit_consensus.py](../experiments/phase2/audit_consensus.py)，[summary.json](../results/literature_diagnosis/gate2-prediction-consensus-20261025/summary.json)保留所有style×axis分层、family区间、随机参考；[逐行记录](../results/literature_diagnosis/gate2-prediction-consensus-20261025/episode_results.json)保留三种预测距离、分数和动作。独立读取原始NPZ重算192次选择、平均regret及101次attention严格最小检查，一致。

## TRAIN动作平均检查：已经完成

从120个TRAIN family按排序后的ID、固定NumPy seed20261025，无放回抽16个family，取它们的query后6步动作。每个family的四sibling×两style动作相同，已检查。该动作库对全部DEV parent相同，先锁定，再进行CPU推理；不根据DEV表现选计划或调样本数。

选择器接收当前/历史特征、年龄和已执行动作摘要，不接收episode ID、语义角色、真实未来特征、删除损失或实际未执行动作。对每个参考计划a计算三个预测，再取

\[
\bar D_{ij}=\frac1{16}\sum_a\operatorname{MSE}(p_i(a),p_j(a)),
\qquad i^*=\arg\min_i\sum_j\bar D_{ij}.
\]

先算每个计划下的距离再平均，避免先平均预测向量时不同动作相互抵消。全部192个parent先锁定动作，评价器才读取实际计划下已经归档的预测误差。将DEV未执行动作置零或改成异常值，192个选择器输入均逐字节不变；同输入重复推理一致。独立代码审查确认输入与原reader的年龄、动作摘要、候选顺序一致，选择接口没有未来信息路径。这是**终局共同parent上的因果输入策略**，不是完整在线episode部署验证。

结果：普通外观0/96丢唯一证据，特殊外观2/96；总体MSE=0.107295，regret=0.006753，family描述性95%区间[0.003974,0.009968]。相对diversity的配对MSE差−0.032143，区间[−0.042782,−0.021575]；相对attention−0.030700，区间[−0.039237,−0.022599]。与实际计划完整向量规则有189/192次选择相同。零精确/容差平局，两次语义失败仍不能被192/192的原型识别率掩盖。

成本不是免费的：16计划×3删除=每个决策48个reader预测样本，192个parent共9,216个。Mac CPU四线程批量推理加距离计算合计2.031秒，中位数10.669毫秒/决策，p95=12.284毫秒；不含checkpoint加载、数据I/O和结果写盘，也不是A100延迟。另有benchmark的96个参考预测和3个实际计划加载检查预测，不混入9,216的主结果计数。

动作库额外持久数据768字节；这不是总策略内存。reader参数861,188字节、buffer5,283,840字节，CPU实现还持有NumPy basis和normalization副本5,324,800字节；上述实际保留数值数组合计11,470,596字节，详见[内存账目](../results/literature_diagnosis/gate2-causal-consensus-20261025/implementation_memory_accounting.json)。每次参考批量的输入tensor约7.06MB、预测约1.97MB、float64距离输入副本约3.93MB，均是明确数组的下界；进程峰值RSS约718MB包含框架和归档映射，不等于有界记忆状态。后续必须同时报告K项存储、控制器/动作库、共享reader与额外计算，不能把这个规则与单次小scorer称为等算力。

只对首个parent的三种实际计划预测做了CPU/A100加载一致性检查：最大特征差6.20e−6、最大loss差7.38e−8，三种删除完整排序一致；没有声称对所有实际计划重新推理。主结果的16计划预测全部有限；NumPy/Accelerate的矩阵运算浮点告警保留在原日志。独立审计从377MB保存预测重算全部192次距离/动作，最大分数差1.11e−16，原生损失和语义标签全部一致。

代码：[run_gate2_causal_consensus.py](../experiments/phase2/run_gate2_causal_consensus.py)与[audit_gate2_causal_consensus.py](../experiments/phase2/audit_gate2_causal_consensus.py)。结果：[协议](../results/literature_diagnosis/gate2-causal-consensus-20261025/protocol.json)、[动作锁定及成本](../results/literature_diagnosis/gate2-causal-consensus-20261025/choices_locked.json)、[汇总](../results/literature_diagnosis/gate2-causal-consensus-20261025/summary.json)、[独立数值审计及全部分层](../results/literature_diagnosis/gate2-causal-consensus-20261025/independent_audit.json)。

## 必须保留的下一步区分

一致性不是正确性或反事实效用。在[A,A′,B]中，三种删除有两种留下互补证据，只有一种丢唯一证据；这个结构有利于“排除离群预测”。**后续推导修正：** 普通加性诱饵[A,B,D]即使只有一种删除完整，medoid仍可能成功；缺两轴的预测误差正交时它必选删D。更明确的结构反例是XOR协同：若两个缺信息预测都是同一条件均值，它们会使medoid排斥唯一完整预测。推导及精确符号预检见[协同设计审计](GATE2_SYNERGY_DESIGN_AUDIT_ZH.md)，未声称视觉reader已实现该理想行为。

不能把诱饵直接追加成[A,A′,B,D]后仍声称是K=2的相同三候选单删除问题。修订后的研究采用分别报告的冗余、诱饵和关系XOR协同三族，各自K=2、三个候选；并增加相对完整K+1预测的敏感度对照。详见[合并研究草案](GATE2_COMBINED_STUDY_DRAFT.md)。草案不是已执行的确认实验，完整K+1 reader也尚未训练和验证。

主要比较包括现有强基线、动作平均的无训练预测一致性、独立逐项学生、集合条件学生、future-attention学生、打乱标签和仅外观控制。无训练规则或独立学生足够时，报告集合条件学习不必要。只有集合条件学生在新确认数据中胜过前两者，才支持proposal相应主张；增加方法复杂度不是默认结果。

## 与CAOTE、ReST-KV及PM的关系

[CAOTE §3.1](https://arxiv.org/html/2504.14051v6)以删一个KV条目前后的attention输出变化计分，包含attention权重与value偏差。[ReST-KV §3.2及附录A](https://arxiv.org/html/2605.08840v1)同样控制KV删除造成的输出重建误差，并加入时间/空间平滑。二者说明“按输出影响淘汰”已有直接先例。它们参照删除前完整attention输出；本次规则比较三个等K保留集合的世界模型未来预测，使用medoid而非完整三项reader作为真值参考。这个区别不能单独证明新颖性，当前应把它作为学习型scorer必须面对的强对照。

PM V6.1的BF-Delta依据已经观察到的当前残差，比较保持/重置假设的预测证据并产生遗忘门；这里依据假设动作下的多个删除预测之间的一致性。可共用“冻结预测模型参与记忆决策”的问题意识，不能称为同一个算法，也不能把CV开发结果当作PM收益。PM本轮只更新连接说明，既有V6.1实验范围不变。

## 复算与公开归档

[本次诊断Release](https://github.com/JasonLIN0226/CV_project/releases/tag/gate2-consensus-20261025)保存377MB原始参考预测、因果输入、动作库、规则锁定、源码、逐样本结果与本报告；模型/checkpoint及原始实际计划预测依赖[原Gate2完整Release](https://github.com/JasonLIN0226/CV_project/releases/tag/gate2-terminal-pilot-20261025)，未重复打包。包内README给出精确命令与路径约定。

复算分为两层：`audit_consensus.py --execute`和`audit_gate2_causal_consensus.py`只读取保存预测，验证选择与指标；`run_gate2_causal_consensus.py prepare/benchmark/score/evaluate`会在CPU重新推理冻结模型，需要原Release的完整run目录。复制到新位置时使用prepare生成新路径的锁定记录，输出到新目录；不要覆盖原来的结果或伪装成新的独立确认。所有数值仍属于旧Gate2开发集。
