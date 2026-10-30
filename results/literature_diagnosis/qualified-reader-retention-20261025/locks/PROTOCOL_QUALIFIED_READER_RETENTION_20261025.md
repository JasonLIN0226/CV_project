# 合格 reader 上的终局保留比较：运行前锁定（开发阶段）

写于 2026-10-25（纽约时间），在生成任何 TRAIN/VAL 标签、训练任何学生、计算任何 16 计划因果选择结果之前。此前只看过"实际计划近似版"无训练规则的结果（见 `qualified_reader_rules_lock.md` 及其结果），本协议不因那些结果调整任何门槛或方法。

## 数据与边界
- 特征：本地 `build/terminal-download-20261025/extracted/run/data/features.npy`，SHA256 `9090806a…e6b1`，与 HPC 运行记录一致；在本机按 TRAIN basis 投影为 128 维后传入（`projected_features.npy` `8d3591e0…1552`、`output_basis_coords.npy` `5970f4f2…a755`）。
- 复现检查已通过：DEV 上 multihead / item_attention / original 的 native 与 full 系数与 A100 存档差 ≤3.0e-4，pooling 权重差 ≤8.7e-6。
- 划分：TRAIN 20261101（48 family）、VAL 20261102（24）、DEV 20261103（24，开发集，已用于 reader 资格与实际计划近似分析）。**CONF 20261104 不打开。**
- Reader：主 `multihead`，次 `item_attention`（两者均通过全部 K2/K3 门槛）；`original` 只作参照。单一 reader 训练 seed17，结论受此限制。

## 方法（同一冻结 reader、R=4、三候选删一、K=2）
- 不依赖 reader：FIFO（删 slot0）、family 固定随机、visual diversity（沿用终局运行的 DEV 决策，特征相同）。
- 依赖 reader：
  - attention：K+1 集合在 query 处的 pooling 权重（按头平均），删最小者；
  - medoid、完整集合敏感度：16 个 TRAIN 参考动作，逐计划、逐 horizon 先算距离再平均；选择器不读实际未执行动作；
  - oracle：32 个评价计划上期望损失最小的删除（容差 1e-6）。
- 学生（`fit_terminal_students.py` 原样，seeds 17/29/43，1000 步，按各自 VAL 标签 CE 选 checkpoint）：
  - 独立 / DeepSets × {删除代价, future-attention, 打乱删除代价}；
  - 只看外观的独立控制（再生成 RGB 的颜色直方图/均值/方差，与原实现一致）；
  - **新增两臂（蒸馏）**：独立 / DeepSets，以 16 计划完整集合敏感度分数为标签（小者删），走同一训练代码。
- 标签：TRAIN 删除代价 = 16 参考计划平均损失；VAL = 32 评价计划平均损失；future-attention = 冻结 reader 以真实未来帧为 query 的 pooling 权重，按计划和 horizon 平均；敏感度标签 = 16 参考计划的完整集合敏感度。损失与距离都在输出基底张成空间内计算，基底外残差对同一 parent 的三个删除相同，不影响任何选择、regret 或居中标签。

## 终点
- 主：必要证据丢失率（按任务、按外观条件）；联合 h3/h6 期望损失 regret（相对 32 计划 oracle）。
- 学生先在 parent 内对 3 个 seed 平均，再按 24 个 family 做 2000 次 bootstrap（seed 20261107）。
- 成本：每次决策的 reader 预测次数；学生参数量。

## 事先写下的预测
| 方法 | R 冗余 | D 诱饵 | S 协同 |
|---|---|---|---|
| medoid | 低丢失 | **高丢失**（近似版已见：两个缺证据预测误差同向） | **高丢失** |
| 完整集合敏感度 | 低 | 低 | 约三成丢失（K3 预测不够"确定"时失效） |
| attention | multihead 差；item_attention 在 D 上好 | 同左 | 差 |
| oracle（MSE） | 低 | 低 | **10–25% 删掉必要证据** |
| 删除代价学生 | 接近 oracle | 接近 oracle | 不优于 oracle 的语义丢失率 |
| 独立学生 | ? | ? | 丢失 ≥50%（设计上界） |
| 蒸馏敏感度学生 | 接近敏感度 | 接近敏感度 | 接近敏感度 |

## 判读规则
1. 若某个无训练规则在三族都接近 oracle，学生只在"摊薄成本"意义上有价值。
2. 只有 DeepSets 删除代价学生在三族上同时优于敏感度、attention 与独立学生，并且后续在 CONF 上复现，才支持 proposal 核心主张。
3. 若 MSE oracle 本身在 S 上频繁删除必要证据，则任何以 MSE 删除代价为标签的方法在 S 上都有相同的上限，报告为监督信号本身的局限，而不是学生失败。
4. 全部结果（含失败）保存；不因结果改方法、门槛或 seed。
