# Gate use-time 标签：反事实排序被修复，但当前学生没有泛化收益

本轮完成了固定预测器、固定容量下的 CPU 可行性实验。结果支持继续诊断监督目标，但**不支持把当前 use-time ridge 学生提升为方法，也不支持立即扩大诱饵实验**。本报告没有 A100 新训练结果；A100 的 reader 检查是独立实验。

## 实际比较的是什么

- 训练：原开发备份的 16 个 TRAIN 场景对、32 条 episode、1,282 个真实 K+1 parent。没有加入确认集训练样本。
- 标签干预：对每个 parent 的三个删除动作，分别不可逆地执行一次删除；随后统一使用冻结的 attention 策略继续到该 episode 的 query；同一个最终记忆集合同时评价 q+3、q+6，取平均 MSE。
- student：复用原 903 维因果输入、parent-centered ridge、固定 λ=1、TRAIN RMS≤1e−6 的统一支持掩码；本次没有架构或超参数搜索。有效支持维度 193。未来 query 时间、未来动作/图像、cue 注释只在离线 teacher/evaluator 中使用。
- 对照：local-cost、use-time-cost、future-attention-label、两种 cost 的 parent 内打乱标签，共五个同参数量学生；另跑原 reader attention 和 visual diversity。
- 评价：已反复检查的 seed-20261022 开发集，96 对、192 条 episode；不是新确认结果。旧 seed-20261023 只读取一个明确事后选定的讲解案例，不参加本次拟合或汇总。

这里的 `future_attention` 是为了隔离标签而拟合的**同架构 ridge 对照**，不是原 PaFu 适配的 MLP，更不是原论文系统。

## 代表案例：同一个 parent 的排序确实反转

episode `587c4cb6ee274da5e3a4b35d`，t=9，候选 `[0,1,5]`，第 5 帧是最后一份 cue，query=17。

| 删除动作 | 原 local 标签：t=12、15 | 固定 attention 续跑后的最终保留 | use-time 标签：t=20、23 |
|---|---:|---|---:|
| 删 0，保 cue | 0.218346 | `[5,13]` | 0.068828 |
| 删 1，保 cue | 0.210544 | `[5,13]` | 0.068828 |
| 删 5，丢 cue | **0.119606** | `[12,13]` | 0.127778 |

所以本例并非仅仅“scorer 没学好”：原标签本身就要求删 cue；改成在使用时刻评价后，两个保 cue 动作并列最优。

此处 0.127778 不等于原 masked-cost 整条轨迹的 0.142450，因为本实验在 t=9 之后统一切换为 attention。保 cue 的 0.068828 也不等于原 attention 的 0.062924，因为这个真实 parent 已经没有第 4 帧，反事实不能复活它。

## TRAIN 标签层面

只看“候选里恰好一份 cue，recent 中没有其它副本，因此一次删除能彻底丢失证据”的 parent。`pre-use` 指两个目标均不晚于 query；结果分支从 query+1 才出现。

| 标签时段 | parent 数 | local argmin 丢 cue | use-time argmin 丢 cue |
|---|---:|---:|---:|
| 两目标都在使用前 | 130 | 111 | 0 |
| 跨越使用时刻 | 162 | 85 | 1 |
| 两目标都在使用后 | 182 | 13 | 2 |

argmin 使用最老项打破并列，不能把表中每个 argmin 都解释成严格标签偏好。容差 1e−8 下，use-time 在全部 474 个风险 parent 中有 8 个最优集合**允许**丢 cue，0 个**要求**丢 cue；其中 7 个三动作全平。pre-use 的 130 个全部只在两个非 cue 删除动作之间并列，**没有一个最优动作会毁掉 cue**。因此 pre-use 的 111→0 不是 tie-breaking 假象。

全部 1,282 个 use-time parent 中，895 个最优动作有并列，442 个三动作全平。这在固定 continuation 下有合理来源：不同当前删除可能后来汇合到同一个最终记忆。标签准确率必须同时查看 regret，不能只报命中率。

## 学生学会了训练样本，却没有泛化

| 策略 | 旧 TRAIN cue 存活 | 旧 TRAIN MSE | 192 条开发集 cue 存活 | 开发集 MSE |
|---|---:|---:|---:|---:|
| local-cost ridge | 6/32 | 0.081632 | 141/192 | 0.112316 |
| use-time-cost ridge | **32/32** | **0.036685** | **74/192** | **0.125385** |
| future-attention ridge | 31/32 | 0.037937 | 126/192 | 0.120131 |
| 打乱 local-cost ridge | 未另测 | — | 126/192 | 0.110272 |
| 打乱 use-time-cost ridge | 未另测 | — | 110/192 | 0.112996 |
| reader attention | 未另测 | — | 192/192 | 0.110337 |
| visual diversity | 未另测 | — | 192/192 | 0.108966 |

use-time 学生在 130 个 pre-use 风险训练 parent 上，130/130 选择最优保 cue 动作，平均标签 regret=0；没有必要把在线失败归结为“这些训练标签拟合不了”。但其开发集 cue 存活为 ID 37/64、long-gap 28/64、distractor-shift 9/64。应先检查小训练分布到新场景的转移，而不是据训练表现宣布可部署收益。

开发集 use-time−local MSE 差为 +0.013069，按场景对 bootstrap 的描述性 95% 区间为 [+0.004477,+0.021088]；use-time−attention 为 +0.015048，[+0.007779,+0.021567]。当前学生确实更差。所有数值使用同一个冻结 reader；cue 存活、特征方向正确和任务收益仍是不同指标。

## 尚未隔离的关键混淆

冻结 reader 的原训练通过 `train_predictor_followup.py:132` 和 `train_predictor_pilot.py:154`，始终在 `rec['query']` 取预测起点。local-cost 标签却在每个 overflow 时刻 t 调用它。query 前的静态场景因此可能超出了 reader 的训练起点分布。

use-time 干预同时做了两件事：把目标放到证据实际生效之后，以及把 reader 评价起点搬回它训练过的 query 分布。**本轮不能把全部变化归因于更好的延迟信用分配。** 更强 reader 及多起点训练检查应单独完成，再判断哪一部分能成为方法贡献。

TD 没有实现。简写的“即时删除代价 + 下一步条目存活价值”还没有定义完整状态、终止条件、其它条目之间的交互和反事实后续策略，不能直接视为有效 Bellman 目标。

## 复算与证据

入口：`experiments/phase2/run_use_time_probe.py`。本次 CPU 运行约 11 秒，读取已有 A100 编码缓存，未启动新编码或新 reader 训练。三个语义测试覆盖不可复活、与完整流式 attention 的 tie-breaking 一致、拒绝非法容量/候选。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 experiments/phase2/run_use_time_probe.py --output /absolute/new-empty-output
python3 -m pytest tests/test_use_time_probe.py -q
```

结果在 `results/literature_diagnosis/use-time-probe-20261024/`：

- `protocol.json`：输入/源代码哈希、reader 绑定、固定训练和评价规则。
- `selected_case.json`：同 parent 的三个删除和两个 horizon。
- `training_parent_results.json` / `use_time_training_labels.npz`：1,282 个反事实标签及 continuation 终点。
- `student_fit_diagnosis.json`：固定拟合的训练排名、ties、regret 和训练 episode 在线重放。
- `online_results.json` / `online_trace.jsonl`：全部七个策略逐 episode 指标和不可逆轨迹。
- `summary.json`：分组统计、结果哈希、时长、限制和输入数值复现检查。

最初的严格 bitwise 输入重建检查未通过，失败目录保存在 `build/use-time-probe-20261024/failed-exact-input-check/`。最终训练仍复用原存档输入，没有替换；14 个抽查 parent 的本地投影重建差异通过 rtol=1e−5、atol=1e−3，逐项最大差异写入 summary。运行中的 NumPy matmul 发出浮点警告，但已保存预测、参数和标签均有限；输出没有把这些警告当成成功验证的替代品。需要完整复算时仍依赖本地 `build/` 的原模型和特征缓存。

后续导出补齐了[原 1,282 个 TRAIN parent 的精确输入与标签](../results/literature_diagnosis/use-time-label-audit-20261024/training_inputs_labels.npz)，哈希与本实验 protocol 一致。它与本目录的 use-time 标签、参数及 parent 记录可以支持无需 GPU 的标签／拟合复查；完整在线重放仍需要相应轨迹缓存。
