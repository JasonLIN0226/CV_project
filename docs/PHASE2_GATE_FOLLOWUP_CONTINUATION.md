# Gate：修正预测器后的 Phase 2 开发续跑

在有用历史已得到开发集支持的 **Gate / unit-columns / epoch 35** 冻结预测器上，实际重跑六类保留策略，并生成对应的有限记忆删除监督。CPU 运行完成，用时 **2.03 秒**；未训练 Phase 3 scorer，未读取测试数组，未关闭 PushT baseline planning 缺口。

这次使用新的显式 `FollowupGateEvaluator` 绑定，不修改旧 `FrozenPredictorEvaluator` 的初始化检查或旧结果。它核对完整协议、缓存/normalization、训练集 basis、epoch 35 的验证选择、模型参数与源码哈希；评估复用已有不可恢复删除、因果 surprise 和等 K 分支语义。

## 六类 baseline：实际结果

8 条 validation 轨迹、4 个独立配对场景，形成 48 条 episode-policy 记录与 96 个 horizon loss。误差为完整归一化特征空间 MSE，越低越好。

| 策略 | h=3 MSE | h=6 MSE | 最大逻辑持久字节 |
|---|---:|---:|---:|
| FIFO | 0.099427 | 0.108295 | 123,615 |
| Reservoir | 0.085370 | 0.094922 | 123,629 |
| Temporal buckets | 0.099427 | 0.108295 | 123,627 |
| Visual diversity | 0.071807 | 0.079869 | 123,627 |
| Prediction surprise | 0.103442 | 0.110227 | 131,466 |
| Attention | 0.073832 | 0.078564 | 123,619 |

相比旧弱预测器的小规模结果，这次 diversity/attention 的平均误差较低。这个变化说明 baseline 排名依赖预测器是否有效利用历史；不能据此宣布一种保留策略普遍胜出，也尚未比较 learned scorer。

各策略同为 **R=4、K=2**，但额外状态字节不相等。Surprise 保存至多 3 个待完成预测的有界数值摘要；其预测仅在三个动作实际执行、目标观测到达后计算。表中不含 Python 分配器和临时 tensor 工作区，共享预测器参数/缓冲单独记账。因此不是等总字节公平比较。

## 删除监督：实际产物

- 四类 parent 策略：FIFO、reservoir、temporal buckets、visual diversity。
- 40 条 Gate train/validation 轨迹，每条在已声明 query 处捕获真实 K+1=3 overflow；160 个 policy-episode origins 去重为 **118 parents**。
- **94 train parents / 24 validation parents**；对应独立统计单位是 **16 train pairs / 4 validation pairs**。
- 每个 parent 分别删掉三个候选之一，每个分支严格保留 K=2，预测 h=3/6。
- 共 **354 forks、708 个有限损失值**，无排除项。全部 708 个损失都从保存的 scorer inputs 重建并重新调用冻结模型，结果逐值完全一致；所有预测向量保存在本地 label 目录。
- `scorer_inputs/` 只含过去/当前特征、时间戳及过去动作摘要。未来目标、预测向量与损失单独保存在 `labels/`；生成过程未访问 hidden simulator state 或 privileged evidence。

开发数据保存在 `build/history-diagnosis-20261020/gate-phase2/`，未覆盖旧 pilot。可发布的结果与逐文件 raw/published SHA256 映射在 [gate_continuation](../results/phase2_history/20261020/gate_continuation/manifest.json)，大模型和 NPZ 数据未加入 Git。

## 绑定及复现

- checkpoint SHA256：`302221849cb424a1d846c3af2932d0ce1f7bdc22965ab23d330c36a827c174d1`
- followup protocol SHA256：`10306756e360abc1ebbdad6393d7482f435523a4b9abb57632a79e75efd6fba7`
- basis SHA256：`9c77bf393000d6911f8cf16c3856d0999a1c9d00f9fbed4e0e5ed3b13aedd320`

在仓库根目录执行；输出目录必须为新的空目录：

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  experiments/phase2/run_gate_followup_phase2.py \
  --snapshot build/hpc-20261020/development-assets \
  --followup build/history-diagnosis-20261020/followup-blas1 \
  --output build/history-diagnosis-20261020/gate-phase2-reproduction
```

4 项新增绑定测试通过，包括拒绝错误 task/epoch/protocol、偷偷替换 basis 或模型 buffer、改变 train membership。实际运行再核对完整 artifact 哈希、冻结状态与输入未变。另有独立审计从保存的预测向量与原始特征重算全部 96 个 baseline 损失和 708 个删除损失，核对 1,068 个连续有界状态事件、分组统计与来源哈希；这次独立审计未重跑推理，和运行器自身的全分支推理重算分别记录。运行器有 300 秒硬上限；实际线程为 OpenBLAS=1、Torch intraop=4。

## 结论边界

这是 Gate 的 Phase 2 开发交付前进了一步：已有可重现的六策略结果，以及与质量改善后的冻结预测器配套的干净监督产物。验证集已用于训练 checkpoint 选择和此次开发检查，只有 4 个独立 pairs；没有独立泛化结论。Motion 仍没有确认有用历史优势，未用其弱模型继续扩大监督。完整 Phase 2、学习式 scorer 的优越性及 PushT planning 均未完成。
