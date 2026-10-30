# 新场景确认：先冻结，再读取 seed-20261023 结果

本轮只确认已有保留探针能否在独立新种子上产生稳定收益。所有旧数据（包括前一轮96对）
都视为开发数据；不再用新场景重训、选 λ、调标准化下限、换 reader 或挑选最好的 probe。

机器可核验的协议是
[locked_protocol.json](../results/literature_confirmation/gate-seed20261023/locked_protocol.json)，
SHA256 `977feb00d44ce388157e05aaa1ec0f28f1960f182b4e46068618571c2615c734`。
它在新数据生成和读取之前写入，绑定 encoder、新运行代码、旧参数、PaFu checkpoint、
epoch-35 reader、basis、normalization 和已有场景元数据。源文件或参数改变后，运行器会拒绝继续。

## 唯一主要判断

使用**原先已经确定的 masked cost ridge**，在全部96对新场景上的在线 MSE，分别减去：

1. 同 mask、同模型容量、同输入的 shuffled-cost ridge；
2. 既有 reader-attention；
3. 既有 diversity。

只有这三个配对差异的双侧95% bootstrap 区间上界**全部小于0**，才认为当前冻结探针在
这一个共同 feature-MSE reader 下获得了新种子上的保留收益。不能挑最好的条件、seed、
探针或其中一个对照替代此规则。这是三个要求的合取，不是从多个显著结果中选择一个。
若失败，如实报告未建立优势；不自动继续搜索模型，不能宣称研究方法已成立。

即使通过，也只确认这个冻结系统和生成器家族的收益。reader 的任务意义、cue 内容与时间/
动作路径的归因必须结合单独诊断；单个 feature-MSE 数字不能替代物理任务或规划验证。

## 固定数据和对照

- 独立 seed **20261023**；ID、long gap、distractor shift 各32对，共96对/192条轨迹。
  场景参数保持原生成器配置，仅更换 seed 和身份命名空间。元数据 scope/split 是
  `literature_confirmation`，不是原 Phase 1 的正式测试集。
- 排除所有旧 metadata 及上次 seed-20261022 的场景种子、ID、pair 重叠。使用同一 pinned
  I-JEPA encoder、旧 TRAIN normalization 和固定 epoch-35 reader。
- 在线13个 arm：原始与 masked 的 cost / attention / shuffled-cost 六个冻结 probe；
  FIFO、diversity、reader-attention、原 WR；PaFu 的原17/23/41三个 seed。
  这些参数原样复用。WR 保持原启动规则，前缀控制属于之前的诊断，没有混入此轮主比较。
- Mask 仍是原 TRAIN RMS ≤1e-6 的710/903列系数置零，不能基于新数据修改；没有拟合调用。
  原 probe 作为负对照完整报告，不因其结果更好或更差而取代主要 arm。

## 同父集合与不可恢复重放

所有策略都使用 R=4、K=2，以及同一个删除动作的 `L=(MSE_h3+MSE_h6)/2`。每个 online
对象只接收当前帧和刚执行的一个动作；未来观察、动作序列、证据与 sibling 标签都在评估
层，不能进入策略状态。已删历史不能恢复。

共同 parent 来自实际 FIFO / diversity / reader-attention 最终 K+1 状态的去重并集，
不由拟合 probe 的输出挑选。对其中每个 parent 枚举三个删除动作，比较六个 probe、
FIFO/diversity/attention 和三个 PaFu student 的同集合 regret。WR 的 Gaussian/controller
状态依赖整条在线路径，因此只评价其自身重放，不伪造一个“相同候选但任意 tracker”对照。
本轮只做局部删除 oracle，不重新做全档案事后 K2 搜索。

所有 online 行保留 h3/h6 损失、真实 resident/candidate 时间戳、局部 regret、逻辑状态
与参数字节、延迟、cue 第一次完整信息丢失和结尾存活率。误差同时沿 sibling 未来分支差异
及正交方向分解，报告分支正确率；这些是只在评估时使用的描述性任务检查，不用于选参。

先平均同一 episode 的多个共同 parent，再平均 sibling 两条轨迹，以96个独立配对为
bootstrap 单位，5,000次、固定 seed20261022。分条件及其他对照为次要描述结果，不做
多重检验校正，也不据此替换主要判断。PaFu 三个 seed 分别保存，另报告 seed 均值对照。

## 执行与边界

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/phase2/run_gate_confirmation.py evaluate \
  --protocol results/literature_confirmation/gate-seed20261023/locked_protocol.json \
  --data build/literature-confirmation-20261023/data \
  --output build/literature-confirmation-20261023/evaluation-v1
```

原运行结果与参数不覆盖。新 encoder 需要实际 A100 GPU/UUID 证明；在线策略和 oracle
使用本地 CPU。新场景的失败不能被“只展示最坏几个视频”替代；portable 归档要保留完整
online/common-parent 行、配对区间、固定主判断、数据来源和文件哈希。

代码入口：[run_gate_confirmation.py](../experiments/phase2/run_gate_confirmation.py)。
数据契约：[gate_confirmation_data.py](../experiments/phase2/gate_confirmation_data.py)。
测试包含未来扰动不改变在线选择、严格 split/seed 隔离、固定 mask、重复 cue 信息生存和
三项合取判断。此文是执行协议，不表示实验已经完成或发现了收益。
