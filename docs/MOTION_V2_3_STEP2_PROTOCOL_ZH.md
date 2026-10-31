# Motion v2.3 第 2 步 protocol

日期：2026-10-31。训练前登记，依据已推送的 [v2.3 修订](MOTION_V2_3_AMENDMENT_ZH.md)。本文件和 `experiments/motion_v2/protocol_step2_v2_3.json` 连同源码、参数、输入 SHA256 提交推送后才运行实际 validation 下界计算、部署验收或科学入口。v2.3 软件测试已按修订先执行，28/28 通过（一次，0.44 秒）；测试只使用人工数组。源码与测试均固定为实际通过版本。

## 沿用的科学方案

除本文件明确新增的评估量和作业检查外，[v2.2 第 2 步方案](MOTION_V2_2_STEP2_PROTOCOL_ZH.md) 的数据、模型、训练和估时参数不变。旧 protocol/JSON/停止记录保持原字节，其旧源码锁指向相应历史提交；新的源码锁只写入本版本。

只使用现有 seed 20261030 的 train 512 对、validation 96 对，共 1,216 episode、39,398 帧。manifest SHA256 为 `9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a`。生成器和 Step 1 审计不改、不重跑，不读取或生成 test。

官方冻结 I-JEPA 为 `facebook/ijepa_vith14_1k`，revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e`，FP32、batch 2、grid2 四格池化 5120 维。归一化只用全部 train 帧，std floor 0.05；原 Phase 1 `fit_basis` 只用 train，rank 128。模型使用原 `DynamicsPredictor`，hidden 128、dropout 0。

预测器 seeds 17/29/43，各 120 epochs；每 epoch 四训练条件各覆盖全部 train：recent-only、full-history、固定 b、random K=2。batch 64 episode=32 pairs；每 epoch 64 steps，总 23,040 optimizer steps。AdamW lr=0.001、weight decay=0.0001、clip=1、StepLR 40/gamma 0.5，原 loss=feature MSE+2×paired residual MSE+0.5×rank loss。非空 memory 训练整段 dropout=0.25，同 pair 共用决定。只按 validation 四训练条件、两 horizon 等权 feature MSE 选最小 checkpoint，严格更小才更新，并列取最早。

decoder 为原 `LinearReadout`，train 两 horizon 共 2,048 行，GroupKFold 5 折按 pair 分组；ridge 候选 `[0.01,0.1,1,10]`，按整体 OOF xy MSE 选择、并列取小者；选定后全 train 拟合并冻结。validation 不参与 decoder 选择或拟合。真实 validation target feature 的 decoder loss 只作诊断。

## 固定历史及位置误差

所有条件共享最近四帧，memory 均早于 recent 窗口；P 为 t_c 至 end 的可见帧，S 为 t_c 之前的可见帧。

| 条件 | memory |
|---|---|
| a | 空（recent-only） |
| b | 第一张 P 帧 + end |
| c | 第一张 S 帧 + 最后一张 S 帧 |
| d | 最后一张 S 帧 + end |

a/c/d 两个 sibling 的所有模型输入张量必须完全相同。隐藏状态只用于特权选帧、train decoder 标签及评估，不能作为预测器输入。feature MSE 为归一化 5120 维平均平方误差；position MSE 为 `((x_hat-x)^2+(y_hat-y)^2)/2`。

## CPU 方向下界与必需判据

CPU helper 在训练前只读 validation 的 evaluation_only target 位置，按冻结的 pair/sibling/horizon 顺序保存真实位置 `[96,2,2,2]`、B `[96,2]`、IDs、源码/protocol/manifest/数组及文件哈希和 status。不拟合模型，零 GPU。新目录必须不存在；文件缺失、非有限数、顺序或哈希不符即停。runner 在训练前将 artifact 与相同 validation 真值逐元素核对；最终统计用同一真值计算 B 并核对，B 不影响训练、checkpoint 或 decoder。

每 pair/horizon：`m=(p0+p1)/2; B=mean_sibling(mean_xy((m-p)^2))`。任意共同预测 q 的 pair 误差等于 `B+mean_xy((q-m)^2)`，因此至少为 B。这是同 pair 的预测相同条件下的位置误差下界，不宣称所有含潜在泄漏的模型必然输出相同预测。

先在 pair 内平均 sibling loss，再平均三个固定训练 seed。对 96 对做 2,000 次 pair bootstrap，seed 901，percentile 2.5%/97.5%，linear quantile。所有绝对指标、差值、horizon、每 seed 附表和 decoder 诊断使用同一个索引数组。B 没有模型 seed 轴，不是额外独立样本。

| 对比（位置 MSE） | h=3 | h=6 | 用途 |
|---|---|---|---|
| d−b | 95% 下界 >0 | 95% 下界 >0 | 原主判据 |
| c−b | 95% 下界 >0 | 95% 下界 >0 | 原次要必需判据 |
| B−b | 95% 下界 >0 | 95% 下界 >0 | 新方向必需判据 |
| d−B | 报告均值与 CI | 报告均值与 CI | d 的额外误差诊断 |

六项必需判据全部通过才算通过。另报告 a−b 和原 feature MSE 对比；B 没有 feature MSE 的对应定义。输出 a/b/c/d×h=3/6 完整均值/CI，按 horizon 并列 B 与真实 target feature 的冻结 decoder position MSE；每 seed 也保留同样的结果表和对比。

## 部署与作业一次性

新根目录 `/scratch/xy3257/motion-v2-20261031-deployment2`；旧 deployment1 全部只读保留。使用旧 venv 绝对路径，以 `-I -B` 禁止写 pyc，TMP/cache 全部指向新目录。复用环境的 import、关键版本、活动依赖检查失败即停，不安装、不换版本。数据包 SHA256 先匹配登记值，再只向新目录解包、逐 NPZ 核对 manifest。当前 partition/account 和 sbatch wrapper 另外在提交前登记，不能用历史值代替当前集群证据。

科学入口仅调用一次 `run_step2.py`，输出目录必须全新，禁止覆盖/恢复/删除。sbatch 请求单 A100、`--no-requeue`、时限≤7,200 秒；提交后立即保存 scontrol。运行前及每至多约 30 秒的预算检查核对原生作业、起始时间、GPU 数量及 Restarts；**Restarts>0 即停**。Requeue 字段保留在证据中，wrapper 的 no-requeue 请求与拒绝已存在输出共同保护一次性原则，不添加用户未要求的 Requeue 单字段结果门槛。

无科学入口的 OOD/CPU 辅助会话调度差异按 v2.3 非阻塞记录。其他可能影响科学结果或一次性原则的哈希、revision、输入/配对审计、GPU 型号数量、预算、NaN/缺项、已存在输出及需要本人认证的差异仍阻塞。保留所有失败现场，不修阈值后重跑。

## 资源估时与结束

预算不变：登记上限 7,200 A100 秒，用户总上限 14,400 秒，既有实际 A100 消耗为零。39,398 帧÷27.15545 fps≈1,451 秒；360 epochs×5.72894 秒≈2,062 秒；其他工作预留 3,600 秒；合计 7,113.250553 秒。编码完成至少 64 帧后更新速率，完成至少一个 epoch 后更新训练率。实际或完整剩余序列的估计超 7,200 秒即停；不降精度、缩预算或重跑。GPU 分配中的初始化、下载、CPU 工作和失败占用都计入。CPU 上提前计算 B 不消耗 A100；GPU 中读取和校验它的开销计入。

输出 scontrol/sacct、阶段时长、环境版本、checkpoint、decoder、逐 pair loss、bootstrap indices、完整指标/每 seed 附表、status.json、SHA256；最终 GPU 用时以原生 Slurm 结算为准。若提前停止，未计算项标为未运行，不能按零值或科学失败解释。无论判据通过或失败都停在第 2 步，不训练保留策略、不进入第 3 步。
