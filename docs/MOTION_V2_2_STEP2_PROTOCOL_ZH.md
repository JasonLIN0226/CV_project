# Motion v2.2 第 2 步 protocol：方向信息是否被预测器使用

日期：2026-10-31。状态：训练前登记。完整可执行参数、源码 SHA256、输入审计及 manifest 哈希见 `experiments/motion_v2/protocol_step2_v2_2.json`。两文件与执行源码必须提交并推送后才能运行。此前所有原始 protocol、停止记录、JSON、源码锁保持不动。

## 前置门槛与数据

v2.2 第 0′ 步重判、Phase 1 headroom 精确比较及真实 v2.1 第 1 步已通过。只使用已生成的 seed 20261030 数据：train 512 对、validation 96 对，共 1,216 episode、39,398 帧。不能生成或打开 test；不能调用旧 pipeline 的全 split 入口。数据 manifest 为 `results/motion_v2/v2_2/step1/data/dataset/manifest.json`；其内容和每个 NPZ 的 SHA256 必须复核。

模型输入仅包括图像特征、相对时间、history 标识、过去动作累计、已知未来动作及当前图像特征。t_c、可见性仅用于特权选帧；位置、速度、label 等隐藏状态只用于 train decoder 与最终评估，不进入预测器。

## 固定四组历史

记 P 为 `[t_c,end]` 内可见帧，S 为 t_c 前可见帧。每组仍有共同的最近四帧；memory 时间严格早于 recent 窗口。

| 条件 | memory 选择 | 目的 |
|---|---|---|
| a | 空 | recent-only |
| b | `[min(P), end]` | 最大跨度新运动证据 |
| c | `[min(S), max(S)]` | 有历史但与方向无关 |
| d | `[max(S), end]` | 终点位置已知、方向未知 |

规则对所有模型 seed 相同，不依赖预测误差或未来 target。a/c/d 在两个 sibling 间的全部模型张量必须逐元素完全一致；d 不能通过选帧时间、动作或填充暴露方向。所有帧索引、原始/投影 feature 输入边界须检查。

## 编码、归一化与 basis

- 原 Phase 1 `FrozenIJepa`：`facebook/ijepa_vith14_1k`，revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e`，官方 PIL processor，无 center crop，无随机权重 fallback。
- 冻结 eval 模式、FP32、batch size 2；grid2 四个不重叠池化格，合计 5120 维。逐帧移动干扰物意味着不依赖旧场景的重复帧缓存收益。
- 仅 train 所有帧拟合特征 mean/std，std floor 0.05。validation 不参与归一化。
- 复用冻结 Phase 1 `model.fit_basis`：rank 128；输入 PCA 仅 train 至 query 的特征；输出 basis 仅 train target residual 和 sibling target differences；PCA seed 901，原迭代数与 whitening 不变。

basis 冻结后，关闭 training dropout，按 b 的输入路径预热全部 1,216 episode 的特征投影、动作和索引缓存；不拟合参数，target 仍与模型输入分离。预热耗时计入 overhead，以免将一次性输入 IO 当成全部 epoch 的稳定训练速率。执行环境锁定 Python 3.12 及 `requirements-gpu.txt` 的关键包版本，完整实际版本另存。原 config 的 cache_dir=null 在运行副本中解析为新 output/hf_cache；HF、Torch、编译缓存也都写在该次新 output 内，不改变权重 revision 或科学参数。

## 预测器与训练预算

复用原 `DynamicsPredictor`（内容加权时序拟合、动作交互及 MLP），不是另换 Transformer 或保留策略。hidden=128、rank=128、模型 dropout=0。

- seeds：17、29、43；各 120 epochs；batch 64 episode＝32 对。
- 每 epoch 四条件各覆盖全部 1,024 train episode：recent-only、full-history、固定 b 的 privileged K=2、每 epoch 每对共同采样的 random K=2。
- 每 epoch 64 optimizer steps，每 seed 7,680 steps，总计 **23,040 steps**。
- AdamW：lr 0.001、weight decay 0.0001；gradient clip 1；StepLR 每 40 epoch 乘 0.5。
- 原 feature MSE + 2×paired residual MSE + 0.5×rank loss，不增加隐藏状态损失。
- 仅训练时对非空 memory 条件以 p=0.25 丢弃全部 memory，同一 pair 共用随机决定；保留 recent=4。这个概率是本次新增登记，不声称 Phase 1 原来使用了该随机 dropout。结合 recent-only 条件，期望无历史曝光为 43.75%。验证/评估 dropout=0。

每 epoch 用 validation 四种训练条件、两个 horizon 的等权平均 feature MSE 选择 checkpoint；严格更小才更新，并列取最早 epoch。不得用 a/b/c/d 的主判据选择超参、追加 epoch 或重跑。保存改进 checkpoint 时使用新文件名，不删除旧文件。

## 只用 train 的冻结位置 decoder

沿用原线性 ridge `LinearReadout.fit/predict`，输入归一化真实 target feature，输出 `[x,y]`；不使用 class 目标。train 两个 horizon 共 2,048 行。

ridge 候选固定 `[0.01,0.1,1,10]`。仅 train 内按 pair 做 GroupKFold 5 折，两个 sibling 和两个 horizon 必须同折；每折 feature mean/std 只在该折训练部分拟合。按全体 out-of-fold 行的整体 xy MSE 选择 ridge，并列取最小 ridge。选定后在全部 train 上拟合一次并冻结，记录 fit IDs、fold membership、每项得分和模型哈希。

validation 不参与 decoder 的 ridge、参数或预处理选择。不裁剪位置预测。decoder 在真实 validation target feature 上的误差只作诊断，不据此调参。

## 指标、配对区间与判据

分别对每个 seed、pair、sibling、条件、horizon 记录：

- feature MSE：归一化 5120 维 target 特征的平均平方误差。
- position MSE：`((x_hat-x)^2+(y_hat-y)^2)/2`，坐标范围按生成器归一化单位。

先在每个 pair 内平均两个 sibling 的 loss，再平均三个固定训练 seed，得到每条件/horizon 的 96 个 pair 值；不先平均预测形成 ensemble。以 seed 901、有放回抽 96 对、共 2,000 次 bootstrap，使用 percentile 2.5%/97.5%（linear quantile）。所有条件、horizon、指标和单 seed 附表共享重采样索引。

报告 a/b/c/d × h=3/6 的均值和 95% CI，以及 d−b、c−b、a−b 的同对 loss 差值 CI；正值表示 b 更好。另报告每个 seed，不挑最好 seed。CI 只描述这三个已拟合 seed 的平均性能的场景抽样不确定性，不把 96×2×3 当成 576 个独立样本。

**主判据：** h=3、h=6 的 position MSE(d)−position MSE(b) 的 95% 下界分别都严格 >0。

**次要必需判据：** 将“b 优于 c”预先明确为 h=3、h=6 的 position MSE(c)−position MSE(b) 的 95% 下界也都严格 >0。b 对 a 与 feature MSE 差值只报告，不替代主判据。

任意必需项不满足、缺 seed/pair、NaN 或输入审计失败即停；全部通过也停。不改阈值重跑，不训练保留策略，不进入第 3 步。

## GPU 用时预估与执行边界

使用一张 A100，登记估时上限 **7,200 A100-GPU 秒（2 小时）**，低于用户授权总上限 14,400 秒。外部提交作业的 wall-time 也不得超过 7,200 秒；本脚本不提交作业或登录。

预估来源是仓库历史 Phase 1 记录，不是本次实测：

- 编码保守速率 27.15545 frame/s，39,398 帧约 1,451 秒。
- 训练/验证保守耗时 5.72894 秒/epoch，360 epochs 约 2,062 秒。
- 初始化、basis、decoder 21 次拟合、读写和最终评估预留 3,600 秒。
- 合计约 7,113 秒，向上登记为 7,200 秒。生成器最大帧数边界 52,288 仅作输入检查，本估计使用已经生成并锁定的真实 39,398 帧。

历史来源：v9 `native_encode_train_completed/fresh/train_encoding_complete.json`；`native_interruption9068/fresh/sequence_budget.json` 与相应训练计时。不得套用混有去重收益的 70.44 logical frame/s。

原生 Slurm 记录必须证明当前作业单张 A100；以当前 GPU allocation 已运行时长作为保守累计起点，包括 runner 开始前的时间，已用的非编码/训练时间从 overhead reserve 扣除，避免重复计算。每次编码 batch 和 optimizer step、每次 decoder fit 前后检查预算；至少完成 64 编码帧后更新吞吐，至少一个完整 epoch 后更新训练率。实际或完整剩余序列的预测用时超出 7,200 秒即停，不降低精度、batch 或 epoch 以绕过预算；失败与占用 GPU 时的 CPU 工作也计入。

不自行登录 OOD/SSH，不访问 xl4358 账号目录。需要认证时停止，由用户本人登录。当前无 GPU 作业被本任务启动；训练结果不得用占位数字冒充。

## 软件预检与结果存放

源码冻结后只允许先运行已登记的纯数组统计单元测试，检查配对聚合、差值方向、两个 horizon 的独立门槛及缺失/非有限数据拒绝。这不是生成 held-out test episode，也不拟合预测器或 decoder。

第 2 步所有输出写新目录，拒绝覆盖或自动恢复旧 attempt。记录 native allocation、encoder revision、环境版本、各阶段耗时、训练日志、checkpoint、decoder、逐 pair loss、bootstrap indices、完整结果表、status.json 和 SHA256。若认证未完成，结果表明确标记“未运行”，不得作出统计通过/失败结论。
