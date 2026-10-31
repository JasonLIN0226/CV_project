# Motion v2.3 执行报告（2026-10-31）

**本轮停止在提交检查器的实现缺陷：A100 作业 16542 只提交一次，在等待资源时被我方检查器误判，随后取消；`run_step2.py` 调用 0 次，实际 A100 用时 0 秒。预测器的六项科学门槛均未评估。** 这不是科学数据不一致，也不是预测器统计判据失败。

工作仍在分支 `experiments/motion-v2-20261030`、[PR #3](https://github.com/JasonLIN0226/CV_project/pull/3)。本报告及 [v2.3 status.json](../results/motion_v2/v2_3/status.json) 是本轮结果入口；原 v2/v2.1 报告、v2.2 进展及停止记录、旧 protocol/JSON 和哈希锁保持原样。此前报告中的“当前状态”仅指其各自版本。没有合并 main、force push、删除文件、训练保留策略或进入第 3 步。

本轮依据先行提交的 [v2.3 修订](MOTION_V2_3_AMENDMENT_ZH.md)、[科学 protocol](MOTION_V2_3_STEP2_PROTOCOL_ZH.md)及[部署 protocol](../experiments/motion_v2/protocol_step2_deployment_v2_3.json)。实际部署并验收的完整 HEAD 为 `dd9582c317cd6388180868ef175960cc2875321d`；科学 protocol SHA256 为 `031e37b3f83d3ec5c05242a72069fc8336b32e771c3e2a8a2a3f6f0302e10875`，部署 protocol SHA256 为 `b952731a4e9ff657662cec44a799ac923249d9e5fe4f88aea0bd2ab33496965b`。两者在本轮实际操作前冻结；没有因失败修改检查器、判据或版本后重提。

**已完成的历史第 0、1 步。** 原第 0 步不能概括为“两个最初脚本均逐数值复现”。实际证据是：Phase 1 `motion_headroom.py` 后来只运行一次，完整解析 JSON 与附件完全一致；seed 9137 的 pair ID/query/gap 尚未与完整原始 manifest 逐对核对，结论仍限定为公开代码重建。v2.1 参考检查此前只运行一次，单帧分类两项为 0.620/0.545、附件为 0.624/0.538，原严格复现停止记录保留；v2.2 按用户事先修订的容差与阈值同侧规则，对同一份保存结果重判通过，没有重跑。不能把该重判称为新的严格逐数值复现。证据见 [Phase 1 对照](../results/motion_v2/v2_2/step0/phase1_headroom/comparison.json)、[v2.2 重判](../results/motion_v2/v2_2/step0/rejudge/decision.json)和[v2.2 报告](MOTION_V2_2_REPORT_ZH.md)。

第 1 步已经按冻结 protocol 一次性生成 seed 20261030 的 train 512 对、validation 96 对，共 1,216 episode、39,398 帧。真实配对结构审计 608/608 通过，pytest 7 项通过。真实 train 的隐藏状态分类准确率为：end-only 0.5000000000、单张非终点新段帧 0.5253188654、旧段加 end 0.5000000000、两张新段帧 0.9756520084，均通过当时登记门槛。train/validation 的在线充分帧保留率均为 FIFO 0%、keep-first 0%、事后充分帧上限 100%、最新两张可见帧结构参照 100%。这些是生成器审计和充分帧保留指标，不是 I-JEPA 预测精度；也不同于参考脚本的四组 500/500 配对审计。证据见[真实数据审计](../results/motion_v2/v2_2/step1/audit.json)、[第 1 步状态](../results/motion_v2/v2_2/step1/status.json)及[哈希清单](../results/motion_v2/v2_2/step1/SHA256SUMS)。

**v2.3 软件、环境和数据准备。** 纯人工数组的软件测试一次通过 28/28，pytest 报告 0.44 秒；没有编码、拟合模型、读取真实 validation/test 或使用 GPU。测试检查的是 B 的代数恒等式、差值方向、独立 horizon 门槛、共享 bootstrap 及非法输入拒绝，不是预测器通过证据。证据见[软件状态](../results/motion_v2/v2_3/software/status.json)、[原始 pytest 输出](../results/motion_v2/v2_3/software/stdout.log)和[软件 SHA256](../results/motion_v2/v2_3/software/SHA256SUMS)。

新部署根目录为 `/scratch/xy3257/motion-v2-20261031-deployment2`。旧 deployment1 的 venv 和 data.zip 只读复用，未重装或换版本。CPU 环境/source import/manifest 预检通过；解包回执确认 2,432 个 train/validation NPZ 文件通过 manifest 哈希核对，没有重新生成数据或打开 test。数据包 SHA256 为 `58f14130ddda659f9c3a11bbbc50d2729cf61b0613da7bbf7522466afa28831d`；manifest SHA256 为 `9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a`。CPU 辅助会话 16537 的实际 CPU 分配与请求不同，按修订记录为非阻塞，没有将其当作单 A100 科学分配。证据见[预检回执](../results/motion_v2/v2_3/deployment2/remote_evidence/preflight/receipt.json)、[预检状态](../results/motion_v2/v2_3/deployment2/remote_evidence/preflight/status.json)、[解包回执](../results/motion_v2/v2_3/deployment2/remote_evidence/unpack_receipt.json)和[辅助会话原始记录](../results/motion_v2/v2_3/deployment2/remote_evidence/auxiliary_job_initial.txt)。

环境为 Linux x86_64、Python 3.12.11、NumPy 1.26.4、scikit-learn 1.7.2、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、transformers 5.5.4；11 项锁定版本与 72 项活动依赖检查通过。科学源码 59/59、部署源码 7/7 哈希保持一致，见[交付完整性复核](../results/motion_v2/v2_3/deployment2/integrity_review.json)。导出后以 `scancel 16537` 释放本次辅助作业，OOD 显示 Completed；未删除会话或文件，见[释放观察记录](../results/motion_v2/v2_3/deployment2/auxiliary_release_observation.json)。

**CPU 方向下界已计算；模型指标未计算。** 在科学 protocol 冻结后、A100 作业提交前，CPU helper 读取了 96 对 validation 的 evaluation_only target 位置，保存真实坐标、逐 pair/horizon 的 B、IDs 和哈希；没有导入或拟合模型。完成时间为 `2026-10-31T21:37:24.213943+00:00`。每对定义 `m=(p0+p1)/2`、`B=mean_sibling(mean_xy((m-p)^2))`；B 是两个 sibling 获得相同位置预测时的最优误差下界，不是训练模型的测量结果。

| horizon | CPU B 均值 | B 的 95% CI | 真实 target feature 的 decoder position MSE | decoder 95% CI |
|---|---:|---|---|---|
| 3 | 0.010503079706867778 | null（未计算） | null（未运行） | null（未运行） |
| 6 | 0.014639254826637952 | null（未计算） | null（未运行） | null（未运行） |

以上 B 均值直接摘自[CPU 下界状态](../results/motion_v2/v2_3/deployment2/remote_evidence/repo/results/motion_v2/v2_3/direction_bound/status.json)，没有在写报告时重算。CPU helper 不生成 bootstrap CI；预定共享 CI 的最终 metrics 阶段没有启动，因此不补算区间。远端保存的 `direction_bound.npz` 文件 SHA256 为 `0a99ba0570ef116a03c24775d311ba25886434e7f1e2c87637bc6e1a800427ea`；本地导回了[文件哈希回执](../results/motion_v2/v2_3/deployment2/remote_evidence/repo/results/motion_v2/v2_3/direction_bound/artifact_sha256.json)及状态，[逐 pair 二进制工件](../results/motion_v2/v2_3/direction_bound/direction_bound.npz) 也已按原字节导回，远端原件继续保留在 deployment2。B 数组内容哈希与 NPZ 文件哈希用途不同，未混用。

**提交失败的具体原因。** `sbatch` 只调用一次，返回 job ID **16542**。随即取得的 scontrol 为 `JobState=PENDING`、`Reason=Resources`、`NumNodes=1-1`、`Restarts=0`，请求 `TresPerNode=gres/gpu:a100:1`。`NumNodes=1-1` 表示节点请求下限和上限都为 1；我方 `submit_step2_v2_3.py` 的 `control_fields()` 却仅接受字符串 `"1"`，抛出 `Native job must request one node`。因此原提交状态是 `stopped_after_submission`。

这是我方检查器没有处理等待状态节点范围表示的实现缺陷，不能据此断言请求了多节点、GPU 型号错误或科学数据不一致。28 项数组测试覆盖统计实现，没有覆盖这个 Slurm 文本解析分支。检查失败后，按冻结部署 protocol 的 failure-stop 规则停止推进并取消该等待作业，没有修改源码后重提。原始证据为[提交回执及 traceback](../results/motion_v2/v2_3/deployment2/remote_evidence/submission/submission.json)、[提交状态](../results/motion_v2/v2_3/deployment2/remote_evidence/submission/status.json)和[立即 scontrol](../results/motion_v2/v2_3/deployment2/remote_evidence/submission/immediate_scontrol.stdout.txt)。`Requeue=1` 被原样保留，依 v2.3 修订它不是本次失败依据；native `Restarts` 始终为 0。

最终原生 sacct 为 `CANCELLED by 4882486`、`Start=None`、`ElapsedRaw=0`、`AllocTRES` 为空。最终 scontrol 同样记录 `CANCELLED`、`RunTime=00:00:00`、`AllocTRES=(null)`、`Restarts=0`；它将 StartTime 和 EndTime 均填为取消时刻 `2026-10-31T21:38:11`，这个字段不能单独解释为已启动执行，应与 sacct 及文件存在性证据共同读取。

只读 `--settle` 也未通过：其实现对所有终态都要求恰好一张已分配 GPU，无法处理“等待中取消、没有任何 AllocTRES”的合法终态，因此保存了 `stopped_accounting_error` 和 `Native TRES does not certify exactly one GPU`。这是第二处检查器边界处理缺陷；原错误回执保留，没有把 settlement status 改写为 passed。实际 0 A100 秒的结论来自上述原生终态事实，而不是虚称结算 helper 通过。证据见[原始 sacct](../results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_sacct.stdout.txt)、[最终 scontrol](../results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_scontrol.stdout.txt)、[结算回执](../results/motion_v2/v2_3/deployment2/remote_evidence/settlement/accounting.json)和[结算错误状态](../results/motion_v2/v2_3/deployment2/remote_evidence/settlement/status.json)。

在终态后导出的[证据清单](../results/motion_v2/v2_3/deployment2/remote_evidence/export_manifest.json)确认：`scientific_job_started`、`repo/results/motion_v2/v2_3/step2_gpu_attempt1`、`scientific-job-16542.stdout.log`、`scientific-job-16542.stderr.log` 均不存在。结合 sacct 的未启动、零分配记录，可以确认科学 wrapper 未进入，`run_step2.py` 调用 0 次。I-JEPA 编码、归一化/basis 拟合、decoder 拟合、三 seed 预测器训练和最终 metrics 均未运行，不存在新 checkpoint 或科学指标文件。

另存的[计费解释](../results/motion_v2/v2_3/deployment2/accounting_interpretation.json)明确区分原结算脚本失败与原生证据确认的零消耗；[机器可读未运行表](../results/motion_v2/v2_3/deployment2/step2_results_not_run.json)包含下列主表、对比和每 seed 单元。

**第 2 步主表。** a 为 recent-only；b 为第一张新段可见帧加 end；c 为第一张与最后一张旧段可见帧；d 为最后一张旧段可见帧加 end。所有条件共享最近四帧。feature MSE 的登记定义是归一化 5120 维均方误差，position MSE 是 x/y 两维均方误差。下表的 `null` 一律表示未运行，不表示误差为零。

| 条件 | horizon | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---|---:|---|---|---|---|
| a | 3 | null | null | null | null |
| a | 6 | null | null | null | null |
| b | 3 | null | null | null | null |
| b | 6 | null | null | null | null |
| c | 3 | null | null | null | null |
| c | 6 | null | null | null | null |
| d | 3 | null | null | null | null |
| d | 6 | null | null | null | null |

必需判据是 d−b、c−b、B−b 的 position MSE 改善在两个 horizon 上各自配对 95% CI 下界严格大于 0，共六项全过。预定先平均 sibling，再平均固定三个 seed，对 96 对做 2,000 次 bootstrap、seed 901，所有指标共享索引；区间只覆盖固定三个 seed 平均表现的场景抽样不确定性。该统计阶段没有运行，不能用已知 B 均值替代 B−b 的 CI。

| 位置对比 | horizon | 均值 | 95% CI | 下界是否严格 >0 | 用途 |
|---|---:|---|---|---|---|
| d−b | 3 | null | null | 未评估 | 必需 |
| d−b | 6 | null | null | 未评估 | 必需 |
| c−b | 3 | null | null | 未评估 | 必需 |
| c−b | 6 | null | null | 未评估 | 必需 |
| B−b | 3 | null | null | 未评估 | 必需 |
| B−b | 6 | null | null | 未评估 | 必需 |
| a−b | 3 | null | null | 不适用 | 仅报告 |
| a−b | 6 | null | null | 不适用 | 仅报告 |
| d−B | 3 | null | null | 不适用 | 仅诊断 |
| d−B | 6 | null | null | 不适用 | 仅诊断 |

d−b、c−b、a−b 的 feature MSE 差值及 CI 也均为 `null`；B 没有 feature MSE 对应定义。总体六项判定为**未评估**，不能称为通过或“预测器还不能用对的帧”。真实 target feature 的 decoder 诊断也未运行。

**每 seed 附表。** 三个 seed 均未启动，下表保留全部 24 个 condition/horizon 单元；没有从人工数组 fixture 填入任何数值。

| seed | 条件 | horizon | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---:|---|---:|---|---|---|---|
| 17 | a | 3 | null | null | null | null |
| 17 | a | 6 | null | null | null | null |
| 17 | b | 3 | null | null | null | null |
| 17 | b | 6 | null | null | null | null |
| 17 | c | 3 | null | null | null | null |
| 17 | c | 6 | null | null | null | null |
| 17 | d | 3 | null | null | null | null |
| 17 | d | 6 | null | null | null | null |
| 29 | a | 3 | null | null | null | null |
| 29 | a | 6 | null | null | null | null |
| 29 | b | 3 | null | null | null | null |
| 29 | b | 6 | null | null | null | null |
| 29 | c | 3 | null | null | null | null |
| 29 | c | 6 | null | null | null | null |
| 29 | d | 3 | null | null | null | null |
| 29 | d | 6 | null | null | null | null |
| 43 | a | 3 | null | null | null | null |
| 43 | a | 6 | null | null | null | null |
| 43 | b | 3 | null | null | null | null |
| 43 | b | 6 | null | null | null | null |
| 43 | c | 3 | null | null | null | null |
| 43 | c | 6 | null | null | null | null |
| 43 | d | 3 | null | null | null | null |
| 43 | d | 6 | null | null | null | null |

每 seed 的 d−b、c−b、a−b、B−b、d−B 均值和 CI 均为 `null`。B 本身不依赖训练 seed，上方两项 CPU B 均值对三个 seed 相同，不把 B 复制成额外样本；B 的共享 bootstrap CI 和 decoder 诊断仍未计算。

**GPU 预算与停止边界。** 科学方案没有改变：39,398 帧、冻结官方 I-JEPA FP32/四格池化 5120 维、train-only 归一化和 rank-128 basis、原 Phase 1 预测器、seeds 17/29/43、各 120 epochs，共 23,040 optimizer steps，以及 train-only decoder。登记总估时为 **7,113.250553 秒**，执行上限 **7,200 A100 秒**，用户总授权上限 **14,400 秒**；这些是预估和限制，不是实际消耗。由于作业在获得分配前取消，**本轮及此前累计实际 A100 用时为 0 秒（0 GPU 小时）**。已经完成的 B、环境和数据准备都在 CPU 辅助会话中进行。

此次最直接的执行风险已经显现：我方部署检查器对 Slurm 合法状态表示及零分配取消终态处理不完整，使一次已授权实验在训练前停止。未启动科学入口并不消除该实现责任。最大的研究风险仍未被检验：隐藏状态下两帧可辨方向，不保证冻结 I-JEPA 特征和规定训练预算的预测器能利用它；CPU B 仅给出对照下界，不能证明 b 超过该下界。

按照冻结失败规则，本轮到此停止，未重新提交、未修阈值重跑、未更换依赖、未覆盖原错误记录。未打开或生成 test episode，未训练保留策略，未启动第 3 步。所有导回原始文本/JSON 的字节数与 SHA256 收录于[远端证据导出清单](../results/motion_v2/v2_3/deployment2/remote_evidence/export_manifest.json)，提交及结算各自的 SHA256 清单和失败状态也原样保留。
