# Motion v2.3.1 第2步运行 protocol

本文件接续 [v2.3 科学 protocol](MOTION_V2_3_STEP2_PROTOCOL_ZH.md) 与 [v2.3.1 运行层修订](MOTION_V2_3_1_AMENDMENT_ZH.md)，科学定义仍全部以v2.3为准。该版本新的JSON重新冻结当前完整源码；旧protocol/JSON/哈希不改。待真实回放全部符合预期并提交推送后执行。

唯一配置差异：`execution.estimated_gpu_seconds` 从7200改为10800。原工作估算7113.250553398305秒、总授权上限14400秒、全部学习参数和判断阈值不变。BudgetGuard仍用已消耗时间加全部剩余工作估计与实际耗时分别检查上限；初始化、sacct启动等待、权重在线下载、GPU分配中的CPU工作、失败占用全部计费。Slurm wall-time=03:00:00，1张A100，--no-requeue，Requeue仅记录、Restarts>0停。

提交与runner共同使用`slurm_records.py`的无副作用字符串解析及字段验证。节点数允许`1`和`1-1`，不允许任何非单节点区间。保留原native要求，实际CPU多于请求不等同于GPU数量不符。启动sacct最多120秒宽限仅针对当前作业行尚缺或PENDING/CONFIGURING同步未就绪：每轮先从scontrol确认RUNNING、身份、单节点/单A100、Restarts=0及有限时限；sacct已存在非空字段矛盾立即失败，未知/终态/坏记录不等待。sacct RUNNING后所有必须字段严格核验。宽限仅首次，运行中复核立即失败。原文与尝试时间写入新日志。

原始16542、16512、12374以及16542取消后的记录只作解析软件回放；逐字段表明确缺证据/N/A，不能拼成当前RUNNING科学作业。回放运行前先提交软件protocol与源码哈希；用户授权遇解析测试失败先修再测，各attempt保留不覆盖。最终全部通过后登记并推送科学及部署protocol，才提交作业。

新deployment3只读复用deployment1既有venv/data.zip，两个旧现场不改。环境import与版本必须通过，不临时重装。包SHA256、2432 NPZ逐文件校验沿用原清单；数据不重新生成。使用原已提交`results/motion_v2/v2_3/direction_bound/`三文件：新protocol显式锁定旧准备protocol、旧helper哈希和工件哈希；旧status保持原样。只兼容准备凭据的历史来源，不重新执行CPU准备，不修改B数值。原runner的实际target/IDs/数组一致性验证不变。

官方I-JEPA revision、FP32/grid2/5120维、train-only归一化与basis、predictor架构、seed17/29/43、120epochs/seed、memory dropout、train-only decoder、checkpoint选择、a/b/c/d历史、B与所有6项CI门槛均不变。h=3/6分别要求d−b、c−b、B−b位置MSE差值的95%配对CI下界>0。d−B与真实feature decoderMSE仅诊断，featureMSE及a−b照常报告。同一bootstrap索引，先平均sibling，再平均固定3seed；2000次、seed901。

提交单个A100作业，仅一次run_step2.py，新输出`results/motion_v2/v2_3_1/step2_gpu_attempt1`，拒绝已有输出或wrapper启动标记。源码/protocol/数据/模型revision、输入/配对审计、GPU类型/数量、预算实际/预估、NaN/缺项、认证要求等原阻塞条件保持。提交后立即scontrol，运行中记录原生复核，终态sacct记录实际A100秒数；无分配取消明确0秒，不伪称获得GPU。六项判据无论通过或失败都停，不访问test，不训练保留策略，不进入第3步，不删除文件。

交付包含逐字段回放、所有软件attempt与哈希、部署回执、原生调度/用量、完整指标/对比与每seed附表；提前停止则未完成项明确未运行。结果进入`results/motion_v2/v2_3_1/`，新报告单独登记，原错误记录保留。
