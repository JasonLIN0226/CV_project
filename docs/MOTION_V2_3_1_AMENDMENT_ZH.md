# Motion v2.3.1 运行层修订登记

日期：2026-10-31。依据用户本轮授权，接续 `ff9cb462f7a78454e897ab14e7571af3ccae5a1a`。本修订与 JSON 先提交推送，再实现和回放；最终源码、回放结果、运行 protocol 再冻结提交推送后，才提交新 A100 作业。原 16542 的提交/取消/结算错误记录、原 protocol/JSON/哈希锁及 deployment1、deployment2 现场保留，不覆盖、不删除。

## 改动范围

科学方案完全沿用 v2.3：生成器及 seed 20261030、已有 train/validation 数据、I-JEPA revision/编码方式、归一化/basis、预测器架构及 seeds 17/29/43、训练条件与步数、train-only decoder、a/b/c/d、B、bootstrap seed/索引/次数及全部六项门槛均不变。仅运行层解析、启动等待和用户授权预算上限改变。`run_step2.py` 仅修改原生记录解析/启动等待及预算常量；以 AST 对照核查科学函数不变，BudgetGuard 判定算法不变。

新增共用 Slurm 解析模块：纯函数仅接收字符串，无 subprocess/文件/时间/环境依赖；提交检查器与 runner 共用。`NumNodes` 接受整数字符串 `1` 或 `min-max`，要求两端都为 1；其余取值、缺项和畸形文本拒绝。scontrol 与 sacct 解析保留原字段与原文，拒绝歧义/重复字段，不以实际 CPU 多于请求为失败。单 A100、Restarts=0、作业身份等其他要求不变；Requeue=1 只记录。

runner 仅在首次启动原生检查中，对 sacct 本作业记录暂缺或仍为非终态且未显示 RUNNING 做最多 120 秒的有界等待；同时每次先检查 scontrol 的作业身份、Restarts、GPU 和其他必须字段，其他不符立即失败。终态、畸形记录、字段不符不等待。命令超时及轮询等待均计入 120 秒；运行中复核不使用该启动宽限。GPU 分配中的等待、在线权重下载、初始化和 CPU 工作均计入作业预算。

## 真实记录回放与修复纪律

采用对应 JSON 列明 SHA256 的原文，不改写 fixture：16542 immediate scontrol（PENDING, NumNodes=1-1）；16512 scontrol（RUNNING, CPU）；12374 sacct（含 header，历史 A100 完成记录）。逐字段记录解析结果、提交/runner 适用要求、通过/失败/缺证据及预期。

预期：16542 满足提交检查；16512 仅证明解析正确，其 GPU 要求必须失败；12374 的 GPU TRES 解析通过，实际 cpu=12 不能造成 GPU 判断失败。历史 COMPLETED 记录不冒充当前 RUNNING 证据，16512 的 CPU=24 对请求4也不失败。另测试等值节点范围、非单节点拒绝、坏/缺失/重复字段、GPU计数、Restarts和120秒重试边界。纯解析回放不导入模型、不打开数据 NPZ、不使用 GPU。用户已授权解析结果不符时先修再测；逐次保留代码哈希、stdout/stderr、status 与 SHA，不覆盖失败记录。所有预期通过后才冻结最终 protocol 和提交作业。

## 预算与现场

原估时约 7113.250553 A100 秒，仅余约1%的旧上限不够覆盖运行波动。用户授权将本次登记上限与 sbatch wall-time 改为 10800 秒（3小时），总上限仍为14400 A100秒；原速率、训练预算和 BudgetGuard 算法不变，所有7200上限约束同步为10800。此前实际GPU累计为0秒。

为完整保留两个旧现场，新建 `/scratch/xy3257/motion-v2-20261031-deployment3` 浅克隆最终登记提交并核对HEAD。只读复用 deployment1 的既有venv及已锁定data.zip；先校验包哈希、再向新目录逐NPZ校验解包，不重装/换版本。复用v2.3已保存CPU B工件及原准备记录，按新运行登记显式锁定原协议与三份工件的哈希，不重算或改写B原件；必要的准备凭据兼容仅属来源验证，不改变数值/选择/评估规则。CPU辅助会话仅供部署与结算；实际CPU数量与请求差异如实记录。

新提交单个A100、--no-requeue、wall-time=03:00:00；唯一一次run_step2.py，新的输出目录与wrapper启动标记拒绝已存在路径。提交立即保存scontrol，Restarts>0停；完成/停止后保存sacct/scontrol和实际分配秒数。运行层结算识别未分配即取消时的0秒证据，原16542结算错误记录不改。需要用户认证即停。其他科学停止条件不变，所有六项通过或任一失败均停，不训练保留策略、不进入第3步、不打开或生成test、不删除任何文件。

## 交付

v2.3.1修订与冻结源码哈希、真实回放逐字段表和每次软件测试记录、部署import/数据核对、job ID与原生scontrol/sacct、a/b/c/d×h=3/6的feature/position MSE与共享配对95% CI、B、d−b/c−b/B−b/d−B及每seed附表、最终status与实际A100秒数。任何未完成指标明确为未运行，保留原始错误和现场。
