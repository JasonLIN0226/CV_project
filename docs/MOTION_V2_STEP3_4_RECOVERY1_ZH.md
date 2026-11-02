# Motion v2 第 3/4 步运行层修订：17761 后重新提交

本登记依据用户在获知 GPU UUID 检查缺陷后明确提出的“重新配置一下再重新提交”。它只授权修复运行检查后重新提交第 3/4 步，不改变科学参数、seed、标签、学生架构、checkpoint 选择、指标、bootstrap 或门槛。原 17761 停止记录、原 protocol/JSON、72 项原源码及两个旧阶段现场全部保留。

## 原因与本次授权的边界

16614 使用旧 Step2 设备检查，未读取 `get_device_properties(0).uuid`。17761 的 sbatch 同样成功，进入单 A100 分配后，新检查器要求 PyTorch UUID 必须以 `GPU-` 开头而停止。官方 PyTorch v2.5.1 `Module.cpp` 的 `_CUuuid.__str__` 返回标准 8-4-4-4-12 格式，不带该前缀；此前模拟测试没有覆盖这种真实 API 表示。原始 UUID 未被记录，因此不把源码推断写成那次设备值的实测记录。

本次只统一合法 UUID 的完整表示，再比对 torch/nvidia-smi/可见设备；仍要求恰好一张 A100、完整相同 UUID、固定 GPU 身份。新增原始 type/repr/str、torch 版本、命令和输出记录，失败也保留。无法取得有效设备身份、截断、MIG、多卡、错误型号或不同 UUID 仍停止；沿用原逻辑，torch 无 UUID 时仅允许 nvidia-smi 对唯一可见设备给出完整身份。

原科学入口 marker 已存在，因此先前自动重提规则不允许继续；用户本次新指令构成只针对 job17761 的显式例外。预检必须重新核对其停止目录全部文件及哈希，确认尚在 initialization、后续科学阶段进度和时间均为 0，且新原生账单仍为 FAILED、单卡48秒、Restarts=0。原报告及旧冻结结算器的 blocked 状态不覆盖。

## 记账与预算

旧已结算 3705 秒，加17761原生48秒，按3753秒计入本次预算。剩余10647秒。估时仍8500秒，单次上限和 wall time 仍10500秒（02:55:00）；最坏3753+10500=14253≤14400，余147秒。分阶段估时450/900/1000/3400/2300/450秒、动态预算公式和10%速率预热规则不变。若发生允许的入口前基础设施失败，其实际秒数进入后续 admission 和 BudgetGuard；不能用估计值替代实际分配。

`sacct --duplicates` 保留全部原文；按 JobIDRaw、JobName、User、Account、Partition、Submit 精确选择本次顶层记录，再按 Cluster/DBIndex 绑定全部作业步。所有历史行及排除理由保留。相同提交身份有多个候选、同一数据库记录重复、身份/资源/时限冲突仍阻塞。全部步骤的时间不重复计入 allocation。

终态仍读取新鲜 scontrol。只有原生明确返回 `Invalid job id specified`、空stdout、returncode1时，允许用保存的原始 startup control 与新鲜终态 sacct 的 Restarts=0、起止时间、节点、资源、提交身份共同结算；不得把 RUNNING 改成终态或伪造 terminal scontrol。其他查询失败不使用该分支。新结算器、提交器与导出器共用这一纯解析实现，旧文件不修改。

## 先登记、验证，再部署

1. 所有修复使用新增 `*_recovery1.py` / wrapper；原72文件逐字节保留。`run_step34_recovery1.py` 只改运行层模块/证据/预算接线，模型载入之后的科学流程及编码函数须 AST 一致。新入口拒绝 CPU 模式，避免再次生成数据。
2. 本文、修订 protocol、源码哈希及测试清单先提交推送，再执行 CPU 软件检查。覆盖 torch2.5.1 UUID 对象、带/不带前缀、大小写、不同卡、非法输入、失败证据、原生历史同号记录、重启/预算/身份矛盾、原科学源码与参数不变、wrapper/提交/导出一致性；另以独立进程重放原98项 Slurm 测试。失败保留记录，修复解析/测试实现须先登记新的执行修订，不启动科学作业。
3. 检查通过后把回执与新源码哈希绑定到 execution protocol，再提交推送。新目录 `/scratch/xy3257/motion-v2-20261101-step34-deployment2` 独立浅克隆并核对完整 HEAD；不修改 deployment1。
4. 复用原 venv，严格 import/版本核对，不安装替代版本。复用原ZIP，通过新增路径绑定的纯传输 helper 验证384 NPZ/391文件；不得生成新数据。旧2432个NPZ、1239项编码/预测器资产继续逐字节校验；不重训预测器、不重新编码已有帧。
5. preflight 和 submit 每个 attempt 各一次；submit helper 只执行一次 sbatch，--no-requeue、单A100、10500秒；立刻保存 scontrol。未知提交结果只读调查，不能再提交一次。

## 停止、跟进与交付

所有科学哈希、数据、模型 revision、输入/配对审计、GPU身份、NaN/缺失、已有新输出目录、Restarts>0或预算超限仍阻塞。辅助CPU会话调度差异记录后继续。遇需要本人完成的认证交还用户；短暂重定向不直接判定认证失败。不得删除文件。

本次 recovery attempt1 是第3/4步总第2次A100提交。只有其原生终态且无.batch、无stdout/stderr、无marker、无科学输出四项全满足时，才允许按原授权最多再一次入口前重提（总最多3次），且重新证明累计+10500≤14400。任何本次科学入口启动之后的失败不再自动重提。

启动后先确认通过设备检查并进入正常科学阶段，再按用户要求每小时跟进；正常排队/运行无实质变化保持安静。保留运行中原生记录，终态按修订结算/导出流程收集证据。第4步无论结果都停，只报告train/validation；不生成或打开CONF/test。报告全部诊断、54学生选步、普通/具备资格的long-gap在线表、配对95%CI与累计A100秒。

依据：[PyTorch v2.5.1 UUID绑定](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/cuda/Module.cpp#L868-L913)；[Slurm sacct官方字段定义](https://slurm.schedmd.com/sacct.html)；原始17761证据位于 `results/motion_v2/step3_4/job17761_stop/`。
