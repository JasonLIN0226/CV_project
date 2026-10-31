# Motion v2.3.1 执行记录

截至 2026-10-31 22:03 UTC，运行层修订、回放测试和部署检查已完成，唯一 A100 作业 **16550** 正在排队。科学指标尚未产生，所有科学门槛均为“未判定”。本记录不是最终运行报告。

- 登记提交：`6dc5ed253dfa305b26be8f19d40edcd6a5878ea7`，已在提交作业之前推送。部署 checkout 固定在此提交；后续结果文档提交不改变作业源码。
- 修订与科学协议见 [修订登记](MOTION_V2_3_1_AMENDMENT_ZH.md)、[第 2 步协议](MOTION_V2_3_1_STEP2_PROTOCOL_ZH.md)。科学参数、seed、阈值及六项必需判据不变；登记运行上限为 10800 A100 秒，用户总上限为 14400 秒。
- 真实原生记录回放 **98/98 通过**；逐字段结果见 [回放报告](MOTION_V2_3_1_REPLAY_ZH.md)。16542 的 `NumNodes=1-1` 通过提交检查；16512 的 CPU 记录可解析、GPU 检查按预期失败；12374 的 A100 TRES 通过。分配的 CPU 多于请求不构成失败。
- 新建 `/scratch/xy3257/motion-v2-20261031-deployment3`，核对完整 HEAD；复用原 venv 的 import/版本检查通过；原数据包 SHA256 与全部 **2432 个 NPZ** 校验通过。
- 单次 sbatch 返回 **16550**，即时 `scontrol`：`PENDING`、`NumNodes=1-1`、`Restarts=0`、`Requeue=1`、`TimeLimit=03:00:00`。提交检查通过；Requeue 如实记录。
- 最近队列状态 `PENDING (Priority)`，预计启动时间为 2026-10-31 23:45:33 UTC（美东 19:45:33），仅为调度估计。截至观察时实际 A100 用时为 **0 秒**；最终用时必须以终态 `sacct` 为准。
- 原作业 16542、两个旧 deployment、全部原停止记录和协议保留。未打开或生成 test，未启动保留策略训练。

远端原生提交证据在 `deployment3/submission/`，部署证据在 `deployment3/preflight/`。尚未下载最终证据；终态后将保存 scontrol/sacct、原始输出哈希及结果表，再判断 d−b、c−b、B−b 在 h=3/6 的配对 95% 下界，并报告 d−B 和真实特征 decoder 误差。

当前最大的未决项是冻结预测器能否超过方向下界 B；回放测试通过只说明运行层按登记解析原生记录，不能支持科学通过结论。
