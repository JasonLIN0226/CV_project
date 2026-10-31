# Motion v2.3.1 基础设施失败核查与有限重提登记

登记日期：2026-10-31。依据用户在 16550 终态交付后的新授权，本登记只增加基础设施核查、独立运行回执和有条件的重提；科学版本固定为 `6dc5ed253dfa305b26be8f19d40edcd6a5878ea7`。旧的单次提交限制仅在本登记列明的范围内由新授权替代；16550 的全部原生记录、状态、报告、哈希和三个 deployment 现场保留不改。

## 原因核查

“云节点开机失败”是用户提出的可能解释，登记时尚未得到原生证据确认。只读收集 16550 的 `sacct --duplicates` 全部可查询字段与所有作业步（不加 `-X`）、`scontrol show node b-19-12`、`scontrol show config` 中 ResumeTimeout / ResumeProgram 相关项，以及 `sinfo -p c12m85-a100-1 -N -o "%N %T %E"`。完整 stdout/stderr、返回码、命令、采集时间和 SHA256 保存至新的核查目录。节点或配置查询不可用时如实记录，不将其当作科学失败或另行修补环境。

以下四项必须逐项给出原生证据或当前文件系统存在性证据；查询失败不能当作不存在：

1. `sacct --duplicates` 不含该作业的 `.batch` 步。
2. 该作业 stdout 和 stderr 两个文件均不存在。
3. `/scratch/xy3257/motion-v2-20261031-deployment3/scientific_job_started` 目录不存在。
4. `/scratch/xy3257/motion-v2-20261031-deployment3/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1` 目录不存在。

四项同时成立才标记“科学入口前的基础设施失败”。原因细节的核查不阻塞本次用户明确授权的重提，但任意启动标记存在、科学产物存在、Restarts > 0、预算超限或科学阶段错误均禁止重提；后续自动重提还必须取得全部四项的肯定证据。

## 不变项与独立回执

远端科学 checkout 保持 `6dc5ed2` 完整提交，科学源码、科学 protocol、config、数据、seed、判据、BudgetGuard 和 wrapper 的全部字节不变。复用原 venv、已校验数据与 CPU B 产物；重新做 import/关键版本核查，不安装或更换依赖。原科学 wrapper 仍使用其不存在的原启动标记和输出目录；历史 16550 的作业日志路径带 job ID，不会与新作业冲突。

新增运行辅助源码单独登记哈希，发布到 `deployment3/runtime_retries_v2_3_1/source/`；不检出或修改旧科学仓库。attempt2 和必要时 attempt3 的提交、预检、结算目录分别位于 `runtime_retries_v2_3_1/attempt2/` 和 `attempt3/`，均拒绝已有输出目录。未知提交结果先只读核查，不能再次 sbatch。任何原回执不覆盖。

若实施中必须改变 wrapper，则先重新冻结哈希并重跑完整 98 项回放，全部通过后才可提交；当前方案保持 wrapper 原样，不触发该条件。新增运行层代码的测试也须登记源码哈希、提交推送后才运行，之后登记最终提交协议并推送，再提交 A100 作业。

## 次数与累计预算

本授权最多包含三次 A100 提交：已终止的 16550 为 attempt1，本次为 attempt2，只有 attempt2 再次满足四项入口前失败条件时才允许一次自动 attempt3。16542 等更早历史尝试不被重开，其实际已分配 GPU 秒数仍按原证据计入累计（此前为 0）。

每次提交前都从已终态的原生 `sacct` 重新核对累计已分配 A100 秒，并要求 `累计 + 10800 ≤ 14400`。当前累计 769 秒，attempt2 预留后为 **11569 秒**，符合上限。attempt3 前累计必须不大于 3600 秒。每次仍是单节点、单张 A100、10800 秒 wall-time、`--no-requeue`，Requeue=1 只记录，Restarts > 0 立即停止。BudgetGuard 逻辑及数值保持冻结版本，不用新代码扩展科学预算。

只要 `scientific_job_started` 目录存在，无论运行结果如何都不重提；任何科学阶段错误都停止。不得删除文件、清理旧现场、恢复或续训、改变阈值、打开或生成 test、训练保留策略。第 2 步无论通过与否均停止。

## 跟进与交付

提交后每 5–10 分钟只读查询 squeue/sacct，持续到原生终态；跟进任务在活跃作业期间保留。每次记录 job ID、节点、启动/结束时间、scontrol/sacct 和累计 A100 秒。新结果放到 `results/motion_v2/v2_3_1_resubmission/`；原 v2.3.1 结果目录及停止记录不改写。

若科学入口开始执行，按冻结协议报告 a/b/c/d × h=3/6 的 feature/position MSE、B、配对 95% CI、d−b/c−b/B−b/d−B 与逐 seed 附表；未产生的指标保持未评估。所有必需门槛不变。运行层源码哈希和测试命令将在独立 JSON protocol 中登记后再执行。
