# Motion v2.3.1 Slurm 离线回放报告（2026-10-31）

**首次登记回放通过：98/98 项测试，失败、错误、跳过均为0；实际GPU用时0秒。** pytest原始输出为 `98 passed in 0.13s`；driver记录CPU墙钟用时0.23558591702021658秒。没有再次运行测试。

本报告只根据已保存的[状态JSON](../results/motion_v2/v2_3_1/software/attempt1/status.json)、[逐字段回放JSON](../results/motion_v2/v2_3_1/software/attempt1/fixture_replay.json)、[stdout](../results/motion_v2/v2_3_1/software/attempt1/stdout.log)和[JUnit XML](../results/motion_v2/v2_3_1/software/attempt1/pytest.xml)整理。完整工件哈希见[SHA256SUMS](../results/motion_v2/v2_3_1/software/attempt1/SHA256SUMS)。报告生成不重放校验器、不解码科学数组、不计算实验指标。

执行源码及软件protocol提交为 `7292744ccbb04ef4b5bd251a0518132a678e1fef`（`7292744`）；[软件protocol](../experiments/motion_v2/protocol_slurm_replay_v2_3_1.json) SHA256为 `71d039297c70c4fe0a39c1ec59abda6941f13a4c683ad2539fcff6309451b060`。它在回放前提交，依据已推送的[v2.3.1运行层修订](MOTION_V2_3_1_AMENDMENT_ZH.md)。测试使用Python 3.12.14、pytest 7.4.4，运行在macOS arm64；源码逐项哈希保存在状态JSON。

离线测试调用与真实提交器、运行器相同的 `validate_submission_control`、`validate_running_control`、`validate_settlement_record` 和共用 `slurm_records` 解析函数；没有仅复制一套测试规则来代替运行时校验。`run_step2.py` 仅作为stdlib模块提供原生记录校验函数，没有调用科学入口。集群命令、真实轮询sleep、模型导入或拟合、真实数据数组读取、test episode访问和GPU使用均为0。

**fixture及身份上下文。** 下列五份原始文本按登记SHA256读取，没有改写为新job或deployment3记录。16542提交及结算回放使用它自己的历史上下文：owner `xy3257`、account `csci_ga_2271-2026fa`、partition `c12m85-a100-1`、时限7200秒、旧deployment2的wrapper和workdir。这个历史时限不改变v2.3.1新作业登记的10800秒上限。

| fixture | 原始证据 | SHA256 |
|---|---|---|
| 16542 PENDING | [原文](../results/motion_v2/v2_3/deployment2/remote_evidence/submission/immediate_scontrol.stdout.txt) | `850e8e394ebe06d301666c0575e3e9c5fec01289c5a17db6d5e513ca657c532d` |
| 16512 CPU RUNNING | [原文](../results/motion_v2/v2_2/deployment1/remote_evidence/native_job_at_stop.txt) | `3e79c9b4fd9e7e2ad172293b04f84ef73677499d05714df968af655f46ad8de7` |
| 12374 历史 COMPLETED | [原文](../results/literature_diagnosis/qualified-reader-retention-20261025/conf/cluster/sacct_12374.txt) | `509f39267779ea64e0b5f5c6b20925b753ba0a5c9eed2f4c8a42d49561e0d4f3` |
| 16542 CANCELLED sacct | [原文](../results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_sacct.stdout.txt) | `c6e562f1d80ec1ea4d4e157775087ae0c305f7004c64fd34b72f40e0731ef676` |
| 16542 CANCELLED scontrol | [原文](../results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_scontrol.stdout.txt) | `87a38c8ba2d7ec1e7b7f357f8e761b8c7c49010cc583bcc8b3ee8a7344dca604` |

16542等待状态为单行scontrol；16512为多行scontrol；12374为带十列表头的pipe分隔sacct，同时含顶层、`.batch`及`.extern`行，回放只选择精确顶层12374。16542取消sacct按已登记的十二列无表头结构解析。每份原文全部解析字段保存在回放JSON的 `cases[].parsed_fields`，下表逐行列出其中全部 `checks[]`，没有省略N/A或预期拒绝项。

**结果含义。** “通过”表示该项与登记预期一致。“预期拒绝”表示校验器实际拒绝了不具备当前科学运行资格的记录，这属于软件测试通过，不是作业或科学实验失败。“N/A：缺证据”表示原始表没有该字段，不能补造；“N/A：上下文不适用”表示历史/辅助记录不承担当前科学作业资格。所有表内结果均与JSON中的expected一致。

| fixture | field | requirement | observed | result |
|---|---|---|---|---|
| 16542 PENDING | `JobId` | 历史提交身份；调用实际共用提交校验器（submission） | `16542` | 通过（符合预期） |
| 16542 PENDING | `Restarts` | 历史提交身份；调用实际共用提交校验器（submission） | `0` | 通过（符合预期） |
| 16542 PENDING | `Account` | 历史提交身份；调用实际共用提交校验器（submission） | `csci_ga_2271-2026fa` | 通过（符合预期） |
| 16542 PENDING | `Partition` | 历史提交身份；调用实际共用提交校验器（submission） | `c12m85-a100-1` | 通过（符合预期） |
| 16542 PENDING | `Command` | 历史提交身份；调用实际共用提交校验器（submission） | `/scratch/xy3257/motion-v2-20261031-deployment2/repo/experiments/motion_v2/run_step2_v2_3.sbatch` | 通过（符合预期） |
| 16542 PENDING | `WorkDir` | 历史提交身份；调用实际共用提交校验器（submission） | `/scratch/xy3257/motion-v2-20261031-deployment2/repo` | 通过（符合预期） |
| 16542 PENDING | `UserId` | 历史提交身份；调用实际共用提交校验器（submission） | `xy3257(4882486)` | 通过（符合预期） |
| 16542 PENDING | `CPUs/Task` | 保留斜线、冒号字段名，不能并入前一字段的值（parser） | `4` | 通过（符合预期） |
| 16542 PENDING | `ReqB:S:C:T` | 保留斜线、冒号字段名，不能并入前一字段的值（parser） | `0:0:*:*` | 通过（符合预期） |
| 16542 PENDING | `NtasksPerN:B:S:C` | 保留斜线、冒号字段名，不能并入前一字段的值（parser） | `1:0:*:*` | 通过（符合预期） |
| 16542 PENDING | `NumNodes` | 单节点：整数1或范围两端均为1（submission/runner） | `1-1` | 通过（符合预期） |
| 16542 PENDING | `TimeLimit` | 该历史请求的时限为7200秒（submission） | `02:00:00` | 通过（符合预期） |
| 16542 PENDING | `ReqTRES` | 请求恰好一张GPU（submission） | `cpu=4,mem=16G,node=1,billing=4,gres/gpu=1` | 通过（符合预期） |
| 16542 PENDING | `TresPerNode` | 保留原始A100类型请求（submission） | `gres/gpu:a100:1` | 通过（符合预期） |
| 16542 PENDING | `JobState` | PENDING可通过提交检查，但不能通过科学RUNNING检查（runner） | `PENDING` | 预期拒绝（实际已拒绝） |
| 16542 PENDING | `AllocTRES` | PENDING尚无分配，不能通过科学单GPU检查（runner） | `(null)` | 预期拒绝（实际已拒绝） |
| 16542 PENDING | `StartTime` | 等待中的提交允许StartTime=Unknown（submission） | `Unknown` | 通过（符合预期） |
| 16542 PENDING | `RunTime` | 等待作业的RunTime为零（parser/runner field） | `00:00:00` | 通过（符合预期） |
| 16542 PENDING | `Requeue` | 仅记录Requeue，不据此拒绝（submission/runner） | `1` | 通过（符合预期） |
| 16542 PENDING | `NumCPUs` | 记录实际CPU数，不要求实际分配必须等于4（record_only） | `4` | 通过（符合预期） |
| 16512 CPU RUNNING | `JobId` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `16512` | 通过（符合预期） |
| 16512 CPU RUNNING | `JobState` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `RUNNING` | 通过（符合预期） |
| 16512 CPU RUNNING | `Restarts` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `0` | 通过（符合预期） |
| 16512 CPU RUNNING | `NumNodes` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `1` | 通过（符合预期） |
| 16512 CPU RUNNING | `NumCPUs` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `24` | 通过（符合预期） |
| 16512 CPU RUNNING | `CPUs/Task` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `4` | 通过（符合预期） |
| 16512 CPU RUNNING | `ReqB:S:C:T` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `0:0:*:*` | 通过（符合预期） |
| 16512 CPU RUNNING | `NtasksPerN:B:S:C` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `1:0:*:*` | 通过（符合预期） |
| 16512 CPU RUNNING | `Requeue` | 保留辅助CPU记录；实际CPU多于4可以接受（parser/auxiliary） | `1` | 通过（符合预期） |
| 16512 CPU RUNNING | `NumNodes` | 单节点（shared node parser） | `1` | 通过（符合预期） |
| 16512 CPU RUNNING | `ReqTRES` | CPU请求，无GPU（auxiliary） | `cpu=4,mem=16G,node=1,billing=4` | 通过（符合预期） |
| 16512 CPU RUNNING | `AllocTRES` | CPU分配必须被科学单GPU门槛拒绝（runner） | `cpu=24,mem=16G,node=1,billing=24` | 预期拒绝（实际已拒绝） |
| 16512 CPU RUNNING | `TimeLimit` | 辅助会话1小时时限，不套用科学wrapper时限（auxiliary） | `01:00:00` | 通过（符合预期） |
| 16512 CPU RUNNING | `StartTime` | 保留原始CPU会话启动时刻（parser） | `2026-10-31T20:42:02` | 通过（符合预期） |
| 16512 CPU RUNNING | `RunTime` | 正确解析辅助会话已运行时长（parser/runner field） | `00:08:21` | 通过（符合预期） |
| 16512 CPU RUNNING | `Account` | 辅助会话上下文不适用科学提交身份要求（submission） | `csci_ga_2271-2026fa` | N/A：上下文不适用（符合预期） |
| 16512 CPU RUNNING | `Partition` | 辅助会话上下文不适用科学提交身份要求（submission） | `n2c48m24` | N/A：上下文不适用（符合预期） |
| 16512 CPU RUNNING | `UserId` | 辅助会话上下文不适用科学提交身份要求（submission） | `xy3257(4882486)` | N/A：上下文不适用（符合预期） |
| 16512 CPU RUNNING | `Command` | 辅助会话上下文不适用科学提交身份要求（submission） | `(null)` | N/A：上下文不适用（符合预期） |
| 16512 CPU RUNNING | `WorkDir` | 辅助会话上下文不适用科学提交身份要求（submission） | `/home/xy3257/ondemand/data/sys/dashboard/batch_connect/sys/burst_jupyter/output/f8b8df63-186d-4e3f-8343-94e6b8be9d76` | N/A：上下文不适用（符合预期） |
| 12374 历史 COMPLETED | `JobID` | 保留历史表头，精确选择顶层job行（parser/history） | `12374` | 通过（符合预期） |
| 12374 历史 COMPLETED | `JobName` | 保留历史表头，精确选择顶层job行（parser/history） | `cv-qualified-reader-conf` | 通过（符合预期） |
| 12374 历史 COMPLETED | `Partition` | 保留历史表头，精确选择顶层job行（parser/history） | `c12m85-a100-1` | 通过（符合预期） |
| 12374 历史 COMPLETED | `State` | 保留历史表头，精确选择顶层job行（parser/history） | `COMPLETED` | 通过（符合预期） |
| 12374 历史 COMPLETED | `ExitCode` | 保留历史表头，精确选择顶层job行（parser/history） | `0:0` | 通过（符合预期） |
| 12374 历史 COMPLETED | `Elapsed` | 保留历史表头，精确选择顶层job行（parser/history） | `00:06:02` | 通过（符合预期） |
| 12374 历史 COMPLETED | `ElapsedRaw` | 保留历史表头，精确选择顶层job行（parser/history） | `362` | 通过（符合预期） |
| 12374 历史 COMPLETED | `Start` | 保留历史表头，精确选择顶层job行（parser/history） | `2026-10-26T20:10:28` | 通过（符合预期） |
| 12374 历史 COMPLETED | `End` | 保留历史表头，精确选择顶层job行（parser/history） | `2026-10-26T20:16:30` | 通过（符合预期） |
| 12374 历史 COMPLETED | `AllocTRES` | GPU计数为1；cpu=12不导致失败，允许generic GPU TRES（shared GPU validator） | `billing=12,cpu=12,gres/gpu=1,mem=24G,node=1` | 通过（符合预期） |
| 12374 历史 COMPLETED | `State` | 历史COMPLETED记录不接受当前RUNNING资格判定（runner live-state） | `COMPLETED` | N/A：上下文不适用（符合预期） |
| 12374 历史 COMPLETED | `UserId` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `Account` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `NumNodes` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `Restarts` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `TimeLimit` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `RunTime` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `Command` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `WorkDir` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `ReqTRES` | 该历史sacct表结构没有此字段（submission/runner） | —（缺证据） | N/A：缺证据（符合预期） |
| 12374 历史 COMPLETED | `visible CUDA device count/model` | 历史文本不能验证当前物理A100设备（runner physical GPU gate） | —（缺证据） | N/A：缺证据（符合预期） |
| 16542 CANCELLED sacct | `JobIDRaw` | 未获得分配即取消的原生证据（settlement） | `16542` | 通过（符合预期） |
| 16542 CANCELLED sacct | `State` | 未获得分配即取消的原生证据（settlement） | `CANCELLED by 4882486` | 通过（符合预期） |
| 16542 CANCELLED sacct | `Start` | 未获得分配即取消的原生证据（settlement） | `None` | 通过（符合预期） |
| 16542 CANCELLED sacct | `ElapsedRaw` | 未获得分配即取消的原生证据（settlement） | `0` | 通过（符合预期） |
| 16542 CANCELLED sacct | `AllocTRES` | 未获得分配即取消的原生证据（settlement） | `""`（空字段） | 通过（符合预期） |
| 16542 CANCELLED sacct | `Restarts` | 原生Restarts仍为零（settlement） | `0` | 通过（符合预期） |
| 16542 CANCELLED sacct | `StartTime` | scontrol取消时刻不能覆盖sacct的Start=None结论（settlement） | `2026-10-31T21:38:11` | 通过（符合预期） |
| 16542 CANCELLED scontrol | `AllocTRES` | 取消时没有已分配GPU（settlement） | `(null)` | 通过（符合预期） |

**实际运行时校验器整体返回。** 逐字段表之外，回放JSON保留了以下共用校验器的实际返回或异常信息。

| fixture | 实际调用 | 实际结果 | 解读 |
|---|---|---|---|
| 16542 PENDING | `validate_submission_control` | 通过 | 使用原历史上下文通过 |
| 16542 PENDING | `validate_running_control` | 预期拒绝 | Current native job identity must be confirmed RUNNING |
| 16512 CPU RUNNING | `validate_running_control` | 预期拒绝 | Native TRES must certify exactly one A100 GPU |
| 16512 CPU RUNNING | `one_gpu` | 预期拒绝 | Native TRES must certify exactly one A100 GPU |
| 12374 历史 COMPLETED | `one_gpu` | 通过 | 只确认历史TRES中的单GPU计数；不声明当前RUNNING或当前物理型号 |
| 16542 CANCELLED sacct | `validate_settlement_record` | 通过 | 返回cancelled_before_allocation、allocated_gpu_count=0、actual_a100_gpu_seconds=0；不推断科学成功 |
| 16542 CANCELLED scontrol | `validate_settlement_record` | 通过 | 返回cancelled_before_allocation、allocated_gpu_count=0、actual_a100_gpu_seconds=0；不推断科学成功 |

这确认了原16542问题的修复：`NumNodes=1-1`与`1`均表示一节点；两端不是1的范围被拒绝。16512实际CPU=24而请求4、12374实际CPU=12均未被误当作GPU检查失败。16512没有GPU，因此科学GPU校验器仍必须拒绝；12374为历史COMPLETED，不能用来冒充当前RUNNING证据。generic与typed GPU TRES若同时出现，计数不相加。

新的零分配结算函数对保存的16542取消证据返回0GPU/0秒，与sacct的Start=None、ElapsedRaw=0、空AllocTRES及scontrol的空分配证据一致。scontrol的StartTime=EndTime=取消时刻不会被当作科学启动证明。这是新代码对旧证据的离线回放；原v2.3 `stopped_after_submission`、`stopped_accounting_error`及所有原始记录保持原样，未把旧helper运行状态改成通过。

**假时钟与边界覆盖。** 其余测试使用明确标注的人工字符串、fake command和fake clock，不把修改过的字符串当作真实fixture。98项通过包括以下范围：

- sacct记录暂缺后RUNNING；PENDING/CONFIGURING使用Start=None/Unknown、AllocTRES为空/(null)，随后RUNNING。
- 每轮先校验scontrol；错误job、Restarts非零/缺失、非单节点、错误GPU数量/类型、非RUNNING、错误时限/起始时间立即失败，不等待。
- sacct已出现的非法字段、矛盾起始时间、畸形行和终态立即失败；不能用启动宽限掩盖。
- 启动总宽限精确120秒，命令耗时与fake polling sleep均计入；达到边界后不再发命令。运行期检查默认0秒宽限。
- 成功启动的等待计入分配用时，通过monotonic补偿防止原生计时暂时滞后造成少计。
- 保留`CPUs/Task`、`ReqB:S:C:T`、`NtasksPerN:B:S:C`独立字段；五列无表头行的末尾空AllocTRES与额外末分隔符均正确处理。
- 缺失/重复/坏字段、非单节点、GPU数量/类型冲突及零分配结算中的矛盾原生证据被拒绝。

B复用来源检查仅读取原protocol、status、artifact清单的JSON及文件原始字节，核对SHA256；没有用NumPy打开NPZ、重算B或修改原B工件。完整科学函数不变性另见[静态AST/JSON核查](../results/motion_v2/v2_3_1/implementation_invariance.json)：七个科学函数和BudgetGuard类与基线相同，配置只将运行估时上限7200改为10800；B helper只有来源验证函数发生变更，数学及target读取函数不变。该静态核查不冒充GPU运行测试。

**证据边界。** 回放通过说明已保存真实格式和登记边界在这版运行层代码中得到预期处理，不证明集群未来一定返回这些格式，也不表示已完成新的A100科学作业。a/b/c/d的feature/position MSE、B−b等六项研究判据不属于这98项软件测试，不能据此宣称预测器过关。没有重跑28项科学统计数组测试，没有训练保留策略或进入第3步。后续实际部署和作业状态应由单独的原生记录与执行报告说明。
