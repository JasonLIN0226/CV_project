# Gate2 终局删除 pilot：运行完成，完整数据核验待恢复传输

2026-10-25。**A100 实验已完成；本页目前只报告已取回的执行日志和预先锁定的设计，完整逐行结果尚未完成独立审计。** 不把日志中的自动判定当成已核验的方法收益。本轮没有训练 retention student，也没有运行新的 PM 实验。

## 已完成与目前可以说什么

| 检查 | 已确认的事实 | 当前解释边界 |
|---|---|---|
| 执行前锁定 | 源码/协议先冻结，110个打包源码文件均与执行前提交 `3bd9f33` 一致，再提交作业 | 没有按结果改生成器、seed、门槛或reader |
| 原始结构 | 28项测试通过：2×2 siblings、相同 recent/动作、两独立语义位、四种未来、真实终端 R4/K2 parent | 原始结构成立不等于特征几何或reader成立 |
| 实际GPU | 作业内 CUDA 与 `nvidia-smi` UUID一致，A100-SXM4-40GB，作业11310 | 门户登录和申请资源与此验证分开 |
| 调度结束 | `COMPLETED / 0:0`，307秒，约0.0853 GPUh | 包含本次编码、训练、评价及打包的分配时间 |
| GEOMETRY日志 | benign：0/48条轨迹保留冗余对；改变副本外观：24/48条，涉及6/12 quartets | 这是专用几何集，不能冒充DEV结果；距离margin/tie待逐行复算 |
| Reader日志 | 总体与两种style合计24/24项门槛通过 | 尚未独立复算各arm的实际准确率、MSE与checkpoint身份 |
| 自动分流 | `candidate_terminal_gap_requires_independent_confirmation` | 仅日志判定；强baseline差距量级、分布、oracle一致性仍待核验 |

这比旧Gate更有针对性：原始任务中两个bit都改变未来，只有两份互补信息才足够；几何日志也显示外观干预确实改变了diversity选择。但现在还不能声称“集合条件打分必要”“新方法有效”或“oracle全部收益可学习”。

## 五条设计要求如何落实

以[执行前协议](GATE2_TERMINAL_PILOT_PROTOCOL.md)和[冻结源码](https://github.com/JasonLIN0226/CV_project/commit/3bd9f33)为准。

1. **几何先于reader。** 使用固定 I-JEPA 和TRAIN-only normalization，调用真实 diversity 实现，保存三对特征距离、实际选择及平局规则。两个style完全共享基础family，只改变重复副本的呈现方式；不把外观名称当作距离结论。
2. **四种sibling。** 00/01/10/11分别改变未来x/y方向，recent与动作相同。A/A′表示被重复的物理轴，B是另一轴；重复x还是y、六种cue顺序全部平衡，不能靠固定“保最后一帧”解题。
3. **真实最后一次删除。** cue在q−6/q−5/q−4，共同FIFO前缀在q形成真实K+1集合，R=4。删除后冻结，不再写入未来空白帧，也没有后续attention策略替候选动作收尾。
4. **只换内容。** 固定两槽age为[6,4]，固定recent及过去/计划动作，比较AB、A′B、AA′、Ablank、Bblank、blank2、wrongA/B。24项门槛要求每个互补arm分别胜过每个单证据arm，并通过两个单bit反事实。预测分类为同quartet四种真实未来特征的原型识别，不能称为物理坐标解码准确率。
5. **语义最优与reader最优分开。** 同parent枚举三个删除，统一h3/h6目标，保存精确和1e−6容差最小集合、共同最优动作及语义正确集合。不同siblings有不同过去cue，因此动作不同不自动证明不可学习；任务语义一致也不能保证reader误差的最优动作一致。

数据规模为TRAIN/VAL/GEOMETRY/DEV 240/48/24/48 quartets。DEV的192条轨迹只对应24个独立family，置信区间按family而非episode重采样。模型选择只使用VAL。本轮没有最终确认集。

## 后续方法比较的解释边界

以下是用户要求保留的后续判读规则；本轮只预注册并执行到reader与终局baseline检查，学生比较尚未单独预注册或运行。

- 强baseline已经保住互补证据并解决实际损失差距：报告负结果，不靠筛选新seed扩大规模。
- attention保留冗余、集合条件学生改善：只有与同容量独立学生及打乱标签控制比较、再经新seed确认，才讨论集合依赖的价值。
- 独立学生也能解决：报告不需要集合条件。尤其本pilot的特殊外观副本可能产生简单规则捷径；这必须成为对照，而不是事后忽略。

本轮学生数量为零。下一步首先完成当前192条DEV结果的独立算术审计，查看损失margin、四个sibling最优集合是否一致、内容对照与原生时间条件是否一致，再决定是否值得预注册学生比较。

## 文件与传输状态

已取回并纳入仓库的证据：

- [执行前冻结凭据](../results/literature_diagnosis/gate2-terminal-pilot-20261025/freeze_receipt.json)、[独立审计脚本冻结凭据](../results/literature_diagnosis/gate2-terminal-pilot-20261025/audit_freeze_receipt.json)
- [原始结构预检](../results/literature_diagnosis/gate2-terminal-pilot-20261025/raw_preflight.json)、[原始RGB接触表](../results/literature_diagnosis/gate2-terminal-pilot-20261025/raw_contact_sheet.png)
- [完整运行日志](../results/literature_diagnosis/gate2-terminal-pilot-20261025/slurm-11310.log)、[Slurm完成记录](../results/literature_diagnosis/gate2-terminal-pilot-20261025/sacct-11310.txt)、[远端打包凭据](../results/literature_diagnosis/gate2-terminal-pilot-20261025/zip_receipt.json)
- [传输状态](../results/literature_diagnosis/gate2-terminal-pilot-20261025/transfer_status.json)

完整远端ZIP含136个源/结果文件，大小345,671,988字节，SHA256为`523654adcb889726af0a8ce8a0dd3e59dba71b8c99e9634a2cd84ad09ed9be5c`。原始文件仍在`/scratch/xy3257/cv-phase2-20261020/gate2-terminal-pilot-20261025/`，未删除。归档包含模型、basis、normalization、完整RGB/features、所有预测/目标、逐行决策和源码。

Jupyter与OOD的下载事件均未返回完成；分块文本导出已校验21/83块，随后内置浏览器出现`ERR_NETWORK_IO_SUSPENDED`，其生成的错误页又被URL策略阻止自动操作，已请求用户手动恢复。**本地完整ZIP、独立数据审计、完整GitHub Release尚未完成。** 不可把当前PR或本页说成已包含全部数值结果；没有为此重跑GPU任务。

完整恢复后运行：

```sh
python3 experiments/phase2/audit_gate2_pilot.py --run <解包目录>/run --source-root <解包目录>/source --output <审计结果.json>
```

审计使用NumPy重算保存的结果，不重新调用encoder或reader；与独立推理复现区分。

![同一family的两种外观与四种真实未来](../results/literature_diagnosis/gate2-terminal-pilot-20261025/raw_contact_sheet.png)
