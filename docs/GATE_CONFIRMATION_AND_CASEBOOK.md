# 从线索诊断到独立确认：目前能支持什么

本轮先在已观察的开发场景中拆开内容、时间和动作，再把已有保留探针原样放到新 seed 的96组配对场景上。**结论是：预测器确实会错误利用某些正确线索，但当前删除价值探针没有在新场景上建立保留优势。** 这两件事需要分开报告。

## 为什么保住线索反而更差

[完整组合诊断](GATE_CUE_FACTOR_DIAGNOSIS.md)使用72个可比较的真实 attention 父集合，每个执行12个内容／年龄／动作组合，共864条预测；144个原始端点逐值复现。

在原始27个失败案例中，端点误差差距的顺序平均分配为：内容+0.045316、年龄+0.000615、已执行动作摘要+0.007793。内容分配最大，动作也有影响；年龄的平均区间跨零。交互项不为零，这些数值是固定模型输入干预的差距分配，不是物理原因百分比。

最直观的案例中，原 noncue 子集误差0.127822；**只换入正确线索内容、保持时间和动作不变，误差升到0.324210**。实际正确 cue 获得0.999771的 attention，却把预测推向错误的 sibling 未来。因此，保存成功、attention 命中和预测利用成功是不同环节。

不过全部72个案例中，48个使用正确 cue 比匹配错误 cue更好，24个更差。不能说整个模型统一学反了。原失败子集中21/27正确 cue更差，但该条件子集均值的区间跨零；它是已观察失败样本的诊断，不是新数据上的普遍反转结论。

## 新场景是否支持可学习的保留收益

事先冻结[协议](GATE_CONFIRMATION_PROTOCOL.md)，使用新 seed20261023、96对／192条轨迹，沿用旧 encoder、normalization、epoch35预测器、三个 PaFu student和六个线性探针。没有重新训练、调参或选取新结果最好的 arm。新数据属于同一个生成器家族的独立种子确认，不是新任务、真实视频或正式Phase1测试。

主要对象固定为 masked cost ridge，只有它对下列三个对照的配对误差差值区间上界都小于零，才通过：

| masked cost 减去对照；负值更好 | 平均差值 | 95%配对区间 | 解释 |
|---|---:|---:|---|
| 同容量 shuffled-cost | −0.000273 | [−0.003626, +0.003581] | 没有确认超过打乱标签 |
| Reader attention | +0.008909 | [+0.004461, +0.013363] | 显著更差 |
| Visual diversity | +0.007823 | [+0.002915, +0.012922] | 显著更差 |

**预先固定的主要判断未通过。** 不能改挑其他探针、条件或单个案例来代替这个判断。完整13个arm、同parent局部regret、任务分支检查、全部行及审计见[确认报告](GATE_FROZEN_CONFIRMATION_REPORT.md)。本轮不把探针升级为proposal的新方法，也不据此宣称删除价值原则不可学。

## 先看这些案例，再读论文

- [CV真实RGB互动图册](../results/literature_diagnosis/visual-casebook-20261022/index.html)：三个预选案例，可播放两条配对轨迹、跳转cue/query/未来。下载HTML后可离线打开。
- [CV中文看图说明](GATE_VISUAL_CASEBOOK.md)：内容读错、年龄参与、正确cue成功对照。图片均为缓存的真实输入和真实未来；特征预测只用数值图展示，没有伪造解码视频。
- [PM已有四个案例](https://github.com/XintongYang1010/predictive-memory-lab/blob/experiments/bank-reference-20261022/docs/PM_CASE_GALLERY_ZH.md)：有害旧值、噪声主导、成功对照、旧A2小效应。仅重画已有结果；缺失原始RGB的部分明确画成残差/时间/attention图。
- [论文阅读顺序与问题](https://github.com/XintongYang1010/predictive-memory-lab/blob/experiments/bank-reference-20261022/docs/PAPER_READING_GUIDE_ZH.md)：PaFu-KV → Worth Remembering → Gated DeltaNet。

## 与两个proposal结合后的下一步

共同研究问题仍然是**给定剩余记忆与当前任务，某条证据的增量贡献是什么**。目前的结果要求先把可用证据、正确读取、噪声偶然收益与事后未来选择分开；oracle gap本身不是可学习空间的证明。

CV下一轮应先建立可靠的固定证据预测对照，核验正确cue与已知控制的映射；若更换预测器，所有保留方法必须共用新预测器，并重新保留未查看的新场景。之后才值得继续比较未来attention和删除价值监督。不能在这批已看过的确认集上反复修模型，再称其独立确认。

PM继续保留已有反例和成功对照：删除有害旧值只是避免错误读取，未必恢复正确新状态；单次噪声实现也可能制造不可利用的oracle改善。旧A2物体坐标修补的账面0.0125%空间不重启为主线。这些是相通的诊断问题，不是同一个机制的跨仓库复现；不合并两仓库的MSE或把相同K称为相同总字节。

与Saining讨论时，可以直接展示失败与成功图，并问：先改善共享reader的任务有效性，还是已有足够可靠的场景能隔离保留策略？当前证据支持先回答这个问题，尚不支持宣称新memory方法已经优于强baseline。

## 执行事实与保留边界

通过Codex内置浏览器完成NYU登录、Jupyter上传和提交。A100作业10167正常完成，`COMPLETED / 0:0`，分配123 GPU秒；192条轨迹、5,822逻辑帧合并为1,440张独立RGB编码。实际设备为A100-SXM4-40GB，Torch GPU UUID与nvidia-smi匹配；数据包下载哈希一致。临时CPU控制会话10162随后结束。详见[执行凭证](../results/literature_confirmation/gate-seed20261023/execution/execution_receipt.json)。

时间顺序为UTC 03:14:30锁协议 →03:20:07提交 →03:21:54开始编码。CPU确认未新增GPU开销。源、参数、数据哈希与先锁后编码记录均保留；导出审计复算保存结果，不冒充独立重跑reader。

正式Phase1仍是Gate155/155、Motion147/155，联合规则未通过；本轮没有改动。Motion历史收益、model-driven PushT baseline planning及proposal最终方法仍未完成。画廊与负结果不是这些交付的替代品。
