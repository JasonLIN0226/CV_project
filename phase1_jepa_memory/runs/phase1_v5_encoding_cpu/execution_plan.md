# CPU 替代路线：已撤销，历史准备记录

> **已于2026-10-13 15:25 EDT由用户撤销CPU替代路线。以下仅保留历史准备记录，不得部署、编码或提交；当前路线见GPU恢复状态。**

2026-10-13，按用户“自行重新调配实验、跑训练、交完整报告”的授权准备。截至15:00EDT，本机已有密钥和Torch已恢复；7135在Ready前结束，正常管理SSH尚未建立会话。没有部署、编码推理、特征拟合、真实训练或新的HPC科学作业。已有 Gate 失败、Motion 公开失败及全部原门槛保留。

## 已完成的准备

独立 CPU 编码包共 23 个源文件；CPU 缓存核验和真实 reader 软件包共 17 个源文件。打包副本分别通过 53 项和 18 项人工软件检查；源码、包内载荷及旧 L4 18 个冻结源文件的 SHA 均重新核验。软件检查不是实际实验结果。

完整数据保持 58 train / 20 validation、78 段开发 RGB、198 个片段、50688 个时点、7524 个查询；冻结官方 IJEPA 原 revision、grid2、float32、batch2、原 causal PTS 选择不变。模型、数据及数组留在 HPC。封存 18 RGB / 9 人和独立确认 seed9137 仍未打开。

## 唯一执行顺序

1. 已实际核验批准密钥和Torch恢复。新7135已在Ready前结束，不重复申请，不执行其guard。优先等待现有7004/7005真实路由；正常管理登录只有实际会话建立后才可用于元数据/传输。网页认证成功不等于SSH登录成功。新分配身份仍须正常公开身份与原生keyscan严格匹配，不复用已结束IP、不用浏览器上传绕过。
2. 读取实际 Slurm、7004/7005 状态及日志。若 7004 已进入合法编码，保留其执行；若仍未起 batch，保存真实启动失败证据后取消其及依赖 7005，验证终态与无并行科学作业，再提交一次独立 CPU 编码。不在备用路线可提交前取消现有机会；不修改原冻结 checkout。
3. CPU 编码固定课程 csci_ga_2271-2026fa / n2c48m24，零 GPU，4 CPU / 16G / 4h。首次固定 512 帧的完整视频哈希、因果解码、编码、写入和哈希时间，加实际加载/检查开销，投影完整 50688 帧必须小于 13800 秒。该 512 帧编码只执行一次并复用。超预算保存真实部分输出并停止；不缩数据、精度或延长预算。实际 CPU 速度尚未知。
4. 绑定实际 CPU 编码 job ID，另行锁定 CPU 缓存审计源与协议。只有父作业实际 COMPLETED 0:0、零 GPU 资源证明、完整 78 文件 dtype/shape/finite/前后 SHA、所有 PTS/episode 摘要及官方权重/processor/source 绑定一致，才可认定完整缓存。
5. 完整审计后，将实际 cache、encoding report、audit 和审计 source SHA 写入新的正式训练协议。当前两个 binding template 均不是有效正式试验协议，不能拿空值提交。CPU reader 保持 256 / 2 layers / 4 heads、128 train-only bases、seeds17/29/43 各30轮、recent8/capacity16、h1/4/16/64、四原策略、原 AdamW / dropout / schedule。4 CPU / 16G / 4h；首完整 train+validation epoch ×90 加 setup 投影须小于13800秒，否则停止，不能减种子或轮次。
6. 实际六份 best/latest 的来源、有限值、优化器、调度、RNG、完整计数和最早验证 best 全部证明后，才做预注册评估。依次执行固定 reader 健康检查、17 个删除选择和共同 32 步 Reservoir 延续、不可逆 FIFO/Reservoir/random 重放及 task-vs-drift/recovery；各阶段正式协议和源在实际运行前冻结。目前仅延续 primitive 和 reader 软件已准备，完整后续科学评估尚未运行。
7. 原 RGB 与官方标签抽帧相位未证明；无标签特征预测可继续，不能猜测语义标签/帧匹配。原 synthetic 没有可臆造的32步延续。Gate REJECTED 仍阻止独立确认；真实实验不能补救其失败。

## 报告与核验

报告分别列 completed / scientific rejection / infrastructure failure / incomplete、原种子和门槛、训练与正式评估、实际资源/时间、已知和未知原因、所有可复核标量和日志。恢复 Motion 最终 Slurm、六实际模型及 compact 结果后独立重放原32检查；旧 helper 的7050 guard须绑定实际新分配。只通过原生 SCP 返回明确小型 JSON、标量 CSV、日志和图，每项前/本地/后 SHA。无真实结果时保留“未运行”。

编码正式准备：[协议](../../../configs/phase1_v5_real_encoding_cpu_protocol.json)；本地核验：[证据](software_and_bundle_proof.json)；当前真实结果：[晨报](../phase1_v5_gate/overnight_report.md)。
