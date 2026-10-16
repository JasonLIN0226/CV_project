# 夜间实验结果报告（持续更新）

截至纽约时间 2026-10-13 07:04 EDT。

**Gate 新因果读出器的正式开发评估：REJECTED。** 三个种子全部120轮、六个实际best/latest和来源/缓存/基底/RNG/优化器/调度/选模证明通过；失败来自原科学门槛，不是训练中断。seed29在三个测试条件、h3/h6全部不满足记忆收益和内容替换干预要求，不能由另两个种子或pooled均值补救。Phase1未通过，独立确认保持未打开。

| seed | 选中epoch | 原记忆收益 | 内容替换干预 | 固定参考recent | 关键绝对CI |
|---|---:|---:|---:|---:|---:|
| 17 | 57 | 6/6 | 6/6 | 6/6 | 2/2 |
| 29 | 4 | 0/6 | 0/6 | 0/6 | 2/2 |
| 43 | 66 | 6/6 | 6/6 | 6/6 | 2/2 |
| all_seeds_mean | — | 6/6 | 6/6 | 6/6 | 2/2 |

原数据证据可读性检查3/3通过；原记忆收益18/24、内容替换18/24、固定参考recent18/24，干扰变化关键绝对CI8/8。必须全部原规则成立才采用；某一类检查通过不能替代其它检查。

seed29相对固定原参考近期预测的latent MSE收益：

| 条件 | h3 | h6 |
|---|---:|---:|
| ID | -2.49% [-3.72, -1.25] | -2.70% [-4.37, -0.97] |
| 长间隔 | -2.01% [-3.16, -0.81] | -0.72% [-2.78, 1.26] |
| 干扰变化 | 8.13% [6.23, 10.03] | 4.95% [3.14, 6.70] |

![Gate各seed、条件和horizon的固定参考收益与原95%配对区间](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/fixed_reference_gain.png>)

[可导出的结果图PDF](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/fixed_reference_gain.pdf>)。该图直接使用已核验Gate结果的24项固定参考比较；红色为未通过切片，平均值不替代逐seed验收。

区间是原规则2000次按完整base-pair重采样的95%配对区间。原规则要求收益点估计至少10%且区间下界大于0（接近零基线按原绝对规则）；没有更改阈值。上述结果说明seed29未可靠利用历史，尚不能确定具体优化、背景或数据设计原因。

![Gate三seed的实际训练与验证曲线](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/training_curves/gate_learning_curves.png>)

训练记录补充：Gate seed29选中第4轮时，近期/关键历史验证MSE为0.10287009/0.10292404；到第120轮，等权训练MSE继续下降，但等权验证MSE升至0.12094947，高于所选0.10290398。曲线显示训练与验证分歧；它与正式记忆收益失败相一致，仍不足以定位具体优化或数据原因。未据此重训、换checkpoint或调参。三份120轮JSON逐项匹配实际模型完成证明中的SHA，原生传输前/本地/后均相同。

[训练记录和曲线摘要](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/training_curves/learning_curve_observations.json>)、[PDF](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/training_curves/gate_learning_curves.pdf>)、[传输核验](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/training_curves/transfer_integrity_0710.json>)。

| 实际阶段 | 状态 |
|---|---|
| Gate训练6990 | COMPLETED0:0；02:45:53–05:01:41 UTC；3×120轮、各7680步 |
| Gate评估6994 | COMPLETED0:0；05:03:23–05:06:12 UTC；34560条完整标量；科学采用REJECTED |
| Motion训练7000 | 已不在活动列表；后续7001完成日志已出现；最终Slurm和seed43完整证明待原生取证 |
| Motion评估7001 | 正常门户stdout明确写REJECTED、34560标量、156.31秒；stderr0；逐seed/原门槛及六模型独立原生复核待取回 |
| 实际Assembly101编码7004 | 门户CONFIGURING；门户最近StartTime为10:52:11 UTC、State=CONFIGURING，此前07:42:08/08:22:08均有记录；重启次数/原因待Slurm取证，日志目录仍空，未证明batch或实际编码启动，尚无cache |
| 完整缓存核验7005 | PENDING/afterok:7004；CPU4/16G/30分钟；尚未读取真实特征 |
| 真实reader | L4版15项本地及HPC人工数据软件检查通过；只部署软件检查副本，无正式新试验协议、真实拟合或训练提交 |

当前唯一科学分配为7004，仍在节点启动阶段；7005依赖排队。门户10:52:11的新StartTime不代表编码程序已运行，也不能据此推断重启发起者。所有阶段严格顺序，不重复或并发提交。Gate失败已保留；Motion和真实辅助可行性不能替代Gate原验收，也不能授权fresh9137确认。

Motion以下训练表记录原生核验截至07:15:59 UTC的历史快照；此后7001公开日志已明确完成并REJECTED，不能把历史95/120当作现在仍在训练。其逐seed验收结果和最终Slurm须原生取证后补全，暂不臆造。

Motion首完整训练/验证轮29.88秒，固定总预算预测10791.4秒 < 13800秒，预算检查通过。seed17/29的实际best/latest均完成120轮/7680步；有限模型/优化器、完整RNG、调度、每条件1024样本、等权验证最早最小值、原basis buffers及学习率逐项核验通过。该历史快照中seed43为95/120、6080步，其当时完整轮边界通过。最终三个模型完成证明仍待取回并另行重算；公开7001日志不能替代这一独立复核。

| Motion seed（07:15历史快照） | 当时完成轮数 | 当时选中epoch | 当时验证MSE | 7001逐seed结果（当前） |
|---|---:|---:|---:|---|
| 17 | 120/120 | 79 | 0.13437433 | 原生结果未取回 |
| 29 | 120/120 | 119 | 0.11576871 | 原生结果未取回 |
| 43 | 95/120 | 暂时55 | 暂时0.15044360 | 原生结果未取回 |

上述是训练/验证与选模记录，不能替代原门槛。Motion原start/end两帧、数据/缓存/128维train-only基底、B2/recent4/h3,h6、MSE+2pair+.5rank、优化器、三seed/120轮及所有原门槛保留；没有恢复旧wide-span/pure-velocity分支。

L4路径依据实际A100启动失败选择：6965/6980被UID0取消且无batch/日志/输出，具体原因未知；6985未起batch后按DDL指令由agent取消，重启发起者未知。原A100准备13文件保留。L4节点广告47000MB主机内存，60G请求不适配，改为32G；encoder官方版本、float32/TF32off/grid2/batch2、78开发录像、50688时点、PTS及首512帧/四小时预算均保持。20项HPC软件测试通过，但实际GPU速度和成功尚未知；单次L4分配尝试，不盲重试或扩大GPU数。

另在7000已分配节点、单CPU线程运行15项人工数据软件检查（1.55秒）：验证未来特征不能改变当前输入、padding不影响预测、验证集污染不能改变训练集拟合、全维损失梯度，以及缺失/错误实际缓存证明会被拒绝。8个软件源文件锁定；没有读取真实特征、拟合真实基底或提交真实reader科学任务。

7005将检查实际7004COMPLETED0:0/一GPU、78完整文件SHA/大小/形状/float32/有限值、全部episode/PTS摘要、实际编码源码/官方权重/处理器来源；10项人工缓存HPC检查通过。核验不拟合基底、不训练。真实reader必须先取得实际完整cache/audit SHA，再建立独立正式训练协议和源锁；不存在的旧6980缓存没有被冒充有效。

已完成数据部分：78开发RGB、完整PTS及稀疏解码检查、开发标签可用性核验。原视频60fps与官方30fps抽帧标注的相位仍未证明。进一步只读核查[HandFormer固定版本的预处理源](https://github.com/s-shamil/HandFormer/blob/640a10a6bd03f85842a2662e67cf30444ab1d673/Preprocess/1_get_clip_data.py)：其30fps标注对应两份60fps姿态样本，并引用预先提取的JPEG路径；该消费者没有提供原MP4到JPEG序号/PTS的抽帧规则。它支持速率比例，不能证明我们C10095_rgb的确切时间相位。5个源码文件与固定commit的Git blob逐项匹配，没有运行该预处理或读取额外标签/特征。禁止猜测label/frame join；合法无标签特征预测继续。[来源审计](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_data/handformer_annotation_phase_source_audit_verified.json>)。封存18段真实RGB/9参与者及fresh9137未打开。

7050此前已实际NotebookReady，公开b-10-107/IP34/ED25519与nativeTorch keyscan逐字匹配；严格原生SSH随后因本机agent无密钥退出255，现有钥匙串也未自动解锁。已一次通知用户在本机运行原hpc.sh login，不要求聊天秘密。科学作业可继续，但逐seed结果与六模型原生复核暂受这一凭据依赖阻塞。此前临时连接取消动作遇到浏览器超时；本次正常门户已明确7050 Completed，无临时资源活动。最终Slurm原因尚未读取，不能推断由先前取消动作还是时限结束。IP34已失效。

本次取证连接7049在15分钟窗口中从未NotebookReady，门户最终Completed；最终Slurm原因尚未读，不推断超时或取消。正常门户SSH终端同样显示连接终止，未执行命令。已按独立记录的一次恢复预算提交7050：课程n2c48m24、零GPU、1CPU/4G/30分钟，仅用于只读取证及小结果原生传输，用完释放；不重复科学任务。7004仍保持原单次L4分配，没有重交、缩小数据或放松验收。

取证过程：6994的原生连接未返回并退出255，原生Slurm随后确认其分配在连接尝试前已经正常结束；转发失败具体机制不推断。通过新的7000分配公开身份与nativeTorchkeyscan精确匹配恢复取证，没有重跑评估，当时没有新认证问题或放宽主机验证；本次新的本机密钥到期依赖见上段。所有权重、数组和视频继续留HPC，只取回明确的JSON/标量CSV/日志。

可复核结果：

- [完整Gate结果](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/report.md>)、[全部原门槛与32项比较](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/development_comparison.json>)。
- [六个实际模型完成证明](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/all_models_completion.json>)；另一段原生只读程序重算与实际评估完全相同。
- [34560条原始标量](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/evaluation/raw_records.csv>)、[逐seed/OOD/配对区间](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/evaluation/metrics.json>)。
- [12个传输前/本地/后SHA核验](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/transfer_integrity.json>)；原始标量SHA16273453160efc3b643cfabbb4c36c2bcc7aa53d5a80346f979960e6b5690e79。
- [32项标量重放](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_gate/evaluation_results/fixed_reference_scalar_replay.json>)：独立于项目stats函数，最大数值差1.38778e-17、所有通过/失败标签一致。这是root数值审计，不是新委托独立审稿或新实验。
- [当前Motion来源、前两个模型完整证明与当前断点](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_motion/monitor_20261013T0710.json>)、[seed17完整训练证明](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_motion/seed17_completion_20261013T0552.json>)、[原basis及当前进度](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_motion/seed17_basis_progress_20261013T0554.json>)，[L4提交](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_encoding_l4/submission_7004.json>)，[缓存核验提交](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_encoding_l4/cache_audit_submission_7005.json>)。

[15项HPC软件检查](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_encoding_l4/real_software_preflight_results/software_test_proof.json>)与[5个小文件原生传输一致证明](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_encoding_l4/real_software_preflight_results/transfer_integrity_20261013T0552.json>)已保存；当前7套冻结科学源逐文件重验不变。

Motion的32项固定参考检查独立重算程序已准备：用已曝光、已保存的旧v3 Motion标量验证，最大差1.73472e-18，通过缺项、重复项、错误关键split三种拒绝检查；旧v3 REJECTED不变。默认入口要求六个当前模型完成证明后才读取新结果。此项仅是历史标量兼容性/程序检查，该历史软件检查没有打开v5 held-out或改变门槛；7001此后已实际运行，新标量仍待原生取回。[软件准备证据](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_motion/scalar_audit_software_preparation.json>)。

半小时监控继续。纽约时间08:00前补充实际完成或未完成/失败的报告；不保证通过，不把排队、软件测试、验证误差下降或oracle诊断写成科学成功。所有原v3/v4失败保持。

共同32步Reservoir延续的12项本地人工软件检查通过：17删除分支同一时钟/容量/随机序列，删除不可逆，拒绝越界时点与非法随机数；进一步要求最后延续步的h64目标也在片段内（索引255允许、256拒绝），不会近边界截短成本或省略长horizon；没有模型推理、真实数据或科学诊断执行，活动/pending源锁未修改。
