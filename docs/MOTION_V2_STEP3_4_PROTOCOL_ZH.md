# Motion v2 第 3/4 步预登记：删除代价学生与在线 B=2 验证

状态：运行前冻结。机器可读登记为 `experiments/motion_v2/protocol_step34.json`，含本文件、配置、复用资产清单及全部执行源码 SHA256。先提交并推送登记，再运行软件测试和 CPU 数据步骤；CPU 审计通过后，将其文件哈希及测试回执写入独立 `protocol_step34_execution.json`，再次提交推送，之后才允许 GPU 作业。原 Step2、原停止记录、Phase1/Gate 源码、已有结果及哈希锁保持不动。

## 1. 授权范围与不变资产

只使用原 train512 对、ordinary validation96 对，及本次新增 long-gap validation96 对。第4步后无论结果如何停止，不生成、打开或选用确认集与 test；不创建第5步数据、不合并 PR。新的工作仍在 `experiments/motion-v2-20261030` 分支及 PR#3。

复用作业16614、科学提交 `6dc5ed253dfa305b26be8f19d40edcd6a5878ea7` 的全部1216个 episode 编码特征、train-only normalization、basis、位置 decoder，以及 predictor seed17/29/43 各自 selection.json 指定的 checkpoint。冻结清单 `step34_assets.json` 从已结算的原 `artifact_sha256.json` 提取：包括1216个特征 NPZ、index/provenance、basis、selection/权重、decoder、配置及原 I-JEPA 缓存。部署前和作业内读前核验，结束时再次核验。不得重训预测器、重拟合 basis/decoder/旧 normalization、重新编码旧帧。旧配置中的通用 `mode=confirm` 字符串不代表本次可访问确认集；manifest 强制只有 train/validation。

原数据 manifest SHA256：`9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a`。旧资源位置是 deployment3/repo，venv 只读复用 deployment1。新部署根为 `/scratch/xy3257/motion-v2-20261101-step34-deployment1`，不得改写或删除旧现场。

## 2. 预测器诊断与新 validation 数据

### 普通间隔：任意两张新段帧

在每个 validation pair 的 `t_c..end` 可见帧中，不放回均匀选两张，排序后作为 K=2 memory；抽样 seed20261102 与 opaque pair ID 组成稳定哈希种子，两个 sibling 共用抽样。只抽一次，不依据预测损失筛选。与原 b（首张新段可见帧+end）、d（最后旧段可见帧+end）、方向下界 B 并列报告。h3/6 各报告 feature MSE、position MSE 及配对95%区间，另报 B−b、d−b、B−random、d−random 等描述性差值。该诊断不增加停止门槛。

### Long-gap validation

新 seed20261101，仅生成 validation96对，gap均匀整数20–32。沿用已冻结 v2.1 场景与渲染：两车道、独立 x_pre/x_end、0–6的t_on、L0 4–10、L1 3–7、速度绝对值0.008–0.012、25%遮挡和可见帧约束等全部保持；只有 seed、split 规模和 gap 范围改变。按原语义 `query=end+gap+4`，R4/B2/h3,6，最后4帧均挡板。新入口显式限制96个 validation pair；原 envs_v2.py 不修改。

CPU 首先核验每对旧段/最近4帧/动作/end位置一致、新段方向相反、可见约束、隐藏状态分离、ID/路径/字节哈希。泄漏阈值与Step1相同：end_only、stale_plus_end 在[.47,.53]，one_post_non_end≤.58，two_post≥.95。探针采用与Step1相同的特征构造和 HistGradientBoostingClassifier(max_iter=200,random_state=0)，固定 frame-choice seed7。本次采用原 train512对拟合四个探针、仅在新 long-gap validation96对评分；不在96对 validation 上拟合或选择探针。这是预先登记的训练/评估划分，区别于Step1在512对 train 上的五折审计；阈值不变。保存 fit IDs、选帧、输入/标签/预测哈希、全部分类器参数、Python/NumPy/sklearn/平台版本。任何一项配对/泄漏审计失败即停，不换seed、不调探针、不重采样。

通过 CPU 审计后，只编码这192个新 episode。使用原官方 I-JEPA revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e`、FP32、无裁剪、四格池化5120维、batch2、离线原缓存；无模型下载或替代版本。仅应用旧 train-only normalization/basis/decoder，不拟合新变换。

冻结预测器在新 validation 上评估 a/b/d 与 B。B为每个 pair、horizon 的两个真实位置相对 sibling 均值的 XY MSE，再平均 siblings。h3和h6各自的 B−position MSE(b) 配对95%下界都>0，才继续报告 long-gap 在线比较；否则后续主张及在线结果限普通间隔。d−b 和 d−B 同时报；B−b 不通过只限制间隔范围，不触发普通间隔实验停止。真实 validation feature 经冻结 decoder 的 position MSE 与 B 并列。

## 3. 父集、标签与学生

### 在线状态来源与输入

从t=4开始，每有一帧 `t−4` 离开recent4，就进入memory；不足2帧时直接填入，此后父集严格为已有2帧+离开的1帧，每次删1，不可复活。

登记四种产生父集的行为策略：FIFO、reservoir、temporal buckets、feature diversity，规则沿用10/29在线框架。每种行为在每个 episode 完整不可逆回放；只在四个预定溢出时刻 `6+floor(j*(query−6)/3), j=0,1,2,3` 记录父集。四行为等权、四时刻等权；相同父集的重复记录保留权重，记录ID含episode/behavior/clock/pool，另报告去重数。train=1024×16=16384条，ordinary validation=192×16=3072条。reservoir以20261103、pair ID、策略名派生RNG，siblings共用。无可见性过滤、无按标签平衡、无事后挑状态。

学生输入沿用10/26：每候选 `[p, p−p_current, log1p(age), past_actions]` 共259维；公共context为recent4的128维投影、四个age与四个past_actions，共524维。过去动作只累计到当前clock。学生不接收 query未来信息、可见性、t_c、label、位置、充分帧标记或监督值。冻结预测器在实际查询时使用原Phase1的已给定未来动作接口；该接口不传给保留策略。

### 四种删除代价与对照

对每个父集的每个删除候选，分别构造：

| 标签 | 时间点 | 损失 |
|---|---|---|
| local_feature | 保留删后原两帧，直接在本episode真实query评价 | feature MSE |
| local_position | 同上 | 冻结decoder的XY position MSE |
| use_time_feature | 在父集clock强制删一张，以固定当前attention继续不可逆回放到query | feature MSE |
| use_time_position | 同上 | 冻结decoder的XY position MSE |

每项先算各 predictor seed 的 h3/6损失，再等权平均3个reader×2个horizon。标签是删后K2原始损失；无需向reader输入未训练的K3，也不减一个共同的K3 loss。保存未聚合的[reader,parent,drop,h]数组及use-time最终两帧，便于复核。

use-time续跑策略固定为三个reader当前候选attention概率的均值（各reader先对3候选softmax，再平均），保留分数最高两帧，tie保留较新帧。所有label共用这一续跑策略；这是本次三reader集成的预登记实现。local时间点按用户本轮定义为真实query，区别于旧Gate某些probe的clock+h局部目标。

future_attention监督：冻结reader的salience以真实 `query+h` 投影为anchor，对三个父集候选softmax，再平均3reader×2h。不是余弦proxy。删除目标是低attention候选。四种删除代价各有一项独立的shuffle对照，使用seed20261108+parent ID生成三候选排列；同parent四种shuffle共享排列，但每种分别置乱自己的代价值，不挑选一种shuffle。合计9个标签臂。

### 固定学生训练与选点

集合学生DeepSets与独立学生各9标签×3seed17/29/43，共54臂。沿用10/26网络：independent宽64、54401参数；DeepSets宽60、54361参数，phi一层ReLU，head一层ReLU再标量；集合模型在head前拼本候选phi与父集mean(phi)。无dropout，无scheduler。Adam lr=.001，weight_decay=0，batch128，有放回均匀采样，1000updates，clip1。每架构/seed在所有标签臂使用相同初始化和批次序列，跨架构参数形状不同，不声称逐权重相同。

输入均值/标准差只由train父集拟合，std floor1e−4。每parent代价减三候选均值，以train所有centered值的全局RMS归一化，floor1e−8，softmax(−centered/scale)作为删除分布；future_attention同样低分优先删除；shuffle复用对应原代价的train scale。以soft cross-entropy训练，每100步在固定ordinary validation父集上的本标签CE选择严格最小值，tie选更早；包含step0。每臂保存1000步loss/grad norm、11次validation CE、选中step、checkpoint哈希、初始化/批次/输入/监督哈希。选中step0明确标为未训练初始化，不当作用该标签训练成功，不重跑。

## 4. 在线validation比较和将来确认的主张

所有方法从空memory开始完整在线回放，以最终保留两帧+recent4输入冻结reader。主指标：两帧都为新段可见帧的成功率，以及h3/h6 feature/position MSE。reader损失先平均三个冻结seed；学生先各自在线回放其三个训练seed，再平均。reservoir用17/29/43三个RNG种子且siblings共用抽样，其余确定性baseline单跑并注明复制seed轴仅用于对齐，绝不伪造独立重复。

通用对手固定7项：FIFO、reservoir、temporal buckets、feature diversity、feature surprise、predictor attention、keep-first。temporal buckets的log2(age/4)桶数、年龄差及较新帧tie顺序，diversity的两候选相互距离加到当前anchor距离均沿用Gate规则。attention为上述当前3reader概率集成。surprise为帧到达时以三步前recent-only冻结reader预测的feature误差，未来动作以最后已知动作保持，平均3reader；最初无法预测的帧分数为−1（所有有效MSE≥0）。因此没有读取未来动作或状态来做在线选择。

非对手参照：最后两张可见历史帧（用隐藏可见性）、特权b、事后最优。事后最优穷举query时所有合法历史帧对，以3reader×h3/6平均position MSE最小为目标、lexicographic最早pair打破tie；它是离线位置损失上限参照，不是可部署策略或成功率oracle。

实现是离线archive加速的逻辑B2不可逆回放；部署scorer API只收到当前3候选及recent4，不能恢复已丢弃帧。报告不把这一实验称为物理内存只存R+B的系统实现。

普通间隔为主。四种删除代价均报告、均与全部7baseline和同架构future-attention比较，不依据结果选择标签。预登记主要候选为 **set/use_time_position**；其余三种为次要候选。每个候选在将来确认集要检验的联合主张是：成功率改善和两个horizon的position MSE改善，相对每个对手的配对95%下界全部>0。此处不做最终主张、不分配确认集seed。所有feature MSE对比也报告，但不加入该联合判据。主要候选三个seed均须选中step>0；否则明确未完成“该标签训练的学生”要求。

建议开确认集仅在普通validation上主要候选的上述联合条件全部满足、主要候选和同架构future-attention对手各三个seed均选中step>0、所有科学/运行审计通过时为“是”；其余情况为“否/当前证据不足”。若future-attention选中step0，只能称作初始化参照，其数值差值照报，不能声称胜过“用future-attention训练”的对手。次要候选即便表现更好也不替换主要候选。long-gap通过reader诊断则单独报告同一套结果，不与ordinary混合；long-gap reader失败时不能宣称长间隔有效。

独立学生同样完整报告。每种删除代价都比较set与independent的成功率及position/feature MSE。若主要候选任一所需差异区间含0，则写明本任务未能清晰区分集合条件作用；不把无显著差异解释成等价证明。“最近两张可见帧”本身是逐帧可解的规则。

## 5. 配对统计与证据

固定seed901、2000次pair bootstrap，双侧percentile95%区间；每次重采样96个pair，保留其两个siblings。先平均siblings，再平均固定的reader/student seeds，最后对pair抽样。同split全部方法、horizon、B与差值共享索引；普通与long-gap各独立报告，同seed不代表跨split配对。CI描述这96对上的不确定性，不是对训练seed总体的外推；多标签结果为预登记的validation探索，未经确认不作最终科学结论。

保存每pair损失/成功率、原始监督、抽样帧、轨迹、逐seed附表、训练选点、原生Slurm记录、status.json及SHA256。统计输出包含d−b、B−b、d−B和任意新段帧诊断；在线主表包含各学生/基线/参照×h3/6、所有登记对比和独立/集合差值。缺失结果标未评估，不补训或选择性省略失败臂。

## 6. GPU估时、执行顺序与停止

总上限14400 A100秒；旧16550=769、16614=2936，累计3705，剩10695。单A100新工作预计8500秒（2.3611h），登记作业上限10500秒（2h55m）；3705+10500=14205≤14400。

| 阶段 | 预计秒 | 依据/覆盖 |
|---|---:|---|
| 初始载入与完整哈希/环境核验 | 450 | 原特征、权重和缓存只读校验 |
| 新编码与两组reader诊断 | 900 | 新数据上界12480帧；16614编码39398帧用1325.8秒，约29.7帧/s，保守预留模型载入和诊断 |
| 父集和全部标签 | 1000 | 19456父集×3删除×3reader×2时间点=350208 sample-reader前向，另含固定attention续跑 |
| 54学生训练及选点 | 3400 | 54000 updates，每步保守约.06秒，另留validation及工件写入 |
| 在线两种间隔及穷举参照 | 2300 | 全部54学生轨迹、7baseline、参照及全部reader损失，long-gap未通过时会少做 |
| 末尾只读复验与打包/写入 | 450 | 全部资产复验与SHA256 |
| 合计 | 8500 | 不是实测保证；超出预算预测即停 |

这里学生/在线新计算无事后microbenchmark调参；估时不够就停止报告。BudgetGuard保留“实际已耗时达到上限，或已耗时+剩余预测超过上限即停”的判定，按本次阶段估算剩余。阶段完成10%后才用实测速率外推，以免I-JEPA载入时间除以第一个batch造成假超限；此前保留登记的该阶段估计。30秒间隔复核原生allocation身份/Restarts/GPU，初始sacct仅允许原登记120秒短暂传播延迟。累计秒最终取top-level sacct ElapsedRaw×GPU数，不重复累加.batch/.extern。

进度按预定工作量逐chunk增加，不能在耗时操作中卡住一个百分比：标签阶段每parent的行为记录1单位、三reader×两h salience及续跑预处理6单位、18个sample-reader损失前向18单位，合计25×19456。在线阶段每student溢出选择60单位、其余帧到达1单位，每个reader sample（含surprise和穷举）20单位；这些权重是运行前的成本近似，不是指标或样本权重。总单位仅由manifest的query长度、预登记方法/seed数及是否获得long-gap资格计算，逐前向/选择累加并核对总数。最终SHA过程中仍检查预算，最后一次检查后才冻结会追加的原生日志哈希；原生终态记账包含所有尾部I/O。

执行顺序：推送本登记→软件单测/真实原生记录回放→CPU新validation生成审计→冻结CPU产物和执行回执再推送→新部署浅克隆并核对HEAD→只读原资产/venv预检→单个A100 sbatch→立即保存scontrol→每5–10分钟保存原生状态→终态结算→原始结果核验/报告/更新PR→停止第4步。

沿用v2.3分级规则：源码/protocol/数据/模型revision哈希改变、输入或pair审计失败、GPU型号/数量不符、Restarts>0、超预算、NaN/缺失、输出目录已存在、科学阶段错误均阻塞；需要本人认证立即停并通知。辅助CPU/OOD会话Requeue/CPU分配高于请求只记录；科学作业Requeue=1记录但不阻塞，wrapper仍请求--no-requeue。不得临时安装/换版本、改科学参数/阈值、看到validation后重跑。

本冻结作业最多3次基础设施条件提交，已知sbatch回执或未知提交结果不能重复执行同一attempt。每次重提前原生终态与四项证据必须全部满足：无.batch、stdout/stderr均不存在、scientific_job_started目录不存在、科学输出目录不存在，且Restarts=0、累计真实A100秒+本次10500≤14400。任何marker存在即不再重提。所有attempt现场、日志和回执保留；不删除文件、不通过缩小预算/科学规模来绕过停止。正常排队/运行无实质变化保持安静；结束/失败/本人认证才通知。

这是运行前承诺。软件测试若在科学入口前发现实现错误，保留失败回执，只修实现并重新冻结源码、提交推送后再测试；不得借此改变科学参数、标签选择或判据。若真实数据/科学检查不过，按上述规则停止，不用修实现名义重复实验。
