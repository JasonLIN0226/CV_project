# Assembly101 真实 GPU 实验报告（已运行结果与未完成项）

更新时间：2026-10-13 20:10 EDT；诊断实际结束2026-10-13 23:53:11UTC，最终原生结果与数值核验已取回。

**三组 GPU reader训练全部完成，各30轮9690步；所有6当前模型通过正式推理前与结束后的数值/来源核验。完整固定视觉评估A/B/C也已结束：健康6/8、删除47/48，均为REJECTED；不可逆重放完整执行并通过轨迹核验。动作语义评估仍缺相位证明，必须列为未完成。最终结论是本轮真实可行性REJECTED，Phase1未通过。**

## 实际执行与资源

| 阶段 | 作业 | 实际状态 | 资源/结果 |
|---|---:|---|---|
| 原Gate训练/评估 | 6990/6994 | COMPLETED0:0；科学REJECTED | 三组各120轮；seed29原收益/内容/固定参考失败 |
| 原Motion训练/评估 | 7000/7001 | COMPLETED0:0；科学REJECTED | 三组各120轮；收益10/24、固定参考11/24、长gap绝对CI0/8 |
| 真实视频编码 | 7187 | COMPLETED0:0；20:31:31–22:28:43UTC | 单A100；78录像、198episodes、50688时点全部特征完成 |
| 全缓存审计 | 7188 | COMPLETED0:0；22:29:36–22:30:27UTC | 0GPU；全78文件和PTS/source/officialweights/processor证明 |
| GPU训练交接 | 7207 | COMPLETED0:0 | 只元数据绑定；自动提交7209 |
| 三组真实 reader训练 | 7209 | COMPLETED0:0；22:33:06–23:23:15UTC | 单A100/请求4CPU60G4h；实际50分09秒；各30轮9690步 |
| 全6模型证明/评估交接 | 7223 | COMPLETED0:0；23:25:06–23:25:30UTC | 只完成/来源数值证明；自动提交唯一7226 |
| 固定模型正式诊断 | 7226 | COMPLETED0:0；23:26:36–23:53:11UTC | 原定0GPU/4科学线程16G2h；A/B/C全部视觉完成，科学REJECTED |
| 动作NLL/活动内漂移/切换/供体恢复 | — | UNAVAILABLE | RGB60fps与标签30fps相位未证明；没有猜测连接或制造供体 |

CPU模型编码/训练替代未执行。7226是原设计对同一GPU训练好的冻结模型进行评估；没有新的拟合或训练。各阶段保持一次一个科学作业；现在全部结束。未追加时间或降低样本/种子/删除分支/延续长度。

## 数据、模型与信息边界

固定78开发RGB（58train/20validation）、198episodes、50688时点；训练136episodes/34816观察/5168查询，验证62episodes/15872观察/2356查询。官方冻结I-JEPA facebook/ijepa_vith14_1k@f157467ea509bc356ff9f61fd3c0d840eec5e04e，5120D、float32、grid2、编码batch2、TF32关闭。128rank归一化和输入/输出基底只拟合34816训练观察和20672训练未来残差，validation/test fit rows=0。

Reader为256hidden、2层4heads、FFN1024、dropout0.1，1779072参数；recent8、capacity16、h1/4/16/64、recent/FIFO/Reservoir/full共同训练；seeds17/29/43各30epoch，AdamW3e-4/wd0.05/warmup5%cosine和historydropout0.15保持。三个最终best分别5/6/7，由原等权validation MSE的最早最小值选出。所有6best/latest、12实际文件SHA、配置/fit/cache/source/RNG/optimizer/scheduler/counts/最早best在任何正式heldout推理前已证明；root再次读取当前实际模型得到完全相同的完成证明。

封存18RGB/9人和fresh9137确认没有打开。Gate/Motion原失败仍保留，当前不允许独立确认。

## GPU训练结果与正式评估的区别

| seed | 实际轮数/步数 | 最早 best epoch | best validation MSE | 第30轮 validation MSE |
|---|---:|---:|---:|---:|
| 17 | 30 / 9690 | 5 | 0.34036500 | 0.39486956 |
| 29 | 30 / 9690 | 6 | 0.33975386 | 0.39401980 |
| 43 | 30 / 9690 | 7 | 0.34271761 | 0.39865064 |


训练MSE在继续下降时，三组validation均在前5–7轮达到最小值、随后升高；这是实际观察到的泛化退化表现。该曲线本身不证明退化原因，也不能据此事后更换训练配置/种子或checkpoint规则。正式评估使用原规则的best，不能以训练完成或validation下降冒充研究通过。

![三组GPU训练与实际健康门槛](/Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/training7209_and_health7226.png)

## A：完整真实健康评估

全部3seeds×2356验证查询×4conditions×4h共113088标量记录，原完整5120D归一化MSE、persistence及fixed-basis truefuture oracle均已计算。统计按query→episode→recording→participant等权，验证10参与者，10000次seed0 bootstrap、双侧95%CI；不能把帧或种子当独立参与者。

主门槛要求h16/h64的每个种子和参与者汇总区间都支持历史收益。**h16仅17和29通过，43与pooled区间跨零；h64全部通过。必要主检查6/8，故A为REJECTED。** h1/h4各0/4正区间，短期结果照实保留，不改主门槛。正数表示full比recent误差更小；CI跨零表示尚不能证明稳定改善。

| h | seed/汇总 | Recent−Full MSE | 95% CI | 原正区间检查 |
|---|---|---:|---|---|
| 1 | 17 | 0.00012829 | [-0.00026164, 0.00067533] | FAIL |
| 1 | 29 | -0.00154260 | [-0.00215166, -0.00100047] | FAIL |
| 1 | 43 | 0.00027636 | [-0.00090603, 0.00201529] | FAIL |
| 1 | pooled | -0.00037932 | [-0.00074084, 0.00009580] | FAIL |
| 4 | 17 | 0.00072163 | [-0.00055343, 0.00236869] | FAIL |
| 4 | 29 | -0.00020210 | [-0.00123234, 0.00078616] | FAIL |
| 4 | 43 | -0.00054848 | [-0.00177244, 0.00081045] | FAIL |
| 4 | pooled | -0.00000965 | [-0.00085269, 0.00103500] | FAIL |
| 16 | 17 | 0.00354113 | [0.00196484, 0.00529251] | PASS |
| 16 | 29 | 0.00156920 | [0.00005162, 0.00325420] | PASS |
| 16 | 43 | -0.00171391 | [-0.00787683, 0.00217355] | FAIL |
| 16 | pooled | 0.00113214 | [-0.00066553, 0.00232555] | FAIL |
| 64 | 17 | 0.00366903 | [0.00061982, 0.00679148] | PASS |
| 64 | 29 | 0.00262572 | [0.00008362, 0.00531769] | PASS |
| 64 | 43 | 0.00275554 | [0.00011104, 0.00586580] | PASS |
| 64 | pooled | 0.00301676 | [0.00184947, 0.00461047] | PASS |


按参与者等权的绝对MSE如下；与训练阶段按查询等权选模的validation MSE统计权重不同，不能直接相减比较。

| seed | h | recent | FIFO16 | Reservoir16 | full | persistence | fixed-basis oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 0.1621551 | 0.1617668 | 0.1618437 | 0.1620268 | 0.1643919 | 0.0511759 |
| 17 | 4 | 0.2957818 | 0.2941320 | 0.2943199 | 0.2950601 | 0.3176215 | 0.0793470 |
| 17 | 16 | 0.4049908 | 0.4004334 | 0.4016569 | 0.4014497 | 0.4452227 | 0.1024793 |
| 17 | 64 | 0.4839566 | 0.4782793 | 0.4796274 | 0.4802876 | 0.5427452 | 0.1209284 |
| 29 | 1 | 0.1639740 | 0.1652169 | 0.1641765 | 0.1655166 | 0.1643919 | 0.0511759 |
| 29 | 4 | 0.2976131 | 0.2974957 | 0.2971052 | 0.2978152 | 0.3176215 | 0.0793470 |
| 29 | 16 | 0.4003741 | 0.3983236 | 0.3988332 | 0.3988049 | 0.4452227 | 0.1024793 |
| 29 | 64 | 0.4772160 | 0.4742187 | 0.4748573 | 0.4745903 | 0.5427452 | 0.1209284 |
| 43 | 1 | 0.1633840 | 0.1632712 | 0.1633082 | 0.1631076 | 0.1643919 | 0.0511759 |
| 43 | 4 | 0.2993856 | 0.2993103 | 0.2990314 | 0.2999340 | 0.3176215 | 0.0793470 |
| 43 | 16 | 0.4064635 | 0.4035076 | 0.4045997 | 0.4081774 | 0.4452227 | 0.1024793 |
| 43 | 64 | 0.4823902 | 0.4782571 | 0.4797840 | 0.4796346 | 0.5427452 | 0.1209284 |


fixed-basis oracle读取真实未来，仅用作已批准诊断下界，不能部署或证明可学习性。根数值审计直接从113088原始标量独立重放分层均值和固定bootstrap，不调用项目stats；与执行摘要最大差 4.2500725161431774e-17，全部16判断一致。这是根数值复核，未新委托独立review，也没有新模型推理/fit/train。17823061字节原始CSV留在HPC，SHA d158c869a132f1db35fbdc2491802c6899a4770dfc13a0338c0facca30eeed2c，前后稳定。

## B：全部17删除分支与32共同延续

全部78录像×FIFO/Reservoir来源×每来源4预先固定均匀状态，共624状态；每状态包含全部3模型×17删除选择×32共同Reservoir后续。完成31824候选标量行，完整继续路径与所有候选均保留，没有缩减或另开补充作业。共同oracle按.5即时4h均值+.5未来32×4h均值选一次，精确平局取最早实际时间，各h16/h64使用同一选择。

**原联合/长h收益门槛为REJECTED：47/48通过，唯一失败为Reservoir来源→对照FIFO、seed29、h16，收益0.000136057，95%CI [-0.000004872984, 0.000267131094]，跨零。** 不能按h16重新挑另一删除条目、丢弃该seed或用汇总救这一项。下表列全部原48检查；每个单元为baseline MSE−同一个joint-oracle MSE，列顺序joint/h16/h64。

| 来源/基线 | seed | joint收益 [95% CI] | h16收益 [95% CI] | h64收益 [95% CI] | 通过 |
|---|---|---|---|---|---:|
| FIFO_FIFO | 17 | 0.000118703 [0.000087890, 0.000155011] | 0.000118539 [0.000075194, 0.000167264] | 0.000220343 [0.000159613, 0.000307022] | 3/3 |
| FIFO_FIFO | 29 | 0.000301081 [0.000200384, 0.000420960] | 0.000356171 [0.000244673, 0.000481657] | 0.000422051 [0.000281037, 0.000616695] | 3/3 |
| FIFO_FIFO | 43 | 0.000192843 [0.000116352, 0.000297053] | 0.000364336 [0.000150842, 0.000690047] | 0.000253847 [0.000152615, 0.000375300] | 3/3 |
| FIFO_FIFO | pooled | 0.000204209 [0.000142871, 0.000286750] | 0.000279682 [0.000171139, 0.000433889] | 0.000298747 [0.000210023, 0.000422651] | 3/3 |
| FIFO_Reservoir | 17 | 0.000133789 [0.000108686, 0.000165971] | 0.000146684 [0.000095229, 0.000199221] | 0.000266000 [0.000198604, 0.000346643] | 3/3 |
| FIFO_Reservoir | 29 | 0.000241051 [0.000168747, 0.000323219] | 0.000346806 [0.000183180, 0.000537520] | 0.000366742 [0.000240608, 0.000490192] | 3/3 |
| FIFO_Reservoir | 43 | 0.000275921 [0.000175688, 0.000402635] | 0.000549632 [0.000275680, 0.000885176] | 0.000313565 [0.000191037, 0.000459239] | 3/3 |
| FIFO_Reservoir | pooled | 0.000216920 [0.000167664, 0.000277589] | 0.000347707 [0.000206911, 0.000494519] | 0.000315436 [0.000242096, 0.000397792] | 3/3 |
| Reservoir_FIFO | 17 | 0.000107328 [0.000067878, 0.000147856] | 0.000127638 [0.000049530, 0.000207138] | 0.000111828 [0.000052697, 0.000175146] | 3/3 |
| Reservoir_FIFO | 29 | 0.000264002 [0.000186171, 0.000338739] | 0.000136057 [-0.000004873, 0.000267131] | 0.000579480 [0.000321856, 0.000925580] | 2/3 |
| Reservoir_FIFO | 43 | 0.000131896 [0.000078106, 0.000190152] | 0.000162048 [0.000090212, 0.000246834] | 0.000116638 [0.000003100, 0.000230732] | 3/3 |
| Reservoir_FIFO | pooled | 0.000167742 [0.000127465, 0.000208464] | 0.000141914 [0.000071818, 0.000211353] | 0.000269315 [0.000178794, 0.000377779] | 3/3 |
| Reservoir_Reservoir | 17 | 0.000265202 [0.000162900, 0.000390864] | 0.000385373 [0.000242423, 0.000580408] | 0.000428618 [0.000192706, 0.000738797] | 3/3 |
| Reservoir_Reservoir | 29 | 0.000318073 [0.000226785, 0.000417217] | 0.000298920 [0.000142450, 0.000476094] | 0.000779965 [0.000489830, 0.001144672] | 3/3 |
| Reservoir_Reservoir | 43 | 0.000254070 [0.000155844, 0.000383155] | 0.000362246 [0.000219439, 0.000542100] | 0.000452797 [0.000280800, 0.000703993] | 3/3 |
| Reservoir_Reservoir | pooled | 0.000279115 [0.000208458, 0.000366464] | 0.000348846 [0.000225345, 0.000502110] | 0.000553794 [0.000402516, 0.000749308] | 3/3 |


oracle使用真实未来，仅为删除收益上界。47个正区间也不证明存在可部署或可学习的删除策略，不能用于改变已经失败的健康A。首8完整all3状态预算投影1318.41秒<6900，实际整套A/B/C程序1585.04秒、Slurm26分35秒，未扩预算。

## C：不可逆固定策略重放完整完成

全部198episodes、3冻结模型、FIFO/Reservoir/Uniform，容量16主与8/32固定次级检查、每episode同38queries×4h。完成812592标量和456192条共享保留轨迹（1782条trajectory，每条256步）。root独立按固定哈希RNG种子重算每一次FIFO、Algorithm-R Reservoir和Uniform的入队/删除，所有trace完全一致；时间顺序、recent8、容量上限、持久载荷及已删内容不回流均通过。该核验没有读取特征作新推理、训练或拟合。

下表是验证参与者等权的绝对归一化MSE，仅描述预测误差。它不构成动作语义、恢复或learned selector成功；不按结果选择容量或策略。

| seed | h | FIFO16 | Reservoir16 | Uniform16 |
|---|---:|---:|---:|---:|
| 17 | 1 | 0.16176682 | 0.16184366 | 0.16177537 |
| 17 | 4 | 0.29413200 | 0.29431989 | 0.29409510 |
| 17 | 16 | 0.40043341 | 0.40165690 | 0.40060069 |
| 17 | 64 | 0.47827931 | 0.47962739 | 0.47842003 |
| 29 | 1 | 0.16521692 | 0.16417650 | 0.16490003 |
| 29 | 4 | 0.29749573 | 0.29710517 | 0.29736161 |
| 29 | 16 | 0.39832359 | 0.39883316 | 0.39828481 |
| 29 | 64 | 0.47421868 | 0.47485731 | 0.47415908 |
| 43 | 1 | 0.16327124 | 0.16330824 | 0.16327353 |
| 43 | 4 | 0.29931031 | 0.29903138 | 0.29910714 |
| 43 | 16 | 0.40350755 | 0.40459968 | 0.40347639 |
| 43 | 64 | 0.47825711 | 0.47978405 | 0.47831315 |


固定容量8/32的次级描述结果如下，按三seed等权汇总。完整全部3seed×2split×3policy×3capacity×4h的216组数值见 [全部重放组MSE表](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_final7226/replay_MSE_all_groups.csv>)；每个参与者的原层级均值见 [实际重放参与者摘要](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_final7226/replay_participant_summary.json>)。

| 固定次级容量 | h | FIFO | Reservoir | Uniform |
|---|---:|---:|---:|---:|
| 8 | 1 | 0.16327162 | 0.16307143 | 0.16318434 |
| 8 | 4 | 0.29689054 | 0.29695821 | 0.29679384 |
| 8 | 16 | 0.40096304 | 0.40226753 | 0.40105793 |
| 8 | 64 | 0.47729915 | 0.47897395 | 0.47735216 |
| 32 | 1 | 0.16352622 | 0.16324808 | 0.16343508 |
| 32 | 4 | 0.29721001 | 0.29684744 | 0.29706114 |
| 32 | 16 | 0.40144901 | 0.40144974 | 0.40109863 |
| 32 | 64 | 0.47703024 | 0.47747604 | 0.47696385 |


![全部删除/延续与不可逆重放结果](/Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/deletion_and_replay7226.png)

## 缺失的语义评估与整体结论

24verb readout/NLL、within-activity drift、switch response、different-activity donor在0/1/2/4/8/16步恢复仍未可用。实际原RGB是60fps、标签约定是30fps抽帧索引，两者抽帧相位没有充分来源证明；不能猜+1/奇偶帧/index÷30的对应，不能制造活动供体或以无分层视觉稳定性代替这些必要任务指标。没有用这些标签重新拟合读出器，也没有打开封存数据。

**本轮可执行的GPU训练、全部固定视觉A/B/C及数值来源核验已经结束，结果为REJECTED；整体真实可行性不成立，Phase1未通过。语义组件列为INCOMPLETE/UNAVAILABLE，未假称全部原计划指标已经完成。** A必要6/8、B47/48仍各有失败；C完整执行不改变前两项失败，也没有语义层面的支持证据。这里的CI与模型选模使用同一开发validation参与者，属于开发诊断，不能当作fresh独立确认。原Gate/Motion失败和原规则保持，封存18RGB/9人及fresh9137确认未打开。

## 来源、实际证据与失败分类

原L4节点7004重启72次且未boot，没有batch/cache，7175也未boot；云控制器原因/演员未知。它们是启动失败。A1007179完成512帧但固定投影20056.84秒>13800主动停止，是预算失败。7184逐字节证明512clock/PTS/RGB相等后，7187仅改SLICE→FRAME解码调度，首512特征SHA与7179完全相同，完整编码成功。7209和7226的健康失败均没有启动/断线/OOM/nonfinite证据；A失败属于原科学门槛。

根监控重新核验14编码、10缓存审计、14GPU训练、15评估交接、14正式评估冻结文件，共5组；全78实际缓存当前大小及前/后SHA稳定，fitNPZ字节与训练证明一致。29份阶段compact JSON/logs/训练标量CSV及10份最终compact JSON/日志的原生传输前/本地/后SHA一致。权重、特征、视频和标签载荷没有取回Mac。

- [实际训练报告](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/training_report7209.json>)；[全部六模型执行证明](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/all_models_completion.json>)。
- [根当前模型/源/78缓存复核](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/root_monitor_all6_health_20261013T2330.json>)。
- [正式模型hash绑定评估协议](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/diagnostic_protocol7226.json>)；[实际自动提交7226](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/diagnostic_submission7226.json>)。
- [健康评估执行摘要](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/visual_health.json>)；[独立标量重放/所有条件绝对MSE](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/root_health_scalar_replay_20261013T2330.json>)。
- [最新实际Slurm/删除进度](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/root_progress_20261013T2334.json>)；[29份compact三SHA](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_completed7209_diagnostic7226/transfer_integrity.json>)。
- [既有标注相位来源审计](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_data/handformer_annotation_phase_source_audit_verified.json>)。

## 最终核验与资源释放

7226实际COMPLETED0:0，23:26:36–23:53:11UTC；科学REJECTED按协议正常退出，stderr0。root再次证明全部6当前模型、5组冻结源、78缓存和fit字节稳定，执行报告SHA ea9fb1c5254ee3bbbd8009bc09ced1650ee7b2b56b2d756a5f8575d6fb2b550f。根完成复核SHA 0e936849ff6158fb8212fadd6c1c89d92cd91e989698cd898da09975ee33254d，B的48组CI和C全部参与者均值与执行结果最大差均0；所有456192保留状态/删除记录按独立固定RNG重放一致。A此前独立原标量bootstrap最大差4.2500725161431774e-17。数值复核没有新模型推理、拟合、训练或独立委托review。

原始标量与大轨迹均留HPC，以下是完整实际大小、行数与字节SHA；Mac只保留compact摘要/证明/小训练CSV/图。

| HPC原始文件 | 行数 | 大小（字节） | SHA256 |
|---|---:|---:|---|
| health_scalar_records.csv | 113088 | 17823061 | d158c869a132f1db35fbdc2491802c6899a4770dfc13a0338c0facca30eeed2c |
| deletion_candidate_records.csv | 31824 | 11494001 | 6677b259d9a0a688cf236bbf5ea0ae7d1dcc1f22b2fe15312a0df3825bc0de7f |
| replay_scalar_records.csv | 812592 | 137128590 | 00ef34029d5b99070f9204fda314991d8a4137ac67c97e587494e5a40ad3bd0a |
| irreversible_replay_trace.csv | 456192 | 89242659 | c8b47e2d7d3b9e236eba5a8f5634df818b70604c046ce19dbdecf42fcf7a394b |


[最终执行报告](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_final7226/diagnostic_report.json>)；[根最终Slurm/模型/所有源缓存与原标量轨迹核验](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_final7226/root_final_audit_7226_20261014T0000.json>)；[最终10文件原生三SHA](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/native_final7226/transfer_integrity.json>)。

因为7226结束后没有有效路由，仅申请一次0GPU/1CPU/1500M/30分钟短连接7233取证。新b-9-9/IP92正常Jupyter公开公钥与Torch native keyscan逐字节匹配后独立严格pin；实际课程/节点/资源核验通过，所有科学与模型证明均在课程Cloud。取回完成后从同一有效会话发送释放请求，连接在取消过程中Broken pipe退出255，没有重连已结束92。正常门户随后证明7233 CANCELLED/Completed、23:57:06–00:04:17UTC/7分11秒；最终native sacct取消演员/退出码未返回，不冒称已读。现无活动科学或临时资源、无新人工认证依赖，也没有发支持邮件或运行CPU模型替代。

[临时取证释放事实](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/control7233_release_verified.json>)。

![取证连接释放的正常门户详情](/Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v5_real_evaluation/control7233_cancelled_detail.png)

本轮作业与可执行视觉结果收集已经结束。缺失的语义来源证明及未成立的科学门槛不通过盲重试、事后改模型/种子/门槛或新确认数据解决；当前试验停止在REJECTED，保留全部实际结果和明确的未完成项。

本轮所有作业、结果取回和核验均结束，cv-hpc半小时监控已删除；不会继续轮询已结束作业。缺失语义项与未成立门槛仍按上文保留。
