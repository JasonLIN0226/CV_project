# 夜间实验结果报告（持续更新）

截至纽约时间 2026-10-13 01:00 EDT：Gate 已完成357/360轮（约99%），前两个种子的实际 best/latest 独立核验通过，第三个继续。当前仍无新版候选的完整科学评估结论，Phase1未通过。半小时监控继续，08:00前汇总真实结果；未完成或失败也如实报告。

| 项目 | 实际状态 |
|---|---|
| Gate seed17 | 120/120；best57，等权验证MSE 0.06682199；实际有限值、RNG、优化器/调度、7680步及最早best证明通过 |
| Gate seed29 | 120/120；best4，等权验证MSE 0.10290398；同样的实际完成证明通过 |
| Gate seed43 | 117/120；当前best66，等权验证MSE 0.06818651；不是最终选模或测试结果 |
| Gate完整评估6994 | PENDING/afterok:6990，尚未执行 |
| Motion训练7000 | PENDING/afterok:6994，尚未训练；三个种子各120轮、原start/end证据及全部原规则 |
| Motion完整评估7001 | PENDING/afterok:7000，尚未执行；全部权重完成证明后才进入测试 |
| Assembly101 | 78段开发RGB和PTS/标签可用性核验完成；仍无实际特征cache或真实reader训练结果 |

所有科学任务依次执行：6990 → 6994 → 7000 → 7001 → 7004（L4编码） → 7005（CPU缓存核验）。当前只有6990运行；五个后续任务只是依赖队列，没有重复或并发科学训练。每项训练仍使用课程CPU4/16G/4小时，评估CPU4/16G/2小时。stderr为空，四套实际冻结源锁均重验一致。

两项候选沿用原数据、缓存、128维train-only基底、B2/recent4/h3,h6、完整配对四条件、MSE+2pair+.5rank和原优化器/调度；唯一变化为已声明的同一通用因果Transformer。Motion不采用旧wide-span或pure-velocity方法，没有重拟合基底、改变目标或阈值。Gate即使失败，也不能由Motion或真实辅助结果补救。

Motion的18项本地和HPC软件检查已通过（数据无关），包含原两帧选择、真正Motion参考绑定、控制前缀、最早best/不完整权重拒绝、原指标和单种子失败不能被pooled平均掩盖。六个原Motion参考best/latest及原34560行结果来源已核验。静态差异检查发现一次尚未转换的参考字典键，已在任何部署/试验前修复；默认本机Python无pytest，改用现有Anaconda测试。新科学训练没有受影响，也没有以软件测试冒充实验成功。

两项完整评估均保留24项固定参考recent与8项关键绝对CI检查（Gate干扰变化、Motion长间隔），以及全部原任务/OOD/逐seed/错误证据方向和真实train-only读出规则。所有6个新best/latest完成证明先于各自评估，加载器明确使用新类及真实签名，不冒用旧DynamicsPredictor哈希。验证误差下降不能替代采用门槛。

可复核文件：

- `monitor_20261013T0426.json`：实际04:25:50 UTC捕获；SHA 1f05bb1ced96bf13d501f50d659e1796ec639d4436f7317a14da8ce7b48ec6ff。
- `../phase1_v5_motion/submission_7000.json`：SHA cd11eaebb79b8b948b4b91a19e3faeb2ce00d943720fd6d37039e047deaa1e22。
- `../phase1_v5_motion/evaluation_submission_7001.json`：SHA 61be773f0e89aa018c48c169b5a719e1d7e9d2b061e62dbe8a9f5c7678e2d4d2。
- 上述紧凑记录传输前/本地/传输后SHA均一致；权重、数组、视频继续留在HPC。

所有原v3/v4失败和GPU启动失败保留。6965/6980未进入编码即被UID0取消，原因未知；6985未起batch后按DDL指令由agent取消。真实特征提取仍需可用的一GPU路径和实际cache核验；30fps标签与60fps原视频抽帧相位仍未证明，不能猜测做label/frame评估。封存18段RGB和fresh9137确认未打开，无Phase1GO。

最新编码路径：一张课程L4任务7004已提交并等待afterok:7001；4CPU/32G/4小时，实际编码未开始。课程g2-standard-12当前UP、每节点L4×1/47000MB，因此原60G主机请求改为32G；全部50,688时点、冻结官方IJEPA版本、float32/TF32off/grid2/batch2、PTS和首512帧预算检查保持。原A100准备13文件逐字节保留，新路径只换独立资源保护/源/输出版本，不保证L4节点能启动或耗时过关。20项HPC软件检查通过，不是实际GPU推理。单次L4分配尝试，不盲重试、不增GPU或降低数据/精度。

CPU缓存核验7005已提交，afterok:7004；零GPU/4CPU/16G/30分钟。10项人工缓存软件测试通过；必须实际7004COMPLETED0:0、全部78特征SHA/大小/float32/形状/有限值、全部PTS摘要、实际编码源/官方权重/处理器绑定成立，才发布完成核验。不会拟合基底或训练模型。真实reader的L4软件15项本地检查通过，但没有新正式训练协议、HPC部署或提交；7004缓存仍不存在，必须先核验实际cache/audit SHA，再绑定新试验。旧6980训练协议未被冒充有效。

新增可复核文件：`../phase1_v5_encoding_l4/submission_7004.json`（SHA5817bdc57f9f0f9047e438f9103cfae7f2394eb7c26b4f3ffbc4764c783a480b）、`cache_audit_submission_7005.json`（SHAa0b223982200afefc822b00a37548300aa461c0bc0c0c9af689a4cbce5d1fcb6）、`monitor_20261013T0459.json`（SHA17947a8020d9fd9ed2c1a501f73322f3f3dc7ffe48db3da09cb16e4b28fd2b66），前/本地/后SHA一致。编码静态同源检查曾因测试AST索引漏计前置import而失败，已在部署前修复；生产编码流程、已有训练和数据没有受影响。
