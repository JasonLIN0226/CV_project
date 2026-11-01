# Motion v2.3.1 有限重提最终报告

日期：2026-11-01（UTC）。**第 2 步通过，并已按协议停止。** 作业 **16614** 完成冻结实验，`runner status=completed_step2_stop`、`step2_gate=passed`。d−b、c−b、B−b 在 h=3/6 的六项 position MSE 改善区间下界全部严格大于 0；其中 B−b 下界分别为 **0.00270860694、0.00602382346**。这表明在当前 validation 与固定三个 seed 上，预测器能利用特权提供的正确新段帧，并超过共同预测的方向下界。

成功作业实际占用 **2936 A100 秒**；加上入口前失败的 16550 的 **769 秒**，累计 **3705 秒（1.029167 A100 小时）**。没有第三次提交、没有 test 评估、没有保留策略训练，也未进入第 3 步。后续工作等待用户确认。

本报告直接整理已校验的 [metrics.json](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/metrics/metrics.json)、[runner status](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/status.json)和[原生结算](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/settlement/accounting.json)，没有重新计算指标、bootstrap 或训练。表格显示 9 位有效数字，完整精度以原 JSON 为准。独立[结果审计](../results/motion_v2/v2_3_1_resubmission/result_audit.json) **116/116 通过，failures 为空**。

## 第 0 步和第 1 步的已保存结果

Phase 1 `motion_headroom.py` 重建的完整 JSON 与附件一致；三个原测试条件的 reject-newest 信息最优率均为 100%，含至少两张可见帧的池比例分别为 60.87%、18.42%、62.63%。这些是公开代码重建结果，**尚未与原 seed 9137 manifest 逐 pair 核对**，没有读取原 test 图像或生成新 test 数据集。

原始 `/scratch/xl4358` 属于另一账号；仍需 xl4358 将两个 manifest 和 `confirmation_test_generation_complete.json` 复制出来，才可完成历史逐对核对。此次没有尝试登录该账号，该限制不阻塞新的 v2.1 train/validation 实验。

v2.1 保存结果按先行登记的 v2.2 规则重判通过：v1 单张非终点新段帧为 0.620，对照 0.624；v2.1 为 0.545，对照 0.538。两张新段帧分别为 0.996、0.993，与参考一致；其余要求精确一致的配对与在线结果均一致。本次没有重跑这些 CPU 测量。

实际生成 seed 20261030 的 train **512 对**、validation **96 对**，共 1216 episodes、39398 帧。结构审计 **608/608**、**7 项 pytest** 通过。

| 真实 train 隐藏状态输入 | 实际准确率 | 第 1 步判定 |
|---|---:|---|
| 终点帧 | 0.5000000000 | 通过 |
| 单张非终点新段帧 | 0.5253188654 | 通过 |
| 旧段帧 + 终点帧 | 0.5000000000 | 通过 |
| 两张新段帧 | 0.9756520084 | 通过 |

train 与 validation 的 FIFO、keep-first 充分帧保留成功率均为 0%，事后充分帧上限及最新两张可见帧结构参照均为 100%。这些结构检查不等同于预测器精度。来源：[v2.2 报告](MOTION_V2_2_REPORT_ZH.md)、[第 1 步 audit](../results/motion_v2/v2_2/step1/audit.json)及[状态](../results/motion_v2/v2_2/step1/status.json)。

## 16550 的基础设施诊断与有限重提

首个 A100 作业 16550 原终态为 `CANCELLED by 0`，计费 769 秒。后续只读诊断收集了 `sacct --duplicates` 的 120 个字段及全部可查询作业步，未加 `-X`；作业步列表为空。

| 入口前失败条件 | 16550 诊断结果 |
|---|---|
| 完整 `sacct --duplicates` 无 `.batch` 步 | proven |
| 该作业 stdout、stderr 两文件不存在 | proven |
| `scientific_job_started` 目录不存在 | proven |
| 原科学输出目录不存在 | proven |

节点 `b-19-12` 原生记录为 `State=DOWN+CLOUD+POWERED_DOWN`，`Reason=16550 finishing : ResumeTimeout reached [slurm@2026-10-31T23:47:14]`；配置查询返回 `ResumeTimeout=6000 sec`、`ResumeProgram=/bin/true`。这些证据支持与 16550 相关的节点恢复超时诊断和科学入口前基础设施失败分类，但不能独自定位云侧具体故障，也不能把 6000 秒配置值当作实际分配用时。

旧 scontrol 记录后来被清理，唯一非阻塞诊断错误为 `Invalid job id specified`。新 helper 仅在此明确错误下使用哈希核验过的原终态 scontrol，且要求新 sacct 与原 allocation row 逐字段一致，从而保留有证据的 `Restarts=0`。没有从记录缺失推断重启次数。来源：[诊断回执](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/diagnosis_16550/receipt.json)、[诊断状态](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/diagnosis_16550/status.json)与[SHA256SUMS](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/diagnosis_16550/SHA256SUMS)。旧 16550 报告和原记录保留，新增证据另行衔接。

用户授权同一冻结版本的 attempt2。新增运行层软件检查 **158/158 通过**；原 **98/98** Slurm 回放保持不变。运行层代码只管理诊断、独立回执、累计预算和一次提交，没有改变科学 wrapper。来源：[有限重提登记](MOTION_V2_3_1_RESUBMISSION_ZH.md)、[运行 protocol](../experiments/motion_v2/protocol_resubmission_runtime_v2_3_1.json)、[新增软件检查](../results/motion_v2/v2_3_1_resubmission/software/attempt1/status.json)、[原回放](MOTION_V2_3_1_REPLAY_ZH.md)。

## 冻结版本、预检与训练完成情况

远端科学 HEAD 保持 **`6dc5ed253dfa305b26be8f19d40edcd6a5878ea7`**。源码、protocol、config、数据、seed、判据、BudgetGuard 和 wrapper 全部原样；科学 protocol SHA256 为 `b88c85ca78a91b722c48bdd8465ce88b324ac281447c4608d1447f1a6b508c60`，config SHA256 为 `5156c095d4e805a60d1f7b29c1491e8fed51b8d612b0173b80d4349d803a5bcd`。科学源码 73/73 项、部署源码 8/8 项均与冻结 6dc 提交一致。完整锁与运行核验见 [registration_verification.json](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/registration_verification.json)。科学目录沿用冻结 wrapper 的 `step2_gpu_attempt1` 名称，作业身份以 16614 和运行层 attempt2 回执为准。

提交前重新完成原 venv import/版本检查，复核 **2432 个 NPZ** 与原 ZIP 的 SHA256，没有生成或改写数据。见[环境复验](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/preflight_environment/receipt.json)、[数据复验](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/submission/data_reverification.json)、[提交回执](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/submission/submission.json)。

实际编码器为冻结官方 `facebook/ijepa_vith14_1k`，revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e`，FP32、grid2 四格池化、5120 维。归一化和 basis 只用 train。沿用登记架构、四类训练历史和 0.25 整段 memory dropout；39398 帧编码完成，三个 seed 每个都完成 120 epochs、7680 optimizer steps，共 360 epochs。checkpoint 只按登记 validation feature MSE 选择：

| seed | checkpoint | checkpoint 选择用 validation MSE | 完成 epochs | optimizer steps |
|---:|---|---:|---:|---:|
| 17 | best_epoch_016.pt | 0.217820185 | 120 | 7680 |
| 29 | best_epoch_011.pt | 0.221463612 | 120 | 7680 |
| 43 | best_epoch_018.pt | 0.220079627 | 120 | 7680 |

这些选择分数对应原四种训练条件下的 validation 评分，不等同于下文单独 a/b/c/d 的指标。来源：[seed 17](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/predictor/motion/seed_17/selection.json)、[seed 29](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/predictor/motion/seed_29/selection.json)、[seed 43](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/predictor/motion/seed_43/selection.json)、[encoder metadata](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/encoder_metadata.json)、[运行环境](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/runtime_environment.json)。

位置 decoder 的拟合和 ridge 分组交叉验证都只用 train，最终 ridge 为 **10**。validation 真实 target features 仅用于冻结 decoder 诊断，不用于其拟合或选择。见[decoder provenance](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/decoder/decoder_provenance.json)与[训练分组 CV](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/decoder/decoder_cv.json)。

运行前全部 96 个 validation pair 输入审计通过：a/c/d 模型输入共 **288 次精确一致比较**，b 两帧时刻合法性 **192 次**；`hidden_state_in_model_inputs=false`。见[输入一致性审计](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/control_input_identity_audit.json)。

## 主结果：三个固定 seed 平均

a 为 recent-only、无 memory；b 为第一张新段可见帧加 end；c 为第一张与最后一张旧段可见帧；d 为最后一张旧段可见帧加 end。共享最近 R=4 帧、历史预算 B=2。下文方向下界也记为 B，由上下文区分。

feature MSE 为 5120 个经 train-only 归一化的 target feature 维度均方误差；position MSE 为解码后归一化 x/y 坐标均方误差，不是像素距离或 RMSE。先平均 pair 内两个 sibling，再平均固定三个 seed，对 96 个场景 pair 做 2000 次 bootstrap，seed 901，percentile 2.5%/97.5%，linear quantile。全部绝对指标、差值、B、decoder 诊断与逐 seed 表共享同一套重采样索引。


| 条件 | h | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---|---:|---:|---|---:|---|
| a | 3 | 0.239936323 | [0.230350805, 0.249590017] | 0.0230899968 | [0.0205893923, 0.0256850592] |
| a | 6 | 0.253781308 | [0.245288267, 0.262451892] | 0.026242516 | [0.0241289094, 0.0284649131] |
| b | 3 | 0.18681108 | [0.176280593, 0.19850645] | 0.00605192017 | [0.00476684491, 0.00758422044] |
| b | 6 | 0.198887371 | [0.189070967, 0.208607819] | 0.00685985888 | [0.00563872707, 0.00820422235] |
| c | 3 | 0.262927121 | [0.252947427, 0.273750449] | 0.0259217187 | [0.0230253473, 0.0294122263] |
| c | 6 | 0.28058246 | [0.269806015, 0.291673482] | 0.0294689294 | [0.0267948636, 0.0325900461] |
| d | 3 | 0.240418074 | [0.231215114, 0.249560356] | 0.0157658745 | [0.0142535569, 0.0173336874] |
| d | 6 | 0.249386873 | [0.240343958, 0.259099636] | 0.0195420055 | [0.0182485947, 0.0208719759] |

## 方向下界 B 与冻结 decoder 诊断

B 来自此前 CPU 保存的 validation 真实位置，表示两个 sibling 只能得到相同预测时的最小 pair position MSE；它没有 seed 轴，也没有拟合过程。已保存 B 均值不变，下列区间来自本次冻结评估的共享 bootstrap。真实 target features 的 decoder 误差仅为诊断，不是预测器误差或门槛。

| h | B 均值 | B 95% CI | 真实 target features 的 decoder position MSE | decoder 95% CI |
|---:|---:|---|---:|---|
| 3 | 0.0105030797 | [0.00981344449, 0.0111894541] | 1.69438289e-05 | [1.28362475e-05, 2.10633683e-05] |
| 6 | 0.0146392548 | [0.0136689525, 0.0155461462] | 1.51445015e-05 | [1.19833977e-05, 1.84810621e-05] |

## 全部对比与六项必需门槛

以下差值直接引用原 metrics JSON。d−b、c−b、a−b、B−b 为正表示 b 误差更小；d−B 为正表示 d 预测还有超出共同预测理论下界的额外误差。B 是位置空间下界，没有对应 feature 差值。


| 对比 | h | feature 差值 | feature 95% CI | position 差值 | position 95% CI |
|---|---:|---:|---|---:|---|
| d−b | 3 | 0.0536069947 | [0.0444530595, 0.0623401197] | 0.00971395428 | [0.00809570123, 0.011269942] |
| d−b | 6 | 0.0504995022 | [0.041439986, 0.0588790572] | 0.0126821466 | [0.0111143479, 0.014214918] |
| c−b | 3 | 0.0761160411 | [0.0655184154, 0.0869849358] | 0.0198697985 | [0.0171754273, 0.0231076341] |
| c−b | 6 | 0.0816950883 | [0.07265509, 0.0912207089] | 0.0226090706 | [0.0200123242, 0.0255864682] |
| a−b | 3 | 0.0531252438 | [0.043557431, 0.0629656807] | 0.0170380766 | [0.0147529123, 0.0193699021] |
| a−b | 6 | 0.0548939364 | [0.0468140524, 0.062879725] | 0.0193826571 | [0.0173377697, 0.0213992617] |
| B−b | 3 | 不适用 | 不适用 | 0.00445115954 | [0.00270860694, 0.00603568078] |
| d−B | 3 | 不适用 | 不适用 | 0.00526279475 | [0.00384336588, 0.00687973534] |
| B−b | 6 | 不适用 | 不适用 | 0.00777939595 | [0.00602382346, 0.0093850643] |
| d−B | 6 | 不适用 | 不适用 | 0.00490275068 | [0.00368297575, 0.00628461541] |

六项必需门槛按原始 `gate` 字段逐项报告。a−b、feature MSE、d−B 和 decoder 诊断不加入门槛，也未据结果改动阈值。

| 必需 position 对比 | h | 95% CI 下界 | 登记要求 | 结果 |
|---|---:|---:|---|---|
| d−b | 3 | 0.00809570123 | 严格 > 0 | 通过 |
| d−b | 6 | 0.0111143479 | 严格 > 0 | 通过 |
| c−b | 3 | 0.0171754273 | 严格 > 0 | 通过 |
| c−b | 6 | 0.0200123242 | 严格 > 0 | 通过 |
| B−b | 3 | 0.00270860694 | 严格 > 0 | 通过 |
| B−b | 6 | 0.00602382346 | 严格 > 0 | 通过 |

**六项全部通过。** b 优于 d 和 c，并低于共同预测下界 B，在本次条件下通过“预测器能用对帧”的判断。d−B 区间也全为正，说明 d−b 改善包含 d 组合处理不够理想的部分；额外 B−b 判据独立要求 b 超过共同预测下界。

## 逐 seed 附表

以下呈现已保存结果，不新增逐 seed 门槛。B 与冻结 decoder 没有 seed 轴，各 seed 的参考表均与上面的 B/decoder 表相同，不重复列示。正式判定使用三固定 seed 平均后的六项门槛，不能把 seed 或 sibling 当作额外独立场景。


### seed 17


| 条件 | h | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---|---:|---:|---|---:|---|
| a | 3 | 0.239031184 | [0.229511471, 0.248554316] | 0.0226663015 | [0.0202433391, 0.0252292853] |
| a | 6 | 0.252556035 | [0.244015558, 0.261498559] | 0.0261887632 | [0.0240577753, 0.028411289] |
| b | 3 | 0.184536554 | [0.173019056, 0.196734456] | 0.00591594662 | [0.00420802075, 0.00792924237] |
| b | 6 | 0.19722937 | [0.186081083, 0.207860691] | 0.0071669503 | [0.00546989348, 0.00901419808] |
| c | 3 | 0.258657455 | [0.248934438, 0.269179246] | 0.0246503886 | [0.0213426407, 0.028176153] |
| c | 6 | 0.275103344 | [0.264179551, 0.286643867] | 0.0281028722 | [0.0251174255, 0.031379731] |
| d | 3 | 0.241481446 | [0.232529183, 0.251161523] | 0.0161804019 | [0.0143335733, 0.0180972286] |
| d | 6 | 0.24897846 | [0.239367182, 0.258516893] | 0.0201163305 | [0.0185388203, 0.0217368178] |


| 对比 | h | feature 差值 | feature 95% CI | position 差值 | position 95% CI |
|---|---:|---:|---|---:|---|
| d−b | 3 | 0.0569448921 | [0.0466235956, 0.0663740668] | 0.0102644553 | [0.00815170205, 0.0122829242] |
| d−b | 6 | 0.0517490904 | [0.0413353358, 0.0617785991] | 0.0129493802 | [0.010855442, 0.0149328706] |
| c−b | 3 | 0.0741209016 | [0.063133577, 0.0856636076] | 0.018734442 | [0.015729024, 0.0220563779] |
| c−b | 6 | 0.0778739744 | [0.0669235941, 0.0891456592] | 0.0209359219 | [0.0178587796, 0.0242287712] |
| a−b | 3 | 0.0544946303 | [0.0444178682, 0.0645822873] | 0.0167503549 | [0.01434825, 0.0192598733] |
| a−b | 6 | 0.0553266648 | [0.0456543384, 0.0644962582] | 0.0190218129 | [0.0166510983, 0.0213873475] |
| B−b | 3 | 不适用 | 不适用 | 0.00458713309 | [0.00238812078, 0.0065463884] |
| d−B | 3 | 不适用 | 不适用 | 0.00567732222 | [0.00393472483, 0.0076643898] |
| B−b | 6 | 不适用 | 不适用 | 0.00747230453 | [0.0052202938, 0.00958897989] |
| d−B | 6 | 不适用 | 不适用 | 0.00547707571 | [0.00389037638, 0.00725646867] |

### seed 29


| 条件 | h | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---|---:|---:|---|---:|---|
| a | 3 | 0.243366397 | [0.23377664, 0.25321819] | 0.0232728491 | [0.0206538948, 0.0259862715] |
| a | 6 | 0.257002384 | [0.248343062, 0.265836817] | 0.0265895996 | [0.0243837044, 0.0288077394] |
| b | 3 | 0.188547629 | [0.178016909, 0.199992644] | 0.00634483944 | [0.00497803399, 0.00784923412] |
| b | 6 | 0.198469504 | [0.18779647, 0.208814046] | 0.00722309176 | [0.00590503872, 0.00864964463] |
| c | 3 | 0.267321783 | [0.255669683, 0.279238183] | 0.0261536856 | [0.0226706734, 0.0302734894] |
| c | 6 | 0.28452887 | [0.271782915, 0.297557497] | 0.0305484355 | [0.0271112859, 0.0347410768] |
| d | 3 | 0.240540367 | [0.230746867, 0.249860891] | 0.0156738375 | [0.0140878541, 0.0173905917] |
| d | 6 | 0.24972556 | [0.240109053, 0.259638669] | 0.0195281531 | [0.0181363161, 0.0209895235] |


| 对比 | h | feature 差值 | feature 95% CI | position 差值 | position 95% CI |
|---|---:|---:|---|---:|---|
| d−b | 3 | 0.0519927378 | [0.0424404811, 0.060973693] | 0.00932899804 | [0.00773038777, 0.0109093656] |
| d−b | 6 | 0.0512560563 | [0.0417205235, 0.0603138865] | 0.0123050613 | [0.0106120671, 0.0139942478] |
| c−b | 3 | 0.0787741546 | [0.0661849729, 0.0918318122] | 0.0198088462 | [0.0165400111, 0.0236245872] |
| c−b | 6 | 0.0860593662 | [0.0749269984, 0.0981436452] | 0.0233253438 | [0.0198784699, 0.0273979401] |
| a−b | 3 | 0.054818768 | [0.0448344721, 0.0646329476] | 0.0169280097 | [0.0146076855, 0.0193029458] |
| a−b | 6 | 0.05853288 | [0.0497044324, 0.0670656601] | 0.0193665079 | [0.0172343686, 0.0215070005] |
| B−b | 3 | 不适用 | 不适用 | 0.00415824026 | [0.00242297581, 0.00576972459] |
| d−B | 3 | 不适用 | 不适用 | 0.00517075777 | [0.00362946129, 0.0068826278] |
| B−b | 6 | 不适用 | 不适用 | 0.00741616307 | [0.00561447445, 0.0091254359] |
| d−B | 6 | 不适用 | 不适用 | 0.00488889825 | [0.00354986367, 0.00637536828] |

### seed 43


| 条件 | h | feature MSE | feature 95% CI | position MSE | position 95% CI |
|---|---:|---:|---|---:|---|
| a | 3 | 0.237411389 | [0.227449945, 0.247643094] | 0.0233308398 | [0.0207137169, 0.0262375363] |
| a | 6 | 0.251785505 | [0.242834059, 0.261056894] | 0.025949185 | [0.0238633076, 0.0281545619] |
| b | 3 | 0.187349056 | [0.175732009, 0.199921345] | 0.00589497445 | [0.00460376432, 0.00734997217] |
| b | 6 | 0.200963241 | [0.190064792, 0.212299587] | 0.00618953458 | [0.00488927491, 0.00772008871] |
| c | 3 | 0.262802123 | [0.250755941, 0.276284034] | 0.0269610818 | [0.0233988057, 0.0313070733] |
| c | 6 | 0.282115165 | [0.269573568, 0.294530303] | 0.0297554806 | [0.0266914975, 0.033499638] |
| d | 3 | 0.23923241 | [0.229501414, 0.249228943] | 0.015443384 | [0.0140306929, 0.0168883766] |
| d | 6 | 0.2494566 | [0.23959332, 0.259779543] | 0.0189815329 | [0.0177070137, 0.0203046477] |


| 对比 | h | feature 差值 | feature 95% CI | position 差值 | position 95% CI |
|---|---:|---:|---|---:|---|
| d−b | 3 | 0.0518833541 | [0.0419778212, 0.0615263711] | 0.0095484095 | [0.00782047096, 0.0111438569] |
| d−b | 6 | 0.0484933598 | [0.0397447058, 0.056869446] | 0.0127919983 | [0.0110806654, 0.0144017641] |
| c−b | 3 | 0.0754530672 | [0.0622373538, 0.0892018258] | 0.0210661074 | [0.0176464798, 0.0251630339] |
| c−b | 6 | 0.0811519243 | [0.0694247005, 0.0944508106] | 0.023565946 | [0.020407202, 0.0272802568] |
| a−b | 3 | 0.0500623332 | [0.0390337468, 0.061449421] | 0.0174358653 | [0.0148578808, 0.0200665908] |
| a−b | 6 | 0.0508222644 | [0.0417919088, 0.0594974525] | 0.0197596505 | [0.0175961004, 0.0218580444] |
| B−b | 3 | 不适用 | 不适用 | 0.00460810526 | [0.00290813702, 0.00611664901] |
| d−B | 3 | 不适用 | 不适用 | 0.00494030425 | [0.00361424712, 0.00639501008] |
| B−b | 6 | 不适用 | 不适用 | 0.00844972024 | [0.00671156468, 0.0100388574] |
| d−B | 6 | 不适用 | 不适用 | 0.00434227808 | [0.00323808308, 0.00555039425] |

## 原生结算与实际 GPU 用时

| 尝试 | 作业 | 节点 | Start / End（UTC） | 终态 | Restarts | A100 秒 |
|---|---:|---|---|---|---:|---:|
| attempt1 | 16550 | b-19-12 | 2026-10-31 22:07:12 / 22:20:01 | CANCELLED by 0；入口前基础设施失败 | 0 | 769 |
| attempt2 | 16614 | b-19-27 | 2026-10-31 23:52:04 / 2026-11-01 00:41:00 | COMPLETED；科学门槛通过 | 0 | 2936 |
| 累计 | — | — | — | 两次已结算，未触发 attempt3 | — | **3705** |

16614 使用单张 **NVIDIA A100-SXM4-40GB**，实际分配 2936 秒，即 0.815556 A100 小时。加上 16550，累计 **3705 秒，即 1.029167 小时**，低于 14400 秒总上限；每次的 10800 秒 wall-time 和冻结 BudgetGuard 没有改变。GPU 分配期间的初始化、下载、CPU 工作、训练及收尾都计入。runner 结束快照约 2914.63 秒不是最终计费，最终采用 sacct 的 2936 秒。科学入口已执行且通过，没有第三次提交，已按协议停止。

原生 `Requeue=1` 按授权修订记录而不阻塞，`Restarts=0`；实际 CPU 分配为 12、请求为 4，不影响单张 A100 的核验。监测用的 CPU 辅助会话 16611 在 00:21:01 UTC 被系统记为 `CANCELLED by 0`，随后仅创建零 GPU 的辅助会话 16620 继续读取证据；科学作业 16614 没有中断或重提。两辅助会话的 AllocTRES 均无 GPU，不计入 A100 秒数，原生记录见[恢复监测快照](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/monitor/20261101T002459Z/native_snapshot.json)。

来源：[16614 accounting](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/settlement/accounting.json)、[settlement status](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/settlement/status.json)、[最终 sacct](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/settlement/final_sacct_duplicates.stdout.txt)、[最终 scontrol](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/settlement/final_scontrol.stdout.txt)、[原 16550 结算](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/settlement/accounting.json)。

## 归档完整性与三个列明的缺失路径

终态 ZIP 为 **1131125 字节**，SHA256 为 `0fcac0a6751afae34c25dd8ddf1902a7458375f00db8dbc30fdeca2313beeb76`。ZIP、四个 sidecar 与内部 **136 个原始源文件**校验通过。远端导出前还核验了原 runner 的 **1295 项**产物清单，原清单未修改。来源：[下载核验](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/download_verification.json)、[导出 manifest](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/export_manifest.json)、[科学文件 inventory](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/scientific_file_inventory.json)、[原 runner SHA256 清单](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/artifact_sha256.json)。

导出记录只列出以下三个缺失路径，全部保留原状，没有补造文件：

| 缺失路径 | 解释与证据 |
|---|---|
| `reused_archive_sha256.txt` | 缺少该独立文本回执；本次 [data_reverification.json](../results/motion_v2/v2_3_1_resubmission/terminal_evidence/runtime_retries_v2_3_1/attempt2/submission/data_reverification.json) 已记录原数据包实际 SHA256/大小和 2432 个 NPZ 的重新核验。 |
| `scientific-job-16550.stderr.log` | 16550 入口前失败的预期不存在项，由四项诊断证明；不属于 16614 科学输出缺失。 |
| `scientific-job-16550.stdout.log` | 同上。 |

较大的 episode 特征、Hugging Face 权重/缓存、predictor checkpoint 与 basis 按小型证据导出范围留在原远端现场，清单保留其路径与 SHA256；它们与上述缺失项不同。本报告仅读取已导出的 JSON，没有打开科学 NPZ 或加载模型。

## 风险、结论范围和停止边界

本次结论限于当前生成器、validation 场景、冻结编码器、指定预算及三个固定 seed：提供正确的新段帧可以改善预测，并超过仅共享终点等信息时的方向下界。这里没有训练自行选择这些帧的保留策略。

最大的研究风险是**特权选帧收益能否由实际保留策略获得，并在独立场景中保持**。当前 CI 描述三个已拟合 seed 条件下的场景 pair 不确定性，不是所有训练随机种子的总体区间；checkpoint 选择和本次门槛评估都使用登记的 validation，尚无独立 test 泛化结论。单帧隐藏状态分类检查也只约束登记分类器及场景分布，不能证明任何模型都无法利用单帧线索。

旧报告、16550 原始回执、协议、状态、哈希锁和三个 deployment 现场全部保留。本轮没有改阈值、重选实验、打开或生成 test、训练保留策略或进入第 3 步。**第 2 步通过后即停止，等待用户确认后续工作。** 当前机器状态见 [status.json](../results/motion_v2/v2_3_1_resubmission/status.json)。
