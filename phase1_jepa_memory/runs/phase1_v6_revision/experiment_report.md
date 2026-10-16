# v6 GPU 实验报告：实际结果与未完成项

更新时间：2026-10-13 22:39 EDT。固定视觉计算及最终原生数值审计完成。

**三组 GPU 训练7251已全部COMPLETED0:0，各30轮/9690步；正式推理前和root独立重算的六个实际模型证明一致。原健康门槛6/8、固定旧参考比较9/16，均REJECTED。B/C完整结果如下。 标签精确相位仍未证明，语义任务评估未完成，Phase1未通过。**

## 本轮改变与失败解释

唯一科学改动为AdamW weight_decay从0.05增加到0.20。原架构、官方冻结编码器、完整缓存、训练数据、三个种子、30轮、学习率/调度和最早等权validation best规则保持。7209 train-only归一化和128rank输入/输出基底逐字节复用，未重新拟合。

上一轮的训练误差持续下降，而验证误差在5/6/7轮达到最低后上升，支持泛化落差假设；增加正则化是一次固定单因素检验，不代表已确认唯一失败机制。新best验证MSE略有下降，但本轮最早best仍为5/6/7。

本轮h16/h64的seed29历史收益95%区间均跨零，原要求是每个种子与pooled都通过，所以6/8仍失败。seed43恢复为正区间，不能替代seed29。与上一轮固定近期/完整历史预测比较也只有9/16支持改善，不能据pooled或平均误差较小称稳定采用。标签缺口独立于这些无标签视觉失败；解锁/启动问题也不用于解释科学失败。

## 作业、资源与范围

| 阶段 | job | 实际状态/时间UTC | 内容 |
|---|---:|---|---|
| 复用真实编码/缓存审计 | 7187/7188 | 已完整COMPLETED0:0 | 全78录像/50688时点与完整缓存证明 |
| 本轮GPU训练 | 7251 | COMPLETED0:0；00:47:36–01:37:50 | 一A100-SXM4-40GB、4科学线程、60G、原4h预算；实际50分14秒 |
| 六模型证明与自动评估交接 | 7253 | COMPLETED0:0；01:39:07–01:39:30 | 来源/数值审计后冻结协议并自动提交唯一7268 |
| 冻结GPU模型固定评估 | 7268 | 原生COMPLETED0:0；01:40:37–02:08:47，28分10秒 | 原4科学线程/16G/零GPU/2h；原全部A/B/C及旧标量比较 |

冻结GPU模型的CPU评估属于原设计，不是CPU模型训练替代。未执行CPU编码/reader训练fallback，未发送支持邮件或访问申请，未增加科学预算、删seed或样本。

固定开发数据为58train/20validation RGB、198episodes、50688时点。train136episodes/34816观察/5168查询，validation62episodes/15872观察/2356查询；封存18RGB/9人及fresh9137确认未打开。

官方I-JEPA revision f157467ea509bc356ff9f61fd3c0d840eec5e04e，float32、grid2、batch2、TF32关闭；输出5120D。Reader hidden256、2层4heads、FFN1024、dropout.1、historydropout.15、recent8/capacity16、h1/4/16/64、recent/FIFO/Reservoir/full；1779072参数。AdamW3e-4/wd.20、5%warmup+cosine。

## 完整训练与选模

| seed | epochs/steps | 最早best | v5 best val MSE | v6 best val MSE | v6末轮train MSE | v6末轮val MSE |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 30/9690 | 5 | 0.34036500 | 0.33851528 | 0.21126342 | 0.38394897 |
| 29 | 30/9690 | 6 | 0.33975386 | 0.33884275 | 0.21171161 | 0.38653680 |
| 43 | 30/9690 | 7 | 0.34271761 | 0.34110368 | 0.21121550 | 0.38986724 |

选模统计按查询等权；下方科学评估按参与者分层等权，两种均值不能直接相减。六个best/latest在任何评估推理前已核验有限值、source/cache/fit/config/RNG、optimizer实际wd.20、原schedule、完整轮数/步数和CSV最早best。

![本轮与上一轮训练及实际健康/固定参考区间](/Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v6_revision/training7251_health7268.png)

## A：完整预测健康评估

完整113088条标量：3seeds×2356validation queries×4conditions×4h；完整5120D normalized MSE、persistence与固定基底truefuture oracle。query→episode→recording→participant等权，10验证参与者，10000次seed0 bootstrap，双侧95%CI。原主门槛要求h16和h64的3seed+pooled全部正区间；实际6/8，REJECTED。h1/h4各0/4，完整保留。

| h/秒 | seed/汇总 | Recent−Full MSE | 95%CI | 判定 |
|---|---|---:|---|---|
| 1/0.5s | 17 | 0.00026071 | [-0.00015607, 0.00079114] | FAIL |
| 1/0.5s | 29 | -0.00242425 | [-0.00319810, -0.00171629] | FAIL |
| 1/0.5s | 43 | -0.00092413 | [-0.00134425, -0.00050496] | FAIL |
| 1/0.5s | pooled | -0.00102923 | [-0.00145479, -0.00065011] | FAIL |
| 4/2s | 17 | 0.00136462 | [-0.00013637, 0.00308674] | FAIL |
| 4/2s | 29 | -0.00103933 | [-0.00198274, -0.00007947] | FAIL |
| 4/2s | 43 | -0.00259868 | [-0.00381825, -0.00147138] | FAIL |
| 4/2s | pooled | -0.00075780 | [-0.00168118, 0.00012976] | FAIL |
| 16/8s | 17 | 0.00514155 | [0.00334571, 0.00751135] | PASS |
| 16/8s | 29 | 0.00074293 | [-0.00065036, 0.00187337] | FAIL |
| 16/8s | 43 | 0.00104734 | [0.00038532, 0.00179583] | PASS |
| 16/8s | pooled | 0.00231061 | [0.00175820, 0.00298557] | PASS |
| 64/32s | 17 | 0.00537268 | [0.00308977, 0.00798169] | PASS |
| 64/32s | 29 | 0.00104713 | [-0.00040009, 0.00227796] | FAIL |
| 64/32s | 43 | 0.00246908 | [0.00125971, 0.00377683] | PASS |
| 64/32s | pooled | 0.00296296 | [0.00200583, 0.00422645] | PASS |

### 参与者等权绝对MSE

| seed | h | recent | FIFO | Reservoir | full | persistence | fixedbasis oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 0.16114383 | 0.16061538 | 0.16077397 | 0.16088312 | 0.16439188 | 0.05117588 |
| 17 | 4 | 0.29458287 | 0.29249766 | 0.29295104 | 0.29321826 | 0.31762146 | 0.07934702 |
| 17 | 16 | 0.40448557 | 0.39895852 | 0.40021420 | 0.39934402 | 0.44522266 | 0.10247927 |
| 17 | 64 | 0.48148398 | 0.47518669 | 0.47666551 | 0.47611130 | 0.54274519 | 0.12092840 |
| 29 | 1 | 0.16503846 | 0.16696935 | 0.16548228 | 0.16746271 | 0.16439188 | 0.05117588 |
| 29 | 4 | 0.29770624 | 0.29780179 | 0.29724047 | 0.29874557 | 0.31762146 | 0.07934702 |
| 29 | 16 | 0.39880237 | 0.39638468 | 0.39679716 | 0.39805944 | 0.44522266 | 0.10247927 |
| 29 | 64 | 0.47405732 | 0.47164579 | 0.47164376 | 0.47301019 | 0.54274519 | 0.12092840 |
| 43 | 1 | 0.16214346 | 0.16265610 | 0.16229208 | 0.16306759 | 0.16439188 | 0.05117588 |
| 43 | 4 | 0.29683444 | 0.29800243 | 0.29708982 | 0.29943313 | 0.31762146 | 0.07934702 |
| 43 | 16 | 0.40367472 | 0.40185643 | 0.40261151 | 0.40262738 | 0.44522266 | 0.10247927 |
| 43 | 64 | 0.47992963 | 0.47675725 | 0.47814285 | 0.47746054 | 0.54274519 | 0.12092840 |

truefuture oracle仅是诊断下界，不可部署、不能证明可学习性。root使用原113088标量独立重放原分层均值/CI，最大差3.5561831257524545e-17；全部判断一致，没有新的模型推理/拟合/训练。

## 预先固定的旧参考比较

比较同一开发查询上的new full与旧7226已执行的recent/full标量，未重新推理旧模型；所有113088个seed/recording/episode/time/policy/horizon键严格一致。h16/h64×2参考×3seed+pooled共16项，实际9/16，REJECTED。原A/B规则保持，不用旧参考结果替代原门槛。

| 固定比较 | seed/汇总 | Old−NewFull MSE | 95%CI | 判定 |
|---|---|---:|---|---|
| old_recent_vs_new_full_h1 | 17 | 0.00127200 | [0.00061811, 0.00190809] | PASS |
| old_recent_vs_new_full_h1 | 29 | -0.00348874 | [-0.00453290, -0.00243473] | FAIL |
| old_recent_vs_new_full_h1 | 43 | 0.00031639 | [-0.00016957, 0.00077916] | FAIL |
| old_recent_vs_new_full_h1 | pooled | -0.00063345 | [-0.00103380, -0.00024670] | FAIL |
| old_full_vs_new_full_h1 | 17 | 0.00114370 | [0.00052275, 0.00178650] | PASS |
| old_full_vs_new_full_h1 | 29 | -0.00194613 | [-0.00279655, -0.00110082] | FAIL |
| old_full_vs_new_full_h1 | 43 | 0.00004003 | [-0.00140090, 0.00108985] | FAIL |
| old_full_vs_new_full_h1 | pooled | -0.00025413 | [-0.00071483, 0.00019080] | FAIL |
| old_recent_vs_new_full_h4 | 17 | 0.00256352 | [0.00152380, 0.00366681] | PASS |
| old_recent_vs_new_full_h4 | 29 | -0.00113251 | [-0.00349217, 0.00038375] | FAIL |
| old_recent_vs_new_full_h4 | 43 | -0.00004756 | [-0.00209922, 0.00136681] | FAIL |
| old_recent_vs_new_full_h4 | pooled | 0.00046115 | [-0.00019950, 0.00112987] | FAIL |
| old_full_vs_new_full_h4 | 17 | 0.00184189 | [0.00030415, 0.00348030] | PASS |
| old_full_vs_new_full_h4 | 29 | -0.00093041 | [-0.00352276, 0.00090004] | FAIL |
| old_full_vs_new_full_h4 | 43 | 0.00050092 | [-0.00105056, 0.00188445] | FAIL |
| old_full_vs_new_full_h4 | pooled | 0.00047080 | [-0.00101845, 0.00166495] | FAIL |
| old_recent_vs_new_full_h16 | 17 | 0.00564683 | [0.00363575, 0.00820638] | PASS |
| old_recent_vs_new_full_h16 | 29 | 0.00231467 | [-0.00179903, 0.00551764] | FAIL |
| old_recent_vs_new_full_h16 | 43 | 0.00383608 | [-0.00081522, 0.00804414] | FAIL |
| old_recent_vs_new_full_h16 | pooled | 0.00393252 | [0.00218606, 0.00567696] | PASS |
| old_full_vs_new_full_h16 | 17 | 0.00210569 | [0.00051464, 0.00400749] | PASS |
| old_full_vs_new_full_h16 | 29 | 0.00074547 | [-0.00300039, 0.00366544] | FAIL |
| old_full_vs_new_full_h16 | 43 | 0.00554999 | [-0.00084758, 0.01484809] | FAIL |
| old_full_vs_new_full_h16 | pooled | 0.00280038 | [0.00085118, 0.00487791] | PASS |
| old_recent_vs_new_full_h64 | 17 | 0.00784528 | [0.00462522, 0.01125611] | PASS |
| old_recent_vs_new_full_h64 | 29 | 0.00420582 | [0.00101929, 0.00748268] | PASS |
| old_recent_vs_new_full_h64 | 43 | 0.00492962 | [-0.00136437, 0.01108969] | FAIL |
| old_recent_vs_new_full_h64 | pooled | 0.00566024 | [0.00310201, 0.00839703] | PASS |
| old_full_vs_new_full_h64 | 17 | 0.00417626 | [0.00163275, 0.00678611] | PASS |
| old_full_vs_new_full_h64 | 29 | 0.00158011 | [-0.00148899, 0.00486512] | FAIL |
| old_full_vs_new_full_h64 | 43 | 0.00217408 | [-0.00192460, 0.00601560] | FAIL |
| old_full_vs_new_full_h64 | pooled | 0.00264348 | [0.00060341, 0.00492088] | PASS |

root独立旧/新原标量重放最大差7.112366251504909e-17，全部32判断一致。旧参考SHA d158c869a132f1db35fbdc2491802c6899a4770dfc13a0338c0facca30eeed2c。

## B：全部17删除分支与32共同延续

已原生取回并复核。

全部624states=78recordings×2source policies×4固定均匀状态，每状态3models×17primary删除×32共同Reservoir延续。共同oracle按.5instant四h均值+.5future32四h均值选一次，同一选择用于h16/h64，exacttie删最早时间，不能逐h重选。

科学状态：REJECTED；48项必要检查实际47项支持。失败项为Reservoir来源/FIFO基线/seed43/h64，点收益0.0000736861，95%CI[-0.0000007579,0.0001566059]。不能忽略区间跨零或逐horizon重选oracle。

| source/基线 | seed | 指标 | 点收益 | 95%CI | 判定 |
|---|---|---|---:|---|---|
| FIFO_FIFO | 17 | joint | 0.00012834 | [0.00008567, 0.00017153] | PASS |
| FIFO_FIFO | 17 | h16 | 0.00017046 | [0.00009150, 0.00025376] | PASS |
| FIFO_FIFO | 17 | h64 | 0.00023744 | [0.00016479, 0.00031241] | PASS |
| FIFO_FIFO | 29 | joint | 0.00031419 | [0.00026074, 0.00037352] | PASS |
| FIFO_FIFO | 29 | h16 | 0.00035815 | [0.00027438, 0.00043742] | PASS |
| FIFO_FIFO | 29 | h64 | 0.00041028 | [0.00033057, 0.00049187] | PASS |
| FIFO_FIFO | 43 | joint | 0.00015054 | [0.00009983, 0.00020050] | PASS |
| FIFO_FIFO | 43 | h16 | 0.00018277 | [0.00010827, 0.00026375] | PASS |
| FIFO_FIFO | 43 | h64 | 0.00019990 | [0.00010280, 0.00031191] | PASS |
| FIFO_FIFO | pooled | joint | 0.00019769 | [0.00015665, 0.00024201] | PASS |
| FIFO_FIFO | pooled | h16 | 0.00023713 | [0.00018811, 0.00028962] | PASS |
| FIFO_FIFO | pooled | h64 | 0.00028254 | [0.00021590, 0.00035458] | PASS |
| FIFO_Reservoir | 17 | joint | 0.00013363 | [0.00009981, 0.00016791] | PASS |
| FIFO_Reservoir | 17 | h16 | 0.00018008 | [0.00008586, 0.00027467] | PASS |
| FIFO_Reservoir | 17 | h64 | 0.00025259 | [0.00019797, 0.00029835] | PASS |
| FIFO_Reservoir | 29 | joint | 0.00025715 | [0.00019710, 0.00031699] | PASS |
| FIFO_Reservoir | 29 | h16 | 0.00032897 | [0.00016782, 0.00049903] | PASS |
| FIFO_Reservoir | 29 | h64 | 0.00036251 | [0.00024981, 0.00047983] | PASS |
| FIFO_Reservoir | 43 | joint | 0.00018700 | [0.00013168, 0.00027290] | PASS |
| FIFO_Reservoir | 43 | h16 | 0.00036099 | [0.00018361, 0.00065878] | PASS |
| FIFO_Reservoir | 43 | h64 | 0.00018707 | [0.00009629, 0.00026624] | PASS |
| FIFO_Reservoir | pooled | joint | 0.00019259 | [0.00015419, 0.00023398] | PASS |
| FIFO_Reservoir | pooled | h16 | 0.00029001 | [0.00016334, 0.00043485] | PASS |
| FIFO_Reservoir | pooled | h64 | 0.00026739 | [0.00021728, 0.00031741] | PASS |
| Reservoir_FIFO | 17 | joint | 0.00009568 | [0.00007572, 0.00011608] | PASS |
| Reservoir_FIFO | 17 | h16 | 0.00011483 | [0.00006176, 0.00016586] | PASS |
| Reservoir_FIFO | 17 | h64 | 0.00014701 | [0.00011951, 0.00017519] | PASS |
| Reservoir_FIFO | 29 | joint | 0.00033553 | [0.00027416, 0.00039904] | PASS |
| Reservoir_FIFO | 29 | h16 | 0.00020791 | [0.00000403, 0.00039114] | PASS |
| Reservoir_FIFO | 29 | h64 | 0.00066660 | [0.00048900, 0.00087642] | PASS |
| Reservoir_FIFO | 43 | joint | 0.00011856 | [0.00007214, 0.00017152] | PASS |
| Reservoir_FIFO | 43 | h16 | 0.00012624 | [0.00007144, 0.00017988] | PASS |
| Reservoir_FIFO | 43 | h64 | 0.00007369 | [-0.00000076, 0.00015661] | FAIL |
| Reservoir_FIFO | pooled | joint | 0.00018326 | [0.00014632, 0.00022092] | PASS |
| Reservoir_FIFO | pooled | h16 | 0.00014966 | [0.00006503, 0.00022709] | PASS |
| Reservoir_FIFO | pooled | h64 | 0.00029577 | [0.00024132, 0.00035518] | PASS |
| Reservoir_Reservoir | 17 | joint | 0.00020515 | [0.00015684, 0.00026527] | PASS |
| Reservoir_Reservoir | 17 | h16 | 0.00026764 | [0.00019015, 0.00035379] | PASS |
| Reservoir_Reservoir | 17 | h64 | 0.00033803 | [0.00021796, 0.00050145] | PASS |
| Reservoir_Reservoir | 29 | joint | 0.00041539 | [0.00027653, 0.00057577] | PASS |
| Reservoir_Reservoir | 29 | h16 | 0.00053403 | [0.00025091, 0.00088535] | PASS |
| Reservoir_Reservoir | 29 | h64 | 0.00079445 | [0.00052783, 0.00110191] | PASS |
| Reservoir_Reservoir | 43 | joint | 0.00019218 | [0.00011574, 0.00029570] | PASS |
| Reservoir_Reservoir | 43 | h16 | 0.00025092 | [0.00016812, 0.00033972] | PASS |
| Reservoir_Reservoir | 43 | h64 | 0.00032906 | [0.00020930, 0.00047654] | PASS |
| Reservoir_Reservoir | pooled | joint | 0.00027091 | [0.00021008, 0.00033268] | PASS |
| Reservoir_Reservoir | pooled | h16 | 0.00035087 | [0.00023098, 0.00047541] | PASS |
| Reservoir_Reservoir | pooled | h64 | 0.00048718 | [0.00037358, 0.00061003] | PASS |

Oracle只报告删除可行性上界，不可部署、不能证明learnability。

## C：不可逆持久载荷重放

来源：原生最终取回与独立审计。

全198episodes×3模型×FIFO/Reservoir/Uniform×capacity16主及8/32次×38queries×4h，实际812592标量、456192行保留轨迹、1782条完整重放轨迹。当前原始标量的参与者均值独立重放最大差为0；当前轨迹逐项固定FIFO/Algorithm R/Uniform随机状态、容量、chronology及no-reentry独立重放通过。没有外部archive取回已删载荷。

root最终独立固定RNG/容量/chronology/no-reentry核验：PASS。

| seed | split | policy | capacity | h | participant等权MSE |
|---|---|---|---:|---:|---:|
| 17 | train | FIFO | 8 | 1 | 0.15630843 |
| 17 | train | FIFO | 8 | 4 | 0.25883614 |
| 17 | train | FIFO | 8 | 16 | 0.32384478 |
| 17 | train | FIFO | 8 | 64 | 0.38755214 |
| 29 | train | FIFO | 8 | 1 | 0.15364827 |
| 29 | train | FIFO | 8 | 4 | 0.24700974 |
| 29 | train | FIFO | 8 | 16 | 0.31044435 |
| 29 | train | FIFO | 8 | 64 | 0.37015870 |
| 43 | train | FIFO | 8 | 1 | 0.15023091 |
| 43 | train | FIFO | 8 | 4 | 0.23845665 |
| 43 | train | FIFO | 8 | 16 | 0.29761678 |
| 43 | train | FIFO | 8 | 64 | 0.35426160 |
| 17 | train | Reservoir | 8 | 1 | 0.15657409 |
| 17 | train | Reservoir | 8 | 4 | 0.26002992 |
| 17 | train | Reservoir | 8 | 16 | 0.32442355 |
| 17 | train | Reservoir | 8 | 64 | 0.38818732 |
| 29 | train | Reservoir | 8 | 1 | 0.15337492 |
| 29 | train | Reservoir | 8 | 4 | 0.24741634 |
| 29 | train | Reservoir | 8 | 16 | 0.31006677 |
| 29 | train | Reservoir | 8 | 64 | 0.36971628 |
| 43 | train | Reservoir | 8 | 1 | 0.15018929 |
| 43 | train | Reservoir | 8 | 4 | 0.23884154 |
| 43 | train | Reservoir | 8 | 16 | 0.29771639 |
| 43 | train | Reservoir | 8 | 64 | 0.35432730 |
| 17 | train | Uniform | 8 | 1 | 0.15634500 |
| 17 | train | Uniform | 8 | 4 | 0.25899860 |
| 17 | train | Uniform | 8 | 16 | 0.32386145 |
| 17 | train | Uniform | 8 | 64 | 0.38758760 |
| 29 | train | Uniform | 8 | 1 | 0.15355056 |
| 29 | train | Uniform | 8 | 4 | 0.24697731 |
| 29 | train | Uniform | 8 | 16 | 0.31023243 |
| 29 | train | Uniform | 8 | 64 | 0.36989231 |
| 43 | train | Uniform | 8 | 1 | 0.15021090 |
| 43 | train | Uniform | 8 | 4 | 0.23852008 |
| 43 | train | Uniform | 8 | 16 | 0.29761232 |
| 43 | train | Uniform | 8 | 64 | 0.35422213 |
| 17 | train | FIFO | 16 | 1 | 0.15618821 |
| 17 | train | FIFO | 16 | 4 | 0.25808282 |
| 17 | train | FIFO | 16 | 16 | 0.32372068 |
| 17 | train | FIFO | 16 | 64 | 0.38742552 |
| 29 | train | FIFO | 16 | 1 | 0.15378208 |
| 29 | train | FIFO | 16 | 4 | 0.24703091 |
| 29 | train | FIFO | 16 | 16 | 0.31078355 |
| 29 | train | FIFO | 16 | 64 | 0.37048399 |
| 43 | train | FIFO | 16 | 1 | 0.15027854 |
| 43 | train | FIFO | 16 | 4 | 0.23836923 |
| 43 | train | FIFO | 16 | 16 | 0.29765497 |
| 43 | train | FIFO | 16 | 64 | 0.35426384 |
| 17 | train | Reservoir | 16 | 1 | 0.15640200 |
| 17 | train | Reservoir | 16 | 4 | 0.25937149 |
| 17 | train | Reservoir | 16 | 16 | 0.32409505 |
| 17 | train | Reservoir | 16 | 64 | 0.38780028 |
| 29 | train | Reservoir | 16 | 1 | 0.15337419 |
| 29 | train | Reservoir | 16 | 4 | 0.24723118 |
| 29 | train | Reservoir | 16 | 16 | 0.31001550 |
| 29 | train | Reservoir | 16 | 64 | 0.36965747 |
| 43 | train | Reservoir | 16 | 1 | 0.15018352 |
| 43 | train | Reservoir | 16 | 4 | 0.23874462 |
| 43 | train | Reservoir | 16 | 16 | 0.29767528 |
| 43 | train | Reservoir | 16 | 64 | 0.35426210 |
| 17 | train | Uniform | 16 | 1 | 0.15622467 |
| 17 | train | Uniform | 16 | 4 | 0.25834331 |
| 17 | train | Uniform | 16 | 16 | 0.32376240 |
| 17 | train | Uniform | 16 | 64 | 0.38747159 |
| 29 | train | Uniform | 16 | 1 | 0.15366058 |
| 29 | train | Uniform | 16 | 4 | 0.24696688 |
| 29 | train | Uniform | 16 | 16 | 0.31046575 |
| 29 | train | Uniform | 16 | 64 | 0.37010827 |
| 43 | train | Uniform | 16 | 1 | 0.15024812 |
| 43 | train | Uniform | 16 | 4 | 0.23844021 |
| 43 | train | Uniform | 16 | 16 | 0.29763674 |
| 43 | train | Uniform | 16 | 64 | 0.35424424 |
| 17 | train | FIFO | 32 | 1 | 0.15615185 |
| 17 | train | FIFO | 32 | 4 | 0.25754440 |
| 17 | train | FIFO | 32 | 16 | 0.32398391 |
| 17 | train | FIFO | 32 | 64 | 0.38761918 |
| 29 | train | FIFO | 32 | 1 | 0.15386595 |
| 29 | train | FIFO | 32 | 4 | 0.24714703 |
| 29 | train | FIFO | 32 | 16 | 0.31104049 |
| 29 | train | FIFO | 32 | 64 | 0.37076587 |
| 43 | train | FIFO | 32 | 1 | 0.15033919 |
| 43 | train | FIFO | 32 | 4 | 0.23841207 |
| 43 | train | FIFO | 32 | 16 | 0.29773314 |
| 43 | train | FIFO | 32 | 64 | 0.35432055 |
| 17 | train | Reservoir | 32 | 1 | 0.15627202 |
| 17 | train | Reservoir | 32 | 4 | 0.25859667 |
| 17 | train | Reservoir | 32 | 16 | 0.32399919 |
| 17 | train | Reservoir | 32 | 64 | 0.38764620 |
| 29 | train | Reservoir | 32 | 1 | 0.15346905 |
| 29 | train | Reservoir | 32 | 4 | 0.24709830 |
| 29 | train | Reservoir | 32 | 16 | 0.31007498 |
| 29 | train | Reservoir | 32 | 64 | 0.36985328 |
| 43 | train | Reservoir | 32 | 1 | 0.15020804 |
| 43 | train | Reservoir | 32 | 4 | 0.23865074 |
| 43 | train | Reservoir | 32 | 16 | 0.29767634 |
| 43 | train | Reservoir | 32 | 64 | 0.35423049 |
| 17 | train | Uniform | 32 | 1 | 0.15618187 |
| 17 | train | Uniform | 32 | 4 | 0.25782262 |
| 17 | train | Uniform | 32 | 16 | 0.32391701 |
| 17 | train | Uniform | 32 | 64 | 0.38755274 |
| 29 | train | Uniform | 32 | 1 | 0.15374769 |
| 29 | train | Uniform | 32 | 4 | 0.24704002 |
| 29 | train | Uniform | 32 | 16 | 0.31071684 |
| 29 | train | Uniform | 32 | 64 | 0.37038516 |
| 43 | train | Uniform | 32 | 1 | 0.15029470 |
| 43 | train | Uniform | 32 | 4 | 0.23843111 |
| 43 | train | Uniform | 32 | 16 | 0.29767786 |
| 43 | train | Uniform | 32 | 64 | 0.35427013 |
| 17 | validation | FIFO | 8 | 1 | 0.16061992 |
| 17 | validation | FIFO | 8 | 4 | 0.29267041 |
| 17 | validation | FIFO | 8 | 16 | 0.39969107 |
| 17 | validation | FIFO | 8 | 64 | 0.47605749 |
| 29 | validation | FIFO | 8 | 1 | 0.16640996 |
| 29 | validation | FIFO | 8 | 4 | 0.29761146 |
| 29 | validation | FIFO | 8 | 16 | 0.39631312 |
| 29 | validation | FIFO | 8 | 64 | 0.47143370 |
| 43 | validation | FIFO | 8 | 1 | 0.16244667 |
| 43 | validation | FIFO | 8 | 4 | 0.29742913 |
| 43 | validation | FIFO | 8 | 16 | 0.40195638 |
| 43 | validation | FIFO | 8 | 64 | 0.47704152 |
| 17 | validation | Reservoir | 8 | 1 | 0.16080359 |
| 17 | validation | Reservoir | 8 | 4 | 0.29323044 |
| 17 | validation | Reservoir | 8 | 16 | 0.40117660 |
| 17 | validation | Reservoir | 8 | 64 | 0.47781577 |
| 29 | validation | Reservoir | 8 | 1 | 0.16522443 |
| 29 | validation | Reservoir | 8 | 4 | 0.29717446 |
| 29 | validation | Reservoir | 8 | 16 | 0.39720741 |
| 29 | validation | Reservoir | 8 | 64 | 0.47226616 |
| 43 | validation | Reservoir | 8 | 1 | 0.16220268 |
| 43 | validation | Reservoir | 8 | 4 | 0.29692023 |
| 43 | validation | Reservoir | 8 | 16 | 0.40295856 |
| 43 | validation | Reservoir | 8 | 64 | 0.47877879 |
| 17 | validation | Uniform | 8 | 1 | 0.16062943 |
| 17 | validation | Uniform | 8 | 4 | 0.29269206 |
| 17 | validation | Uniform | 8 | 16 | 0.39975713 |
| 17 | validation | Uniform | 8 | 64 | 0.47614900 |
| 29 | validation | Uniform | 8 | 1 | 0.16605624 |
| 29 | validation | Uniform | 8 | 4 | 0.29736635 |
| 29 | validation | Uniform | 8 | 16 | 0.39628703 |
| 29 | validation | Uniform | 8 | 64 | 0.47127618 |
| 43 | validation | Uniform | 8 | 1 | 0.16236018 |
| 43 | validation | Uniform | 8 | 4 | 0.29726829 |
| 43 | validation | Uniform | 8 | 16 | 0.40199401 |
| 43 | validation | Uniform | 8 | 64 | 0.47718085 |
| 17 | validation | FIFO | 16 | 1 | 0.16061538 |
| 17 | validation | FIFO | 16 | 4 | 0.29249766 |
| 17 | validation | FIFO | 16 | 16 | 0.39895852 |
| 17 | validation | FIFO | 16 | 64 | 0.47518669 |
| 29 | validation | FIFO | 16 | 1 | 0.16696935 |
| 29 | validation | FIFO | 16 | 4 | 0.29780179 |
| 29 | validation | FIFO | 16 | 16 | 0.39638468 |
| 29 | validation | FIFO | 16 | 64 | 0.47164579 |
| 43 | validation | FIFO | 16 | 1 | 0.16265610 |
| 43 | validation | FIFO | 16 | 4 | 0.29800243 |
| 43 | validation | FIFO | 16 | 16 | 0.40185643 |
| 43 | validation | FIFO | 16 | 64 | 0.47675725 |
| 17 | validation | Reservoir | 16 | 1 | 0.16077397 |
| 17 | validation | Reservoir | 16 | 4 | 0.29295104 |
| 17 | validation | Reservoir | 16 | 16 | 0.40021420 |
| 17 | validation | Reservoir | 16 | 64 | 0.47666551 |
| 29 | validation | Reservoir | 16 | 1 | 0.16548228 |
| 29 | validation | Reservoir | 16 | 4 | 0.29724047 |
| 29 | validation | Reservoir | 16 | 16 | 0.39679716 |
| 29 | validation | Reservoir | 16 | 64 | 0.47164376 |
| 43 | validation | Reservoir | 16 | 1 | 0.16229208 |
| 43 | validation | Reservoir | 16 | 4 | 0.29708982 |
| 43 | validation | Reservoir | 16 | 16 | 0.40261151 |
| 43 | validation | Reservoir | 16 | 64 | 0.47814285 |
| 17 | validation | Uniform | 16 | 1 | 0.16062610 |
| 17 | validation | Uniform | 16 | 4 | 0.29253513 |
| 17 | validation | Uniform | 16 | 16 | 0.39913818 |
| 17 | validation | Uniform | 16 | 64 | 0.47537732 |
| 29 | validation | Uniform | 16 | 1 | 0.16652128 |
| 29 | validation | Uniform | 16 | 4 | 0.29758799 |
| 29 | validation | Uniform | 16 | 16 | 0.39628232 |
| 29 | validation | Uniform | 16 | 64 | 0.47141331 |
| 43 | validation | Uniform | 16 | 1 | 0.16253913 |
| 43 | validation | Uniform | 16 | 4 | 0.29769956 |
| 43 | validation | Uniform | 16 | 16 | 0.40188984 |
| 43 | validation | Uniform | 16 | 64 | 0.47689101 |
| 17 | validation | FIFO | 32 | 1 | 0.16069347 |
| 17 | validation | FIFO | 32 | 4 | 0.29263669 |
| 17 | validation | FIFO | 32 | 16 | 0.39879488 |
| 17 | validation | FIFO | 32 | 64 | 0.47493818 |
| 29 | validation | FIFO | 32 | 1 | 0.16729537 |
| 29 | validation | FIFO | 32 | 4 | 0.29803160 |
| 29 | validation | FIFO | 32 | 16 | 0.39673931 |
| 29 | validation | FIFO | 32 | 64 | 0.47222754 |
| 43 | validation | FIFO | 32 | 1 | 0.16286933 |
| 43 | validation | FIFO | 32 | 4 | 0.29869098 |
| 43 | validation | FIFO | 32 | 16 | 0.40210803 |
| 43 | validation | FIFO | 32 | 64 | 0.47692990 |
| 17 | validation | Reservoir | 32 | 1 | 0.16075017 |
| 17 | validation | Reservoir | 32 | 4 | 0.29275149 |
| 17 | validation | Reservoir | 32 | 16 | 0.39945655 |
| 17 | validation | Reservoir | 32 | 64 | 0.47575811 |
| 29 | validation | Reservoir | 32 | 1 | 0.16598621 |
| 29 | validation | Reservoir | 32 | 4 | 0.29743519 |
| 29 | validation | Reservoir | 32 | 16 | 0.39660011 |
| 29 | validation | Reservoir | 32 | 64 | 0.47155415 |
| 43 | validation | Reservoir | 32 | 1 | 0.16249896 |
| 43 | validation | Reservoir | 32 | 4 | 0.29756499 |
| 43 | validation | Reservoir | 32 | 16 | 0.40236909 |
| 43 | validation | Reservoir | 32 | 64 | 0.47755163 |
| 17 | validation | Uniform | 32 | 1 | 0.16068410 |
| 17 | validation | Uniform | 32 | 4 | 0.29258313 |
| 17 | validation | Uniform | 32 | 16 | 0.39885624 |
| 17 | validation | Uniform | 32 | 64 | 0.47500601 |
| 29 | validation | Uniform | 32 | 1 | 0.16689182 |
| 29 | validation | Uniform | 32 | 4 | 0.29778969 |
| 29 | validation | Uniform | 32 | 16 | 0.39652801 |
| 29 | validation | Uniform | 32 | 64 | 0.47181477 |
| 43 | validation | Uniform | 32 | 1 | 0.16273686 |
| 43 | validation | Uniform | 32 | 4 | 0.29827691 |
| 43 | validation | Uniform | 32 | 16 | 0.40204491 |
| 43 | validation | Uniform | 32 | 64 | 0.47690196 |

## 标签对齐：已做工作与剩余依赖

已核验固定官方HF完整README、0based/inclusive消费代码、官方action-recognition的37个文本Git blob及13次历史变更；未找到原30fps抽帧生产脚本。消费代码不能单独决定原RGB抽帧相位。[官方基线固定版本](https://github.com/assembly-101/assembly101-action-recognition/tree/00d3c3bb3af75ce38ae406614a7cb034aecdebc0)。

固定官方HF发布的5763条路径元数据已完整检查。进一步在HPC用受限range读取72GB AssemblyPoses.zip的658938字节目录，发现3381项中包含逐录像timestamp。仅读取预先批准的9011-c13b…160915开发录像419325字节时间戳元数据，keys0..21275连续，首9881.016/末10235.599秒；未读取pose/像素/标签/封存数据，完整member留HPC。

用户完成本机解锁后，已核验同一批准公钥并恢复原生取证。对同批准C10095 RGB MP4已实际读取container/stream元数据，前后文件SHA稳定：60fps、21277帧、start_time=0、time_base=1/15360；官方timestamp有21276项，其相对60fps时钟残差为[-0.000333333334,0.000666666667]秒。没有嵌入capture timecode；时间戳与视频还存在一帧计数差。这些元数据不能证明哪一个raw frame生成30fps annotation-frame0。没有读取像素/pose/标签数组或封存数据，也没有选择标签join。

另对官方账号7个公开仓库完成目录盘点，新核验download-scripts固定版本01835b08aebe563148bea21a7be7a7674bc432cb的3个文本源码及action-anticipation固定版本7ae5cf6328bc4413a279b2664318dafe2845ce36的10个文本源码，Git blob逐项一致。下载脚本只转移原视频；anticipation消费fps=30的已有特征/标签，没有原RGB→标注抽帧producer。这是有界源码审计，不是所有未公开生产资料不存在的证明。

官方旧pose链接正常浏览器实际AccessDenied，没有发送访问申请。另核验27个官方AssemblyHands toolkit文本Git blob；其visualization/video_visualizer.py声明ANNOT_FPS60、RECTIFIED_VIDEO_FPS30，针对不同手姿标注与rectified视频，不能嫁接为原Assembly101动作抽帧生产证明。[固定源码](https://github.com/facebookresearch/assemblyhands-toolkit/blob/8d4b7f1bce3f99874514199aae7504f78ebf3517/visualization/video_visualizer.py)。

全50688时点在phase0/1/60两个条件假设下的verb union差异为train1234/34816、validation493/15872，合计1727/50688=3.407%。这不是相位证明，不能忽略或挑一个看似更好的join。

24verb readout/NLL、within-activity drift、switch response、different-activity donor和0/1/2/4/8/16恢复均UNAVAILABLE。没有制造供体或用无分层视觉稳定性替代语义门槛。下一项必要证据是权威生产脚本或带原始索引的实际标注帧，可据其与批准开发RGB做精确匹配；现有timestamp及消费者不足以替代。继续只读调查新来源，不能原样重复相位搜索/审计或声称标签已解决。

## 来源与最终核验

28明确compact JSON/小trainingCSV/日志原生before/local/afterSHA一致，见native7251_7268/transfer_integrity_final.json。另3份小型官方目录/timestamp结构/RGB时钟摘要三SHA一致，见primary_sources/timestamp_metadata_native_transfer_integrity.json。七组冻结源、78cache前后SHA、全部6模型数值来源与fit字节已root重算严格等于正式执行证明。大raw CSV/trace/features/weights/videos/labels数组未下载到Mac。

| 证据 | SHA256 |
|---|---|
| native7251_7268/training_report.json | e7da2f91dbc05f9cd5c6ab349c63f60848ca79da8059674db378bc6c62f50b73 |
| native7251_7268/all_models_completion.json | b8f41d7579c92722a7eca0b80f63c80aede6347ae744bf7b47de425b39cdca85 |
| native7251_7268/protocol.json | 07594ef120ac2e32c0e56d22701f8365fda44447089b86bde5cb55e157455222 |
| native7251_7268/source_lock_diagnostic.json | 7ba613fb2c3fb9419059a8bb8035846bc2ec74a4f5ef78b17458c7682db3e379 |
| native7251_7268/visual_health.json | 380c887ff32e2621bac2114477af838e0eebf93f9ef728529eb45bb2450dc89b |
| native7251_7268/fixed_old_reference_comparison.json | 805cd6776b5997dcbc41aa002bfb166ccf6aca4ea627c02b26a62860f285bbdf |
| native7251_7268/root_health_scalar_replay_7268_20261014T0144.json | 4363b5fa89b7c86e8f6760251331ee7d85d3a431a8765e423eaccc4d03124f94 |
| native7251_7268/root_fixed_old_reference_replay_7268_20261014T0144.json | ed58b1292bc026548f5b88f7b6aa347324d897790fae1161bf92b5d2d69abe21 |
| native7251_7268/root_monitor_all6_health_7268_20261014T0144.json | 7aae66dc3cead95c49ec1d3bea59e028a45174f699622d0813020778dcf4a454 |

完整输出哈希如下；当前HPC全部输出字节、大小和SHA已root重新核验等于执行报告，大CSV/trace留HPC；紧凑摘要另外完成原生三SHA传输。

| 已执行输出 | bytes | 执行报告声明SHA256 |
|---|---:|---|
| deletion_candidate_records.csv | 11493578 | ae0dd111c7b9522d5d90d7dbe1ca2929152bb6912e9065c4dfe4925519df369b |
| deletion_summary.json | 13133 | 241e47199a153b625855065ae86c0e27deed207e112aab2b864e37aa9fd6a9ed |
| root_health_scalar_replay_7268_20261014T0144.json | 12091 | 4363b5fa89b7c86e8f6760251331ee7d85d3a431a8765e423eaccc4d03124f94 |
| fixed_old_reference_comparison.json | 18755 | 805cd6776b5997dcbc41aa002bfb166ccf6aca4ea627c02b26a62860f285bbdf |
| health_scalar_records.csv | 17823306 | c17feed30bd23ff932170d22d9b2c80f4a49c23e9880e7c113c113aca8add108 |
| irreversible_replay_trace.csv | 89242659 | c8b47e2d7d3b9e236eba5a8f5634df818b70604c046ce19dbdecf42fcf7a394b |
| root_monitor_all6_health_7268_20261014T0144.json | 31339 | 7aae66dc3cead95c49ec1d3bea59e028a45174f699622d0813020778dcf4a454 |
| visual_health.json | 9442 | 380c887ff32e2621bac2114477af838e0eebf93f9ef728529eb45bb2450dc89b |
| replay_participant_summary.json | 154073 | 0f36627f4f8618eb766ec63018100a3877512519617cd287d2465e7712fca85c |
| replay_scalar_records.csv | 137127625 | c6ae685ec81fddf9db33c58da21863ca1c1926c80aa0d2d477db0ed6fe2d13fb |
| all_models_completion.json | 2074 | b8f41d7579c92722a7eca0b80f63c80aede6347ae744bf7b47de425b39cdca85 |
| root_fixed_old_reference_replay_7268_20261014T0144.json | 6707 | ed58b1292bc026548f5b88f7b6aa347324d897790fae1161bf92b5d2d69abe21 |

最终Slurm/全部文件大小SHA/31824候选数/48CI/812592重放标量/456192固定RNG轨迹已原生独立重放证明。

最终审计SHA：8ff4208ba1ac66d21a36543d7c2c525d5ac80885269ddaa7e48ee36f47a7f7a8。

## 结论与继续执行

本次固定增加正则化的试验未达到原采用门槛；所有种子和失败项保留。GPU训练与全部可用视觉评估、最终模型/源码/缓存/原始标量/保留轨迹取证已完成。临时原生取证连接7271用途完成后由agent释放，实际CANCELLED by3850228/0:0（02:16:37–02:35:05UTC），正常门户Completed；没有活跃科学任务或认证待办。标签语义依赖仍未解决，不能把可用视觉计算完整称作全部语义实验完成。原Gate/Motion REJECTED保持，不能由真实分支救回；sealed/fresh9137确认不启动。继续有依据的原始生产来源调查，不原样重训失败trial、放宽门槛或扫描参数。
