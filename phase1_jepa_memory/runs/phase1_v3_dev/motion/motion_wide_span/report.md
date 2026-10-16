# 开发比较：单项修改结果（不构成 Phase 1 GO）

**DEVELOPMENT_PASS** · 冻结 I-JEPA · 种子 [17, 29, 43]

同一个预测器比较四种历史输入。所有超参数和推进阈值在训练前固定，checkpoint只由validation选择。

## 四条件预测

表中为预先指定seed的平均normalized MSE；配对准确率表示预测更接近本条轨迹的未来还是sibling未来（chance=50%）。

|任务|测试|h|近期|完整历史|关键B=2|随机B=2|关键历史配对准确率|
|---|---|---:|---:|---:|---:|---:|---:|
|motion|test_id|3|0.14472|0.07918|0.07808|0.15167|97.9%|
|motion|test_id|6|0.15306|0.07965|0.07905|0.15586|99.1%|
|motion|test_long_gap|3|0.17954|0.13589|0.13307|0.19253|98.8%|
|motion|test_long_gap|6|0.18547|0.15081|0.15021|0.20414|98.7%|
|motion|test_distractor_shift|3|0.29866|0.25238|0.25105|0.29492|88.2%|
|motion|test_distractor_shift|6|0.30299|0.25421|0.25361|0.30045|92.1%|

## 关键历史相对近期观测的收益

百分比为误差下降，括号为95%配对置信区间；gate任务指标是lane错误率，motion是位置MSE。区间按完整pair重采样，条件于本次预先指定的seed。

|任务|测试|h|latent误差下降|任务误差下降|错误证据方向|
|---|---|---:|---|---|---|
|motion|test_id|3|46.1% [44.0%, 48.1%]|85.7% [83.6%, 87.6%]|1.481 [1.443, 1.521]|
|motion|test_id|6|48.4% [46.4%, 50.2%]|87.4% [85.4%, 89.2%]|1.462 [1.420, 1.503]|
|motion|test_long_gap|3|25.9% [23.1%, 28.8%]|79.0% [76.7%, 81.2%]|1.217 [1.162, 1.272]|
|motion|test_long_gap|6|19.0% [16.0%, 22.0%]|76.9% [74.5%, 79.1%]|1.180 [1.119, 1.236]|
|motion|test_distractor_shift|3|15.9% [14.1%, 17.7%]|47.3% [42.3%, 52.1%]|1.097 [1.042, 1.149]|
|motion|test_distractor_shift|6|16.3% [14.2%, 18.4%]|54.5% [50.1%, 58.9%]|1.169 [1.116, 1.218]|

错误证据方向应为正：保持时间/动作/近期输入不变，只换关键历史内容，预测应向sibling的错误未来移动。

## 证据与数据检查

|任务|测试|历史证据probe|近期probe|
|---|---|---:|---:|
|motion|test_id|100.0%|50.0%|
|motion|test_long_gap|100.0%|50.0%|
|motion|test_distractor_shift|100.0%|50.0%|

数据审计：PASS；共1984 pair / 3968 episode。
近期RGB/动作严格相同，历史/可见未来不同，pair不跨split；feature配对相等审计通过。遮挡、提示、真实未来请结合figures/pair_*.png检查。
motion单帧对照、true-target decoder及逐seed完整指标保存在probes/metrics.json、evaluation/metrics.json。

## 训练与开销

两个任务分别训练，同一任务的四种条件共用模型。视觉投影仅在train拟合，训练不读取simulator标签。

实验臂：`motion_wide_span`；任务：['motion']。所有新实验臂按四条件、全部 horizon 等权 validation normalized MSE 选 checkpoint，平局取较早轮次。
门控候选仅删除排序项；运动候选仅改变两帧选取。各任务的训练目标与选帧规则以 resolved_config.json 中固定实验臂为准。

- motion/17：完成120轮，最佳epoch34，累计训练1114.3秒，best.pt SHA256 `87b0ce6c360b2250732707c9f20d51ea5b06c1d232b0e7ec7c1367433c4b0bca`。
- motion/29：完成120轮，最佳epoch48，累计训练1089.5秒，best.pt SHA256 `7350f14b1118ea208934433fcb9bdbef34199debb95445051dcbf28c2b543f51`。
- motion/43：完成120轮，最佳epoch49，累计训练1108.7秒，best.pt SHA256 `ad273333769a8dc20f16ba2eda8637a969e5e0b8819cc5d1411e174b64ca3fbb`。
- 数据生成：137.0秒；冻结编码：3108.8秒。
- 累计训练为完整轮次耗时，不含排队、中断等待或未完成轮次；evaluation中checkpoint的training_seconds记录选中模型保存时的累计值。
- 5120维float32每条历史20KiB，B=2为40KiB；本次latency在CPU上测量缓存特征预测器forward，不包含编码或端到端延迟，见evaluation/latency_memory.json。

## 阶段结论

- 当前实验臂满足任务内的原检查；仍须独立开发比较支持候选后才能进入完整确认。

GO要求证据可读、latent与任务收益达到原10%阈值且区间支持、错误证据方向正确、ID/OOD与三seed成立。REVISE/INCONCLUSIVE如实保留；不代表历史原则上无用，也不证明方法创新。

只看本报告即可了解主结果。原始评估在evaluation/raw_records.csv，固定方案在resolved_config.json，逐条件训练曲线在predictor/{task}/seed_*/training.csv。figures只展示真实RGB，没有生成未来画面。
