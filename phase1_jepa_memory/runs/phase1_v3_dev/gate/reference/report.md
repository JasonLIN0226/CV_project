# 开发比较：单项修改结果（不构成 Phase 1 GO）

**DEVELOPMENT_REVISE** · 冻结 I-JEPA · 种子 [17, 29, 43]

同一个预测器比较四种历史输入。所有超参数和推进阈值在训练前固定，checkpoint只由validation选择。

## 四条件预测

表中为预先指定seed的平均normalized MSE；配对准确率表示预测更接近本条轨迹的未来还是sibling未来（chance=50%）。

|任务|测试|h|近期|完整历史|关键B=2|随机B=2|关键历史配对准确率|
|---|---|---:|---:|---:|---:|---:|---:|
|gate|test_id|3|0.10075|0.05096|0.05135|0.08963|92.3%|
|gate|test_id|6|0.09970|0.05710|0.05736|0.09278|92.1%|
|gate|test_long_gap|3|0.10168|0.05082|0.04731|0.09781|97.7%|
|gate|test_long_gap|6|0.09950|0.04935|0.04902|0.09603|97.0%|
|gate|test_distractor_shift|3|0.13659|0.14659|0.14811|0.16873|85.2%|
|gate|test_distractor_shift|6|0.13854|0.15633|0.15761|0.17433|84.0%|

## 关键历史相对近期观测的收益

百分比为误差下降，括号为95%配对置信区间；gate任务指标是lane错误率，motion是位置MSE。区间按完整pair重采样，条件于本次预先指定的seed。

|任务|测试|h|latent误差下降|任务误差下降|错误证据方向|
|---|---|---:|---|---|---|
|gate|test_id|3|49.0% [42.6%, 55.2%]|84.6% [76.0%, 92.7%]|1.585 [1.455, 1.711]|
|gate|test_id|6|42.5% [35.0%, 49.3%]|84.6% [75.8%, 92.7%]|1.573 [1.439, 1.702]|
|gate|test_long_gap|3|53.5% [48.0%, 57.8%]|95.3% [90.4%, 99.2%]|1.599 [1.511, 1.674]|
|gate|test_long_gap|6|50.7% [45.4%, 55.4%]|94.3% [88.0%, 99.0%]|1.645 [1.553, 1.727]|
|gate|test_distractor_shift|3|-8.4% [-14.6%, -2.7%]|71.6% [60.4%, 81.2%]|1.175 [1.013, 1.324]|
|gate|test_distractor_shift|6|-13.8% [-19.3%, -8.5%]|70.1% [59.6%, 79.7%]|1.116 [0.954, 1.266]|

错误证据方向应为正：保持时间/动作/近期输入不变，只换关键历史内容，预测应向sibling的错误未来移动。

## 证据与数据检查

|任务|测试|历史证据probe|近期probe|
|---|---|---:|---:|
|gate|test_id|100.0%|50.0%|
|gate|test_long_gap|100.0%|50.0%|
|gate|test_distractor_shift|100.0%|50.0%|

数据审计：PASS；共1984 pair / 3968 episode。
近期RGB/动作严格相同，历史/可见未来不同，pair不跨split；feature配对相等审计通过。遮挡、提示、真实未来请结合figures/pair_*.png检查。
motion单帧对照、true-target decoder及逐seed完整指标保存在probes/metrics.json、evaluation/metrics.json。

## 训练与开销

两个任务分别训练，同一任务的四种条件共用模型。视觉投影仅在train拟合，训练不读取simulator标签。

实验臂：`reference`；任务：['gate']。所有新实验臂按四条件、全部 horizon 等权 validation normalized MSE 选 checkpoint，平局取较早轮次。
门控候选仅删除排序项；运动候选仅改变两帧选取。各任务的训练目标与选帧规则以 resolved_config.json 中固定实验臂为准。

- gate/17：完成120轮，最佳epoch78，累计训练1066.6秒，best.pt SHA256 `c2a59b114eeb912a9975c6b07e934fef94c937628800d8151f64e4a126e7f9e1`。
- gate/29：完成120轮，最佳epoch43，累计训练1057.9秒，best.pt SHA256 `4f2c9ef7fa95ac77d46dc0d157f340893f76819123826f58dfa50b20efcf8ca0`。
- gate/43：完成120轮，最佳epoch84，累计训练1115.0秒，best.pt SHA256 `d1a00c955dfaf97787cdcd2b9b6cdd250222f418692583f498e8560c6ac0a88c`。
- 数据生成：137.0秒；冻结编码：3108.8秒。
- 累计训练为完整轮次耗时，不含排队、中断等待或未完成轮次；evaluation中checkpoint的training_seconds记录选中模型保存时的累计值。
- 5120维float32每条历史20KiB，B=2为40KiB；本次latency在CPU上测量缓存特征预测器forward，不包含编码或端到端延迟，见evaluation/latency_memory.json。

## 阶段结论

- 记忆收益：8项未通过或不确定，完整切片依据见decision.json。
- full-history：6项未通过或不确定，完整切片依据见decision.json。

GO要求证据可读、latent与任务收益达到原10%阈值且区间支持、错误证据方向正确、ID/OOD与三seed成立。REVISE/INCONCLUSIVE如实保留；不代表历史原则上无用，也不证明方法创新。

只看本报告即可了解主结果。原始评估在evaluation/raw_records.csv，固定方案在resolved_config.json，逐条件训练曲线在predictor/{task}/seed_*/training.csv。figures只展示真实RGB，没有生成未来画面。
