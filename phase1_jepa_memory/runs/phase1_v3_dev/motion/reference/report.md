# 开发比较：单项修改结果（不构成 Phase 1 GO）

**DEVELOPMENT_REVISE** · 冻结 I-JEPA · 种子 [17, 29, 43]

同一个预测器比较四种历史输入。所有超参数和推进阈值在训练前固定，checkpoint只由validation选择。

## 四条件预测

表中为预先指定seed的平均normalized MSE；配对准确率表示预测更接近本条轨迹的未来还是sibling未来（chance=50%）。

|任务|测试|h|近期|完整历史|关键B=2|随机B=2|关键历史配对准确率|
|---|---|---:|---:|---:|---:|---:|---:|
|motion|test_id|3|0.14516|0.07874|0.08342|0.15100|96.6%|
|motion|test_id|6|0.15259|0.07847|0.08174|0.15461|97.8%|
|motion|test_long_gap|3|0.17823|0.13607|0.14279|0.19203|96.9%|
|motion|test_long_gap|6|0.18309|0.15178|0.16208|0.20209|96.6%|
|motion|test_distractor_shift|3|0.28633|0.25414|0.25423|0.31002|85.8%|
|motion|test_distractor_shift|6|0.29085|0.25546|0.25693|0.31392|87.6%|

## 关键历史相对近期观测的收益

百分比为误差下降，括号为95%配对置信区间；gate任务指标是lane错误率，motion是位置MSE。区间按完整pair重采样，条件于本次预先指定的seed。

|任务|测试|h|latent误差下降|任务误差下降|错误证据方向|
|---|---|---:|---|---|---|
|motion|test_id|3|42.5% [40.2%, 44.8%]|82.2% [79.3%, 85.1%]|1.448 [1.400, 1.496]|
|motion|test_id|6|46.4% [44.4%, 48.3%]|84.5% [82.0%, 87.0%]|1.423 [1.381, 1.467]|
|motion|test_long_gap|3|19.9% [16.1%, 23.2%]|73.7% [70.8%, 76.5%]|1.155 [1.094, 1.214]|
|motion|test_long_gap|6|11.5% [7.9%, 14.9%]|71.6% [68.6%, 74.4%]|1.105 [1.042, 1.166]|
|motion|test_distractor_shift|3|11.2% [9.3%, 13.1%]|54.0% [49.3%, 58.2%]|1.059 [1.009, 1.108]|
|motion|test_distractor_shift|6|11.7% [9.3%, 14.1%]|56.9% [52.3%, 61.4%]|1.094 [1.045, 1.138]|

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

实验臂：`reference`；任务：['motion']。所有新实验臂按四条件、全部 horizon 等权 validation normalized MSE 选 checkpoint，平局取较早轮次。
门控候选仅删除排序项；运动候选仅改变两帧选取。各任务的训练目标与选帧规则以 resolved_config.json 中固定实验臂为准。

- motion/17：完成120轮，最佳epoch35，累计训练1123.8秒，best.pt SHA256 `dba52d80a90e86b24db1b58a1835558c83a8fdc1563ccdc9b06c8b28ed397cf2`。
- motion/29：完成120轮，最佳epoch52，累计训练1081.8秒，best.pt SHA256 `cb25f0a404598b6ca50fd5f44934877d2a3591373f44ee12e814d47cb3e3054f`。
- motion/43：完成120轮，最佳epoch25，累计训练1086.8秒，best.pt SHA256 `60c486d8d0659f3943e41d188aaf7c16c43d344b134f6062a581e294ef199931`。
- 数据生成：137.0秒；冻结编码：3108.8秒。
- 累计训练为完整轮次耗时，不含排队、中断等待或未完成轮次；evaluation中checkpoint的training_seconds记录选中模型保存时的累计值。
- 5120维float32每条历史20KiB，B=2为40KiB；本次latency在CPU上测量缓存特征预测器forward，不包含编码或端到端延迟，见evaluation/latency_memory.json。

## 阶段结论

- 记忆收益：2项未通过或不确定，完整切片依据见decision.json。
- 区间：1项未通过或不确定，完整切片依据见decision.json。

GO要求证据可读、latent与任务收益达到原10%阈值且区间支持、错误证据方向正确、ID/OOD与三seed成立。REVISE/INCONCLUSIVE如实保留；不代表历史原则上无用，也不证明方法创新。

只看本报告即可了解主结果。原始评估在evaluation/raw_records.csv，固定方案在resolved_config.json，逐条件训练曲线在predictor/{task}/seed_*/training.csv。figures只展示真实RGB，没有生成未来画面。
