# 第一阶段实验结果

**REVISE** · 冻结 I-JEPA · 种子 [17, 29, 43]

同一个预测器比较四种历史输入。所有超参数和推进阈值在训练前固定，checkpoint只由validation选择。

## 四条件预测

表中为预先指定seed的平均normalized MSE；配对准确率表示预测更接近本条轨迹的未来还是sibling未来（chance=50%）。

|任务|测试|h|近期|完整历史|关键B=2|随机B=2|关键历史配对准确率|
|---|---|---:|---:|---:|---:|---:|---:|
|gate|test_id|3|0.10513|0.04979|0.05014|0.09149|93.4%|
|gate|test_id|6|0.10468|0.05860|0.05952|0.09452|92.7%|
|gate|test_long_gap|3|0.10557|0.04950|0.04604|0.09966|98.4%|
|gate|test_long_gap|6|0.10444|0.05033|0.05027|0.09770|97.7%|
|gate|test_distractor_shift|3|0.12616|0.10526|0.10537|0.12754|86.8%|
|gate|test_distractor_shift|6|0.12788|0.11346|0.11313|0.13204|86.2%|
|motion|test_id|3|0.15796|0.08118|0.08560|0.15936|99.6%|
|motion|test_id|6|0.16306|0.08046|0.08350|0.15986|99.5%|
|motion|test_long_gap|3|0.17970|0.13121|0.14935|0.18892|98.8%|
|motion|test_long_gap|6|0.18406|0.14371|0.16748|0.19747|98.4%|
|motion|test_distractor_shift|3|0.25172|0.19392|0.19695|0.25904|97.3%|
|motion|test_distractor_shift|6|0.25378|0.19108|0.19383|0.25676|97.4%|

## 关键历史相对近期观测的收益

百分比为误差下降，括号为95%配对置信区间；gate任务指标是lane错误率，motion是位置MSE。区间按完整pair重采样，条件于本次预先指定的seed。

|任务|测试|h|latent误差下降|任务误差下降|错误证据方向|
|---|---|---:|---|---|---|
|gate|test_id|3|52.3% [46.3%, 57.7%]|86.5% [78.9%, 93.5%]|1.586 [1.468, 1.700]|
|gate|test_id|6|43.1% [36.3%, 49.4%]|85.2% [76.3%, 93.2%]|1.577 [1.444, 1.701]|
|gate|test_long_gap|3|56.4% [52.4%, 59.6%]|96.6% [92.2%, 99.7%]|1.654 [1.575, 1.720]|
|gate|test_long_gap|6|51.9% [47.3%, 55.9%]|95.6% [90.1%, 99.7%]|1.703 [1.619, 1.780]|
|gate|test_distractor_shift|3|16.5% [9.2%, 22.9%]|73.2% [62.5%, 82.6%]|1.278 [1.104, 1.430]|
|gate|test_distractor_shift|6|11.5% [5.3%, 17.4%]|73.2% [62.5%, 82.8%]|1.258 [1.092, 1.411]|
|motion|test_id|3|45.8% [44.2%, 47.4%]|85.7% [83.9%, 87.4%]|1.579 [1.542, 1.616]|
|motion|test_id|6|48.8% [47.3%, 50.3%]|88.3% [86.6%, 89.9%]|1.562 [1.524, 1.601]|
|motion|test_long_gap|3|16.9% [13.0%, 20.2%]|82.1% [80.3%, 83.8%]|1.286 [1.239, 1.335]|
|motion|test_long_gap|6|9.0% [5.2%, 12.5%]|80.2% [78.3%, 81.9%]|1.238 [1.185, 1.292]|
|motion|test_distractor_shift|3|21.8% [20.1%, 23.4%]|49.5% [43.7%, 54.5%]|1.261 [1.222, 1.297]|
|motion|test_distractor_shift|6|23.6% [21.9%, 25.4%]|56.0% [50.9%, 60.5%]|1.295 [1.254, 1.332]|

错误证据方向应为正：保持时间/动作/近期输入不变，只换关键历史内容，预测应向sibling的错误未来移动。

## 证据与数据检查

|任务|测试|历史证据probe|近期probe|
|---|---|---:|---:|
|gate|test_id|100.0%|50.0%|
|gate|test_long_gap|100.0%|50.0%|
|gate|test_distractor_shift|100.0%|50.0%|
|motion|test_id|100.0%|50.0%|
|motion|test_long_gap|100.0%|50.0%|
|motion|test_distractor_shift|100.0%|50.0%|

数据审计：PASS；共1984 pair / 3968 episode。
近期RGB/动作严格相同，历史/可见未来不同，pair不跨split；feature配对相等审计通过。遮挡、提示、真实未来请结合figures/pair_*.png检查。
motion单帧对照、true-target decoder及逐seed完整指标保存在probes/metrics.json、evaluation/metrics.json。

## 训练与开销

两个任务分别训练，同一任务的四种条件共用模型。训练目标是latent MSE、配对差分MSE与配对排序损失；validation使用同一目标。视觉投影仅在train拟合，训练不读取simulator标签。

- gate/17：完成120轮，最佳epoch12，累计训练1071.5秒，best.pt SHA256 `e3b8b10bebfb29de75aa4585a91685fbe7206dbb8c54ab364584723853228cf7`。
- motion/17：完成120轮，最佳epoch12，累计训练1036.6秒，best.pt SHA256 `faffcce5ef1b2978a8546d13152f2d8347dfd9d55e87c70143391b957d3746ae`。
- gate/29：完成120轮，最佳epoch13，累计训练1038.7秒，best.pt SHA256 `460b34e67d9d7c621b2d1edc19863c140bd98b36ccad5f8f789f4a42e017b876`。
- motion/29：完成120轮，最佳epoch10，累计训练1063.7秒，best.pt SHA256 `3af5c7449a86f16160f30121df25cff6ff15c6538f0e907d9569d6c0cd15205d`。
- gate/43：完成120轮，最佳epoch48，累计训练1042.4秒，best.pt SHA256 `e208f2f5ef75da83f1ad597b4e93b886cf4fc65261b012c8d19ae72ed8813e44`。
- motion/43：完成120轮，最佳epoch11，累计训练1050.3秒，best.pt SHA256 `6e47f3286e3984805f60a67dceb5667a8564f1afc28a84439b32c993d2b23c53`。
- 数据生成：137.0秒；冻结编码：3108.8秒。
- 累计训练为完整轮次耗时，不含排队、中断等待或未完成轮次；evaluation中checkpoint的training_seconds记录选中模型保存时的累计值。
- 5120维float32每条历史20KiB，B=2为40KiB；本次latency在CPU上测量缓存特征预测器forward，不包含编码或端到端延迟，见evaluation/latency_memory.json。

## 阶段结论

- 记忆收益：2项未通过或不确定，完整切片依据见decision.json。
- 区间：3项未通过或不确定，完整切片依据见decision.json。

GO要求证据可读、latent与任务收益达到原10%阈值且区间支持、错误证据方向正确、ID/OOD与三seed成立。REVISE/INCONCLUSIVE如实保留；不代表历史原则上无用，也不证明方法创新。

只看本报告即可了解主结果。原始评估在evaluation/raw_records.csv，固定方案在resolved_config.json，逐条件训练曲线在predictor/{task}/seed_*/training.csv。figures只展示真实RGB，没有生成未来画面。
