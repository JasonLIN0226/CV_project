# 第一阶段实验结果

**REVISE** · 冻结 I-JEPA · 种子 [17, 29, 43]

同一个预测器比较四种历史输入。所有超参数和推进阈值在训练前固定，checkpoint只由validation选择。

## 四条件预测

表中为预先指定seed的平均normalized MSE；配对准确率表示预测更接近本条轨迹的未来还是sibling未来（chance=50%）。

|任务|测试|h|近期|完整历史|关键B=2|随机B=2|关键历史配对准确率|
|---|---|---:|---:|---:|---:|---:|---:|
|gate|test_id|3|0.23373|0.23346|0.23337|0.23334|50.1%|
|gate|test_id|6|0.22503|0.22524|0.22537|0.22504|50.4%|
|gate|test_long_gap|3|0.22377|0.22400|0.22347|0.22390|49.9%|
|gate|test_long_gap|6|0.22198|0.22277|0.22224|0.22185|50.1%|
|gate|test_distractor_shift|3|0.28190|0.27656|0.27779|0.27895|50.1%|
|gate|test_distractor_shift|6|0.27956|0.27363|0.27377|0.27639|50.0%|
|motion|test_id|3|0.26150|0.22305|0.22311|0.25245|79.4%|
|motion|test_id|6|0.25721|0.21263|0.21262|0.24369|80.6%|
|motion|test_long_gap|3|0.28968|0.35360|0.33962|0.29357|35.0%|
|motion|test_long_gap|6|0.29496|0.36253|0.34951|0.30130|36.1%|
|motion|test_distractor_shift|3|0.34762|0.31440|0.31631|0.34475|78.1%|
|motion|test_distractor_shift|6|0.34878|0.30957|0.31140|0.34240|77.2%|

## 关键历史相对近期观测的收益

百分比为误差下降，括号为95%配对置信区间；gate任务指标是lane错误率，motion是位置MSE。区间按完整pair重采样，条件于本次预先指定的seed。

|任务|测试|h|latent误差下降|任务误差下降|错误证据方向|
|---|---|---:|---|---|---|
|gate|test_id|3|0.2% [-0.2%, 0.5%]|-3.4% [-6.8%, 0.0%]|-0.001 [-0.002, -0.000]|
|gate|test_id|6|-0.2% [-0.4%, 0.1%]|-3.1% [-6.5%, 0.3%]|-0.001 [-0.002, -0.000]|
|gate|test_long_gap|3|0.1% [-0.2%, 0.4%]|-1.6% [-4.2%, 1.0%]|0.000 [-0.001, 0.001]|
|gate|test_long_gap|6|-0.1% [-0.5%, 0.2%]|-2.3% [-4.4%, -0.3%]|0.000 [-0.001, 0.001]|
|gate|test_distractor_shift|3|1.5% [0.8%, 2.2%]|-0.8% [-2.3%, 0.8%]|-0.000 [-0.002, 0.001]|
|gate|test_distractor_shift|6|2.1% [1.3%, 2.9%]|-1.0% [-2.6%, 0.5%]|0.000 [-0.001, 0.002]|
|motion|test_id|3|14.7% [13.0%, 16.3%]|53.5% [49.3%, 57.4%]|0.957 [0.914, 0.998]|
|motion|test_id|6|17.3% [15.5%, 18.9%]|54.0% [50.2%, 57.5%]|0.927 [0.884, 0.967]|
|motion|test_long_gap|3|-17.2% [-19.9%, -14.7%]|-35.2% [-40.6%, -30.0%]|-0.307 [-0.361, -0.254]|
|motion|test_long_gap|6|-18.5% [-20.9%, -16.1%]|-32.3% [-37.2%, -27.6%]|-0.297 [-0.348, -0.247]|
|motion|test_distractor_shift|3|9.0% [7.8%, 10.2%]|49.7% [45.0%, 53.9%]|0.855 [0.810, 0.896]|
|motion|test_distractor_shift|6|10.7% [9.5%, 11.9%]|49.1% [44.8%, 53.0%]|0.826 [0.783, 0.866]|

错误证据方向应为正：保持时间/动作/近期输入不变，只换关键历史内容，预测应向sibling的错误未来移动。

## 证据与数据检查

|任务|测试|历史证据probe|近期probe|
|---|---|---:|---:|
|gate|test_id|100.0%|50.0%|
|gate|test_long_gap|100.0%|50.0%|
|gate|test_distractor_shift|100.0%|50.0%|
|motion|test_id|94.5%|50.0%|
|motion|test_long_gap|93.4%|50.0%|
|motion|test_distractor_shift|51.2%|50.0%|

数据审计：PASS；共1408 pair / 2816 episode。
近期RGB/动作严格相同，历史/可见未来不同，pair不跨split；feature配对相等审计通过。遮挡、提示、真实未来请结合figures/pair_*.png检查。
motion单帧对照、true-target decoder及逐seed完整指标保存在probes/metrics.json、evaluation/metrics.json。

## 训练与开销

训练目标是latent MSE加配对差分MSE。每轮四种条件均衡覆盖所有训练pair；validation使用同一目标。训练不读取simulator标签。

- seed17：完成120轮，最佳epoch61，累计训练1236.7秒，best.pt SHA256 `da3ab9697914da57211a8783768a794f000980a28e94688510fde1f529574d63`。
- seed29：完成120轮，最佳epoch119，累计训练1253.7秒，best.pt SHA256 `30a7e6ebc330ca4da41dcb76704250fdd271a4cb13ab730a98c9f7a0ec711069`。
- seed43：完成120轮，最佳epoch85，累计训练1273.3秒，best.pt SHA256 `059ac692de44ff235dc906afe8157c7dd0bf58e33a5bfc301b4cb92f7e2e44a6`。
- 数据生成：98.6秒；冻结编码：2291.3秒。
- 累计训练为完整轮次耗时，不含排队、中断等待或未完成轮次；evaluation中checkpoint的training_seconds记录选中模型保存时的累计值。
- 5120维float32每条历史20KiB，B=2为40KiB；本次latency在CPU上测量缓存特征预测器forward，不包含编码或端到端延迟，见evaluation/latency_memory.json。

## 阶段结论

- 证据：1项未通过或不确定，完整切片依据见decision.json。
- 记忆收益：36项未通过或不确定，完整切片依据见decision.json。
- 内容替换：30项未通过或不确定，完整切片依据见decision.json。
- full-history：6项未通过或不确定，完整切片依据见decision.json。
- 区间：1项未通过或不确定，完整切片依据见decision.json。

GO要求证据可读、latent与任务收益达到原10%阈值且区间支持、错误证据方向正确、ID/OOD与三seed成立。REVISE/INCONCLUSIVE如实保留；不代表历史原则上无用，也不证明方法创新。

只看本报告即可了解主结果。原始评估在evaluation/raw_records.csv，固定方案在resolved_config.json，逐条件训练曲线在predictor/seed_*/training.csv。figures只展示真实RGB，没有生成未来画面。
