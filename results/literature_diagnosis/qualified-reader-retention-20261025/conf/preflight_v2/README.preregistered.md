# CONF 一次性确认：按修订预检门槛继续，CONF 尚未打开

**状态（2026-10-26，修订前置登记）：A100 未提交；CONF 未生成、未编码、未评价。** 用户明确授权保留原停止记录，替换数值复现门槛并继续。下面先登记修订内容和原因，再执行修订版 DEV 预检；三条科学主张不变。

## 预检修订 v2 与原因（在继续计算和打开 CONF 前登记）

来源：用户于 2026-10-26 的明确指令。原逐元素相对误差会放大接近零的敏感度误差：敏感度是两个几乎相同预测之差的平方。已有 DEV parent21 的三个归档候选值约为 `0.093 / 0.012 / 7.8e-9`，原检查测到近零候选的浮点抵消敏感性，而删除选择全部一致。因此改为同一 parent 的归档尺度判定，并单独严格检查选择。原停止报告及原始数组全部保留，不把原记录改写成通过；本修订不根据 CONF 结果制定。

修订仅替换预检，模型、学生、输入、数值计算、checkpoint、三条主张及 bootstrap（seed20261107、2000 次）均不改：

1. **sensitivity / medoid_sum / eval_cost：**逐 parent 定义 `tol = 1e-4 * max(old[0:3])`。每个候选都须满足 `abs(new-old) <= tol`。argmin 必须一致；唯一豁免是该 parent 归档第二小值减最小值 `<= tol`。报告归档平局 parent 数、实际使用豁免数及未豁免 mismatch 数。
2. **attention：**数值仍要求 `abs(new-old) <= 1e-4 * max(old[0:3])`。argmin 必须一致；唯一豁免为归档 `log(second_smallest / smallest) <= 1e-4`，报告平局及实际豁免数。完全相等的正权重属于平局；若遇到非正权重或非有限数值，停止报告，不擅自定义额外豁免。
3. **四组学生、全量 DEV：**逐 parent 定义 `tol = 1e-4 * max(1, max(abs(old_logits)))`，所有 logits 都须满足 `abs(new-old) <= tol`。drop 必须一致；唯一豁免为归档 logits 前两名之差 `<= tol`。报告归档平局数、实际豁免数与未豁免 mismatch 数。standard 使用再生并通过哈希检查的 appearance，distilled 传 None，不训练。

执行顺序：先读取已保存 `multihead_development_first256.npz` 按新门槛重判，**不重新推理 multihead**；再重算 item_attention 前256个 parent，再预测四组学生的全量DEV。任何一项不过仍立即停止，不打开CONF。通过后才从原流程第1步继续A100生成与编码。

## 追加平局稳健性报告（同样在 CONF 前登记）

- 对 `full_sensitivity`、`medoid`、`oracle`，逐行把 `value - min(value) <= 1e-4 * max(value)` 的候选视为平局集合；报告集合大小大于1的 parent 数。
- 平局平均丢失率：每个 parent 在平局集合中错误删除的候选比例，再按原评价口径聚合；保持与 argmin 同样的 family 单位。逐 reader、逐 R/D/S 报告 argmin 与平局平均及差值。
- **三条主张仍以原 DEV 同样的 argmin 评价作为正式判定。** 稳健性版本只替换上述三种规则的丢失率，学生及 attention 保持原判定；若某条主张的真值翻转，标为“不稳健”，不改正式通过/不通过结论。
- 在打开 CONF 前，用已有 DEV 的完整 stageA 数组核验并记录参照：用户报告两个 reader、三族、上述三种方法共18格丢失率在平局平均下均不变；本轮重算该只读参照，若不一致如实报告，不改阈值。

以上修订记录先落盘；执行代码、修订后DEV凭证与CONF开启凭证将记录相应文件哈希。

## 原停止记录（保留，不改写）

- multihead 前 256 个 DEV parent 的 `sensitivity` 有 16/768 个单元超过逐元素相对误差 `1e-4`，最大为 `0.002483151026204096`；四组数组的 argmin 全部一致，其他三个数组数值门槛通过。
- 最大相对误差单元的归档/重算值为 `7.776472774821103e-9` / `7.795782931172149e-9`，绝对差 `1.9310156351045237e-11`。未使用小绝对差来豁免既定相对误差门槛；误差原因尚未确认。
- item_attention 的 DEV 重算及四组学生的完整 DEV 预测尚未执行，不能写为通过。appearance 再生 2208 个独特证据帧，RGB 哈希不一致为 0。
- 集群仅上传了编码脚本、sbatch 和只读检查脚本，尚未执行远端 inspect 或 sbatch；无 A100 job id、无 CONF ZIP SHA。没有删除任何本地或集群文件。
- [复现报告](preflight/dev_reproduction.json)、[失败单元](preflight/failed_gate_detail.json)、[路径适配 diff](preflight/path_adaptation.diff)、[完整停止说明](preflight/DEV_REPRODUCTION_STATUS.md)。原始代码与重算数组保存在本地 `build/qualified-reader-conf-20261026/dev-preflight/`。
- `conf_integrity.py`、`conf_predict.py`、`conf_claims.py` 是本轮准备的草稿，未执行且未完成最终审查，不作为已验证的确认入口。

## 要确认的三条主张（在看 CONF 之前写定）
1. S 协同：集合条件删除代价学生（3 seed 平均）的必要证据丢失率低于完整集合敏感度、按头平均 attention 和集合 future-attention 学生；两个 reader 各自成立。
2. 完整集合敏感度在 R/D 上的丢失率与 oracle 相差不超过 10 个百分点（item_attention reader）。
3. medoid 在 D 与 S 上的丢失率高于完整集合敏感度（两个 reader 各自成立），即其失败机制在新场景重复。

所有学生 checkpoint、reader、规则、标签与评价代码保持 DEV 时的版本（哈希见上级目录 `manifest.json`）。

## 步骤
1. 把本目录的 `conf_encode_project.py` 上传到集群 `/scratch/xy3257/cv-phase2-20261020/qualified-reader-conf-job/`。
2. `sbatch conf.sbatch`。脚本复用终局运行冻结的 bundle、venv 与 I-JEPA 缓存：生成 CONF 记录，用 TRAIN normalization 编码，按冻结 basis 投影，打包成 `conf_transfer.zip`（约 30–40 MB）。
3. 下载 `conf_transfer.zip`，在 CPU 上用本目录上级的 `pipe.py` / `run_main.py` / `evaluate_dev.py` 评价冻结方法。
