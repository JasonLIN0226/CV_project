# DEV 复现：按预设门槛停止

状态：**未通过，CONF 未读取/评价**。2026-10-26 UTC 17:49:47–17:49:51。

仅执行了原始 DEV 的 multihead 前 256 个 parent 重算。按逐元素 `abs(new-old)/abs(old) <= 1e-4` 检查，sensitivity 的 16 / 768 个单元超限；其他三个数组通过，四个数组的 argmin 均完全一致，没有依赖平局豁免。

| 数组 | 最大绝对差 | 最大逐元素相对差 | argmin 不同数 | 门槛 |
|---|---:|---:|---:|---|
| medoid_sum | 3.58258e-7 | 3.38982e-6 | 0 | 通过 |
| sensitivity | 1.55545e-7 | 2.48315e-3 | 0 | 不通过 |
| attention | 1.78814e-7 | 2.28250e-6 | 0 | 通过 |
| eval_cost | 6.72832e-8 | 7.29026e-7 | 0 | 通过 |

最大相对差发生在 DEV parent 21（`1d45614bd60211e984c50a7e`）候选 2：归档值 `7.776472774821103e-9`，本机值 `7.795782931172149e-9`，绝对差 `1.9310156351045237e-11`。这是已保存数组的只读核查，未重跑推理。未据此增加绝对误差容忍，也未更换数值后端继续尝试。

按用户“任何一项不一致：停下报告，不要打开 CONF”的要求，item_attention 及四组学生全 DEV 预测检查尚未执行；不能把学生门槛写为通过。

## 已完成的前置工作

- 原始 `RES/code/*.py`、学生 ZIP 与 `manifest_sha256.txt` 对应项全部核对通过；原代码保存在 `original_code/`。
- 四个 ZIP 校验 CRC 后解压到 `students/<reader>_<mode>/`，每个目录独立，manifest 在根。没有改模型、normalization 或源 SHA。
- `appearance.py` 重建了原 TRAIN/VAL/DEV 6912 个记录所用的 2208 个独特证据帧，RGB 哈希不一致为 0。
- `pipe.py`、`run_main.py`、`evaluate_dev.py` 的数值函数 AST 与原版一致；Store 的 AST 在两处文件路径表达式替换后完全一致。`experiments/phase2/*` 未修改。

## 路径适配

新增 `RES/code/path_config.py`，以 `QR_REPO/RUN/XFER/AGG/RECORDS/ACTIONS/RGB_INDEX/BASELINE_DECISIONS/APPEARANCE_OUTPUT` 环境变量配置路径。默认均为本仓库的原 DEV 输入。`QR_SPLIT` 控制 evaluator，`QR_SPLITS` 控制 run_main；CONF 配置未执行。evaluate_dev 在非 development split 上不调用缺失的 uniform/diversity 存档，并明确输出 `not_computed`。原有 bootstrap、指标及选择计算未变。

本次环境：Python 3.12.14、NumPy 1.26.4、PyTorch 2.14.1，CPU 2 线程、deterministic algorithms。版本差异尚未被证明是误差原因。

证据：`dev_reproduction.json`、`dev_reproduction.log`、`failed_gate_detail.json`、`path_and_appearance_audit.json`、`initial_manifest_verification.json`、`path_adaptation.diff`、`multihead_development_first256.npz`。
