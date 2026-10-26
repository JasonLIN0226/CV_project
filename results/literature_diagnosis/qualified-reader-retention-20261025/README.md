# qualified-reader-retention-20261025

**2026-10-26 更新：**[一次性 CONF 已运行](conf/README.md)，三条预注册主张全部通过，平局平均未改变判定。A100 job12374 只生成/编码新CONF，冻结预测与评价在CPU完成，无训练或调参。完整二进制数据、学生与复现数组见 [Release](https://github.com/JasonLIN0226/CV_project/releases/tag/qualified-reader-retention-confirmation-20261026)。`manifest_sha256.txt` 保留原DEV包清单；本轮代码路径适配和新增产物另由执行锁、完成凭证及发布清单记录。下面是原DEV包的历史说明，包含当时的“CONF未打开”状态。

合格 reader（multihead 主、item_attention 次）上的终局 R/D/S 保留比较，DEV 开发结果。报告见 `docs/QUALIFIED_READER_RETENTION_DEV_ZH.md`。全部在 CPU 上完成，未使用 GPU，CONF 未打开。

- `locks/`：运行前协议与哈希（先于任何标签/学生/16 计划结果写定）；实际计划近似版的单独锁定。
- `code/`：`pipe.py`（复现终局运行的 reader 输入、学生输入与选择器；DEV 系数与 A100 存档差 ≤3.0e-4）、`run_main.py`（阶段 A：标签、选择器、oracle）、`run_students.py`（调用原样的 `fit_terminal_students.py`）、`evaluate_dev.py`、`appearance.py`（再生成证据帧 RGB，0 个哈希不一致）、`score_qualified_rules.py`、`diagnose_failures.py`。脚本里的绝对路径指向分析时的临时目录，复跑时需改为本地路径。
- `stageA/`：每个 reader、每个 split 的逐 parent 数组（16 计划 medoid/敏感度分数、TRAIN 删除代价与 future-attention 标签、VAL/DEV 32 计划期望损失、query 处 attention）。
- `students/`：4 组学生（2 reader × standard/distilled）的全部 checkpoint、prefit manifest、学习曲线、DEV 预测（zip）。
- `eval/`：DEV 评价（含 family bootstrap 区间、分外观条件计数）；实际计划近似版首轮汇总。
- `conf/`：CONF 一次性确认的作业脚本与要确认的三条主张，**尚未运行**。
- `manifest_sha256.txt`：本目录全部文件的 SHA256 与字节数。

输入：本地 `build/terminal-download-20261025/extracted/run/data/features.npy`（SHA256 `9090806a…e6b1`）按 TRAIN basis 投影后的 `build/_claude_transfer_20261025/projected_features.npy`（`8d3591e0…1552`）与 `output_basis_coords.npy`（`5970f4f2…a755`）；reader checkpoint 来自 Release `reader-aggregation-20261025`。
