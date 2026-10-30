# Motion v2 / v2.1 执行报告（2026-10-30）

**最新状态：v2.1 第 0′ 步原样 CPU 复现完成，但两项数值与附件不同，按已登记规则停止。** 第 1 步生成器和第 2 步预测器均未启动。这不是预测器统计判据失败，也不是新生成的 train 数据审计失败。

## v2.1 本轮执行与登记

v2.1 修订明确允许 `t_c` 处的位置跳变，并以共同 `x_end` 锚定新段，已解决上一轮终点不一致问题。新增三个附件文件原样导入，SHA256 全部与预期一致；此前五个附件文件保持逐字节不变。

运行前已提交并推送 protocol：`5ee647a7b9234b8a127d80d87f38c1eebcedb30e`。登记文件为 [protocol_step0_v2_1.json](../experiments/motion_v2/protocol_step0_v2_1.json)，SHA256 为 `84103cf885a9928453623d1cdeadeb897598949cdcf547bf1d4cae6897c5ba7a`。其中冻结了源文件哈希、依赖版本、参数、运行顺序和“解析后的 JSON 必须完全相同，否则停止，不改参数／版本重跑”。

实际原脚本命令（经一次性 runner 记录输出）：

```sh
python -I experiments/motion_v2/analysis/motion_v2_1_checks.py 1000
```

仅执行一次，退出码 0；CPU 墙钟用时 **44.406 秒**，实际 **0 A100-GPU 秒**。环境为 macOS arm64、Python 3.12.14、NumPy 1.26.4、Pillow 11.3.0、scikit-learn 1.7.2、SciPy 1.17.1；完整依赖见 [requirements-cpu.lock](../experiments/motion_v2/requirements-cpu.lock)。所有原始输出、stderr、差异及状态均保留在 [本轮结果目录](../results/motion_v2/step0_v2_1/v2_1_reference)。

## 复现结果与停止原因

| 跨场景分类项目（long gap） | v1 参考 | v1 实际 | v2.1 参考 | v2.1 实际 |
|---|---:|---:|---:|---:|
| 只有终点帧 | 50.0% | 50.0% | 50.0% | 50.0% |
| 一张非终点新段帧 | 62.4% | **62.0%** | 53.8% | **54.5%** |
| 旧段帧 + 终点帧 | 不适用 | 不适用 | 50.0% | 50.0% |
| 两张新段帧 | 99.6% | 99.6% | 99.3% | 99.3% |

仅以上两处单帧准确率不同：v1 差 −0.4 个百分点，v2.1 差 +0.7 个百分点。参见 [comparison.json](../results/motion_v2/step0_v2_1/v2_1_reference/comparison.json)。

- 参考脚本的四组配对审计（v1/v2.1 × ordinary/long gap）全部 **500/500**，与附件完全一致。
- 所有 online B=2 策略数字均完全一致；v2.1 的 FIFO、keep-first 都为 0，充分帧标签上限及结构参照均为 100%。
- v2.1 的 54.5% 仍低于修订的 58% 单帧阈值，两帧为 99.3%。但这不能替代本步骤要求的严格参考复现，也不能记作尚未生成的真实 train 数据通过第 1 步审计。

**按用户“不一致即停”和事先提交的 protocol，当前停止，不放宽匹配标准，不换依赖或 seed 重跑。** 后续原 Phase 1 headroom 脚本、manifest 逐对核对、第 1 步及第 2 步均未运行。

附件没有提供原始依赖版本；其分类器也没有显式设置 `random_state`。这些是复现时需要记录的条件，**本次没有证据证明它们导致了上述差异**，也没有通过试跑寻找能匹配参考值的环境。

## 原始 seed 9137 manifest 的定位状态

本地候选 `results/phase1_independent_confirmation/execution_v7_metadata/native_v6_evidence/fresh/dataset/manifest.json` 已仅按 JSON 元数据检查：共 2,432 条，Gate/Motion 各 train 1,024 条、validation 192 条，没有三个测试 split。没有打开记录指向的图像、NPZ 或特征文件。该候选不能完成原测试池的 pair id/query/gap 核对。

已检查本地 build/results 中 127 个 manifest JSON 和 19 个 ZIP 内的 33 个 manifest JSON，没有找到完整原始副本。仓库配置与 staged_data.py 指向以下历史 HPC 候选，当前仍未验证文件存在或取得内容：

- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/dataset/manifest.json`
- `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/locked_test_generation/dataset/manifest.json`
- 完整文件的历史校验回执：`/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/confirmation_test_generation_complete.json`

Chrome 无痕 Dashboard 当时显示已登录 `xy3257`，但进入 Files 后跳转至 Microsoft 登录流程并停在“Working…”，未进入文件列表；SSH 非交互认证也未成功。因此不能说已完成 HPC 文件核查。CPU 复现随后已触发独立停止条件，本轮不再处理登录或启动任何作业。

## 交付状态与风险

| 用户要求 | 当前结果 |
|---|---|
| 附件完整性 | 原五文件与新增三文件均保留原字节，8/8 预期 SHA256 一致 |
| 第 0′ 步 v2.1 CPU 复现 | 运行一次；两项分类准确率不匹配，停止 |
| 原第 0 步 Phase 1 headroom / manifest 对照 | 未运行；完整原始 manifest 尚待取得 |
| 第 1 步真实生成器 / train 审计 | 未实现、未运行；参考脚本的 500/500 不算此步骤通过 |
| 第 2 步 (a) recent-only | h=3/6 特征 MSE、位置误差及配对 95% CI 均无结果 |
| 第 2 步 (b) 两张正确帧 | 同上，无结果 |
| 第 2 步 (c) 两张旧段帧 | 同上，无结果 |
| 第 2 步是否过关 | 未评估 |
| GPU | 未进入 Step 2 估时登记；实际 0 A100-GPU 秒 |
| test / 保留策略 | 未打开或生成实验 test 数据；未训练保留策略 |

最大风险是跨环境参考复现尚不一致，且隐藏状态的高可辨识性尚未证明能传递到真实 RGB/I-JEPA 特征和预测器。当前结果支持修订代理的配对结构和定性 headroom，但不足以给真实生成器或第 2 步判定通过。

当前机器可读总状态见 [status.json](../results/motion_v2/status.json)；本次测量及其 SHA256 见 [本轮 status.json](../results/motion_v2/step0_v2_1/v2_1_reference/status.json) 和 [本轮 SHA256SUMS](../results/motion_v2/step0_v2_1/v2_1_reference/SHA256SUMS)。所有 main 中原有的 Phase 1、Gate、results 文件及既有哈希锁保持不变；本分支新增的运行 protocol 也未在运行后修改。

---

## 历史：原 v2 首次预检（已由 v2.1 修订解决）

以下为 `e07333e` 时的历史停止报告，描述的是当时状态，不是本轮 v2.1 状态。


**状态：在运行任何实验前停止。** 参考实现与要求的配对终点约束冲突；没有修改设计、运行两个 headroom 脚本、生成数据或使用 GPU。此停止是用户要求的“不一致就停”，不是第 2 步预测器统计判据失败。

## 工作范围与来源

分支 `experiments/motion-v2-20261030` 从本次 fetch 得到的最新 `main`（`9d26be1a85b64548602c17231169d274f70914b8`）建立。另建工作目录，原 checkout 保持不动。

`motion_v2_for_codex.md` 内嵌的 shell 块已原样执行，仅写出五个附件文件并计算 SHA256。附件本身 SHA256 为 `780675a4639262af910be9586932af489373bb51429742e11f3c46592433db42`。五个文件均与附件给出的预期哈希完全一致；两个 JSON 是导入的参考结果，**不是本次复现结果**。原草案中的“已完成”和数字均按原文保留，只陈述附件作者的结果。

原用户请求决定本次执行范围：只允许第 0–2 步；遇到不一致即停；第 2 步无论通过与否都停；不训练保留策略。附件草案提及的后续步骤和“先修预测器”不构成本次继续运行、改阈值或重跑的授权。本次没有实验运行，因此没有事后补签运行 protocol。

## 停止证据

用户要求和[附件协议第 2 节](MOTION_V2_PROTOCOL_ZH.md#2-motion-v2-设计)均规定：配对样本旧运动相同、最终速度方向相反、`end` 位置严格相同。

但 [reference `episode()`](../experiments/motion_v2/analysis/motion_v2_headroom.py#L46-L63) 的实际定义是：

- 第 48–49 行：`L1` 在 3–7，`end = t_c + L1 - 1`。
- 第 57–61 行：`x(t_c) = 0.5`，`v1 = (2 * label - 1) * s`，`s` 在 0.008–0.012。
- 第 62–63 行：从该共同位置以正、负速度向前推进。

即便给予该代理最有利的配对解释——共享所有随机量，只改变 label——也有：

```text
x_plus(end)  = 0.5 + s * (L1 - 1)
x_minus(end) = 0.5 - s * (L1 - 1)
end_position_difference = 2 * s * (L1 - 1) > 0
```

两条终点位置都在 `[0.428, 0.572]` 内，不触发 `[0.08, 0.92]` 的反射。差值范围界为 `[0.032, 0.144]`，绝不为零。例如合法参数 `L1=3, s=0.01` 给出 `0.52` 和 `0.48`。这是从源码直接推导的例子，不是执行脚本得到的测量。

此外，脚本没有构造共享场景的 sibling 对：第 121–124 行用不断推进的 RNG 生成独立 episode，仅交替 label。因此参考结果也未审计配对的旧帧、动作、最近四帧或终点一致性。

## 对 headroom 解释的影响

原实现保证 `end` 可见，而其两种方向的 `end` 球心分别位于 0.5 两侧。因此仅该帧的绝对位置已经携带方向/label 信息，不支持草案“只保留一张时无法区分配对样本”的论证。这不等于已训练的 I-JEPA 预测器一定能利用该线索；本次没有相关编码或学习实验。

第 55、110、124 行把成功定义为“选中的两帧均标为 sufficient”，把上限定义为“历史至少含两张 sufficient 帧”。这个定义可衡量预设帧类别的保留率，但没有直接验证像素恢复速度、配对终点一致或两帧的联合必需性。故附件的 100% 不能当作这些性质已经审计通过的证据。

不自动把共同 `t_c` 改成共同 `end`：这样会影响共同旧轨迹到新轨迹的衔接，需要明确是否允许 `t_c` 处的位置跳变等设计选择。这里保留原附件供审阅，不擅自选择替代实现。

## 执行结果

| 项目 | 本次状态 |
|---|---|
| 第 0 步：五个文件导入与 SHA256 | 5/5 一致 |
| 第 0 步：两个 CPU 脚本复现 | 未运行；不能报告数字已复现 |
| seed 9137 manifest 逐项对照 | 未运行；候选文件存在，但未打开 |
| 第 1 步：v2 生成器、pytest、在线模拟 | 未实现／未运行；源码预检触发停止 |
| 第 2 步：(a) recent-only | h=3/6 特征 MSE、位置误差、95% CI 均无结果 |
| 第 2 步：(b) 正确两帧 | h=3/6 特征 MSE、位置误差、95% CI 均无结果 |
| 第 2 步：(c) 旧运动两帧 | h=3/6 特征 MSE、位置误差、95% CI 均无结果 |
| 第 2 步是否过关 | 未评估，不应记为统计失败 |
| GPU 估时与实际用时 | 尚未进入估时阶段；实际 **0 A100-GPU 秒** |
| test 数据 | 未打开、未生成 |
| 保留策略／第 3 步 | 未启动 |

manifest 候选为 `results/phase1_independent_confirmation/execution_v7_metadata/native_v6_evidence/fresh/dataset/manifest.json`（1,468,996 字节）；这里只记录路径和元信息，没有核实其内容、seed 或与重建的匹配关系。用户指定的 Phase 1 source/config 路径存在。所有 Phase 1、Gate、已有 results 文件及既有哈希锁保持不动。

最大风险是把按人为 sufficient 标签计算的 headroom，当作已经满足配对约束且确实需要两帧的任务证据；这样后续预测器即使改善，也可能利用单帧绝对位置线索。需要先由用户明确配对轨迹衔接与“联合必需”的要求，再编写含参数、停止条件及源码哈希的运行 protocol，提交后才可启动实验。

## 附件完整性

| 文件 | SHA256 | 与附件预期比较 |
|---|---|---|
| `docs/MOTION_V2_PROTOCOL_ZH.md` | `7c3bd7acff814554a0fd65d4e9c02753893787c05a229163fdfec110ca9c6762` | 一致 |
| `experiments/motion_v2/analysis/motion_headroom.py` | `989afebc5c2fd2ece882442c5dd508b2d2afc0102d2b623033002c0fa8e9cd78` | 一致 |
| `experiments/motion_v2/analysis/motion_v2_headroom.py` | `00d9420a0512682186bda466f08e8a653aa1b9ac3bf06a34206246dd02d26dcf` | 一致 |
| `results/motion_v2/headroom/motion_v1_pool_headroom.json` | `09c9606f98960092b0b1ee8467fe08d453c52884c2df5c7a15afbf31cced16df` | 一致 |
| `results/motion_v2/headroom/motion_headroom_online.json` | `d11f3e62f9b7237b925361e20b601ddc3a2be9bb93d7824138a6f45c2a218ecd` | 一致 |

机器可读状态见 [status.json](../results/motion_v2/status.json)；本次新增文件的 SHA256 清单见 [SHA256SUMS](../results/motion_v2/SHA256SUMS)，清单不包含自身。
