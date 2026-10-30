# Motion v2：附件导入与停止报告（2026-10-30）

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
