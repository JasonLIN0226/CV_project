# 给 @JasonLIN0226 的 Phase 1 交接清单

基于已审计 commit `2c73c6bca11a9396cdb0546b8bf78b52445e7b22`：目前已有代码和结果汇总，以下材料可让 Phase 2 接着原模型开展。**先交小文件、源码和模型配套即可，大缓存可后补。** 请给可访问的共享路径或下载位置，不需要提供账号密码。

## 优先交接

1. **10 月 19 日最终 Gate/Motion 修复源码和配置。** 包含 Gate 关闭完整 temporal-extrapolation 分支的实现，以及 Motion 的 observed-RGB guard、observation-innovation map 和组合逻辑；附各自 protocol、source lock、运行命令、依赖版本。Motion 已知原执行目录是 `/scratch/xl4358/phase1_motion_observation_mix_locked_test/`。Gate 只核实到 job `8332`，请你提供实际源码目录。当前仓库没有这两套最新修复源码对应的 lock/protocol 文件。
2. **两个已拟合的 train-only maps。** 请给实际文件、加载代码、拟合配置、输入/输出含义、使用的 train episode IDs、SHA-256。当前证据没有 map 文件名或路径，因此不预设位置。
3. **配套 predictor 与 frozen reference checkpoints。** 两个任务、seeds `17/29/43` 的六个 candidate `best.pt` 位于 `/scratch/xl4358/phase1_predictor_repair_v2/execution/training/job_7574/predictor/` 下；另需各 reference 目录中的六个 `best.pt`。若最新版本更换过其中任何模型，请明确替代关系，保留对应 hashes 和完成记录。
4. **与这些 checkpoints 配套的 transforms 和数据身份。** 每个任务的 `basis.npz` 及 JSON、`features/normalization.npz` 及 JSON、resolved config、dataset manifest、feature index、相关 provenance/audit 小文件。请保留原 episode/pair IDs、data seed、模型 seeds、train/validation/test 划分、生成配置和源版本；说明哪些开发/test 数据已被查看，哪些仍未使用。旧记录的数据 seed 是 `6301`；不能把后来重建的数据当作原始未见测试。

5. **如果已有 PushT / baseline planning 实现，请补源码、模型配置、动作和时间定义、运行命令及 rollout 记录。** 如果此前没有实现，说明尚未做即可，不需要补造结果。本分支已独立完成真实 PushT 观测接口和官方 I-JEPA 编码；baseline model 驱动的 planning 验证仍需补齐。

对应 reference 根目录：

- `/scratch/xl4358/phase1_v3_dev/gate/reference/`
- `/scratch/xl4358/phase1_v3_dev/motion/reference/`

**请附一份 SHA-256 manifest**：逐文件列原路径、交接包内相对路径、大小、SHA-256、用途及其 checkpoint/config 对应关系。已有 [资产清单](../experiments/phase2/required_phase1_assets.json) 包含 21 个旧版本必要资产的已发表 hashes，可用作核对；它不涵盖全部最新修复文件，也不能替代完整数据目录。

## 可以后补或重建

- **大体积特征缓存和 RGB 数据**：有原缓存最快；没有则可按源码、配置和 split manifest 重新生成/编码。重建后须核对身份、hash 和配套 transforms，记录新 provenance，不能直接假定与旧 checkpoint 完全一致。
- **官方 I-JEPA 权重/processor**：可重新下载 `facebook/ijepa_vith14_1k`，固定 revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e`，无需你重复上传。
- **旧 raw metrics、logs、诊断用 state decoder/probes**：复核原实验的 task-state 指标时需要；不是启动 Phase 2 接口开发的前提。若最新推理代码直接依赖某个 readout，请随模型配套交接，不能归入后补。后补时请保留对应 job、模型和数据 hash。

## 收到文件后如何核对

把交接包解压到自己的目录，保留 `/scratch/xl4358/` 后面的相对目录结构，例如 `/path/to/intake/phase1_v3_dev/gate/reference/...`，运行：

```bash
python3 -m experiments.phase2.verify_assets --root /path/to/intake
```

如果文件重新分目录，可写一个 JSON，把清单中的**完整原路径**映射到实际本地路径，然后运行：

```bash
python3 -m experiments.phase2.verify_assets --mapping /path/to/path_mapping.json
```

映射值是绝对路径；同时提供 `--root` 时，也可用该目录内的相对路径。工具只读文件字节计算 SHA-256，区分缺失、内容不匹配与读取失败，不下载、不执行模型、不反序列化 `.pt`/`.npz`。退出码 `0` 仅表示清单内必要资产全部匹配；最新 repair/maps、完整数据目录和科学结果仍需单独核实。

在交接期间，Phase 2 通用记忆接口、baseline、删除监督合同和 PushT 接口可以独立继续。原任务的额外 fresh-confirmation/T8 流程不新增为本次 proposal 的执行门槛。

## 20 October 早期独立试跑记录

不等待以上原始资产，新开发分支已完成 A100 上的公开 I-JEPA 编码、128 条新 Gate/Motion 受控轨迹、两个新初始化预测器的训练/冻结试跑、238 个去重父状态的删除监督，以及六类 retention baseline 的小规模验证比较（16 条验证轨迹，96 条策略预测记录）。该轮 181 项软件测试全部通过。预测器验证误差略差于初始化，六策略结果也只来自重复使用的开发验证集，不据此宣称效果提升或 Phase 2 全部完成。实际日志、代码版本和限制见 [运行记录](PHASE2_HPC_RUN.md)。这些新数据和权重已备份，不替代上面用于接续和复核你原实验的交接包；baseline-model planning 仍待完成。

## 20 October 最新独立推进状态

后续已完成历史使用诊断，以及三种输入投影 × 两个任务 × 40 epochs 的固定 CPU 对照；选模包含 epoch 0。Gate 去掉白化的 epoch 35 候选获得开发集历史收益：recent-only MSE 0.102571，正确 privileged K=2 历史 0.074275，错误历史 0.196949。Motion 仍未建立正确历史相对 recent-only 的净收益。这里只使用四对反复检查的验证场景，不是独立泛化或 learned retention 结论，详见 [诊断报告](PHASE2_HISTORY_DIAGNOSIS.md)。

该 Gate 模型已重新绑定并运行六类 baseline（48 条 episode-policy 记录、96 个损失），生成新的删除监督（94 train + 24 validation parents，354 forks、708 个损失），与早期双任务试跑产物分别保留。详见 [Gate 续跑记录](PHASE2_GATE_FOLLOWUP_CONTINUATION.md)。另已独立采集 48 条 PushT 轨迹及遮挡/无遮挡对照，每种条件 4,656 帧；这批新数据尚未编码，也未完成模型驱动 planning，详见 [PushT 准备记录](PHASE2_PUSHT_PLANNING_PREPARATION.md)。最新完整软件检查为 **229 项通过，零失败、错误和跳过**。

上述代码、文档、图表和可发布 JSON 证据均在 [PR #1](https://github.com/JasonLIN0226/CV_project/pull/1)。模型、特征、RGB/action 和预测向量二进制保留在本地 `build/`，未加入 Git。以上进展可独立继续，但不能替代本清单要求的原始模型交接；原模型复现所需材料仍通过 [issue #2](https://github.com/JasonLIN0226/CV_project/issues/2) 跟踪。Proposal 下一步是训练 past/present-only set-conditioned scorer 及等容量 attention-target 对照，进行独立不可逆流式评估，并继续补齐 Motion 和 PushT planning。
