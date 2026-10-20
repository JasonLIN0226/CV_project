# 给 @JasonLIN0226 的 Phase 1 交接清单

基于已审计 commit `2c73c6bca11a9396cdb0546b8bf78b52445e7b22`：目前已有代码和结果汇总，以下材料可让 Phase 2 接着原模型开展。**先交小文件、源码和模型配套即可，大缓存可后补。** 请给可访问的共享路径或下载位置，不需要提供账号密码。

## 优先交接

1. **10 月 19 日最终 Gate/Motion 修复源码和配置。** 包含 Gate 关闭完整 temporal-extrapolation 分支的实现，以及 Motion 的 observed-RGB guard、observation-innovation map 和组合逻辑；附各自 protocol、source lock、运行命令、依赖版本。Motion 已知原执行目录是 `/scratch/xl4358/phase1_motion_observation_mix_locked_test/`。Gate 只核实到 job `8332`，请你提供实际源码目录。当前仓库没有这两套最新修复源码对应的 lock/protocol 文件。
2. **两个已拟合的 train-only maps。** 请给实际文件、加载代码、拟合配置、输入/输出含义、使用的 train episode IDs、SHA-256。当前证据没有 map 文件名或路径，因此不预设位置。
3. **配套 predictor 与 frozen reference checkpoints。** 两个任务、seeds `17/29/43` 的六个 candidate `best.pt` 位于 `/scratch/xl4358/phase1_predictor_repair_v2/execution/training/job_7574/predictor/` 下；另需各 reference 目录中的六个 `best.pt`。若最新版本更换过其中任何模型，请明确替代关系，保留对应 hashes 和完成记录。
4. **与这些 checkpoints 配套的 transforms 和数据身份。** 每个任务的 `basis.npz` 及 JSON、`features/normalization.npz` 及 JSON、resolved config、dataset manifest、feature index、相关 provenance/audit 小文件。请保留原 episode/pair IDs、data seed、模型 seeds、train/validation/test 划分、生成配置和源版本；说明哪些开发/test 数据已被查看，哪些仍未使用。旧记录的数据 seed 是 `6301`；不能把后来重建的数据当作原始未见测试。

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
