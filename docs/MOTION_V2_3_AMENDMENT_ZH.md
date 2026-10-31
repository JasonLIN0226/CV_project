# Motion v2.3 修订登记（训练前）

日期：2026-10-31。依据用户本轮明确修订，接续 v2.2；本文件及对应 `experiments/motion_v2/amendment_v2_3.json` 提交并推送后，才进行实现、软件测试、部署或计算。基线提交为 `91a7837d4fed2b39c558291f735bfcb33480f874`，分支仍为 `experiments/motion-v2-20261030`，审阅 PR 为 #3，不合并、不 force push。

## 停止规则分级

阻塞并停止：源码/protocol/数据哈希、模型 revision、输入审计或配对一致性不符；GPU 型号或数量不符；预算实际或预测超限；NaN、缺项；输出目录已存在；出现需要用户本人完成的认证；科学作业 `Restarts > 0`。不得修改阈值、临时替换版本、重装依赖、重跑科学入口或绕过失败。

非阻塞、记录后继续：不运行科学入口的辅助会话（OOD Jupyter、CPU 准备会话）的调度属性，包括 Requeue、实际 AllocTRES 与请求不同。作业 16512 是 `ood-jupyter` 辅助会话，表单附加选项未必生效，且其中没有科学计算。故 v2.2 将其 Requeue 差异判为科学推进的阻塞条件过严；保留原记录，不追改原结果。防止重复运行依靠科学入口拒绝已存在输出目录，以及单次 A100 sbatch 的 `--no-requeue`，并核对原生 Restarts。

A100 科学作业提交后立即保存 `scontrol show job`；运行前及运行中检查原生作业身份、单 A100 分配和 Restarts，完成或停止后保存 `sacct`。科学入口只调用一次。辅助会话的调度豁免不适用于科学作业的型号、数量、时长或重启检查。

## 无模型方向下界 B

训练条件仍是 recent-only、full-history、固定 b 和 random K=2。d 只在 random K=2 中偶尔出现，因此 d−b 可能包含预测器处理 d 组合不佳造成的额外误差。新下界用于要求 b 超越同一 pair 两个 sibling 获得相同预测时的最佳位置误差。

对 validation 每个 pair、每个 h∈{3,6}，令真实二维位置为 p₀、p₁，m=(p₀+p₁)/2，定义：

`B = mean_sibling(mean_xy((m - p_sibling)^2))`。

对于两个 sibling 的共同预测 q，pair 平均位置 MSE 恰为 `B + mean_xy((q-m)^2)`，故至少为 B。B 只从 validation 的 evaluation_only 真实 target 位置计算，不拟合模型，计算本身零 GPU；不访问 test。位置归一化和 MSE 的 xy 平均定义与 v2.2 相同。

新增必需判据：h=3、h=6 各自 `B − position MSE(b)` 的配对 95% CI 下界严格大于 0。b 的 loss 先平均两个 sibling，再平均固定三个训练 seed；B 每 pair/horizon 一个值，不将其当作额外 seed 或独立样本。

原必需判据不变：h=3/6 的 `position MSE(d)−position MSE(b)`、`position MSE(c)−position MSE(b)` 的配对 95% 下界分别严格大于 0。三种必需对比、两个 horizon 共六项必须全部通过。`a−b` 和 feature MSE 仍报告。

新增非门槛诊断：`position MSE(d)−B` 及 CI；在真实 validation target features 上的冻结 decoder position MSE 与 B 并列报告。绝对 B、a/b/c/d、所有差值及每 seed 附表共享同一套 96-pair、2,000 次、seed 901 bootstrap 索引，percentile 2.5%/97.5%，linear quantile。区间只覆盖固定三个已拟合 seed 平均表现的场景抽样不确定性。

## 实现、测试与科学 protocol 的顺序

本修订先提交推送。随后修改 `step2_metrics.py`、必要的 runner 接线和对应测试，使 validation 隐藏状态只参与评估统计，不进入预测器输入、训练、checkpoint 或 decoder 选择。旧 protocol、旧 JSON、原停止记录不改写；旧源码由基线 Git 提交保留，旧源码锁仍描述原版本，不为新代码改写旧锁。

训练前进行零 GPU 纯数组单元测试：共同预测恒有 pair loss≥B；达到 midpoint 时取等号；B−b 与 d−B 的差值方向正确；两个 horizon 独立判定；所有指标复用相同 bootstrap；非法形状、缺项及非有限值拒绝。软件测试使用人工坐标/loss，不生成 episode，不读实际 validation/test，不导入或拟合模型。测试结果、源码 SHA256、命令、环境和 status 写入新 `results/motion_v2/v2_3/software/`；失败即停，不根据结果改阈值重试。

测试通过后再提交推送 v2.3 第 2 步科学 protocol，锁定新源码、原数据和模型 revision、统计规则、实际命令及原预算。B 的实际 validation 计算在该科学 protocol 推送后、训练前于 CPU 上完成，并保存逐 pair 值和 SHA256。不得因 B 数值调整训练或判据。

## 部署与复用

保留 `/scratch/xy3257/motion-v2-20261031-deployment1` 全部现场，不删除、不改写。新建先确认不存在的 `/scratch/xy3257/motion-v2-20261031-deployment2`，浅克隆包含 v2.3 科学 protocol 的已推送登记提交并核对精确 HEAD。

允许只读复用 deployment1 的 venv 与已上传 data.zip。数据包先验证 SHA256 `58f14130ddda659f9c3a11bbbc50d2729cf61b0613da7bbf7522466afa28831d`；仅向 deployment2 新数据目录解包，逐 NPZ 校验 manifest，拒绝任何已存在目标。manifest SHA256 保持 `9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a`。

venv 做 import 和关键版本检查，禁止生成 pyc 或向旧环境/旧缓存写入；所有临时文件、日志和新缓存放 deployment2。失败即停，不临时重装或换版本。可建立零 GPU 辅助会话供核验、提交和最终原生结算；其调度差异按本修订非阻塞记录。实际 GPU partition/account 与单 A100 参数只依据当前可读集群状态确定，提交脚本和部署参数在提交前登记，不凭历史假定。

科学作业使用 sbatch、单张 A100、`--no-requeue`、wall-time≤7,200 秒，仅一次 `run_step2.py`。原估时上限 7,200 A100 秒、用户总上限 14,400 秒和速率更新/提前停止规则不变；GPU 分配中的初始化、CPU 工作、下载和失败占用也计入。保持原 encoder、架构、seeds 17/29/43、预算、训练条件、checkpoint 和 train-only decoder 规则，不增加实验。

## 交付与边界

记录部署检查、A100 job ID、scontrol/sacct、a/b/c/d×h=3/6 的 feature/position MSE 与 CI、并列 B、d−b/c−b/B−b/d−B、每 seed 附表、实际 A100 秒数、status.json 与 SHA256。遇阻塞即停；六项科学判据无论通过或失败也停。没有训练保留策略、没有第 3 步、没有打开或生成 test。原 manifest 回溯仍非阻塞。
