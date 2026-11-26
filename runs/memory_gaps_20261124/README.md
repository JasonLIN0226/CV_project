# Comparison gaps filled on 24 Nov 2026 (Burst job 42464, one A100)

Code: branch `motion-v2/clean-rerun` (commits `ec4a10a`, `d8cb945`, `45b9266`). Full outputs stay on Burst
under `/scratch/xy3257/mv2_runs/` (Motion v2) and `/scratch/xy3257/cv_project_runs/reproduction/synergy_gap/`
(Synergy). Every number below is a development or confirmation result as labelled; nothing was tuned on it.
`extract.json` holds the numbers of sections 1 and 2 and the batch-equivalence and stage-A checks, with the SHA256
of the Burst files they come from.

The code was reorganised afterwards without changing what it computes: the Motion v2 baselines are the
manifest tasks `motion_v2.literature.*` (one method and reader seed per call, pooled by
`scripts/aggregate_results/motion_v2_literature.py`), the scorer of section 2 is
`methods/deletion_cost/motion_v2.py` (task `motion_v2.registered.scorer`), and the Synergy reruns are
`synergy.literature_gpu.train` and `synergy.literature_gpu.paired`.

## 1. Motion v2: the three literature baselines on the frozen readers

`scripts/train_baselines/motion_v2.py`, every method trained against one reader (seeds 17, 29, 43) on the
1,024 training episodes and scored on the 96 validation scene pairs under all three readers (the path of
the published Motion v2 table). Position MSE of the decoded ball position, lower is better. Gate's and
Motion's batch builders gave identical reader predictions (max difference 0).

| Method | Memory | Ordinary +3 | Ordinary +6 | +3 per seed (17 / 29 / 43) | Long gap +3 |
|---|---|---:|---:|---|---:|
| reference: post-hoc best pair | privileged | 0.002535 | 0.003139 | | 0.004508 |
| reference: the two cue frames (b) | privileged | 0.006052 | 0.006860 | | 0.017639 |
| **attention (project baseline)** | bounded | **0.009147** | **0.011148** | | **0.025911** |
| PaFu-KV teacher top 2 | privileged (future) | 0.009318 | 0.011440 | 0.009039 / 0.009279 / 0.009637 | 0.026249 |
| PaFu-KV | bounded | 0.009374 | 0.011395 | 0.008714 / 0.009301 / 0.010107 | 0.027428 |
| future-attention student (set) | bounded | 0.009756 | 0.011751 | | 0.025384 |
| reference: last two visible frames | privileged (visibility) | 0.011120 | 0.013360 | | 0.029212 |
| FrameMorrow teacher top 2 (I-JEPA) | privileged (future) | 0.017450 | 0.021113 | | 0.040327 |
| Compress to Remember, two slots | bounded | 0.019030 | 0.022648 | 0.015015 / 0.018601 / 0.023472 | 0.042064 |
| Compress to Remember, full history | unbounded | 0.020215 | 0.024244 | 0.021618 / 0.019017 / 0.020010 | 0.050327 |
| FrameMorrow published (I-JEPA) | unbounded | 0.020716 | 0.024554 | 0.020937 / 0.018490 / 0.022721 | 0.041604 |
| ours, registered deletion-cost student (set, use-time position) | bounded | 0.021401 | 0.025305 | | 0.041540 |
| FIFO | bounded | 0.023793 | 0.027355 | | 0.042148 |
| FrameMorrow online (I-JEPA) | bounded | 0.024301 | 0.027911 | 0.023894 / 0.024193 / 0.024816 | 0.042066 |

No literature baseline beats the attention policy on either interval; PaFu-KV is within noise of it.
FrameMorrow uses the project's I-JEPA features here, not DINOv2 as published.

## 2. Motion v2: deletion-cost scorer v2 (development; negative)

`scripts/train_scorer/motion_v2_deletion_cost_v2.py` (optimal-continuation costs, zero-initialised residual
on the readers' salience, checkpoint chosen on a dev split of 54 training scene pairs). On dev, step 0 (the
attention policy) scored +3 0.004369 / +6 0.005130 and every trained checkpoint was worse (best +3 0.0053 to
0.0057 per seed), so selection kept step 0 for all three seeds: v2 adds nothing over attention. It was not
evaluated on validation (no look taken). The same attention policy scores 0.0044 on dev episodes and 0.0091 on
validation: the frozen readers are far better on the episodes they were trained on, so teacher labels built
from training episodes reflect reader memorisation.

## 3. Synergy, item_attention reader: ours vs Compress to Remember, paired

Compress to Remember rerun with three seeds on TRAIN features re-encoded on the GPU (S regret per seed
+0.0037 / +0.0057 / +0.0035; the original run reported 0.0037). Our released set students, replayed exactly.
Stage A on the locked confirmation set reproduced the released arrays: largest absolute difference 3.3e-6 over
all stage-A arrays of both readers (eval_cost 8.9e-7 item_attention, 2.2e-7 multihead), with the same argmin
everywhere (`extract.json`, synergy_stage_a_against_released).
`scripts/aggregate_results/synergy_paired.py`, 48 families, 10,000 bootstrap draws; negative = ours lower regret.

| Task | Ours | Compress to Remember | Ours minus C2R | 95% interval | Families where ours is lower |
|---|---:|---:|---:|---|---:|
| S (synergy) | 0.0020 | 0.0043 | -0.0023 | [-0.0051, +0.0006] | 67% |
| R | 0.0171 | 0.0018 | +0.0153 | [+0.0092, +0.0221] | 27% |
| D | 0.0014 | 0.0001 | +0.0013 | [+0.0004, +0.0025] | 40% |

Ours has the lower S regret but the interval includes zero; ours is clearly worse on R and D. FrameMorrow on
item_attention and Compress to Remember on multihead were still running when this was written
(both were completed in `runs/synergy_paired_20261125/`).
