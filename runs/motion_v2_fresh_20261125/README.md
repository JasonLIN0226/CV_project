# Motion v2: our scorer trained on fresh scenes (25–26 Nov 2026; development, negative)

Burst jobs 43438 (fresh scenes, loss tables, first scorer run), 43547 (variants) and 43573 (holdout check), one
A100 each, branch `motion-v2/clean-rerun` at commits `995d299`, `88c7bc6` and `d0aa049`. The scripts were then
reorganised into the manifest tasks `motion_v2.fresh.*` (`configs/motion_v2/fresh.yaml`) without changing what they
compute; the validation step now also saves per-episode losses. Full outputs stay on
Burst under `/scratch/xy3257/cv_project_runs/motion_v2_fresh/`. Validation was not read.

**Why fresh scenes.** The frozen readers were trained on the 1,024 registered training episodes and are far better
on them: the attention policy scores +3 position MSE 0.0044 on training scene pairs and 0.0091 on validation.
Deletion-cost labels built on those episodes describe what the readers memorised (`runs/memory_gaps_20261124/`,
section 2). `environments/motion_v2/fresh.py` draws new scenes from the registered generator under a split name
of its own (`fresh_train_20261125`; the copied generator reproduces the registered train and validation scenes
exactly, `tests/motion_v2/test_fresh.py`). Scene pairs 0–511 give training states, 512–607 choose checkpoints,
rounds and the variant (dev), and 608–703 check the choice once (holdout).

## 1. The readers behave on fresh scenes as on validation

Attention policy on fresh dev: +3 0.0095, +6 0.0113 (validation 0.0091 / 0.0111; registered training scene pairs
0.0044 / 0.0051). The best pair chosen after the fact scores 0.0026 (mean of +3 and +6) against attention's
0.0104, so the room to improve is large.

## 2. Variants, chosen on fresh dev (`variants_dev.json`; every round of every seed in `runs_extract.json`)

Position MSE, mean of +3 and +6, of each seed's selected checkpoint (step 0, the attention policy, is always a
candidate), mean over seeds 17, 29 and 43. Attention: 0.010388.

| Variant | Label | Rounds with the scorer's own states | Dev | Per seed (17 / 29 / 43) |
|---|---|---:|---:|---|
| min | minimum over reachable pairs | 0 | 0.010388 | step 0 on all three |
| min_dagger | minimum | 3 | **0.009663** | 0.009806 / 0.009210 / 0.009974 |
| soft | soft minimum, T = 0.002 | 0 | 0.010336 | 0.010388 / 0.010231 / 0.010388 |
| soft_dagger | soft minimum | 3 | 0.009875 | 0.010203 / 0.009911 / 0.009510 |

Chosen: min_dagger. Warning signs before the holdout: each seed's value is the best of about 50 checkpoints, and
dev MSE moves by about 0.002 between neighbouring checkpoints (job logs of 43547); the last checkpoint of every round was worse than
attention (0.0115 to 0.0135); and on dev states the scorers did not systematically pick the cheapest deletion more often than
attention does (top-1 0.634 to 0.649 against 0.646 with the minimum label).

## 3. Holdout check (`holdout.json`)

Fresh scene pairs 608–703, used for nothing else. Paired with attention per scene pair (mean of siblings,
horizons and seeds); 95% interval over the 96 scene pairs. Negative = lower error than attention.

| Variant | Dev | Holdout | Minus attention | 95% interval | Pairs lower |
|---|---:|---:|---:|---|---:|
| attention | 0.010388 | 0.010813 | | | |
| min | 0.010388 | 0.010813 | 0 (same choices) | | |
| **min_dagger** (chosen) | 0.009663 | 0.010521 | −0.000293 | −0.002027 to +0.001350 | 47% |
| soft | 0.010336 | 0.010854 | +0.000041 | −0.000530 to +0.000612 | 48% |
| soft_dagger | 0.009875 | 0.011328 | +0.000515 | −0.001092 to +0.001987 | 36% |

Best pair after the fact on the holdout: 0.0030. (The recorded `holdout.json` counts min as lower in 27% of pairs:
float rounding of identical choices; `evaluation/metrics/motion_v2.paired` now ignores differences below 1e-12.)

## Reading

The dev gain of min_dagger was selection among many noisy checkpoints: on scenes not used to choose it, no
variant differs from the attention policy. So validation was not read
(`motion_v2.validate.deletion_cost_v2` refuses without a holdout gain), and on Motion v2 our scorer does not beat
attention. Training on scenes the readers never saw removed the memorisation bias but not the failure: the
scorer still does not learn which frames to keep, although the best pair is far better than attention (0.0030
against 0.0108 on the holdout). Why is open.
