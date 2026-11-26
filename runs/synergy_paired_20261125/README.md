# Synergy: our released students against Compress to Remember and FrameMorrow, paired (25 Nov 2026)

Burst jobs 42464 (24 Nov), 43211, 43322 and 43323 (25 Nov; each earlier one was reclaimed by Burst and the next
resumed), one A100 each, `burst/synergy_literature_gap.sh` on branch `motion-v2/clean-rerun` (commit `45b9266`;
now the manifest tasks `synergy.literature_gpu.train` and `synergy.literature_gpu.paired`). This completes the
comparison begun in `runs/memory_gaps_20261124/README.md` section 3.

- Baselines retrained with three seeds (17, 29, 43) on TRAIN features re-encoded on the GPU
  (`configs/synergy/gpu_features.yaml`): Compress to Remember on both readers, FrameMorrow (published variant) on
  item_attention. Our released set students (`runs/repro_20261111/synergy/replay/students_ours_*`) replayed
  exactly.
- Locked confirmation set, 48 families (576 R, 576 D, 2,304 S records). Regret = loss of the policy's deletion
  minus the best single deletion, averaged over seeds; statistic = family mean of ours minus baseline; 95%
  interval resamples the 48 families (10,000 draws, seed 20261124; `scripts/aggregate_results/synergy_paired.py`).
  Negative = ours lower regret. Files: `paired_<reader>_ours_vs_<baseline>.json` in this folder.

| Reader | Baseline | Task | Ours | Baseline | Ours minus baseline | 95% interval | Families where ours is lower |
|---|---|---|---:|---:|---:|---|---:|
| item_attention | Compress to Remember | R | 0.0171 | 0.0018 | +0.0153 | [+0.0092, +0.0221] | 27% |
| item_attention | Compress to Remember | D | 0.0014 | 0.0001 | +0.0013 | [+0.0004, +0.0025] | 40% |
| item_attention | Compress to Remember | S | 0.0020 | 0.0043 | −0.0023 | [−0.0051, +0.0006] | 67% |
| item_attention | FrameMorrow (published) | R | 0.0171 | 0.0275 | −0.0104 | [−0.0183, −0.0023] | 69% |
| item_attention | FrameMorrow (published) | D | 0.0014 | 0.0305 | −0.0291 | [−0.0340, −0.0245] | 100% |
| item_attention | FrameMorrow (published) | S | 0.0020 | 0.0483 | −0.0463 | [−0.0501, −0.0422] | 100% |
| multihead | Compress to Remember | R | 0.0150 | 0.0026 | +0.0124 | [+0.0078, +0.0178] | 25% |
| multihead | Compress to Remember | D | 0.0009 | 0.0006 | +0.0003 | [−0.0010, +0.0018] | 50% |
| multihead | Compress to Remember | S | 0.0036 | −0.0046 | +0.0082 | [+0.0059, +0.0105] | 15% |

Reading:

- Ours has clearly lower regret than FrameMorrow on every task (item_attention).
- Against Compress to Remember, ours is not clearly better anywhere: on item_attention its S regret is lower but
  the interval includes zero; it is clearly worse on R on both readers, on D on item_attention, and on S on
  multihead, where Compress to Remember's negative regret (it merges the three cards into two learned items
  instead of deleting one) beats every deletion policy.
- The retrained baselines reproduce the registered values (`configs/reference_values.json`): Compress to Remember
  S 0.0043 against 0.0037 on item_attention (seeds 0.0037 / 0.0057 / 0.0035) and −0.0046 against −0.0046 on
  multihead; FrameMorrow S 0.0483 against 0.0483.
