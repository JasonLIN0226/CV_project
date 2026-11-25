# Gate on GPU: our scorer and two-slot Compress to Remember over extra initialisation seeds (25 Nov 2026)

Burst job 43212 (one A100, partition `c12m85-a100-1`), branch `motion-v2/clean-rerun` (driver as of `72d7614`).
Driver `scripts/train_baselines/gate_init_seeds.py` via `burst/gate_init_seeds.sh`; summary
`scripts/aggregate_results/gate_init_seeds.py` -> `summary.json` (this folder). Unit files stay on Burst under
`/scratch/xy3257/cv_project_runs/reproduction/gate_init_seeds/`.

**Question.** Yixiang's CPU sweep (`runs/gate_seed_sweep_20261124/` on `clean`: training seeds 1–10 on the three
predictors, Torch CPU) found two-slot Compress to Remember collapsing in 5 of 30 runs and ours in 0 of 30, and
left open whether the 2 of 2 collapses on Torch GPU mean the method always collapses on GPU.

**Setup.** The team bundle's three frozen predictors (seeds 17, 29, 43) and 192 validation episodes, horizons 3
and 6, as in the panel. Initialisation seeds 101, 202, 303 (not used by the CPU sweep). Ours: the registered
paired fit with only the initial weights and minibatch order taken from the extra seed (the mirror reproduces
the registered fit exactly for seed = predictor: max |dw| 0 on all three predictors). Compress to Remember:
the Gate implementation unchanged, two persistent slots, 3,000 steps, the extra seed as its training seed.
`CUBLAS_WORKSPACE_CONFIG` was not set (warn-only determinism), so the GPU compressor runs are not guaranteed to be
bit-reproducible.

**Result.** Route accuracy (%), validation:

| | ours | Compress to Remember, two slots |
|---|---:|---:|
| units | 9 | 9 |
| collapsed (<= 60) | 0 | 0 |
| mean | 94.62 | 94.59 |
| min to max | 93.23 to 95.31 | 90.63 to 96.35 |
| standard deviation over units | 0.72 | 1.77 |

| predictor | init 101 (ours / C2R) | init 202 | init 303 |
|---|---|---|---|
| 17 | 94.01 / 95.31 | 95.05 / 95.31 | 95.05 / 94.01 |
| 29 | 94.01 / 95.31 | 93.23 / 90.63 | 95.05 / 93.49 |
| 43 | 94.53 / 94.53 | 95.31 / 96.35 | 95.31 / 96.35 |

Paired over the 9 (predictor, init) units: ours minus Compress to Remember +0.03 points, 95% interval over units
[−0.78, +0.95], over the 192 episodes [−1.88, +1.97]; ours higher in 3, Compress to Remember in 5, tied in 1.

**Reading.** No collapse in 9 GPU runs on Burst, so the Torch GPU 2 of 2 did not recur here; at the CPU sweep's
rate (5 of 30) nine runs without a collapse has probability about 0.19, so this does not show that GPU runs
never collapse. When it trains, two-slot Compress to Remember matches our accuracy; our scorer's advantage on
Gate is stability (0 collapses in 39 runs across CPU and GPU, against 5 of 39) and a smaller spread.
