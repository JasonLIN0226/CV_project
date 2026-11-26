# Gate on GPU: our scorer and two-slot Compress to Remember over extra initialisation seeds (25 Nov 2026)

Burst job 43212 (one A100, partition `c12m85-a100-1`), branch `motion-v2/clean-rerun`: driver
`scripts/train_baselines/gate_init_seeds.py` as of `72d7614`, summary `scripts/aggregate_results/gate_init_seeds.py`
as of `d24666e` -> `summary.json` (this folder). Unit files stay on Burst under
`/scratch/xy3257/cv_project_runs/reproduction/gate_init_seeds/`.

Both scripts were later removed as duplicates: the driver's fit of our scorer is, statement for statement,
`methods/deletion_cost/training.fit_scorers_with_training_seed`, and its Compress to Remember call is the one of
`scripts/train_baselines/gate.py --train-seed` (only the checkpoint's tag differs). The same run is now the
manifest tasks `gate.sweep_gpu.*` (`configs/gate/seed_sweep_gpu.yaml`), pooled by
`scripts/aggregate_results/gate_seed_sweep.py --sweep sweep_gpu`, which also gives the paired interval over cells.

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
never collapse. On these 9 runs two-slot Compress to Remember matched our accuracy (+0.03 points, interval
−0.78 to +0.95) and spread more (standard deviation 1.77 against 0.72). This is not true of Gate in general: in
the CPU sweep its 25 runs that trained averaged 93.60 against our 94.92, and ours was higher in 24 of 30 cells.
Across CPU and GPU, ours collapsed in 0 of 39 runs and two-slot Compress to Remember in 5 of 39.

The manifest form of this run, `gate.sweep_gpu.*`, has not been run itself; the numbers above come from the
earlier driver named at the top, which ran the same fits.
