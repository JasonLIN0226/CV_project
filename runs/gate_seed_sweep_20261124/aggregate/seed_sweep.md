# Gate seed sweep: run gate_seed_sweep_20261124

Route accuracy (%) on the 192 validation episodes, both horizons. A run below 75.00 collapsed.

| Method | Runs | Collapsed | Mean, all runs | Mean, runs that trained | Min | Max | Panel-style value per training seed: mean (min to max) | Threads |
|---|---|---|---|---|---|---|---|---|
| Compress to Remember, two slots | 30 | 5 | 87.33 | 93.60 | 55.21 | 96.35 | 87.33 (69.01 to 95.14) | 8 |
| ours, deletion-cost scorer | 30 | 0 | 94.92 | 94.92 | 92.97 | 96.09 | 94.92 (93.92 to 95.49) | 8 |

Same predictor and training seed, higher route accuracy: Compress to Remember in 5, ours in 24, tied in 1 of 30 cells.

## ours, deletion-cost scorer

| Training seed | predictor 17 | predictor 29 | predictor 43 |
|---|---|---|---|
| 1 | 96.09 | 94.79 | 95.05 |
| 2 | 95.31 | 95.31 | 94.27 |
| 3 | 95.31 | 94.27 | 96.09 |
| 4 | 94.27 | 92.97 | 94.53 |
| 5 | 95.31 | 95.31 | 93.75 |
| 6 | 94.79 | 93.23 | 94.53 |
| 7 | 96.09 | 93.75 | 94.79 |
| 8 | 95.57 | 95.05 | 94.53 |
| 9 | 95.57 | 95.57 | 95.05 |
| 10 | 95.57 | 95.57 | 95.31 |

## Compress to Remember, two slots

| Training seed | predictor 17 | predictor 29 | predictor 43 |
|---|---|---|---|
| 1 | 94.79 | 96.09 | 94.53 |
| 2 | 55.47 | 94.79 | 93.75 |
| 3 | 93.49 | 90.89 | 55.73 |
| 4 | 95.31 | 85.68 | 95.05 |
| 5 | 95.05 | 94.53 | 95.57 |
| 6 | 92.19 | 55.47 | 93.49 |
| 7 | 91.93 | 91.15 | 96.35 |
| 8 | 94.01 | 57.81 | 55.21 |
| 9 | 95.31 | 95.57 | 94.27 |
| 10 | 95.31 | 86.46 | 94.53 |
