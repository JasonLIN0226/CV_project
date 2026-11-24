# Motion v2 online validation replayed on the frozen assets (24 Nov 2026)

`scripts/evaluate/motion_v2_online.py` with `configs/motion_v2/frozen.yaml` on NYU Burst
(`BM_MOTION_ROOT=/scratch/xy3257`), code at commit `d6d3f4b` of `motion-v2/clean-rerun`
(package source SHA256 `9c730416d6060e272749524f2ff7afd8a5356efe0fbec6e1ef63ee73aeb80dff`).
One NVIDIA A100-SXM4-40GB, Python 3.11.18, torch 2.5.1+cu121, numpy 1.26.4, 4 threads,
deterministic algorithms, no TF32, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Result: the raw online losses of every method are bit-identical to the tracked step-3/4 evidence
(job 17925) on both intervals: ordinary 28/28 identical, long gap 28/28 identical** (maximum absolute
difference 0 for position, feature and success arrays).

| Interval | Job | Inputs verified | Methods identical | +3 position MSE: FIFO / attention / ours (set, use-time position) / reference b / post-hoc best |
|---|---|---|---|---|
| ordinary | 41903 | step2 1232, dataset manifest, evidence 171, long-gap receipt 391, long-gap features 194 | 28 / 28 | 0.023793 / 0.009147 / 0.021401 / 0.006052 / 0.002535 |
| long gap | 41941 | step2 1232, dataset manifest, evidence 143, long-gap receipt 391, long-gap features 194 | 28 / 28 | 0.042148 / 0.025911 / 0.041540 / 0.017639 / 0.004508 |

Notes:

- Job 41903 was terminated by Slurm (`CANCELLED ... DUE to SIGNAL Terminated`, 28 minutes into a
  3-hour allocation) after the ordinary interval had finished and been compared, and before the
  long-gap post-hoc oracle finished; its task JSON was therefore not written and `replay1.log` is the
  record of the ordinary comparison. The long gap was rerun alone in job 41941.
- On the long gap, every deletion-cost and shuffled-label student of one architecture has the same
  loss (set 0.041540, independent 0.041756), consistent with those students keeping their initial
  weights (see `configs/reproduction.yaml`, motion_v2 student_selection).
- Full outputs (raw losses, traces, summaries) are on Burst under `/scratch/xy3257/mv2_runs/`
  (`replay1/online/ordinary`, `replay2/online/long_gap`, `replay2/online_replay_seed0.json`).

## Logs

`check.log` (`--verify-only`):

```
[online_replay seed 0 +    31s] inputs match their checksums: step2 1232, dataset ok, evidence 171, longgap_receipt 391, longgap_features 194
[online_replay seed 0 +    31s] wrote /scratch/xy3257/mv2_runs/check/online_replay_seed0.json
```

`replay1.log` (job 41903, `--intervals ordinary long_gap`, terminated during the long gap):

```
[online_replay seed 0 +     5s] inputs match their checksums: step2 1232, dataset ok, evidence 171, longgap_receipt 391, longgap_features 194
[online_replay seed 0 +     8s] replaying ordinary into /scratch/xy3257/mv2_runs/replay1/online/ordinary
[online_replay seed 0 +   396s] ordinary: {'identical': 28}
[online_replay seed 0 +   396s]   baseline/fifo                    +3 position MSE replay 0.023793 tracked 0.023793 (identical)
[online_replay seed 0 +   396s]   baseline/attention               +3 position MSE replay 0.009147 tracked 0.009147 (identical)
[online_replay seed 0 +   396s]   student/set/use_time_position    +3 position MSE replay 0.021401 tracked 0.021401 (identical)
[online_replay seed 0 +   396s]   reference/b                      +3 position MSE replay 0.006052 tracked 0.006052 (identical)
[online_replay seed 0 +   396s]   reference/posthoc_best           +3 position MSE replay 0.002535 tracked 0.002535 (identical)
[online_replay seed 0 +   396s] replaying long_gap into /scratch/xy3257/mv2_runs/replay1/online/long_gap
```

`replay2.log` (job 41941, `--intervals long_gap`):

```
[online_replay seed 0 +    16s] inputs match their checksums: step2 1232, dataset ok, evidence 143, longgap_receipt 391, longgap_features 194
[online_replay seed 0 +    20s] replaying long_gap into /scratch/xy3257/mv2_runs/replay2/online/long_gap
[online_replay seed 0 +   859s] long_gap: {'identical': 28}
[online_replay seed 0 +   859s]   baseline/fifo                    +3 position MSE replay 0.042148 tracked 0.042148 (identical)
[online_replay seed 0 +   859s]   baseline/attention               +3 position MSE replay 0.025911 tracked 0.025911 (identical)
[online_replay seed 0 +   859s]   student/set/use_time_position    +3 position MSE replay 0.041540 tracked 0.041540 (identical)
[online_replay seed 0 +   859s]   reference/b                      +3 position MSE replay 0.017639 tracked 0.017639 (identical)
[online_replay seed 0 +   859s]   reference/posthoc_best           +3 position MSE replay 0.004508 tracked 0.004508 (identical)
[online_replay seed 0 +   859s] wrote /scratch/xy3257/mv2_runs/replay2/online_replay_seed0.json
```
