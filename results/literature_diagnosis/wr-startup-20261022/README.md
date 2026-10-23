# WR startup: shared initial-observation prefix

Completed fixed diagnostic: 32 existing training episodes and all 192 existing
fresh-development episodes, WR/FIFO/attention/diversity, original input versus
four repeated copies of the first observed frame prepended for **every** policy.
No extra task information, predictor fitting, threshold tuning or GPU work.

- [Frozen protocol](protocol.json), [summary](summary.json), [all 1,792 result rows](episode_results.json).
- [Matched policy comparisons](paired_policy_comparisons.json), [independent arithmetic audit](audit.json).
- [896 exact predictor clock-shift checks](shift_invariance.json).
- [Complete WR scalar/retention trace](wr_trace.jsonl.gz), [two fixed descriptive examples](paired_examples.json).
- [Figure PNG](figures/wr_startup.png) / [PDF](figures/wr_startup.pdf).
- [Artifact manifest](manifest.json), [executed source snapshot](source/diagnose_wr_startup.py).

All 62 early fresh cues are recovered; all 130 later-cue decisions are unchanged.
Among the 62 rescued episodes, feature MSE improves in 41 and worsens in 21.
This isolates admission startup, not the remaining reader/metric problem.
The prefix is an explicit diagnostic counterfactual, not a new deployable baseline
or an original Worth Remembering reproduction.

The original four-policy fresh rows reproduce the prior result exactly. The
predictor's output is unchanged when an original kept set is shifted by four
along with its query/actions. The prefix changes the selection process, so it
is supplied to every control; attention has one changed filler choice, while
FIFO/diversity stay identical on every episode. Same K does not imply equal
time or controller bytes: WR's maximum logical state increases by 32 bytes.

Re-run into a new empty directory using the previously verified local assets:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python experiments/phase2/diagnose_wr_startup.py \
  --output build/literature-diagnosis-NEW/wr-startup
```

The default paths are recorded in that runner; model/features remain outside
Git. The 136,972,645-byte prediction/target archive is retained locally with its
hash in `manifest.json`. To replot using only this portable directory:

```sh
python results/literature_diagnosis/wr-startup-20261022/source/export_report.py \
  --root . --output results/literature_diagnosis/wr-startup-20261022 --render-only
```

The exporter verifies raw predictions and old-row identity when doing a full
export. `--render-only` does no raw-array validation and executes no model.
