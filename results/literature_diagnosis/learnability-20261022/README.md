# Causal oracle-learnability diagnostic

Fixed old-TRAIN-only ridge probes; 16 training pairs and 96 fresh development
pairs. `protocol.json` was written before labels/fitting. Cost, attention and
shuffled-cost probes share 903 features/weights and lambda=1. The three retained
parameter/scale arrays are published as JSON. This is not the final proposed
selector or an untouched confirmation result.

`audit.json` independently reconstructs causal training inputs, fresh parent
costs, probe choices, grouped confidence intervals and irreversible trace
invariants. It also describes training fit and input shift without model tuning.
`ambiguity.json` separates complete past from bounded state: the repeated states
are after information has been lost; a better earlier retention policy could
avoid them. `bad_cases.json` gives deterministic selection rules and complete
three-way losses. The full traces and binary training arrays remain in ignored
`build/literature-diagnosis-20261022/learnability-v2`, hash-bound by the raw summary.

The earlier v1 run was superseded only to fix sibling weighting in summary
statistics (average parents within each episode before averaging siblings).
The v2 protocol/hyperparameters were unchanged; fits and per-case predictions
are identical. No outcome-driven model selection was performed.

Run on the recorded local development assets:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/phase2/diagnose_oracle_learnability.py --output build/learnability-NEW
.venv/bin/python experiments/phase2/export_oracle_learnability.py --source build/learnability-NEW --output results/learnability-NEW
```

A later **post-hoc train-support numerical ablation** is preserved in
[`support_mask/`](support_mask/README.md), with its protocol written before
masked replay. It zeroes coefficients on the fixed TRAIN-RMS floor columns
without refitting. Both versions must be reported together: their cost-vs-
attention ranking reverses, and neither cost probe reliably beats shuffled
labels. See the joint [diagnosis](../../../docs/ORACLE_LEARNABILITY_DIAGNOSIS.md).
The original result files and source snapshots above are unchanged.
