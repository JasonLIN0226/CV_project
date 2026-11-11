# Motion v2: original run scripts (not runnable from here yet)

These are the entry points that produced the published Motion v2 numbers, copied byte for byte
from `experiments/motion_v2/` at commit `fe69074`:

| File | What it does |
|---|---|
| `run_step1.py` | dataset generation and audit |
| `run_step2.py` | I-JEPA encoding, normalisation, training of the four reference readers, checkpoint selection, state decoder |
| `run_step34.py`, `run_step34_recovery1.py` | teacher labels, student fitting and online scoring (ordinary and long gap) |
| `step34_runtime*.py`, `step34_accounting_recovery1.py`, `slurm_records.py`, `prepare_direction_bound_v2_3.py` | their helpers: allocation checks, time budget, Slurm records, the direction bound |
| `config_*.json`, `protocol_*.json`, `amendment_*.json`, `step34_assets.json` | the frozen configs and the checksums each step verifies before it starts |

Each step refuses to start unless its own source files and inputs match checksums registered
under their old `experiments/motion_v2/...` paths, the allocation is one A100, and the frozen
assets (dataset, features, readers) are present. None of that holds here: the paths moved and the
assets were never released to this account. The modules these scripts call (environment, generator,
reader, metrics, students) are one folder up and are covered by `tests/motion_v2/`.

To rerun Motion v2 with this layout: obtain the frozen assets, re-register the checksums for the
new paths, and add thin `scripts/` entry points as for Gate and Synergy.
