# Frozen clean-future predictor source

The October 22 clean-future Gate experiment used a newer `probes.py`, saved
here. The other predictor source files retain their historical paths and match
the clean-future source lock. The older `probes.py` stays in the historical
Phase 1 source tree to preserve that publication's reproducibility.

Verify the full 53-file clean-future lock with:

```bash
python tools/verify_gate_clean_future_source.py
```

To reconstruct the exact predictor source in a new local directory:

```bash
python experiments/gate_clean_future/frozen_control/materialize_snapshot.py --output ./work/clean_future_control
export PYTHONPATH="$PWD/work/clean_future_control/src:$PWD/experiments/predictor/control/src:$PWD"
```

The original HPC-specific launchers and paths still require adaptation for a
different machine. No training or data regeneration is needed to reconstruct
the source snapshot.
