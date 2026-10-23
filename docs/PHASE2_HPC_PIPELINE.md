# Bounded course A100 encoding pipeline

`experiments/phase2/run_hpc_pipeline.py` runs independent preparation/encoding work within an existing allocation. It does not submit jobs, install software, train predictors/scorers, produce deletion supervision, or validate model-based planning.

CPU preparation must provide `TASK_ROOT/repo`, `TASK_ROOT/venv`, and the immutable official I-JEPA snapshot in `TASK_ROOT/cache/ijepa`. Commit and push the complete source first, then place the exact clean commit in the HPC checkout. Pass the complete expected Git SHA; the driver rejects a different HEAD or dirty/untracked checkout. Keep scheduler logs outside the repository.

From a CPU/login session, with actual `TASK_ROOT` and `EXPECTED_COMMIT` values:

```bash
sbatch --chdir="$TASK_ROOT" --output="$TASK_ROOT/slurm-%j.log" \
  "$TASK_ROOT/repo/experiments/phase2/encoder_pipeline.sbatch" \
  "$TASK_ROOT" "$EXPECTED_COMMIT"
```

The batch requests course account `csci_ga_2271-2026fa`, partition `c12m85-a100-1`, one GPU, four CPUs, 24 GiB RAM, 45 minutes, and no requeue. The driver bounds total work to 2,400 seconds and preserves cleanup margin. No submission is made by this documentation or the Python driver.

Stages under `TASK_ROOT/runs/SLURM_JOB_ID`:

1. Run the real CUDA PushT/I-JEPA encoding pilot with a 600-second worker timeout and actual A100 UUID verification. `--cache-dir TASK_ROOT/cache` is correct here: the pilot appends `ijepa`.
2. Generate new small controlled data only if `TASK_ROOT/data` is absent/empty. If already populated, preserve it. Validate the exact seed `20261020`, 16 training pairs per task, four pairs per other task/split, retained source identities, archive checksums, causal observation boundary, and explicit development split roles.
3. Encode controlled observations and audit varied-history/dropout training samples with `--cache-dir TASK_ROOT/cache/ijepa --device cuda --local-files-only`. This stage fits train-only feature normalization but does not train a predictor. Official weights are used offline in both encoding stages.

Every subprocess stage records its command, log, timestamps, timeout, process group, exit code, and log hash. `pipeline_manifest.json` records source hashes, expected/actual commit, scope, hardware proof, and terminal status, including failures/interruption. Timeout/TERM stops the active process group; the pilot also handles TERM by stopping its own worker group. Existing per-job evidence is not overwritten. A later allocation can validate and reuse prepared data and compatible feature caches while writing a fresh run directory.

`completed` means these preparation/encoding/sample-audit stages completed. It does **not** mean scientific Phase 2, baseline prediction comparisons, deletion-supervision generation, or Phase 1 baseline-model planning are complete. Driver wall time is recorded separately from total allocation use; obtain final allocation time from scheduler accounting.
