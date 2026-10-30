# Bounded real I-JEPA encoding pilot on the existing A100

This wrapper connects the actual PushT RGB/masking CLI to the original frozen
I-JEPA encoder. It performs inference only. It does not train a predictor, prove
history necessity, run MPC, submit a job, or automatically requeue anything.

Run inside the already active single-A100 course allocation, using a new output
directory and cache storage on scratch. Replace the sample paths with the actual
workspace and interpreter; no stale job/node identifier is built into the runner.

```bash
python experiments/phase2/run_encoder_pilot.py \
  --output /scratch/NETID/cv-project/encoder-pilot-NEW \
  --cache-dir /scratch/NETID/cv-project/cache \
  --timeout-seconds 1800
```

The environment needs this repository's model dependencies and the PushT packages
documented in `PHASE2_PUSHT.md`. Import resolution includes the retained Phase 1
source directly, so an editable package install is not required. Missing CUDA,
weights, dependencies, Slurm allocation identity, or hardware identity fails loudly.
The official checkpoint is `facebook/ijepa_vith14_1k` at revision
`f157467ea509bc356ff9f61fd3c0d840eec5e04e`, with the retained official processor,
float32 weights and four-cell grid pooling. Missing weights may download into
the explicitly supplied scratch cache; no random or pixel-only encoder is substituted.

The parent runs one child process with a 1,800-second default timeout. Timeout or
interruption terminates the child's process group (TERM, then KILL if necessary)
and writes a failed/interrupted `run_manifest.json`. The allocation's own Slurm
time limit remains the outer limit. This runner does not extend that allocation.

## Evidence checked

- Exactly one CUDA device is visible and its name is A100. The **actual Torch CUDA
  device UUID** must match `nvidia-smi`; machine inventory order is not used to pick
  a GPU. If the Torch version lacks UUID properties, the runner explicitly records
  a fallback mapping of its own CUDA worker PID from `nvidia-smi compute-apps` and
  rejects missing or ambiguous matches.
- Host, Slurm job ID, GPU UUID/model/memory/driver, Torch/CUDA versions, source and
  encoder-configuration hashes are saved. No environment-variable dump, credential,
  token, cookie, or authentication material is written.
- The original `FrozenIJepa` loader executes. A thin audited subclass asserts all
  encoder modules are in eval mode, all parameters have gradients disabled, and
  all weights are float32 on the verified CUDA device. It records parameter counts,
  resolved processor/model metadata and parameter versions; versions must remain
  unchanged after inference.
- The existing paired PushT CLI generates 25 actual RGB observations per variant,
  with identical seeds/actions, masks applied before encoding and later visible
  targets. All observed/control features must be finite and have shape (25,5120).
  Visible RGB must retain matching features, while blacked-out/control features
  must differ somewhere during the mask interval.

Outputs include `worker.log`, `hardware.json`, `encoder_config.json`,
`frozen_encoder_audit.json`, `encoding_check.json`, `run_manifest.json`, and the
existing paired RGB/action/feature trace under `pusht/`. The manifest hashes saved
artifacts and records elapsed runtime. `baseline_model_planning_verified` and
`history_necessity_validated` remain false even when encoding completes.
