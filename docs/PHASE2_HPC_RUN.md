# Course HPC execution record — 20 October 2026

A100 job **9285 completed successfully** on 20 October 2026, followed by completed fresh CPU predictor, deletion-supervision and six-policy validation pilots for Gate and Motion. The hardware UUID was checked against both PyTorch and `nvidia-smi`; official I-JEPA encoded real PushT observations and all 128 new controlled episodes. These are development execution results; model-driven planning validation remains incomplete.

## Final validation, backups and resource release

At source commit `6ac378b3d24a39611cb55c862e934e06381cbae0`, the six actual retention baselines completed in **7.51 CPU seconds**: **96 episode/policy rows and 192 h3/h6 losses** over 16 validation episodes (eight independent pairs). FIFO and temporal buckets had the lowest mean losses in this small pilot. Checkpoint selection had already used this validation split; the comparison is not untouched-test confirmation. Observation capacity was equal (R=4, K=2), while total auxiliary bytes differed. See [six-policy methods and results](PHASE2_RETENTION_BASELINE_PILOT.md).

The final local independent suite passed **181 tests, zero failures/errors/skips**, plus both published-evidence verifiers, dependency checks and a real PushT trace. The 29 recorded source/test hashes match the executed code in `6ac378b`. The separate Linux six-policy preflight passed 18 tests. [Software evidence](../results/phase2_independent/20261020/latest_software_checks.json) preserves the check scope; it does not itself claim model evaluation.

All **472 files** in the full new development-data/model backup and all **nine files** in the six-policy backup were verified after download. An independent audit reran all 1,428 deletion losses from the frozen models and raw inputs, agreeing within **4.79e-8** across different PyTorch versions (tolerance 1e-6). A separate baseline audit recomputed all 192 losses from saved predictions exactly, checked pair aggregates, online traces and checkpoint identities; this latter audit did not rerun model inference. The original archives remain outside Git, while portable manifests and summaries are committed.

After these checks and downloads, the remaining CPU job **9280 was canceled**. A fresh reload of My Interactive Sessions showed **9280 Completed**; immediately before cancellation it was the only queued/running job shown for the account. A100 job 9285 had already completed with scheduler-recorded elapsed time 169 seconds. No new task jobs remain running. The CPU completion observation comes from the portal, not a final exported `sacct` record; no exact final CPU allocation duration is claimed. See [release and backup receipt](../results/phase2_hpc/20261020/completion_receipt.json).

## Completed A100 and fresh predictor runs

Replacement job **9285** ran on an **NVIDIA A100-SXM4-40GB**, with one visible CUDA device and a matching physical UUID. The exact encoder checkout was `6f8703ca11bf47a996605401ab55e303e403db98`; all recorded source hashes match that commit. Scheduler accounting reports `COMPLETED`, exit `0:0`, **05:05:41–05:08:30 UTC**, one GPU allocated for **169 seconds**. The driver took 169.28 seconds. The earlier failed launch remains recorded below; its final zero-duration accounting does not reconstruct the provisioning interval observed earlier.

- Real PushT paired observations produced finite float32 features of shape **25 × 5120**. Parameters remained frozen, all modules remained in evaluation mode, visible control features matched, and masked observations changed the features. This establishes the encoder interface, not history necessity or planning.
- The controlled encoder stage completed in **117.02 seconds**: **128 episodes / 3,616 frames**, 5120-dimensional official I-JEPA features and **864 audited sample rows**. Train-only normalization was independently recomputed exactly from 64 training episodes / 1,696 frames. Test-split features were encoded, but no test target samples or prediction metrics were generated.
- After the encoder process exited, the checkout advanced to `1d18e93a6fe8f8e23a358407a0d5ec399f8ede23`. **15 targeted Linux checks passed**. The new CPU predictor pilot ran **05:08:53–05:09:08 UTC**, taking **15.32 seconds**, on existing CPU job 9280.
- Both Gate and Motion completed **4 epochs / 24 optimizer steps**, seed 17, six varied-history training modes and 0.15 whole-memory dropout. Each model has 211,040 parameters, input rank 128 and train-fitted output rank 95. Actual parameter changes and exact frozen validation replay were verified. Validation selected epoch 1 for both tasks; its MSE was **0.107931** (Gate) and **0.154756** (Motion). These are development checkpoint-selection scores, not a baseline comparison or generalization finding.

The mean validation MSE before any optimizer step was **0.107552** (Gate) and **0.154291** (Motion). The selected trained checkpoints are about **0.35% / 0.30% worse**, and epochs 2–4 deteriorated further. This small pilot therefore establishes execution and freezing, **not improved prediction quality**. The negative development result is retained; it is not a new pass/fail gate from the proposal.

The new models preserve the supplied predictor architecture and locked source, and do not reproduce the teammate's unavailable repaired Phase 1 checkpoints. No retention scorer or model-driven planner was trained in these runs. A subsequent bounded deletion pilot completed separately: 238 new development parents (190 train / 48 validation), 714 equal-capacity forks and 1,428 finite h3/h6 losses in 10.21 CPU seconds. Thirty parents reproduced exactly. See [deletion evidence](PHASE2_DELETION_PILOT.md). This does not establish scorer quality or full scientific Phase 2 completion.

The completed evidence export contains **43 files**, all independently SHA-256 verified after download. All recorded encoder and trainer source hashes were also checked against their respective Git commits. Full originals live in ignored `build/hpc-20261020/training-completed/`; portable records are [`a100_encoding.json`](../results/phase2_hpc/20261020/a100_encoding.json), [`fresh_predictor_training.json`](../results/phase2_hpc/20261020/fresh_predictor_training.json), and [`scheduler_after_encoding.json`](../results/phase2_hpc/20261020/scheduler_after_encoding.json). The previous initial export remains unchanged.

## Verified CPU work

The CPU environment uses Python 3.11.14, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, transformers 5.5.4, NumPy 1.26.4 and gym-pusht 0.1.8. Dependency checks and imports passed. Preparation downloaded the official I-JEPA snapshot pinned to `f157467ea509bc356ff9f61fd3c0d840eec5e04e`; downloading weights is separate from validating them in a GPU execution.

Between **04:55:36 and 04:56:01 UTC**, the independent runner passed **138 tests: 112 Phase 2 and 26 inherited, with zero failures, errors or skips**. Both published Phase 1 evidence verifiers, `pip check` and the real PushT interface trace also passed. Counts were verified from the archived XML. Every one of the 19 recorded source/test hashes matches the exact commit, including the corrected controlled-data resource-accounting code.

A fresh controlled Gate/Motion dataset was generated on the remote CPU: **128 episodes, 64 pairs and 3,616 RGB frames**, seed **20261020**. Native validation and the observation-boundary audit passed. Each task has 16 training pairs and four pairs in each other split; all `test_*` roles remain development smoke. Recent length 4, capacity 2, horizons 3/6 and inherited dynamics are unchanged. This is new data, not recovery of the teammate's missing dataset. No predictor, scorer, scientific deletion dataset or model-driven planning result was produced by these CPU stages.

## A100 allocation snapshot

Job **9284** was submitted at **04:56:12 UTC** to `c12m85-a100-1`, with the exact source commit above. At the archived snapshot, `scontrol` reported `CONFIGURING` and `sacct` reported `RUNNING`; allocation accounting listed one GPU, 12 CPUs and 24 GiB RAM, with a 45-minute limit. Those scheduler states establish resource allocation only. No physical GPU UUID, successful CUDA invocation or encoder result is contained in this snapshot.

The batch script requested `--no-requeue`, but the actual scheduler record was **`Requeue=1`, `Restarts=0`**. Two `scontrol update` attempts were reported to return an unspecified error; their output is not included in this exported archive. Requeue was therefore **not verified disabled**. Later scheduler or encoder results must be recorded as a new observation, without rewriting this initial state.

CPU preparation ran in job **9280**, whose exported allocation listed CPU and memory resources without a GPU TRES. The dataset preparation manifest correctly leaves GPU allocation seconds unknown under Slurm and points to scheduler accounting. The independent CPU runner's zero submission/use counters are scoped to that command; they are not total session accounting and do not cover job 9284. Final allocation use requires the completed scheduler record, including setup and idle time.

### Later environment-retrieval failure

After the archived snapshot, the operator observed job **9284** back in pending/held state with reason **`user env retrieval failed requeued held`**. No `slurm.log` or hardware proof had appeared. This is a launch-stage environment failure, not a completed encoding run. This later observation was reported during execution and is not part of the initial exported archive linked below.

Before the guarded retry, the scheduler already reported 9284 `CANCELLED`, retaining the environment-retrieval failure reason. A replacement, **9285**, was submitted at **05:04:06 UTC**, preserving the exact source commit and original receipt. It exports only a small normal shell environment through explicit `--export=ALL --no-requeue`, with a shorter 40-minute limit. The immediate scheduler receipt still showed `Requeue=1`; no claim is made that requeue was disabled. No `SBATCH_*` variables were present in the submitting process, so inherited `SBATCH_*` options do not explain the observed failure. Successful recovery requires actual worker output; these later observations are not in the initial archive.

## Portable proof and provenance

- [`cpu_checks.json`](../results/phase2_hpc/20261020/cpu_checks.json) preserves the original CPU result, source/log hashes, XML-derived counts, controlled-data preparation/validation and the separate scheduler snapshot.
- [`environment_preparation.json`](../results/phase2_hpc/20261020/environment_preparation.json) preserves the preparation commands/status and package versions. Both records include hashes of their original manifests and the exported evidence index.

All **14 exported file hashes** were independently checked after download. Task-specific absolute paths are replaced with `<TASK_ROOT>` in portable copies; raw XML hostnames, private node names and user/group identifiers are omitted. Full original evidence is retained locally in `build/hpc-20261020/initial/`, which is excluded from Git.

The original `environment.txt` editable-package line references earlier commit `514012d2325b29b78882a0850b72186e0774a3a4`, because that listing was captured during environment preparation. It is not the later test-source identity. The CPU manifest and independently verified file hashes establish the executed `6f8703c` source; neither the old listing nor its hash was rewritten.

For the intended GPU stages and limits, see [the pipeline contract](PHASE2_HPC_PIPELINE.md). Completion of preparation or encoding alone does not establish scientific Phase 2 completion or Phase 1 baseline-model planning.
