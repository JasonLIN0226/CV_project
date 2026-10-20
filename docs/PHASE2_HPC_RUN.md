# Course HPC execution record — 20 October 2026

The remote Linux CPU preparation and checks passed at exact source commit **`6f8703ca11bf47a996605401ab55e303e403db98`**. At the archived **04:57:24 UTC** snapshot, A100 job **9284** had an allocation and remained `CONFIGURING` in `scontrol`. This record does not establish that CUDA execution or encoding had started.

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
