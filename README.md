# Bounded Visual Memory for World Models

A world model that can keep only a few past frames has to decide which ones to keep. This project
studies that decision on controlled tasks: every method sees the same **four recent frames** and
may keep **two memory items** (R = 4, B = 2), on top of a frozen I-JEPA ViT-H/14 encoder and a
frozen future predictor. Methods differ only in what fills the two memory slots.

This file is the single guide: [results](#current-results) · [layout](#repository-layout) ·
[1. environment](#1-set-up-the-environment) · [2. models](#2-get-the-models-download-or-train) ·
[3. evaluation](#3-evaluate-reproduce-the-panel) · [checks](#checks) · [next steps](#next-steps).

## Current results

The results panel is `reports/memory_results_20261111/memory-panel-baselines.html` (open it in a
browser). Its numbers are frozen in `configs/reference_values.json`; section 3 recreates them.

| Environment | Metric | No memory | Oracle | Strongest baseline | Ours (deletion-cost scorer) |
|---|---|---:|---:|---:|---:|
| **Gate**: an arrow panel shows a direction once, early; later the ball takes the upper or lower route | route accuracy, validation, higher is better | 50.00 | 95.57 | 90.28 (PaFu-KV) | **94.53** |
| **Synergy**: three cue cards arrive, one must be deleted; two cards decide the future together | prediction regret, locked confirmation, 2,304 S records, lower is better | 0.0476 (FIFO) | 0.0000 | −0.0046 (Compress to Remember) | **0.0036** |
| **Motion v2**: a ball is kicked into a new lane, then hidden | position MSE at +3, validation, lower is better | 0.0231 | 0.0061 | 0.0091 (attention) | 0.0214 |
| **PushT**, hidden centre of gravity | latent prediction at +12 | not run | not run | not run | not run |

What to keep in mind when reading the table:

- **Gate** is scored on the validation set (192 episodes, three predictor seeds, two horizons); there is no independent test set. The oracle knows where the cue is. Compress to Remember as published reads the whole history and reaches 95.31; restricted to two slots it reaches 79.51 and is unstable across seeds (50.5 / 93.8 / 94.3).
- **Synergy**: regret is error above the best single deletion, not MSE. Compress to Remember blends the three cards into two learned items instead of deleting one and scores below its own teacher; the cause is untested. The next baselines are the distilled-sensitivity scorer (0.0046) and full-set sensitivity (0.0077).
- **Motion v2**: our scorer never improved on its initial weights during training, so 0.0214 is an untrained network. The three literature baselines were not run on Motion v2.
- **PushT**: the corpora exist; no predictor has been trained.

Every row of the panel (16 methods on Gate, 16 on Synergy, 14 on Motion v2), its dataset, split,
seeds, checkpoints, tasks and status is listed in `configs/reproduction.yaml`.

## Repository layout

| Where | What |
|---|---|
| `src/bounded_memory/methods/deletion_cost/` | Our method: scorer, teacher, online retention, training, ablations |
| `src/bounded_memory/methods/{pafu_kv,pacc,framemorrow}/` | The three literature baselines (PaFu-KV, Compress to Remember, FrameMorrow), one file per environment |
| `src/bounded_memory/environments/{gate,synergy,motion_v2,pusht}/` | Data generation, feature encoding, predictor training, asset reading, protocol constants |
| `src/bounded_memory/{encoders,predictors,evaluation,data}/` | Pinned encoders, predictor architectures (`predictors/phase1/` is the frozen predictor source), metrics, oracles, heuristics, I/O |
| `configs/reproduction.yaml` | The manifest: every result, how it is produced, and every compute task |
| `configs/<environment>/*.yaml` | One config per asset set; paths use `${BM_SCRATCH}`, `${BM_RUN}`, `${BM_REPO}` |
| `scripts/<task>/<environment>.py` | Thin entry points: explicit config, seed, device, inputs and output folder |
| `torch/` | `cluster.sh` (every cluster setting, as environment variables), `setup_env.sbatch`, and `make_jobs.py`, which turns manifest tasks into Slurm arrays |
| `burst/` | The same tasks on the course cluster: one GPU job that runs many tasks through `scripts/run_tasks.py` (see `burst/README.md`) |
| `runs/<run_id>/` | Small JSON outputs and aggregate tables of one run; heavy files stay on scratch |
| `tests/` | Guards (code moved unchanged), old-versus-new migration checks, unit tests |
| `docs/`, `results/`, `reports/`, `handoff/`, `tools/` | The historical record, unchanged (see the last section) |

## 1. Set up the environment

One environment for everything: Python 3.11 and `requirements.txt`.

**Your own cluster settings.** No file in the repository holds an account or a personal path.
Each one is an environment variable defined in `torch/cluster.sh`, with a default built from your
cluster user name, so teammates can share the same code and differ only in these values:

| Variable | What it is | Default |
|---|---|---|
| `BM_ACCOUNT` | Slurm account to charge | your Slurm default account |
| `BM_SCRATCH` | datasets, features, checkpoints, logs and run outputs | `/scratch/$USER/cv_project_runs` |
| `BM_REPO` | the checkout on the cluster (the Mutagen target) | detected from where the scripts are |
| `BM_HF_HOME` | Hugging Face cache and token | `/scratch/$USER/hf_cache` |
| `BM_CONDA_SH` | conda's activation script | `/scratch/$USER/miniconda3/etc/profile.d/conda.sh` |
| `BM_CONDA_ENV` | the environment every job activates | `/scratch/$USER/miniconda3/envs/cvmem` |
| `BM_SBATCH_ARGS` | extra `sbatch` options for every job (partition, QOS, constraint) | none |
| `BM_GPU_TYPE` | GPU model for the tasks that need a GPU (`l40s`, `a100`, ...) | any GPU |
| `BM_GPU_SBATCH_ARGS` | extra `sbatch` options for GPU tasks only | none |

To change one, do not edit `torch/cluster.sh`. Either keep your values in `torch/.env` (one
`KEY=value` per line, no quotes; git ignores the file and Mutagen syncs it to the cluster):

```bash
# torch/.env
BM_ACCOUNT=torch_pr_000_mylab
BM_GPU_TYPE=a100
```

or export it for a single command:

```bash
ssh torch 'BM_ACCOUNT=torch_pr_000_mylab bash /scratch/$USER/CV_project/torch/generated/<run_id>/submit.sh'
```

An exported variable wins over `torch/.env`, which wins over the default. `submit.sh` prints the
values it used. In the commands of this guide, `torch` is the SSH alias of the cluster login node
and `/scratch/$USER/CV_project` is the checkout; replace them if yours differ.

**On the cluster.** Install Miniconda (under `/scratch/$USER/miniconda3`, or point `BM_CONDA_SH`
and `BM_CONDA_ENV` at your own), then build the environment:

```bash
ssh torch 'bash /scratch/$USER/CV_project/torch/submit_setup_env.sh'
```

This creates the conda environment (Torch 2.5.1 with CUDA 12.1, then `requirements.txt`) and
caches the pinned I-JEPA encoder under `BM_HF_HOME`.

**On a laptop** (tests, aggregation, the panel):

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps        # optional: makes bounded_memory importable everywhere
```

**Hugging Face access.** The Gate models and dataset are private team repositories
(`cv-project-team5guys/gate-retention-models-v1`, `cv-project-team5guys/gate-clean-future-trainval-v1`).
Ask an organisation member for access, create a token with read access to the organisation, and
store it on the cluster under `BM_HF_HOME` (`HF_HOME=$BM_HF_HOME huggingface-cli login`). The Synergy
releases are public.

Edit code locally and let Mutagen sync it; run all compute through `sbatch`, never on a login node.

## 2. Get the models: download or train

All compute is started the same way: pick task prefixes from the manifest, generate the job
arrays locally, submit on the cluster.

```bash
python torch/make_jobs.py --run-id <run_id> --select <task prefixes>
ssh torch 'bash /scratch/$USER/CV_project/torch/generated/<run_id>/submit.sh'
```

The generated files are the same for everyone; `submit.sh` reads your settings on the cluster.
Every seed and variant in the manifest becomes one array index. Jobs run from a copy of `src/`,
`scripts/` and `configs/` taken at submission, skip outputs that already exist, and resume long
training from checkpoints.

| Environment | Download the released models (`--select`) | Train them yourself (`--select`) |
|---|---|---|
| Gate | `gate.assets`: the team bundle (three predictors, the scorers for seeds 17/29/43, dataset, features), checksum-verified | `gate.retrain_scorer`: our scorer on the released predictors. `gate.retrain`: everything from scratch (dataset, I-JEPA features, three predictors, scorer, PaFu-KV) |
| Synergy | `synergy.assets.release`: readers, basis, encoded confirmation set, released students | `synergy.assets.trainvaldev` then `synergy.retrain`: re-encoded train/validation features, teacher labels, all student arms. `synergy.retrain_readers`: the two readers |
| Motion v2 | not available: the frozen readers, features and dataset were never released | `motion_v2.retrain.students`: the 54 students from the tracked teacher labels. Reader training is not runnable yet (see next steps) |
| PushT | nothing to download | `pusht.prepare_data`: the two corpora. No predictor exists |

The released Gate predictors and Synergy features and readers were produced on a GPU, and a CPU
gives slightly different features. The tasks above run on CPU. Each has a GPU twin that repeats it
on the hardware class of the release: `gate.retrain_gpu`, `synergy.assets.trainvaldev_gpu` with
`synergy.retrain_gpu`, and `synergy.retrain_readers_gpu`. Set `BM_GPU_TYPE` for them.

Where a GPU is scarce, one job should do all the GPU work instead of one job per seed.
`scripts/run_tasks.py` runs the selected tasks inside one allocation, in dependency order, three
at a time on the one GPU, and continues an unfinished run when started again. The course cluster
(NYU cloud bursting, one A100) uses it: [`burst/README.md`](burst/README.md) has the commands and
says where the results appear. Those scripts have not run on the course cluster yet.

The literature baselines have no released checkpoints. They are trained inside the evaluation
tasks of section 3 (`gate.replay.{pafu_kv,pacc,framemorrow}`, `synergy.replay.literature`).

## 3. Evaluate: reproduce the panel

Two paths are kept apart. **Replay** loads the released checkpoints and scores them again; this
recreates the panel. **Retrain** produces new measurements, reported in their own tables and never
written into the panel.

```bash
# replay: downloads, old-versus-new checks, every bar of the panel
python torch/make_jobs.py --run-id <run_id> --select gate.assets gate.validate gate.replay
python torch/make_jobs.py --run-id <run_id> --select synergy.assets synergy.validate synergy.replay
python torch/make_jobs.py --run-id <run_id> --select motion_v2.replay
ssh torch 'bash /scratch/$USER/CV_project/torch/generated/<run_id>/submit.sh'      # after each generate step

# retrain (optional): see the table in section 2
python torch/make_jobs.py --run-id <run_id> --select gate.retrain_scorer gate.retrain synergy.retrain synergy.retrain_readers motion_v2.retrain

# pool the run and rebuild the panel (locally, once Mutagen has synced runs/<run_id>/ back)
python scripts/aggregate_results/aggregate.py --run runs/<run_id>      # per-seed and pooled tables, differences from the frozen values
python scripts/aggregate_results/build_panel.py --run runs/<run_id>    # runs/<run_id>/aggregate/panel.html and panel_check.json
```

`runs/<run_id>/aggregate/tables.md` lists every result per seed and pooled, with a verdict against
`configs/reference_values.json`. The manifest gives each result a tolerance: `exact` for
deterministic replay, `tie_sensitive` where exactly tied candidates are broken by CPU rounding,
`per_seed` for models trained in the run, `unstable` for training known to flip between runs.

| Environment | Replay | Retrain |
|---|---|---|
| Gate | every bar of the panel, from the team bundle | scorer on the released predictors; the full chain from scratch |
| Synergy | every bar of the panel, from the public releases, for both readers and R/D/S | teacher labels and every student arm on the released readers; the two readers on re-encoded features |
| Motion v2 | the published tables, recomputed from the tracked loss arrays; policies cannot be re-scored without the frozen readers and features | the 54 students from tracked labels |
| PushT | nothing to replay | data preparation only |

The latest run is `runs/repro_20261111/`; its tables and panel are in `runs/repro_20261111/aggregate/`,
and `panel_check.json` there counts the bars that match the frozen values (42 of 46 on 11 November
2026; the four Gate bars of Compress to Remember and FrameMorrow were still training).

## Checks

```bash
python -m pytest -q tests                                 # everything that needs no assets
BM_GATE_ROOT=<Gate asset tree> python -m pytest -q tests/test_migration_gate.py
BM_SYN=<Synergy assets> BM_SYN_STUDENTS=<released students> python -m pytest -q tests/test_migration_synergy.py
```

- `tests/test_refactor_guards.py`: the package holds the same code as the original experiment
  files (compared with commit `fe69074` through git), the historical record is unchanged, and the
  reproduction starts nothing outside `scripts/`.
- `tests/test_run_tasks.py`, `tests/test_burst.py`: the single-job runner (order, tasks at once,
  memory budget, failures, continuing) and the course-cluster scripts, with stand-ins for `sbatch`
  and conda.
- `tests/test_migration_*.py`: old code and new package on the same assets give identical arrays,
  predictions, kept frames, metrics, labels and trained weights. The old code is taken from git
  (`tests/legacy_tree.py`), or from `BM_LEGACY_TREE` where git is not available.

## Next steps

1. **Motion v2 with the new layout.** Obtain the frozen assets from the teammate's storage
   (dataset archive, step-2 readers and features, long-gap features, step-3/4 CPU archive). The
   original run scripts in `src/bounded_memory/environments/motion_v2/pipeline/` verify checksums
   registered under their old paths and require one A100, which this account cannot request:
   re-register them, add `scripts/` entry points as for Gate and Synergy, then rerun the readers
   and the online evaluation and run PaFu-KV, Compress to Remember and FrameMorrow there.
2. **Motion v2, our method.** The deletion-cost students kept their initial weights. Find out why
   the teacher labels are not learnable before comparing against baselines.
3. **PushT.** Encode the v5 marker corpus, train a predictor, fix the metric (latent prediction at
   +12), then run the same ladder: no memory, oracle, baselines, ours.
4. **Gate.** Run Compress to Remember with two slots on more seeds (it fails on one of three on CPU
   and on both seeds tried on GPU). The comparison is on validation data; an independent test set
   has not been generated.
5. **Synergy.** Test why Compress to Remember scores below its own teacher.
6. **Code.** `predictors/dynamics.py` and `predictors/inputs.py` duplicate parts of
   `predictors/phase1/`; remove the duplicates and rerun the migration checks.

## The historical record

`docs/`, `results/`, `reports/` and `handoff/` are unchanged and remain the record of how each
result was first produced: protocols, preregistrations, raw outputs, checksums and reports in
English and Chinese. Start with `docs/GATE_RETENTION_COMPARISON_LATEST.md` (Gate),
`docs/QUALIFIED_READER_RETENTION_DEV_ZH.md` and
`results/literature_diagnosis/qualified-reader-retention-20261025/conf/README.md` (Synergy),
`docs/MOTION_V2_STEP3_4_RECOVERY1_REPORT_ZH.md` (Motion v2) and `docs/PHASE2_PUSHT.md` (PushT).

Those documents and their checksum files name paths under `experiments/`. That folder held the
original experiment code; everything still used was moved into `src/bounded_memory/` and the folder
was removed. A path such as `experiments/gate_clean_future/materialize.py` now means the file at
commit `fe69074`:

```bash
git show fe69074:experiments/gate_clean_future/materialize.py
```

`results/publication_manifest.json` describes the whole tree at that commit and verifies only there.
