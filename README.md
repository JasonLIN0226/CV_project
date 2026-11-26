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
| **PushT** (v5c): the hidden centre of gravity is drawn on the block early, among distractor dots; a later push turns the block by class | latent prediction error at +12, validation, lower is better | 0.743 | 0.474 | 0.618 (visual diversity) | **0.473** |

What to keep in mind when reading the table:

- **Gate** is scored on the validation set (192 episodes, three predictor seeds, two horizons); there is no independent test set. The oracle knows where the cue is. Compress to Remember as published reads the whole history and reaches 95.31. Restricted to two slots, its training sometimes collapses to near chance (50–58), so a three-seed value is a draw: 79.51 in the release (50.5 / 93.8 / 94.3), 95.31 replayed on NYU Burst CPU (95.83 / 94.27 / 95.83), 55.2 and 52.9 on the two Torch GPU seeds tried. Over 10 training seeds on each predictor (30 runs per method, Torch CPU, 8 threads, `runs/gate_seed_sweep_20261124/aggregate/seed_sweep.md`), its three-seed value ranges from 69.01 to 95.14 and ours from 93.92 to 95.49; it is higher on 1 of the 10 training seeds. It collapsed in 5 of 30 runs and averaged 93.60 over the 25 that trained, 87.33 over all (PaFu-KV, one training seed per predictor, is 90.28); ours never collapsed and averaged **94.92**, higher in 24 of the 30 runs with the same predictor and seed number. With 8 threads pinned it still collapses, so the thread count is not the cause; the release's seed-17 collapse did not recur on Torch with 1 or 8 threads (90.89, 95.57). Torch GPU's 2 of 2 is unlikely at one in six but rests on two runs. On a Burst A100, three new initialisation seeds on each predictor (`runs/gate_init_seeds_gpu_20261125/`), neither method collapsed in 9 runs: ours 94.62, two-slot Compress to Remember 94.59, tied (paired +0.03 points, 95% interval −0.78 to +0.95); ours varies less (standard deviation 0.72 against 1.77).
- **Synergy**: regret is error above the best single deletion, not MSE. Compress to Remember blends the three cards into two learned items instead of deleting one and scores below its own teacher; the cause is untested. The next baselines are the distilled-sensitivity scorer (0.0046) and full-set sensitivity (0.0077). Paired against the retrained baselines (48 families, 95% intervals, `runs/synergy_paired_20261125/`): ours has clearly lower regret than FrameMorrow on R, D and S (item_attention); against Compress to Remember it is lower on S on item_attention but not significantly (−0.0023, interval −0.0051 to +0.0006), and clearly higher on R (both readers), on D (item_attention) and on S (multihead).
- **Motion v2**: our scorer never improved on its initial weights during training, so 0.0214 is an untrained network. The three literature baselines now run on Motion v2 (`runs/memory_gaps_20261124/README.md`): none beats attention (+3: PaFu-KV 0.0094, two-slot Compress to Remember 0.0190, FrameMorrow 0.0207). The frozen readers memorised the training episodes (attention 0.0044 on them, 0.0095 on fresh scenes from the same generator), so our scorer was trained on fresh scenes (`environments/motion_v2/fresh.py`, `runs/motion_v2_fresh_20261125/`): the variant chosen on fresh dev (0.0097 against attention's 0.0104) does not differ from attention on a fresh holdout (0.0105 against 0.0108, 95% interval −0.0020 to +0.0014), so validation was not read. The online validation now replays on the frozen assets (28 of 28 methods identical on both intervals, `runs/motion_v2_frozen_replay_20261124/README.md`).
- **PushT** (v5c; 96 validation episodes, three predictor seeds; `runs/pusht_cog_20261124/README.md`): error is the mean squared error over 5,120 standardized I-JEPA features at query + 12; the oracle keeps the marker frames (privileged). Our scorer keeps a marker frame in 287 of 288 episode runs (96 episodes × 3 seeds; one miss in seed 43) and matches the oracle, because one marker frame carries almost all the value: with only one marker frame in memory the error is within ±0.02 of the oracle's two; visual diversity keeps one in 44% (without distractors, on v5b, it kept 99% and tied ours at 0.498 vs 0.501); FIFO, temporal buckets and random deletion stay at no memory. Sibling identification (is the prediction closest to its own sibling's future; chance 0.33): ours 0.87, oracle 0.86, visual diversity 0.57. The literature baselines were not run on PushT.

Every row of the panel (16 methods on Gate, 16 on Synergy, 14 on Motion v2), its dataset, split,
seeds, checkpoints, tasks and status is listed in `configs/reproduction.yaml`.

Plain-language progress notes from team members, including work not yet merged here: `progress/`.

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
| `torch/` | `cluster.sh` (every cluster setting, as environment variables), `setup_env.sbatch`, `make_jobs.py`, which turns manifest tasks into Slurm arrays, and `submit_pack.sh`, one GPU job for many tasks |
| `burst/` | The same tasks on the course cluster: one GPU job that runs many tasks through `scripts/run_tasks.py` (see `burst/README.md`) |
| `runs/<run_id>/` | Small JSON outputs and aggregate tables of one run; heavy files stay on scratch |
| `tests/` | Guards (code moved unchanged), old-versus-new migration checks, unit tests |
| `docs/`, `results/`, `reports/`, `handoff/`, `tools/` | The historical record, unchanged (see the last section) |

## 1. Set up the environment

All environments use Python 3.11 and `requirements.txt`.

**On Burst (course OOD)**, run `bash burst/setup.sh` from the checkout. It creates a uv
environment at `.venv`, installs the CUDA 12.1 dependencies, and caches the encoder. Burst
selects this environment automatically for interactive and batch runs. See the
[Burst guide](burst/README.md) for login and launch commands.

**Your own Torch cluster settings.** Torch reads your account and paths from the environment.
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

**On Torch.** Install Miniconda (under `/scratch/$USER/miniconda3`, or point `BM_CONDA_SH`
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
| Motion v2 | not released; the frozen readers, features and dataset are on a teammate's Burst scratch, checksum-verified by `configs/motion_v2/frozen.yaml` | `motion_v2.retrain.students`: the 54 students from the tracked teacher labels. Reader training is not runnable yet (see next steps) |
| PushT | the team bundle [`cv-project-team5guys/pusht-hidden-cog-v1`](https://huggingface.co/datasets/cv-project-team5guys/pusht-hidden-cog-v1) (private Hugging Face dataset; download with `hf download cv-project-team5guys/pusht-hidden-cog-v1 --repo-type dataset --local-dir <dir>`): train/validation corpora and I-JEPA features of v5c, v5b, v5b_t128, v5 and v4, the v5c and v5b predictors and scorers (seeds 17/29/43), and the results; `SHA256SUMS` checks every file. Its README says which data each checkpoint was trained on | `pusht.prepare_data` (v5, v4, v5b, v5c), then for v5b/v5c `pusht.encode` (GPU), `pusht.train_predictor`, `pusht.retention` |

The released Gate predictors and Synergy features and readers were produced on a GPU, and a CPU
gives slightly different features. The tasks above run on CPU. Each has a GPU twin that repeats it
on the hardware class of the release: `gate.retrain_gpu`, `synergy.assets.trainvaldev_gpu` with
`synergy.retrain_gpu`, and `synergy.retrain_readers_gpu`. Set `BM_GPU_TYPE` for them.

Where a GPU is scarce, one job should do all the GPU work instead of one job per seed.
`scripts/run_tasks.py` runs the selected tasks inside one allocation, in dependency order, three
at a time on the one GPU, and continues an unfinished run when started again; the state of every
task is in `runs/<run_id>/pack_status.txt`.

```bash
ssh torch 'bash /scratch/$USER/CV_project/torch/submit_pack.sh --run-id <run_id> --select <task prefixes>'
```

On NYU Burst the runner defaults to one A100. Set `BM_GPU_COUNT=2` to schedule ready CUDA tasks
on two GPUs. The [Burst guide](burst/README.md) gives the launch command and result paths.

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

# Gate seed sweep (optional): two-slot Compress to Remember and ours, training seeds 1 to 10 on each predictor
python torch/make_jobs.py --run-id <run_id> --select gate.assets.team_bundle gate.sweep
# the same on a GPU with training seeds 101, 202, 303 (gate.sweep_gpu); Synergy literature baselines on
# GPU-encoded features, paired with our students per family (synergy.literature_gpu.train, synergy.literature_gpu.paired)

# Motion v2 on the frozen study assets (BM_MOTION_ROOT; on Burst, see burst/README.md): online replay,
# literature baselines, and our scorer on fresh scenes (holdout check; validation only after a holdout gain)
bash burst/submit.sh --run-id <run_id> --select motion_v2.online motion_v2.literature motion_v2.fresh

# pool the run and rebuild the panel (locally, once Mutagen has synced runs/<run_id>/ back)
python scripts/aggregate_results/aggregate.py --run runs/<run_id>      # per-seed and pooled tables, differences from the frozen values
python scripts/aggregate_results/build_panel.py --run runs/<run_id>    # runs/<run_id>/aggregate/panel.html and panel_check.json
python scripts/aggregate_results/gate_seed_sweep.py --run runs/<run_id> # the seed sweep: collapses, means, head-to-head (aggregate/seed_sweep.md)
python scripts/aggregate_results/gate_seed_sweep.py --run runs/<run_id> --sweep sweep_gpu  # the GPU sweep (aggregate/seed_sweep_gpu.md)
```

`runs/<run_id>/aggregate/tables.md` lists every result per seed and pooled, with a verdict against
`configs/reference_values.json`. The manifest gives each result a tolerance: `exact` for
deterministic replay, `tie_sensitive` where exactly tied candidates are broken by CPU rounding,
`per_seed` for models trained in the run, `unstable` for training known to flip between runs.

| Environment | Replay | Retrain |
|---|---|---|
| Gate | every bar of the panel, from the team bundle | scorer on the released predictors; the full chain from scratch |
| Synergy | every bar of the panel, from the public releases, for both readers and R/D/S | teacher labels and every student arm on the released readers; the two readers on re-encoded features |
| Motion v2 | the published tables, recomputed from the tracked loss arrays; with the frozen assets, `scripts/evaluate/motion_v2_online.py` re-scores every policy online (one GPU) | the 54 students from tracked labels; with the frozen assets, the literature baselines and our scorer on fresh scenes |
| PushT | nothing to replay | the full chain: corpora, encoding and probes, predictor, retention (recorded in `runs/pusht_cog_20261124/`) |

The latest run is `runs/repro_20261111/`; its tables and panel are in `runs/repro_20261111/aggregate/`,
and `panel_check.json` there counts the bars that match the frozen values: 43 of 46 identical; 2 within
the per-seed tolerance (Compress to Remember on the whole history 95.14 against 95.31, FrameMorrow
54.51 against 54.34); and two-slot Compress to Remember at 95.31 against 79.51 (see Current results).
The Gate files of Compress to Remember and FrameMorrow were produced on NYU Burst CPU on 12 November
(`runs/repro_20261111/pack_status.txt`); everything else ran on Torch. The Gate seed sweep is
`runs/gate_seed_sweep_20261124/`: the sweep ran the code of commit b626562 (`run.json` records fd8cc3e
with those changes not yet committed), the two thread probes ran fd8cc3e.

## Checks

```bash
python -m pytest -q tests                                 # everything that needs no assets
BM_GATE_ROOT=<Gate asset tree> python -m pytest -q tests/test_migration_gate.py
BM_SYN=<Synergy assets> BM_SYN_STUDENTS=<released students> python -m pytest -q tests/test_migration_synergy.py
```

- `tests/test_refactor_guards.py`: the package holds the same code as the original experiment
  files (compared with commit `fe69074` through git), the historical record is unchanged, and the
  reproduction starts nothing outside `scripts/`.
- `tests/test_run_tasks.py`, `tests/test_burst.py`, `tests/test_torch_pack.py`: the single-job runner (order, tasks at once,
  memory budget, failures, continuing) and the Burst and Torch launch scripts, using a local stand-in for Slurm.
- `tests/test_migration_*.py`: old code and new package on the same assets give identical arrays,
  predictions, kept frames, metrics, labels and trained weights. The old code is taken from git
  (`tests/legacy_tree.py`), or from `BM_LEGACY_TREE` where git is not available.

## Next steps

1. **Motion v2 literature baselines.** Done on the frozen assets (`scripts/train_baselines/motion_v2.py`,
   `runs/memory_gaps_20261124/`). Retraining the readers still needs the original run scripts in
   `src/bounded_memory/environments/motion_v2/pipeline/` re-registered.
2. **Motion v2, our method.** Training on fresh scenes (tasks `motion_v2.fresh.*`) removed the readers'
   memorisation but the scorer still does not beat attention on the fresh holdout
   (`runs/motion_v2_fresh_20261125/`). Find out why it cannot learn which frames to keep, although the best
   pair is far better (0.0030 against 0.0108); validation stays unread (`motion_v2.validate.deletion_cost_v2`
   refuses without a holdout gain).
3. **PushT.** v5c runs the full ladder except the literature baselines (`runs/pusht_cog_20261124/`).
   Next: PaFu-KV, Compress to Remember and FrameMorrow on v5c; the natural variant v4 (the cue is a push
   and the block's response, read against the present) with the same predictor and scorer; then planning
   with the learned predictor.
4. **Gate.** The comparison is on validation data; an independent test set has not been
   generated. Two-slot Compress to Remember collapsed in 5 of 30 CPU runs and 2 of 2 Torch GPU runs,
   but in none of 9 Burst GPU runs (`runs/gate_init_seeds_gpu_20261125/`); the Torch GPU collapses are
   not explained.
5. **Synergy.** Test why Compress to Remember scores below its own teacher.
6. **GPU retraining on Burst** (run `ood_gpu_repro`, outputs on branch `debug/burst-repro-20261112`).
   `gate.retrain_gpu.assets` and `synergy.retrain_readers_gpu.train` stopped on the deterministic
   cuBLAS error, which is now fixed, so the 31 tasks after them have not run. Rerun the same command
   to continue. In that run our Synergy student with the multihead reader failed on seed 29
   (regret 0.0271, against 0.0037 and 0.0038 on the other two seeds).
7. **Code.** `predictors/dynamics.py` and `predictors/inputs.py` duplicate parts of
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
