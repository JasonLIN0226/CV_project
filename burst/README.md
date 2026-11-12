# Running on the course cluster (NYU cloud bursting)

The course partition `c12m85-a100-1` (account `csci_ga_2271-2026fa`) has one A100 per job. One
job therefore runs the whole GPU chain: `scripts/run_tasks.py` starts the tasks of `configs/reproduction.yaml` in
dependency order and keeps three running at once on the one GPU. The tasks are the same as on
Torch (`torch/`); only how they are started differs.

**Not yet run on the course cluster.** The scripts are tested locally with stand-ins for `sbatch`
and conda (`tests/test_burst.py`, `tests/test_run_tasks.py`). Unverified there: `sbatch` from an
Open OnDemand session, the job size, and `gpu_driver.sh`.

## 1. Set up once

Open a terminal in a session at https://ood.burst.hpc.nyu.edu, then:

```bash
cd /scratch/$USER
git clone -b clean https://github.com/JasonLIN0226/CV_project.git
cd CV_project
bash burst/setup.sh        # about 15 minutes: Miniforge, the Python 3.11 environment, the I-JEPA weights
bash burst/hf_login.sh     # paste a Hugging Face token that can read the team organisation (Gate bundle)
```

## 2. Run

```bash
# what the job would run; nothing is submitted
bash burst/submit.sh --run-id burst_a100 --dry-run --select gate.assets.team_bundle synergy.assets.release

# a first small job (a few minutes): the downloads. Checks the account, the environment, the token and the GPU.
bash burst/submit.sh --run-id burst_a100 --select gate.assets.team_bundle synergy.assets.release

# the full GPU chain: Gate from scratch, Synergy features, labels, students and readers
bash burst/submit.sh --run-id burst_a100 --select gate.assets.team_bundle synergy.assets.release \
    gate.retrain_gpu synergy.assets.trainvaldev_gpu synergy.retrain_gpu synergy.retrain_readers_gpu \
    --skip gate.retrain_gpu.compare_cpu
```

`gate.retrain_gpu.compare_cpu` is left out because it compares with the tree built on CPU, which
is on Torch.

Running the same command again continues an unfinished run: after a time limit, a failure, or a
cancelled job, finished tasks are skipped. If the GPU runs out of memory, run it again with fewer
tasks at once: `BM_PARALLEL=2 bash burst/submit.sh ...`.

If `sbatch` does not work inside the session, start a session on the GPU partition and run the
tasks in its terminal instead (same arguments):

```bash
nohup bash burst/run_here.sh --run-id burst_a100 --select ... > run_here.log 2>&1 &
```

## 3. Where the results are

| What | Where |
|---|---|
| Is it queued or running | `squeue -u $USER` |
| State of every task, rewritten whenever one starts or ends | `runs/burst_a100/pack_status.txt` |
| Job log | `/scratch/$USER/cv_project_runs/logs/bm_burst_<job id>.log` |
| Log of one task | `/scratch/$USER/cv_project_runs/reproduction/burst_a100/logs/<task>_<index>.log` |
| Results per seed and pooled, compared with the frozen values | `runs/burst_a100/aggregate/tables.md` (written when the job ends) |
| Result files | `runs/burst_a100/gate/retrain_gpu/`, `runs/burst_a100/synergy/retrain_gpu/`, `runs/burst_a100/synergy/retrain_readers_gpu/` |
| Rebuilt Gate tree against the team's | `runs/burst_a100/gate/retrain_gpu/gate_assets_diff.json` |

To bring the results to your laptop:

```bash
bash burst/pack_results.sh burst_a100      # writes ~/burst_a100_results.tar.gz; download it in the Open OnDemand file browser
# on the laptop, inside the checkout:
tar xzf burst_a100_results.tar.gz && python scripts/aggregate_results/aggregate.py --run runs/burst_a100
```

## Settings

`burst/cluster.sh` lists every setting with its default (account, partition, job size, tasks at
once, folders). To change one, export it before the command or put `KEY=value` lines in
`burst/.env`, which git ignores.

| File | Purpose |
|---|---|
| `cluster.sh` | settings |
| `setup.sh`, `hf_login.sh` | one-time setup |
| `submit.sh`, `job.sbatch` | one GPU job that runs the selected tasks |
| `run_here.sh`, `gpu_driver.sh` | the same inside a GPU session, without `sbatch` |
| `pack_results.sh` | archive of a run's results |
