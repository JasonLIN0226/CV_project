# Literature memory baselines

Three published memory mechanisms, run against the repo's own rungs (recent-only, oracle, heuristics,
deletion-cost scorer) with the same frozen predictor, R = 4 recent frames and B = 2 memory items.

| Method | Paper | Official code | File |
|---|---|---|---|
| PaFu-KV | [arXiv 2601.21896](https://arxiv.org/abs/2601.21896) | none (the authors' repo holds only a README) | `pafu.py` |
| Compress to Remember (PACC) | [arXiv 2609.36364](https://arxiv.org/abs/2609.36364) | none | `pacc.py` |
| FrameMorrow | [arXiv 2609.38839](https://arxiv.org/abs/2609.38839) | [YinBo0927/FrameMorrow](https://github.com/YinBo0927/FrameMorrow), MIT | `framemorrow_ref/` (unchanged copy), `fm.py` (adapter) |

Each file's docstring lists what is kept from the paper and what had to change. Two points decide how a
number should be read:

- **Bounded store or not.** PaFu-KV is an online bounded policy as published. Compress to Remember keeps a
  growing archive and FrameMorrow ranks the whole past, so each is run twice: as published (reads the
  whole eligible history at the query) and as an online 2-slot variant that is not in the paper. Rows are
  labelled accordingly.
- **Two PaFu-KV rows.** `pafu.py` follows the paper: one score per frame, written once and never updated.
  `pafu_repo.py` re-runs the repo's earlier adaptation (`experiments/phase2/literature_baselines.py`), which
  re-scores the three resident candidates together at every eviction.
- **Teacher signal.** PaFu-KV distils attention, FrameMorrow distils visual similarity between a past frame
  and the realised future, Compress to Remember matches the frozen predictor's full-history output. None
  uses the predictor's future error per deleted frame, which is what the deletion-cost scorer uses.

## Environments

| Environment | Entry point | Assets |
|---|---|---|
| Gate (192 validation episodes) | `gate_run.py` | Default: the team's frozen bundle from the private HF org (`torch/gate_team_assets.sbatch`): original predictors, feature cache and scorers. "Ours" is the published checkpoint, and every shared policy is compared row by row with the team's original validation output. Fallback without HF access: `torch/gate_assets.sbatch` rebuilds everything from the repo generator (the rebuilt dataset is byte-identical to the team's) and retrains ours. |
| Synergy / terminal R, D, S (3,456 confirmation records) | `synergy_run.py` | Frozen readers, confirmation features and the team's frozen student outputs come from the public GitHub releases. TRAIN features are re-encoded by `torch/synergy_assets.sbatch`; the regenerated records are byte-identical to the released ones. |
| Motion v2 | not run | The frozen Motion v2 reader and features exist only on a teammate's scratch space. |
| PushT | not run | No PushT predictor exists yet. |

## Running on torch

```bash
ssh torch 'cd /scratch/ys4907/CV_project && sbatch experiments/memory_baselines/torch/setup_env.sbatch'   # once
ssh torch 'cd /scratch/ys4907/CV_project && A=$(sbatch --parsable experiments/memory_baselines/torch/gate_team_assets.sbatch) \
  && sbatch --array=0-8,15-17 --dependency=afterok:$A experiments/memory_baselines/torch/gate_baselines.sbatch'
ssh torch 'cd /scratch/ys4907/CV_project && S=$(sbatch --parsable experiments/memory_baselines/torch/synergy_assets.sbatch) \
  && sbatch --dependency=afterok:$S experiments/memory_baselines/torch/synergy_baselines.sbatch'
python -m experiments.memory_baselines.gate_run --out results/memory_baselines/gate_team --aggregate       # pools seeds and parts
```

Gate array tasks, one per predictor seed (17, 29, 43): 0-2 core (reference rungs, PaFu-KV, Compress to
Remember), 3-5 FrameMorrow as published, 6-8 FrameMorrow online, 9-14 FrameMorrow on the project's I-JEPA
features (optional), 15-17 the earlier-repo PaFu-KV row.

A Gate task stopped by its time limit continues when the same array index is resubmitted: the compressor
training saves a checkpoint every 100 steps under `/scratch/ys4907/cv_project_runs/gate_checkpoints/`, and a
resumed run takes the same steps as an uninterrupted one. The 2-slot compressor variant needs about 4.5 hours
on CPU and about 3 minutes on a GPU.

`synergy_released.py` scores every released Synergy policy (the project's own baselines and students) from
the released arrays, with no training, and writes `results/memory_baselines/synergy/released_scores.json`.

Jobs run on CPU by default; add `--gres=gpu:1 --constraint=l40s --export=ALL,DEVICE=cuda` for a GPU.
The conda env is `/scratch/ys4907/miniconda3/envs/cvmem`. Large files (dataset, features, checkpoints) stay
under `/scratch/ys4907/cv_project_runs/`; only small JSON results are written into `results/memory_baselines/`.

Software checks: `python -m pytest -q tests/test_memory_baselines.py`.
