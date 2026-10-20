# Fresh CPU predictor pilot

`experiments/phase2/train_predictor_pilot.py` trains a **newly initialized** supplied `phase1.model.DynamicsPredictor` on the new development dataset after real official I-JEPA encoding and train-only normalization have completed. It does not resume or reproduce the latest repaired Phase 1 predictor, use missing external checkpoints, train a retention scorer, generate deletion labels, or run planning.

Run inside the existing CPU allocation, after the real encoded report is complete:

```bash
"$TASK_ROOT/venv/bin/python" "$TASK_ROOT/repo/experiments/phase2/train_predictor_pilot.py" \
  --data-root "$TASK_ROOT/data" --encoded-report "$TASK_ROOT/runs/9284/encoded" \
  --output "$TASK_ROOT/runs/new-cpu-predictor-pilot" \
  --epochs 4 --seed 17 --dropout 0.15 --batch-size 32 --threads 4 --timeout-seconds 1200
```

`9284` identifies the currently planned encoder allocation; use the actual completed report path when running elsewhere. Output must be new. The parent limits runtime to at most 20 minutes, hides CUDA from the worker, records commands/source hashes/settings/Git HEAD/logs/exit status, and terminates the worker process group on timeout. A partial run retains completed-epoch training snapshots and must not be reported as a completed pilot. No job is submitted.

The exact encoder runtime configuration comes from the completed encoding report, because encoder fields participate in the cache signature. The earlier raw data preparation configuration is not interchangeable. The report, cache, normalizer, observation archives, and retained-source hashes are checked before training. Only train/validation feature arrays and actions are loaded by the trainer; no test prediction metrics are computed.

The new output contains explicit symlink read views of the source `dataset` and `features`. This lets the unchanged train-only `fit_basis` write new task bases under the new output's `predictor` directory without overwriting original data/cache/configuration. Original artifact hashes are checked again at completion. The existing basis algorithm caps rank by available training samples: with 16 training pairs per task, output rank is at most 95 despite a configured maximum of 128. This is the supplied algorithm, not a copied 128-dimensional old basis.

Both Gate and Motion use one seed, unchanged predictor architecture/hyperparameters and the supplied paired regression objective (MSE plus the configured paired/ranking auxiliary terms). Batches preserve sibling adjacency. Each epoch balances the new adapter's six training modes: random fixed/variable subsets, FIFO, reservoir, temporal buckets, and visual diversity. These six modes are not the six completed retention baseline results. Whole-memory dropout applies only to training; validation uses fixed epoch/draw zero without dropout. The legacy trainer is not called because it uses different privileged/full-history conditions and no enabled memory dropout.

Checkpoints are selected by the earliest minimum validation feature MSE averaged equally across the six fixed modes. Per-task logs retain training loss/MSE, optimizer steps, validation mode scores, sample/dropout counts, and elapsed time. `latest.pt` contains optimizer/scheduler/RNG state at completed epoch boundaries; `best.pt` is the selected training checkpoint. `frozen.pt` contains the selected state/configuration and provenance for later use, after freezing and exact validation-score replay. No test scores select a checkpoint.

The pilot demonstrates that real-cache, varied-history/dropout predictor training executes. Four epochs on this small development dataset do not establish scientific superiority, generalization, full Phase 2 completion, or readiness of a PushT predictor. Absolute-coordinate PushT actions remain incompatible with this inherited additive-control bridge.

Local tests use synthetic vectors explicitly labeled `NOT_JEPA`; they check real optimizer updates, validation checkpoint selection/freeze replay, no test-feature access, train-only basis fitting, and rejection of incomplete real-encoder reports. They are software checks, not A100 or research evidence.
