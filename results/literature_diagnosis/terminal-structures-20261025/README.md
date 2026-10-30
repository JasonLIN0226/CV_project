# Terminal three-structure DEV run — 2026-10-25

**Completed, reader gate failed for the combined study. No students trained. CONF was not generated or opened.** R/D have failing appearance conditions; S passes K2 in all conditions but fails K3. S baseline diagnostics remain interpretable as development evidence. See [Chinese report](../../../docs/TERMINAL_THREE_STRUCTURES_REPORT_ZH.md), [frozen preregistration](../../../docs/TERMINAL_STRUCTURES_PREREGISTRATION_20261025.md), and [six failure/control cards](casebook/index.html).

## Read first

- `execution.json`, `sacct-receipt.txt`, `slurm-11441.log`: completed A100 job11441, 22m24s, exact stop status.
- `reader_qualification.json`: all condition gates; `summary.json`: baseline metrics and cost.
- `decisions.csv`: all 1728 parent × six method decisions; `decision_summary.json`: exact choices, expected deletion losses, and prediction-distance scores.
- `student_comparison_summary.json` / `student_comparison_family_metrics.json`: despite the pipeline filename, this run is **baseline_only=true**; no student result is present. Intervals use 24 base families, not 1728 independent samples.
- `student_coefficient_audit.json`: independent arithmetic recomputation from saved prediction coefficients; no new reader inference.
- `protocol.json`, `reader_locked.json`, `learning_curve.json`: frozen source hashes and VAL checkpoint selection.
- `package_manifest.json`: full archive member checksums and the three derived arrays omitted from transport.

## Full raw data and checkpoint

[Release](https://github.com/JasonLIN0226/CV_project/releases/tag/terminal-structures-20261025)

| Asset | Bytes | SHA256 |
|---|---:|---|
| terminal-structures-results.zip | 345875527 | d6b700cd17fedf0423be026654ac84bf48ac7953bfd31f973230ef958a05217c |
| terminal-execution-receipts.zip | 9450 | 9a4ac3239ca8a4fb018a173e30f386e781d2f53a3b9898cb86ab748c77fe0cc9 |
| terminal-structures-source-20261025.zip | 654909 | f41497a439f39895055e396beb0bc458e20cc746d5c71391eb986fde945fe806 |

GitHub server digests match local and HPC receipts. Main result archive contains all 163 manifest members: frozen source, model checkpoint, basis, exported inputs and prediction coefficients, action banks, all raw baseline decisions and 21312 reader intervention rows. It omits only `unique_rgb.bin`, `features.npy`, `features_raw.npy`, which remain on HPC and have recorded hashes. Reproduce RGB with the frozen generator; local reconstruction matches the entire 529376256-byte archived RGB stream SHA256 (`fab455aef9d42d9671239f43d748f938420a12edc5fd0ab3730b48d3fdb3e8a5`). Re-encoding requires the stated I-JEPA model and dependencies; we do not claim independent re-encoding or retraining.

## Recompute the post-run audit

After extracting the result ZIP to a new folder, run from the CV checkout with NumPy installed:

```bash
python experiments/phase2/audit_terminal_archive.py --extracted /absolute/path/to/extracted --output /absolute/path/to/separate-audit-output --repository .
```

The output directory must be outside the archive. This verifies manifest/source/Git binding and qualification arithmetic, recomputes baseline aggregates and forecast distances from saved coefficients, and writes the explicitly posthoc geometry/native-correct subset diagnosis. It does not run a reader, train, re-encode, change choices or open confirmation. Outputs are published under `independent_audit/`. The visual renderer separately checks each regenerated displayed frame against the archived RGB index.

## Scope

R/D/S are synthetic terminal decisions, K=2 from three observed candidates. Semantic loss is distinct from reader feature-MSE regret. In S the expected-loss oracle deletes necessary evidence in 159/1152 cases despite passing aggregate K2 gates. Full sensitivity lacks a qualified K3 reference. These are development findings, not independent confirmation, learned-scorer superiority, online delayed credit, PushT success, or a new PM experiment. The new relational structure's scalar-score upper bound is a design property, not a new algorithm contribution.
