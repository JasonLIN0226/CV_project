# Execution-only amendment

Native job8530 failed before the Python entrypoint because Slurm executes a copy of stage.sh from its spool directory. It consumed4CPU seconds and0GPU seconds; no new data was generated.

A separate immutable execution namespace fixes only the explicit shell launch root and output path. The original package, failed receipt and stderr remain preserved. All model, data, optimization, inference, evaluation and statistical Python files are byte-identical to the12-fixture checked package. Fresh sample counts, seeds,120epochs and original acceptance rules are unchanged.

New root: /scratch/xl4358/phase1_independent_confirmation_execution_v1

Source lock: 3c735c89a31219ad17be3a3c9d30d208e4a85a5a07e41f69c3c17c0054e7ed7e

New native CPU preflight receipt:8532, submitted once. No duplicate submission or automatic failure retry is permitted.
