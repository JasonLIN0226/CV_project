# Publication scope

This is a source and results publication, not a new training run or web deployment.

Included:

- The local shared experimental package, configurations, tests and original scripts.
- Frozen scientific snapshots, source locks, protocols and available native job/audit records.
- Locally available CSV/JSON scientific results, including earlier negative and incomplete experiments.
- Scientific figures and existing Word/PDF reports, with the latest report highlighted at repository root.
- Historical FAR and paper-reproduction adapters/results, clearly separate from proposal Phase One claims.
- An explicit copied-file SHA-256 manifest and exclusion inventory.

Excluded:

- Model weights and generated array/cache files; synthetic dataset manifests are retained.
- HPC-only raw arrays/traces that were never transferred to this workstation.
- Python bytecode, runtime binaries, wheels, font caches, upload ZIP bundles and per-page QA screenshots.
- Git internals, login secrets, private keys and private environment files.
- Pristine upstream FAR/SPACE checkouts; pinned identities and origin URLs are in `THIRD_PARTY.md`.

`publication/source_and_results_manifest.json` lists copied file sizes and hashes. `publication/excluded_artifacts.json` lists excluded local paths and reasons. A clean clone can inspect the results and independently recompute the transferred pair-level statistics. Full model replay needs the external assets described in `REPRODUCIBILITY.md`.

Machine-specific paths in immutable historical records identify where evidence was originally produced. They are not current connection details or credentials. No old allocation or SSH endpoint should be reused.
