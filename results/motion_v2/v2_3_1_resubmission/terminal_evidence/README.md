# Job 16614 terminal evidence

The frozen, separately registered CPU exporter ran once after native terminal
state and settlement. Its 136 copied source files are preserved byte for byte,
including native stdout/stderr whitespace. `export_manifest.json` and
`scientific_file_inventory.json` are the exporter's original generated inventories.

`export_sidecars/` preserves all five downloaded files in their original names:
the ZIP, manifest, scientific inventory, exporter status, and `SHA256SUMS`.
The archive is 1,131,125 bytes; its SHA256 is
`0fcac0a6751afae34c25dd8ddf1902a7458375f00db8dbc30fdeca2313beeb76`.
The four sidecar checksum entries and all 136 inner source entries were verified
before extraction. `download_verification.json` records that local verification;
this README is also a local delivery addition, not a remote scientific output.

Primary records:

- Science: `repo/results/motion_v2/v2_3_1/step2_gpu_attempt1/metrics/metrics.json`
  and the adjacent `status.json`, input audit, decoder, selections, and logs.
- Final accounting: `runtime_retries_v2_3_1/attempt2/settlement/accounting.json`
  plus unmodified `final_scontrol.stdout.txt` and `final_sacct_duplicates.stdout.txt`.
- Original infrastructure failure: `runtime_retries_v2_3_1/diagnosis_16550/`.
- Submission, data/environment revalidation, and timestamped native monitoring:
  `runtime_retries_v2_3_1/attempt2/`.

The three recorded missing expected paths are not missing scientific results.
16550's stdout/stderr were never created, as documented by the pre-entry failure
diagnosis. The legacy `reused_archive_sha256.txt` does not exist; the attempt-2
`data_reverification.json` instead records the archive, manifest, and all 2432
train/validation data-file checks, with its hash bound to the submission receipt.

The inventory hashes 1293 regular scientific files and records three symlinks.
Large checkpoints, encoded-feature archives, and downloaded model weights remain
at the original HPC deployment; the manifest lists exclusions explicitly. Their
remote hashing is not a claim that these large files were downloaded or locally
rehash-verified. No scientific arrays were decoded, models run, or bootstrap
intervals recomputed during export and delivery.
