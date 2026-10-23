# Historical source snapshots

These byte-identical files preserve the source hashes recorded by the 20 October
receipts. They were copied from commit `951a6f9` before the 22 October CI test-isolation
fix. Their filenames are their SHA-256 digests; no historical receipt is rewritten.

| Original path | Snapshot SHA-256 | Why the working file changed |
|---|---|---|
| `tests/test_phase2_prepare_pusht_planning_data.py` | `fd06b36e1e29f6857954004a14324043f2f0d58b3c4ee1c84e89e2e472fb9bba` | The fake-environment unit test now injects fake package provenance, so it does not require the optional real simulator. |
| `tools/verify_phase2_history.py` | `39ff9f82dbffcc75f41e7c212759e80cde569965ee40bf396588e8eb344b452e` | The verifier can verify these exact historical source snapshots when the current working revision differs. |

The current verifier reports every snapshot fallback explicitly. It checks the
original receipt digest against these bytes; it does not interpret historical
test counts as validation of later code. Current tests and CI validate current
code separately. Scientific runtime source and saved experiment results did not
change in this fix.
