# Controlled Gate cue-factor diagnosis

Read the [full report](../../../docs/GATE_CUE_FACTOR_DIAGNOSIS.md). This is a fixed, exploratory diagnosis on all 72 comparable actual attention parents from the already observed fresh development set, with 8 content/age/action combinations and 4 matched sibling-wrong-cue combinations per episode.

All 144 original endpoint vectors reproduce exactly. In the original 27 failures, the six-order average endpoint allocation is content +0.045316, age +0.000615 and executed-action summary +0.007793. Correct cue is worse than matched wrong cue in 21/27 failures, while it is better in 48/72 of the full set. These are conditional model-input diagnostics, not deployable policies, causal physics or new-scene confirmation.

- `protocol.json`: frozen design, all arms, representative selection rules, runtime and input/source hashes.
- `episode_rows.json`: every prediction metric and reader path value.
- `factor_effects.json`: all main effects, interactions and exact endpoint allocations.
- `correct_minus_wrong.json`: all four content comparisons with identical age/actions.
- `cases.json`, `representatives.json`, `case_cards.json`: original parent identities, frame times, fixed representative choices and their 12-arm results.
- `reader_path_effects.json`: descriptive effects on attention/time/memory/velocity/transport paths.
- `independent_audit.json`: full metric and six-permutation allocation verification from saved outputs.
- `source/`, `figures/`, `manifest.json`: exact source snapshots, scientific plot and portable file hashes.

The large saved prediction archive stays under ignored build; its hash and size are recorded in the manifest. No prior frozen source or report was modified by this task.
