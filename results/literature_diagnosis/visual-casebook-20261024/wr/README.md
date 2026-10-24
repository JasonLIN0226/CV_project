# WR startup: an existing storage-stage failure

This package exports an already-completed diagnostic. It runs no model, policy,
encoder, simulator, training, or oracle search. `case.json` contains every saved
WR step for both input versions and paths to exact cached PNG observations.

The selected fresh-development episode is `1eb3ccff25424834a0780a10`. Selection
is post hoc: among the 41 rescued episodes whose archived feature MSE improved,
sort by cue onset and episode ID, then take the first. This is an explanatory
case, not a held-out performance estimate.

Original cue copies occur at t=2,3,4. The cold-start onset has a valid surprise
score but no available MAD threshold. Original t=2 exits recent at t=6 and is
deleted from retained memory at t=8; other equivalent copies remain until t=10.
Thus **t=10 is the first complete cue loss**, not t=8. In the shared-prefix arm,
the onset event is confirmed at original t=3 / execution time 7. The final
retained original timestamps are cold [12,13], prefix [2,13]; query t=17.

Archived mean h3/h6 feature MSE: 0.08686156369655085 → 0.06107325690368278.
No alpha diagnostic is inferred, and pictures are never predicted RGB.

The full denominator is 192 fresh episodes: 62 early cues rescued (41 MSE
improvements, 21 worsened), 130 later-cue episodes with unchanged memory/loss.
Fixing storage therefore does not guarantee improved prediction.

Both arms have R=4/K=2, but the common four-frame prefix adds processing and
scalar state. It repeats the first already-observed frame with zero added
actions. Original t maps to execution t+4; negative original times denote
prefix copies, not new observations. Selected-case peak logical state rises
124,215→124,255 bytes; maxima across the complete WR diagnostic rise
124,431→124,463. This is not equal-total-cost superiority. The archived 896
shift-only frozen-predictor checks have exactly zero prediction difference.

Reproduce with `.venv/bin/python experiments/phase2/export_wr_startup_case.py`.
Input hashes, PNG pixel roundtrips, trace residency, first loss, population
counts and output hashes are verified or recorded in `provenance.json`.
