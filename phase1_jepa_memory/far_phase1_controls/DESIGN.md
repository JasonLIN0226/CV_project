# Phase1 explicit memory controls

Add48 complete records (8 existing scenes, seeds0/1/2,2 controls),3498 target steps. Reuse the immutable96-record pilot, giving144 paired records/10494 target steps. Same official FAR predictor, DDIM20, stride4, full return and metrics. No training.

Current-only reads exactly one true current latent and zero explicit past. Recent-only stores FIFO16 past but reads the most recent11 plus current directly, without learned retrieval. Old FIFO used FAR to retrieve11 from its16, so it remains a separate comparison. No valid padding or repeated frames. Native variable context compatibility must be demonstrated on GPU; otherwise stop with a truthful failure boundary.

Primary: Reservoir against both controls on final-quarter DreamSim. Average three seeds within each of8 independent scenes; paired10000 bootstrap and exact256 signflip, Holm correction for two primary comparisons. Report all six methods, all scenes, seeds and secondary metrics. Both lower95% bounds positive and Holm p<.05 are the bounded criterion. Existing original Phase1 GO thresholds remain unchanged. This addition is prospective for new controls, but follows observed pilot values and selects Reservoir from that pilot; it is not independent confirmation.

Estimate additional50–90 minutes on one A100, excluding queue; hard cap2 hours, soft110 minutes. First full clip across both controls and all seeds must project the entire fixed run within soft budget. CPU audit/aggregation follows automatically on afterany; incomplete or failed science yields partial audit only. Preserve old96 records and all frozen sources; no adaptive reduction or blind resubmission.
