# Final Phase 1 result

Fresh seed 9137 independent confirmation is complete. Gate passes 155/155; Motion passes 147/155; the independent audit passes. Phase1_GO is false under the unchanged registered rule.

The original prediction checks (51 per task) include three evidence probes. These probes must not be added again to the 310 total. Both tasks also pass 32 fixed-reference checks. Gate passes all 72 deletion checks; Motion passes 64/72. The eight failures concern long-gap cross-horizon deletion versus reject-newest, across three seeds and their mean at both horizons.

The deletion choice improves feature MSE by 9.4–11.1%, but does not establish positive decoded-position gain: all eight paired 95% intervals contain zero. This is a limitation of the present deletion signal for the task, not proof that historical memory is useless. These checks are correlated; 302/310 is a descriptive count rather than an accuracy estimate.

The report explains objectives, data, preprocessing, each test, numerical results, figures, limitations and next steps. See [Word report](../reports/phase1_independent_confirmation_20261020/Phase1_Independent_Confirmation_Report.docx) and [verified data](../results/phase1_independent_confirmation/local_verified_results_20261020.json).

This controlled I-JEPA study uses a supervised train-only position readout and an observed-RGB guard. It does not establish V-JEPA transfer, real-video performance, online utility estimation or learned-policy advantage. Retention-policy training remains blocked by the registered Motion criterion.
