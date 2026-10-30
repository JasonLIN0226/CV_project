# Post-hoc train-support numerical ablation

Preserves the preceding unmasked run. No refitting: each arm's 903-coefficient
vector has coefficients zeroed where its old-TRAIN parent-centered feature RMS
is at the fixed 1e-6 numerical floor (710 columns). The mask, architecture,
causal observations, normalization and byte allocation match across all arms.
This is one exploratory numerical diagnosis after seeing unstable input norms,
not a selected method or an untouched confirmation experiment.

The protocol amendment was saved before masked choices and losses were read.
`audit.json` independently recomputes all common-parent and online deletion
choices, cached losses, irreversible memory invariants and pair confidence
intervals. The unmasked coefficients remain in the parent directory; this mask
and those coefficients exactly reproduce the masked parameters. No extra
training, checkpoint changes or GPU execution were involved.

Cost vs attention changes sign after masking, but cost does not reliably beat
the shuffled-label null or strong existing controls. Neither variant establishes
oracle learnability or unlearnability. See the joint diagnosis document.
