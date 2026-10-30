# Exact symbolic preflight, not visual results

This directory contains an exhaustive finite toy check. No learned reader,
encoder, scorer, new visual dataset, GPU job, or training was used. The reader
is an analytically specified conditional mean for independent fair bits.

With a qualified complete-set reference:

| Structure | Medoid preserves necessary evidence | Full-set sensitivity |
|---|---:|---:|
| Redundancy | 24/24 | 24/24 |
| Orthogonal additive decoy | 48/48 | 48/48 |
| XOR, visible roles | 0/48 | 48/48 |
| Relational XOR, duplicate visible tag identifies pair | 0/96 | 96/96 |

Visible-role XOR is also solved48/48 by a fixed independent scorer that simply
gives the visible decoy the lowest score. XOR alone does not establish the need
for set-conditioned eviction.

In relational XOR, the three visible tags are `(u,u,v)` and the XOR target uses
the two matching-tag payloads. All tag assignments, payloads and orders are
balanced. A single local type is a singleton with probability1/3. Enumerating
all75 weak orders on the four `(tag,payload)` types gives best independent
retention48/96, while an all-tied ranking gives32/96. Independent does not mean
random.

The stronger conditional check allows arbitrary slot/age-dependent scores.
Fix chronological payloads, ages, nuisance and a nonleaking common context.
The six nonconstant tag patterns remain equally likely, and six `(slot,tag)`
scores can be ordered in720 ways. None correctly deletes the singleton on more
than3/6 patterns. Earliest-slot tie breaking induces one of these total orders.
All8 payload conditions have equal pattern multiplicity in the enumerated toy.
Thus the50% bound also covers independent position-aware scores **under these
conditional-balance assumptions**. A visual generator must verify them; the
bound does not cover pooled candidate features, recurrent state revealing tags,
or styling/context that leaks which tag is duplicated.

`summary.json` also records the general squared-distance condition for additive
decoys, an aligned-error counterexample, and a wrong-full-reference counterexample
for sensitivity. Full-set prediction quality must be qualified, not assumed.
XOR produces two target classes from four input-bit pairs; four-way outcome
classification is invalid.

From the repository root:

```bash
python experiments/phase2/audit_terminal_synergy_preflight.py --output results/literature_diagnosis/terminal-synergy-preflight-20261025
```

`ideal_cases.json`, `relational_cases.json` and `independent_weak_orders.json`
contain every enumerated row. Selection uses exact rational arithmetic and an
earliest-slot exact-tie rule. No random seed or parameter search is used.
