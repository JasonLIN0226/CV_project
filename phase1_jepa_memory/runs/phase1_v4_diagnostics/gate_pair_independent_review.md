# Gate pair independent review

**PASS — diagnostic audit only; not Phase 1 GO.**

Reviewed: 2026-10-12T15:55:05.027873+00:00

The original ranked reference meets all six fixed distractor diagnostic rules. No-rank remains NOT_SUPPORTED. This supports testing the already fixed, single-factor common-anchor candidate; it does not establish candidate success.

The separate audit used NumPy directly and did not import the original statistics or diagnostic functions. It reproduced all original calibration means and CI endpoints exactly and all 384 hybrid slice results within 1.11e-15. Each arm has 17,280 finite pair rows, 144 complete per-seed slices, 34,560 original episode rows, and 2,880 direct recent prediction equality proofs.

Bootstrap: 2,000 complete-pair resamples with default_rng(0); numerator and denominator remain paired. Seeds are averaged within each pair before pooled resampling. Original 10% point / positive CI-lower rule and low-baseline absolute fallback are unchanged.

## reference: SUPPORTED

| Seed | Horizon | Relative gain | 95% paired CI | Rule |
|---|---:|---:|---|---|
| 17 | 3 | 20.8764% | [16.1128%, 24.9928%] | PASS |
| 29 | 3 | 19.5232% | [12.9469%, 25.5129%] | PASS |
| 43 | 3 | 15.9128% | [8.5355%, 22.3818%] | PASS |
| all_seeds_mean | 3 | 18.8325% | [12.8903%, 24.1015%] | PASS |
| 17 | 6 | 16.6329% | [12.4683%, 20.5039%] | PASS |
| 29 | 6 | 19.4283% | [12.4481%, 25.4751%] | PASS |
| 43 | 6 | 14.5425% | [8.4407%, 20.1401%] | PASS |
| all_seeds_mean | 6 | 16.7516% | [11.4393%, 21.6070%] | PASS |

ID: no privileged slice shows increased MSE; all six per-seed absolute-improvement CI lower bounds are positive.

## gate_no_rank: NOT_SUPPORTED

| Seed | Horizon | Relative gain | 95% paired CI | Rule |
|---|---:|---:|---|---|
| 17 | 3 | 15.2983% | [10.0159%, 19.9864%] | PASS |
| 29 | 3 | 22.8800% | [17.0833%, 27.8424%] | PASS |
| 43 | 3 | 9.8814% | [3.1875%, 15.9166%] | FAIL |
| all_seeds_mean | 3 | 15.7085% | [10.0544%, 20.7259%] | PASS |
| 17 | 6 | 14.8976% | [9.7746%, 19.5312%] | PASS |
| 29 | 6 | 18.2360% | [12.5207%, 23.1305%] | PASS |
| 43 | 6 | 12.5525% | [5.3278%, 19.2204%] | PASS |
| all_seeds_mean | 6 | 15.2015% | [9.3816%, 20.2925%] | PASS |

ID: no privileged slice shows increased MSE; all six per-seed absolute-improvement CI lower bounds are positive.

## Interpretation limits

- The hybrid is a sibling-pair oracle diagnostic, not a legal single-trajectory model.
- Common-component error is not evidence that background caused the failure.
- A training anchor may change the recent teacher and contrast predictions; the diagnostic cannot guarantee its result.
- This uses revealed development trajectories. Full original task rules, the fixed reference baseline guard, and fresh confirmation remain required.

## Source integrity

Native-export SHA256: `65d9098be7143663381a232239dd67d98750550818d6bb9930a17d53ac42c2c6`.

The companion JSON records actual pair/report/raw/config/calibration/verification/log hashes and numerical discrepancy bounds. No old run/source file, checkpoint or feature array was changed.
