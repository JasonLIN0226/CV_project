# Frozen gate common replacement diagnostic

**SUPPORTED — diagnostic only; not Phase 1 GO.**

For paired targets m±d and frozen history predictions c±v, replace only c by the same checkpoint's shared recent prediction r. The hypothetical pair-average normalized MSE is recent common MSE + unchanged history contrast MSE. No new predictions are created.

This is an oracle pair diagnostic: evaluating the construction requires both sibling predictions. It is not a deployable single-trajectory model, does not establish that common error is background error, and does not predict that an anchor-training loss will work.

Core criterion: privileged B2 distractor shift must meet the original 10% point improvement (or original low-baseline absolute rule) and a strictly positive 95% paired CI lower bound in all three seeds × both horizons. Bootstrap: 2,000 resamples, seed 0, complete base pairs; pooled seeds are averaged within each pair, and pooled CIs are conditional on those seeds.

## reference: SUPPORTED


| Split | Seed | h | Recent MSE | History MSE | Hybrid MSE | Hybrid gain | 95% paired CI | History→hybrid MSE reduction | Status |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| validation | 17 | 3 | 0.09900414 | 0.04415032 | 0.03943900 | 60.164% | [54.723%, 64.764%] | 0.00471132 | SUPPORTED |
| validation | 29 | 3 | 0.10094731 | 0.04678030 | 0.04278937 | 57.612% | [51.073%, 62.966%] | 0.00399094 | SUPPORTED |
| validation | 43 | 3 | 0.10109633 | 0.04682171 | 0.04227318 | 58.185% | [52.301%, 62.947%] | 0.00454853 | SUPPORTED |
| validation | all_seeds_mean | 3 | 0.10034926 | 0.04591744 | 0.04150052 | 58.644% | [52.858%, 63.379%] | 0.00441693 | SUPPORTED |
| validation | 17 | 6 | 0.09852073 | 0.05101482 | 0.04143917 | 57.939% | [53.376%, 61.997%] | 0.00957564 | SUPPORTED |
| validation | 29 | 6 | 0.10036838 | 0.04861798 | 0.04436321 | 55.800% | [50.248%, 60.696%] | 0.00425477 | SUPPORTED |
| validation | 43 | 6 | 0.09839832 | 0.04814147 | 0.04136989 | 57.957% | [53.836%, 61.690%] | 0.00677158 | SUPPORTED |
| validation | all_seeds_mean | 6 | 0.09909581 | 0.04925809 | 0.04239076 | 57.222% | [52.572%, 61.193%] | 0.00686733 | SUPPORTED |
| test_id | 17 | 3 | 0.10018987 | 0.05042236 | 0.04624602 | 53.842% | [47.177%, 60.065%] | 0.00417634 | SUPPORTED |
| test_id | 29 | 3 | 0.10120207 | 0.05131448 | 0.04662716 | 53.927% | [47.504%, 59.849%] | 0.00468732 | SUPPORTED |
| test_id | 43 | 3 | 0.10085634 | 0.05230028 | 0.04740756 | 52.995% | [45.993%, 59.355%] | 0.00489272 | SUPPORTED |
| test_id | all_seeds_mean | 3 | 0.10074943 | 0.05134571 | 0.04676025 | 53.588% | [46.972%, 59.668%] | 0.00458546 | SUPPORTED |
| test_id | 17 | 6 | 0.09900688 | 0.05669580 | 0.05045671 | 49.037% | [41.557%, 56.112%] | 0.00623909 | SUPPORTED |
| test_id | 29 | 6 | 0.10134454 | 0.05804277 | 0.05290542 | 47.796% | [40.237%, 54.567%] | 0.00513736 | SUPPORTED |
| test_id | 43 | 6 | 0.09874649 | 0.05733694 | 0.05095048 | 48.403% | [40.774%, 55.453%] | 0.00638645 | SUPPORTED |
| test_id | all_seeds_mean | 6 | 0.09969930 | 0.05735850 | 0.05143754 | 48.407% | [41.004%, 54.968%] | 0.00592097 | SUPPORTED |
| test_long_gap | 17 | 3 | 0.10122238 | 0.04710645 | 0.04144295 | 59.058% | [53.345%, 63.205%] | 0.00566349 | SUPPORTED |
| test_long_gap | 29 | 3 | 0.10217868 | 0.04518817 | 0.04077006 | 60.099% | [54.281%, 64.424%] | 0.00441812 | SUPPORTED |
| test_long_gap | 43 | 3 | 0.10164127 | 0.04964658 | 0.04228125 | 58.401% | [52.459%, 62.915%] | 0.00736533 | SUPPORTED |
| test_long_gap | all_seeds_mean | 3 | 0.10168078 | 0.04731373 | 0.04149809 | 59.188% | [53.753%, 63.268%] | 0.00581565 | SUPPORTED |
| test_long_gap | 17 | 6 | 0.09883449 | 0.04940404 | 0.04269629 | 56.800% | [52.035%, 60.952%] | 0.00670775 | SUPPORTED |
| test_long_gap | 29 | 6 | 0.10127656 | 0.04876284 | 0.04527366 | 55.297% | [49.951%, 59.807%] | 0.00348918 | SUPPORTED |
| test_long_gap | 43 | 6 | 0.09840298 | 0.04889582 | 0.04337104 | 55.925% | [50.507%, 60.638%] | 0.00552477 | SUPPORTED |
| test_long_gap | all_seeds_mean | 6 | 0.09950468 | 0.04902090 | 0.04378033 | 56.002% | [50.855%, 60.404%] | 0.00524056 | SUPPORTED |
| test_distractor_shift | 17 | 3 | 0.14743660 | 0.17214354 | 0.11665717 | 20.876% | [16.113%, 24.993%] | 0.05548637 | SUPPORTED |
| test_distractor_shift | 29 | 3 | 0.12867823 | 0.13698123 | 0.10355616 | 19.523% | [12.947%, 25.513%] | 0.03342507 | SUPPORTED |
| test_distractor_shift | 43 | 3 | 0.13364389 | 0.13520958 | 0.11237740 | 15.913% | [8.535%, 22.382%] | 0.02283218 | SUPPORTED |
| test_distractor_shift | all_seeds_mean | 3 | 0.13658624 | 0.14811145 | 0.11086358 | 18.833% | [12.890%, 24.102%] | 0.03724787 | SUPPORTED |
| test_distractor_shift | 17 | 6 | 0.14864942 | 0.18682692 | 0.12392476 | 16.633% | [12.468%, 20.504%] | 0.06290216 | SUPPORTED |
| test_distractor_shift | 29 | 6 | 0.12432654 | 0.13535028 | 0.10017207 | 19.428% | [12.448%, 25.475%] | 0.03517821 | SUPPORTED |
| test_distractor_shift | 43 | 6 | 0.14264120 | 0.15064945 | 0.12189761 | 14.542% | [8.441%, 20.140%] | 0.02875184 | SUPPORTED |
| test_distractor_shift | all_seeds_mean | 6 | 0.13853905 | 0.15760888 | 0.11533148 | 16.752% | [11.439%, 21.607%] | 0.04227740 | SUPPORTED |

ID slices where replacement increases privileged error: 0 (including pooled slices).
## gate_no_rank: NOT_SUPPORTED


| Split | Seed | h | Recent MSE | History MSE | Hybrid MSE | Hybrid gain | 95% paired CI | History→hybrid MSE reduction | Status |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| validation | 17 | 3 | 0.09912001 | 0.04128619 | 0.03836329 | 61.296% | [56.399%, 65.533%] | 0.00292290 | SUPPORTED |
| validation | 29 | 3 | 0.10163300 | 0.04744004 | 0.04322118 | 57.473% | [51.495%, 62.498%] | 0.00421886 | SUPPORTED |
| validation | 43 | 3 | 0.10131275 | 0.04628493 | 0.04172666 | 58.814% | [52.826%, 63.612%] | 0.00455827 | SUPPORTED |
| validation | all_seeds_mean | 3 | 0.10068858 | 0.04500372 | 0.04110371 | 59.177% | [53.747%, 63.625%] | 0.00390001 | SUPPORTED |
| validation | 17 | 6 | 0.09756956 | 0.04703254 | 0.04001201 | 58.991% | [54.220%, 63.075%] | 0.00702052 | SUPPORTED |
| validation | 29 | 6 | 0.09832099 | 0.04823360 | 0.04297009 | 56.296% | [49.778%, 61.601%] | 0.00526351 | SUPPORTED |
| validation | 43 | 6 | 0.09867421 | 0.04895977 | 0.04192646 | 57.510% | [52.772%, 61.659%] | 0.00703330 | SUPPORTED |
| validation | all_seeds_mean | 6 | 0.09818825 | 0.04807530 | 0.04163619 | 57.596% | [52.718%, 61.798%] | 0.00643911 | SUPPORTED |
| test_id | 17 | 3 | 0.10070103 | 0.05179735 | 0.04818431 | 52.151% | [44.559%, 59.245%] | 0.00361304 | SUPPORTED |
| test_id | 29 | 3 | 0.10171835 | 0.05758039 | 0.05100538 | 49.856% | [42.270%, 57.356%] | 0.00657502 | SUPPORTED |
| test_id | 43 | 3 | 0.10110754 | 0.05461292 | 0.04933270 | 51.208% | [43.979%, 58.185%] | 0.00528022 | SUPPORTED |
| test_id | all_seeds_mean | 3 | 0.10117564 | 0.05466356 | 0.04950746 | 51.068% | [43.760%, 58.019%] | 0.00515609 | SUPPORTED |
| test_id | 17 | 6 | 0.09888989 | 0.05791253 | 0.05164147 | 47.779% | [39.723%, 55.744%] | 0.00627106 | SUPPORTED |
| test_id | 29 | 6 | 0.09907760 | 0.05825270 | 0.05286900 | 46.639% | [38.269%, 54.576%] | 0.00538370 | SUPPORTED |
| test_id | 43 | 6 | 0.09931838 | 0.05915119 | 0.05214682 | 47.495% | [39.878%, 54.989%] | 0.00700437 | SUPPORTED |
| test_id | all_seeds_mean | 6 | 0.09909529 | 0.05843881 | 0.05221910 | 47.304% | [39.424%, 54.897%] | 0.00621971 | SUPPORTED |
| test_long_gap | 17 | 3 | 0.10067559 | 0.04498858 | 0.04001993 | 60.249% | [54.226%, 64.781%] | 0.00496865 | SUPPORTED |
| test_long_gap | 29 | 3 | 0.10287218 | 0.04892618 | 0.04315302 | 58.052% | [52.586%, 62.642%] | 0.00577317 | SUPPORTED |
| test_long_gap | 43 | 3 | 0.10071263 | 0.04680989 | 0.04030563 | 59.980% | [54.582%, 64.287%] | 0.00650426 | SUPPORTED |
| test_long_gap | all_seeds_mean | 3 | 0.10142014 | 0.04690822 | 0.04115952 | 59.417% | [53.939%, 63.732%] | 0.00574869 | SUPPORTED |
| test_long_gap | 17 | 6 | 0.09769292 | 0.04800546 | 0.04137175 | 57.651% | [52.832%, 61.957%] | 0.00663372 | SUPPORTED |
| test_long_gap | 29 | 6 | 0.09870337 | 0.05072153 | 0.04464502 | 54.768% | [48.725%, 59.783%] | 0.00607650 | SUPPORTED |
| test_long_gap | 43 | 6 | 0.09800508 | 0.04939715 | 0.04331339 | 55.805% | [49.476%, 61.290%] | 0.00608377 | SUPPORTED |
| test_long_gap | all_seeds_mean | 6 | 0.09813379 | 0.04937471 | 0.04311005 | 56.070% | [50.401%, 60.877%] | 0.00626466 | SUPPORTED |
| test_distractor_shift | 17 | 3 | 0.15737879 | 0.15687105 | 0.13330251 | 15.298% | [10.016%, 19.986%] | 0.02356854 | SUPPORTED |
| test_distractor_shift | 29 | 3 | 0.12928997 | 0.13926025 | 0.09970837 | 22.880% | [17.083%, 27.842%] | 0.03955188 | SUPPORTED |
| test_distractor_shift | 43 | 3 | 0.14804162 | 0.16132253 | 0.13341297 | 9.881% | [3.187%, 15.917%] | 0.02790956 | NOT_SUPPORTED |
| test_distractor_shift | all_seeds_mean | 3 | 0.14490346 | 0.15248461 | 0.12214128 | 15.709% | [10.054%, 20.726%] | 0.03034333 | SUPPORTED |
| test_distractor_shift | 17 | 6 | 0.15589821 | 0.17214754 | 0.13267305 | 14.898% | [9.775%, 19.531%] | 0.03947450 | SUPPORTED |
| test_distractor_shift | 29 | 6 | 0.12996610 | 0.14293483 | 0.10626548 | 18.236% | [12.521%, 23.131%] | 0.03666935 | SUPPORTED |
| test_distractor_shift | 43 | 6 | 0.13099460 | 0.14668554 | 0.11455151 | 12.552% | [5.328%, 19.220%] | 0.03213403 | SUPPORTED |
| test_distractor_shift | all_seeds_mean | 6 | 0.13895297 | 0.15392264 | 0.11783001 | 15.202% | [9.382%, 20.292%] | 0.03609263 | SUPPORTED |

ID slices where replacement increases privileged error: 0 (including pooled slices).

All six conditions, validation/ID/long-gap/distractor splits, per-seed and pooled values are retained in gate_common.json. Missing pair calibration is INCOMPLETE: adding/subtracting marginal confidence limits would not recover the required paired interval.
