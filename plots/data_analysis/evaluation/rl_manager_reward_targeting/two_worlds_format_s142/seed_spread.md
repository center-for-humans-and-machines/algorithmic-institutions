# Seed spread

The spread is the measurement, not a nuisance. Each row gives the three seeds, their range, and the gap the experiment is trying to resolve -- the baseline clone against the best rule. When the range exceeds that gap, a mean over three seeds is not a result.

| statistic | rl_s42 | rl_s43 | rl_s44 | range | clone | best rule | gap |
|---|---|---|---|---|---|---|---|
| punish_rate | 0.4233 | 0.1020 | 0.5626 | **0.4606** | 0.2902 | 0.6931 | 0.4030 |
| mean_punishment | 1.9199 | 1.2144 | 1.1543 | **0.7656** | 1.7126 | 1.4123 | 0.3003 |
| common_good | 11.7393 | 11.9493 | 13.0223 | **1.2830** | 13.1569 | 13.3607 | 0.2038 |

Best rule by common good: `rl_pc_s44`.
