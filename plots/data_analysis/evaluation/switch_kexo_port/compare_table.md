# switch-kexo-port: k-one-hot joint-exodus switch in the frontier stack

Scores are multiples of the human noise ceiling (<= 1 at the ceiling, 1-2 minor, 2-5 clear, > 5 not reproduced); delta = after - before, negative is an improvement. `before` = the frontier baseline `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun`; `after` = the same stack with only the switch model swapped to `switch_exodus_k_onehot`, `23_2g8a_switch_kexo_port_self_gnncopar1_contr_stimulus_skip_contr_gnn_kexo_switch_curpun`. Run `lin_multinomial_copula_self`.

## Per row

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.860 | 0.841 | -0.019 | <= 1 -> <= 1 |
| CB | 0.788 | 0.804 | 0.016 | <= 1 -> <= 1 |
| CC | 0.889 | 0.829 | -0.060 | <= 1 -> <= 1 |
| CD | 0.810 | 0.807 | -0.003 | <= 1 -> <= 1 |
| CE | 1.057 | 1.031 | -0.026 | 1-2 -> 1-2 |
| CF | 0.828 | 0.840 | 0.012 | <= 1 -> <= 1 |
| CG | 1.554 | 1.101 | -0.452 | 1-2 -> 1-2 |
| SA | 0.785 | 0.789 | 0.004 | <= 1 -> <= 1 |
| SB | 1.006 | 0.898 | -0.108 | 1-2 -> <= 1 |
| SC | 1.427 | 1.329 | -0.099 | 1-2 -> 1-2 |
| PA | 0.660 | 0.617 | -0.043 | <= 1 -> <= 1 |
| PB | 0.969 | 0.948 | -0.021 | <= 1 -> <= 1 |
| PC | 0.907 | 0.898 | -0.009 | <= 1 -> <= 1 |
| PD | 0.722 | 0.736 | 0.013 | <= 1 -> <= 1 |
| RCA | 1.633 | 1.507 | -0.126 | 1-2 -> 1-2 |
| RCB | 1.545 | 1.396 | -0.149 | 1-2 -> 1-2 |
| RCC | 1.530 | 1.613 | 0.083 | 1-2 -> 1-2 |
| RCD | 1.309 | 1.689 | 0.380 | 1-2 -> 1-2 |
| RCE | 0.894 | 0.952 | 0.058 | <= 1 -> <= 1 |
| RSA | 1.070 | 1.644 | 0.574 | 1-2 -> 1-2 |
| RPA | 0.693 | 0.648 | -0.045 | <= 1 -> <= 1 |
| RPB | 0.847 | 0.773 | -0.075 | <= 1 -> <= 1 |
| mean | 1.036 | 1.031 | -0.004 |  |
| rows <= 1 | 13.000 | 14.000 | 1.000 |  |

## RCE per band: OLS slope of next-round change on punishment received

| stage | slope_0-4 | slope_5-9 | slope_10-14 | slope_15-19 | signs | signs_vs_human |
|---|---|---|---|---|---|---|
| human | 0.140 | 0.104 | -0.077 | -0.161 | ++-- |  |
| before | 0.095 | 0.020 | -0.058 | -0.160 | ++-- | ++-- (4/4) |
| after | 0.076 | 0.033 | -0.020 | -0.092 | ++-- | ++-- (4/4) |

## Gates

- gate 1 (band upgrade on SC or RCD): {'SC': False, 'RCD': False}
- gate 2 (mean <= 1.10 x 1.0357 = 1.1392): after mean 1.0314 -> True
- RCE protected-row failures: {'band drop': False, 'sign lost': False, 'magnitude halved': True}
- verdict: **[FAIL]**
