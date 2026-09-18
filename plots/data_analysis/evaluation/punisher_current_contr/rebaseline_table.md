# Punisher current-contribution re-baseline

Scores are multiples of the human noise ceiling (<= 1 at the ceiling, 1-2 minor, 2-5 clear, > 5 not reproduced); delta = after - before, negative is an improvement. `before` = the source sim with the prev-contribution punisher, rescored with the 22-row suite; `after` = the same contributor/switch stack with the punisher retrained on the current contribution.

## Summary

| case | after_available | mean_before | mean_after | mean_delta | rows_le1_before | rows_le1_after | RCE_before | RCE_after | RCE_band |
|---|---|---|---|---|---|---|---|---|---|
| a_vnode | False | 1.099 | nan | nan | 12 | None | 1.100 | nan | 1-2 |
| b_skip | False | 1.096 | nan | nan | 13 | None | 0.906 | nan | <= 1 |
| c_infl | False | 1.119 | nan | nan | 12 | None | 0.989 | nan | <= 1 |
| d_kexo | False | 1.205 | nan | nan | 12 | None | 0.832 | nan | <= 1 |
| e_lin | False | 1.729 | nan | nan | 11 | None | 1.092 | nan | 1-2 |
| e_gnn | False | 1.866 | nan | nan | 7 | None | 0.986 | nan | <= 1 |

## RCE per band: OLS slope of next-round change on punishment received

Human pattern: comply when punished at low contribution, withdraw at high (++--). `signs_vs_human` counts matching signs.

| case | stage | slope_0-4 | slope_5-9 | slope_10-14 | slope_15-19 | signs | signs_vs_human |
|---|---|---|---|---|---|---|---|
| human |  | 0.140 | 0.104 | -0.077 | -0.161 | ++-- |  |
| a_vnode | before | 0.062 | 0.012 | -0.008 | -0.037 | ++-- | ++-- (4/4) |
| b_skip | before | 0.073 | 0.054 | -0.047 | -0.025 | ++-- | ++-- (4/4) |
| c_infl | before | 0.033 | 0.061 | -0.051 | -0.231 | ++-- | ++-- (4/4) |
| d_kexo | before | 0.103 | 0.037 | -0.116 | -0.174 | ++-- | ++-- (4/4) |
| e_lin | before | 0.062 | 0.043 | 0.047 | -0.042 | +++- | +++- (3/4) |
| e_gnn | before | 0.090 | 0.021 | -0.005 | -0.000 | ++-- | ++-- (4/4) |

## Per case

### a: PR 179 group vnode, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/a_vnode/scores.csv`), after `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.961 | nan | nan | <= 1 |
| CB | 0.953 | nan | nan | <= 1 |
| CC | 0.920 | nan | nan | <= 1 |
| CD | 0.922 | nan | nan | <= 1 |
| CE | 1.111 | nan | nan | 1-2 |
| CF | 1.076 | nan | nan | 1-2 |
| CG | 0.899 | nan | nan | <= 1 |
| SA | 0.864 | nan | nan | <= 1 |
| SB | 1.111 | nan | nan | 1-2 |
| SC | 0.977 | nan | nan | <= 1 |
| PA | 0.582 | nan | nan | <= 1 |
| PB | 0.919 | nan | nan | <= 1 |
| PC | 0.865 | nan | nan | <= 1 |
| PD | 0.775 | nan | nan | <= 1 |
| RCA | 1.400 | nan | nan | 1-2 |
| RCB | 2.315 | nan | nan | 2-5 |
| RCC | 1.660 | nan | nan | 1-2 |
| RCD | 1.340 | nan | nan | 1-2 |
| RCE | 1.100 | nan | nan | 1-2 |
| RSA | 1.355 | nan | nan | 1-2 |
| RPA | 1.311 | nan | nan | 1-2 |
| RPB | 0.758 | nan | nan | <= 1 |
| mean | 1.099 | nan | nan |  |
| rows <= 1 | 12.000 | nan | nan |  |

### b: PR 181 stimulus skip, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/b_skip/scores.csv`), after `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.848 | nan | nan | <= 1 |
| CB | 0.830 | nan | nan | <= 1 |
| CC | 0.821 | nan | nan | <= 1 |
| CD | 0.796 | nan | nan | <= 1 |
| CE | 0.910 | nan | nan | <= 1 |
| CF | 0.887 | nan | nan | <= 1 |
| CG | 1.310 | nan | nan | 1-2 |
| SA | 0.784 | nan | nan | <= 1 |
| SB | 1.040 | nan | nan | 1-2 |
| SC | 1.023 | nan | nan | 1-2 |
| PA | 0.630 | nan | nan | <= 1 |
| PB | 0.901 | nan | nan | <= 1 |
| PC | 0.877 | nan | nan | <= 1 |
| PD | 0.854 | nan | nan | <= 1 |
| RCA | 1.469 | nan | nan | 1-2 |
| RCB | 2.087 | nan | nan | 2-5 |
| RCC | 1.613 | nan | nan | 1-2 |
| RCD | 2.205 | nan | nan | 2-5 |
| RCE | 0.906 | nan | nan | <= 1 |
| RSA | 1.236 | nan | nan | 1-2 |
| RPA | 1.227 | nan | nan | 1-2 |
| RPB | 0.847 | nan | nan | <= 1 |
| mean | 1.096 | nan | nan |  |
| rows <= 1 | 13.000 | nan | nan |  |

### c: PR 177 inflated gmlp, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/c_infl/scores.csv`), after `plots/simulation/23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 1.442 | nan | nan | 1-2 |
| CB | 1.017 | nan | nan | 1-2 |
| CC | 0.860 | nan | nan | <= 1 |
| CD | 0.961 | nan | nan | <= 1 |
| CE | 0.803 | nan | nan | <= 1 |
| CF | 0.855 | nan | nan | <= 1 |
| CG | 1.842 | nan | nan | 1-2 |
| SA | 0.916 | nan | nan | <= 1 |
| SB | 0.965 | nan | nan | <= 1 |
| SC | 1.233 | nan | nan | 1-2 |
| PA | 0.621 | nan | nan | <= 1 |
| PB | 0.952 | nan | nan | <= 1 |
| PC | 0.891 | nan | nan | <= 1 |
| PD | 0.919 | nan | nan | <= 1 |
| RCA | 1.862 | nan | nan | 1-2 |
| RCB | 1.462 | nan | nan | 1-2 |
| RCC | 1.327 | nan | nan | 1-2 |
| RCD | 1.353 | nan | nan | 1-2 |
| RCE | 0.989 | nan | nan | <= 1 |
| RSA | 1.335 | nan | nan | 1-2 |
| RPA | 1.275 | nan | nan | 1-2 |
| RPB | 0.731 | nan | nan | <= 1 |
| mean | 1.119 | nan | nan |  |
| rows <= 1 | 12.000 | nan | nan |  |

### d: PR 174 k-one-hot switch on gmlp copula, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/d_kexo/scores.csv`), after `plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 1.609 | nan | nan | 1-2 |
| CB | 0.936 | nan | nan | <= 1 |
| CC | 1.036 | nan | nan | 1-2 |
| CD | 1.116 | nan | nan | 1-2 |
| CE | 0.947 | nan | nan | <= 1 |
| CF | 1.345 | nan | nan | 1-2 |
| CG | 2.079 | nan | nan | 2-5 |
| SA | 0.810 | nan | nan | <= 1 |
| SB | 0.891 | nan | nan | <= 1 |
| SC | 0.980 | nan | nan | <= 1 |
| PA | 0.619 | nan | nan | <= 1 |
| PB | 0.960 | nan | nan | <= 1 |
| PC | 0.888 | nan | nan | <= 1 |
| PD | 0.865 | nan | nan | <= 1 |
| RCA | 3.507 | nan | nan | 2-5 |
| RCB | 1.910 | nan | nan | 1-2 |
| RCC | 1.088 | nan | nan | 1-2 |
| RCD | 0.732 | nan | nan | <= 1 |
| RCE | 0.832 | nan | nan | <= 1 |
| RSA | 1.317 | nan | nan | 1-2 |
| RPA | 1.314 | nan | nan | 1-2 |
| RPB | 0.727 | nan | nan | <= 1 |
| mean | 1.205 | nan | nan |  |
| rows <= 1 | 12.000 | nan | nan |  |

### e: main gnn x gnn, lin_multinomial (no copula)

Run `ah group_switching managed by lin_multinomial_self`; before `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/e_main/scores.csv`), after `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.772 | nan | nan | <= 1 |
| CB | 0.691 | nan | nan | <= 1 |
| CC | 1.606 | nan | nan | 1-2 |
| CD | 0.650 | nan | nan | <= 1 |
| CE | 1.332 | nan | nan | 1-2 |
| CF | 0.814 | nan | nan | <= 1 |
| CG | 9.850 | nan | nan | > 5 |
| SA | 0.721 | nan | nan | <= 1 |
| SB | 0.754 | nan | nan | <= 1 |
| SC | 3.270 | nan | nan | 2-5 |
| PA | 0.634 | nan | nan | <= 1 |
| PB | 0.878 | nan | nan | <= 1 |
| PC | 0.778 | nan | nan | <= 1 |
| PD | 2.935 | nan | nan | 2-5 |
| RCA | 2.035 | nan | nan | 2-5 |
| RCB | 1.928 | nan | nan | 1-2 |
| RCC | 1.539 | nan | nan | 1-2 |
| RCD | 2.772 | nan | nan | 2-5 |
| RCE | 1.092 | nan | nan | 1-2 |
| RSA | 0.909 | nan | nan | <= 1 |
| RPA | 1.268 | nan | nan | 1-2 |
| RPB | 0.814 | nan | nan | <= 1 |
| mean | 1.729 | nan | nan |  |
| rows <= 1 | 11.000 | nan | nan |  |

### e: main gnn x gnn, gnn punisher

Run `ah group_switching managed by gnn_self`; before `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/e_main/scores.csv`), after `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.842 | nan | nan | <= 1 |
| CB | 0.685 | nan | nan | <= 1 |
| CC | 1.712 | nan | nan | 1-2 |
| CD | 0.670 | nan | nan | <= 1 |
| CE | 1.356 | nan | nan | 1-2 |
| CF | 0.852 | nan | nan | <= 1 |
| CG | 10.138 | nan | nan | > 5 |
| SA | 0.659 | nan | nan | <= 1 |
| SB | 0.744 | nan | nan | <= 1 |
| SC | 3.455 | nan | nan | 2-5 |
| PA | 1.267 | nan | nan | 1-2 |
| PB | 1.114 | nan | nan | 1-2 |
| PC | 1.160 | nan | nan | 1-2 |
| PD | 2.823 | nan | nan | 2-5 |
| RCA | 2.367 | nan | nan | 2-5 |
| RCB | 1.886 | nan | nan | 1-2 |
| RCC | 1.407 | nan | nan | 1-2 |
| RCD | 2.893 | nan | nan | 2-5 |
| RCE | 0.986 | nan | nan | <= 1 |
| RSA | 1.137 | nan | nan | 1-2 |
| RPA | 1.555 | nan | nan | 1-2 |
| RPB | 1.343 | nan | nan | 1-2 |
| mean | 1.866 | nan | nan |  |
| rows <= 1 | 7.000 | nan | nan |  |
