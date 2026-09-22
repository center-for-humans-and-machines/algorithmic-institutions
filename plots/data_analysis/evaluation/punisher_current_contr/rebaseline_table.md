# Punisher current-contribution re-baseline

Scores are multiples of the human noise ceiling (<= 1 at the ceiling, 1-2 minor, 2-5 clear, > 5 not reproduced); delta = after - before, negative is an improvement. `before` = the source sim with the prev-contribution punisher, rescored with the 22-row suite; `after` = the same contributor/switch stack with the punisher retrained on the current contribution.

## Summary

| case | after_available | mean_before | mean_after | mean_delta | rows_le1_before | rows_le1_after | RCE_before | RCE_after | RCE_band |
|---|---|---|---|---|---|---|---|---|---|
| a_vnode | True | 1.099 | 1.108 | 0.009 | 12 | 9 | 1.100 | 1.268 | 1-2 -> 1-2 |
| b_skip | True | 1.096 | 1.036 | -0.060 | 13 | 13 | 0.906 | 0.894 | <= 1 -> <= 1 |
| c_infl | True | 1.119 | 1.101 | -0.017 | 12 | 12 | 0.989 | 0.851 | <= 1 -> <= 1 |
| d_kexo | True | 1.205 | 1.188 | -0.017 | 12 | 8 | 0.832 | 0.705 | <= 1 -> <= 1 |
| e_lin | True | 1.729 | 1.740 | 0.011 | 11 | 13 | 1.092 | 0.998 | 1-2 -> <= 1 |
| e_gnn | True | 1.866 | 1.709 | -0.156 | 7 | 8 | 0.986 | 1.005 | <= 1 -> 1-2 |

## RCE per band: OLS slope of next-round change on punishment received

Human pattern: comply when punished at low contribution, withdraw at high (++--). `signs_vs_human` counts matching signs.

| case | stage | slope_0-4 | slope_5-9 | slope_10-14 | slope_15-19 | signs | signs_vs_human |
|---|---|---|---|---|---|---|---|
| human |  | 0.140 | 0.104 | -0.077 | -0.161 | ++-- |  |
| a_vnode | before | 0.062 | 0.012 | -0.008 | -0.037 | ++-- | ++-- (4/4) |
| a_vnode | after | 0.049 | -0.007 | -0.017 | 0.040 | +--+ | +--+ (2/4) |
| b_skip | before | 0.073 | 0.054 | -0.047 | -0.025 | ++-- | ++-- (4/4) |
| b_skip | after | 0.095 | 0.020 | -0.058 | -0.160 | ++-- | ++-- (4/4) |
| c_infl | before | 0.033 | 0.061 | -0.051 | -0.231 | ++-- | ++-- (4/4) |
| c_infl | after | 0.068 | 0.109 | -0.020 | -0.193 | ++-- | ++-- (4/4) |
| d_kexo | before | 0.103 | 0.037 | -0.116 | -0.174 | ++-- | ++-- (4/4) |
| d_kexo | after | 0.120 | 0.071 | -0.112 | -0.205 | ++-- | ++-- (4/4) |
| e_lin | before | 0.062 | 0.043 | 0.047 | -0.042 | +++- | +++- (3/4) |
| e_lin | after | 0.064 | 0.092 | 0.044 | -0.098 | +++- | +++- (3/4) |
| e_gnn | before | 0.090 | 0.021 | -0.005 | -0.000 | ++-- | ++-- (4/4) |
| e_gnn | after | 0.073 | 0.050 | 0.013 | 0.029 | ++++ | ++++ (2/4) |

## Per case

### a: PR 179 group vnode, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/a_vnode/scores.csv`), after `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.961 | 1.198 | 0.237 | <= 1 -> 1-2 |
| CB | 0.953 | 1.213 | 0.260 | <= 1 -> 1-2 |
| CC | 0.920 | 0.991 | 0.071 | <= 1 -> <= 1 |
| CD | 0.922 | 1.153 | 0.231 | <= 1 -> 1-2 |
| CE | 1.111 | 1.010 | -0.101 | 1-2 -> 1-2 |
| CF | 1.076 | 1.026 | -0.051 | 1-2 -> 1-2 |
| CG | 0.899 | 0.884 | -0.015 | <= 1 -> <= 1 |
| SA | 0.864 | 0.866 | 0.001 | <= 1 -> <= 1 |
| SB | 1.111 | 1.007 | -0.104 | 1-2 -> 1-2 |
| SC | 0.977 | 1.356 | 0.378 | <= 1 -> 1-2 |
| PA | 0.582 | 0.706 | 0.124 | <= 1 -> <= 1 |
| PB | 0.919 | 0.912 | -0.007 | <= 1 -> <= 1 |
| PC | 0.865 | 0.933 | 0.067 | <= 1 -> <= 1 |
| PD | 0.775 | 0.999 | 0.224 | <= 1 -> <= 1 |
| RCA | 1.400 | 1.423 | 0.023 | 1-2 -> 1-2 |
| RCB | 2.315 | 1.474 | -0.841 | 2-5 -> 1-2 |
| RCC | 1.660 | 1.711 | 0.051 | 1-2 -> 1-2 |
| RCD | 1.340 | 1.428 | 0.088 | 1-2 -> 1-2 |
| RCE | 1.100 | 1.268 | 0.168 | 1-2 -> 1-2 |
| RSA | 1.355 | 1.284 | -0.070 | 1-2 -> 1-2 |
| RPA | 1.311 | 0.723 | -0.588 | 1-2 -> <= 1 |
| RPB | 0.758 | 0.809 | 0.051 | <= 1 -> <= 1 |
| mean | 1.099 | 1.108 | 0.009 |  |
| rows <= 1 | 12.000 | 9.000 | -3.000 |  |

### b: PR 181 stimulus skip, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/b_skip/scores.csv`), after `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.848 | 0.860 | 0.012 | <= 1 -> <= 1 |
| CB | 0.830 | 0.788 | -0.042 | <= 1 -> <= 1 |
| CC | 0.821 | 0.889 | 0.068 | <= 1 -> <= 1 |
| CD | 0.796 | 0.810 | 0.015 | <= 1 -> <= 1 |
| CE | 0.910 | 1.057 | 0.148 | <= 1 -> 1-2 |
| CF | 0.887 | 0.828 | -0.059 | <= 1 -> <= 1 |
| CG | 1.310 | 1.554 | 0.243 | 1-2 -> 1-2 |
| SA | 0.784 | 0.785 | 0.001 | <= 1 -> <= 1 |
| SB | 1.040 | 1.006 | -0.034 | 1-2 -> 1-2 |
| SC | 1.023 | 1.427 | 0.404 | 1-2 -> 1-2 |
| PA | 0.630 | 0.660 | 0.030 | <= 1 -> <= 1 |
| PB | 0.901 | 0.969 | 0.068 | <= 1 -> <= 1 |
| PC | 0.877 | 0.907 | 0.030 | <= 1 -> <= 1 |
| PD | 0.854 | 0.722 | -0.132 | <= 1 -> <= 1 |
| RCA | 1.469 | 1.633 | 0.163 | 1-2 -> 1-2 |
| RCB | 2.087 | 1.545 | -0.541 | 2-5 -> 1-2 |
| RCC | 1.613 | 1.530 | -0.083 | 1-2 -> 1-2 |
| RCD | 2.205 | 1.309 | -0.896 | 2-5 -> 1-2 |
| RCE | 0.906 | 0.894 | -0.012 | <= 1 -> <= 1 |
| RSA | 1.236 | 1.070 | -0.166 | 1-2 -> 1-2 |
| RPA | 1.227 | 0.693 | -0.534 | 1-2 -> <= 1 |
| RPB | 0.847 | 0.847 | -0.000 | <= 1 -> <= 1 |
| mean | 1.096 | 1.036 | -0.060 |  |
| rows <= 1 | 13.000 | 13.000 | 0.000 |  |

### c: PR 177 inflated gmlp, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/c_infl/scores.csv`), after `plots/simulation/23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 1.442 | 1.508 | 0.066 | 1-2 -> 1-2 |
| CB | 1.017 | 1.174 | 0.157 | 1-2 -> 1-2 |
| CC | 0.860 | 0.968 | 0.108 | <= 1 -> <= 1 |
| CD | 0.961 | 1.100 | 0.138 | <= 1 -> 1-2 |
| CE | 0.803 | 0.850 | 0.047 | <= 1 -> <= 1 |
| CF | 0.855 | 0.884 | 0.030 | <= 1 -> <= 1 |
| CG | 1.842 | 1.854 | 0.012 | 1-2 -> 1-2 |
| SA | 0.916 | 0.887 | -0.029 | <= 1 -> <= 1 |
| SB | 0.965 | 0.896 | -0.070 | <= 1 -> <= 1 |
| SC | 1.233 | 1.579 | 0.345 | 1-2 -> 1-2 |
| PA | 0.621 | 0.782 | 0.161 | <= 1 -> <= 1 |
| PB | 0.952 | 1.014 | 0.062 | <= 1 -> 1-2 |
| PC | 0.891 | 0.950 | 0.059 | <= 1 -> <= 1 |
| PD | 0.919 | 0.920 | 0.001 | <= 1 -> <= 1 |
| RCA | 1.862 | 1.690 | -0.173 | 1-2 -> 1-2 |
| RCB | 1.462 | 0.966 | -0.496 | 1-2 -> <= 1 |
| RCC | 1.327 | 1.381 | 0.055 | 1-2 -> 1-2 |
| RCD | 1.353 | 1.262 | -0.091 | 1-2 -> 1-2 |
| RCE | 0.989 | 0.851 | -0.138 | <= 1 -> <= 1 |
| RSA | 1.335 | 1.166 | -0.168 | 1-2 -> 1-2 |
| RPA | 1.275 | 0.735 | -0.540 | 1-2 -> <= 1 |
| RPB | 0.731 | 0.810 | 0.079 | <= 1 -> <= 1 |
| mean | 1.119 | 1.101 | -0.017 |  |
| rows <= 1 | 12.000 | 12.000 | 0.000 |  |

### d: PR 174 k-one-hot switch on gmlp copula, lin_multinomial copula

Run `ah group_switching managed by lin_multinomial_copula_self`; before `plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/d_kexo/scores.csv`), after `plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 1.609 | 1.703 | 0.094 | 1-2 -> 1-2 |
| CB | 0.936 | 1.144 | 0.208 | <= 1 -> 1-2 |
| CC | 1.036 | 1.134 | 0.098 | 1-2 -> 1-2 |
| CD | 1.116 | 1.322 | 0.206 | 1-2 -> 1-2 |
| CE | 0.947 | 1.022 | 0.075 | <= 1 -> 1-2 |
| CF | 1.345 | 1.369 | 0.024 | 1-2 -> 1-2 |
| CG | 2.079 | 1.665 | -0.414 | 2-5 -> 1-2 |
| SA | 0.810 | 1.082 | 0.272 | <= 1 -> 1-2 |
| SB | 0.891 | 0.931 | 0.040 | <= 1 -> <= 1 |
| SC | 0.980 | 1.076 | 0.096 | <= 1 -> 1-2 |
| PA | 0.619 | 0.856 | 0.237 | <= 1 -> <= 1 |
| PB | 0.960 | 1.011 | 0.050 | <= 1 -> 1-2 |
| PC | 0.888 | 0.975 | 0.087 | <= 1 -> <= 1 |
| PD | 0.865 | 0.732 | -0.134 | <= 1 -> <= 1 |
| RCA | 3.507 | 3.428 | -0.079 | 2-5 -> 2-5 |
| RCB | 1.910 | 1.332 | -0.579 | 1-2 -> 1-2 |
| RCC | 1.088 | 1.169 | 0.081 | 1-2 -> 1-2 |
| RCD | 0.732 | 0.760 | 0.029 | <= 1 -> <= 1 |
| RCE | 0.832 | 0.705 | -0.127 | <= 1 -> <= 1 |
| RSA | 1.317 | 1.196 | -0.121 | 1-2 -> 1-2 |
| RPA | 1.314 | 0.736 | -0.578 | 1-2 -> <= 1 |
| RPB | 0.727 | 0.791 | 0.064 | <= 1 -> <= 1 |
| mean | 1.205 | 1.188 | -0.017 |  |
| rows <= 1 | 12.000 | 8.000 | -4.000 |  |

### e: main gnn x gnn, lin_multinomial (no copula)

Run `ah group_switching managed by lin_multinomial_self`; before `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/e_main/scores.csv`), after `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.772 | 0.969 | 0.197 | <= 1 -> <= 1 |
| CB | 0.691 | 0.850 | 0.159 | <= 1 -> <= 1 |
| CC | 1.606 | 1.821 | 0.215 | 1-2 -> 1-2 |
| CD | 0.650 | 0.843 | 0.193 | <= 1 -> <= 1 |
| CE | 1.332 | 1.311 | -0.021 | 1-2 -> 1-2 |
| CF | 0.814 | 0.891 | 0.077 | <= 1 -> <= 1 |
| CG | 9.850 | 10.282 | 0.432 | > 5 -> > 5 |
| SA | 0.721 | 0.821 | 0.099 | <= 1 -> <= 1 |
| SB | 0.754 | 0.837 | 0.083 | <= 1 -> <= 1 |
| SC | 3.270 | 3.440 | 0.170 | 2-5 -> 2-5 |
| PA | 0.634 | 0.648 | 0.014 | <= 1 -> <= 1 |
| PB | 0.878 | 0.808 | -0.070 | <= 1 -> <= 1 |
| PC | 0.778 | 0.764 | -0.014 | <= 1 -> <= 1 |
| PD | 2.935 | 3.205 | 0.270 | 2-5 -> 2-5 |
| RCA | 2.035 | 1.938 | -0.097 | 2-5 -> 1-2 |
| RCB | 1.928 | 1.021 | -0.908 | 1-2 -> 1-2 |
| RCC | 1.539 | 1.618 | 0.079 | 1-2 -> 1-2 |
| RCD | 2.772 | 2.933 | 0.161 | 2-5 -> 2-5 |
| RCE | 1.092 | 0.998 | -0.094 | 1-2 -> <= 1 |
| RSA | 0.909 | 0.944 | 0.035 | <= 1 -> <= 1 |
| RPA | 1.268 | 0.684 | -0.584 | 1-2 -> <= 1 |
| RPB | 0.814 | 0.665 | -0.149 | <= 1 -> <= 1 |
| mean | 1.729 | 1.740 | 0.011 |  |
| rows <= 1 | 11.000 | 13.000 | 2.000 |  |

### e: main gnn x gnn, gnn punisher

Run `ah group_switching managed by gnn_self`; before `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch` (rescored, `plots/data_analysis/evaluation/punisher_current_contr/before/e_main/scores.csv`), after `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch_curpun`.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.842 | 0.788 | -0.054 | <= 1 -> <= 1 |
| CB | 0.685 | 0.681 | -0.004 | <= 1 -> <= 1 |
| CC | 1.712 | 1.569 | -0.143 | 1-2 -> 1-2 |
| CD | 0.670 | 0.664 | -0.006 | <= 1 -> <= 1 |
| CE | 1.356 | 1.301 | -0.055 | 1-2 -> 1-2 |
| CF | 0.852 | 0.809 | -0.043 | <= 1 -> <= 1 |
| CG | 10.138 | 9.221 | -0.917 | > 5 -> > 5 |
| SA | 0.659 | 0.883 | 0.224 | <= 1 -> <= 1 |
| SB | 0.744 | 0.900 | 0.156 | <= 1 -> <= 1 |
| SC | 3.455 | 2.973 | -0.482 | 2-5 -> 2-5 |
| PA | 1.267 | 1.309 | 0.042 | 1-2 -> 1-2 |
| PB | 1.114 | 1.150 | 0.036 | 1-2 -> 1-2 |
| PC | 1.160 | 0.995 | -0.166 | 1-2 -> <= 1 |
| PD | 2.823 | 2.690 | -0.133 | 2-5 -> 2-5 |
| RCA | 2.367 | 2.086 | -0.280 | 2-5 -> 2-5 |
| RCB | 1.886 | 1.213 | -0.672 | 1-2 -> 1-2 |
| RCC | 1.407 | 1.301 | -0.107 | 1-2 -> 1-2 |
| RCD | 2.893 | 2.846 | -0.047 | 2-5 -> 2-5 |
| RCE | 0.986 | 1.005 | 0.019 | <= 1 -> 1-2 |
| RSA | 1.137 | 1.004 | -0.133 | 1-2 -> 1-2 |
| RPA | 1.555 | 0.888 | -0.667 | 1-2 -> <= 1 |
| RPB | 1.343 | 1.332 | -0.012 | 1-2 -> 1-2 |
| mean | 1.866 | 1.709 | -0.156 |  |
| rows <= 1 | 7.000 | 8.000 | 1.000 |  |
