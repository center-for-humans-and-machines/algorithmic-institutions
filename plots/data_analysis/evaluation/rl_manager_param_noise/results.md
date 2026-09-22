# Parameter-space noise: results

Evaluated policy, pooled over the last 10 of 200 evaluation points.

## Did the shape recover

| manager | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} |
|---|---|---|---|---|---|---|
| rl_pnoise_s42 | 9.359 | 1.780 | 1.944 | 4.959 | 5.000 | 5.000 |
| rl_pnoise_s43 | 18.193 | 5.069 | 0.080 | 0.000 | 0.000 | 0.000 |
| rl_pnoise_s44 | 13.740 | 3.937 | 0.023 | 0.909 | 5.510 | 5.998 |
| rl_pnoise_s45 | 0.023 | 0.024 | 0.811 | 1.000 | 1.000 | 1.000 |
| rl_pnoise_s46 | 0.999 | 0.884 | 0.051 | 0.000 | 0.000 | 0.000 |
| artificial punisher (clone) | 4.003 | 2.605 | 1.725 | 1.099 | 0.835 | 0.338 |
| human managers | 4.755 | 2.973 | 1.672 | 0.978 | 0.692 | 0.267 |

| manager | verdict | verdict_shape_only | tie_attenuated | monotonicity | rho_contribution_punishment | tau_b | n_distinct_bins | n_zero_bins | relative_range | contrast | contrast_sd_across_eval_points |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rl_pnoise_s42 | no clean targeting (not monotone) | no clean targeting (not monotone) | False | none | 0.441 | 0.540 | 5 | 0 | 2.048 | 4.359 | 3.684 |
| rl_pnoise_s43 | targets free-riders | targets free-riders | False | decreasing | -0.973 | -0.939 | 4 | 3 | 10.320 | 18.193 | 2.404 |
| rl_pnoise_s44 | no clean targeting (not monotone) | no clean targeting (not monotone) | False | none | -0.018 | 0.012 | 6 | 0 | 4.716 | 7.742 | 5.393 |
| rl_pnoise_s45 | inverted | inverted | False | increasing | 0.982 | 0.954 | 4 | 0 | 1.782 | -0.977 | 0.051 |
| rl_pnoise_s46 | targets free-riders | targets free-riders | False | decreasing | -0.940 | -0.853 | 5 | 2 | 2.794 | 0.999 | 0.107 |
| artificial punisher (clone) | targets free-riders | targets free-riders | False | decreasing | -1.000 | -1.000 | 6 | 0 | 2.125 | 3.665 |  |
| human managers | targets free-riders | targets free-riders | False | decreasing | -1.000 | -1.000 | 6 | 0 | 2.430 | 4.488 |  |

Row counts per bin:

| manager | n[{0}] | n[1-5] | n[6-10] | n[11-15] | n[16-19] | n[{20}] |
|---|---|---|---|---|---|---|
| rl_pnoise_s42 | 78947 | 254912 | 188661 | 114086 | 40938 | 136377 |
| rl_pnoise_s43 | 40660 | 193182 | 371334 | 193935 | 49277 | 143528 |
| rl_pnoise_s44 | 51409 | 225480 | 323605 | 145177 | 34509 | 111875 |
| rl_pnoise_s45 | 122537 | 308678 | 248015 | 160219 | 48199 | 148259 |
| rl_pnoise_s46 | 102737 | 291909 | 301818 | 174372 | 46874 | 134110 |
| artificial punisher (clone) | 409838 | 1096963 | 1212987 | 818694 | 264735 | 831287 |
| human managers | 809 | 1955 | 2614 | 1794 | 510 | 1232 |

## What it spent

Paired within-run comparison against the artificial punisher, same
episodes, same contributors, so no separate baseline run is needed.

Read `payoff_per_member_minus_clone`, NOT the summed version.
`group_payoff_sum` sums over whoever is in the group and membership
is endogenous -- players switch in response to punishment -- so a
manager that merely retains more members scores higher on the sum
without anyone being better off. `common_good` is the per-capita
pool and is the cleanest welfare number here.

Targeting and restraint come apart: a seed that targets well and
overpays is not a seed that helps.

| manager | punishment | common_good | contribution | group_payoff_sum | opp_sum_payoff | opp_punishment | rl_avg_group_size | opp_avg_group_size | payoff_per_member | opp_payoff_per_member | payoff_per_member_minus_clone | payoff_sum_minus_clone | punishment_minus_clone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rl_pnoise_s42 | 3.384 | 9.464 | 8.087 | 62.989 | 103.184 | 1.710 | 3.415 | 4.585 | 18.447 | 22.503 | -4.056 | -40.195 | 1.674 |
| rl_pnoise_s43 | 1.853 | 12.660 | 9.093 | 95.664 | 84.968 | 1.632 | 4.218 | 3.782 | 22.678 | 22.469 | 0.209 | 10.696 | 0.221 |
| rl_pnoise_s44 | 2.906 | 10.452 | 8.391 | 75.311 | 95.931 | 1.711 | 3.762 | 4.238 | 20.021 | 22.634 | -2.612 | -20.620 | 1.195 |
| rl_pnoise_s45 | 0.502 | 12.345 | 8.057 | 106.495 | 82.202 | 1.470 | 4.408 | 3.592 | 24.161 | 22.883 | 1.279 | 24.294 | -0.968 |
| rl_pnoise_s46 | 0.346 | 12.559 | 8.089 | 110.179 | 79.589 | 1.509 | 4.480 | 3.520 | 24.595 | 22.609 | 1.985 | 30.590 | -1.163 |

## Was the mechanism live

Final adapted scale and the divergence it held, per seed. A scale
at the cap with the divergence below target means weight noise
could not match epsilon-greedy's displacement and the arm
under-explored.

| job_id | param_noise_scale | param_noise_target | param_noise_divergence | param_noise_divergence_l2 |
|---|---|---|---|---|
| rl_pnoise_s42 | 0.2481 | 1.2188 | 1.7842 | 0.0313 |
| rl_pnoise_s43 | 0.2714 | 1.4320 | 0.9999 | 0.0367 |
| rl_pnoise_s44 | 0.1994 | 1.3525 | 0.5214 | 0.0183 |
| rl_pnoise_s45 | 2.6235 | 1.4308 | 16.7192 | 0.2035 |
| rl_pnoise_s46 | 2.5718 | 1.4768 | 1.0794 | 0.0952 |
