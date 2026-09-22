# Behaviour versus evaluated, 200-step guard pilots

Mean punishment served by the RL manager to its own group. The
behaviour rows are the rollouts that fill the replay buffer; the
evaluated rows are the fully deterministic rollout at the same
update steps, every exploration mechanism off. A ratio away from 1
describes what the buffer holds; it is not a defect, because DQN is
off-policy and is meant to evaluate the greedy policy whatever
collected the data.

| run | sampling | behaviour | evaluated | ratio | update steps |
|---|---|---|---|---|---|
| rl_epsgreedy_guard | eps-greedy | 4.9307 | 4.1913 | 1.176 | 20 |
| rl_pnoise_guard | param-noise | 4.9737 | 4.1941 | 1.186 | 20 |

## Policy shape and targeting

Mean punishment per contribution bin, evaluation-suite RPA bins.

`rho` is the count-weighted rank correlation between the contribution
bin and the punishment served. It is invariant to ANY monotone
rescaling of punishment, so no amount of punishing harder or softer
can move it -- which `contrast`, a difference of bin means, cannot
say. Negative is the human sign: punish the free-rider, leave the
full contributor alone. Human managers sit at -1.0, a manager with
the inversion at +1.0. `rho` carries no magnitude, so read it with
`contrast_over_mean`, and read neither when `profile_snr` is small --
below about 2 the six bin means are within their own sampling noise
and `rho` is ranking noise. NaN means a flat profile: nothing to rank.

| manager | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} | rho_contribution_punishment | contrast | contrast_over_mean | mean_punishment | profile_snr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rl_epsgreedy_guard [eps-greedy] | 6.014 | 6.004 | 6.011 | 5.997 | 6.027 | 6.001 | -0.540 | 0.014 | 0.002 | 6.006 | 2.903 |
| rl_epsgreedy_guard [greedy] | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 |  | 0.000 | 0.000 | 5.000 | inf |
| rl_pnoise_guard [greedy] | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 |  | 0.000 | 0.000 | 5.000 | inf |
| rl_pnoise_guard [param-noise] | 6.555 | 6.089 | 5.752 | 5.532 | 5.546 | 5.716 | -0.827 | 0.839 | 0.143 | 5.848 | 3.023 |
| artificial punisher (clone) | 3.722 | 2.658 | 1.720 | 1.094 | 0.844 | 0.322 | -1.000 | 3.400 | 2.087 | 1.630 |  |
| human managers | 4.755 | 2.973 | 1.672 | 0.978 | 0.692 | 0.267 | -1.000 | 4.488 | 2.430 | 1.847 |  |

Row counts per bin:

| manager | n[{0}] | n[1-5] | n[6-10] | n[11-15] | n[16-19] | n[{20}] |
|---|---|---|---|---|---|---|
| rl_epsgreedy_guard [eps-greedy] | 162744 | 246607 | 320817 | 229082 | 85746 | 298069 |
| rl_epsgreedy_guard [greedy] | 166465 | 264968 | 351029 | 258455 | 92682 | 314709 |
| rl_pnoise_guard [greedy] | 166418 | 260175 | 349885 | 257098 | 93183 | 315140 |
| rl_pnoise_guard [param-noise] | 167001 | 247804 | 331710 | 242879 | 90534 | 313865 |
| artificial punisher (clone) | 442067 | 876416 | 1209967 | 890903 | 279313 | 947541 |
| human managers | 809 | 1955 | 2614 | 1794 | 510 | 1232 |

## Does the buffer's shape differ the way uniform noise would

A description of what was SAMPLED, not of what was learned.

Within each contribution bin, epsilon-greedy should drag the
behaviour mean toward the uniform mean of 15 by exactly
`eps * (15 - evaluated)` -- no free parameters. `drag_slope`
regresses the observed per-bin shift on that prediction through
the origin. Weight noise has no uniform action mean to drag
toward, so its slope should collapse even where the shifts
themselves are large; `mean_shift_spread` is the standard
deviation across episodes of each bin's behaviour mean, small
when every episode is flattened the same way and large when each
episode carries its own contingency.

| job_id / sampling | drag_slope | mean_abs_shift | contrast_evaluated | contrast_behaviour | contrast_flattening | mean_shift_spread |
|---|---|---|---|---|---|---|
| rl_epsgreedy_guard / eps-greedy | 1.0089 | 1.0089 | 0.0000 | 0.0137 |  | 0.0515 |
| rl_pnoise_guard / param-noise | 0.8651 | 0.8651 | 0.0000 | 0.8387 |  | 1.5848 |

## Noise scale

Final scale 3.01562, holding divergence 0.365 punishment levels (l2 0.06513, w1 3.894).
