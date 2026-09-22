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

## Policy shape

Mean punishment per contribution bin, evaluation-suite RPA bins.
Human managers fall from 4.76 at contribution 0 to 0.27 at 20; two
of the three finished learned seeds rose instead. `contrast` is
{0} minus {20}: positive is the human sign.

| manager | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} | n[{0}] | n[1-5] | n[6-10] | n[11-15] | n[16-19] | n[{20}] | contrast_0_minus_20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rl_epsgreedy_guard [eps-greedy] | 6.014 | 6.004 | 6.011 | 5.997 | 6.027 | 6.001 | 162744.000 | 246607.000 | 320817.000 | 229082.000 | 85746.000 | 298069.000 | 0.014 |
| rl_epsgreedy_guard [greedy] | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | 166465.000 | 264968.000 | 351029.000 | 258455.000 | 92682.000 | 314709.000 | 0.000 |
| rl_pnoise_guard [greedy] | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | 166418.000 | 260175.000 | 349885.000 | 257098.000 | 93183.000 | 315140.000 | 0.000 |
| rl_pnoise_guard [param-noise] | 6.555 | 6.089 | 5.752 | 5.532 | 5.546 | 5.716 | 167001.000 | 247804.000 | 331710.000 | 242879.000 | 90534.000 | 313865.000 | 0.839 |
| artificial punisher (clone) | 3.722 | 2.658 | 1.720 | 1.094 | 0.844 | 0.322 | 442067.000 | 876416.000 | 1209967.000 | 890903.000 | 279313.000 | 947541.000 | 3.400 |
| human managers | 4.755 | 2.973 | 1.672 | 0.978 | 0.692 | 0.267 | 809.000 | 1955.000 | 2614.000 | 1794.000 | 510.000 | 1232.000 | 4.488 |

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
