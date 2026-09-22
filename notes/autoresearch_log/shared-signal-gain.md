# Shared-signal response gain: does the contributor under-follow its own group's level?

Branch `auto/shared-signal-gain`, created from `origin/auto/sim-timeout-imputation` at `3fe1f44` (PR #197), which carries the most corrected simulation. A measurement, not an experiment: nothing is trained, no copula is recalibrated, no simulation is run, and no row is added to the evaluation suite. Every input is a file already committed on this branch or on `origin/auto/seed-spread-noise-floor`.

## 1. The question

Every teacher-forced verification in this campaign has measured a player's response to their **own** punishment, and it comes back essentially exact: on human trajectories the trunk gives +0.148 and +0.104 in the two low contribution bands against the human +0.140 and +0.104 (PR #183, PR #197 section 3).

Nobody had measured the response to a **shared** signal — how strongly a player follows their own group's level. It is a different weight on a different input, and it is the one that decides whether a group moves as a body rather than its members drifting apart. PR #191 built an off-manifold probe that shifts a group's recent contribution level and reads the predicted expectation, but it shifted *every* contribution in the game at once (own history included) and it compared three models against each other; it never computed a human benchmark. That benchmark is what this branch supplies.

The motivation is PR #202's result: a gap between the two groups carries forward about as well in the simulation as in real games (between membership changes, 0.8973 against 0.9098, inside the reseed floor), so the defect is not persistence; what is short by more than six reseed standard deviations is the **per-round innovation** of the group mean. Under-following of the group's own level was the standing candidate explanation.

## 2. Specification, and why

The estimand is `b` in

    c[i, t+1] = a * c[i, t] + b * m[-i, g(i), t] + k + e,

with `m[-i, g, t]` the **leave-one-out mean of the valid contributions of the other members of i's group at round t**. Every choice below is a decision the question does not fix; each is stated with its reason and each is checked against at least one alternative in section 3.5.

**Leave-one-out, not the group mean including self.** The group mean including i is `(c[i] + n*m[-i]) / (n+1)`, so a regression on it would load i's own stickiness onto the group coefficient — with group sizes down to two, more than a third of it. The LOO mean is the part of the group's level that is not the player.

**The own lag is in the regression.** `c[i, t]` and `m[-i, t]` are correlated (the within-round correlation in the human frame is what makes a group a group), so a specification without the own lag splits the credit arbitrarily and reports the sum. It is reported anyway as an alternative — human 0.7305, trunk 0.7296 — and it is uninformative about the split precisely because it is that sum.

**With an intercept.** Unlike PR #202's gap, which is symmetric about zero by construction, a contribution level is positive and the two regressors do not span the mean; the intercept is the level the process reverts to and dropping it would force the slopes to absorb it.

**Teacher-forced conditional expectations for the model, not simulated draws.** By iterated projections, both regressors are history-measurable, so the population projection of a realised `c` equals the projection of `E[c | history]`. The human regression and the teacher-forced regression therefore estimate the same coefficient on the same states, and the comparison is like-for-like row by row. The same argument says the regression on a **simulation's own realised trajectories** already is the teacher-forced coefficient at the simulation's states, which is why no forward pass over simulated states was needed.

**Paired bootstrap for the human-versus-model interval.** The human arm and the teacher-forced arm share the games, the rows and the regressors and differ only in the response, so games are resampled once and used in both. For the human-versus-simulation comparisons, which are different games, they are resampled independently.

**Group of one is dropped.** A player alone in a group has no peers, so the shared signal does not exist for them: 228 of the human's 8,796 transitions (2.6%) and ~420 of each simulated arm's 18,400. They are reported in `coverage.md` and excluded from every regression; specifications at 2, 3 and 4 peers are in section 3.5.

**Membership changes are a round-level flag, taken from the rosters.** A transition is a *change* transition when either group's member set differs between `t` and `t+1`, exactly PR #202's convention, and for the same reason: building the roster from valid contribution rows instead of from membership puts a human timeout into the change bucket and nothing of the simulation's. Results are reported pooled, on stable transitions and on change transitions separately.

**The peer set is the group at round `t`** — the group whose round-`t` outcome the player watched. The alternative, the group at `t+1` (the one the player is in when it acts, which is what the model's `agent_group` feature encodes), is reported in section 3.5; the two coincide on every transition that moves nobody, which is 80% of the sample.

**No attenuation correction in the headline.** The realised peer mean is what the player is shown, not a noisy proxy for a latent group propensity — PR #202's argument, unchanged. The correction matters less here than there anyway: the human arm and the teacher-forced arm use the *identical* regressor, so the attenuation factor is identical (0.5734 in both) and cancels from the difference.

**Thirds** are rounds 1-8 / 9-16 / 17-24 (`round_number` 0-7 / 8-15 / 16-23), the blocks the spread finding is stated in; a transition is assigned to the block containing `t`. **Clustering by game** throughout, 50 human games and 100 simulated episodes, with a 2000-draw game-cluster bootstrap on every difference.

**Arms.** `human` is `experiments/2group_8agent_50ep.csv` through `evaluation_suite.convert.load_human`, the canonical single-copy frame. `tf_trunk` is the bare stimulus-skip contributor `group_switching_contribution_50ep_vnode_stimulus_skip` teacher-forced over those same 50 games. `sim_noise_off` is PR #197's arm C (`23_2g8a_sim_timeout_rho0`), the serving-fixed frontier stack with the contribution copula off; `sim_noise_on` is its arm B; the two pre-serving-fix parents are carried as a guard.

**Noise floor, both routes.** PR #195's five reseed replicas of the same contributor plus the shipped sixth draw. Their six simulations give the run-to-run spread of the *realised* coefficient; their six bare artifacts, teacher-forced over the same human games, give the spread of the *teacher-forced* coefficient. The second floor is small by construction — it holds the states fixed and varies only the weights — so it is reported beside, never instead of, the human arm's own 50-game interval.

**Tooling.** `scripts/data_analysis/shared_signal_gain.py`: `analyse` locally on CPU, `tf` and `probe` on Raven because the trunk is a `torch_geometric` graph network and a forward pass cannot run on macOS. Tables and the figure: `plots/data_analysis/evaluation/shared_signal_gain/`. Remote isolation dir `~/repros/ai-runs/shared-signal-gain` (jobs 30399690 and 30399691, 58 s and 54 s on one node each, exit `0:0`); delete it when this PR closes.

## 3. Results (measured)

### 3.1 Controls

The teacher-forced tensor frame and the evaluation suite's canonical frame are the same 50 games: 9,320 valid contributions of 9,600 agent-rounds in both (the suite's 280 human timeouts), and the human coefficient computed from the tensor frame reproduces the one computed from `load_human` **to every digit** on all 43 quantities (`quantities.csv`, columns `human` and `tf_frame_human_y`). The regression code that scores the model is therefore the code that scores the human, on the same rows.

Coverage: 8,796 human transitions, 8,568 with at least one peer, 1,706 crossing a membership change, 548 of which move the player themselves; mean peer count 4.11 against 4.38 (`sim_noise_off`) and 4.46-4.48 across the six reseed arms. The teacher-forced mean of `E[c]` is 9.4676 against the human mean contribution 9.4572.

### 3.2 The coefficient, human against the teacher-forced trunk (headline)

95% intervals from game-clustered standard errors; the difference's interval and p-value from the 2000-draw paired game bootstrap.

| sample | human | trunk (teacher-forced) | difference | 95% CI | p | tf seed sd |
|---|---|---|---|---|---|---|
| **all transitions, pooled** | **0.2274** [0.1905, 0.2643] | **0.2364** [0.2121, 0.2607] | **-0.0090** | [-0.0305, +0.0119] | **0.332** | 0.0030 |
| all, 1-8 | 0.2199 [0.176, 0.264] | 0.2198 [0.188, 0.252] | +0.0001 | [-0.0296, +0.0259] | 0.944 | 0.0071 |
| all, 9-16 | 0.1985 [0.146, 0.251] | 0.2074 [0.175, 0.240] | -0.0089 | [-0.0422, +0.0251] | 0.575 | 0.0047 |
| all, 17-24 | 0.2217 [0.165, 0.279] | 0.2305 [0.193, 0.268] | -0.0087 | [-0.0421, +0.0300] | 0.594 | 0.0035 |
| **stable, pooled** | **0.2666** [0.2289, 0.3043] | **0.2559** [0.2314, 0.2804] | **+0.0107** | [-0.0129, +0.0348] | **0.409** | 0.0021 |
| stable, 1-8 | 0.3091 [0.255, 0.363] | 0.2500 [0.213, 0.287] | **+0.0591** | [+0.0208, +0.0958] | **0.003** | 0.0074 |
| stable, 9-16 | 0.2306 [0.177, 0.284] | 0.2276 [0.195, 0.261] | +0.0029 | [-0.0380, +0.0485] | 0.928 | 0.0040 |
| stable, 17-24 | 0.2368 [0.180, 0.294] | 0.2376 [0.202, 0.273] | -0.0008 | [-0.0389, +0.0413] | 0.944 | 0.0039 |
| **change, pooled** | **0.0802** [0.0069, 0.1534] | **0.1608** [0.1232, 0.1983] | **-0.0806** | [-0.1402, -0.0261] | **0.003** | 0.0076 |

The own lag beside it: human 0.7058 [0.674, 0.738], trunk 0.6919 [0.675, 0.709]. The **total gain** `a + b`, which is exactly the group mean's own lag-one coefficient because leave-one-out means average to the group mean, is **0.9332** (human) against **0.9283** (trunk) on all transitions, and 0.9747 against 0.9460 on stable ones. The share of that total carried by the shared signal is 0.2437 (human) against 0.2547 (trunk).

### 3.3 The simulation's own realised trajectories

Same regression, the simulation's states and its own draws; games resampled independently in the two arms.

| arm | all, pooled | 1-8 | 9-16 | 17-24 | stable, pooled | change, pooled | a (all) |
|---|---|---|---|---|---|---|---|
| human | **0.2274** | 0.2199 | 0.1985 | 0.2217 | **0.2666** | **0.0802** | 0.7058 |
| sim_noise_off | **0.2511** [0.2239, 0.2784] | 0.2535 | 0.2425 | 0.2272 | **0.2626** | **0.2063** | 0.6555 |
| sim_noise_on | 0.2673 [0.2382, 0.2965] | 0.2634 | 0.2320 | 0.2538 | 0.2812 | 0.2112 | 0.6754 |
| sim_noise_off_prefix | 0.2769 | 0.2476 | 0.2759 | 0.2728 | 0.2931 | 0.2137 | 0.6241 |
| sim_noise_on_prefix | 0.2806 | 0.2610 | 0.2623 | 0.2717 | 0.2896 | 0.2459 | 0.6611 |

Human minus `sim_noise_off`: all **-0.0237** [-0.0693, +0.0220] p = 0.319; stable **+0.0040** [-0.0414, +0.0517] p = 0.845; change **-0.1262** [-0.2159, -0.0426] p = 0.002.

### 3.4 The run-to-run floor

`seed_sd` is over PR #195's five reseed replicas plus the shipped sixth draw; the realised route uses their six simulations, the teacher-forced route their six bare artifacts over the human games. Full table: `noise_floor.md`.

| quantity | route | seed mean | seed sd | seed range | human | model | difference | in seed sd |
|---|---|---|---|---|---|---|---|---|
| b, all, pooled | realised | 0.2849 | **0.0128** | 0.2729-0.2992 | 0.2274 | 0.2511 | -0.0237 | 1.86 |
| b, stable, pooled | realised | 0.2995 | **0.0120** | 0.2810-0.3119 | 0.2666 | 0.2626 | +0.0040 | **0.34** |
| b, all, 17-24 | realised | 0.2592 | 0.0198 | 0.2394-0.2931 | 0.2217 | 0.2272 | -0.0055 | **0.28** |
| b, stable, 17-24 | realised | 0.2629 | 0.0212 | 0.2392-0.2967 | 0.2368 | 0.2337 | +0.0032 | **0.15** |
| b, change, pooled | realised | 0.2233 | 0.0308 | 0.1833-0.2538 | 0.0802 | 0.2063 | -0.1262 | **4.10** |
| a, all, pooled | realised | 0.6746 | 0.0110 | 0.6662-0.6934 | 0.7058 | 0.6555 | +0.0503 | 4.57 |
| a + b, stable | realised | 0.9720 | 0.0086 | 0.9567-0.9805 | 0.9747 | 0.9181 | +0.0566 | 6.55 |
| b, all, pooled | teacher forced | 0.2407 | **0.0030** | 0.2364-0.2441 | 0.2274 | 0.2364 | -0.0090 | 2.97 |
| b, stable, pooled | teacher forced | 0.2584 | **0.0021** | 0.2559-0.2607 | 0.2666 | 0.2559 | +0.0107 | 5.12 |
| b, stable, 17-24 | teacher forced | 0.2363 | 0.0039 | 0.2330-0.2428 | 0.2368 | 0.2376 | -0.0008 | **0.20** |
| b, change, pooled | teacher forced | 0.1729 | 0.0076 | 0.1608-0.1805 | 0.0802 | 0.1608 | -0.0806 | 10.62 |
| b, without the own lag | teacher forced | 0.7315 | 0.0010 | 0.7296-0.7325 | 0.7305 | 0.7296 | +0.0009 | **0.90** |
| b, other group's mean (d) | teacher forced | 0.0292 | 0.0012 | 0.0277-0.0308 | 0.0337 | 0.0277 | +0.0060 | 5.11 |

The teacher-forced floor is an order of magnitude tighter than the realised one (0.002-0.008 against 0.012-0.031), because it holds the states fixed and varies only the trained weights. It is therefore **not** the binding uncertainty on a human-to-model difference: the human arm's own 50-game interval is +-0.037 on the pooled coefficient and the paired difference's is +-0.021 to +-0.024, both an order of magnitude wider. Both are reported; the verdict in section 5 is read against the wider one.

### 3.5 Alternative specifications (pooled, all transitions)

| specification | human | trunk (tf) | difference | sim_noise_off |
|---|---|---|---|---|
| headline: own lag + LOO peer mean | 0.2274 | 0.2364 | -0.0090 | 0.2511 |
| without the own lag | 0.7305 | 0.7296 | +0.0009 | 0.6647 |
| + the other group's mean | 0.2180 | 0.2267 | -0.0087 | 0.2411 |
| + own and peer punishment | 0.2394 | 0.2371 | +0.0023 | 0.2536 |
| within player (episode x player FE) | 0.2474 | 0.2377 | +0.0097 | 0.1960 |
| within round (round FE) | 0.2297 | 0.2379 | -0.0082 | 0.2515 |
| within episode | 0.1258 | 0.1700 | -0.0442 | 0.1174 |
| peer set = the group at t+1 | 0.2622 | 0.2558 | +0.0064 | 0.2613 |
| at least 2 peers | 0.2408 | 0.2589 | -0.0181 | 0.2713 |
| at least 3 peers | 0.2494 | 0.2717 | -0.0223 | 0.2761 |
| at least 4 peers | 0.2508 | 0.2797 | -0.0289 | 0.2765 |
| stable and at least 2 peers | 0.2811 | 0.2782 | +0.0029 | 0.2850 |
| dc on (peer mean - own), pull form | 0.2774 | 0.2901 | -0.0127 | 0.3235 |
| errors-in-variables corrected (attenuation 0.5734 in both) | 0.3965 | 0.4122 | -0.0157 | 0.4710 |

The other group's mean carries a small, non-zero weight in both: human +0.0337, trunk +0.0277, `sim_noise_off` +0.0351 — one eighth of the own group's, i.e. both the people and the model treat "my group" and "the other group" as different inputs, which is what makes `b` a shared *group* signal rather than a room-level one.

The own lag under player fixed effects falls from 0.706 to 0.363 in the human arm and from 0.692 to 0.502 in the trunk, so roughly half the pooled `a` is persistent individual heterogeneity rather than round-to-round stickiness. The two are not comparable at face value: the FE estimator's Nickell bias scales with the residual variance, and the trunk's teacher-forced residual is 1.65 against the human 3.65.

### 3.6 The interventional probe (route b)

PR #191's probe, refitted to this question: instead of shifting every contribution in the game, each condition shifts exactly one channel of the target player's state and leaves the rest of the tensor alone, with `prev_contribution` rebuilt by the loader's own t-1 roll. `own` shifts only the target's own history, `peers` shifts only the players sharing the target's group at that round, `all` reproduces PR #191's everything-shifted total. Gains are per unit of the **realised** shift of the corresponding regressor, which is smaller than delta because the grid clips.

The three conditions partition the tensor, so their three responses must add to the everything-shifted one; they do, to 0.03-0.4 contribution units out of 4.6 (`additivity` in `probe.csv`), the residual growing with |delta| as the model's nonlinearity bites.

**All-rounds shift, trunk, valid rows with at least one peer (n = 7,336).** `peers` is the shared-signal gain, `all` is PR #191's everything-shifted gain normalised the same way (by the realised shift), `seed peers` is the mean and sd of `peers` over the trunk plus the five reseeds.

| delta | own | **peers** | other | all (normalised) | seed peers, mean (sd) |
|---|---|---|---|---|---|
| -6 | 0.8457 | **0.1588** | -0.0030 | 0.9543 | 0.1691 (0.0087) |
| -4 | 0.7938 | **0.1485** | -0.0010 | 0.9192 | 0.1687 (0.0134) |
| -2 | 0.8111 | **0.1436** | -0.0089 | 0.9520 | 0.1662 (0.0153) |
| +2 | 0.7601 | **0.1751** | -0.0109 | 0.9472 | 0.1901 (0.0085) |
| +4 | 0.7778 | **0.1832** | -0.0060 | 1.0143 | 0.2004 (0.0104) |
| +6 | 0.8009 | **0.1966** | -0.0080 | 1.0568 | 0.2030 (0.0073) |

Restricting to stable rounds moves nothing (peers 0.1439-0.1998); restricting to at least two peers moves nothing (0.1466-0.2041). The everything-shifted gain 0.919-1.057 reproduces PR #191's 0.957-1.023 for this same artifact, the residual difference being their narrower common set (own previous contribution in [6, 14], which drops the rows the grid clips).

**One-round shift, trunk, delta +-4, stimulus round r (1-indexed) read at r+1** (only that round's contributions are moved, so the recurrent state carries an unshifted history):

| stimulus round | own (-4 / +4) | **peers (-4 / +4)** | other |
|---|---|---|---|
| 3 | 0.436 / 0.410 | **0.136 / 0.175** | -0.012 |
| 7 | 0.372 / 0.376 | **0.106 / 0.138** | -0.005 |
| 11 | 0.367 / 0.318 | **0.085 / 0.115** | -0.005 |
| 15 | 0.293 / 0.307 | **0.056 / 0.086** | -0.003 |
| 19 | 0.315 / 0.324 | **0.062 / 0.079** | -0.004 |

The share of the model's total response carried by the shared channel, `peers / (own + peers)`, is **0.150-0.197** under the all-rounds shift and **0.159-0.300** under the one-round shift, against the regression's **0.2547** for the trunk and **0.2437** for the human (`share_all` in `quantities.csv`).

### 3.7 Where the spread deficit actually comes from, given these numbers

PR #202 measured the group mean's lag-one coefficient and its innovation and put 46.6% of the whole spread shortfall on the innovation against 6.9% on the gain. The decomposition here says what `b` can and cannot do about that. Averaging the per-player equation over a group, and using that leave-one-out means average to the group mean,

    m[g, t+1] = (a + b) * m[g, t] + k + mean_i(e[i, t+1]),

so `b` enters the group's dynamics **only through the sum `a + b`** and not at all through the innovation, whose sd is `sd(e) / sqrt(n)` under independent draws and larger only to the extent the draws are correlated within the round. A `b` deficit could therefore only have produced a *gain* deficit, which PR #202 measured and found inside the reseed floor — and there is no `b` deficit to begin with (section 3.2).

## 4. Notes (inference)

1. **The hypothesis is refuted. The contribution model does not under-respond to its group's level; it responds to it very slightly more than people do.** Pooled over all transitions the human weight on the leave-one-out group mean is 0.2274 and the teacher-forced trunk's is 0.2364, a difference of **-0.0090** with a paired game-bootstrap interval of [-0.031, +0.012] at p = 0.332. On stable transitions — the ones where the same people face each other again and the shared signal is cleanest — it is 0.2666 against 0.2559, **+0.0107** [-0.013, +0.035] at p = 0.409. The sign is not even consistent between the two samples, which is what a null looks like.

2. **In the last third, where the spread defect lives, the match is essentially exact.** Stable transitions, rounds 17-24: human 0.2368, trunk 0.2376, difference **-0.0008**, 0.20 teacher-forced seed sd and less than a fortieth of the human interval's half-width. On all transitions in that block it is -0.0087. There is no late-game under-following, which is the specific place the defect was supposed to be.

3. **The one cell that fires is the first third of stable transitions, and it is in the right direction but the wrong place.** Human 0.3091 against trunk 0.2500, +0.0591 [+0.021, +0.096], p = 0.003 — nine cells were tested, so it survives Bonferroni at 0.027, and it is worth stating against my own conclusion. It is robust to every specification tried on that cell: +0.056 adding the other group's mean, +0.059 adding punishment, +0.068 within player, +0.054 and +0.062 at two and three peers, and it is present in both halves of the block (+0.050 over rounds 1-4, +0.078 over rounds 5-8). It says people herd more strongly than the model in the opening rounds, when nobody has a history and the group's level is almost the only information there is. But the symptom it would have to explain is a group-mean spread that *stalls in the last third* after tracking the human through the first (PR #191 section 3.5: copula-off arms 3.39 / 4.33 / 4.17 against human 4.19 / 5.54 / 6.09), and the coefficient in that last block is matched to three decimal places. It is also the cell the closed loop partly repairs on its own: `sim_noise_off` gives 0.2860 there against the teacher-forced 0.2500 and the human 0.3091. An opening-rounds herding deficit is a real, small, separately interesting finding; it is not the campaign's defect.

4. **The two routes agree once the difference between a projection and a derivative is taken seriously, and neither shows a deficient shared channel.** The regression's `b` (0.236) is larger than the probe's peer gain (0.144-0.197 all-rounds, 0.056-0.175 one-round), and that is expected rather than a discrepancy: the regression is the *projection* of the response onto the lagged group mean, so it also collects the part of the association that runs through the group's persistent level, which the recurrent state carries; the probe is the *derivative*, holding everything else at its realised value. The human coefficient is a projection too, which is exactly why the projection is the like-for-like statistic and the probe is the corroboration. What the probe adds is the decomposition a regression cannot give: the shared channel's share of the model's total response is 0.150-0.197 (all-rounds) and 0.159-0.300 (one-round), bracketing the regression's 0.2547 for the model and 0.2437 for the human. The probe also reproduces PR #191's everything-shifted gain for this artifact (0.919-1.057 against their 0.957-1.023), which is the check that the two probes are the same instrument pointed at different channels.

   Two further things the probe shows that the regression cannot. First, the shared channel's *one-round* gain decays across the game (0.175 at round 3 to 0.079 at round 19 for delta +4) while the own channel decays much less (0.410 to 0.324) — later in the game the model leans on its recurrent state rather than on the immediate lag, and the shared input is the one it drops. Second, **the regression's weight on the other group's mean is confounded and the probe says so**: the interventional gain of the other group is -0.011 to -0.001 for the trunk, i.e. indistinguishable from zero, while the regression attributes +0.028 to it. The other group's level proxies the game's overall level, which the model tracks by other means; the same caution presumably applies to the human's +0.034, and neither can be read as a between-group comparison effect.

5. **Even a confirmed deficit could not have explained the symptom, and that is worth recording as the general lesson.** Section 3.7's arithmetic is exact: leave-one-out means average to the group mean, so `b` affects the group's trajectory only through the sum `a + b` — the gain PR #202 already measured — and contributes nothing at all to the per-round innovation, which is the 46.6% term. "A group's average only moves when its members move together" is true of the *innovation*, and the innovation is a within-round property of the joint distribution over a group's agents, not a weight on a lagged input. Any future hypothesis of the form "the model under-weights input X" should be checked against this identity before it is run: if X is a lagged, group-level regressor, it cannot move the innovation.

6. **What the measurement does find is the opposite deficit, at membership changes.** The trunk follows the group it *was* in at 0.1608 across a reshuffle where people follow it at 0.0802 — a difference of -0.0806 [-0.140, -0.026], p = 0.003, and 10.6 teacher-forced seed sd. The realised simulation is worse still (0.2063, -0.1262 against the human). Read forwards: a person who has just been reshuffled largely stops tracking their old group's level, and the artificial human keeps tracking it at two thirds of its stable weight. This is the individual-level mirror of PR #202's note 4, where the human *gap* survives a reshuffle better than the simulation's (0.5065 against 0.3349) — the two are consistent, because the human gap survives through *who moves* while the model's members keep following a level that is no longer theirs.

7. **The own lag is where the model and the people differ most on this panel, not the shared signal.** Human `a` 0.7058 against the trunk's 0.6919 and `sim_noise_off`'s 0.6555; the total `a + b` on stable transitions is 0.9747 (human), 0.9460 (trunk) and 0.9181 (simulation, 6.55 reseed sd below the human). The shortfall is therefore in the *total* persistence and it is carried by the own channel, and it grows from the teacher-forced conditional (-0.029) to the closed loop (-0.057). This is not the shared-signal story and it is not a gain the group-level measurement of PR #202 detected, because that one is attenuated by the sampling noise in the observed group mean (PR #202 section 3.6, attenuation 0.672 human / 0.505 simulation) while the individual-level decomposition is not.

8. **The simulated arms follow their groups slightly *more* than people do, not less, and the copula pushes it further that way.** `sim_noise_off` 0.2511 and `sim_noise_on` 0.2673 against the human 0.2274; both pre-serving-fix parents are higher again (0.2769, 0.2806). Every one of those differences is negative, i.e. the model over-follows, and the two copula-on arms are further from the human than their copula-off pairs on all three samples. Since the copula changes no weight in any model, this is the closed loop's states, not its conditional: correlated draws make the peer mean a better predictor of a player's own draw.

9. **The measurement convention is not doing the work here, unlike in PR #202.** The suite drops a human timeout and scores a simulated one at the env's imputed 9, which PR #202 found worth about 40% of its pooled difference. It cannot do that to the headline comparison, because the human arm and the teacher-forced arm are the *same rows of the same file*: the only convention that enters both is the LOO mean's treatment of a timeout, and it enters identically. It does enter the human-versus-simulation comparison of section 3.3, which is the weaker of the two routes for exactly that reason.

10. **No scored row.** The quantity is matched at the human value on the samples that isolate the mechanism, so there is nothing to score; and on the realised route the run-to-run floor is 0.0128 against a human-to-simulation difference of 0.0237 (1.86 seed sd) and 0.0040 on stable transitions (0.34 seed sd), which is below the floor before any scoring machinery is applied. Nothing is added to the evaluation suite on this branch.

## 5. Verdict

**The contribution model does not under-respond to its group's level. The hypothesis is dead.** On human trajectories the weight a player puts on the leave-one-out mean of their own group's previous contributions is **0.2274** [0.1905, 0.2643] and the trunk's teacher-forced weight on the same rows is **0.2364** [0.2121, 0.2607]; the paired game-bootstrap difference is **-0.0090** [-0.0305, +0.0119] at p = 0.332, i.e. the model follows its group very slightly *more* than people do. On the transitions that isolate the mechanism — stable rosters — it is 0.2666 against 0.2559, **+0.0107** [-0.0129, +0.0348] at p = 0.409, and the sign does not even agree with the pooled one. By thirds the difference is +0.0001 / -0.0089 / -0.0087 on all transitions and +0.0591 / +0.0029 / -0.0008 on stable ones: **in the last third, where the spread defect lives, the two coefficients agree to 0.0008**, 0.20 teacher-forced reseed sd and a fortieth of the human interval's half-width. The one cell that fires is the opening third of stable transitions (+0.0591, p = 0.003), which is real and robust across seven specifications but is neither the size nor the place of the symptom.

**The conclusion survives every alternative checked.** Dropping the own lag: 0.7305 against 0.7296. Adding the other group's mean: 0.2180 against 0.2267. Adding own and peer punishment: 0.2394 against 0.2371. Within player: 0.2474 against 0.2377. Within round: 0.2297 against 0.2379. Peer set defined by the group at t+1: 0.2622 against 0.2558. At two, three and four peers: 0.2408 / 0.2494 / 0.2508 against 0.2589 / 0.2717 / 0.2797. In the pull form (change on the distance to the group): 0.2774 against 0.2901. Errors-in-variables corrected: 0.3965 against 0.4122, with the attenuation factor identical at 0.5734 because the regressor is the same column in both arms. No specification puts the human more than 0.06 above the model and most put the model above the human.

**The two routes agree.** The interventional probe, shifting only a target's same-group peers and leaving its own history alone, gives a shared-signal gain of 0.144-0.197 across deltas against an own gain of 0.760-0.846, i.e. a shared share of 0.150-0.197 (0.159-0.300 under a one-round shift) against the regression's 0.2547 for the model and 0.2437 for the human. The difference in level between the two routes is the difference between a projection and a derivative, not a disagreement, and the probe reproduces PR #191's everything-shifted gain for this artifact (0.919-1.057 against 0.957-1.023). The probe also shows the regression's +0.028 weight on the *other* group is confounded: the interventional gain there is -0.011 to -0.001.

**In the closed loop the following does not weaken; it strengthens slightly.** `sim_noise_off` gives 0.2511 [0.2239, 0.2784] against the human 0.2274, and 0.2626 against 0.2666 on stable transitions (+0.0040, **0.34 reseed sd**, the smallest difference in the whole file). The reseed floor over PR #195's six arms is 0.0128 pooled and 0.0120 on stable transitions. Whatever the closed loop degrades, it is not this. The copula moves it further from the human, not closer (0.2673), and so do both pre-serving-fix parents (0.2769, 0.2806).

**Two things the measurement found that were not asked for.** First, at membership changes the model keeps following the group it *was* in at 0.1608 where people follow it at 0.0802 (-0.0806 [-0.140, -0.026], p = 0.003), and the closed loop makes it worse (0.2063). Second, the deficit that does exist on this panel is in the *own* channel and hence in the total: `a + b` on stable transitions is 0.9747 (human), 0.9460 (teacher-forced trunk) and 0.9181 (simulation, 6.55 reseed sd below the human), a gap that grows from the conditional to the loop.

**No scored row.** The quantity is matched at the human value where it matters, the realised route's human-to-simulation difference on stable transitions is 0.34 reseed sd, and nothing is added to the evaluation suite on this branch.

## 6. What this leaves for a successor

- **Do not re-run this measurement, and do not propose another "the model under-weights input X" hypothesis without checking the identity in section 3.7 first.** `scripts/data_analysis/shared_signal_gain.py analyse` regenerates every local number from committed inputs in about two minutes; `tf` and `probe` need Raven (the trunk is a `torch_geometric` graph network) and together take under two minutes of compute on one node. The identity is the reusable part: averaging the per-player equation over a group shows that any weight on a *lagged, group-level* regressor enters the group's trajectory only through the total gain `a + b` and contributes nothing at all to the innovation, which is the term PR #202 put 46.6% of the spread shortfall on. Three of this campaign's hypotheses have now died against a version of that observation.

- **The live target is unchanged and is now the only one standing: the within-round joint distribution over a group's agents.** PR #202 named it (the group-level shock is 0.5231 of the arm's own individual sd against the human 0.6018), PR #191's note 5 named it, and the campaign's section 6 has named it since PR #140. Both group-level *weights* are now measured and matched — the gain across rounds (PR #202) and the weight on the shared input (here) — so the remaining object is not a coefficient on any lagged feature at all. It is the correlation between the draws of agents who share a group at one round, and it is trainable without a trajectory-level objective.

- **The total persistence `a + b` is short and the own channel carries it; that is a cheap follow-up.** Stable transitions: 0.9747 (human), 0.9460 (teacher-forced), 0.9181 (simulation), the last 6.55 reseed sd below the human, with `a` alone 0.7058 / 0.6919 / 0.6555. PR #202's group-level gain could not see this because the observed group mean attenuates it (their attenuation factors 0.672 and 0.505); the individual-level decomposition is attenuation-free in `a` because the own lag is measured exactly. Note before acting on it that roughly half of the pooled `a` is persistent individual heterogeneity (under player fixed effects the human's falls 0.706 -> 0.363), and that the fixed-effects comparison between the arms is not clean because Nickell bias scales with the residual variance and the teacher-forced residual is 1.65 against the human 3.65. A candidate would have to raise own-channel persistence without raising it inside a player, which is a different and harder target than it looks.

- **The reshuffle remains the switch-slot hypothesis, and this branch adds the individual-level half of the evidence.** PR #202 measured it at the group level (the human gap survives a reshuffle at 0.5065 against the simulation's 0.3349) and left it as an untouched declaration. Here the same thing appears per player: a person who has just been reshuffled largely stops tracking their old group's level (b falls from 0.267 to 0.080) and the artificial human keeps tracking it at two thirds of its stable weight (0.256 to 0.161), with the closed loop worse again (0.263 to 0.206). The two are the same defect seen from two sides. A successor declaring it should measure the correlation between a switcher's own contribution and the gap between their old and new group before building anything; the human sample is 548 ego-move transitions and 1,706 roster-change transitions, so a bootstrap and not a new simulation is the right instrument.

- **The opening-third herding gap is a small, separate, contribution-slot lead.** +0.0591 [+0.021, +0.096] on stable transitions in rounds 1-8, robust across seven specifications and present in both halves of the block, with the model at 0.2500 against the human 0.3091. It is worth about a tenth of the pooled coefficient and it is in the block where the group-mean spread already matches (PR #191: 3.39 against the human 4.19 is the *closest* of the three blocks in ratio terms), so it is unlikely to be worth a band on any row; but if a contributor candidate is built for another reason, this cell is a free check of whether it herds like a person when nobody has a history yet.

- **The regression weight on the other group is not a between-group comparison effect.** Both the human (+0.034) and the model (+0.028) load a small positive coefficient on the other group's mean, but the interventional gain of that channel in the model is -0.011 to -0.001. The regressor proxies the game's overall level. Anyone building a feature out of "what the other group is doing" should measure it interventionally first.
