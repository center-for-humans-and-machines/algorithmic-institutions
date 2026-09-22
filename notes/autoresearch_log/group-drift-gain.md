# Group drift gain: is the missing late divergence a gain problem?

Branch `auto/group-drift-gain`, created from `origin/auto/sim-timeout-imputation` at `3fe1f44` (PR #197), which carries the most corrected simulation. A measurement, not an experiment: nothing is trained, no copula is recalibrated, no simulation is run, and no row is added to the evaluation suite. Everything below comes from files already committed on this branch or on `origin/auto/seed-spread-noise-floor`.

## 1. The question

Real groups keep pulling apart as a game runs: the sd of the group mean contribution rises across thirds of the game, 4.19 then 5.54 then 6.09 (PR #186 §3.5, PR #191 §3.5). Every simulated arm with the shared-noise machinery off starts alike and then stalls or reverses in the last third. Four candidate causes have been eliminated — the emission head and the off-manifold gain (PR #191), the shared-noise strength and its persistence (PR #186), model uncertainty (PR #188) and the routing of punishment to the output (PR #184). The untested hypothesis is a **gain**: that in real games a gap between the two groups feeds itself forward and grows, and that in the simulation it decays.

With two groups there is effectively one number per round, the gap between their mean contributions, so the hypothesis reduces to one coefficient: regress `gap[t+1]` on `gap[t]`. Above one, gaps grow; below one, they shrink. This branch estimates it for the human games, for the simulation with the shared-noise machinery off (the honest arm), and for the simulation with it on.

## 2. Specification, and why

Every choice below is a decision the question does not fix. Each is stated with its reason and each is checked against at least one alternative in §3.6.

**Raw gap, not normalised.** A lag-one slope is already scale free: multiplying every contribution by a constant leaves it unchanged. A *time-varying* normaliser would be worse than useless, because it folds the normaliser's own trend into the coefficient — and the individual-level spread trends over the game in both sources, which is precisely the trend under examination. The normalised version is reported as an alternative and moves both arms the same way.

**Unweighted, one row per realised transition.** Group sizes run 1 to 8, so a gap between a one-player group and a seven-player group is a noisy quantity, and the usual reflex is to weight by precision or to correct for attenuation. Both are wrong here, for the same reason: the realised group mean is *what the agents are shown*. Both the real player and the artificial human condition on the realised mean of their group, not on a latent group propensity, so its sampling noise is part of the stimulus and not measurement error standing between the analyst and a truth. The population of realised transitions is the population of interest. Precision weighting, size restrictions and an explicit errors-in-variables correction are all reported as alternatives.

**No intercept.** The group labels are arbitrary — the human CSVs carry every game twice with the labels mirrored and the suite keeps one copy — so the gap is symmetric about zero by construction and its mean must be zero. Fitting an intercept anyway returns -0.134 (human) and +0.124 (simulation) on a variable with sd ~6, and moves the slope by less than 0.002 in both arms.

**No episode fixed effects, and no other conditioning.** A persistent per-episode group offset is exactly the object under test — it is what the copula supplies and what PR #186 found to be worth twice any emission head — so within-episode demeaning would absorb the mechanism and, with 23 transitions per game, add Nickell bias on top. Conditioning on punishment, on group size or on round would turn the coefficient into a partial derivative and forfeit the one thing that makes it readable: that it is the *total* forward transmission of a gap, the thing the phrase "feeds itself forward" means.

**Lag one, with the innovation reported beside it.** This is the choice that most needed checking, and the check changed how the result reads. Between-group spread does not grow because beta exceeds one; it grows as `sigma / sqrt(1 - beta^2)` towards a steady state, so a model can fail on spread through the gain, through the innovation sd, or through both. Reporting beta alone would have been half the measurement. An AR(2) fit is reported as a form check.

**Clustering by game**, since agents within a game share a trajectory; 50 human games, 100 simulated episodes. Analytic clustered intervals throughout, with a 2000-draw episode-cluster bootstrap on every pooled coefficient and on every human-minus-simulation difference.

**Thirds** are rounds 1-8 / 9-16 / 17-24 (`round_number` 0-7 / 8-15 / 16-23), the blocks PR #186 and PR #191 state the finding in; a transition is assigned to the block containing `t`.

**Membership-change transitions** are identified from the realised rosters, not from the round number: a transition is a switch transition when the set of members of either group differs between `t` and `t+1`. This matters. Building the roster from *valid contribution* rows instead of from membership puts a human timeout into the switch bucket and nothing of the simulation's, and doing so inflated the human switch-round coefficient from 0.51 to 0.69 in the first pass of this analysis. The roster is membership; the group mean is over valid rows.

**Arms.** `human` is `experiments/2group_8agent_50ep.csv` through `evaluation_suite.convert.load_human`, so the flip augmentation and the human timeouts are handled exactly as everywhere else. `sim_noise_off` is PR #197's arm C, `plots/simulation/23_2g8a_sim_timeout_rho0/`, the serving-fixed frontier stack with the contribution copula off. `sim_noise_on` is its arm B, `..._gnn_switch_simtimeout/`, the same stack with the copula on. The two pre-serving-fix parents (arms D and A, `23_2g8a_sim_timeout_rho0_base/` and `..._gnn_switch_timeout/`) are carried through every table as a guard that nothing here is an artefact of PR #197's fix.

**Noise floor.** PR #195's five reseed replicas of the accepted contributor plus the shipped sixth draw, `23_2g8a_contr_stimulus_skip_seed{1..5}_self_gnncopar1_contr_gnn_switch_ceiling/per_round.parquet` and the shipped `..._ceiling/`, read from `origin/auto/seed-spread-noise-floor` into a scratch directory and passed to the script with `--seed-dir`. Same architecture, same config, same data, same simulation seed: the spread over the six is the run-to-run floor of every quantity in this file.

**Tooling.** `scripts/data_analysis/group_drift_gain.py`, local CPU, no model forward pass and no cluster time. Tables and the figure: `plots/data_analysis/evaluation/group_drift_gain/`.

## 3. Results (measured)

### 3.1 Controls

The human group-mean spread by block reproduces the number this whole question rests on: **4.2411 / 5.5549 / 6.0696** against PR #186's quoted 4.19 / 5.54 / 6.09 (0.3% on the last two blocks, 1.2% on the first, the cell sets differing only in how rounds with one empty group are handled). The human individual contribution sd is **6.3179** against their 6.32.

Roster changes occur at decision rounds and nowhere else: `switch_outside_decision` is **0** in all eleven arms. The human frame has 280 NaN contributions in 9,600 agent-rounds (2.92%); every simulated frame has none, because the env records the imputed 9 and `load_sim` has no validity column (PR #197 §4 note 2). 1,039 of 1,200 human game-rounds have both groups non-empty, giving 961 transitions of which 169 cross a membership change; the simulated arms give ~1,800 transitions of which ~290 do.

### 3.2 The coefficient, all transitions

95% intervals from game-clustered standard errors. `sigma` is the innovation sd, `stat.sd` the steady state `sigma / sqrt(1 - beta^2)` it implies.

| arm | pooled | 1-8 | 9-16 | 17-24 | sigma (pooled) | stat.sd |
|---|---|---|---|---|---|---|
| human | **0.8356** [0.7774, 0.8939] | 0.7567 [0.651, 0.862] | 0.8227 [0.724, 0.922] | 0.9036 [0.829, 0.978] | 4.2149 | 7.6734 |
| sim_noise_off | **0.7905** [0.7508, 0.8302] | 0.7824 [0.725, 0.840] | 0.7506 [0.691, 0.810] | 0.8509 [0.785, 0.917] | 3.6459 | 5.9527 |
| sim_noise_on | **0.8168** [0.7704, 0.8632] | 0.7688 [0.711, 0.827] | 0.8262 [0.765, 0.888] | 0.8650 [0.786, 0.945] | 3.7557 | 6.5099 |
| sim_noise_off_prefix | 0.7609 [0.717, 0.805] | 0.7346 | 0.7418 | 0.8333 | 3.4499 | 5.3166 |
| sim_noise_on_prefix | 0.8161 [0.769, 0.863] | 0.7541 | 0.8171 | 0.8807 | 3.6357 | 6.2912 |

### 3.3 Between membership changes (the transitions that isolate the mechanism)

| arm | pooled | 1-8 | 9-16 | 17-24 | sigma (pooled) | stat.sd |
|---|---|---|---|---|---|---|
| human | **0.9098** [0.8595, 0.9601] | 0.8811 [0.762, 1.000] | 0.9169 [0.834, 1.000] | 0.9192 [0.850, 0.988] | 3.8019 | 9.1590 |
| sim_noise_off | **0.8973** [0.8671, 0.9275] | 0.9076 [0.852, 0.963] | 0.8795 [0.832, 0.927] | 0.9071 [0.854, 0.961] | 3.0413 | 6.8891 |
| sim_noise_on | **0.9100** [0.8775, 0.9426] | 0.9138 [0.862, 0.965] | 0.9245 [0.874, 0.974] | 0.8887 [0.814, 0.963] | 3.1503 | 7.5987 |
| sim_noise_off_prefix | 0.8804 [0.847, 0.914] | 0.8959 | 0.8765 | 0.8685 | 2.8807 | 6.0747 |
| sim_noise_on_prefix | 0.9209 [0.891, 0.950] | 0.9115 | 0.9006 | 0.9493 | 3.0661 | 7.8639 |

### 3.4 At membership changes

| arm | pooled | 1-8 | 9-16 | 17-24 | sigma (pooled) |
|---|---|---|---|---|---|
| human | **0.5065** [0.3331, 0.6798] | 0.4306 [0.201, 0.660] | 0.5131 [0.288, 0.738] | 0.6935 [0.360, 1.027] | 5.1824 |
| sim_noise_off | **0.3349** [0.2146, 0.4553] | 0.4014 [0.247, 0.556] | 0.2893 [0.098, 0.480] | 0.2658 [-0.028, 0.559] | 4.9750 |
| sim_noise_on | **0.3828** [0.2482, 0.5174] | 0.2849 [0.108, 0.462] | 0.4190 [0.259, 0.579] | 0.6087 [0.323, 0.895] | 5.2055 |
| sim_noise_off_prefix | 0.2909 [0.163, 0.419] | 0.2586 | 0.2787 | 0.4764 | 4.6356 |
| sim_noise_on_prefix | 0.3706 [0.235, 0.506] | 0.2945 | 0.4684 | 0.3726 | 4.8635 |

Human n per cell 169 / 64 / 71 / 34; `sim_noise_off` 297 / 125 / 120 / 52.

### 3.5 Human minus simulation, with games resampled whole in both arms

2000 draws, episodes resampled independently in each arm, no retraining variation included (that is §3.7).

| sim arm | sample | human | sim | difference | 95% CI | p (two sided) |
|---|---|---|---|---|---|---|
| sim_noise_off | all | 0.8356 | 0.7905 | +0.0451 | [-0.0317, +0.1088] | 0.246 |
| sim_noise_off | between changes | 0.9098 | 0.8973 | **+0.0125** | [-0.0509, +0.0660] | **0.690** |
| sim_noise_off | at changes | 0.5065 | 0.3349 | +0.1715 | [-0.0311, +0.3857] | 0.089 |
| sim_noise_on | all | 0.8356 | 0.8168 | +0.0188 | [-0.0589, +0.0895] | 0.627 |
| sim_noise_on | between changes | 0.9098 | 0.9100 | **-0.0002** | [-0.0604, +0.0581] | **0.985** |
| sim_noise_on | at changes | 0.5065 | 0.3828 | +0.1237 | [-0.0828, +0.3391] | 0.271 |

### 3.6 Alternative specifications (pooled, all transitions)

| specification | human | sim_noise_off | sim_noise_on | difference h - off |
|---|---|---|---|---|
| headline: raw gap, unweighted, no intercept | 0.8356 | 0.7905 | 0.8168 | +0.0451 |
| with intercept (const -0.134 / +0.124 / +0.070) | 0.8347 | 0.7888 | 0.8164 | +0.0459 |
| precision weighted, w = n0 n1 / (n0 + n1) | 0.8530 | 0.7847 | 0.8123 | +0.0683 |
| both groups >= 2 members | 0.8812 | 0.7750 | 0.8163 | +0.1062 |
| both groups >= 3 members | 0.8704 | 0.7806 | 0.7808 | +0.0898 |
| gap normalised by the round's individual sd | 0.7917 | 0.7587 | 0.7753 | +0.0330 |
| AR(2), sum of the two lags | 0.8628 | 0.8267 | 0.8631 | +0.0361 |
| AR(2), lag one / lag two | 0.7340 / +0.129 | 0.6722 / +0.155 | 0.6670 / +0.196 | |
| errors-in-variables corrected (attenuation 0.672 / 0.505 / 0.574) | 1.2427 | 1.5666 | 1.4229 | **-0.3239** |

The same cuts restricted to transitions between membership changes: human 0.9319 / 0.9270 / 0.9805 at min group size 2 / 3 / 4 against `sim_noise_off` 0.8864 / 0.8967 / 0.8809 and `sim_noise_on` 0.9268 / 0.9285 / 0.9350. The human interval at min size 4 is [0.888, 1.074] on 180 transitions and the simulation's is [0.812, 0.949] on 435; they overlap. Group size distributions are close across arms (share of transitions with min group size 1 / 2 / 3 / 4: human 0.234 / 0.275 / 0.264 / 0.227, `sim_noise_off` 0.221 / 0.270 / 0.214 / 0.295).

**The measurement convention costs about 40% of the pooled difference.** The suite NaNs a human timeout and scores a simulated one at the env's imputed 9, on 2.24% of simulated agent-rounds (PR #197 §1). Pushing the same share of human contributions to 9 (20 draws) moves the human coefficient from 0.8356 to **0.8176** (sd 0.0056) on all transitions and from 0.9098 to **0.8906** (sd 0.0056) between changes. On that footing the pooled difference is +0.027 and the between-changes difference is **-0.007**, i.e. it changes sign.

### 3.7 The run-to-run floor over PR #195's six reseeds

`seed_sd` is over the five replicas plus the shipped draw; `in_seed_sd` is |human - `sim_noise_off`| in those units. Full table: `noise_floor.md`.

| quantity | seed mean | seed sd | seed range | human | sim_noise_off | human - off | in seed sd |
|---|---|---|---|---|---|---|---|
| beta, all, pooled | 0.8008 | **0.0174** | 0.7817-0.8276 | 0.8356 | 0.7905 | +0.0451 | 2.59 |
| beta, all, 17-24 | 0.8380 | **0.0326** | 0.8036-0.8814 | 0.9036 | 0.8509 | +0.0527 | 1.62 |
| beta, between changes, pooled | 0.9180 | **0.0141** | 0.9018-0.9346 | 0.9098 | 0.8973 | +0.0125 | **0.89** |
| beta, between changes, 1-8 | 0.9064 | 0.0314 | 0.8657-0.9463 | 0.8811 | 0.9076 | -0.0264 | 0.84 |
| beta, between changes, 9-16 | 0.9378 | 0.0146 | 0.9256-0.9622 | 0.9169 | 0.8795 | +0.0374 | 2.56 |
| beta, between changes, 17-24 | 0.9064 | 0.0226 | 0.8704-0.9278 | 0.9192 | 0.9071 | +0.0121 | **0.54** |
| beta, at changes, pooled | 0.3073 | **0.0570** | 0.2497-0.4146 | 0.5065 | 0.3349 | +0.1715 | 3.01 |
| beta, at changes, 17-24 | 0.3198 | 0.1049 | 0.1884-0.4704 | 0.6935 | 0.2658 | +0.4277 | 4.08 |
| beta, AR(2) sum | 0.8356 | 0.0163 | 0.8189-0.8644 | 0.8628 | 0.8267 | +0.0361 | 2.21 |
| beta, all, min group >= 2 | 0.8054 | 0.0160 | 0.7920-0.8350 | 0.8812 | 0.7750 | +0.1062 | 6.62 |
| sigma, between changes | 2.8688 | 0.1234 | 2.7403-3.0273 | 3.8019 | 3.0413 | +0.7606 | **6.16** |
| sigma / individual sd, between changes | 0.4639 | 0.0207 | 0.4398-0.4989 | 0.6018 | 0.5231 | +0.0787 | **3.79** |
| implied steady-state sd, between changes | 7.2834 | 0.3580 | 6.7075-7.7041 | 9.1590 | 6.8891 | +2.2699 | 6.34 |
| sd of the gap, 17-24 | 4.8578 | 0.3383 | 4.4983-5.4055 | 7.3465 | 5.4796 | +1.8670 | **5.52** |
| individual contribution sd | 6.1847 | 0.1008 | 6.0680-6.3247 | 6.3179 | 5.8138 | +0.5042 | 5.00 |

### 3.8 What actually produces the missing spread

Three of every four transitions leave the rosters alone and the fourth reshuffles them, so the gap process has four parameters: a gain and an innovation sd on each kind of step. Solving the four-step cycle for its fixed point gives the steady-state spread the process settles at. Swapping one parameter at a time from `sim_noise_off` to the human closes this much of the 5.807 -> 7.204 shortfall:

| swapped to the human value | cycle sd | share of the shortfall closed |
|---|---|---|
| none (`sim_noise_off` as is) | 5.8073 | 0% |
| gain between changes, 0.8973 -> 0.9098 | 5.9029 | **6.9%** |
| gain at changes, 0.3349 -> 0.5065 | 6.1978 | 28.0% |
| innovation sd between changes, 3.041 -> 3.802 | 6.4575 | **46.6%** |
| innovation sd at changes, 4.975 -> 5.182 | 5.9488 | 10.1% |
| both gains | 6.3262 | 37.2% |
| both innovation sds | 6.5851 | 55.7% |
| human | 7.2035 | 100% |

For `sim_noise_on` the same table reads 6.253 -> 7.204 with the gain between changes worth **-0.2%**, the gain at changes 35.7%, the innovation sd between changes 60.0% and the innovation sd at changes -1.7%.

### 3.9 The copula's effect on the coefficient

Two independent (serving fix on / off) pairs, each differing only in whether the contribution copula is on.

| quantity | fix, off -> on | parent, off -> on | seed sd |
|---|---|---|---|
| beta, all, pooled | 0.7905 -> 0.8168 (+0.0263) | 0.7609 -> 0.8161 (+0.0552) | 0.0174 |
| beta, between changes, pooled | 0.8973 -> 0.9100 (+0.0127) | 0.8804 -> 0.9209 (+0.0405) | 0.0141 |
| beta, at changes, pooled | 0.3349 -> 0.3828 (+0.0479) | 0.2909 -> 0.3706 (+0.0797) | 0.0570 |

The direction is the same in both pairs on all three samples. With the copula on, the between-changes coefficient lands on **0.9100** against the human **0.9098**.

## 4. Notes (inference)

1. **The drift failure is not a gain problem.** On the transitions that isolate the mechanism — between membership changes, where the same people face each other again — the human coefficient is 0.9098 and the honest simulated arm's is 0.8973. The difference is +0.0125, its game-clustered bootstrap interval is [-0.051, +0.066] at p = 0.69, and it is **0.89 reseed sd**, i.e. smaller than the spread produced by retraining the same model with a different seed. By thirds it is -0.026, +0.037, +0.012: there is no late-third gain deficit at all, which is the specific place the defect was supposed to live. The hypothesis is not merely unproven; the measurement is a null with an interval tight enough to exclude anything like the size of the symptom.

2. **The pooled difference of +0.045 is real but it decomposes into two things, neither of which is a gain.** About 40% of it is the measurement convention: the suite drops a human timeout and scores a simulated one at 9, and putting the human data on the same footing takes 0.0451 to 0.027 and takes the between-changes difference to -0.007 (§3.6). The rest is the membership-change transitions carrying a 17% weight in the pooled sample at a coefficient difference of 0.17. Nothing is left over for the ordinary round-to-round gain.

3. **What is short is the innovation, not the gain, and it is short for a group-level reason.** Between membership changes the innovation sd is 3.80 in the humans and 3.04 in the honest arm, 6.16 reseed sd apart. That is not just simulated individuals being less variable: normalised by each arm's own individual contribution sd it is 0.6018 against 0.5231, still 3.79 reseed sd apart. The cycle decomposition (§3.8) puts 46.6% of the whole spread shortfall on that one number against 6.9% on the ordinary gain. The reading is the one PR #186 and the campaign's §6 already point at: the per-round shock that moves a *whole group* together is too small, because per-agent sampling is too close to independent. It is a single-round property, and it is a property of the joint distribution over a group's agents rather than of any agent's conditional mean.

4. **The second lever is the reshuffle, and it is the one place a gain reading survives.** At membership changes the human coefficient is 0.5065 and the honest arm's 0.3349, a difference of 0.1715 at 3.01 reseed sd, p = 0.089, closing 28% of the spread shortfall. Read forwards: in the human games a gap survives a reshuffle better than a random reallocation would leave it, and increasingly so as the game runs (0.431, 0.513, 0.694 across thirds); in the simulation it survives less and decreasingly so (0.401, 0.289, 0.266). The obvious mechanism is assortative switching — who moves where is correlated with how much they give — and it is a property of the *switch* model and its coupling to contributions, not of the contribution model's gain. The evidence is suggestive rather than settled: the human sample is 169 transitions, the by-thirds cells are 64 / 71 / 34, and the pooled difference's interval covers zero.

5. **The one cut where the difference is larger is worth stating against my own conclusion.** Restricting to transitions where both groups have at least two members raises the human pooled coefficient to 0.8812 while the simulation's falls to 0.7750, a difference of 0.106 at 6.6 reseed sd; between changes at min size 4 it is 0.9805 against 0.8809. The human coefficient rises with the size cut (0.910, 0.932, 0.927, 0.981) and the honest arm's does not (0.897, 0.886, 0.897, 0.881). Two things keep this from overturning Note 1. The intervals overlap heavily at every cut (at min size 4, [0.888, 1.074] against [0.812, 0.949] on 180 and 435 transitions), and the copula-on arm, which differs from the copula-off arm in no part of any model, tracks the human at every cut (0.9268 / 0.9285 / 0.9350) — so the cut is separating arms by their between-agent correlation, which is Note 3, rather than by a gain.

6. **The errors-in-variables alternative reverses the sign of the comparison, and its message is the ordering rather than its numbers.** Correcting for the sampling noise in the observed group mean gives 1.24 for the humans and 1.57 for the simulation, because the simulation needs the larger correction (attenuation factor 0.505 against 0.672) — its within-group variance is larger relative to its gap variance, which is Note 3 again. Both corrected values exceed one, which cannot be right for a process whose spread is bounded, and they should not be: the method-of-moments factor uses the cross-sectional within-group variance, which contains persistent individual heterogeneity and therefore overstates how far a group mean wanders from its own level. They are an upper bound. The usable content is that **on any latent-gap reading the simulated gain is at or above the human one**, which points the same way as the headline.

7. **The shared-noise machinery does inflate this coefficient, and it inflates it the way the symptom would be inflated by cheating.** The copula raises the coefficient in both independent (fix on / fix off) pairs on all three samples, by +0.013 to +0.080, and on the between-changes measure it takes the honest arm's 0.8973 to 0.9100 against the human's 0.9098 — an exact landing produced by a per-episode, per-group latent that no part of any model learned. It raises the at-changes coefficient too (+0.048 and +0.080), which is the sharpest form of the point: the copula's latent is attached to the *group label*, so it survives a reshuffle perfectly, whereas the human gap survives a reshuffle because of *who moves*. The machinery therefore reproduces the mechanism's signature as well as its symptom, and a scored row on this coefficient would be satisfiable by turning the copula up. That is a reason against the row independently of the noise argument.

8. **The serving fix is not what any of this is about.** Every contrast holds in the pre-fix pair as well (§3.2-3.4): between changes, parent-off 0.8804 and parent-on 0.9209; at changes, 0.2909 and 0.3706. The fix moves the coefficients by 0.01-0.04, the same order as the copula, and does not change any sign or ordering.

9. **A scored row on this coefficient would be noise-dominated from birth.** PR #195's floor is 0.138 for a typical scored row, with ten of 22 rows ungateable on a single run. The floor for the quantity measured here is 0.0141 on the between-changes coefficient, and the human-to-simulation difference it would have to resolve is 0.0125 — a signal-to-noise of **0.89**, below one before any scoring machinery is applied, and that is on the honest arm, on the specification that isolates the mechanism, at full sample. The pooled variant reaches 2.59 reseed sd, but 40% of that is the timeout convention (§3.6) and the rest is the reshuffle mixture, so the row would be measuring two other things under a misleading name. The row that *would* clear the floor is the innovation (6.16 reseed sd, 3.79 after normalising) or the steady-state spread (6.34) — and the suite already has CG, whose ratio of group-mean spread to individual spread is the same quantity in reduced form. **Recommend against the row.**

## 5. Verdict

**The simulation's failure to keep groups drifting apart is not a gain problem.** Between membership changes the lag-one gap coefficient is **0.9098** [0.8595, 0.9601] in the human games and **0.8973** [0.8671, 0.9275] in the honest copula-off simulation; the difference is +0.0125 with a game-clustered bootstrap interval of [-0.051, +0.066] (p = 0.690) and is **0.89 reseed sd**, below the run-to-run floor. By thirds the difference is -0.026 / +0.037 / +0.012 — in the last third, where the defect lives, it is 0.012 and 0.54 reseed sd. The conclusion survives every alternative specification checked: an intercept (slope moves < 0.002), precision weighting, an AR(2) form (sums 0.863 against 0.827), normalising the gap by the round's individual spread (0.792 against 0.759) and an errors-in-variables correction, which reverses the sign of the comparison in the simulation's favour.

**The cause is elsewhere, and the measurement names it.** The per-round innovation of the gap is 3.80 in the humans and 3.04 in the honest arm (6.16 reseed sd apart, and 3.79 apart after normalising by each arm's own individual spread), and a four-parameter cycle model puts **46.6%** of the whole spread shortfall on that one number against **6.9%** on the ordinary gain. The second lever, worth **28.0%**, is what happens at a reshuffle: the human coefficient there is 0.5065 against 0.3349 and rises across thirds (0.431 / 0.513 / 0.694) where the simulation's falls (0.401 / 0.289 / 0.266), which is a property of the switch model's coupling to contributions rather than of the contributor's gain.

**The shared-noise machinery inflates the coefficient, on both independent pairs and on every sample** (+0.013 to +0.080), landing the between-changes value on 0.9100 against the human 0.9098 with no part of any model changed. It fakes the mechanism as well as the symptom, and it fakes it in a legible way: its latent is attached to the group label, so it survives a reshuffle perfectly, where the human gap survives a reshuffle because of who moves.

**No scored row.** The floor for this coefficient over PR #195's six reseeds is 0.0141 and the difference it would resolve is 0.0125 — signal-to-noise 0.89, and the machinery that does not improve any model raises it past the human value. Nothing is added to the evaluation suite on this branch.

## 6. What this leaves for a successor

- **Do not re-run the gain measurement.** `scripts/data_analysis/group_drift_gain.py` regenerates every number here from committed inputs on a laptop in under a minute; the only external input is PR #195's six `per_round.parquet`, passed with `--seed-dir`.
- **The live target is the group-level innovation, and it is a single-round property.** The honest arm's per-round gap shock is 0.5231 of its own individual sd against the human 0.6018. That is between-agent correlation within a round, which is the campaign's known independence floor (§6), and it is trainable without any trajectory-level objective — the object is the joint distribution over a group's agents at one round, not the propagation of state across rounds. What is *not* indicated is anything that steepens the contributor's response to its group mean: that gain is already at the human value and PR #191 showed the off-manifold gain is flat-to-rising in this trunk.
- **The reshuffle is an untouched, switch-slot hypothesis worth a declaration.** The human gap survives a membership change at 0.51 and increasingly so late in the game; the simulation's at 0.33 and decreasingly. If switching were assortative on contribution in the human data and is not in the simulation, that is a switch-model deficit with a clean behavioral sentence and a 28% share of the spread shortfall. It needs its own measurement first — the correlation between a switcher's own contribution and the gap between their old and new group — on more than 169 transitions, which means the human data is the binding constraint and a bootstrap, not a new simulation, is the right instrument.
- **Anyone quoting the pooled coefficient should quote the convention beside it.** 40% of the pooled human-simulation difference is the suite scoring a simulated timeout at 9 and dropping a human one. PR #197 §5 already escalated this to the maintainer; it shows up here as a 0.018 bias in a 0.045 difference, and it will keep contaminating any group-level statistic until `per_round.parquet` carries a validity column.
- **The copula's effect on this coefficient is a usable diagnostic for any future noise model.** A device that supplies between-group dispersion honestly should raise the innovation, not the gain. This one raises the gain on every sample and in both pairs, and raises it most at exactly the transitions where a label-attached latent cannot be right.
