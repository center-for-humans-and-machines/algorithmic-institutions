# What does the punishment landscape look like, once the rule family is continuous?

## 1. Declaration

**Slot:** none. Nothing is trained, nothing is recalibrated, no artifact changes. Like all three of its ancestors this is a measurement branch: it replaces a handful of named rules with a five-parameter family, searches the family, and reports the shape of what it finds.

**Parent:** `origin/base/rule-sigmoid` at `b6501ae` -- `auto/rule-inverted-targeting` (PR [#217](https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/217)) with `auto/free-punishment-fix` merged in. Branch `auto/rule-sigmoid-family`. Isolated remote dir `~/repros/ai-runs/rule-sigmoid` (delete when the PR closes). Byte-identical artifacts to the paired arms: the four sha256s are printed in every job log.

**The free-punishment fix matters for this arm specifically.** Every earlier contribution-keyed rule punished 100% of timed-out cells, which is wasted spend, and under the old env accounting it was also free. On this base `punish()` zeroes the action wherever the player gave no input, so the env and the corrected accountings coincide and no rule in the family can buy anything there. A family searched on the old base would have had a parameter direction whose payoff was an accounting artefact.

### The question

The rule arms so far have searched by hand: a threshold at four levels crossed with three amounts, a proportional rate, two human-shaped tables, then a set of inverted mirrors. That is enough to answer sign questions -- does punishing pay, does direction matter -- and it is not enough to answer shape questions. A hand-picked grid cannot say whether the objective has a sharp optimum or a flat ridge, and it cannot say whether the incumbent is at the top of a hill or merely the best of five points on a plateau.

So: a continuous family, a space-filling design, and a surrogate fitted to all of it.

### The family

```
f(c)    = 1 / (1 + exp((c - c0) / tau))
m_ep(t) = ((T - t) / T) ** gamma_ep          T = 24, t = round number
m_sw(s) = ((s + 1) / S) ** gamma_sw          S = 4, s = rounds until reshuffle
p       = round(P_max * f(c) * m_ep(t) * m_sw(s)), clipped to [0, 30]
```

Five parameters: `P_max`, `c0`, `tau`, `gamma_ep`, `gamma_sw`.

**The logistic was chosen because it nests the incumbent continuously.** As `tau -> 0` it becomes a hard step, so `P_max = 10, c0 = 9.5, tau -> 0, gamma = 0` is exactly `thr9_p10` -- punish 10 whenever a player contributed 9 or less -- the best manager measured anywhere in this project. `test_sigmoid_rule.py::test_reproduces_thr9_p10_against_the_rule_family` asserts that cell by cell, on every contribution level and every round, against `RuleBasedManager(rule="threshold", ...)`'s own output. A family that cannot reproduce the incumbent is not a generalisation of it.

**The multipliers run on the remaining horizon, not the elapsed one.** Punishment is an investment: it is charged now and collected later, through the contributions it raises. What is left to collect is what remains of the episode and what remains of this player's tenure before the next reshuffle -- a player who moves away next round returns nothing. `gamma = 0` switches a multiplier off, so the family also nests every memoryless rule the earlier arms ran.

### Two objectives, reported separately, because they disagree

Measured in this same setting, correctly-targeted rule minus never-punishing:

| quantity | difference | 95% CI |
|---|---|---|
| total contribution | +8.12 | [4.01, 12.22] |
| common pool | +1.07 | [-5.79, 7.93] |

Punishment gains 8.12 contribution, worth 13.0 of pool at the 1.6 multiplier, and costs 12.72 in punishment spend. Net 0.3. **On the pool objective punishing is close to break-even by construction, so the optimal `P_max` may genuinely be near zero; on contribution it is not.** Both are optimised and both are reported. They are never averaged and neither is picked silently.

Both are **seat totals per round**: the focal manager's group's summed contribution, and `1.6 * sum(c) - sum(p)` over that group. A total, not a per-member average, because the competing setting prices membership -- a rule that raises contributions and loses the members who make them has not gained anything (`rule-vs-clone-paired.md` section 3.6).

### How targeting is measured, and how it is not

**Three numbers together, never the rank alone.** Every rollout accumulates the full 21 x 31 contribution-by-punishment table on the focal seat's valid cells -- the agent-round joint distribution, not six bin means -- and targeting is reported as a triple:

| statistic | what it is | what it catches |
|---|---|---|
| `rho` | tie-corrected Spearman rho from that table | aim, invariant to force |
| `magnitude` | range of the bin means over their mean | whether the aim has any size |
| `noise_gate` | that range over its own standard error across episodes | whether the size is real |

Both halves of that are measured traps, not preferences.

*Rank is needed because a difference of bin means measures force.* A sibling arm's largest apparent shape difference, **-11.1**, was entirely a level artefact: mean punishment down to a third, punish rate up threefold, the contribution-to-punishment relationship unchanged to the third decimal. And the human managers and the clone differ by **1.088** on a difference of bin means while being identical on rank -- both strictly monotone decreasing across all six bins -- with rescaling recovering most of the gap, so **42%** of that apparent difference in aim was force.

*Magnitude and the gate are needed because rank is blind to flatness.* Rank discards size entirely, so a profile falling 5.00 to 4.99 scores like one falling 4.76 to 0.27; a sibling's exploration buffer scored **-0.540** on a relationship whose spread relative to its own mean was **0.002** -- a flat policy plus sampling noise, ranked. That risk is larger here than in any hand-picked arm, because a search over a thousand candidates will produce rules that punish almost nothing, and rank would call them beautifully targeted. **The objective is untouched by this: it was never the rank.** A rule is called targeting only when it is strong in rank *and* non-negligible in magnitude *and* above the gate. `scripts/tests/test_rule_sigmoid_targeting.py` pins all three properties.

The level (`mean_p_valid`) and the rate (`punish_rate`, `mean_p_given_positive`) are reported beside them, so a reader can always separate aim from force.

**The leaver diagnostic is reported as a ranking, not as a sign test.** It is here because it reads straight off the recorded rounds with no counterfactual, and it does reproduce the ordering of managers -- it correlates with policy shape at r = -0.95 over ten of them. But its zero point does not separate the classes: across four inverted managers only one crossed zero (+0.291) and the other three sat between -0.02 and -1.21. Its noise floor is about 0.577, the size of the differences a fine contrast would ask it to resolve. So no rule is called correctly or incorrectly targeted on the sign of its `c_gap`, and no paired contrast rests on it.

**Timed-out cells are excluded from every shape table in this arm.** A player who gave no input is recorded at the imputed contribution 9 with a forced-zero punishment, so the cell is not a decision; counting it drags the `6-10` column down with rows that never happened. `evaluation_suite.convert.load_sim` does **not** apply that mask (`load_human` does), so every simulated policy-shape table built through the shared loader in this project carries those rows. That is a live defect in frozen shared surface: it is named here and left for the maintainer, not fixed on this branch. This arm builds its own tables in `paired_rollout.contingency`, which masks on `contribution_valid` at source.

### The setting: competing, not self-play

The rule holds group 0 against the behavioural clone in group 1, members free to move every fourth round. Not self-play, because the parent arms established that self-play rankings do not survive competition: `prop10` led by +12.5 pool points per seat in self-play and came out at **-15.6** [-23.1, -8.0] against a live rival.

### Artifact naming contract

| what | path |
|---|---|
| the family | `src/aimanager/manager/sigmoid_rule.py`; also `RuleBasedManager(rule="sigmoid", ...)` |
| batched paired rollout | `src/aimanager/manager/paired_rollout.py` |
| batched clone opponent | `src/aimanager/manager/linear_opponent.py` (taken unchanged from `origin/auto/rl-manager-evolution-strategies`) |
| design | `scripts/rule_sigmoid/design.py` |
| sweep | `scripts/rule_sigmoid/sweep.py`, `scripts/rule_sigmoid/sweep.slurm` |
| surrogate | `scripts/rule_sigmoid/fit_surrogate.py` |
| validation design | `scripts/rule_sigmoid/validation_design.py` |
| tables and figures | `scripts/rule_sigmoid/report.py` -> `plots/data_analysis/rule_sigmoid/` |
| tests | `src/aimanager/tests/test_sigmoid_rule.py`, `test_paired_rollout.py`, `test_linear_opponent.py` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Implement the family; pin the `thr9_p10` reproduction and both multipliers in tests. | **done** |
| 2 | Build a batched paired rollout so a thousand-point design is affordable, and pin its seat accounting, its dispatch and its leaver diagnostic in tests. | **done** |
| 3 | Calibrate the new harness against the established simulation path on the rules both can run. | |
| 4 | Sobol design, fit seeds; GP with a noise term; optimum, length scales, Hessian, flat region, per objective. | |
| 5 | Validate the chosen parameters, the incumbents and ridge/boundary probes on held-out seeds. | |
| 6 | Report realised spend, policy shape on the evaluation suite's bins beside the human and the clone, an intensity-invariant targeting statistic, and the leaver diagnostic as a ranking. | |
| 7 | Log, PR. | |

## 3. Results

### 3.1 The harness reproduces the established simulation path (measured)

The sweep does not go through `simulate.py`. It runs the same env, the same four artifacts (sha256s printed in every job log) and the same protocol -- 2 x 8 agents, 24 rounds, `switch_every` 4, the rule in group 0 and the clone in group 1 -- but on the batch dimension, with one parameter vector per episode. That is new code between the models and the numbers, so it is checked against numbers the old path produced before it is used for anything.

One rollout, 1024 episodes, seed 42, against the published `26_rule_inverted_targeting` figures (300 episodes, `simulate.py`, the corrected accounting):

| manager | members, here | members, published | mean p, here | mean p, published | mean c, here | mean c, published | `c_gap`, here | `c_gap`, published |
|---|---|---|---|---|---|---|---|---|
| `never` | 4.62 | 4.57 | 0 | 0 | 8.64 | 8.48 | -1.45 | -1.20 |
| `thr9_p10` | 3.91 | 3.92 | 2.90 | 2.80 | 11.74 | 11.48 | -3.64 | -3.51 |
| `ah_punisher` (control) | 3.99 | 4.00 | 1.85 | ~1.9 | 10.03 | ~10.2 | -2.26 | -2.35 |

Membership agrees to 0.05 of a member, realised spend to 0.1, mean contribution to 0.26 and the leaver gap to 0.14 -- the last well inside its own 0.577 noise floor. The common pool agrees less tightly (`thr9_p10` 60.69 here against 62.32 published, `never` 62.64 against 62.22) but within about one standard error of this run's own 1.5, and the two runs are different RNG streams with the free-punishment fix in place here and not there.

**Throughput, which is what made the design affordable.** A rollout costs about the same whatever its batch size, because the cost is per-round Python and model-call overhead rather than arithmetic -- the contribution GNN is 35 KB. Measured: 8 rollouts of 768 episodes in 42 seconds on an A100, and 3 seconds for a 192-episode rollout on four CPU threads. `simulate.py` needs about 77 seconds for 100 episodes of one pairing. So the sweep runs on CPU nodes, and a thousand design points at 512 episodes each is an hour of ordinary batch time rather than a GPU campaign: **1,030 points x 512 episodes x 2 seeds = 1.05M episodes in 8 array tasks of about 3.5 minutes each.**

### 3.2 What was run (measured)

| arm | design | seeds | episodes / point | jobs |
|---|---|---|---|---|
| design sweep | 1,024 Sobol + 6 anchors | 42, 43 (**fit**) | 512 | 30422080_[0-6], 30422657_7 |
| surrogate check | the same 1,030 | 44 (**never fitted on**) | 256 | 30422482_[0-3] |
| validation | 46 rows: optima, incumbents, ridge and boundary probes | 45, 46, 47 (**held out**) | 2,048 | 30423322, 30423323 |
| cross-check | `simulate.py`, 10 pairings | 42, 43, 44 | 100 | 30423324-6 |

The design is scrambled Sobol over `P_max` in [0, 30], `c0` in [0, 20], `log10 tau` in [-2, 1], and both exponents in [0, 3]. `tau` is sampled logarithmically because it spans its interesting range multiplicatively: at 0.01 the logistic is a hard threshold on every integer contribution, at 10 it is nearly flat across the whole scale.

**Two of the three box edges are not arbitrary and one is.** `P_max = 30` is the action space's own ceiling (`n_punishments = 31`) and `c0` in [0, 20] is the contribution scale. Nothing fixes `gamma <= 3`, so the validation design carries probes at `gamma = 4` and `6` as well as at `-0.5` and `-1`, which is where the "punishment is an investment" premise gets tested rather than assumed.

**The anchors reproduce the contrast this arm was given.** `thr9_p10` minus `never`, measured here at 1,024 episodes each:

| quantity | here | quoted in the brief |
|---|---|---|
| total contribution | **+7.86** | +8.12 [4.01, 12.22] |
| common pool | **+1.73** | +1.07 [-5.79, 7.93] |

So the harness is measuring the same thing the objectives were defined on, and the break-even arithmetic carries over: punishment buys 7.86 contribution, worth 12.6 of pool, and costs 10.8 in spend.

### 3.3 The surrogate predicts a seed it never saw, to within that seed's own noise (measured)

The whole 1,030-point design was re-run on seed 44 at 256 episodes, and the already-fitted GPs -- loaded from disk, not refitted -- were asked to predict it.

| objective | held-out RMSE | that seed's own measurement noise | R2 against noise | R2 against variance | Spearman | bias |
|---|---|---|---|---|---|---|
| total contribution | 1.810 | 1.765 | **-0.05** | 0.762 | 0.861 | +0.05 |
| common pool | 2.973 | 2.871 | **-0.07** | 0.872 | 0.833 | +0.07 |

**The residual is the measurement noise and essentially nothing else.** An R2-against-noise of about zero is the target, not a failure: it says the surrogate's error on unseen data is the same size as the error of simply measuring that point again, so there is no structure left for it to have learned and none it has invented. A surrogate that had fitted noise would score well in sample and badly here.

That matters because the noise term was the thing most likely to be got wrong. `sklearn`'s `normalize_y` divides the target by its own standard deviation and does **not** scale `alpha` or the `WhiteKernel` with it -- checked empirically rather than assumed, because supplying the measured standard errors in the wrong units would have handed the fit a noise level about 60x too small and produced exactly the sharp false optimum the noise term exists to prevent. With it right, the fitted unexplained noise is **0.36** (contribution) and **0.59** (pool) against measured standard errors of 0.87 and 1.41: the per-point sampling error already accounts for most of the scatter.

### 3.4 The two optima agree on whom to punish and disagree on when (measured)

Maximising each posterior mean over the box:

| objective | `P_max` | `c0` | `tau` | `gamma_ep` | `gamma_sw` | surrogate value | its sd | at a box edge |
|---|---|---|---|---|---|---|---|---|
| total contribution | **30.0** | 8.97 | 0.206 | **0.00** | 1.35 | 50.12 | 0.66 | `P_max`, `gamma_ep` |
| common pool | **30.0** | 7.14 | 0.425 | **3.00** | 1.42 | 71.50 | 1.22 | `P_max`, `gamma_ep` |

They agree on four of the five parameters and disagree completely on the fifth, and the fifth is the episode horizon. Both want the action space's ceiling, both aim just below the untreated mean contribution, both use a fairly sharp threshold, and both discount towards the reshuffle at `gamma_sw` around 1.4. **The contribution optimum wants no episode discount at all -- punish as hard in round 23 as in round 0 -- and the pool optimum wants the most the box allows.**

That is not a difference of degree dressed up. The set of parameter vectors the surrogate cannot distinguish from its own optimum, within one measurement standard error:

| parameter | contribution | common pool | overlap |
|---|---|---|---|
| `P_max` | 24.8 - 29.8 | 25.8 - 29.8 | yes |
| `c0` | 7.8 - 9.9 | 6.0 - 8.5 | yes |
| `tau` | 0.05 - 0.69 | 0.10 - 1.13 | yes |
| **`gamma_ep`** | **0.03 - 0.78** | **1.70 - 2.92** | **none** |
| `gamma_sw` | 0.33 - 1.93 | 0.91 - 2.14 | yes |

(5th to 95th percentile of a 65,536-point Sobol sweep of the box kept within `delta` of the optimum; `delta` is 0.94 for contribution and 1.53 for pool, each the standard error of a single well-measured design point.)

**Only `gamma_ep` fails to overlap, and it is the flattest direction in both surrogates.** The ARD length scales say so directly -- on the unit box, contribution 1.17 / 0.89 / 1.09 / **4.44** / 2.61 and pool 2.00 / 1.30 / 1.57 / **4.14** / 3.76, in the order of the table -- and so does the Hessian: the curvature at the contribution optimum runs from **-215** along `c0` to **+0.66** along `gamma_ep`, a ratio of 325, and at the pool optimum from **-269** to **-0.93** along a direction that is 87% `gamma_ep`, a ratio of 290.

So the landscape is a long flat ridge in the episode exponent, and the two objectives sit at opposite ends of that same ridge. A point estimate of either would have hidden both facts.

**But the ridge being flat is exactly why the disagreement is cheap.** Measured, on the fit seeds: the best design point for the pool (`sobol0279`) scores 72.92 pool / 49.05 contribution, and the best for contribution (`sobol0176`) scores 69.44 / 50.31. Taking the wrong objective's rule costs about **3.5 pool points or 1.3 contribution points** -- two to three standard errors, not a change of regime. The two optima are separated in the one parameter whose separation matters least. That is the honest way to put "do they differ in kind": their *parameter settings* do and do not overlap, and their *consequences* barely do.

### 3.5 The second harness agrees on every contrast, and on levels only to a common offset (measured)

`simulate.py`, 10 pairings, three seeds of 100 episodes, the established path (jobs 30423324-6). Focal seat against the clone, compared with the sweep on the same fit seeds:

| quantity | agreement |
|---|---|
| realised mean punishment | to **0.002 - 0.016** |
| mean contribution per valid cell | to **0.02 - 0.30** |
| leaver gap `c_gap` | to **0.02 - 0.44**, inside its own 0.577 noise floor |
| seat total contribution | sim **4.0 - 5.1 higher**, against a sim seed sd of 3.3 - 5.0 |
| seat common pool | sim **5.9 - 6.7 higher**, against a sim seed sd of 5.2 - 8.7 |
| seat membership | sim **0.31 - 0.48 higher**, against a sim seed sd of 0.10 - 0.27 |

**Every contrast agrees; only the level is offset**, and the level is offset because the sim run holds about 0.45 more members in its focal seat across all of its pairings at once:

| contrast (focal seat, against the clone) | `simulate.py` | sweep | difference |
|---|---|---|---|
| `thr9_p10` - `never`, contribution | +8.91 | +7.86 | -1.05 |
| `thr9_p10` - `never`, pool | +2.02 | +1.73 | -0.29 |
| `thr9_p10` - `never`, members | -0.51 | -0.67 | -0.16 |
| clone - `never`, contribution | +3.32 | +3.19 | -0.13 |
| clone - `never`, pool | -2.67 | -2.13 | +0.55 |
| clone - `never`, members | -0.40 | -0.52 | -0.13 |

Every claim in this arm is a contrast between two rules measured in the same harness, so a common seat-size offset cancels out of all of them.

**On the level, it is the fresh simulation that is the outlier, not the sweep.** Against the published parent-arm figures (`rule-inverted-targeting` section 3.3 and `rule-vs-clone-paired` section 3.7, both `simulate.py` at 300 episodes), focal membership against the clone:

| manager | published | sweep | this cross-check run |
|---|---|---|---|
| `never` | 4.57 | **4.54** | 4.86 |
| `thr9_p10` | 3.92 | **3.87** | 4.35 |
| clone (symmetric control) | 4.00 | **4.02** | 4.46 |

The sweep lands within 0.05 of the published numbers on all three; this cross-check run sits about 0.4 above both. Its own symmetric control is asymmetric (4.46 focal against 3.54 rival where the published control was 4.00 / 4.00), which is where the offset comes from. **This is reported rather than explained.** The cross-check config carries a different manager set from the parent's, and `MultiManager` evaluates every manager in a file on every round, so the RNG stream differs -- the parent arms' note 4 on exactly this point. That accounts for a redraw; whether it accounts for a redraw this large is not established here, and it is left open.

### 3.6 Held-out validation: both fitted rules beat the incumbent, decisively (measured)

52 rules on seeds 45, 46 and 47 -- never used to fit anything -- at 2,048 episodes each, so **6,144 episodes per rule**. Focal seat against the clone. Standard errors are in `validation_table.csv`; they run 0.34-0.40 on contribution and 0.54-0.71 on the pool, so a difference of two rules carries about +/- 0.55 and +/- 0.9 respectively.

| rule | `P_max` | `gamma_ep` | total contribution | common pool | members | realised spend | severity when it fires |
|---|---|---|---|---|---|---|---|
| `best_design_pool` | 27.2 | 2.90 | 47.45 | **70.21** | 4.38 | 1.30 | 6.32 |
| **`opt_pool`** (fitted) | 30.0 | 3.00 | 47.02 | **69.71** | 4.38 | 1.26 | 6.83 |
| `best_cap20_pool` | 18.2 | 2.84 | 44.89 | **67.53** | 4.38 | 0.98 | 4.52 |
| `best_cap10_pool` | 9.3 | 1.09 | 43.35 | **65.19** | 4.41 | 0.94 | 3.05 |
| **`opt_contribution`** (fitted) | 30.0 | 0.00 | **49.92** | 64.87 | 4.01 | 3.74 | 17.03 |
| `best_design_contribution` | 23.0 | 0.80 | 47.62 | 65.31 | 4.07 | 2.68 | 6.17 |
| `best_cap10_contr` | 9.7 | 0.63 | 44.44 | 64.20 | 4.27 | 1.61 | 3.84 |
| `thr4_p10` | 10.0 | 0 | 42.61 | 62.02 | 4.29 | 1.44 | 10.00 |
| **`thr9_p10`** (incumbent) | 10.0 | 0 | 44.64 | 60.19 | 3.87 | 2.90 | 10.00 |
| `never` | 0 | -- | 37.30 | 59.68 | 4.58 | 0 | -- |
| `ah_punisher` (clone control) | -- | -- | 40.09 | 56.85 | 4.04 | 1.80 | 5.97 |

**Against `thr9_p10`, plainly:**

| claim | margin | standard error | verdict |
|---|---|---|---|
| `opt_pool` on the pool | **+9.52** | 0.86 | beats it, 11 sd |
| `opt_contribution` on contribution | **+5.28** | 0.53 | beats it, 10 sd |
| `opt_pool` on contribution | +2.38 | 0.52 | also beats it |
| `opt_contribution` on the pool | +4.68 | 0.91 | also beats it |

**This is not a tie and it is not close.** Both fitted rules beat the incumbent on both objectives at once, and the pool margin is more than ten times its own standard error.

### 3.7 The margin survives a hard severity cap, and the capped rule wins on spend as well (measured)

The unconstrained optima sit at `P_max = 30`, the top of the action space, where the contribution model has almost no evidence. The capped champions are the best design points whose `P_max` never exceeds 20 or 10, so nothing they can ever issue is an extrapolation.

| rule | max punishment it can issue | spend | common pool | vs `thr9_p10` |
|---|---|---|---|---|
| `best_cap20_pool` | 18.2 | 0.98 | 67.53 | **+7.34** (8.5 sd) |
| `best_cap10_pool` | 9.3 | 0.94 | 65.19 | **+5.00** (5.9 sd) |
| `thr9_p10` | 10.0 | 2.90 | 60.19 | -- |

**`best_cap10_pool` cannot punish harder than the incumbent, punishes a third as much on average, and produces five more pool points.** That margin is a claim about the model's behaviour inside the region the model was trained on, which the `P_max = 30` optima are not. Its cost is 1.29 contribution points against `thr9_p10` (2.4 sd), so it is not free -- it is the pool-side end of the same trade the two objectives disagree about.

It wins by holding its members rather than by raising contributions: **4.41 members against 3.87**, +0.54, while mean contribution per valid member is lower. The whole margin is membership times a slightly smaller per-member pool.

**At matched spend the family dominates the incumbent across the range.** Binning the 1,024 design points by realised spend, the best attainable pool per decile (fit seeds):

| spend decile | 0.09 | 0.34 | 0.57 | 0.83 | 1.14 | 1.50 | 1.95 | 2.57 | 3.48 | 5.75 |
|---|---|---|---|---|---|---|---|---|---|---|
| best pool | 65.3 | 66.6 | 68.1 | 70.6 | **72.9** | 70.0 | 69.8 | 69.4 | 66.4 | 63.8 |
| best contribution | 41.3 | 42.8 | 44.4 | 46.7 | 49.0 | 47.8 | 49.2 | **50.3** | 49.8 | 50.2 |

`thr9_p10` spends 2.80 and reaches 61.3 pool on the same seeds; the best rule at *one tenth* of that spend reaches 65.3. The pool optimum sits at spend 1.14 and the contribution optimum at 2.57, which is the trade in one line.

### 3.8 What the horizon exponents actually do (measured)

**The episode exponent is a lever on spend, and the two objectives pull it in opposite directions.** Probing outside the design box at the fitted optima:

| `gamma_ep` | `opt_pool`: pool | `opt_pool`: spend | `opt_contribution`: contribution | `opt_contribution`: spend |
|---|---|---|---|---|
| -1.0 | 65.35 | 3.59 | 50.67 | 4.41 |
| -0.5 | 65.18 | 3.45 | **51.22** | 4.10 |
| 0.0 | -- | -- | 49.92 | 3.74 |
| 3.0 (fitted) | **69.71** | 1.26 | -- | -- |
| 4.0 | 68.91 | 1.10 | -- | -- |
| 6.0 | 68.06 | 0.88 | -- | -- |

**The premise the multiplier was built on holds for the pool and fails for contribution.** Punishment as an investment -- discount it as the horizon runs out -- is right on the pool: inverting the exponent costs 4.4 pool points, and the fitted value of 3.0 is genuinely near the optimum rather than pinned by the box (4.0 and 6.0 are both slightly worse). On total contribution the opposite is true: the box edge at 0 *was* binding, and pushing to `gamma_ep = -0.5` -- punish **harder** as the episode runs out -- buys another **+1.30** contribution (2.4 sd). Contribution does not care that the investment has no time to pay back; the pool does, because it pays the bill.

**The reshuffle exponent works through its alignment with the switch decision, not only through the length of the remaining tenure.** `m_sw`'s trough sits at `s = 0`, which is by construction the round the switch predictor's decision is read, so the two explanations coincide at phase 0. Rotating the cycle keeps the same four multipliers and the same average discount and moves the trough off that round:

| phase | `opt_pool` pool | `opt_contribution` pool | `opt_contribution` contribution |
|---|---|---|---|
| 0 (aligned) | **69.71** | **64.87** | **49.92** |
| 1 | 67.13 | 61.44 | 47.08 |
| 2 | 65.99 | 58.59 | 45.43 |
| 3 | 64.72 | 58.68 | 45.59 |

**Misaligning the cycle costs 2.6 to 5.0 pool points for the pool optimum and 3.4 to 6.3 for the contribution optimum**, against a standard error of about 0.9. `opt_contribution` is the clean case: its `gamma_ep` is exactly 0, so its episode multiplier is off entirely and the phase changes *nothing* except where the trough falls in the reshuffle cycle. The effect is therefore reshuffle alignment and nothing else.

So the honest reading of `gamma_sw` is narrower than the reasoning it was built on: what it buys is **not punishing on the round the switch decision is taken**. Whether that is responding to the incentive or exploiting the switch predictor is not settled here -- both would produce this measurement -- but the "remaining tenure" story alone does not, because it is indifferent to phase.

## 4. Notes

1. **Measured against inferred.** Sections 3.1 to 3.7 are measurements. The *readings* are inferences and are marked as such where they appear: that `gamma_ep` trades contribution for spend, that `gamma_sw` works by moving punishment away from the switch-decision round, and that the fitted optimum's advantage over the incumbent is a redistribution of spend rather than more of it. The first and third are supported by the round-resolved series and by the matched-spend table; the second is supported by the phase probe, which is a direct measurement of that specific mechanism and not an argument about it.

2. **The design box is not the family, and one of its edges is arbitrary.** `P_max <= 30` is the action space's own ceiling and `c0` in [0, 20] is the contribution scale, but nothing fixes `gamma <= 3`. Both fitted optima sit on a `gamma_ep` edge -- the contribution optimum at 0, the pool optimum at 3 -- so the box is binding for both, in opposite directions. Section 3.6 reports what the probes outside it found instead of leaving the reader to wonder.

3. **The unconstrained optimum is a claim about the model, not about people.** Both fitted optima sit at `P_max = 30`, and the contribution model has evidence for almost nothing up there: 4.49% of its training rows follow a punishment above 10 and 1.50% follow one above 20 (manager review S2). This is the same warning the first rule sweep attached to `prop10`, and it applies here with more force because a search will find whatever the model rewards, including in regions where the model is guessing. The capped champions exist so the headline does not have to rest on an extrapolation.

4. **The objective is not the targeting statistic.** `rho`, `magnitude` and `noise_gate` are descriptive: they are computed after the fact and never entered the search. A search that optimised a rank correlation would have selected rules that punish almost nothing, which is precisely the failure the noise gate is there to detect; that failure cannot occur here because the two objectives are seat totals.

5. **The leaver gap is a ranking here and nothing more.** It is reported for every validated rule because it reads straight off the recorded rounds with no counterfactual, and it reproduces the ordering; but its zero point does not separate correctly- from incorrectly-targeted managers and its noise floor of 0.577 is the size of a fine contrast. No claim in this log rests on its sign.

6. **A live defect in shared code is named and not fixed.** `evaluation_suite.convert.load_sim` does not mask a timed-out player's imputed contribution the way `load_human` masks it, so every simulated policy-shape table built through the shared loader carries those rows in its `6-10` bin. This arm builds its own tables with the mask applied at source (`paired_rollout.contingency`). The shared directory is frozen surface and the fix is the maintainer's call.

7. **Nothing here was trained and nothing was recalibrated.** The four artifacts are byte-identical to the parent arms' and their sha256s are printed in every job log. The whole arm is about 1.6M simulated episodes, which is roughly an hour of ordinary CPU batch time across a handful of array tasks -- reproducing it needs no GPU at all.

### Successor

1. **The cheap evaluation is the reusable thing here, more than the rule.** `paired_rollout` turns a competing-setting evaluation from 77 seconds per 100 episodes into a batched rollout whose cost barely depends on how many episodes or how many policies it carries, on CPU nodes rather than the GPU queue every other arm is waiting in. Any arm that wants to *compare* managers rather than train one can use it, and a trained manager can sit in either seat as long as it exposes `predict(state) -> (punishment, None)`.

2. **Re-point the RL manager's baseline again, and at a capped rule.** The parent arms concluded that the honest bar in the competing setting is the clone's seat and `never`'s seat. That is now too low: a five-parameter rule whose severity never exceeds the incumbent's clears both comfortably. The bar for a trained manager should be the capped champion in section 3.7, reported at matched spend, with group size beside the pool.

3. **Put the horizon multipliers in the manager's observation, not only in a rule.** The one thing the family found that the earlier arms could not express is *when* to punish, and it is worth more than anything the threshold shape buys. A learned manager that cannot see the round number or the rounds-to-reshuffle cannot represent the policy that wins here. Both are already in the env state.

4. **The `gamma_ep` box edge is the open question this arm leaves.** Both optima sit on it, in opposite directions, and section 3.6's probes bound it only at the four values they test. A successor that wants the actual optimum should re-run the design with `gamma_ep` in [-1, 6] rather than widen it by hand.

5. **Nothing here needs a retrain and nothing here was one.** Eight CPU array tasks of about 3.5 minutes reproduce the design; two more reproduce the validation; three 12-minute GPU jobs reproduce the cross-check.
