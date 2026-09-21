# Does punishing pay, and can a simple rule beat our clone of a human manager?

## 1. Declaration

**Slot:** none. This is not a model experiment -- nothing is trained, nothing is recalibrated, no artifact changes. It is the cheap control for the expensive RL question: before spending training runs on a manager, find out with rules whether punishment buys anything at all in this world, and what a trained manager would have to beat.

**Parent:** `auto/sim-timeout-imputation` at `3fe1f44` -- the most corrected simulation. Branch `auto/rule-based-manager-sweep`, PR opens with `--base auto/sim-timeout-imputation`. Isolated remote dir `~/repros/ai-runs/rule-managers` (delete when the PR closes).

**Prior work reused.** Branch `origin/99-rule-based-manager-strategy-testing` (PR #99) already ran a `RuleBasedManager` with two rules (`prev_c_threshold`, `tactical_early`) against the pre-correction stack. Its shape -- a manager class dispatched through `MANAGER_CLASS`, one pairing per rule, self-play against the artificial humans -- is what this branch builds on; its rules are subsumed by the threshold family below (its `prev_c_threshold` keyed on the *previous* round, which the punisher-timing work since then has shown is the wrong round to key on).

### The question

Punishment is immediately negative-sum for the group: the pool is `1.6*sum(c) - sum(p)`, so one point of punishment costs the pool one point now and only pays later, through raised contributions, at 1.6 per contribution unit. **One point of punishment therefore has to buy more than 0.625 contribution units to break even.** The simulated contributors are known to under-react to punishment once the simulation runs closed-loop (RCB self-play 0.797 against a teacher-forced 0.095, PR #183). So it is entirely possible that against these players the best policy is not to punish -- and if so, that explains in advance anything a trained manager does.

### The rule family, one sentence each

Every rule reads the **current** round's contribution `c` -- the quantity the human manager saw when deciding (review section 3.1) -- and returns a punishment in 0..30.

| manager | definition |
|---|---|
| `never` | Never punish: `p = 0` for every agent in every round. |
| `thr{T}_p{A}` | Punish a flat `A` whenever this round's contribution is at most `T`, and nothing otherwise. `T` in {4, 9, 14, 19}, `A` in {2, 5, 10}. |
| `prop{R}` | Punish in proportion to the shortfall from the maximum: `p = round(R * (20 - c))`. `R` in {0.1, 0.25, 0.5, 1.0}. |
| `human_mean` | Punish the rounded average a human manager gave at that contribution level, `round(E[p|c])`, read off the human data. |
| `human_severity` | Fire with the human managers' own frequency at that contribution level, `P(p>0|c)`, and when it fires punish the rounded human severity `round(E[p|p>0,c])` -- the human shape with its sparsity and its severity kept apart. |
| `ah_punisher` | **The baseline to beat:** the artificial punisher, our clone of a human manager -- the frontier stack's `punishment_multinomial_timeout_severity_copula` bundle, in the identical setup. |

**Why this space.** Three degrees of freedom and no more: *whether* to punish (`never` against everything else), *who* (the threshold `T`, i.e. how far down the contribution scale the manager reaches), and *how hard* (`A`, or the rate `R`). The thresholds bracket the contribution distribution at its quartile-ish points; the amounts bracket the human mean severity (`E[p|p>0] = 5.86`) by a factor of ~2.5 either way, so the sweep contains rules that punish far less and far more than a human did. The two human-shaped rules exist so that "the human policy's shape, without the human policy's history-dependence" is in the ranking -- the difference between them and `ah_punisher` is exactly what the punisher's memory and its copula buy.

### The two accountings

Per review findings D1 and D2, the environment's own numbers are not the whole truth, so every manager is reported on both:

* **env** -- what `ArtificialHumanEnv` computes. A timed-out player's contribution is zeroed (correct) and the punishment aimed at them is zeroed too, so it is **free**; that player's own payoff is discarded, which the real game paid.
* **corrected** -- computed directly from contributions and punishments. Punishment aimed at a timed-out player is **charged** to the pool, because the artificial humans were shown it and reacted to it, so it was really spent. The timed-out player is paid `20 - 0 - 0 + common_good`, as the real game paid them.

### Artifact naming contract

| what | path |
|---|---|
| rule family | `src/aimanager/manager/api_manager.py::RuleBasedManager` |
| config generator | `scripts/data_analysis/rule_manager_configs.py` |
| sim configs | `configs/simulation/manager_testing/24_rule_managers_s{seed}_{a,b,c}.yml` |
| analysis | `scripts/data_analysis/rule_manager_sweep_report.py` |
| tables and figures | `plots/data_analysis/evaluation/rule_based_managers/` |
| tests | `src/aimanager/tests/test_rule_managers.py` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Read PR #99's `RuleBasedManager`, the frontier sim config and the manager review; generalise the rule family and pin each rule's mapping in a unit test. | **done** -- 10 tests pass on Raven |
| 2 | Give the simulation the two things the comparison needs: `contribution_valid` in `per_round.parquet`, and an opt-in `reseed_per_run` so every manager in one config starts from the same seed. | **done** |
| 3 | Run the 19 rules plus the artificial punisher against the frontier stack, seed 42, standard protocol. | **done** -- jobs 30400716 / 30400717 / 30400718 |
| 4 | Rank on both accountings; policy shapes; timeout-cell exploitation rates. | **done** -- section 3 |
| 5 | Re-run everything on seeds 43 and 44, and add a head-to-head arm in which the decisive managers share one RNG stream; judge every margin against that spread. | **done** -- jobs 30400850 / 30400852-30400856 (sweep), 30400886 / 30400888 / 30400889 (head-to-head) |
| 6 | Log, PR. | **done** |

## 3. Results

### 3.0 What was run

Twelve simulations, all `COMPLETED`, all on the frontier stack with byte-identical artifacts and the standard protocol (2 groups x 8 agents, 24 rounds, 100 episodes, `save_per_round: true`); nothing was trained or recalibrated.

| arm | configs | jobs | managers per config |
|---|---|---|---|
| sweep, seeds 42 / 43 / 44 | `24_rule_managers_s{42,43,44}_{a,b,c}.yml` | 30400716-8, 30400850, 30400852-6 | 7 / 7 / 6, 20 in total |
| head-to-head, seeds 42 / 43 / 44 | `24_rule_managers_h2h_s{42,43,44}.yml` | 30400886, 30400888, 30400889 | 8 |

Every table below **pools the three seeds** -- 300 episodes per manager -- and every interval is a 95% bootstrap over those episodes. The seed-by-seed tables are `summary_s4{2,3,4}.csv` and `seed_spread_{sweep,h2h}.csv`.

**The arms exist for different jobs, and cross-arm numbers are not comparable.** `MultiManager` evaluates *every* manager in a config on *every* round, so what a run draws from the RNG depends on the config's manager set; two managers are stream-comparable only inside one config. The sweep maps the landscape (20 managers, 3 shards); the head-to-head puts the decisive eight in one file so their differences are as close to the rule alone as this simulation allows. The same manager therefore carries different numbers in the two arms (e.g. `ah_punisher` 113.18 in the sweep, 111.06 head-to-head) -- an ordinary redraw, inside the bootstrap interval. A claim counts here only if both arms agree.

### 3.1 The headline: the head-to-head arm, 300 episodes each

Common good = the pool `1.6*sum(c) - sum(p)` per round, summed over **both** groups (both sides carry the same manager). `_si` = the same rule with `skip_invalid`, i.e. never punishing a player who gave no input.

| manager | common good, env | 95% CI | common good, corrected | group payoff sum, env | group payoff sum, corrected | mean contribution | share punished | mean p | mean p given p>0 |
|---|---|---|---|---|---|---|---|---|---|
| `prop10` | **136.04** | [127.5, 143.9] | 133.07 | 136.04 | 134.47 | 14.36 | 0.544 | 5.64 | 10.36 |
| `prop10_si` | 133.63 | [125.4, 141.4] | 133.63 | 133.63 | 138.82 | 14.24 | 0.550 | 5.76 | 10.47 |
| `thr9_p10` | 123.79 | [117.4, 130.5] | 122.33 | 167.73 | 169.45 | 11.61 | 0.280 | 2.80 | 10.00 |
| `thr9_p5` | 121.59 | [115.4, 127.2] | 120.85 | 179.07 | 182.50 | 10.83 | 0.368 | 1.84 | 5.00 |
| `human_severity` | 116.22 | [109.7, 122.3] | 115.48 | 178.34 | 181.65 | 10.34 | 0.302 | 1.74 | 5.77 |
| `thr9_p5_si` | 114.67 | [108.8, 120.2] | 114.67 | 174.42 | 179.48 | 10.38 | 0.401 | 2.00 | 5.00 |
| **`ah_punisher`** (baseline) | 111.06 | [104.7, 117.6] | 111.04 | 174.74 | 179.65 | 10.01 | 0.308 | 1.87 | 6.08 |
| *human managers (real)* | *103.80* | *[90.9, 116.4]* | *103.80* | *171.96* | *179.59* | *9.57* | *0.315* | *1.85* | *5.86* |
| `never` | **99.63** | [93.9, 105.6] | 99.63 | **194.33** | **199.20** | 7.93 | 0 | 0 | -- |

The human row is a **reference level, not a contestant**: it is the real 50 games, with real contributors. Only the simulated managers are comparable to one another.

### 3.2 Does punishing pay? Yes, and not marginally

Margin over `never`, in common good, 95% bootstrap (head-to-head / 20-rule sweep):

| manager | vs `never`, head-to-head | vs `never`, sweep |
|---|---|---|
| `prop10` | **+36.45** [+26.6, +46.3] | **+33.55** [+23.7, +43.3] |
| `thr9_p10` | +24.20 [+15.2, +32.8] | +28.64 [+20.0, +37.3] |
| `thr9_p5` | +22.10 [+13.6, +30.2] | +19.36 [+10.7, +28.2] |
| `human_severity` | +16.59 [+8.0, +25.0] | +9.31 [+0.8, +17.9] |
| `ah_punisher` | +11.41 [+2.7, +20.0] | +14.88 [+6.2, +23.6] |

In the full sweep, **15 of the 18 punishing rules beat `never`** with an interval that excludes zero; `thr4_p5` (+7.23) and `thr4_p2` (+1.56) are indistinguishable, and only `thr19_p10` (-7.79) is worse -- not significantly. Seed by seed, `never` finished 20th of 21 rows (s42), 19th (s43) and 21st, last, (s44).

**Why, arithmetically.** The pool pays 1.6 per contribution unit and charges 1 per punishment point, so punishment pays above **0.625 contribution units bought per point**. Measured against the `never` arm (`contr_bought_per_punishment` in the summary tables), every rule in the sweep buys between **0.83 and 1.47** -- except `thr19_p10` at **0.53**, the one rule below the break-even and the one rule that loses to `never`. The prediction and the outcome agree rule by rule.

### 3.3 Does a rule beat our clone of a human manager? Yes -- two of them, in both arms

Margin over `ah_punisher`, in common good:

| manager | head-to-head | sweep | verdict |
|---|---|---|---|
| `prop10` | **+25.03** [+14.8, +35.3] | **+18.61** [+8.6, +28.7] | beats it in both arms |
| `thr9_p10` | **+12.78** [+3.5, +21.8] | **+13.69** [+4.4, +23.0] | beats it in both arms |
| `prop10_si` | +22.67 [+12.5, +32.8] | (not run) | beats it |
| `thr9_p5` | +10.68 [+1.9, +19.3] | +4.42 [-4.9, +13.7] | only in one arm |
| `prop05` | (not run) | +10.70 [+1.0, +20.5] | only in one arm |
| `thr14_p5` | (not run) | +10.24 [+0.7, +19.5] | only in one arm |
| `human_severity` | +5.17 [-3.7, +14.2] | -5.63 [-14.5, +3.4] | no |
| `never` | -11.38 [-20.2, -2.5] | -14.89 [-23.7, -5.7] | loses |

The margin to beat is set by run-to-run variation, which is large: the seed-to-seed standard deviation of a manager's 100-episode mean common good runs from 0.86 (`ah_punisher`) to 10.9 (`thr19_p10`), median 5.0 (`seed_spread_sweep.csv`), and the 300-episode bootstrap half-width is +/-3 to +/-8. **`thr9_p10`'s +12.8 / +13.7 and `prop10`'s +25.0 / +18.6 are 2.5x to 5x that.** Everything below roughly +10 is inside the noise and flips between arms -- `thr9_p5` is the clear example, significant head-to-head and not significant in the sweep.

The simplest robust winner is one sentence: **punish 10 whenever a player contributed 9 or less, and nothing otherwise.**

### 3.4 The two accountings rank the managers the same way; a different pair of accountings does not

Rank agreement over the managers, Spearman:

| comparison | sweep (pooled) | head-to-head (pooled) |
|---|---|---|
| common good, env vs corrected | **0.999** | **0.983** |
| group payoff sum, env vs corrected | 0.982 | 0.967 |
| common good vs group payoff sum, env | **-0.486** | **-0.667** |
| common good vs group payoff sum, corrected | -0.468 | -0.700 |

**Correcting the environment's accounting does not change the ranking.** Charging the punishment aimed at timed-out players costs a contribution-keyed rule 0.7 to 3.0 common-good points (`prop10` the most, -2.94), costs `never`, `ah_punisher` and the `_si` twins essentially nothing, and produces exactly one adjacent swap anywhere in the sweep (`thr4_p10` and the human row at s42; `prop10` and `prop10_si` head-to-head).

**The accounting that does flip the ranking is the other one: the common pool against the group payoff sum.** `never` is last-but-one of 21 on the pool and **first** on the payoff sum; `prop10` is first on the pool and 20th on the payoff sum. Against `ah_punisher`, `never` is **-14.89 [-23.7, -5.7]** on the pool and **+17.22 [+13.0, +21.5]** on the payoff sum. The break-evens explain it exactly: the pool needs 0.625 contribution units per punishment point, the payoff sum needs `2 / 0.6 = 3.33` (review S1), and no rule in the sweep buys more than 1.47. So *every* rule pays on the pool and *no* rule pays on the payoff sum.

This matters directly for the expensive question this control was run for: `reward_mode: sum` in both RL configs optimises the group payoff sum. **A manager trained on that reward, against these players, is being pointed at an objective whose optimum in this rule family is "never punish".**

### 3.5 Policy shapes

`policy_shape_h2h_pooled.jpg` puts every manager on one axis (mean punishment against the contribution being responded to, valid player-rounds, bins with n >= 20); the numbers are in the matching CSVs.

* **Real human managers** decline 4.76 -> 0.27 from c = 0 to c = 20, concave, with a step down around c = 10 and near-zero at the maximum. Mean 1.85, P(p>0) = 0.315, mean severity 5.86.
* **`ah_punisher`, our clone, reproduces that shape closely** -- 5.02 -> 0.33, mean 1.87, P(p>0) = 0.308, severity 6.08 -- the three lines (human 4.76 -> 0.27, clone 5.02 -> 0.33, `human_severity` 4.74 -> 0.28) are almost on top of each other in the figure. That is a good sanity check on the clone and it is also the point: the clone is a faithful copy of a policy that is not the best policy here.
* **`human_severity`**, the memoryless rule built from the human `P(p>0|c)` and `E[p|p>0,c]` tables, lands at 116.22 head-to-head -- slightly above the clone but inside the interval. Whatever the punisher's memory and copula buy, it is not common good.
* **The winners look nothing like a human.** `prop10` is the straight line `20 - c`, mean punishment 5.64 (3.0x the human) reaching 20. `thr9_p10` is a step: 10 at c <= 9, 0 above, mean 2.80 (1.5x the human) and it punishes fewer players than a human does (P(p>0) = 0.280 vs 0.315) but three times harder when it does.

### 3.6 Finding D1 in practice: the rules punish every timed-out player, every time

With the timeout serving fix in place a player who gave no input is served contribution 0, so **every contribution-keyed rule punishes 100.0% of timed-out agent-rounds at its maximum severity** -- `thr4_p2` as much as `prop10`. The artificial punisher punishes 0.3% of them (it carries a timeout feature); real human managers punished **0 of 280**.

| | share of timeout cells punished | share of the manager's total punishment spent there |
|---|---|---|
| every threshold / proportional / `human_mean` rule | 1.000 | 2.3% - 10.9% (`thr4_p10` highest) |
| `human_severity` | 0.448 | 4.5% |
| `ah_punisher` | 0.003 | 0.03% |
| human managers (real) | 0.000 | 0% |

Timeouts are 1.9% of simulated agent-rounds (2.9% in the human data), which is why the *accounting* effect is small (section 3.4). The *behavioural* effect is bigger, because the artificial humans are shown that punishment and react to it: the `skip_invalid` twins lose 2.41 (`prop10`: 136.04 -> 133.63) and 6.92 (`thr9_p5`: 121.59 -> 114.67) common-good points. **This changes one conclusion and not the others.** `thr9_p5`'s win over the clone falls from +10.68 [+1.9, +19.3] to +3.73 [-5.1, +12.6] and stops being distinguishable from zero; `prop10`'s survives intact at +22.67 [+12.5, +32.8]. `thr9_p10` was not run with `skip_invalid` -- it spends 6.5% of its punishment on timeout cells, so the same correction would be expected to cost it a few points of its +12.8.

### 3.7 What real managers achieved

Over the 50 human games (single copy, the convention the evaluation suite uses), per round, both groups summed: common good **103.80**, group payoff sum **171.96** on the environment's rule and **179.59** once the timed-out players are paid -- the D2 gap, 4.4% here across all group-rounds and 30.7% restricted to the ones containing a timeout, as the review reported. Mean contribution 9.57, mean punishment 1.85 on rows where both the player and the manager acted.

## 4. Notes

1. **The rules key on the current round, not the previous one.** PR #99's `prev_c_threshold` punished round *t* on round *t-1*'s contribution. Since then the punisher re-baseline (PR #196 and its ancestors) established from the accounting identity that the human manager punished round *t*'s contributions, so every rule here reads `data["contribution"]`. This is not a cosmetic difference: a lagged rule punishes a player who has already recovered, which is a different policy, not a noisier version of the same one.
2. **One config per shard, not one per rule, with `reseed_per_run`.** The simulation seeds once at start-up and then runs its pairings in sequence, so in a multi-pairing config run *k* inherits whatever RNG state runs 1..*k*-1 left behind -- the comparison would partly be a comparison of positions in the file. `reseed_per_run` (opt-in, default off so no existing config changes its numbers) restarts each run from the config's seed. The check that it works is `ah_punisher`: it is the last run of the last shard and must still reproduce the standalone frontier run `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout` exactly.
3. **`skip_invalid` is off by default and that is the measurement, not an oversight.** With the timeout serving fix in place a timed-out player is served contribution 0, so every contribution-keyed rule punishes them at its maximum severity by construction -- and under the env accounting that punishment is free (D1). Leaving the flag off is what makes the rate visible; a `skip_invalid` arm isolates what it costs. Measured: 100.0% of timeout cells hit by every contribution-keyed rule, 2.3-10.9% of a rule's total punishment spent there, worth 2.4 (`prop10`) to 6.9 (`thr9_p5`) common-good points (section 3.6).

4. **`reseed_per_run` does what it can and no more, and the head-to-head arm exists because of what it cannot.** Verified: with it on, every run in a config draws the same validity pattern over all 19,200 agent-rounds and the same episode-0 round-0 contributions; without it the runs simply inherit the previous run's RNG position. But the streams diverge from episode 1 onwards, because a manager that punishes differently makes the simulated players act differently and consume a different number of draws. Worse, `MultiManager` evaluates *every* manager in the config on every round, so a config's stream also depends on its manager **set** -- which is why the three sweep shards are not stream-comparable with each other and why the same manager moves by up to 9 points between the sweep and the head-to-head arm. The fix used here is design, not code: put the managers whose difference carries a claim in one config, and only make claims that both arms support. Pairing the draws properly would need a per-model RNG stream keyed on the episode, which is a change to the simulation protocol and out of scope for this branch.

5. **The top of the ranking is model extrapolation and should be read as such.** `prop10` punishes with a mean of 5.64 and a maximum of 20, and drives mean contribution to 14.36 -- 50% above anything the human games produced (9.57). Review S2 measured that only **4.49%** of the contribution model's training rows follow a punishment above 10 and **1.50%** follow one above 20. So the claim "`prop10` produces 136 of common good" is a claim about the model's behaviour far outside its evidence, not a claim about people. `thr9_p10` (mean 2.80, severity 10) sits at the edge of that evidence; `thr9_p5` (mean 1.84, severity 5) is squarely inside it, which is why it is the rule to quote when the question is about people rather than about the model.

6. **The clone is faithful and that is the point.** `ah_punisher` reproduces the human policy's shape, rate and severity to within a few percent (section 3.5) and produces more common good than the real managers did (111.06 against 103.80 -- different contributors, so a reference not a contest). It is nonetheless beaten by two one-line rules. Copying the average human manager is therefore not a strong baseline for the common good; it is a strong baseline for *being human*.

7. **What this predicts about the RL run, before it happens.** Under `reward_mode: sum` the reward is the group payoff sum, where *no* rule in this family pays for punishing and `never` is the single best policy by +17.2 [+13.0, +21.5] over the clone (section 3.4). If a manager trained on that reward converges to punishing very little, that is the reward specification working as written, not a training failure -- and it will look like a failure against `CLAUDE.md`'s stated objective, the common pool, where the same policy is last. The two can be told apart in one plot: the trained manager's realised mean punishment against `never`'s 0 and the clone's 1.87.

### Successor

The obvious next steps, in the order they are worth doing:

1. **Settle the reward specification before the training run, not after.** The decision is now a measured one: the pool and the payoff sum disagree at Spearman -0.49 to -0.67 over 21 managers, and they disagree about the sign of punishing. Review S1 raised it as a question; this branch makes it a quantified one.
2. **Give the RL manager these rules as a floor, not the clone.** The bar for "the trained manager learned something" should be `thr9_p10` at 123.8 / 126.8 common good, not `ah_punisher` at 111.1 / 113.2 and not `never` at 99.6 / 98.2. A trained manager that does not clear a one-line threshold rule has not earned its training budget.
3. **Fix D1 at the source rather than per-manager.** `skip_invalid` is a property of my rule class; the env still lets *any* manager -- including a trained one -- punish a timed-out player for free, and the RL manager observes `contribution_valid`, so unlike a rule it can learn to exploit it deliberately. The review's shape of the fix (zero the action at `~contribution_valid` inside `punish`) is one line and would close it for every manager at once. It is not done here because it is a change to shared env behaviour and belongs in its own experiment.
4. **Widen the rule family only where the evidence is.** The interesting unexplored direction is not more severity -- that is already outside the training distribution -- but *conditioning*: rules keyed on the group mean, on the player's own trend, or on the round number. The flat 0.83-1.47 range of contribution bought per punishment point across this whole family suggests the lever is who you punish, not how hard.
5. **Nothing here needs a retrain, and nothing here was one.** If the conclusions are to be checked, the cheapest check is more seeds on the head-to-head config: it is a 10-minute GPU job.
