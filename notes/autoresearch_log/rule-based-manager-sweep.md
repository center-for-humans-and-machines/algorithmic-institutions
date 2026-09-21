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
| 3 | Run the 19 rules plus the artificial punisher against the frontier stack, seed 42, standard protocol. | pending |
| 4 | Rank on both accountings; policy shapes; timeout-cell exploitation rates. | pending |
| 5 | Re-run the top few managers on further seeds; judge every margin against that spread. | pending |
| 6 | Log, PR. | pending |

## 3. Results

(pending)

## 4. Notes

1. **The rules key on the current round, not the previous one.** PR #99's `prev_c_threshold` punished round *t* on round *t-1*'s contribution. Since then the punisher re-baseline (PR #196 and its ancestors) established from the accounting identity that the human manager punished round *t*'s contributions, so every rule here reads `data["contribution"]`. This is not a cosmetic difference: a lagged rule punishes a player who has already recovered, which is a different policy, not a noisier version of the same one.
2. **One config per shard, not one per rule, with `reseed_per_run`.** The simulation seeds once at start-up and then runs its pairings in sequence, so in a multi-pairing config run *k* inherits whatever RNG state runs 1..*k*-1 left behind -- the comparison would partly be a comparison of positions in the file. `reseed_per_run` (opt-in, default off so no existing config changes its numbers) restarts each run from the config's seed. The check that it works is `ah_punisher`: it is the last run of the last shard and must still reproduce the standalone frontier run `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout` exactly.
3. **`skip_invalid` is off by default and that is the measurement, not an oversight.** With the timeout serving fix in place a timed-out player is served contribution 0, so every contribution-keyed rule punishes them at its maximum severity by construction -- and under the env accounting that punishment is free (D1). Leaving the flag off is what makes the rate visible; a `skip_invalid` arm isolates what it costs.
