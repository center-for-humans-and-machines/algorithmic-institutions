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

**Throughput, which is what made the design affordable.** A rollout costs about the same whatever its batch size, because the cost is per-round Python and model-call overhead rather than arithmetic -- the contribution GNN is 35 KB. Measured: 8 rollouts of 768 episodes in 42 seconds on an A100, and 3 seconds for a 192-episode rollout on four CPU threads. `simulate.py` needs about 77 seconds for 100 episodes of one pairing. So the sweep runs on CPU nodes, and a thousand design points at 512 episodes each is an hour of ordinary batch time rather than a GPU campaign.

## 4. Notes

*(to be filled)*

### Successor

*(to be filled)*
