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
| 6 | Report realised spend, policy shape on the evaluation suite's bins beside the human and the clone, and the leaver diagnostic. | |
| 7 | Log, PR. | |

## 3. Results

*(to be filled)*

## 4. Notes

*(to be filled)*

### Successor

*(to be filled)*
