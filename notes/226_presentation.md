# Presentation material (#226)

Raw material for the presentation artifact. Levin's stack only
(`24_LEVIN_vnode_skip_timeoutpun`).

## Story

1. **Why the autoresearch campaign started.** The artificial human stack
   (contribution, switch, punishment) was exploitable by a no-punishment policy, which
   kept RL training from finding intelligible policies other than "don't punish". This
   was documented with head-to-head matchups between the AH manager and the
   k-parameterized rule-based manager, above all the rule-based manager at different
   leniencies: none of them could beat the zero-punishment dummy on our win rate.
2. **Our win rate:** in each episode, the group whose payoff summed over its members,
   averaged over the 24 rounds, is higher wins; the win rate is the share of episodes
   won.
3. **The claim to test.** In our last meeting with Levin it was said that the optimized
   stack is no longer exploitable, so we set out to prove it. With our win rate we
   could not (section 3).
4. **Levin's definition of a win is different.** Each manager plays against the AH
   manager; its group's common pool is compared with what the zero punisher's group
   gets against the AH manager in a separate run (section 6).
5. **Its weakness:** it only compares how different strategies do against the AH model
   and picks the winner by who did better against it, so there are no head-to-heads.
6. **Even under that definition no winner can be declared.** In the opposite-group-AH
   table the rules have positive margins and larger groups than the AH control, but
   the results are not decisive (section 6).
7. **Head to head with margins, the harsh rules and AH clearly lose; the lenient rules don't win either** (section 7).

## Win criteria

| Criterion | Score per group | Who is compared | Decision |
|---|---|---|---|
| Our win rate | Payoff sum over the members per round (`20·members + 0.6·Σc − Σp`, empty group = 0), averaged over the episode | The two managers in the same episode | Share of episodes won |
| Per-capita contribution (#108, #116) | Mean contribution per member per round (empty group = 0), averaged over the episode | The two managers in the same episode | Share of episodes won |
| Per-capita common good (#108, #116) | Mean common good per member per round (empty group = 0), averaged over the episode | The two managers in the same episode | Share of episodes won |
| Levin's (#219) | Common pool per round (`1.6·Σc − Σp`, empty group = 0), averaged over the episode | A manager's group against AH vs zero's group against AH, separate runs | Mean margin, 95% bootstrap interval |
| Head to head with margin | Common pool, as in Levin's | The manager's group vs the zero group in the same episode | Win % and mean margin, 95% bootstrap interval |

## 1. Levin's stack on the evaluation suite (PR #225)

Seeds 42–46, scored with the #225 evaluation code (mean ± sd over seeds; lower is
better, ≤ 1 = at the human noise ceiling).

| Row | Levin's stack |
|---|---|
| CB: contribution over rounds | 1.02 ± 0.32 |
| CD: contribution distribution | 0.96 ± 0.32 |
| CF: share at 0 and 20 per round | 1.00 ± 0.23 |
| CG: spread between groups | 1.33 ± 0.35 |
| SC: switching | 1.31 ± 0.35 |
| PD: punishment within groups | 0.79 ± 0.07 |
| RCE: response to punishment | 0.93 ± 0.04 |
| **22-row mean** | **1.07 ± 0.06** |
| rows ≤ 1 | 13.4 |
| mean contribution (human 9.46) | 9.96 ± 0.60 |
| players at 20, last third (human 17%) | 22% |

## 2. Reproduction of Levin's stack

| Check | Result |
|---|---|
| 22-row mean per seed, this branch vs `dev` (seeds 42–46) | 1.051, 1.050, 1.173, 1.034, 1.059 on both: identical |
| Linear punishers retrained from the configs | identical |
| GNN models retrained from the configs | differences at GPU-noise level |

## 3. Win rates: payoff sum, head to head

Per group and round: sum of the members' payoffs (`20·members + 0.6·Σc − Σp`), empty
group = 0; averaged over the episode's 24 rounds; the higher group wins the episode.
Seed 42, 200 episodes per matchup (both positions pooled), ±3.5 points of noise.

### The rules against AH

| Policy | Policy win % | Policy / AH mean |
|---|---|---|
| rule k1 | 34.5 | 77.6 / 104.8 |
| rule k2 | 50.5 | 95.8 / 92.1 |
| rule k4 | 58.0 | 104.0 / 85.6 |
| rule k8 | 62.5 | 109.4 / 83.2 |

### AH and the rules against zero

| Policy | Policy win % | Policy / zero mean |
|---|---|---|
| AH | 35.0 | 80.7 / 112.8 |
| rule k1 | 27.0 | 69.2 / 121.8 |
| rule k2 | 32.0 | 76.4 / 119.2 |
| rule k4 | 48.0 | 95.7 / 107.1 |
| rule k8 | 42.5 | 93.6 / 107.3 |

## 4. Win rates: per-capita contribution, head to head

Per group and round: mean contribution per member, empty group = 0; averaged over the
episode's 24 rounds; the higher group wins the episode. Seed 42, 200 episodes per
matchup (both positions pooled).

### The rules against AH

| Policy | Policy win % | Policy / AH mean |
|---|---|---|
| rule k1 | 48.5 | 9.84 / 10.03 |
| rule k2 | 51.5 | 9.41 / 9.15 |
| rule k4 | 49.0 | 8.69 / 8.64 |
| rule k8 | 50.0 | 8.24 / 8.51 |

### AH and the rules against zero

| Policy | Policy win % | Policy / zero mean |
|---|---|---|
| AH | 49.0 | 8.38 / 8.07 |
| rule k1 | 48.0 | 8.92 / 8.67 |
| rule k2 | 46.0 | 8.13 / 8.13 |
| rule k4 | 59.5 | 8.87 / 8.29 |
| rule k8 | 45.5 | 7.51 / 7.84 |

## 5. Win rates: per-capita common good, head to head

Per group and round: mean common good per member, empty group = 0; averaged over the
episode's 24 rounds; the higher group wins the episode. Seed 42, 200 episodes per
matchup (both positions pooled).

### The rules against AH

| Policy | Policy win % | Policy / AH mean |
|---|---|---|
| rule k1 | 44.5 | 13.47 / 14.69 |
| rule k2 | 54.0 | 14.02 / 13.14 |
| rule k4 | 54.0 | 13.45 / 12.45 |
| rule k8 | 55.5 | 13.18 / 12.30 |

### AH and the rules against zero

| Policy | Policy win % | Policy / zero mean |
|---|---|---|
| AH | 41.0 | 11.97 / 13.12 |
| rule k1 | 37.0 | 11.94 / 14.15 |
| rule k2 | 37.5 | 11.85 / 13.26 |
| rule k4 | 53.0 | 13.83 / 13.49 |
| rule k8 | 44.0 | 11.93 / 12.75 |

## 6. Levin's criterion: common pool, opposite group = AH (as in #219)

Per group and round: common pool `1.6·Σc − Σp`, empty group = 0; averaged over the
episode's rounds. AH holds the opposite group in every pairing; each manager's group
is compared with zero's group against AH (separate pairings). Margin = group minus
zero's group, 95% bootstrap over episodes.

| Manager | Pool / members | vs zero's group |
|---|---|---|
| zero (reference) | 60.9 / 4.45 | — |
| AH (control) | 53.4 / 4.00 | −7.5 [−15.4, +0.7] |
| rule k1 | 56.1 / 3.50 | −4.8 [−13.5, +3.8] |
| rule k2 | 63.5 / 3.95 | +2.6 [−6.7, +12.0] |
| rule k4 | 61.4 / 4.22 | +0.4 [−8.0, +8.4] |
| rule k8 | 61.2 / 4.34 | +0.3 [−8.3, +8.6] |

## 7. Common pool, head to head against zero

Same pool; each manager's group against the zero group in the same episode (paired by
episode). Margin with 95% bootstrap over episodes.

| Manager | Win % | Margin |
|---|---|---|
| AH | 39.0 | −10.9 [−19.6, −2.1] |
| rule k1 | 32.0 | −20.8 [−30.5, −11.3] |
| rule k2 | 35.0 | −18.3 [−27.4, −8.9] |
| rule k4 | 48.0 | −2.3 [−11.7, +7.1] |
| rule k8 | 45.0 | −6.8 [−15.6, +2.3] |
