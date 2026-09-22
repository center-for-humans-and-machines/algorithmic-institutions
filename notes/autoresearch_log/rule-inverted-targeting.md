# Does punishing the WRONG people cost anything in this world?

## 1. Declaration

**Slot:** none. Nothing is trained, nothing is recalibrated, no artifact changes. Like both its ancestors this is a measurement branch: it adds one rule to an existing family and runs it beside the rules already there.

**Parent:** `auto/rule-vs-clone-paired` at `d1ca18c` (PR [#209](https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/209)), itself on `auto/rule-based-manager-sweep` (PR [#207](https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/207)). Branch `auto/rule-inverted-targeting`, PR against `auto/rule-vs-clone-paired`. Isolated remote dir `~/repros/ai-runs/rule-inverted` (delete when the PR closes). Byte-identical artifacts, protocol and seed discipline to the parent's paired arm; the four artifact sha256s are printed in every job log.

The parent branch is the base rather than the sweep because it is a strict descendant of it (`git merge-base` is the sweep's tip, `4c0ef83`), so it carries both the rule family and the paired harness. Basing on the sweep would have meant re-deriving the harness.

### The question

Learned RL managers in this project come out **inverted**: they punish full contributors hardest and leave free-riders alone, the reverse of the human policy. An intervention probe on the contributor model (PR [#215](https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/215)) established two things about the *model*: the targeting signal is real, correctly signed and survives retraining, but it is about a fifth of the human gradient (0.066 against 0.301); and at the ceiling, where two of three learned seeds actually aim, the response is close to inert -- punishing a full contributor moves its next contribution by **-0.246** against the human **-7.035**, about 3.5%.

From that it is tempting to conclude that punishing the wrong people is nearly free. **But that is a claim about consequences, and the probe cannot make it.** The probe forces one punishment and reads one contribution response, holding everything else fixed. A manager's consequence in this world runs through the whole closed loop: the contribution response, the payoff charged for punishment, and -- the quantity the parent branch showed dominates -- **whether the members stay**. A per-cell response of -0.25 says nothing about whether a group governed that way holds together.

This branch tests the claim by simulating an inverted manager.

### Why the competing setting, not self-play

The parent established that self-play rankings do not survive competition. `prop10` led by +12.5 pool points per seat in self-play and came out at **-15.6** [-23.1, -8.0] against a live rival, because punishing bleeds members when there is somewhere better to go. Self-play makes group size meaningless -- both seats carry the same policy, so the split is noise -- and group size is exactly the channel through which an inverted rule would be expected to hurt. Running this arm in self-play would answer a question nobody is asking.

So: the paired harness, one rule and one rival per world, one group each, members free to move every fourth round.

### Choosing the mirror, and why the suggested name needed checking

The suggestion this arm inherited was `inv_thr11_p10`: punish 10 whenever a player contributed 11 or more, as the mirror of `thr9_p10`, punish 10 whenever a player contributed 9 or less. The name is defensible but it is not the only mirror, and the two senses of "mirror" disagree here.

`thr9_p10` fires on c in {0..9}: **ten of the twenty-one levels**.

* **Matched on levels.** The exact reflection of `thr9_p10` under c -> 20 - c fires on c in {11..20}: ten levels, the same amount 10, and both rules spare the midpoint c = 10. That is `inv_thr11_p10`, and on the definition it is as faithful a mirror as exists.
* **Matched on spend.** Contributions in this world are **massed low** -- the untreated distribution (`never_vs_never`, three seeds, 57,600 agent-rounds) has mean 8.73. So the two halves of the axis are not equally populated, and the level-matched mirror fires on far fewer cells: **P(c >= 11) = 0.342** against **P(c <= 9) = 0.565**. It under-spends by about 40%. A rule that punishes 40% less often is not a clean test of *direction*; if it did less damage, one could not tell whether that was the direction or the lower intensity.

The spend match is available and almost exact. **P(c >= 7) = 0.567** against `thr9_p10`'s **0.565** -- a gap of 0.15 percentage points -- and the match survives restricting to cells where the player gave an input (0.558 against 0.557). That is `inv_thr7_p10`. It buys the intensity match by giving up the level match: fourteen levels, not ten.

**Both are run**, because neither criterion dominates and the disagreement is informative:

| rule | fires on | levels | P(fire) on the untreated distribution |
|---|---|---|---|
| `thr9_p10` (correctly targeted) | c <= 9 | 10 of 21 | 0.565 |
| `inv_thr11_p10` (inverted, level-matched) | c >= 11 | 10 of 21 | 0.342 |
| `inv_thr7_p10` (inverted, spend-matched) | c >= 7 | 14 of 21 | 0.567 |

If the two inverted rules agree, the reading does not depend on which sense of "mirror" is used. The thresholds were read off the parent arm's own control, ex ante, not tuned to the answer.

### A separate hypothesis the broad mirrors cannot test: the near-ceiling band

The maintainer raised a distinct idea that deserves its own seat rather than an argument: **punishing nearly-full contributors may be a valid strategy, because they are close to the ceiling and therefore cheap to push the rest of the way up.**

`inv_thr11_p10` does not test that. It fires on everyone from 11 upward, which spans the entire withdrawal zone, so it mixes the maintainer's idea with ordinary indiscriminate over-punishment. A rule that fires **only on the near-ceiling band** does test it.

`band16_*` fires on c >= 16 and leaves everyone below untouched -- five of the twenty-one levels. But only **0.186** of cells sit at c >= 16 in this distribution, so a narrow band cannot both fire gently and spend as much as a broad rule. Rather than pick one horn, the band is run at **two intensities**:

| rule | fires on | levels | P(fire) untreated | untreated spend | matched to |
|---|---|---|---|---|---|
| `band16_p10` | c >= 16, punish 10 | 5 of 21 | 0.186 | **1.86** | the clone's realised 1.9-2.0 |
| `band16_p20` | c >= 16, punish 20 | 5 of 21 | 0.186 | **3.72** | `thr9_p10`'s realised 3.71 in the parent arm |

Together they separate the **band** from the **intensity** inside the near-ceiling family, which is what decides whether amount dominates direction. `band16_p10` is the "mild wrong-direction punishment" case; `band16_p20` buys the spend match by punishing each hit twice as hard.

**Why this is worth the extra seats.** The intervention surface says the contributor response flips sign near contribution 12, so punishment above that should produce withdrawal rather than compliance (-0.246 at c = 20 for the model, -7.035 for humans). On that evidence the hypothesis should fail. But the learned seeds complicate it: seed 44 is inverted and reached 14.14 common good, second only to the good rule and above never-punishing at 12.78, while seed 42 is inverted much harder and finished last at 10.74. Mild wrong-direction punishment did fine and severe wrong-direction punishment did badly, which suggests amount may dominate direction. **If the near-ceiling rule performs comparably to the correctly-targeted one, the individual-level withdrawal result and the group-level outcome disagree, and that disagreement is the finding.** It is reported either way.

### The five focal rules

| rule | fires on | family |
|---|---|---|
| `thr9_p10` | c <= 9, punish 10 | correctly targeted |
| `inv_thr11_p10` | c >= 11, punish 10 | broad inverted, level-matched |
| `inv_thr7_p10` | c >= 7, punish 10 | broad inverted, spend-matched |
| `band16_p10` | c >= 16, punish 10 | near-ceiling, clone intensity |
| `band16_p20` | c >= 16, punish 20 | near-ceiling, spend-matched |

**One asymmetry cannot be designed away and is reported rather than corrected.** The env serves a timed-out player contribution 0, so at the rule's input a timeout is indistinguishable from a total free-rider. A low-threshold rule therefore punishes **every** timed-out player and an inverted rule can **never** reach one. `auto/free-punishment-fix` closes this at the env and had not landed when this ran, exactly as in both ancestors. Section 3.6 quantifies it; it works against the correctly-targeted rule, not against the inverted ones, so it cannot manufacture the result below.

### The pairings

Fourteen per seed, one config per seed, one manager set (the parent's note 4: `MultiManager` evaluates every manager in a file each round, so a run's RNG consumption depends on the file's manager *set*). `group_0` carries the focal manager, `group_1` the rival.

| family | pairings |
|---|---|
| controls | `ah_punisher_vs_ah_punisher`, `never_vs_never` |
| vs the clone | `thr9_p10`, `inv_thr11_p10`, `inv_thr7_p10`, `band16_p10`, `band16_p20`, `never` -- each against `ah_punisher` |
| vs never-punish | `thr9_p10`, `inv_thr11_p10`, `inv_thr7_p10`, `band16_p10`, `band16_p20`, `ah_punisher` -- each against `never` |

**`thr9_p10` and `never` are re-run here rather than quoted from the parent.** Because the RNG a run consumes depends on the config's manager set, and this config's set differs from the parent's, the parent's numbers for those two are not stream-comparable with these. Re-running them puts all five focal rules in one set of simulations where they are.

### The two accountings

As both ancestors. Every number is recomputed from contributions and punishments; neither the env's `common_good` state field (the per-capita **share**) nor the human data's `common_good` column (the undivided **pool**) is read, since the two names collide.

* **env** -- punishment aimed at a timed-out player is dropped, so it is free; that player's own payoff is discarded.
* **corrected** -- that punishment is charged to the pool, because the artificial humans were shown it and reacted to it.

Every margin below is on the **corrected** accounting.

### Artifact naming contract

| what | path |
|---|---|
| the rule | `RuleBasedManager(rule="inv_threshold", ...)` in `src/aimanager/manager/api_manager.py` |
| config generator | `scripts/data_analysis/rule_manager_configs.py --inverted` |
| sim configs | `configs/simulation/manager_testing/26_rule_inverted_targeting_s{42,43,44}.yml` |
| job template | `scripts/simulate_config_iso.slurm <config>` |
| analysis | `scripts/data_analysis/rule_inverted_targeting_report.py` |
| tables and figures | `plots/data_analysis/evaluation/rule_inverted_targeting/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Add `inv_threshold` to the rule family without disturbing the existing rules; pin it with tests. | **done** -- `--paired` regenerates byte-identically; 14/14 tests pass on Raven |
| 2 | Choose the mirrors from the parent's own untreated distribution, ex ante, and run both senses of "mirror". | **done** -- section 1 |
| 3 | Add the near-ceiling band at two intensities, so the maintainer's hypothesis is tested rather than argued. | **done** -- section 1 |
| 4 | Run seeds 42/43/44 in the paired setting with `thr9_p10` and `never` in the same manager set. | **done** -- section 3.1 |
| 5 | Report the realised spend of every rule so a reader can check the matching. | **done** -- section 3.2 |
| 6 | Separate the cost of punishing from the cost of punishing the wrong people. | **done** -- section 3.4 |
| 7 | Report the near-ceiling result plainly, whichever way it falls. | **done** -- section 3.5 |
| 8 | Log, PR. | **done** |

## 3. Results

**The headline: the inversion is heavily penalised in this world. Punishing the wrong people is not close to free.** The level-matched inverted rule loses **-32.75 pool points** [-38.14, -27.40] against never-punishing in the same seat, while the correctly-targeted rule loses **+0.10** [-7.07, +7.45] -- indistinguishable from never. That is 10.5x the median seed spread.

### 3.1 What was run (measured)

Three simulations, all `COMPLETED`, on the frontier stack with byte-identical artifacts and the standard protocol (2 groups x 8 agents, 24 rounds, 100 episodes, `save_per_round: true`, `reseed_per_run: true`). Nothing was trained or recalibrated.

| seed | config | job | wall |
|---|---|---|---|
| 42 | `26_rule_inverted_targeting_s42.yml` | 30414875 | 17:59 |
| 43 | `26_rule_inverted_targeting_s43.yml` | 30414876 | 17:57 |
| 44 | `26_rule_inverted_targeting_s44.yml` | 30414877 | 17:59 |

Each file is 14 pairings x 100 episodes x 24 rounds x 8 agents = 268,800 agent-rounds. Every table below **pools the three seeds** (300 episodes per pairing) and every interval is a 95% bootstrap over episodes.

An earlier submission (30414716/17/18) was cancelled by this arm at 11:26:08 and resubmitted at 11:27:24, after the near-ceiling rules were added; `sacct` shows all three cancelled in the same second under this arm's own job name, in this arm's own directory. Nothing external cancelled them, and the completed jobs ran the full 14-pairing, five-focal config.

**The dispatch check passes 19/19 seats**, on two separate tests. Each seat carries its manager's signature on cells where the player gave an input -- a `never` seat punishes only 0, a threshold seat only 0 or its amount and *every* cell on its side of the cut. And each seat shows the *expected timeout behaviour*: `thr9_p10` punishes **1.000** of its timed-out cells, every inverted rule punishes **0.000** of them. Static dispatch, or dispatch keyed on the initial rather than the current group, could not produce this.

That second test is worth stating because it caught a real discrepancy rather than papering over one. `per_round.parquet` records a timed-out player's **own** contribution while the manager was served **0**, so the recorded contribution and the rule's input disagree on exactly those cells (1.9% of them). The parent's report already restricted its `prop10` check to valid cells for the same reason; here the disagreement is additionally checked in its own right, because it is the one asymmetry between the two directions that cannot be designed away.

**The seed-to-seed spread, measured here, is the yardstick** (`seed_spread.csv`, sd of the three per-seed 100-episode means): group size **0.22** members (max 0.49), common pool **3.13** (max 9.38). Close to the parent's 0.16 and 3.15.

**The symmetric controls.** With the same manager on both sides, focal minus rival:

| control | group size | common pool |
|---|---|---|
| `ah_punisher_vs_ah_punisher` | +0.08 [-0.32, +0.46] | +0.99 [-7.14, +9.01] |
| `never_vs_never` | -0.33 [-0.72, +0.05] | **-7.69** [-14.06, -1.31] |

Group size shows no seat effect. The pool control in `never_vs_never` does exclude zero at -7.7, slightly larger than the parent's -6.11 [-13.21, +0.94]. **So a pool margin smaller than about 8 points should not be read as the rule here.** Every margin this log rests on is between 21 and 42 points, so none of them is at risk; but the correctly-targeted rule's +0.10 and +3.75 are *inside* that band, and are reported as "indistinguishable from never", not as a win.

### 3.2 Is the mirror fair? The realised spend of every rule (measured)

Realised mean punishment per member-round on the focal seat, cells where the player gave an input (`mirror_match_pooled.csv`). The ex-ante column is what each rule would fire on the untreated distribution; the realised columns are what it actually spent, and they differ because a rule moves the contributions it then reads.

| rule | levels fired | P(fire) ex ante | realised mean p, vs clone | realised mean p, vs never |
|---|---|---|---|---|
| `thr9_p10` (correct) | 10 of 21 | 0.634 | **2.80** | **3.26** |
| `inv_thr11_p10` (inverted, level-matched) | 10 of 21 | 0.280 | **2.17** | **1.98** |
| `inv_thr7_p10` (inverted, spend-matched) | 14 of 21 | 0.487 | **4.49** | **4.40** |
| `band16_p10` (near-ceiling, mild) | 5 of 21 | 0.144 | **1.07** | **1.14** |
| `band16_p20` (near-ceiling, matched) | 5 of 21 | 0.144 | **1.62** | **1.73** |
| `ah_punisher` (the clone, reference) | -- | -- | ~1.9 | ~2.0 |

**The ex-ante matching did not survive contact with the closed loop, and it did not need to.** `inv_thr7_p10` was chosen to match `thr9_p10`'s firing rate and ended up spending 60% more; `inv_thr11_p10` was chosen to under-spend and ended up spending 22% *less*. The reason is visible in the contribution column of section 3.3: `thr9_p10` drives mean contribution up to 11.5, which lifts players out of its own firing zone, while the inverted rules drive it down to 6.8-8.5, which lifts players out of *theirs*.

What matters is that **the two broad inverted rules bracket the correct rule on realised spend** (2.17 and 4.49 against 2.80), so no reading below depends on an intensity advantage. In particular `inv_thr11_p10` punishes **less** than `thr9_p10` and still loses 32.8 pool points to it.

### 3.3 The levels: what each focal seat holds and produces (measured)

Focal seat, pooled, per round, corrected accounting (`levels_pooled.csv`).

**Rival = the clone (`ah_punisher`):**

| focal | mean p | members (of 8) | common good, group total | common good per member | mean c |
|---|---|---|---|---|---|
| `never` | 0 | **4.57** [4.39, 4.75] | **62.22** [57.19, 67.01] | 12.82 [12.00, 13.64] | 8.48 |
| `thr9_p10` | 2.80 | 3.92 [3.73, 4.12] | **62.32** [56.85, 67.69] | **14.02** [13.09, 14.97] | 11.48 |
| `band16_p10` | 1.07 | 4.04 [3.85, 4.23] | 40.51 [37.87, 43.34] | 9.68 [9.17, 10.22] | 7.78 |
| `band16_p20` | 1.62 | 3.78 [3.59, 3.98] | 32.29 [30.03, 34.75] | 7.91 [7.50, 8.33] | 7.12 |
| `inv_thr11_p10` | 2.17 | 3.55 [3.36, 3.75] | 29.51 [27.60, 31.62] | 7.96 [7.51, 8.45] | 7.72 |
| `inv_thr7_p10` | 4.49 | **2.78** [2.62, 2.93] | **20.75** [19.24, 22.41] | 6.96 [6.49, 7.48] | 8.49 |

**Rival = never-punish:**

| focal | mean p | members (of 8) | common good, group total | common good per member | mean c |
|---|---|---|---|---|---|
| `never` (control) | 0 | 3.83 [3.64, 4.03] | 44.40 [40.57, 48.41] | 10.49 [9.79, 11.21] | 7.29 |
| `thr9_p10` | 3.26 | 3.36 [3.16, 3.55] | **48.13** [42.94, 53.36] | **11.53** [10.57, 12.49] | 10.44 |
| `band16_p10` | 1.14 | 3.46 [3.28, 3.64] | 35.43 [32.80, 38.15] | 9.28 [8.76, 9.85] | 7.47 |
| `band16_p20` | 1.73 | 3.07 [2.90, 3.24] | 27.10 [24.96, 29.24] | 7.61 [7.18, 8.04] | 7.00 |
| `inv_thr11_p10` | 1.98 | 2.81 [2.65, 2.96] | 21.83 [20.25, 23.48] | 6.95 [6.54, 7.39] | 6.83 |
| `inv_thr7_p10` | 4.40 | **2.24** [2.12, 2.36] | **17.26** [15.99, 18.69] | 6.58 [6.09, 7.08] | 8.34 |

**Every inverted rule loses on both quantities at once** -- fewer members *and* less common good per member each. That is the difference from the parent branch, where `prop10` lost members but raised its stayers' contributions to the highest figure in either log. Here there is no compensating gain to point at: the inverted seats hold fewer people who contribute less.

### 3.4 The decision: direction, separated from amount (measured)

Two contrasts, both on the focal seat with the rival and the seat held fixed (`decision_pooled.csv`). **The never-punish contrast carries both the cost of punishing and the cost of mis-targeting; only the `thr9_p10` contrast isolates direction.**

**Contrast 1 -- focal seat MINUS the never-punish seat. Is the inversion penalised at all?**

| focal | rival = clone | rival = never |
|---|---|---|
| `thr9_p10` (correct) | **+0.10** [-7.07, +7.45] | **+3.75** [-2.81, +10.30] |
| `band16_p10` | **-21.71** [-27.38, -16.16] | **-8.99** [-13.72, -4.36] |
| `band16_p20` | **-29.94** [-35.55, -24.32] | **-17.31** [-21.68, -13.02] |
| `inv_thr11_p10` | **-32.75** [-38.14, -27.40] | **-22.56** [-26.81, -18.57] |
| `inv_thr7_p10` | **-41.50** [-46.82, -36.25] | **-27.14** [-31.28, -23.17] |

**This is the answer to the question the arm was run to settle.** The inverted rule loses substantially to never-punishing -- 22 to 42 pool points, 7x to 13x the median seed spread and 2.3x to 4.5x the max, every interval far clear of the -7.7 seat control. The correctly-targeted rule, by contrast, is indistinguishable from never-punishing. **Punishing the wrong people is not approximately free; it is the single most expensive thing a manager does in this world.**

**Contrast 2 -- focal seat MINUS the `thr9_p10` seat. Does direction matter, holding the amount roughly fixed?**

| focal | realised p vs `thr9_p10`'s | rival = clone | rival = never |
|---|---|---|---|
| `band16_p10` | 1.07 vs 2.80 (**less**) | **-21.81** [-28.18, -15.74] | **-12.70** [-18.58, -6.95] |
| `band16_p20` | 1.62 vs 2.80 (**less**) | **-30.04** [-36.08, -24.13] | **-21.01** [-26.58, -15.53] |
| `inv_thr11_p10` | 2.17 vs 2.80 (**less**) | **-32.84** [-38.69, -27.10] | **-26.27** [-31.80, -20.85] |
| `inv_thr7_p10` | 4.49 vs 2.80 (more) | **-41.59** [-47.43, -36.13] | **-30.85** [-36.23, -25.50] |

**Direction is not explained by amount.** Three of the four wrong-direction rules punish *less* than `thr9_p10` and every one of them loses 12 to 33 pool points to it. `inv_thr11_p10` is the clean case: it fires on the same number of levels at the same amount, spends 22% less in realisation, and still ends 32.8 points below.

Amount does matter *within* the wrong direction -- the four rules order monotonically by spend, 40.5 / 32.3 / 29.5 / 20.8 against the clone -- but the ordering runs the wrong way and extrapolates to `never`. **Inside the wrong-direction family the best attainable policy is to punish nothing at all.** In the right direction, spending 2.80 buys back everything it costs.

### 3.5 The near-ceiling hypothesis was tested and did not hold (measured)

The hypothesis: punishing nearly-full contributors may be a valid strategy, because they are close to the ceiling and so cheap to push the rest of the way up.

**It fails, on its own terms and at its own chosen intensity.**

* `band16_p10` -- the mild near-ceiling rule, realised spend **1.07**, *below* the clone's 1.9-2.0 -- produces **40.51** [37.87, 43.34] against `thr9_p10`'s 62.32 and never's 62.22. It loses **-21.81** [-28.18, -15.74] to the correctly-targeted rule and **-21.71** [-27.38, -16.16] to never-punishing.
* `band16_p20`, at spend 1.62, is worse still: 32.29, losing -30.04 and -29.94.

**The manner of the failure is the informative part, and it is not the one the parent branch would predict.** `band16_p10` does **not** lose members: 4.04 [3.85, 4.23] against `thr9_p10`'s 3.92, a difference of **+0.11** [-0.15, +0.38] -- indistinguishable, and above the 4-of-8 start. It keeps its people. What it loses is what those people produce: **9.68 common good per member against 14.02**, a gap of **-4.34** [-5.44, -3.22], and mean contribution **7.78 against 11.48**.

So the near-ceiling rule is not punished by the migration channel that dominated the parent branch. It is punished by the contribution response directly: it pushes its near-ceiling contributors *down*, not up. Mean contribution under it (7.78) sits **below** never-punishing's (8.48), which is the sharpest single statement of the result -- a manager that punishes only the best contributors, gently, ends up with a group that contributes less than one that does nothing at all.

**The individual-level measurement and the group-level outcome agree, so there is nothing to reconcile.** The intervention surface says the contributor response flips sign near contribution 12, so punishment above that produces withdrawal rather than compliance. That is exactly what the group-level run shows, and the two are consistent. Had they disagreed, the disagreement would have been the finding; they do not.

This also does not support the "amount dominates direction" reading the learned seeds suggested. `band16_p10` **is** the mild wrong-direction case -- milder than our clone of a human manager -- and mildness did not rescue it. At matched or lower intensity, direction still costs about 22 pool points.

### 3.6 The mechanism: the inverted rule sheds its contributors and keeps its free-riders (measured)

Comparing the members who leave a seat with those who stay, at the round the move is decided (`who_leaves_pooled.csv`, 3,900-8,100 decisions per seat). `c_gap` is leavers' mean contribution minus stayers'.

| seat's manager | leave rate | c, leavers vs stayers | **c_gap** | p_gap |
|---|---|---|---|---|
| `thr9_p10` (vs clone) | 0.26 | 9.59 vs 13.10 | **-3.51** | +2.64 |
| `ah_punisher`, the clone (control) | 0.24 | 8.45 vs 10.81 | **-2.35** | +1.87 |
| `never` (control) | 0.23 | 6.27 vs 7.47 | **-1.20** | 0 |
| `band16_p10` (vs clone) | 0.24 | 7.55 vs 6.66 | **+0.89** | +0.74 |
| `band16_p20` (vs clone) | 0.27 | 7.14 vs 5.81 | **+1.33** | +2.00 |
| `inv_thr11_p10` (vs clone) | 0.28 | 7.61 vs 5.92 | **+1.69** | +1.15 |
| `inv_thr7_p10` (vs never) | 0.46 | 8.35 vs 6.51 | **+1.83** | +1.53 |

**The sign of `c_gap` flips with the direction of the rule, in every seat.** Where the manager punishes low contributors -- or does not punish at all -- the members who leave are the ones who contributed *less* than those who stay, so the group that remains improves. Where the manager punishes high contributors, the members who leave are the ones who contributed *more*, so the group that remains degrades.

That is the whole result in one mechanism, and it is a composition effect the parent branch never saw because every rule it ran pointed the right way. An inverted manager does not merely fail to discipline free-riders: it actively selects for them, by driving out precisely the members worth keeping. The punishment gap stays positive in every seat -- the punished always leave -- but *who* gets punished decides whether that selection helps or hurts.

### 3.7 What the rival collects (measured)

The parent's cross-group public good, re-measured. The rival seat, facing this rule, changing nothing itself:

| rival | facing | group size | common pool |
|---|---|---|---|
| `never` | `inv_thr7_p10` | **+1.60** [+1.36, +1.83] | **+21.68** [+14.37, +29.02] |
| the clone | `inv_thr7_p10` | **+1.27** [+1.01, +1.51] | **+16.22** [+8.76, +23.69] |
| `never` | `thr9_p10` | +0.48 [+0.20, +0.75] | **+13.41** [+6.84, +20.11] |
| `never` | `inv_thr11_p10` | **+1.03** [+0.78, +1.27] | +10.77 [+3.51, +17.72] |
| the clone | `band16_p10` | +0.00 [-0.26, +0.28] | -0.97 [-8.18, +6.32] |

The inverted rules export members roughly twice as fast as the correctly-targeted one, which is consistent with section 3.6: the people they drive out are the good contributors, and they are worth more to whoever receives them. `band16_p10` is again the exception -- it exports nobody, and its rival gains nothing -- confirming that its loss is a contribution effect rather than a migration effect.

### 3.8 Free punishment on timed-out cells, carried forward (measured)

`auto/free-punishment-fix` still had not landed. As established in section 3.1, the asymmetry is total: `thr9_p10` punishes 1.000 of its timed-out cells, every inverted rule punishes 0.000 of them.

**This works entirely against the correctly-targeted rule and cannot have manufactured the result.** `thr9_p10` is the only focal here paying for punishment the env then discards, and it is the one that comes out ahead; closing D1 would improve its position and leave every inverted rule's unchanged. The env and corrected accountings differ by 1 to 2 pool points per seat and change no conclusion in this log.

## 4. Notes

1. **Measured against inferred.** Sections 3.1 to 3.8 are measurements: realised spends, group sizes, pools recomputed from contributions and punishments, leave rates and leaver/stayer gaps, all with episode bootstraps and all judged against this branch's own seed spread and seat controls. The *reading* of section 3.6 -- that the inverted rule selects for free-riders -- is an inference from a measured sign flip in `c_gap` across seven seats. It is a strong and consistent inference, but this run does not decompose the resulting pool loss into the composition channel and the direct contribution-response channel, and it should not be quoted as though it did. Section 3.5 bounds them indirectly for `band16_p10`, where membership does not move and the loss must therefore be contribution response.

2. **The ex-ante mirror matching did not survive the closed loop, and the log says so rather than quietly re-deriving it.** The thresholds were chosen from the parent's untreated distribution before the run and are not tuned to the answer. Realised spends came out different (section 3.2). The claim rests on the fact that the two broad inverted rules *bracket* `thr9_p10` on realised spend and both lose heavily, plus the monotone ordering within the wrong-direction family -- not on any single pair being exactly matched.

3. **The correctly-targeted rule is not shown to beat never-punishing here.** +0.10 [-7.07, +7.45] and +3.75 [-2.81, +10.30] are inside the `never_vs_never` seat control's -7.7, so they are reported as indistinguishable. This arm is not evidence that punishing pays; it is evidence that punishing *the wrong people* is very costly. The parent's conclusion -- that in a competitive world punishing barely pays for itself -- stands unchallenged.

4. **`never_vs_never`'s pool control excludes zero here** (-7.69 [-14.06, -1.31]) where the parent's did not (-6.11 [-13.21, +0.94]). The manager set differs, so the RNG stream differs, and this is an ordinary redraw rather than a new finding. It is reported because it sets the floor below which a pool margin should not be read, and because a successor pooling more seeds should watch whether it persists.

5. **One asymmetry between manager types is left in place and is not confounding.** The clone is a `LinearManager` reading the raw round history, so a switcher's `prev_punishment` is what they really received in the other seat. This applies identically in the clone-against-clone control and cancels out of every contrast reported.

6. **The near-ceiling rules are `inv_threshold` at 16, not a new mechanism.** They are named for what they do rather than for the hypothesis they test, and both intensities are reported whichever way they fell. They fell against the hypothesis.

### Successor

1. **The exploration arms are testing the right thing.** The four running arms (`rl-anneal-local`, `rl-bootstrapped-dqn`, `rl-es`, `rl-param-noise`) are trying to fix an inversion that this arm shows the environment prices heavily -- 22 to 42 pool points against never-punishing, 12 to 33 against the correct rule. The learned managers failed at something the environment did reward, so an exploration method that finds the correct targeting has a large prize waiting. Had the answer gone the other way, those arms would have been measuring a difference the world does not price; they are not.

2. **Report `c_gap` for every trained manager.** Section 3.6 gives a single-number diagnostic that separates a correctly-targeted manager from an inverted one without needing a counterfactual: the contribution gap between leavers and stayers on its own seat. It is negative for every right-direction policy measured here and positive for every wrong-direction one. It is cheap, it reads off `per_round.parquet`, and it would have flagged the learned managers' inversion from their own simulation output.

3. **Decompose the two channels that `band16_p10` separates by accident.** It loses 21.8 pool points while holding its members exactly, so its loss is pure contribution response; the broad inverted rules lose members as well. A successor that masks the contribution model's cross-group edges at simulation time (the parent's successor item 4) would bound the graph channel and complete the decomposition this run leaves open.

4. **Do not use this arm as evidence that punishing pays.** See note 3. The honest bar for a trained manager in the competing setting is still the clone's seat and `never`'s seat, with group size reported alongside the pool, exactly as the parent's successor item 1 said.

5. **`auto/free-punishment-fix` is worth landing before any positive claim about a punishing policy**, since it is the correctly-targeted rule that pays the free-punishment bill. It cannot change this log's sign, but it would sharpen `thr9_p10`'s position against `never`.

6. **Nothing here needs a retrain, and nothing here was one.** Three 18-minute GPU jobs reproduce every number in this log.
