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

*(filled in below from the measured run)*
