# Can a rule beat our clone of a human manager when they share one world and compete for members?

## 1. Declaration

**Slot:** none. Nothing is trained, nothing is recalibrated, no artifact changes. Like its parent this is a measurement branch: it re-asks the sweep's question in the setting the RL manager will actually be trained in.

**Parent:** `auto/rule-based-manager-sweep` at `4c0ef83` (PR [#207](https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/207)), itself on `auto/sim-timeout-imputation`. Branch `auto/rule-vs-clone-paired`, PR against `auto/rule-based-manager-sweep`. Isolated remote dir `~/repros/ai-runs/rule-vs-clone` (delete when the PR closes). Byte-identical artifacts, protocol and seed discipline to the parent's head-to-head arm.

**`auto/free-punishment-fix` had not landed when this ran.** It is not on `origin`; it exists only in a sibling's local worktree. So this arm runs without it, and reports how much of each rule's punishment lands on timed-out cells, exactly as the sweep did (section 3.6 there), so the result can be corrected later.

### The question

The sweep established that punishing pays and that two one-line rules beat the clone. But `MultiManager` separates managers along the **batch** dimension: every manager in a config was evaluated on every round of its own parallel population, and a pairing's `group_0` and `group_1` carried the *same* manager. Each manager therefore governed its own world and never faced another. Meanwhile in training the learner holds one group while an opponent holds the other, and members are free to move between them every fourth round.

Those are different games. In self-play, "how big is the group this manager holds" has no meaning -- both seats are the same policy, so the split is noise. In the paired game it is the central quantity: a manager can **win or lose members to a rival**, and a policy that produces a high pool for the members it keeps is worth little if it cannot keep them.

This branch puts a rule and a rival in the **same world, one group each**, and asks whether the sweep's conclusions survive the change of setting.

### Two rivals, because they are two different games

| rival family | what a member is choosing between |
|---|---|
| the clone (`ah_punisher`) | two disciplined groups; both seats punish, so moving does not escape punishment |
| never-punish (`never`) | a disciplined group and a **refuge**; a punishing rule is asking its members to accept a cost they could avoid simply by moving |

The second is the harder test and the one most likely to overturn the sweep's ranking. The sweep could not pose it at all.

### The pairings

Twelve per seed. `group_0` carries the focal manager, `group_1` the rival.

| family | pairings |
|---|---|
| controls | `ah_punisher_vs_ah_punisher`, `never_vs_never` |
| vs the clone | `prop10`, `thr9_p10`, `thr9_p5`, `human_severity`, `never` -- each against `ah_punisher` |
| vs never-punish | `prop10`, `thr9_p10`, `thr9_p5`, `human_severity`, `ah_punisher` -- each against `never` |

The two symmetric controls are what say whether an asymmetry is the **rule** or the **seat**: in them both seats carry the same policy, so any persistent difference between group 0 and group 1 is positional. `never_vs_ah_punisher` and `ah_punisher_vs_never` are seat swaps of each other and connect the two families, giving a second, direct reading of the seat effect on an asymmetric pairing.

All twelve live in **one config per seed** with one shared manager set, because `MultiManager` evaluates every manager in a file each round, so a run's RNG consumption depends on the file's manager *set* (parent log, note 4). One file means one manager set means the pairings are as stream-comparable as this simulation allows.

### The two accountings

As the parent. Every number is recomputed from contributions and punishments; neither the env's `common_good` state field (the per-capita **share**) nor the human data's `common_good` column (the undivided **pool**) is read, since `per_round.parquet` is written from the env and the two names collide (`notes/autoresearch_log/manager-common-pool-reward.md`).

* **env** -- punishment aimed at a timed-out player is dropped, so it is free; that player's own payoff is discarded.
* **corrected** -- that punishment is charged to the pool, because the artificial humans were shown it and reacted to it; the timed-out player is paid `20 - 0 - 0 + share`, as the real game paid them.

### Artifact naming contract

| what | path |
|---|---|
| config generator | `scripts/data_analysis/rule_manager_configs.py --paired` (reuses the parent's rule definitions, so the rules are byte-identical) |
| sim configs | `configs/simulation/manager_testing/25_rule_vs_clone_paired_s{42,43,44}.yml` |
| analysis | `scripts/data_analysis/rule_vs_clone_paired_report.py` |
| tables and figures | `plots/data_analysis/evaluation/rule_vs_clone_paired/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Establish whether the simulation path can express a two-manager pairing at all, before writing anything. | **done** -- it can; section 3.1 |
| 2 | Emit the twelve-pairing configs from the parent's generator so the rule definitions cannot drift, and run seeds 42/43/44 on the corrected stack. | **done** -- jobs 30401279 / 30401290 / 30401300 |
| 3 | Score each seat separately: group size over rounds, the undivided pool recomputed from contributions and punishments, contributions, punishments, contributor payoff. | **done** |
| 4 | Judge every difference against the seed-to-seed spread measured **here**, and put the sweep's margins and this setting's side by side. | **done** -- sections 3.5, 3.6 |
| 5 | Log, PR. | **done** |

## 3. Results

### 3.1 The simulation could already express a pairing; nothing new was introduced (measured)

**No new simulation mode was needed, and none was written.** `simulate.py` has carried a `pairings` mode since PR #93 (`9feceef`, "Add pairings + dynamic per-round manager dispatch"), and the parent branch's own frontier config already uses it -- for self-play, with the same manager on both sides, which is why the parent's log reads as though the capability did not exist.

What it does, read off the code rather than the commit message:

* A config may carry `pairings:`, a list of `{name, group_0, group_1}`. The simulation then builds **one run per pairing** instead of one run per manager.
* Inside the round loop, when a pairing is active the per-agent manager list is **rebuilt every round from the live `state["agent_group"]`**: `group_map = [pairing["group_0"], pairing["group_1"]]; groups = [group_map[g] for g in current_agent_group]`. A player who switches seats at round *s* is punished by the **other** manager from round *s* onwards. That is the training-time semantics, not an approximation of it.
* `MultiManager.get_punishments` then masks per manager: `create_data` fills every cell whose agent is not in that manager's group with the manager's own default, and the `in_group` flag is `g1 == g2`. Each manager sees only the players it holds, including in the history -- a switcher's rounds under the other seat read as defaults.

So the pairing is expressed by configuration alone. The only thing this branch adds to the simulation is **nothing**: the code change is a config generator and a report.

**Two limits worth stating, because they bound what the numbers mean.**

1. `MultiManager` evaluates *every* manager in a config on *every* round regardless of which pairing is running, so a run's RNG consumption depends on the config's manager **set** (parent log, note 4). All twelve pairings therefore live in one file with one manager set; across seeds the difference is an ordinary redraw, which the three seeds bound.
2. The clone is a `LinearManager`, which declares `needs_rounds` and consumes the raw round history rather than the masked `create_data` view. Its features are all per-agent (`contribution`, `contribution_max`, `contribution_valid`, `prev_contribution`, `prev_punishment`, `round_number`, `is_first`) so it cannot read the rival group's state; but a switcher's `prev_punishment` is the punishment they really received in the *other* seat. That is arguably right -- the player carries their history across the move -- and in any case it applies identically in the clone-against-clone control, so it cancels out of every contrast reported here. The rules are unaffected: they read only the current round's contribution.

**The dispatch is checked from the output, not assumed.** `check_dispatch` reads each rule's signature out of `per_round.parquet`: the seat holding `never` must show punishment 0 everywhere, a threshold seat only 0 or its amount, a `prop10` seat exactly `20 - c` on cells where the player gave an input. Static dispatch, or dispatch keyed on the initial rather than the current group, would fail these.

### 3.2 What was run, and the spread every number below is judged against (measured)

Three simulations, all `COMPLETED`, all on the frontier stack with byte-identical artifacts and the standard protocol (2 groups x 8 agents, 24 rounds, 100 episodes, `save_per_round: true`, `reseed_per_run: true`). Nothing was trained or recalibrated.

| seed | config | job | wall |
|---|---|---|---|
| 42 | `25_rule_vs_clone_paired_s42.yml` | 30401279 | 15:28 |
| 43 | `25_rule_vs_clone_paired_s43.yml` | 30401290 | 15:14 |
| 44 | `25_rule_vs_clone_paired_s44.yml` | 30401300 | 15:18 |

Each file is 12 pairings x 100 episodes x 24 rounds x 8 agents = 230,400 agent-rounds. Every table below **pools the three seeds** (300 episodes per pairing) and every interval is a 95% bootstrap over episodes.

**The dispatch check passes 14/14 seats.** Every seat carries its manager's signature in the output: a `never` seat punishes 0 and nothing else, a `thr9_p10` seat only 0 or 10, a `thr9_p5` seat only 0 or 5, a `prop10` seat exactly `20 - c` wherever the player gave an input. Static dispatch, or dispatch keyed on the initial rather than the current group, could not produce this.

**The seed-to-seed spread, measured here, is the yardstick** (`seed_spread.csv`, standard deviation of the three per-seed 100-episode means):

| quantity | median sd across seats | max sd |
|---|---|---|
| group size (rounds 16-23) | **0.16 members** | 0.32 |
| common pool per seat per round | **3.15** | 6.08 |

Every difference reported below is quoted against those two numbers. Anything under ~0.3 members or ~6 pool points is not a finding here.

**The symmetric controls say the seat itself is worth nothing.** With the same manager on both sides, focal minus rival:

| control | group size | common pool |
|---|---|---|
| `ah_punisher_vs_ah_punisher` | **-0.00** [-0.39, +0.39] | +3.14 [-4.66, +10.96] |
| `never_vs_never` | -0.34 [-0.71, +0.06] | -6.11 [-13.21, +0.94] |

Neither excludes zero on group size. So an asymmetry bigger than about 0.4 members is the rule, not the seat -- which is exactly what the controls were run for.

### 3.3 The headline: every punishing rule LOSES members, and the harder it punishes the more it loses (measured)

Focal minus rival, same world, paired by episode, 300 episodes. Group size is the mean over all 24 rounds; the rules start at 4 of 8 by construction.

| focal | rival = the clone | rival = never-punish |
|---|---|---|
| `prop10` (mean p 8.0) | **-1.67** [-2.01, -1.32] | **-2.55** [-2.88, -2.21] |
| `thr9_p10` (mean p 3.7) | **-0.63** [-1.00, -0.26] | **-1.29** [-1.65, -0.93] |
| `thr9_p5` (mean p 2.2) | +0.03 [-0.36, +0.42] | **-1.18** [-1.57, -0.83] |
| `human_severity` (mean p 2.0) | -0.00 [-0.38, +0.37] | **-1.14** [-1.51, -0.76] |
| `ah_punisher`, the clone (mean p 2.0) | *-0.00, the control* | **-1.51** [-1.87, -1.15] |
| `never` | **+0.79** [+0.42, +1.15] | *-0.34, the control* |

Read down the columns and the result is one sentence: **against a rival that punishes, the amount of membership a manager loses is ordered by how hard it punishes; against a rival that does not punish, everyone who punishes loses about the same large amount, the clone included.**

* `prop10`, the sweep's winner, ends up holding **3.16 of 8 members against the clone and 2.73 against never-punish** -- it is a group of under three in a world of eight. That is 10x and 16x the group-size seed spread.
* **`never` is the only manager that gains members**, +0.79 against the clone. Not punishing is a membership advantage.
* **The clone bleeds to the refuge exactly as the rules do**, -1.51 [-1.87, -1.15]. This is not a rule-versus-human story. It is a punishment-versus-no-punishment story, and our clone of a human manager is on the losing side of it.
* `thr9_p5` and `human_severity` -- the two mildest punishers -- hold their own against the clone and only lose against the refuge. Punishing at roughly human intensity is survivable when the alternative also punishes, and not when it does not.

### 3.4 Who leaves: the low contributors, the ones the rule just punished (measured)

Switching is decided at round *s* and applied at *s+1*, so the decision rounds are *s* with `(s+1) % 4 == 0`. Comparing the members who leave a seat with those who stay, at the moment the move is decided (`who_leaves_pooled.csv`, 4,800-9,300 decisions per seat):

| seat | leave rate | contribution, leavers vs stayers | punishment, leavers vs stayers |
|---|---|---|---|
| `prop10` vs never | **0.39** | 10.18 vs 15.29 (**-5.11**) | 9.82 vs 4.71 (**+5.11**) |
| `prop10` vs clone | **0.34** | 10.33 vs 15.62 (-5.28) | 9.67 vs 4.38 (+5.28) |
| `thr9_p10` vs clone | 0.28 | 9.34 vs 13.04 (-3.70) | 4.66 vs 2.07 (+2.58) |
| `ah_punisher` (control) | 0.24 | 8.79 vs 11.28 (-2.49) | 3.26 vs 1.11 (+2.16) |
| `never` (control) | **0.22** | 7.17 vs 8.87 (-1.70) | 0 vs 0 |
| `never` vs clone | **0.21** | 7.50 vs 8.97 (-1.47) | 0 vs 0 |

Two things are true in **every one of the 24 seats**: leavers contributed less than stayers, and leavers had been punished more than stayers. The leave rate is ordered by the seat's mean punishment, from `never`'s 0.21-0.22 to `prop10`'s 0.34-0.39.

So the mechanism is not subtle. A contribution-keyed rule punishes its low contributors; the simulated players it punishes leave; the harder the rule punishes, the more of them leave. Note also that even `never` loses its low contributors at 0.21-0.22 -- there is a baseline churn that is not about punishment at all, and the punishing rules add to it.

### 3.5 The rule pays for enforcement and the neighbour collects it (measured)

The same contrast on the **rival's** seat: what the rival gains merely from having this rule across the fence, with no change to its own policy. Rival seat of the pairing against the rival seat of the symmetric control.

| rival | facing | group size | common pool |
|---|---|---|---|
| the clone | `prop10` | **+0.83** [+0.58, +1.10] | **+20.07** [+12.77, +27.31] |
| `never` | `prop10` | **+1.11** [+0.85, +1.36] | **+18.91** [+11.91, +25.70] |
| the clone | `thr9_p10` | +0.32 [+0.05, +0.58] | **+8.92** [+1.42, +16.12] |
| `never` | `thr9_p10` | +0.48 [+0.22, +0.74] | **+7.87** [+1.54, +14.33] |
| the clone | `thr9_p5` | -0.02 [-0.29, +0.26] | +3.19 [-4.01, +10.64] |
| `never` | `human_severity` | +0.40 [+0.13, +0.67] | +5.53 [-1.68, +12.56] |

And the matching loss on the seat that does the punishing (focal seat against the same seat of its control):

| focal | rival | group size | common pool |
|---|---|---|---|
| `prop10` | clone | **-0.83** [-1.10, -0.58] | **-15.62** [-23.08, -7.95] |
| `prop10` | never | **-1.11** [-1.36, -0.85] | **-18.22** [-25.10, -11.55] |
| `thr9_p10` | clone | -0.32 [-0.58, -0.05] | -3.54 [-10.90, +3.98] |
| `thr9_p5` | clone | +0.02 [-0.26, +0.29] | +2.17 [-5.17, +9.58] |
| `never` | clone | **+0.40** [+0.13, +0.66] | +1.04 [-6.11, +7.94] |

The two tables are near mirror images, and that is the finding: **punishment in this world is a cross-group public good.** `prop10` raises contributions -- its own stayers reach 15.6 against the clone's 11.3 -- but it pays the whole punishment bill (mean 8.0 per member per round) and exports the disciplined players, so the pool it holds falls by 15.6 while the neighbour's rises by 20.1 for doing nothing.

The environment permits this by construction and it is worth stating plainly: the contribution model's graph is **fully connected across both groups**, with membership entering only as a `same_group` edge feature rather than as a hard mask (`SameGroupEdgeEncoder`, `environment.batch_edge_index`). So a punished player influences the other group's players directly, as well as by moving into their group. This run does not separate those two channels; it measures their sum.

### 3.6 Does the sweep's conclusion survive the change of setting? No (measured)

The sweep's number is a whole-population total in self-play -- both seats carried the manager, so it counts two groups. A paired margin changes one seat, so the like-for-like sweep figure is halved. That halving is a first-order normalisation, not an identity, and the claim rests on the **sign** and on the size against each setting's own spread, not on the arithmetic matching.

| manager | sweep, self-play, both groups | sweep, per seat | **this setting, focal seat vs the clone's seat** | members |
|---|---|---|---|---|
| `prop10` | +24.98 | +12.49 | **-15.62** [-23.08, -7.95] | -0.83 [-1.10, -0.58] |
| `thr9_p10` | +12.73 | +6.37 | **-3.54** [-10.90, +3.98] | -0.32 [-0.58, -0.05] |
| `thr9_p5` | +10.53 | +5.27 | +2.17 [-5.17, +9.58] | +0.02 [-0.26, +0.29] |
| `human_severity` | +5.16 | +2.58 | -1.21 [-8.25, +5.86] | -0.00 [-0.27, +0.26] |
| `never` | -11.43 | -5.72 | +1.04 [-6.11, +7.94] | +0.40 [+0.13, +0.66] |

**Neither of the sweep's two winners survives.**

* **`prop10`'s win does not merely vanish, it reverses.** Expected +12.5 per seat, measured **-15.6** -- a swing of 28 points against a pool seed spread of 3.15 (max 6.08), so 4.6x to 9x the noise. The sweep's largest, most confident result is the one the change of setting destroys most completely.
* **`thr9_p10`'s win does not survive.** Expected +6.4, measured -3.5 [-10.9, +4.0]: indistinguishable from zero and pointing the wrong way. The sweep called this "the simplest robust winner"; in a competitive world it is not a winner at all.
* **`never`'s loss does not survive either.** The sweep had `never` last but one of 21, -5.7 per seat against the clone. Here it is **+1.0 [-6.1, +7.9] on the pool and +0.40 [+0.13, +0.66] on members** -- no worse than the clone at filling its own pool, and the only manager that reliably grows.

The sweep's ranking and this one agree on nothing except that `thr9_p5` and `human_severity` are indistinguishable from the clone. **The sweep's headline sentence -- "punish 10 whenever a player contributed 9 or less" -- does not carry over to the game the RL manager will be trained in.**

What does carry over is the sweep's *arithmetic*, in a new place. The sweep found punishment pays above 0.625 contribution units bought per punishment point. That break-even is still right for a closed group; what the paired setting adds is that a manager does not keep the contributions it buys. It keeps a shrinking share of them.

### 3.7 Contributions, punishments and contributor payoff, per seat (measured)

Pooled, per round, both seats (`summary_pooled.csv`; per-round trajectories in `trajectories_vs_ah_punisher_pooled.jpg` and `trajectories_vs_never_pooled.jpg`).

| pairing | seat | manager | members | pool (corr.) | pool per member | mean c | mean p | contributor payoff |
|---|---|---|---|---|---|---|---|---|
| `ah_punisher_vs_ah_punisher` | focal | clone | 4.00 | 59.55 | 13.34 | 10.25 | 1.91 | 92.21 |
| | rival | clone | 4.00 | 56.41 | 12.63 | 9.92 | 2.03 | 90.37 |
| `never_vs_never` | focal | never | 3.83 | 51.70 | 12.40 | 8.44 | 0 | 96.95 |
| | rival | never | 4.17 | 57.82 | 12.86 | 8.53 | 0 | 106.14 |
| `prop10_vs_ah_punisher` | focal | `prop10` | 3.16 | 43.92 | 10.85 | 12.45 | 8.01 | 44.30 |
| | rival | clone | 4.84 | **76.55** | **15.09** | 11.22 | 2.09 | 111.51 |
| `prop10_vs_never` | focal | `prop10` | **2.73** | 33.54 | 8.71 | 11.59 | 8.88 | 33.72 |
| | rival | never | **5.28** | **76.76** | 14.54 | 9.45 | 0 | **135.75** |
| `thr9_p10_vs_ah_punisher` | focal | `thr9_p10` | 3.68 | 56.08 | 13.04 | 11.25 | 3.71 | 76.56 |
| | rival | clone | 4.32 | 65.29 | 14.02 | 10.63 | 2.00 | 99.04 |
| `thr9_p5_vs_ah_punisher` | focal | `thr9_p5` | 4.02 | 61.68 | 13.75 | 10.57 | 2.11 | 92.25 |
| | rival | clone | 3.98 | 59.48 | 13.44 | 10.30 | 1.86 | 92.20 |
| `never_vs_ah_punisher` | focal | never | **4.40** | 60.68 | 13.01 | 8.65 | 0 | 111.84 |
| | rival | clone | 3.60 | 51.34 | 12.27 | 9.65 | 1.86 | 82.30 |
| `ah_punisher_vs_never` | focal | clone | 3.25 | 40.24 | 10.55 | 8.75 | 2.01 | 70.88 |
| | rival | never | 4.75 | 61.13 | 12.42 | 8.10 | 0 | 119.10 |

Three things worth naming.

1. **The size effect is not the whole story.** `prop10`'s seat is worse than the clone's *per member* too -- 10.85 against 15.09, a gap of -4.25 [-5.33, -3.13]. It raises contributions (12.45 against 11.22) and spends more than the increase is worth: `1.6 x 12.45 - 8.01 = 11.9` against `1.6 x 11.22 - 2.09 = 15.9`.
2. **On contributor payoff the ordering is even starker**, because payoff charges punishment at 2 rather than 1 (review S1). `prop10`'s members take 44.30 against the clone's 111.51 in the same world. `never`'s seat pays its members the most anywhere in the table.
3. **`never_vs_ah_punisher` and `ah_punisher_vs_never` are seat swaps and agree.** never's seat: 4.40 members / 60.68 pool in one, 4.75 / 61.13 in the other. The clone's seat: 3.60 / 51.34 and 3.25 / 40.24. The result does not depend on which seat a manager sits in -- a second confirmation of section 3.2's controls.

### 3.8 Free punishment on timed-out cells, carried forward (measured)

`auto/free-punishment-fix` had not landed, so the sweep's finding D1 is still live here and is reported so the numbers can be corrected later.

| seat's manager | share of timeout cells punished | share of that seat's total punishment spent there |
|---|---|---|
| every contribution-keyed rule (`prop10`, `thr9_p10`, `thr9_p5`) | **1.000** | 5-6% |
| `human_severity` | 0.43-0.50 | 4-5% |
| `ah_punisher`, the clone | 0.00-0.01 | ~0% |
| `never` | 0 | 0 |

Identical in shape to the sweep's section 3.6. The env and corrected accountings differ by 1 to 2 pool points per seat (`prop10` the most: 45.08 env against 43.92 corrected) and **change no conclusion in this log** -- every margin above is on the corrected accounting, and the env accounting moves each by about a point in the direction that flatters the rules. Since the punishing rules are the ones losing, closing D1 would make their position slightly worse, not better.

## 4. Notes

1. **Measured against inferred.** Sections 3.2 to 3.8 are measurements: group sizes, pools recomputed from contributions and punishments, leave rates and leaver/stayer gaps, all with episode bootstraps and all judged against this branch's own seed spread. The *reading* of section 3.5 -- that punishment is a cross-group public good -- is an inference from two measured mirror-image tables plus one code fact (the graph is fully connected across groups). It is a strong inference but it is not a decomposition: this run does not separate the graph channel from the migration channel, and it should not be quoted as though it did.

2. **The comparison with the sweep is a comparison of two different quantities, deliberately.** The sweep measured a manager's whole-population common good in self-play; this measures the pool of the one group a manager holds while a rival holds the other. There is no arithmetic that makes them the same number, and the halving in section 3.6 is a first-order normalisation offered so the two can be read together. What is being claimed is about signs and about sizes relative to each setting's noise, and both of those are unambiguous for `prop10`.

3. **The sweep's numbers were not wrong; they answered a different question.** Inside a closed group, punishing at `prop10`'s intensity does produce a large common good, and this run reproduces that locally: `prop10`'s *stayers* contribute 15.6, the highest figure anywhere in either branch. The sweep's error was not arithmetic, it was that self-play made group size meaningless and so hid the price of producing that contribution.

4. **`never`'s membership advantage is not an argument that never-punishing is good.** `never`'s seat has the lowest mean contribution in the table (8.4-8.7) and only draws level with the clone on the pool because it spends nothing. What the result says is narrower and more useful: in a world where leaving is cheap, a manager cannot be evaluated on the group it *starts* with, and the cost of enforcement falls entirely on the enforcer while part of the benefit does not.

5. **The controls did real work and should be kept in any successor.** Without `ah_punisher_vs_ah_punisher` and `never_vs_never` there would be no way to tell a -0.6 member difference from a seat artefact. They came out at -0.00 and -0.34, which is what licenses reading everything above +/-0.4 as the rule. The seat-swap pair (`never_vs_ah_punisher` / `ah_punisher_vs_never`) is a cheap second check and agreed.

6. **One asymmetry between manager types is left in place and is not confounding.** The clone is a `LinearManager` and reads the raw round history, so a switcher's `prev_punishment` is what they really received in the other seat; a `create_data` manager would see a default there. This applies identically in the clone-against-clone control, so it cancels out of every contrast reported. It is listed here because a successor that puts two *learned* managers in the two seats will have to decide the question deliberately.

7. **Free punishment on timed-out cells is still open** (section 3.8) and works against the punishing rules, so closing it cannot rescue `prop10` or `thr9_p10`. The correction is worth applying before any *positive* claim about a punishing policy in this setting.

### Successor

1. **Re-point the RL manager's baseline.** The parent log's successor item 2 said the bar for a trained manager should be `thr9_p10` at 123.8 common good rather than the clone. That recommendation was made in the self-play setting and **does not survive**: in the paired game `thr9_p10` is -3.5 [-10.9, +4.0] against the clone's own seat and loses 0.3 members. The honest bar in the competitive setting is the clone's seat (59.6 pool, 4.00 members) and `never`'s seat (51.7-60.7 pool, 3.9-4.4 members), with group size reported alongside the pool in every case.

2. **Settle the reward specification with group size in it.** The parent showed the pool and the payoff sum disagree at Spearman -0.49 to -0.67. This branch adds a third consideration that neither captures: a manager rewarded on its own group's undivided pool has a direct incentive to *acquire members*, and one rewarded per capita does not. `reward_mode: common_pool` (`origin/auto/manager-common-pool-reward`) is the undivided pool, so it already carries that incentive -- which is a design decision that should be made on purpose rather than inherited.

3. **Expect the learner to discover under-punishment, and do not read it as a training failure.** Under any per-seat reward in this setting, punishing costs members and the benefit leaks to the rival. A trained manager that converges to punishing near `thr9_p5`'s intensity or below is responding correctly to the game as specified. The discriminating plot is the trained manager's realised mean punishment and its held group size, together, against `never`'s (0, 3.9-4.4) and the clone's (2.0, 4.00).

4. **Separate the two channels by which a rule helps its rival.** Section 3.5 measures the sum of graph spillover and migration. Masking the contribution model's edges by `agent_group` at simulation time (not retraining) and re-running two pairings would bound the graph channel. This is a simulation-behaviour change and belongs in its own experiment.

5. **The cheap extension is more rivals, not more rules.** Everything interesting here came from changing *what the rule competes against*, not from the rule family. The obvious missing rival is a manager that punishes only newcomers, or only the lowest contributor -- policies that discipline without giving the median member a reason to move.

6. **Nothing here needs a retrain, and nothing here was one.** Three 15-minute GPU jobs reproduce every number in this log.
