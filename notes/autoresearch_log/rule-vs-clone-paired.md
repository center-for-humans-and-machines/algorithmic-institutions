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

*(filled in below once the runs returned)*
