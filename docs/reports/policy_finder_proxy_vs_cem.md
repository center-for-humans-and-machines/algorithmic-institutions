# Policy finder through the Claude proxy, against Cem's campaign

The policy finder was rerun from scratch through the Claude proxy on 2026-11-04. Each agent
ran in its own locked container, started by the container spawner, and every simulation ran
on Raven through the command server. The campaign is compared here with Cem's hand-run
campaign of 2026-10-07 to 10-29, from his observatory snapshot of 2026-10-29. The plan is
`docs/plans/archive/policy-finder-proxy-adapter.md`. The observation page is the artifact
"Policy Finder Proxy Campaign".

## Summary

- **At least Cem's level, likely a little above.** This campaign's best rule,
  `pf-20261104-12-v7`, scores 73.4 ± 1.1 at its GP peak. Cem's best scores 72.8 ± 1.1 at
  its GP peak and 73.2 ± 1.1 as a best of 6. On a common seed (head to head, below), this
  campaign's top two rules score 72.2 and 72.1 against 70.3 for Cem's top rule, ± 0.8 each.
  Paired on matched episodes, both are +1.9 ± 1.1, about 1.7 standard errors: suggestive,
  not conclusive.
- **Iteration does the work, in both campaigns.** First versions score alike: a mean GP peak
  of 66.5 here and 66.8 for Cem. Iterating the best lineages gains about 3 points at the
  top, from 70.3 to 73.4 here and from 69.9 to 72.8 for Cem. Cem chose by hand which
  lineages to iterate; here the driver chose the top 4 by score.
- **Switch-timed wins.** 21 of the 36 versions time their punishment around the switch
  decisions. Their mean GP peak is 71.0, against 67.2 for the 14 scored front-loaded ones,
  and no front-loaded version beats Levin's rule (69.8). Cem's best rule is switch-timed
  too, but his second best (`pf-20261008-8-v6`, 72.8) is front-loaded.
- **No rule needed a person, the harness did.** The campaign took 7 h 50 min
  (09:30–17:17), including a 1 h pause after generation 1 and a Raven outage of about
  40 min that needed a manual restore. Five harness faults surfaced, each once, and each
  was fixed in code (Process, below). The cost was 36 agent runs at a median of 8 min each
  ($44 API-equivalent, on the plan, not billed) and about 12 GPU-hours on Raven.

## Setup

| | Cem | This campaign |
|---|---|---|
| Rule inputs | per-player summaries (pre-#245) | raw round history (#245) |
| Params, v1 / later | up to 5 / unbounded | the same |
| Sweep | 256 Sobol points × 500 episodes, seed 42, 2 parts | the same |
| Final run | GP peak + top 5 at 2000 episodes, seed 1000 | the same |
| Score | best of the 6 | the GP peak; the best of 6 is reported beside it |
| Agent | Sonnet relay, then the `policy-finder` subagent on `opus` | the same; `opus` resolved to `claude-opus-5-5`, the relay to `claude-sonnet-5-5` |
| Isolation | sandbox on the shared machine | one container per agent; network to `api.anthropic.com` only; an export without git history or other lineages |
| Lineages | 15 lineages, 38 versions, iterated by hand | 12 fresh lineages, then 6 generations iterating the top 4: 36 versions |

A version's GP peak is the Gaussian process's peak, rerun at 2000 episodes on fresh seeds.
Its best of 6 is the highest of that peak and the 5 best sweep points rerun the same way.
That selection reads high by about 1 point (see Scores). Every pool is the managed group's
mean common pool against `ah`, ±1 standard error (about 1.1).

## Scores

### Per generation

The 12 lineages ran once each (generation 1). Then, in each generation, the 4 lineages with
the highest GP peak so far got a new version.

| gen | versions | best GP peak | best best-of-6 | mean GP peak | campaign best so far | improved on their lineage |
|---|---|---|---|---|---|---|
| 1 | 12 | 70.50 (`-7`) | 70.50 | 66.47 | 70.50 | – |
| 2 | 4 | 71.91 (`-9-v2`) | 71.91 | 69.97 | 71.91 | 3 of 4 |
| 3 | 4 | 73.02 (`-12-v3`) | 73.03 | 70.94 | 73.02 | 3 of 4 |
| 4 | 4 | 72.89 (`-12-v4`) | 73.69 | 71.16 | 73.02 | 1 of 4 |
| 5 | 4 | 73.29 (`-12-v5`) | 73.29 | 71.23 | 73.29 | 1 of 4 |
| 6 | 4 | 72.73 (`-9-v6`) | 74.18 | 71.12 | 73.29 | 0 of 4 |
| 7 | 4 (1 failed) | 73.43 (`-12-v7`) | 73.43 | 72.15 | 73.43 | 2 of 3 |

The same 4 lineages (`-12`, `-9`, `-5`, `-7`) held the slots from generation 2 to the end.

| Lineage | GP peak by version (v1 to v7) |
|---|---|
| `-12` | 70.3, 70.8, 73.0, 72.9, 73.3, 72.4, **73.4** |
| `-9` | 67.1, 71.9, **72.9**, 71.3, 71.8, 72.7, 71.7 |
| `-5` | 67.3, 67.6, 68.3, 71.3, 70.2, 70.6, **71.4** |
| `-7` | **70.5**, 69.6, 69.5, 69.2, 69.7, 68.8, failed |

Most of the gain came in generations 2 and 3. After that, the top lineage moved within its
standard error, while the mean of the iterated four kept rising, from 70.0 to 72.2.

### Against Cem's 37 validated versions

GP peaks and best-of-6 for Cem's versions are read from the validation tables in his
snapshot. Lineage best is the highest of a lineage's versions.

| | This campaign (12 lineages) | Cem (14 validated lineages) |
|---|---|---|
| Best GP peak | **73.43** (`-12-v7`) | 72.8 (`pf-20261008-3-v6`) |
| Best best-of-6 | **74.18** (`-9-v6`) | 73.2 (`pf-20261008-3-v6`) |
| First versions: mean / median / best GP peak | 66.5 / 66.0 / 70.5 | 66.8 / 67.0 / 69.9 |
| Median lineage best, GP peak | 66.0 | 69.2 |
| Median lineage best, best of 6 | 67.4 | 69.5 |
| Lineages above Levin's rule (69.8), GP peak | 4 of 12 | 5 of 14 |
| Lineages iterated | 4, automatically | 5, by hand |

Cem's lineage median is higher for two reasons. He iterated 5 of his 14 lineages, and
several of his un-iterated first versions scored 69 to 70; here, the 8 lineages that were
not selected kept their first version (63.9 to 66.3). The two campaigns' tops differ by
less than one standard error. That is why the head-to-head below matters.

Best of 6 reads about 1 point above the GP peak whenever a sweep point beats the peak by
luck. Of Cem's 37 versions, 21 took a sweep point as their best; here it was 20 of 35.
Because of that selection, the GP peak is this campaign's score.

## Head to head on a common seed

Config `configs/simulation/manager_testing/27_LEVIN_pf_proxy_vs_cem_batched.yml`, Raven
job 31911380. Each manager played `ah` on both group sides, 2000 episodes per side, on
seed 5000, which no sweep or final run used. The managers are this campaign's three lineage
bests at their GP peaks and Cem's top rule (`pf-20261008-3-v6`, point s201, ported to the
#245 inputs). The pool is definition 4 of #226 (`pool_scores.against_anchor`), the
sweeps' own score, pooled over both sides.

| Manager | Pool vs `ah` | as group 0 | as group 1 | Members | Paired against Cem's |
|---|---|---|---|---|---|
| `pf-20261104-12-v7` | **72.24 ± 0.79** | 71.57 | 72.91 | 4.34 | +1.95 ± 1.09 |
| `pf-20261104-9-v3` | **72.14 ± 0.80** | 72.32 | 71.96 | 4.30 | +1.86 ± 1.11 |
| `pf-20261008-3-v6` (Cem) | 70.28 ± 0.77 | 70.09 | 70.47 | 4.30 | – |
| `pf-20261104-5-v7` | 68.08 ± 0.79 | 67.06 | 69.11 | 4.13 | −2.20 ± 1.11 |
| `zero` (never punishes) | 60.24 ± 0.67 | 59.79 | 60.70 | 4.60 | −10.04 ± 1.03 |
| `ah` against `ah` | 56.21 ± 0.71 | | | 4.00 | |

"Paired" takes the difference episode by episode: the same seed, side and episode index
for both managers. The paired standard error is no smaller than the unpaired one, because
the games diverge after the first round's punishment.

**Reading.**
- The references match their earlier values: `ah` against `ah` 56.2 (56.9 in Cem's
  snapshot), `zero` 60.2 (59.5).
- Every campaign score reads high, because each top was selected from many noisy finals.
  This campaign's top two dropped by 1.2 and 0.8 from their own finals (73.4 and 72.9).
  Cem's rule dropped by 2.5 from its GP peak (72.8) and 2.9 from its best of 6 (73.2). It
  scored 72.7 ± 1.6 at seed 2000 in `26_LEVIN`, so part of that drop is this seed's draw.
- Two of this campaign's lineages, found independently, both land about 2 points above
  Cem's best. That is evidence the campaign at least matched his, and likely did slightly
  better. A second seed would settle it: one 13-minute job.
- `pf-20261104-5-v7`, third here, falls below Cem's rule. Its final run had it at 71.4.

## Ideas

Categories come from Claude reading each rule and its final values, as in the
`policy-finder-summarizer` skill, using Cem's category names so the campaigns compare.
They are kept in `pf_runs/worktrees/summary.json`, outside every agent's reach.

| Policy category | Versions | Mean GP peak | Best GP peak |
|---|---|---|---|
| switch-timed | 21 | 71.0 | 73.43 (`-12-v7`) |
| front-loaded | 15 (14 scored) | 67.2 | 69.66 (`-7-v5`) |

By generation, switch-timed took over the iterated slots: 5 of 12 versions in generation 1,
2 of 4 in generations 2 and 3, and 3 of 4 from generation 4 on. Cem's ten best lineages
split 5 and 5.

**Shared with Cem.** Every lineage fines a contribution shortfall and stops or fades toward
the end game, and every rule spares players who timed out. The best rules of both campaigns
fine hardest right after a switch and least, or not at all, just before the next switch
decision. Cem's `pf-20261008-3-v6` does this with a group-mean bar; this campaign's `-9`
does the same with a bar at about 80% of the rest of the group's mean.

**New here.**
- **A blended bar** (`-12-v3` to `-v7`): the bar sits between a fixed norm (about 8.6) and
  the rest of the group's mean, about 60% of the way to the mean, and fines are dealt only
  in the two rounds after each switch, halving each block. It is the campaign's best rule.
- **Group size scaling** (`-5-v7`, `-12-v4`/`-v5`): fines scale with the cube of the group's
  size over 4, so a shrinking group is fined much less. Cem's `pf-20261008-8` went the
  other way: heavier in big groups.
- **Newcomers** (`-7-v7`): late in the game, fines only players who just switched in. Its
  sweep crashed (see Process), so it was never tested.
- **Tried and switched off by the sweep:** a guard on the rival group's pool (`-5-v6`,
  `-7-v4`, `-9-v5`), scaling by how much the rest of the group gives (`-12-v6`), and a
  discount for repeat fines (`-7-v5`). Each sat at its off end at the GP peak.

**Lineages that changed character.** `-5` moved from front-loaded (v2, v3) to switch-timed
(v4 on), and gained 3 points as it did. `-7` moved the other way, from a switch-timed v1 to
front-loaded from v2 on, and never beat its first version.

## Process

| | |
|---|---|
| Versions | 36 run: 35 scored, 1 failed (`-7-v7`) |
| Agent time per version | median 7.8 min (4.5 to 20.4) |
| Agent tokens, all versions | Opus 5.5: 1.13 M output, 48.1 M cache read, 2.65 M cache write; the Sonnet 5.5 relay: 54 k output |
| API-equivalent cost | $44.24 in all, median $1.18 per version (the plan, not billed) |
| Raven | about 4.4 h of sweep jobs (2 parts in parallel) and 2.9 h of final runs, wall clock: about 12 GPU-hours |
| Wall clock per generation, after the fixes | 22–38 min |

**Failures by step.** Every one of these was the harness, not a rule. Each was fixed in
code, and the affected versions were resumed or rerun:

| Step | What happened | Versions | Fix |
|---|---|---|---|
| image check | 12 parallel `validate-rule` runs on the Raven login node: thread limits, the container driver, interrupts. These were counted as verdicts. | 4 in gen 1 | serialised to 2 slots, thread caps, only `Invalid:` is a verdict (44b0de29) |
| notes check | agents inserted text above `## Key findings` with a script and deleted the heading | 3 in gen 1 (rerun fresh) | the notes are written append-only, with a heading check before finishing (f25e6a85) |
| fetch | two pipelines fetching at once copied each other's half-written files | 1 | fetch lock, rsync temp files skipped, size check (44d04009) |
| all cluster steps | Raven unreachable for about 40 min (gen 3) | 4, resumed | an outage no longer resubmits or fails a version, and the driver waits for Raven without spending retries (5eb3a609); rsync's vanished-files exit accepted (eb79e452) |
| sweep | `-7-v7` built a tensor with `th.arange` without `device=`, which fails on the GPU at every point while every CPU check passed | 1 (failed) | `validate-rule` rejects such constructors, and the prompt says the rule runs on the GPU (804a70af) |

`-7-v7` stayed failed, as the plan's rules say: a failed version counts as no improvement.
It could not have changed the top: its lineage trailed by 3 points.

Versions that ran ahead of their generation's selection and were never adopted
(`-1-v2`, `-11-v2`) stopped before their sweep and are not counted. The 9 lineages of
generation 1 that passed on the first try wrote their notes under the earlier notes prompt.
That changed nothing in their rules or scores.

## Confounds

- **Inputs.** The rules here read the raw round history (#245), which holds strictly more
  than Cem's per-player summaries. The top three rules do not use the extra: they read only
  this round's contributions, who is in the group, and the round number. So the inputs
  cannot explain the top results. Cem's top rule needed porting to run on these inputs
  (5c84b57c, checked identical).
- **Iteration.** Automatic top-4 selection against Cem's choice by hand. The driver kept
  iterating `-7` for 5 generations without a gain. A person would probably have dropped it.
- **Model.** The `opus` alias resolved to `claude-opus-5-5` here. Cem's runs used whatever
  it resolved to in October.
- **Score.** The GP peak here, a best of 6 for Cem. Both are reported for both campaigns
  above. The head-to-head removes the selection and the seed difference.
- **Param bound.** The same in both: up to 5 in a first version, unbounded later. 3 of
  this campaign's versions use 6 parameters.
- **Isolation.** The agents here could not see other lineages, git history, or Levin's and
  Cem's rules. The export enforced this by its path list, a lineage filter and a name check.
  Each lineage started blind to every other, so the campaigns' shared ideas were found
  independently.
