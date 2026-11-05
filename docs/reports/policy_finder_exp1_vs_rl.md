# The policy finder on experiment 1, against its RL manager

Experiment 1 deployed a DQN manager: a recurrent graph network trained to raise the average
payoff of artificial-human contributors (one group of 4, 24 rounds, no switching). This
report asks whether the policy finder would have done as well. The policy finder has agents
design interpretable rule-based managers, then lets a Sobol sweep, a GP fit and a final run
choose each rule's values. It was given the same clones, the same game and the same
objective. The task is `docs/tasks/autoresearch_an_old_policy.md`; the plan is
`docs/plans/archive/policy-finder-exp1.md`. The observation page is the artifact
"Policy Finder Experiment 1"
(https://claude.ai/artifact/PMfYuxEbjvmijyBpznvxnt).

## Summary

- **The best rule comes within 1 point of the RL manager; none beats it.** On fresh seeds,
  22,000 games per rule, the three lineage bests score 655.9, 655.0 and 654.9 (± 0.4); the
  deployed RL manager scores 656.8 ± 0.4 over 23,000 games. The best rule (`-3-v4`) trails
  by 0.9 ± 0.5, which does not separate it clearly from a tie; the campaign's top (`-7-v3`)
  and `-8-v7` trail by 1.8 and 1.9 ± 0.5. All sit about 67 points above never punishing
  (588.9) and about 108 above the human-like manager clone (548.7).
- **Single runs mislead.** One 2000-game run has a standard error of 1.3, and runs of the
  same manager spread over 2.5–5 points. On seed 6000 alone `-7-v3` scored 658.0 and seemed to
  tie the RL manager (+0.8 ± 1.8); on Raven's seeds 7000–7004 alone `-3-v4` tied it exactly
  (656.1 each). Neither was a selection effect (the three were picked on the campaign's own
  finals, before any of these seeds ran), just noise that the pooled 22,000 games average
  out.
- **No leak.** An independent audit of the 38 agent runs' isolation and transcripts found no
  route by which the RL manager's behaviour or scores could have reached an agent (see
  Leak audit).

- **The rules converged on the RL manager's strategy, written down.** The three best
  lineages found the same two-phase rule independently:
  - rounds 0 to about 6: a steep, convex fine on everyone below about 16 (30 for a zero
    contribution, about 7 for 10);
  - until about round 15: only near-zero contributors are fined, with 20–30 points;
  - nothing in the last 8–10 rounds.

  The RL manager does much the same: 7.3 points on average in round 0, fading to almost
  nothing by round 10 and to none from round 20; 12.4 points for contributions of 0–2,
  under 1 for 12 and above.
- **Iteration helped, and stalled early.** First versions averaged 642.4 (best 649.7). The
  campaign best reached 656.05 in generation 3 and did not move after that, while the
  iterated four kept improving on average, from 650.1 to 654.1.
- **The run needed no fixes.** 36 versions, all scored, in 4 h 21 min. Cost: 36 agent runs
  at a median of 11 min ($49 API-equivalent, on the plan, not billed) and about 5 hours of
  Raven job time.

## Setup

| | |
|---|---|
| Game | one group of 4, 24 rounds, no switching; endowment 20, multiplier 1.6, punishment 0–30 charged to the player and the pool |
| Timeouts | a timed-out player contributes 0, cannot be punished and is out of the round's pool; the rule sees its contribution as 0 and `valid` 0 (the clones are fed their trained placeholder, which the rule never sees) |
| Clones | experiment 1's `21_contribution_model_v4` and `22_contribution_valid_model_v4`, restored from the git LFS store, the ones the RL manager was trained against |
| Score | the average payoff: per round, the mean payoff of the valid players, summed over the 24 rounds, averaged over games |
| Rule inputs | `c`, `valid`, `p` over the game so far, and `t` |
| What agents saw | the pilot data the clones were trained on and reference sims under zero punishment and the human-like manager clone; nothing of the RL manager or of experiment 1's live data |
| Budget | 12 fresh lineages, then 6 generations on the top 4: 36 versions; 256 Sobol points × 500 games; GP peak + top 5 at 2000 games |
| Agent | the `policy-finder-exp1` definition (`claude-opus-5-5`, relayed by `claude-sonnet-5-5`) |

**Reproduction check.** Before any agent ran, the restored clones and the RL manager were
re-simulated (`28_EXP1_baselines`, 1000 games each). Under experiment 1's own formula they
reproduce the recorded payoffs within about one standard error:

| Manager | Recorded | Reproduced | Policy-finder score |
|---|---|---|---|
| Deployed RL (17, "pgh") | 656.4–659.0 | 660.7 ± 1.8 | 659.8 ± 1.8 |
| Zero punishment | 587.1–592.2 | 587.9 ± 2.2 | 591.1 ± 2.2 |
| Human-like manager clone | 550.5 | 546.3 ± 3.5 | 546.3 ± 3.5 |

The best RL candidate of experiment 1 (`gamma_08_heavy`, 662.6 recorded) was
`18_exp2_gamma_8_heavy`, which is no longer here, on Raven or on S3, so it is left out.

## Scores

| gen | versions | best GP peak | best best-of-6 | mean GP peak | campaign best so far | improved on their lineage |
|---|---|---|---|---|---|---|
| 1 | 12 | 649.75 (`-3`) | 649.75 | 642.36 | 649.75 | – |
| 2 | 4 | 651.87 (`-3-v2`) | 653.81 | 650.12 | 651.87 | 3 of 4 |
| 3 | 4 | 656.05 (`-7-v3`) | 656.05 | 650.46 | 656.05 | 2 of 4 |
| 4 | 4 | 655.91 (`-3-v4`) | 656.55 | 652.57 | 656.05 | 2 of 4 |
| 5 | 4 | 654.06 (`-8-v5`) | 656.64 | 652.42 | 656.05 | 1 of 4 |
| 6 | 4 | 655.87 (`-3-v6`) | 655.90 | 653.26 | 656.05 | 1 of 4 |
| 7 | 4 | 655.32 (`-3-v7`) | 656.01 | 654.13 | 656.05 | 2 of 4 |

The same four lineages held the slots from generation 2 on. Final scores carry a standard
error of about 1.3.

| Lineage | GP peak by version (v1 to v7) |
|---|---|
| `-7` | 649.7, 647.9, **656.0**, 653.9, 652.8, 653.5, 655.3 |
| `-3` | 649.7, 651.9, 648.3, **655.9**, 653.9, 655.9, 655.3 |
| `-8` | 648.2, 651.0, 651.8, 650.6, 654.1, 654.3, **654.5** |
| `-11` | 648.8, 649.7, 645.7, 649.9, 648.9, 649.3, **651.5** |

The other eight lineages kept their first version: 647.8, 647.6, 647.2, 646.9, 642.4,
642.3, 623.6 and 614.1. The two lowest are the only mild rules, with fines of at most
about 13 points. All 36 versions beat never punishing (591.1); none of the campaign's own
finals reached the RL manager's 659.8 on the reproduction seed. Those finals are not
directly comparable with the RL manager's numbers: every version's final ran on the same
seeds (1000 on, in batches of 1000 games), the RL manager on others.

## Head to head on fresh seeds

Three runs, all of the three lineage bests at their GP peaks and the deployed RL manager,
each alone on the clones; no sweep, final run or baseline used these seeds:
- `30_EXP1_pf_vs_rl`: seed 6000 on Raven, 2000 games, with zero punishment and the
  human-like clone too (job 31924549);
- `31_EXP1_confirm/seed_7000`–`7004` on the local CPU image (torch 2.5), 2000 games each,
  run while Raven refused logins;
- the same five configs on Raven (torch 1.11, jobs 31933686–31933693), fetched to
  `plots/simulation/31_EXP1_confirm_raven/`.

The two platforms draw different random streams for the same seed, so the local and Raven
runs of seeds 7000–7004 are independent games. Each unbatched sim draws all its games from
one stream, so managers do not share games, and a difference carries about √2 times a
single score's error.

| Manager | Seed 6000 | 7000–7004 local | 7000–7004 Raven | All fresh seeds | vs RL | Campaign final |
|---|---|---|---|---|---|---|
| Deployed RL | 657.2 ± 1.2 | 657.1 ± 0.6 | 656.1 ± 0.6 | **656.8 ± 0.4** (23,000 games¹) | – | – |
| `-3-v4` (pf_3) | 654.6 ± 1.3 | 655.9 ± 0.6 | 656.1 ± 0.6 | **655.9 ± 0.4** | −0.9 ± 0.5 | 655.91 |
| `-7-v3` (pf_7) | 658.0 ± 1.3 | 654.7 ± 0.6 | 654.7 ± 0.6 | 655.0 ± 0.4 | −1.8 ± 0.5 | 656.05 |
| `-8-v7` (pf_8) | 655.1 ± 1.3 | 654.7 ± 0.6 | 655.0 ± 0.6 | 654.9 ± 0.4 | −1.9 ± 0.5 | 654.47 |
| Zero punishment | 587.7 ± 1.5 | – | – | 588.9 ± 1.1 (4,000²) | | |
| Human-like clone | 548.8 ± 2.4 | – | – | 548.7 ± 1.7 (4,000²) | | |

¹ with the reproduction run (seed 3000, 1000 games, 659.8). ² with `28_EXP1_baselines` and
`29_EXP1_reference`.

Per 2000-game run the managers spread over 2.5–5 points (`-7-v3`: 653.2 to 658.0), about what
a standard error of 1.3 implies for eleven runs. Under experiment 1's own formula (payoff
recomputed on every row, seed 6000) the order is the same.

**Survivor bias.** The top three were chosen as the best of 36 campaign finals, which would
inflate their finals. On fresh seeds they moved by −1.0, 0.0 and +0.4: little inflation,
since all finals shared seeds and the ranking compared like with like. What made `-7-v3`
look like a tie was the single head-to-head seed, not the selection. The campaign ranked
`-7-v3` first and `-3-v4` second; on fresh seeds their order flips (0.9 ± 0.5 apart), so the
campaign's ranking of its top versions is itself within noise.

**Where the points go.** Over the 22,000 fresh games the rules punish 10–12 points more per
game than the RL manager (102–104 against 92.5), most of it in rounds 4–6, and get 0.25–0.35
more contribution per player-round (15.8–15.9 against 15.6). The extra contribution is worth
about +3.5 to +5 per game (0.6 per point contributed), the extra fines cost about 5–6 (each
point costs the fined player 1 and the pool 1, so 0.5 of the group mean). The net, −1 to
−2, is the measured gap. The cumulative payoff gap by round (chart 4 on the page) shows it:
the rules fall 4–6 points behind in rounds 4–6 and win back 1–3 in the last eight rounds.

**Same opening, different tail.** Mean punishment per valid player, by round (seed 6000; seeds 7000–7004 agree within 0.2):

| Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7–14 | 15–17 | 18–23 |
|---|---|---|---|---|---|---|---|---|---|---|
| `-7-v3` | 7.40 | 4.69 | 3.18 | 2.40 | 1.97 | 1.70 | 1.49 | 0.31–0.38 | 0–0.43 | 0 |
| Deployed RL | 7.34 | 5.00 | 3.50 | 2.25 | 1.46 | 0.84 | 0.60 | 0.50 → 0.11 | 0.12–0.08 | ≤ 0.05 |

The first four rounds match to within half a point. From round 4 the rule keeps its
opening dose two rounds longer, then switches to fining only free riders at a flat rate
until round 15; the RL manager fades smoothly to almost nothing by round 15 and trails off
until round 22.

Mean punishment by contribution:

| Contribution | 0–2 | 3–5 | 6–9 | 10–12 | 13–15 | 16–20 |
|---|---|---|---|---|---|---|
| `-7-v3`, rounds 0–5 | 27.4 | 16.4 | 10.1 | 5.2 | 1.4 | 0 |
| Deployed RL, rounds 0–5 | 27.8 | 16.7 | 7.7 | 2.6 | 1.2 | 0.56 |
| `-7-v3`, rounds 6–23 | 8.4 | 0.6 | 0.3 | 0.3 | 0.1 | 0 |
| Deployed RL, rounds 6–23 | 4.3 | 1.0 | 0.1 | 0.1 | 0.0 | 0.0 |

In the opening the two dose zero to five points almost identically; the rule is harder on
middling contributions (6–12) and never fines a contribution of 16 or more, which the RL
manager occasionally does. Later the rule is about twice as hard on free riders and
leaves everyone else alone. Contributions follow: the rule's groups give 0.2–0.5 more per
round from round 4 on (16.9 against 16.6 at round 8, 14.9 against 14.4 at round 23, seeds
7000–7004), which does not pay for the extra fines.

So the policy finder, given the RL manager's clones and objective and none of its
behaviour, wrote six-parameter rules that play like it and score 1–2 points (0.1–0.3%)
less, mostly because they keep the opening dose two rounds too long. The rule can
be read in full; the RL manager is a recurrent graph network. `-7-v3` at its GP peak:

```python
cc = c[..., -1]                                    # this round's contribution (0 if timed out)
shortfall = th.clamp(20 - cc, min=0) / 20
fine = 29.7 * th.pow(shortfall, 2.18)              # 29.7 for 0, 15.9 for 5, 6.6 for 10, 1.5 for 15
opening = (t < 7).float()                          # rounds 0-6: everyone below 20
net = ((t >= 7) & (t < 16)).float() * (cc <= 2).float()   # rounds 7-15: only 0-2
punishment = fine * (opening + 0.85 * net)
```

Timed-out players are never fined, whatever the rule returns.

## Ideas

Categories come from Claude reading each rule at its final values (the
`policy-finder-summarizer` skill), in `pf_runs/worktrees_exp1/summary.json`. Experiment 1
has no switches, so its categories are drawn up afresh rather than taken from experiment 2.

| Policy category | What it does | Versions | Mean | Best |
|---|---|---|---|---|
| opening shock + late net | steep convex fines on everyone below a high bar for the first 5–10 rounds; then only near-zero contributors are fined, hard, until round 11–17; then nothing | 14 | 654.2 | 656.05 (`-7-v3`) |
| front-loaded stop | fines everyone below a bar from round 0, then stops or fades out by round 7–14 | 21 | 645.7 | 651.87 (`-3-v2`) |
| sustained fade | fines below a bar through most of the game, fading toward the end | 1 | 642.2 | – |

The worst two-phase version (651.5) roughly ties the best one-phase version (651.9). Every
best lineage started as a linear, one-phase rule at about 648–650, gained when it moved to
a convex dose, then gained again when it added the late net. Lineage `-11` reached the
two-phase shape only in its last version (651.5), with a linear dose, a hint that the
convex dose is also worth a few points.

Ideas tried and dropped:
- a strike limit, which stops punishing a player already punished 3 times (`-11-v3`,
  645.7, the lineage's lowest);
- in `-7-v5` to `-v7`, an opening that fines only players near or below the group's mean;
- in `-8-v7`, a late phase that fines players dropping from a full 20 instead of free
  riders (654.5, the lineage's best by 0.2).

## Process

| | |
|---|---|
| Versions | 36 run, 36 scored, none failed |
| Wall clock | 4 h 21 min (01:03–05:24), generations of 23–52 min |
| Agent time per version | median 11.0 min (4.5 to 36.6) |
| Agent tokens | Opus 5.5: 1.16 M output, 50.7 M cache read, 2.95 M cache write; Sonnet 5.5 relay: 47 k output |
| API-equivalent cost | $48.95 in all, median $1.19 per version (on the plan, not billed) |
| Raven | 2.7 h of sweep jobs (2 parts in parallel) and 2.5 h of final runs, wall clock |

Building the setting:
- the restored models;
- a loader for the old RL checkpoints;
- the `exp1` rule inputs, sweep metric, export and agent definition;
- the reproduction and reference sims;
- two smoke runs.

That took about 2 hours before the campaign. The experiment 2 path was left byte-identical.

## Leak audit

The opening fines match the RL manager's closely, so a separate agent audited whether
anything about the RL manager could have reached the policy-finder agents. It rebuilt the
agents' view with the export code at the campaign's base commit and searched all 38 agent
transcripts (36 campaign, 2 smoke), tool results included.

- **The view.** It is copied without `.git`. `28_EXP1_baselines.yml`, the only file in the
  worktree with RL numbers, is denied, and so is `artifacts/manager/`. The 30/31 configs and
  plots, `temp/`, `experiment_analysis/`, notebooks and the experiment 1 reports are outside
  the instance checkout. `29_EXP1_reference` holds only the zero and human-like runs.
  `basics_exp1.md` and the agent definition describe the game and the data, not a strategy.
- **The container.** Network egress is limited to `api.anthropic.com`, and the agent
  definition disallows WebFetch and WebSearch. Clone and RL checkpoints are not in the
  container.
- **The sweeps.** Exp1 sweeps run no anchor, so they never simulate the RL manager.
- **The transcripts.** The only tools used were Bash, Read, Write, Edit and Agent. No web
  call and no git command was made. There were zero hits for the RL run's names or paths.
  The remaining "RL" mentions are code comments and filenames of old configs that were
  never opened.

The top lineages reasoned their way to the shape from the pilot data and the reference sims:
- never punishing beats the human-like manager on the clones;
- a fine has to buy about 3.3 contribution points later to pay;
- its effect persists in the pilots;
- players giving 16 or more do not respond to fines.

The sweeps then tuned the opening against the same clones the RL manager was trained on.
Two optimisers of the same objective on the same simulator ending at similar opening fines
is what one would expect.

The audit could not check:
- the original exports, which were deleted, so the view was rebuilt;
- the egress rule at the network level;
- every transcript read in full.

## Confounds

- **Inputs.** The rules read the full history. The RL manager read this round's
  contributions and validity, last round's punishment and the round, plus whatever its RNN
  carried. Two of the three best rules (`-7-v3`, `-3-v4`) use only this round's
  contribution and the round number. The third (`-8-v7`) also reads last round's
  contribution. None asks for more than the RL manager had.
- **The clones were trained on 8- and 16-round pilots** and play 24 rounds here. That holds
  for the RL manager too, and the comparison is on the clones, not on humans.
- **Search budget.** The RL manager was trained for 100k DQN updates with hand-tuned
  hyperparameters, and its candidates were chosen among several runs. The policy finder ran
  36 agent sessions and about 9,000 simulated sweep points. Neither side's budget was
  matched to the other's.
- **Selection.** The campaign's top was picked from 36 noisy finals, and the RL manager was
  deployed after experiment 1's own comparisons. The fresh seeds remove both from the
  estimates above, but not from which managers were compared.
- **Platform.** Seeds 7000–7004 ran twice, on the local CPU image (torch 2.5) and on Raven
  (torch 1.11). Their random streams differ, so these are different games; the gaps to the
  RL manager differ between the two by about 1 point, within their errors (± 0.8).
