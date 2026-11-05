# Policy finder against several governors (exp2_multi)

Plan: `docs/plans/archive/policy-finder-multi-opponent.md`. Task:
`docs/tasks/a_new_kind_of_autoresearch.md`. Page: the exp2_multi observation page,
https://claude.ai/artifact/BgP93sxr11nxwZ4poFvjLg (built from `pf_runs/view_multi`).

## Summary

- The campaign ran 12 fresh lineages and 6 more generations on the best 4: 36 versions, all
  scored, in 5 h 6 min, $42 API-equivalent. Each rule was scored as the plain mean of its
  group's pool against four fixed governors, in both seats. The agents did not know who
  the governors were.
- **No new rule beats `prev_best`** (`pf-20261104-12-v7`, the earlier exp2 campaign's best).
  `prev_best` was found against `ah` alone and was one of the four opponents here. On
  six fresh seeds (24,000 games per opponent), `prev_best` scores 69.67 ± 0.16 and the best
  new rule, `pf-20261106-7-v5`, 68.43 (−1.24 ± 0.23). `prev_best` is ahead on every seed
  and against every opponent, including the held-out ones and self-play. Cem's best
  (68.77) is also ahead of the new rules, though not decisively (−0.34 ± 0.23 against
  `-7-v5`).
- **Why the new rules lose.** They have nearly the same structure as `prev_best`, but fine
  less in rounds 4–15. That saves pool early and costs more from round 12 on, when their
  groups give less. Group sizes match within 0.3.
- **Adaptation bought nothing.** Eleven of twelve first versions read the other group's
  fines. Each lineage measured the effect and dropped it. Setting the remaining
  adaptation weight of `-9-v5` to 0 changes its score by −0.01 ± 0.22.
- **The agents profiled the opponents accurately** from per-round data, under opaque names.
  None named or guessed an identity.
- **Campaign finals ran 0.4–1.0 high** for the three lineage bests (survivor bias, as in
  exp1).

## Setup

- **Game.** Experiment 2's clones (the group-switching contribution, validity and switch
  models), 8 players in two groups, 24 rounds, switching after rounds 3, 7, 11, 15 and 19.
  The rule governs one group; an opponent governs the other.
- **Score.** Definition 4 of #226. Per round, the group pool is 1.6·Σc − Σp over the
  group's members; an empty group scores 0. The pool is averaged over rounds and games and
  pooled over both seats against each opponent. The objective is the plain mean over:
  - Levin's rule at `sigmoid_opt_pool` (`levin_sigmoid.yml`);
  - the human-manager clone `ah`;
  - `prev_best`;
  - `zero` (never punishes).
- **Budget.** Sweeps play 256 Sobol points × 64 games per opponent and seat; finals 6 points
  × 1000.
- **What agents saw.** Only the objective "good against any governor". From version 2 on,
  they also saw their own results by opponent under the opaque keys `opponent_1..4`, and
  the per-round frames of their sims. No reference sims, no names. The export's leak
  check rejects any file naming an opponent.

## Campaign

| Version | Final, mean | opp_1 Levin | opp_2 `ah` | opp_3 `prev_best` | opp_4 `zero` |
|---|---|---|---|---|---|
| `pf-20261106-9-v5` | 68.93 | 70.82 | 69.94 | 73.98 | 61.00 |
| `pf-20261106-11-v7` | 68.91 | 70.94 | 70.00 | 74.38 | 60.31 |
| `pf-20261106-7-v5` | 68.86 | 71.65 | 69.20 | 74.15 | 60.46 |
| `pf-20261106-5-v3` | 68.52 | 70.15 | 70.00 | 72.98 | 60.93 |

- The generation-1 best was 65.6 (`-9`).
- From generation 2, the same four lineages (`-9`, `-11`, `-7`, `-5`) continued.
- The best reached 68.93 in generation 5 and did not move after.
- Full table: `pf_runs/campaign/exp2-multi-2026-11.summary.md`.

## Head to head on fresh seeds

`33_MULTI_h2h`, seeds 20000–25000 on Raven. Configs come from
`scripts/data_analysis/exp2_multi_h2h_configs.py`, scores from
`scripts/data_analysis/exp2_multi_h2h.py` (→ `plots/simulation/33_MULTI_h2h/h2h_summary.json`).
- **Rows:** the three lineage bests at their GP peaks, a `w = 0` ablation of `-9-v5`, and
  five references.
- **Columns:** each training opponent, the held-out Cem's best (`pf-20261008-3-v6`) and
  #207's `thr9_p10`, and self-play.
- **Games:** both seats, 2000 games per pairing and seat per seed, so 24,000 games per cell.
- **Errors:** cells are independent games. Standard errors are about 0.32 per cell and 0.16
  for the mean of four.

| Governor | Mean of 4 | vs `prev_best` | Levin | `ah` | `prev_best` | `zero` | Held out | Itself | Final |
|---|---|---|---|---|---|---|---|---|---|
| `prev_best` | **69.67** | – | 70.83 | 72.91 | 73.70* | 61.26 | 74.52 | 73.70 | – |
| Cem's best | 68.77 | −0.90 | 70.67 | 71.69 | 72.62 | 60.11 | – | 71.00 | – |
| `-7-v5` | **68.43** | −1.24 ± 0.23 | 70.70 | 71.41 | 72.09 | 59.51 | 73.45 | 72.34 | 68.86 |
| `-11-v7` | 68.04 | −1.63 ± 0.23 | 70.06 | 70.57 | 71.88 | 59.66 | 72.82 | 70.72 | 68.91 |
| `-9-v5` | 67.98 | −1.70 ± 0.23 | 69.09 | 71.04 | 71.73 | 60.06 | 72.62 | 69.27 | 68.93 |
| `-9-v5`, w = 0 | 67.99 | −1.69 ± 0.23 | 69.30 | 70.63 | 71.95 | 60.06 | 72.35 | 66.77 | – |
| Levin's rule | 66.23 | −3.44 | 68.09* | 68.73 | 70.29 | 57.83 | 70.61 | 68.09 | – |
| `zero` | 58.43 | −11.24 | 60.14 | 60.19 | 62.38 | 51.02* | 62.62 | 51.02 | – |
| `ah` | 53.94 | −15.74 | 56.14 | 55.76* | 57.71 | 46.15 | 58.42 | 55.76 | – |

\* A governor against itself is its self-play.

- **`prev_best` per seed:** 69.31–69.90; it is first on all six seeds. The best new rule per
  seed ranges 67.90–68.73.
- **Without `prev_best`'s column**, the gap remains: over Levin, `ah` and `zero`,
  `prev_best` scores 68.33 and `-7-v5` 67.21.
- **Ranking among the new rules:** `-7-v5` leads `-11-v7` by 0.39 ± 0.22 and `-9-v5` by
  0.45 ± 0.22 on the objective, and by 0.6–0.8 ± 0.3 on the held-out pair. That is
  suggestive, not decisive. The campaign's finals ranked them the other way round, within
  0.07.
- **Survivor bias:** fresh minus own final is −0.95 (`-9-v5`), −0.87 (`-11-v7`) and −0.43
  (`-7-v5`). On the seed-8000 reference matrix `prev_best` scored 69.99, against 69.67 here.

## Why the new rules lose

The three bests have the same structure as `prev_best`:
- they fine a member's shortfall below a norm built from the other members' contributions;
- they fine mostly in each 4-round block's opening rounds;
- the fine fades by a factor per block, and stops by round 16–20.

The rules:
- **`prev_best`:**
  - bar = 0.4·8.6 + 0.6·(mates' mean), at most 15;
  - 9.2 per point short in rounds 0–1 of each block, ×0.56 per block;
  - rounds 0–15 only.
- **`-7-v5`:**
  - target = 0.46·10 + 0.54·(others' mean);
  - 5.0 per point short in a block's first round, ×0.61 in the second, ×0.38 in the
    third, none in the fourth;
  - ×0.56 per block.
- **`-11-v7`:**
  - shortfall below the others' mean minus 1.4;
  - 8.6 per point in a block's first round, ×1.32 in its second, none in its last two
    rounds (before round 20);
  - ×0.44 per block, fading out over rounds 20–23.
- **`-9-v5`:** only members in their first four rounds with the group.
  - Per point below the group mean minus 1.1: 12.2 in round 0, then 3.9·(1 + 0.3·q),
    where q is the other governor's mean fine so far.
  - No fines in a block's last two rounds or from round 16.

The by-round frames (`scripts/data_analysis/exp2_multi_charts.py`, charts 3–5 on the page)
show where the gap comes from:

- **Fines.** The new rules fine 0.3–1 point less per member in rounds 4–15 against every
  opponent. `-9-v5` fines up to 2.2 less against `zero`. Rounds 0–3 are within 0.5,
  except `-9-v5`.
- **Pool over the game.** Summed over rounds and divided by 24, each new rule leads by
  0.3–2 points after the first blocks, because it spends less on fines. It loses that lead
  mostly from round 12–16 on, when `prev_best`'s groups still contribute more.
- **Members.** Group sizes stay within 0.3 of `prev_best`'s. The extra fines do not drive
  players out.

So the new rules settled on a milder dose than `prev_best`'s. The sweeps could not show
that this was a mistake:
- a single sweep point has a standard error of about 2.1 on the mean (4 opponents × 128
  games);
- along `-7-v5`'s slope `k`, the sweep's averages over k = 4–13 are flat within that noise
  (66.1, 65.4, 65.9 for k in (4, 6], (6, 9], (9, 13]).

The exp2 campaign that found `prev_best` used 500 games per point against `ah` alone. That
is a narrower objective, but it was measured about as precisely per point.

## Adaptation

- **Version 1.** Eleven of the twelve first versions read the other group's fines (`p`
  masked to players outside the group). The aims, quoted from their notes:
  - to stay the milder manager ("so the group stays the more lenient one", `-1`; "below 1
    keeps this group the milder one, which draws switchers", `-10`);
  - or to take licence from a harsh one ("the other manager's harshness licenses ours",
    `-9`; "a harsher rival lets us enforce more without losing members", `-3`).
- **Later versions.** Each continuing lineage measured the effect in its sweep and dropped
  it:
  - `-5-v2`: "moved the fitted pool by 0.13";
  - `-7-v2`: v1's relative cap "shut punishment off next to the non-punishing opponent 4";
  - `-9-v2`: "t_end and w do nothing (effects 0.0001 and 0.02)";
  - `-11-v3`: an opponent-scaled term "negative against every opponent".
- **`-9-v5`** reintroduced a licence weight after its per-opponent peaks lined up with the
  opponents' harshness. It estimated that "an oracle that knew the opponent would add
  ~1.9" (`-9-v4`). `-9-v6` dropped it again ("length scale 100").
- **The ablation agrees.** `-9-v5` with w = 0 is 67.99 against 67.98, on the objective and
  on every training opponent within 0.5. The weight matters only in self-play (69.27
  against 66.77), where the opponent is itself.

No rule tells the opponents apart to play a different strategy for each. The agents
considered it:
- `-7-v7` rejected telling opponent 4 apart: the per-opponent surfaces were "too noisy";
- `-11-v5` put the most that full tailoring could buy at +0.7.

## What the agents learned about the opponents

By their second or third version, all four continuing lineages had profiled the opponents
from the per-round frames, under the opaque keys. Their profiles were accurate.
- **Levin's rule:** "30 / 18 / 0 / 0 to contributors at 0-4 / 5-9 / 10-14 / 15-20 in round
  0, decaying through each block but spiking again at every block start" (`-7-v3`).
- **`ah`:** "graded mild punishment of everyone … steady over the game" (`-7-v3`).
- **`prev_best`:** "30 / 16 to 0-4 / 5-9 contributors in the first two rounds of each of the
  first four blocks … decaying per block" (`-7-v3`); "a block-timed style like mine"
  (`-9-v2`).
- **`zero`:** "opponent 4 never punishes" (all four lineages).

A read-through of every note, rule and analysis script found no name or guess of an
opponent's identity: no Levin, sigmoid, clone, `ah` as a governor, or earlier campaign.
The export's leak check (`leak_terms`) passed for all 36 runs.

## Caveats

- **Few opponents.** The objective weights four governors equally. Three of them are harsh
  block-start finers, and their scores move together. A different panel could rank the
  rules differently. The held-out pair, though, agrees with the training panel.
- **Seat.** The clones carry a small group-0 advantage (+0.4 ± 0.15 per group on the
  seed-8000 matrix; plan, Progress). Every score here pools both seats, which cancels it.
- **Search, not the objective, is the bottleneck.** `prev_best` scores higher on this very
  objective, so the objective had room the search did not find. More games per sweep
  point, or seeding a lineage from `prev_best`, would test whether the multi-opponent
  search can improve on it at all.

## Files

- **Configs:** `configs/simulation/opponents/exp2_multi.yml`,
  `configs/simulation/manager_testing/32_LEVIN_multi_reference.yml` and
  `33_MULTI_h2h/seed_*_{rules,refs,ablation}.yml`.
- **Rules:** `configs/managers/rule_based/pf-20261106-{9-v5,11-v7,7-v5}.yml`.
- **Analysis:** `scripts/data_analysis/exp2_multi_h2h{,_configs}.py` and
  `exp2_multi_charts.py`.
- **Instance branches:** `policy-finder/pf-20261106-1..12`.
