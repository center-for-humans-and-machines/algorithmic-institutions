# [ACTIVE] Policy finder with common games, fresh seeds and a paired baseline

Follows `docs/plans/archive/policy-finder-pool-difference.md`; report
`docs/reports/policy_finder_pool_difference.md`, "Why the search fell short".

## Why

The exp2_diff campaign's search was blind in its later generations.
- From version 2 on, a sweep's points spread by about 2.7. A single point's noise is 2.85, so
  the spread is all noise.
- Finals reused the same 2000 games for every version. Seven generations of selection on them
  rewarded rules that suited those games: `-5-v7` gains 2.52 over `zero` on them, 1.42 on fresh
  seeds.
- Rules played on identical games are correlated 0.89–0.96 per game. Paired differences need 9–25×
  fewer games than unpaired ones.

## Changes (branch `pf-crn`, kept only if the campaign is better)

1. **Common games within a sweep.**
   - Every design point plays the same games: its batches are seeded by their position within the
     point, not by their global index.
   - `simulate.py`: a sim key `seed_cycle: k` seeds batch i with `seed + i mod k`.
   - `generate_sim_config.py`: for a setting with `common_games`, each batch holds exactly one
     point's games, and `seed_cycle` is the point's batch count. All parts of a sweep share one
     seed.
2. **Fresh seeds per version.** Sweep and final seeds come from the version's name (CRC32): one
   seed for its sweep, another for its final. Nothing is selected twice on the same games, and
   the fixed map of lucky design points is gone.
3. **Finals paired with a baseline.**
   - Each final also plays `zero` against `ah` on the same games, in both seats.
   - Every final point gets `gain`, its paired lead over `zero`'s lead, game by game (`gain_se`).
   - The pipeline records it, and the campaign ranks lineages by it.
   - `zero` and not a stronger rule: a reference rule in the finals would show the agents another
     instance's rule, which the export's leak check rejects. Agents already see `zero` in the
     reference sims.
4. **Sweep size unchanged:** 256 points × 500 games.

The agent's definition says that a sweep's points share their games and that the final's `gain`
ranks versions. The GP still fits the lead.

## The test

Same campaign shape as `exp2-diff-2026-11`: 12 fresh lineages plus 6 generations on the top 4,
on `pf-crn`, prefix `pf-20261113`.

Then a head-to-head on fresh seeds, itself with common games, so every row plays the same games:
- the new top 3 and the old top 3;
- the multi leader (`-9-v5`, w = 0) and `zero`;
- each against `ah` in both seats.

**Better** means the new campaign's best fresh-seed lead is clearly above the old campaign's best
(`-6-v6`, 15.96). A rule above the multi leader (17.11) would settle it.

One campaign each is one draw, so a small difference is not evidence. The report says how much
of the gain is the method and how much could be luck.

## Progress

- 2026-11-13: plan; implementation started on `pf-crn`.
- 2026-11-13: changes 1–4 built and tested (`1a864f06`).
  - A local sim of a final with a non-punishing point gives a gain over `zero` of exactly 0.0, game
    for game.
  - Evaluation scripts: `exp2_diff_crn_h2h_configs.py` and `exp2_diff_h2h.py --paired`.
- 2026-11-13 16:51: blocked at the smoke.
  - The policy-finder subagent's first Opus request is stopped by the API's safety classifier
    (`reasoning_extraction`), every attempt.
  - It happens with yesterday's unchanged agent definition too (`yannik/handover`), so it is not
    this branch's edits.
  - Request IDs: `req_011Ch1GoTnfARuH2rZxJA4Xw`, `req_011Ch1GqieBGWvyf76y1o1D8`,
    `req_011Ch1GwivTa61sqXe696cMe`, `req_011Ch1Gyyb5HXjhQMF5HkJJh`.
  - No campaign started.
