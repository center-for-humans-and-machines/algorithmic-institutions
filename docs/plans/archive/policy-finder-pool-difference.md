# [DONE] Policy finder scored by the lead over `ah` (experiment 2)

Task: `docs/tasks/a_new_kind_of_autoresearch.md`.

## Why

Every experiment 2 campaign so far scored a rule by its own group's common pool:
- the proxy campaign against `ah`;
- the multi-opponent campaign averaged over four governors.

This campaign scores a rule by how far its group's pool runs ahead of `ah`'s group's pool,
in the games where the two play each other. Only `ah` is played, not an average of
governors.

## What stays

- **The game:** Levin's stack (`24_LEVIN` AH block): 2 groups of 4, 24 rounds, switching
  every 4 rounds.
- **The rule's inputs:** `c`, `valid`, `p`, `in_group`, `t`.
- **The pipeline, the campaign driver and the observation app.**
- **The setting mechanism** (`SETTINGS` in `manager/rule.py`, `generate_sim_config.py` and
  `export_instance.py`).
- **The agent's view is exp2's:**
  - it knows it plays `ah`;
  - it reads the `25_LEVIN_run1` reference sims (`ah` and `zero` against each other) and
    the human data;
  - it sees its lineage's earlier results.

## The objective

Per game, take each group's common pool, `1.6·Σc − Σp`, per round, averaged over the rounds
(an empty group counts 0). The score is the rule's group's pool minus `ah`'s group's pool,
averaged over games.

- **Seats:** the rule plays both seats, half the games each. This cancels the small seat
  effect (+0.37 per group), which a difference would double.
- **What is kept:** `sweep.json` holds both pools beside the score (`own`, `other`) and the
  two group sizes.
- **Games:** 250 per seat per sweep point (500, as in exp2), and 1000 per seat per final
  point (2000).
- **Reference bar:** the `33_MULTI_h2h` head-to-head (24,000 games per pairing, both seats)
  gives every reference manager's lead over `ah`. `ah` against itself is 0 by symmetry.

## Steps

1. **Metric:** `pool_diff_vs_opponents` in `simulation/sweep.py`, beside
   `pool_vs_opponents`.
   - The same games: every point against each opponent in both seats.
   - Scored per game as own pool minus the opponent's pool.
   - `pool_scores.against_anchor` now also returns the anchor's group's pool and size
     (`other`, `other_members`) for the same episodes.
2. **Setting `exp2_diff`:**
   - `rule.py`, `cli.py`, `instance_settings.py`: the setting is exp2's game.
   - `generate_sim_config.py`: opponents file `configs/simulation/opponents/exp2_diff.yml`,
     which holds `ah` only.
   - `pipeline.SETTING_EPISODES`: (250, 1000). The score in the pipeline state also keeps
     `own` and `other`.
   - `fit_surrogate.EDGE_EFFECTS`: 2.0. A member moving between groups changes both pools,
     so the lead spreads about twice as wide as a pool.
3. **The agent's side:**
   - `.claude/agents/policy-finder-exp2-diff.md`: `policy-finder.md` with the new objective,
     both seats and the new `sweep.json` fields.
   - `docs/reports/basics_exp2_diff.md`: `basics.md` with the aim and the score rewritten.
   - Both are installed by the export (`export_instance.SETTINGS["exp2_diff"]`).
   - Every other setting's export denies the new game doc.
4. **Observation app:**
   - `PF_SETTING=exp2_diff` labels the score "lead over ah".
   - The Sweep tab shows both pools and both group sizes.
   - The Agents table adds `own pool` and `ah pool`.
   - References are the h2h leads (`references.json`).
5. **Smoke:**
   - one fresh lineage at small size;
   - then one `--iterate-on` at full size.
6. **Campaign:** `campaigns/exp2-diff-2026-11.yml` (`pf-20261112`): 12 fresh lineages plus
   6 generations on the top 4, 36 versions.
7. **Head to head on fresh seeds:**
   - who: the top 3 against `ah`, both seats;
   - seeds: 20000–25000, the `33_MULTI_h2h` seeds;
   - comparison: the reference managers' leads over `ah`.
8. **Report and artifact:**
   - `docs/reports/policy_finder_pool_difference.md`;
   - an observation page with the head-to-head;
   - what the rules do differently from pool-maximising ones (group sizes, fines, timing).

## Before the campaign

- **Raven reachable from the command server.**
- **Container spawner up.**
- **Disk:** at least about 5 GB free. The proxy campaign's worktrees took 3.8 GB, of which
  3.3 GB were sweep parquets.

## Cost (estimate)

This is the proxy campaign's budget:
- the same games per sweep and final point;
- about 12 GPU-hours;
- about 5 h wall clock;
- about $45 API-equivalent.

## Progress

- 2026-11-12: setting, metric, agent definition, game doc, page and tests built on `pf-diff`.
  - Blocked at start: Raven unreachable from the command server (exit 255), the container
    spawner not resolvable, 2.0 GB free disk.
- 2026-11-12 15:35: blockers resolved (Raven reachable, spawner up, 14 GB free).
- 2026-11-12: smoke passed.
  - The first export failed the leak check: the `33_MULTI_h2h` configs name earlier
    instances. They are now in the global deny list, and exp2_diff exports also leave out
    experiment 1's reference sims.
  - `pfsmoke-diff` (16 points × 50 games per seat, final 6 × 200): lead 11.8 ± 3.2 (own 59.8,
    ah 48.0) in 23 min. The agent read the objective as sorting: fine persistent free riders
    just before a switch so that they move to ah's group, and leave everyone else alone.
  - `pfsmoke-diff-v2` (`--iterate-on`, full size: 256 × 250 per seat, final 6 × 1000 per
    seat): lead 16.4 ± 1.4 (own 61.8, ah 45.4) in 27 min.
- 2026-11-12 16:30: campaign `exp2-diff-2026-11` started on `pf-diff`.
- 2026-11-12 21:10: campaign done.
  - 36 versions, all scored, in 4 h 40 min; $49.5 API-equivalent.
  - 9 of the 12 first finals collapsed to never punishing (16.76 on the finals' seed, the games
    `zero` plays).
  - Lineage bests: `-5-v7` 19.28, `-6-v6` 18.85, `-2-v7` 18.62, `-1-v6` 17.70.
- 2026-11-12: steps 7–8 done.
  - Head-to-head (`34_DIFF_h2h`, seeds 20000–31000, 48,000 games each, against `ah` in both
    seats):

    | Manager | Lead over `ah` |
    |---|---|
    | multi `-9-v5`, w = 0 | 17.11 ± 0.33 |
    | `-6-v6` | 15.96 |
    | `-2-v7` | 15.88 |
    | `-5-v7` | 15.65 |
    | Cem's best | 15.08 |
    | `prev_best` | 14.89 |
    | `zero` | 14.23 ± 0.29 |

    The new rules beat `zero` by 1.4–1.7. None beats the multi rule.
  - Sweep noise: lead se about 2.9 per point. Shared random streams by design index: point 245
    gives `zero`-like rules 20.83.
  - Report: `docs/reports/policy_finder_pool_difference.md`.
  - Page: https://claude.ai/artifact/HXiGzw3FpZkv8Hq4arGrAk.
