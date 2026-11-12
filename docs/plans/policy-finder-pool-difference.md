# [ACTIVE] Policy finder scored by the lead over `ah` (experiment 2)

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
