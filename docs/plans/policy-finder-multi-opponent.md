# [ACTIVE] Policy finder against several governors (experiment 2)

Task: `docs/tasks/a_new_kind_of_autoresearch.md`.

## Why

Every experiment 2 policy-finder campaign so far scored a rule on one number: the common
pool of its group against `ah`, the clone of a human governor. The best rules are tuned to
that one opponent. The task asks for rules that do well against several governors, and
expects good rules to adapt to whatever the other governor does. They may read the other
governor's punishments, but never its identity.

## What stays

- **The game:** Levin's stack (`24_LEVIN` AH block), 2 groups of 4, 24 rounds, switching
  every 4 rounds.
- **The rule's inputs:** `c`, `valid`, `p`, `in_group`, `t`.
  - `p` already holds every player's punishment by either governor, so the rule sees the
    other governor's behaviour.
  - No input names the opponent, and none is added.
- **The pipeline:** export, agent, sweep, GP fit, final run, the campaign driver and the
  observation app.
- **The setting mechanism** added for experiment 1 (`SETTINGS` in `manager/rule.py`,
  `generate_sim_config.py` and `export_instance.py`).

## The objective

The score is the mean of four common goods. Each is the pool `1.6·Σc − Σp` of the rule's own
group, per round, averaged over rounds and games, against one opponent:

| Opponent | What it is | Manager config |
|---|---|---|
| `ah` | clone of a human governor (multinomial logistic regression + severity copula) | as in the `25_LEVIN_run1` template |
| `zero` | never punishes | `DummyManager(0)` |
| `levin` | Levin's sigmoid rule (#219) at `sigmoid_opt_pool` (punishes up to 30), deleted in #235; restored and ported to today's inputs (`cc = c[..., -1]`) | `configs/managers/rule_based/levin_sigmoid.yml` + params |
| `prev_best` | the best experiment 2 rule found so far, fixed for the campaign | `pf-20261104-12-v7` at its GP peak (`27_LEVIN` config) |

- **Weighting:** plain mean of the four raw pools, all in the same units.
- **Seating:** the rule plays each opponent from both groups, half the games each. A
  point's pool against an opponent pools both seats, so no score depends on the
  seat.
- **The four pools** are kept per point in `sweep.json` and in every final, not only their
  mean. The GP fits the mean.

## Steps

1. **Restore the opponents.**
   - Port Levin's sigmoid rule from `origin/230-rule-config` to the current rule format.
   - Check that it reproduces #219's pool against `ah`: 69.8 at 2000 games, seed 1000, on
     the pre-#245 code. A port that changes behaviour shows up there.
   - Pin `prev_best` as a rule YAML + params file.
   - Add both, and every config that names the opponents, to the export's `DENY`.
2. **Reference matrix** (`32_LEVIN_multi_reference`).
   - Every opponent against every opponent: `ah`, `zero`, `levin`, `prev_best`, plus Cem's
     best as a held-out manager.
   - 2000 games per pairing, batched, a fresh seed.
   - This gives each reference manager its four-opponent score, the bar a new rule must
     clear. It is for the report and the page only; agents never see it.
3. **The `exp2_multi` setting.**
   - `generate_sim_config.py`: a setting with a list of opponents instead of one anchor. One
     pairing per point per opponent and seat (8); 64 games per pairing in sweeps (512 per
     point, about today's 500) and 1000 per pairing in finals (8000 per point).
   - `sweep.py`: a metric `pool_vs_opponents`. Reuse `pool_scores.against_anchor` per
     opponent, then take the mean.
     - `sweep.json` keeps `pool` (the mean, so the fit, pipeline and campaign read it
       unchanged), plus `pools: {opponent: (mean, se)}`.
     - `merge_sweeps` checks that the opponent list matches across parts.
   - The fit: `EDGE_EFFECTS` entry for the new metric.
   - The pipeline: final runs at the GP peak + top 5 against all four opponents.
   - Exp2 and exp1 configs must regenerate byte for byte, as they did for the exp1 change.
4. **The agent's side: nothing about the opponents up front, their own results by opaque
   name afterwards.**
   - The objective as the agent reads it: write a rule that keeps its group's common good
     high whatever the other governor does. The opponents are not named, counted or
     described.
   - The rule learns about its opponent only in simulation, through `p`, the other
     governor's punishments, like any other input.
   - Every sweep and final reports the mean and the four per-opponent pools under opaque,
     fixed keys (`opponent_1` … `opponent_4`). The same key always means the same opponent,
     across points, versions and lineages.
     - A first version therefore knows nothing about the opponents.
     - A later version sees its lineage's earlier results. From those it can learn that the
       opponents differ, and where its rule does well or badly.
   - The real names live only on the pipeline's side.
     - Sweep and final configs, outputs, plots and `sweep.json` use the opaque keys.
     - The key → opponent map sits in the setting, which is outside the export.
     - The leak check in `export_instance.py` rejects any export holding a real opponent's
       name or rule file.
   - No reference sims of any governor are exported. `25_LEVIN_run1` (`ah` and `zero`
     against `ah`) leaves the exp2-multi view.
   - The game doc is a variant, `docs/reports/basics_exp2_multi.md`, swapped in as exp1's
     was. It is `basics.md` with "The simulation" rewritten:
     - the rule manages one group and another governor, unnamed, the other;
     - the score is the rule's group's common pool, averaged over the opponents it meets.

     No `ah`, and no comparison with another group's pool. Agents keep the game
     description and the human data.
   - A `policy-finder-exp2-multi` agent definition carries the new objective. The export
     installs it as `policy-finder`, as for exp1.
5. **Observation app.**
   - Per-version columns for the four pools beside the mean, in the Agents table, the Sweep
     tab and the campaign summary.
   - References as a 4-column matrix.
   - The task asks for the four scores to be kept apart.
6. **Smoke:** one fresh lineage at small size, then one `--iterate-on` at full size, as
   for exp1.
7. **Campaign:** `campaigns/exp2-multi-2026-11.yml`, 12 fresh lineages + 6 generations on
   the top 4, 36 versions.
8. **Head to head on fresh seeds.** Each run plays every opponent on a seed no sweep used:
   - the campaign's top 3;
   - `prev_best`, Cem's best, Levin's rule, `ah`, `zero`.

   Three groups of opponents:
   - the four training opponents;
   - held-out ones: Cem's best and a threshold rule from #207 (`thr9_p10`);
   - self-play: each rule against itself.

   Repeat on more seeds until the decisive gaps are clear of their errors (the exp1 lesson:
   one 2000-game run misled).
9. **Does the rule adapt?**
   - Punishment profiles of the top rules by opponent (by round, by contribution, by the
     opponent's recent punishments).
   - Against the best non-adaptive version in the campaign, i.e. one that never reads
     other-group `p`.
   - Whether the rule's code reads the other group's punishments at all.
10. **Report and artifact.**
    - `docs/reports/policy_finder_multi_opponent.md`.
    - An observation page with the four scores separated and comparison charts like
      exp1's.

## Decisions

Agreed 2026-11-05:

- **Opponents:** `ah`, `zero`, Levin's rule at `sigmoid_opt_pool`, and
  `pf-20261104-12-v7` at its GP peak, fixed for the whole campaign.
- **What agents see:** nothing about the opponents up front.
  - The objective is a rule that is good against any governor.
  - The rule learns its opponent only through the punishments it observes in simulation.
  - Agents see their own results as the mean plus four pools under opaque, consistent keys,
    so from the second version on they can tell that opponents differ.
  - No reference sims of any governor.
- **Weighting:** the plain mean of the four raw pools.
- **Games:** 125 per opponent per sweep point, 2000 per opponent per final point.
  Revised 2026-11-05: both seats, so 64 per opponent and seat per sweep point (128 per
  opponent), 1000 per opponent and seat per final point (2000 per opponent).

Defaults, unless changed:

- **Start:** 12 fresh lineages, then 6 generations on the top 4, 36 versions, as in both
  earlier campaigns.
- **Held-out opponents** for the head-to-head: Cem's `pf-20261008-3-v6`, #207's
  `thr9_p10`, and self-play.

## Cost (estimate)

The last exp2 campaign took about 12 GPU-hours (4.4 h of sweep jobs, 2.9 h of finals, wall
clock) and $44 API-equivalent.

This campaign plays 125 games per opponent per sweep point and 2000 per final point:
- sweeps take about as long as before;
- finals take 4×, about 11–12 h of final jobs instead of 2.9 h;
- about 20–25 GPU-hours in all, and the agents' cost as before.

## Progress

- 2026-11-05: decisions agreed (above); implementation started on `pf-multi-opponent`.
- 2026-11-05: steps 1–5 built (3dc21c2a, a9852a91, 044dfbcc, 453744d7, 6a69054b).
  - Levin's rule ported. On random histories it returns exactly the original's
    punishments. In group 0 it scores 69.1 ± 1.0 against `ah` (#219: 69.8; the pre-#235
    code run here: 69.5 ± 1.5).
  - **Seat effect: none to speak of; corrected.** I first read about 3 points of group-0
    advantage from two hand-picked comparisons (Levin 69.1 against 65.6, `prev_best`
    74.8 against 72.1). Two checks, using independent games:
    - the within-game gap in self-play, and in both orders of each opponent pair, gives
      +0.6 ± 0.2 per group here;
    - the older run (seed 42) leans the other way, at −1.6 ± 1.1 and −0.7 ± 1.0.

    Mechanically, the contribution and switch clones read `agent_group`. But both were
    trained (2026-09) on the flip-doubled file (#103, 2026-06-10): every game twice,
    group labels swapped, CV grouped by `pair_id`. The group node is symmetric to
    1e-16. So any asymmetry is residual fitting noise.

    Decided (2026-11-05): no further measurement. Sweeps and finals play every opponent
    in both seats, and the reference matrix is rerun with both seats for every row.
    `pool_scores.seated` stays as a tool; nothing scores with it.
  - **Reference matrix, first run** (`32_LEVIN_multi_reference`, job 31935035, seed
    8000, group 0 only, mean over the four opponents; superseded by the both-seat rerun):

    | Manager | Mean |
    |---|---|
    | `prev_best` | 70.8 |
    | Cem's best | 69.3 |
    | Levin | 66.5 |
    | `zero` | 58.7 |
    | `thr9_p10` | 57.9 |
    | `ah` | 54.8 |

    Every manager scores lowest against `zero` (44.6–63.1).
  - **Export.** The first smoke's export failed the opponent-term check on old plotting
    and baseline scripts, the punisher clone's configs and a login script; they are now
    denied for exp2_multi.
    - What remains is code (`src/`, which names manager types generically), the player
      clones' configs, the game doc and the human data.
    - Also found: `30_EXP1`/`31_EXP1_confirm` named exp1 instances and would have failed
      every later export; they are now in the global deny list.
- 2026-11-05 14:41: smoke (step 6) passed.
  - `pfsmoke-multi`: 16 points × 50 games per opponent, final 6 × 200; 59.8 ± 1.5 in
    16 min. Knowing only "another manager", the agent made its rule adaptive: it caps its
    own punishment at the other manager's recent mean, read from `p`.
  - `pfsmoke-multi-v2` (`--iterate-on`, full size: 256 × 125 per opponent in 2 parts,
    final 6 × 2000 per opponent): 69.06 ± 0.55 in 26 min (agent 5 min, sweep 11 min,
    final 5 min). Its export held v1's results under `opponent_1..4` only. The agent
    read them per opponent, then profiled each opponent from the per-round frames, e.g.
    "opponent_4 never punishes".
  - The page (`PF_SETTING=exp2_multi`) shows a column per opponent and the
    per-opponent table with the references.
