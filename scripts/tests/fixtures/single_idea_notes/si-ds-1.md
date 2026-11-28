# pfsmoke-20261127-1-v2

## Explorations

1. **Where the seed rule falls short.** Read `configs/managers/rule_based/pfsmoke-20261127-1.yml`
   and its notes: it is the given seed, `punishment = 0` for everyone, no
   params, so its whole sweep is one point. Its sweep
   (`plots/simulation/policy_finder/pfsmoke-20261127-1_sweep/sweep.json`)
   scored pool 65.37 (se 10.06) at seed 42; its fresh-seed final
   (`final/sweep.json`) scored 55.31 (se 6.78). Read the reference sims
   `25_LEVIN_run1_ah_zero_pairings_batched`: only the plots are mounted here,
   the per-round parquet files are absent. In those plots `ah`'s group reaches
   a per-capita common good of ~15-16 with contributions ~10-11 and mild
   punishment (~1-4 points), while the `zero` group reaches ~12.5 and ~8.5.
   So costly punishment can pay in this stack, and the seed leaves that on the
   table. Next: what punishment shape the human managers actually used, since
   `ah` is trained on them.
2. **What the humans and the AH models compare against.** Read
   `scripts/baselines/handcrafted_grid.py` and `src/aimanager/generic/graph.py`.
   The punishment feature list keeps `contribution_mean_group` legal for the
   punishment target and its comment records the human punishment data:
   corr(p_t, c_t) = -0.28 vs corr(p_t, c_{t-1}) = -0.19, and
   P(p>0 | c_t<=4, c_{t-1}=20) = 0.57 vs 0.18 the other way round. The
   contribution model is fed `own_grp_prev_mean_contr` (leave-one-out mean of
   the own group's previous contributions) and `prev_punishment`; graph.py
   cites a teacher-forced human punishment-response slope of 0.7973 (in
   self-play it collapsed, and the reference stack uses the stimulus-skip
   model built to restore the direct response). So humans punish current low
   contributions relative to their group, and the simulated humans respond to
   punishment. I want the rule to punish a player by its shortfall below the
   mean of the other valid members of its group. The paper I rely on is Fehr &
   Gächter (2000): punishment rises with how far a contribution falls short of
   the others' average, and punishment opportunities sustain contributions.
   Next: estimate the human punishment-per-shortfall slope so I can place
   `k`'s range.
3. **The analysis metric, before running it.** I will use the human
   experiments only, deduplicated per `docs/reports/basics.md` on (session,
   episode, round_number, participant_code), dropping player and manager
   timeouts. For every remaining player-round I compute the shortfall
   `s = max(0, mean contribution of the other valid in-group members - own
   contribution)` and regress `punishment` on `s` with an intercept; the
   metric is the OLS slope, in punishment points per contribution point of
   shortfall. A slope clearly above 0.3 speaks for the change (humans raise
   punishment with the shortfall below peers); a slope near 0 or below speaks
   against. I expect 0.3-1.0 if the human manager is group-relative, and I
   will centre `k`'s sweep range around it while keeping a wide log range.
4. **The analysis number.** Ran
   `scripts/policy_finder/pfsmoke-20261127-1/pfsmoke-20261127-1-v2_analysis.py .`
   once on the human data (test_data first, for structure only). It printed
   `0.648966`: one punishment point per 1.54 contribution points of shortfall
   below the valid group peers. That is well above my 0.3 support threshold
   and close to the 1/1.6 = 0.625 break-even where a punishment point's 1.6
   pooled return from one extra contribution point just covers its cost, so
   the human reaction is consistent with costly but productive group-relative
   punishment. Next: write the rule with one parameter `k` and put its log
   range around 0.65, wide enough to test both mild and strong versions.

5. **Drafting the rule.** Wrote
   `configs/managers/rule_based/pfsmoke-20261127-1-v2.yml` as the seed rule
   plus one idea: each round, a player in the manager's group is punished by
   `k` times its current contribution's shortfall below the mean contribution
   of the other valid members of that group; timeouts are left out of the mean
   and get 0. I kept a single parameter so the 16 points scan `k` finely. The
   log range `[0.05, 3]` has its 16 Sobol values from 0.061 to 2.771, with
   0.617 and 0.679 next to the human estimate 0.649 and a low end that
   approximates the seed. `validate-rule` passed with 0, 4 max params and 16
   Sobol points, so the code runs at rounds 0, 11 and 23 on every point.

## Key findings

1. The seed version never punishes and its fresh-seed final pool is 55.31
   (se 6.78), while in the reference sims `ah`'s mild punishment (about 1-4
   points) sustains per-capita common good ~15-16 versus ~12.5 for the
   zero-punisher, so punishment can raise the pool (Exploration 1).
2. Human punishment is mainly a response to the current contribution
   (corr -0.28 vs -0.19 lagged; P(p>0 | c_t<=4, c_{t-1}=20) = 0.57 vs 0.18),
   the simulated contribution model reads the own group's leave-one-out mean
   contribution and previous punishment, and the teacher-forced human response
   slope is 0.7973 punishment points per contribution point, so a
   group-relative shortfall rule fits both the data and the simulator
   (Exploration 2).
3. On the human data the punishment-on-shortfall-below-peers slope is
   0.648966 points per contribution point, above the 0.3 support threshold and
   close to the 1/1.6 = 0.625 cost break-even (Exploration 4).
4. A single `k` with a log range [0.05, 3] covers from near-zero punishment
   to strong punishment, and its 16-point design has two values (0.617,
   0.679) bracketing the measured human slope (Exploration 5).

## Hypothesis

A good manager in this game keeps the group's contributions high by
punishing only the free riders, and by punishing them in proportion to how far
they fall short of what their group peers gave: the group mean is a norm that
travels with the players across switches, it never punishes the best
contributor, and it costs the common pool less than a blanket high absolute
target. Punishment must still be modest because every point deducted from a
punished player also leaves the group's pool; it pays only if a punishment
point buys at least 1/1.6 contribution points back. The rule has one
parameter, `k`, the punishment points per contribution point of shortfall
below the mean of the other valid group members, swept log-uniformly over
[0.05, 3]: the low end reproduces the seed, the middle brackets the human
slope 0.649 and the cost break-even 0.625, and the high end tests whether a
strong deterrence effect outweighs the extra pool cost.

### Justification

Finding 1 shows the seed leaves pool on the table and that mild punishment is
not merely costly in this stack; finding 2 shows the human manager and the
trained humans both react to the current contribution relative to the group,
which is exactly the comparison the rule makes; finding 3 gives the magnitude
of `k`, 0.649, which is above the 0.3 that would support the mechanism and
just above the 1/1.6 break-even, so a pass through the middle of the sweep is
expected to pay for itself; finding 4 shows the sweep can locate that region
and still probe both harmless-mild and strong values. The hypothesis would be
contradicted if the human slope had been near zero or negative, or if the
simulated contribution response were so weak that the best sweep point sat at
the bottom of `k` and scored below the seed. I did not model antisocial or
retaliatory punishment, punishment history, group-size-dependent severity, or
a separate cap; the one-idea change and the single parameter keep `k`
identifiable, timeouts are charged 0 by the game anyway, and the manager's
0-30 clamp bounds the amount.

## Change

- Base version: pfsmoke-20261127-1
- Operation: add
- Idea: punish a player in proportion to how far its current contribution falls short of the mean contribution of the other valid members of its group.
- Justification: paper and analysis
- Paper: Fehr, E., & Gächter, S. (2000). Cooperation and Punishment in Public Goods Experiments. American Economic Review, 90(4), 980-994.
- Claim: Punishment in the public-goods stage is aimed at free riders and increases with how far a player's contribution falls short of the other group members' average, and the punishment opportunity sustains high contributions.
- Analysis: scripts/policy_finder/pfsmoke-20261127-1/pfsmoke-20261127-1-v2_analysis.py, OLS slope of punishment on the shortfall below the valid group peers' mean contribution, printed 0.648966.
- Expected: raise the group's common pool over the zero-punishment seed by holding contributions at the group norm while punishing only the laggards; low `k` should approach the seed, and `k` around 0.6-1 should pay for itself because the human punishment response (~0.65-0.8) is above the 1/1.6 break-even.
