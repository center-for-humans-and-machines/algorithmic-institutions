# pfsmoke-20261127-1-v2

## Explorations

1. **What is the seed rule and how did it do?** Read
   `configs/managers/rule_based/pfsmoke-20261127-1.yml`: it sets
   `punishment = th.zeros_like(c[..., -1])` and declares no parameters, so the
   sweep is one point. Its `final/sweep.json` reports a fresh-seed pool of
   55.31 (SE 9.89) for the single point and 55.31 (SE 6.78) for the GP copy;
   the sweep-seed point was 65.37. The rule never punishes, so it keeps all
   players but free-rides are not deterred. That is the number to beat.
2. **What does punishment do in the reference stack?** Read the reference
   plots in `plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/`.
   `ah` punishes low contributions (plot: mean punishment falls from about 4
   at round 0 to about 1 at round 24, and it is roughly 5 points for c=0 to
   0.3 for c=20 in its fitted model); its group's per-capita common good is
   higher (~15 vs ~13) but its group shrinks to ~3.1 members while the
   zero-punisher's group grows to ~4.7-4.9. Because the score is the group's
   *total* pool, the zero manager's larger group wins on total pool despite
   lower per-capita contributions. So any punishment I add has to be mild and
   targeted: it must not buy contribution at the price of a smaller group.
3. **Which model artifacts can I read without touching the data?** Read the
   `ah` joblib (`punishment_multinomial_timeout_severity_copula.joblib`) and
   the contribution/switch baseline joblibs. `ah`'s features are current
   contribution, `contribution_max`, `contribution_valid`, prev contribution,
   prev punishment, round and `is_first`; its fitted mean punishment is
   monotone decreasing in current contribution (about 4.2 at c=0, 1.0 at c=10
   and 0.3 at c=20 at round 4) and decreasing over rounds. The switch baseline
   has `win_punishment_mean_group` with a positive switch coefficient
   (+0.618), consistent with punishment making members leave. The
   contribution model was trained as a punishment-response experiment
   (`stimulus_skip`, `group_vnode`) with `prev_punishment` as a node feature,
   so punishment also acts as a group-level signal, not only an individual
   deterrent.
4. **The one change.** The base rule has no punishment idea at all, so I add
   one: peer sanctioning against a contribution norm. A member is punished in
   proportion to the shortfall of their current contribution below the
   group's norm; the norm is the group's mean valid contribution, but never
   below a `floor`. This has two readings: `k` is points of punishment per
   point of shortfall, and `floor` is the minimum contribution norm the group
   is held to. Comparing against the group's actual average keeps the rule
   from punishing high contributors when the group is already cooperating,
   while the floor keeps a low-contributing group from settling at a low
   norm. This is one idea (how a shortfall becomes punishment), not a
   round-dependence or a cap.
5. **Analysis pre-registration.** I will estimate the within-player effect of
   punishment on the next round's contribution, because the added rule only
   helps the pool if punishment makes a sanctioned member contribute more. The
   metric is `b`, the OLS coefficient on `punishment_t` in a regression of
   `contribution_{t+1}` on own `punishment_t`, own `contribution_t`, the mean
   current contribution of the other valid members of the same group, and the
   round, with player-by-game fixed effects; it uses only consecutive,
   same-group, valid observations and prints `b` in contribution points per
   punishment point. A value **for** the change is `b > 0.625`: at the 1.6
   return rate the group pool then recovers one punishment point's direct cost
   within one round (1.6 * 0.625 = 1). A value **against** is `b <= 0` (punished
   members do not contribute more). A value between 0 and 0.625 is positive but
   not self-financing in one round; repeated group signalling could still make
   a mild rule worthwhile. I wrote
   `scripts/policy_finder/pfsmoke-20261127-1/pfsmoke-20261127-1-v2_analysis.py`
   for this and will debug it on `test_data`, then run it once with `.` as root.
6. **The analysis run.** Ran
   `scripts/policy_finder/pfsmoke-20261127-1/pfsmoke-20261127-1-v2_analysis.py`
   once on the real data (root `.`); it printed **0.074260**. The metric is
   the fixed-effects slope of next-round contribution on current punishment.
   That is positive, so punishment does move a member's next contribution in
   the right direction, but it is far below the 0.625 one-round break-even:
   own punishment alone would not pay for itself within one round. The result
   therefore argues for a *mild* rule, and for relying partly on the
   group-level signalling of punishment rather than on squeezing a large
   individual response; it argues against a large `k`. (The test-data debug
   run printed 0.087676, which is meaningless as expected; the real run is
   the one above.)
7. **The rule I drafted.** Copied the seed and added the peer-sanctioning
   idea: for the manager's own valid members,
   `mean = mean current contribution`, `norm = max(mean, floor)`,
   `punishment = k * max(0, norm - c)`, and zero outside the manager's group
   or for timed-out players. `k` is a float in
   `[0.1, 2, log]`: the analysis' 0.074 says the useful values are small, so
   the log range spends its points at the low end but still reaches 2 for a
   group-signalling effect that the one-number analysis cannot see.
   `floor` is an int in `[0, 16]`: 0 gives the pure below-mean norm, and 16 is
   an ambitious minimum norm; the sweep can choose how much floor, if any, a
   floor is worth. The manager's integer flooring means `k * shortfall < 1`
   is already an implicit tolerance, so I did not add a third parameter.

## Key findings

1. The seed's fresh-seed final result is **55.31** (SE 9.89), while its
   sweep-seed point was 65.37; that is the number this version has to beat
   (Exploration 1).
2. In the reference stack `ah` punishes low contributions mildly (roughly 4.2
   points at c=0 down to 0.3 at c=20, round 4) and its group's per-capita
   common good is higher, but its group shrinks to about **3.1** members while
   the zero-punisher's group grows to about **4.7-4.9**; because the score is
   the group's total pool, the no-punishment manager wins on total pool
   (Exploration 2).
3. The one-run analysis gives **b = 0.074260** contribution points of
   next-round contribution per punishment point, positive but far below the
   0.625 one-round break-even, so only a mild punishment can be justified by
   own deterrence (Exploration 6).
4. The switch baseline puts a positive coefficient (+0.618) on the group's
   punishment for leaving, so every punishment point also buys some exit risk;
   this reinforces making the sanction narrow and small (Exploration 3).
5. The drafted rule validated with **16 distinct design points** in the
   16-point Sobol design (Exploration 7 and the validate-rule run).

## Hypothesis

A good manager in this game should sanction free riding, because unpunished
low contributions decay and because the artificial contributors were trained
on a punishment-response stack; but the manager must sanction *narrowly and
mildly*, because punishment is charged to the common pool and makes members
leave, and the score is the group's total pool rather than its per-capita
common good. The rule therefore punishes each own-group valid member in
proportion to how far below the group's contribution norm they fall, where the
norm is the group mean but at least `floor`. `k` (points of punishment per
point of shortfall) is swept log-uniformly over `[0.1, 2]`: the analysis'
0.074 puts the plausible useful values at the low end, while the upper end
leaves room for the group-level signalling channel the one-number analysis
cannot see. `floor` (the minimum contribution norm) is an integer swept over
`[0, 16]`: 0 recovers the pure below-mean peer norm, 16 is an ambitious
minimum; the sweep decides whether a floor adds anything beyond the mean.
With integer punishment flooring, `k * shortfall < 1` is already an implicit
tolerance, so there is no separate tolerance parameter.

### Justification

Findings 1-4 support the hypothesis by number. The seed's 55.31 leaves room,
but the reference shows the total-pool cost of broad punishment (finding 2),
so the sanction should target only below-norm members; the analysis' 0.074260
finding 3 bounds `k` low rather than high; the switch coefficient in finding 4
makes the `floor`/mean comparison narrow by design. The paper supports adding
free-rider punishment at all, and the analysis supports its mild form. What
would contradict the hypothesis: a negative punishment-response slope (it was
+0.074), or a sweep in which any positive `k` lowers the simulated pool
because the group shrinks; if that happens, the lower end of `k` and a floor
near 0 should win. What I did not model: a round-dependent taper (the shortfall
naturally shrinks as contributions rise), an explicit per-player cap, a
leave-one-out norm, or retaliation/history beyond one punishment lag; adding
those would be further ideas, and the base has room for only this one.

## Change

- Base version: pfsmoke-20261127-1
- Operation: add
- Idea: peer sanctioning in proportion to a member's shortfall below the group's contribution norm, where the norm is the group mean with a minimum floor.
- Justification: both
- Paper: Fehr, E. & Gächter, S. (2002). Altruistic punishment in humans. Nature 415, 137-140.
- Claim: In repeated public-goods games, opportunities to punish free riders substantially raise contributions and can sustain near-full cooperation, whereas contributions decay without punishment.
- Analysis: scripts/policy_finder/pfsmoke-20261127-1/pfsmoke-20261127-1-v2_analysis.py, fixed-effects slope of next-round contribution on current punishment, 0.074260 contribution points per punishment point.
- Expected: with a small `k` and a floor near typical contributions, raising free riders' next contributions will add more pool than the narrow punishment costs; if exit risk dominates, the sweep should pick the low end of `k` and a floor of 0, leaving a purely mean-based, very mild sanction.
