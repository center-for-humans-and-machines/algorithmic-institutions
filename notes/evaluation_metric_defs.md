# Definitions of Evaluation Metrics

Metrics for issue #128: we simulate many episodes with trained models and compare
the statistics of the simulation against the human experiment. Every metric is
computed once on the human data and once on the simulated data; the score says
how far apart the two are. Rollouts involve three models (manager, contribution,
switching), so credit cannot be cleanly assigned -- C rows say the most about the
contribution model, S rows about the switching model, P rows about the manager.
The P rows only score managers that are meant to mimic human punishment; for an
RL manager they describe the policy rather than score it.

## Shared conventions

- **Human reference data:** `experiments/2group_8agent_50ep.csv`. Every real game
  appears twice in the file with group labels swapped (the flip augmentation);
  we keep one copy per game, same as the linear-model training pipeline.
- **No-input rows:** rounds where a player or manager did not respond in time are
  excluded, same as the masking used in training.
- **Switching:** derived from group membership changes and anchored to the
  decision round (3, 7, 11, 15, 19). Human decisions with a selection timeout
  are excluded from switching denominators (the `switch_valid` policy from
  training). The simulation has no timeouts, so all its opportunities are valid.
- **Empty groups:** a (game, round) where all players sit in one group. Each
  metric below states how it handles this. Metrics over individual players are
  not affected (the players in the surviving group still count); metrics that
  need both groups drop the round; SC keeps these rounds on purpose.
- **`common_good`** is not used by any metric and is excluded from the common
  data format.

## Score types

Each metric has exactly one canonical score. Three score types exist:

1. **abs Δ statistic** -- absolute difference between the human value and the sim
   value of a statistic. When the metric is "per X" (round, opportunity, bin),
   the difference is computed per stratum and the strata are averaged with
   weights equal to each stratum's human frequency, fixed once from the full
   deduped human data (issue #132's scoring schema; for round strata this is
   close to uniform because rounds are roughly equally populated).
2. **signed Δ std** -- sim standard deviation minus human standard deviation,
   reported signed and in raw units. Diagnostic only, never a canonical score;
   kept for exactly three rows (CA, CC, CE).
3. **EMD** -- earth mover's distance between the human and sim distributions
   (`scipy.stats.wasserstein_distance`).

Whenever a canonical score says "per X, averaged", the average uses the same
human-frequency stratum weights, regardless of whether the per-stratum value is
an abs Δ or an EMD.

## Contribution distributions (C)

**CA -- participant mean contributions:** each participant's average contribution
over their rounds; how much participants differ from each other.
Canonical: EMD. Diagnostic: signed Δ std. Empty groups: not affected.

**CB -- round mean contributions:** the average contribution in each of the 24
rounds; how contributions rise and fall over the game.
Canonical: abs Δ per round, averaged over the 24 rounds. Empty groups: not affected.

**CC -- group mean contributions:** the average contribution of each group in
each (game, round); how much groups differ from each other.
Canonical: EMD. Diagnostic: signed Δ std. Empty groups: an empty group has no
mean and produces no observation; the surviving group's mean is kept.

**CD -- raw contributions:** every single contribution, pooled; the natural
shape of contribution behavior.
Canonical: EMD. Empty groups: not affected.

**CE -- signed group contribution differences:** for each (game, round), group
0's mean contribution minus group 1's; how far the two groups of a game drift apart.
Canonical: EMD. Diagnostic: signed Δ std. Empty groups: the whole (game, round)
drops, because the difference needs both groups.

**CF -- boundary shares:** the share of contributions equal to 0 and the share
equal to 20, per round; how much behavior polarizes at the extremes.
Canonical: abs Δ per (round, boundary) cell, averaged over the 48 cells.
Empty groups: not affected.

## Switching distributions (S)

**SA -- overall switch rate:** the share of valid switching opportunities where
the player actually switched, all opportunities pooled.
Canonical: abs Δ of the two rates. Empty groups: not affected.

**SB -- switch rate per opportunity:** the same rate computed separately at each
of the 5 decision rounds (3, 7, 11, 15, 19); when in the game people switch.
Canonical: abs Δ per opportunity, averaged over the 5. Empty groups: not affected.

**SC -- size of the larger group:** for every (game, round) from round 4 on, the
number of players in the larger group (4 means balanced, 8 means everyone
together); how segregated games become.
Canonical: EMD. Empty groups: KEPT -- a larger-group size of 8 is the strongest
segregation signal and dropping it would censor exactly what SC measures.
Rounds 0-3 are excluded because groups are always 4-4 before the first switch.

## Punishment distributions (P)

**PA -- raw punishments:** every single received punishment, pooled, zeros
included; the natural shape of punishment behavior.
Canonical: EMD. Empty groups: not affected.

**PB -- mean punishment per round:** the average punishment received in each of
the 24 rounds; how punishment intensity develops over the game.
Canonical: abs Δ per round, averaged over the 24 rounds. Empty groups: not affected.

**PC -- share of zero punishments per round:** the share of players who received
no punishment in each round; whether the manager punishes at all, as opposed to
how much.
Canonical: abs Δ per round, averaged over the 24 rounds. Empty groups: not affected.

## Responses (R) -- documented here, implemented on a later branch

A response is conditioned on the stimulus it reacts to: these check whether the
models have the right mechanisms, not just the right states. Contribution change
means next round's contribution minus this round's, with the stimulus taken this
round. The R rows can have empty strata (a sim that never punishes or never
switches produces no observations in some bins); how to handle that is decided
on the R branch.

**RCA -- contribution change by round type:** how contributions change after four
kinds of round: no switch was allowed, the player switched, the player chose to
stay, the player stayed but their group's composition changed.
Canonical: EMD per round type, averaged over the 4 types.

**RCB -- reaction to punishment:** the average contribution change of punished
non-full contributors, split by punishment rate bins (0, 0.25], (0.25, 0.5],
(0.5, 1], > 1; how strongly people respond to being punished.
Canonical: abs Δ per bin, averaged over the 4 bins.

**RCC -- reaction at the ceiling:** among full contributors (gave 20), the average
contribution change of punished minus unpunished players; RCB's rate is
undefined at 20, so the ceiling gets its own contrast.
Canonical: abs Δ of the punished-minus-unpunished contrast.

**RCD -- switching pull:** for each switch, regress the switcher's contribution
change on the gap between the receiving group's mean contribution and their own;
whether switchers adapt toward their new group.
Canonical: abs Δ of the regression slope (the pull coefficient).

**RSA -- switching after punishment:** the share of punished players who switch
at the next opportunity, split by punishment size bins 1-3, 4-15, 16+; who
leaves after being punished.
Canonical: abs Δ per bin, averaged over the 3 bins.

**RPA -- the manager's policy:** the distribution of punishments given at each
contribution level bin {0}, 1-5, 6-10, 11-15, 16-19, {20}; how punishment
depends on what the player contributed.
Canonical: EMD per bin, averaged over the 6 bins.

**RPB -- punishment by group size:** the distribution of punishments in group
size bins 1-3, 4-5, 6-8, from round 4 on; whether punishment depends on how
many players the manager governs. Empty groups drop out by construction.
Canonical: EMD per bin, averaged over the 3 bins.
