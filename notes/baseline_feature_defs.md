# Definitions of Features

Two families after the #123 re-anchoring, one per target:

- **Current family (no prefix) -- switch target.** The switch decision is anchored at the pre-switch round s (rounds 3, 7, 11, ... 0-indexed with switch_every=4): the choice is made after round s is fully played, the membership change materialises at s+1. Features at row s therefore legally see round s's contributions, punishments and common good.
- **Prev family (`prev_` prefix) -- contribution target.** The contribution at round t must not see round t's outcomes, so it uses last round's values. Membership itself resolves before contributing, so current membership-derived features (sizes, tenure counters) are legal for both targets.

Group/Other is always keyed to the agent's CURRENT group id at the row being read.

Leak rule: any current-family feature that reads round-t contributions/punishments/common good (self values, group/other/gap means, win_* windows) is ILLEGAL for the contribution target and rejected with a hard error at config validation.

## Current -- Self

**contribution:** own contribution this round
**punishment:** own punishment this round
**payoff:** own payoff this round = 20 - contribution - punishment + common_good

## Current -- Group

- Average over the current group's members this round, excluding the agent itself (we have self set anyways so we keep the agent itself out of it).

**contribution_mean_group:** this round's average contribution of the current group
**punishment_mean_group:** this round's average punishment of the current group
**payoff_mean_group:** this round's average payoff of the current group
**common_good:** this round's common good = (1.6 · Σ contributions − Σ punishments)/n_valid for the current group
**group_size:** current group size

## Current -- Other

**contribution_mean_other:** this round's average contribution of the current opponent group
**punishment_mean_other:** this round's average punishment of the current opponent group
**payoff_mean_other:** this round's average payoff of the current opponent group
**common_good_other:** this round's common good of the current opponent group
**group_size_other:** current opponent group size

## Current -- Gap/Delta

**contribution_mean_gap:** contribution_mean_group - contribution_mean_other
**punishment_mean_gap:** punishment_mean_group - punishment_mean_other
**payoff_mean_gap:** payoff_mean_group - payoff_mean_other
**common_good_gap:** common_good - common_good_other
**group_size_delta:** group_size - group_size_other

## Current -- Since Switch

- Moving average of the current-family group values over the agent's tenure, INCLUDING the current round.
- Resets when the agent arrives in a new group; the arrival round is included (it is the new group's outcome, observed after playing it).

**win_contribution_mean_group:** average contribution_mean_group since the agent's last switch, through this round
**win_punishment_mean_group:** average punishment_mean_group since the agent's last switch, through this round
**win_common_good:** average common_good since the agent's last switch, through this round
**win_payoff_mean_group:** average payoff_mean_group since the agent's last switch, through this round
**win_group_size:** average group_size since the agent's last switch, through this round

**win_contribution_mean_other:** average contribution_mean_other since the agent's last switch, through this round
**win_punishment_mean_other:** average punishment_mean_other since the agent's last switch, through this round
**win_common_good_other:** average common_good_other since the agent's last switch, through this round
**win_payoff_mean_other:** average payoff_mean_other since the agent's last switch, through this round
**win_group_size_other:** average group_size_other since the agent's last switch, through this round

## Prev -- Self

**prev_contribution:** own contribution from previous round. defaults in the 0th round.
**prev_punishment:** own punishment from previous round. defaults in the 0th round.
**prev_payoff:** own payoff from previous round. defaults in the 0th round.

## Prev -- Group

- Crucially, this definition includes quantities of the agents who left the group at t-1 that the focus agent joined at t in its average calculation.
- If a switch happened then the averages are shown for the group that is switched to that the agent did not belong to in the t-1 round.
- This does not include the agent itself in its averaging. We have self set anyways so we keep the agent itself out of it. So in a round where the agent is still in the same group it sees the average of the group without its value's effect.

**prev_contribution_mean_group:** previous round's average contribution of the current group
**prev_punishment_mean_group:** previous round's average punishment of the current group
**prev_payoff_mean_group:** previous round's average payoff of the current group
**prev_common_good:** previous round's common_good = (1.6 · Σ contributions − Σ punishments)/n_valid for the current group
**prev_group_size:** previous group size of the current group

## Prev -- Other

**prev_contribution_mean_other:** previous round's average contribution of the current opponent group
**prev_punishment_mean_other:** previous round's average punishment of the current opponent group
**prev_common_good_other:** previous round's common_good = (1.6 · Σ contributions − Σ punishments)/n_valid for the current opponent group
**prev_payoff_mean_other:** previous round's average payoff of the current opponent group
**prev_group_size_other:** previous group size of the current opponent group

## Prev -- Gap/Delta

**prev_contribution_mean_gap:** prev_contribution_mean_group - prev_contribution_mean_other
**prev_punishment_mean_gap:** prev_punishment_mean_group - prev_punishment_mean_other
**prev_common_good_gap:** prev_common_good - prev_common_good_other
**prev_payoff_mean_gap:** prev_payoff_mean_group - prev_payoff_mean_other
**prev_group_size_delta:** prev_group_size - prev_group_size_other

## Prev -- Since Switch

- The win_* windows as they stood at t-1: moving average of the prev-observed group values over the agent's tenure (excluding t round values of course).
- In the 0th round after a switch all except group size figures are 0; the size windows keep the arrival value (the previous size of the joined group).
- The 0th round of the episode reads the defaults, like every prev_ feature.

**prev_win_contribution_mean_group:** the moving average contribution of the current group's peers since the last switch of the agent (excluding t round contributions of course)
**prev_win_punishment_mean_group:** the moving average punishment of the current group's peers since the last switch of the agent (excluding t round punishments of course)
**prev_win_common_good:** the moving average common good of the current group since the last switch of the agent (excluding t round common good of course)
**prev_win_payoff_mean_group:** the moving average payoff of the current group's peers since the last switch of the agent (excluding t round payoff of course)
**prev_win_group_size:** the moving average group size of the current group since the last switch of the agent (excluding t round group size of course)

**prev_win_contribution_mean_other:** the moving average contribution of the current opponent group since the last switch of the agent (excluding t round contributions of course)
**prev_win_punishment_mean_other:** the moving average punishment of the current opponent group since the last switch of the agent (excluding t round punishments of course)
**prev_win_common_good_other:** the moving average common good of the current opponent group since the last switch of the agent (excluding t round common good of course)
**prev_win_payoff_mean_other:** the moving average payoff of the current opponent group since the last switch of the agent (excluding t round payoff of course)
**prev_win_group_size_other:** the moving average group size of the current opponent group since the last switch of the agent (excluding t round group size of course)

## Structural (shared, legal for both targets)

**round_number:** current round number
**switched_last_choice:** whether the agent switched at the most recent decision round before this row
**rounds_since_switch:** the number of rounds since the last switch of the agent (0 at the arrival round)
