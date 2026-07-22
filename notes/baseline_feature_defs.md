# Definitions of Features

## Self

**prev_contribution:** own contribution from previous round. defaults in the 0th round.
**prev_punishment:** own punishment from previous round. defaults in the 0th round.
**prev_payoff:** own payoff from previous round. defaults in the 0th round.

## Group

- Crucially, this definition includes quantities of the agents who left the group at t-1 that the focus agent joined at t in its average calculation.
- If a switch happened then the averages are shown for the group that is switched to that the agent did not belong to in the t-1 round.
- This does not include the agent itself in its averaging. We have self set anyways so we keep the agent itself out of it. So in a round where the agent is still in the same group it sees the average of the group without its value's effect.

**prev_contribution_mean_group:** previous round's average contribution of the current group
**prev_punishment_mean_group:** previous round's average punishment of the current group
**prev_payoff_mean_group:** previous round's average payoff of the current group
**prev_common_good:** previous round's common_good = (1.6 · Σ contributions − Σ punishments)/n_valid for the current group 
**prev_group_size:** previous group size of the current group

## Other

**prev_contribution_mean_other:** previous round's average contribution of the current opponent group 
**prev_punishment_mean_other:** previous round's average punishment of the current opponent group
**prev_common_good_other:** previous round's common_good = (1.6 · Σ contributions − Σ punishments)/n_valid for the current opponent group
**prev_payoff_mean_other:** previous round's average payoff of the current opponent group
**prev_group_size_other:** previous group size of the current opponent group

## Gap/Delta

**prev_contribution_mean_gap:** prev_contribution_mean_group - prev_contribution_mean_other
**prev_punishment_mean_gap:** prev_punishment_mean_group - prev_punishment_mean_other
**prev_common_good_gap:** prev_common_good - prev_common_good_other
**prev_payoff_mean_gap:** prev_payoff_mean_group - prev_payoff_mean_other
**prev_group_size_delta:** prev_group_size - prev_group_size_other

## Own Since Switch

- In the 0th round after a switch all except group size figure has 0.

**win_contribution_mean_peers:** the moving average contribution of the current group's peers since the last switch of the agent (excluding t round contributions of course)
**win_punishment_mean_peers:** the moving average punishment of the current group's peers since the last switch of the agent (excluding t round punishments of course)
**win_common_good_peers:** the moving average common good of the current group's peers since the last switch of the agent (excluding t round common good of course)
**win_payoff_mean_peers:** the moving average payoff of the current group's peers since the last switch of the agent (excluding t round payoff of course)
**win_group_size:** the moving average group size of the current group since the last switch of the agent (excluding t round group size of course)

## Other Since Switch

- In the 0th round after a switch all except group size figure has 0.

**win_contribution_mean_other:** the moving average contribution of the current opponent group since the last switch of the agent (excluding t round contributions of course)
**win_punishment_mean_other:** the moving average punishment of the current opponent group since the last switch of the agent (excluding t round punishments of course)
**win_common_good_other:** the moving average common good of the current opponent group since the last switch of the agent (excluding t round common good of course)
**win_payoff_mean_other:** the moving average payoff of the current opponent group since the last switch of the agent (excluding t round payoff of course)
**win_group_size_other:** the moving average group size of the current opponent group since the last switch of the agent (excluding t round group size of course)

## Since Switch as of t-1 (switch target)

- One-round-shifted copies of the win_* windows: the window value as it stood at t-1.
- Needed for the switch target: win_* at round t resets on does_switch[t] (the target itself), while the prev_win_* reset only ever uses switches before t. Leak-safe at decision rounds.

**prev_win_contribution_mean_peers:** win_contribution_mean_peers as of the previous round
**prev_win_punishment_mean_peers:** win_punishment_mean_peers as of the previous round
**prev_win_common_good_peers:** win_common_good_peers as of the previous round
**prev_win_payoff_mean_peers:** win_payoff_mean_peers as of the previous round
**prev_win_group_size:** win_group_size as of the previous round

**prev_win_contribution_mean_other:** win_contribution_mean_other as of the previous round
**prev_win_punishment_mean_other:** win_punishment_mean_other as of the previous round
**prev_win_common_good_other:** win_common_good_other as of the previous round
**prev_win_payoff_mean_other:** win_payoff_mean_other as of the previous round
**prev_win_group_size_other:** win_group_size_other as of the previous round

## Group/Other as of t-1 (switch target)

- Twins of the group/other/gap means that keep t-1 MEMBERSHIP: t-1 values averaged over the group as it stood before the round-t decision.
- Needed for the switch target: the standard means aggregate over the round-t roster, i.e. the post-decision membership (the target itself). Leak-safe at decision rounds.
- To be removed by #123 (re-anchoring the switch decision at the pre-switch round makes plain current features legal).

**lag_contribution_mean_peers:** previous round's average contribution of the t-1 group-mates (excluding self)
**lag_punishment_mean_peers:** previous round's average punishment of the t-1 group-mates (excluding self)
**lag_payoff_mean_group:** previous round's average payoff of the t-1 group (including self; shared common good)
**lag_contribution_mean_other:** previous round's average contribution of the t-1 opponent group
**lag_punishment_mean_other:** previous round's average punishment of the t-1 opponent group
**lag_common_good_mean_other:** previous round's common good of the t-1 opponent group
**lag_payoff_mean_other:** previous round's average payoff of the t-1 opponent group
**lag_contribution_mean_gap:** lag_contribution_mean_peers - lag_contribution_mean_other
**lag_punishment_mean_gap:** lag_punishment_mean_peers - lag_punishment_mean_other
**lag_common_good_mean_gap:** own t-1 group's common good - lag_common_good_mean_other
**lag_payoff_mean_gap:** lag payoff of the t-1 group-mates - lag_payoff_mean_other

## Structural

**round_number:** current round number
**switched_last_choice:** whether the agent switched in the last switch decision round
**prev_switched_last_choice:** whether the agent switched in the previous last switch decision round (needed for switch to prevent leakage)
**rounds_since_switch:** the number of rounds since the last switch of the agent
**group_size:** current group size
