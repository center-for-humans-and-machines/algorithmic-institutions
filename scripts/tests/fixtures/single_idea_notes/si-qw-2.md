# pfsmoke-20261127-1-v2

## Explorations

1. **The base rule's limitation.** The base version `pfsmoke-20261127-1` never punishes anyone (punishment = 0). Looking at the human data in `experiments/2group_8agent_50ep.csv`, contributions range from 0 to 20, and punishments range from 0 to 9. In group_id=0 episodes (where the group does well), common_good averages around 40-90, while group_id=1 episodes have lower averages (12-26). This suggests that managers who punish low contributors achieve higher common pools. The base rule's inability to discipline free riders likely explains its poor performance. Next, I need to design a punishment scheme based on contributions.

2. **Analysis of contribution variability.** To quantify how much human contributors vary, I computed the standard deviation of contributions per group per episode, averaged across all episodes. The result is 5.5. High variability (std dev > 4) suggests many free riders. If a manager punishes low contributors, future contributions should be more equal (lower std dev). The target std dev for a good manager's group should be around 3.5 or less, indicating more equal contributions. This supports a threshold-based punishment rule.

3. **Punishment based on shortfalls.** The natural idea is to punish players who contribute below a threshold. From the data, the endowment is 20, and contributions range 0-20. A player who contributes little relative to what they could should be punished. The base version should be modified to punish based on contribution shortfalls. I will implement a rule that punishes players who contribute below a threshold `k`, with punishment proportional to the shortfall.

## Key findings

1. The base version never punishes, so free riders are not disciplined, leading to low common pools in the other group. Human data shows that groups with more equal contributions and some punishment achieve higher common goods.

## Hypothesis

A good manager punishes players who contribute below a threshold, to encourage higher contributions and increase the common pool. The rule should punish players whose contribution is below a threshold `k`, with punishment proportional to the shortfall.

### Justification

1. Human data shows contribution ranges 0-20, with some players contributing very little (0-5) while others contribute a lot (10-14). Groups with higher average contributions have much higher common goods (e.g., 89.0 vs 12.8).
2. The base rule's failure to punish free riders explains why it performs poorly - there's no cost to free riding.
3. A threshold-based punishment scheme directly addresses this by making free riding costly.

## Change

- Base version: pfsmoke-20261127-1
- Operation: modify
- Idea: Replace the constant zero punishment with a punishment that is proportional to how much a player's contribution falls below a threshold `k`, with a multiplicative factor `p_mult`
- Justification: analysis of human data showing contribution patterns and their relationship to common pool outcomes
- Paper: none
- Claim: none
- Analysis: scripts/policy_finder/pfsmoke-20261127-1-v2/pfsmoke-20261127-1-v2_analysis.py, metric = contribution_std_dev, value = 5.5 (high variability suggests free riders are not disciplined), target = 3.5 for (lower variability indicates more equal contributions with punishment)
- Expected: The rule should achieve a higher common pool than `ah` by punishing low contributors and encouraging higher voluntary contributions
