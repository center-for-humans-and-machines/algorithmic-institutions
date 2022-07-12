# RL manager

![RL Manager](../notebooks/manager_evaluation/plots/ah_simple_complex/metrics.jpg)

# Rule Manager

## Rule

$pun = (20-cont) \cdot s + (cont != 20) \cdot  c - b$

## Importance of factors

### Slope and constant punishment of defects

![Common Good S C](../notebooks/manager_evaluation/plots/v1_2_1/common_good_s_c.jpg)

### Slope and constant forgiveness

![Common Good S B](../notebooks/manager_evaluation/plots/v1_2_1/common_good_s_b.jpg)

### Conclusion

- Slope is the most important factor.
- Therefore we set in the following c == 0 and b == 0.
- A slope around 1.2 appears optimal for the complex AH.
- For simple AH there does not appear to a clear maximum.

## Evolution over the rounds

### Contribution and punishments over rounds

![Con Pun Over Rounds](../notebooks/manager_evaluation/plots/v1_2_1/over_rounds.jpg)

This plot shows the evolution of contributions and punishments over 16 rounds.

### Common Good and Cumulative Common Good

![Common Good Over Rounds](../notebooks/manager_evaluation/plots/v1_2_1/over_rounds_cg.jpg)

## Applying the rule only partially

### Apply the rule only to 1,2,3,4 agents

![Different Agents](../notebooks/manager_evaluation/plots/v1_agents_1/metric_ah.jpg)

### Apply the rule only to some rounds - Total over episode

![Some Rounds](../notebooks/manager_evaluation/plots/v1_rounds_1/metric_ah.jpg)

### Apply the rule only to some rounds - Per round

![Some Rounds](../notebooks/manager_evaluation/plots/v1_rounds_1/metric_ah.jpg)

## Next

- Policy Gradient?
- Debug with simple or even more simple model
