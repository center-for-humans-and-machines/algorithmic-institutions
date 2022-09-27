# Comparision of the humanlike, RL manager and the pilots

We are simulating 100 games each of groups of 4 artificial contributors managed by either
the humanlike manager and the RL manager. We also compare artificial
contributors trained on the full dataset with artificial contributor that were
trained only on those human contributors that where managed by a human manager.

## Collective behavior

![Learning curve](../notebooks/test_manager/plots/simulate_mixed/comparison_pilot.jpg)

We can make the following observations:
1. The RL manager (green) archives a larger common good compared to the rule based (brown) and the human manager from
   the pilot (violet)
2. The RL manager (green, red) punishes stronger then the human manager (violet), in particular in early rounds.
3. The RL performance is robust, however reduced, when managing alternative
   artificial participants trained on a subset of the data (red)
4. Punishment level of the artificial human manager (blue, orange) matches that
   of the human manager (violet).
5. Contribution of artificial human contributors managed by the artificial human
   manager (blue) are slightly below the empirical contributions by
   actual human participants (violet). Correspondingly the common good is
   slightly lower.
6. However, when humans artificial human contributors trained on the pilot with
   a human manager is used (orange), contributions and punishments matches the empirical
   levels (violet).

## Policies

We also show the respective empirical policies (i.e. the relationship between
contributions and punishments).

![Learning
curve](../notebooks/test_manager/plots/simulate_mixed/comparison_pilot_policy.jpg)

Average punishments of the RL manager (red, green) are considerable higher then punishments
of (artificial) human manager (blue, orange, violet). This applies to all contribution levels, but is
particular prominent for contributions below 14 points.

The policy of the artificial human manager (blue, orange) is well aligned with
the empirical policy found for human manager (violet).

