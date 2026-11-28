# pfsmoke-20261127-1-v4

## Explorations

1. **The base version never punishes.** Looked at `experiments/2group_8agent_50ep.csv` to understand human punishment patterns. Found that humans use contingent punishment: low contributors (0-5) get punished ~33% of the time while high contributors (16-20) get punished only ~6% of the time. The correlation between contribution and punishment is -0.23. Average punishment for low contributors (<10) is 2.59 vs 0.78 for high contributors (>=10). This shows punishment incentivizes contribution, as high contributors get higher payoffs (23.68 vs 21.78) and contribute to larger pools (108.98 vs 35.00). So version 4 should add contingent punishment based on contribution level.

## Key findings

1. Humans use contingent punishment: low contributors get punished more than high contributors (32.9% vs 5.9% chance of >0 punishment).
2. Punishment is effective: high contributors get higher payoffs (23.68 vs 21.78) and contribute to larger pools (108.98 vs 35.00).
3. The base version (v1) never punishes, which is suboptimal.

## Hypothesis

A good manager punishes low contributors to incentivize higher contributions to the common pool. The punishment should be contingent on contribution level, with higher punishment for lower contributors. The rule uses a threshold parameter: players contributing below the threshold get punished proportionally to how far below the threshold they are, while players at or above the threshold get no punishment.

### Justification

The key findings show that human managers use contingent punishment (finding 1), and this punishment is effective at incentivizing contribution (finding 2). The rule implements this by punishing players who contribute below a threshold, with the punishment intensity proportional to how far below the threshold they are. This matches human behavior while being simple and effective.

## Change

  - Base version: pfsmoke-20261127-1
  - Operation: add
  - Idea: Add contingent punishment based on contribution level: players contributing below a threshold get punished proportionally to how far below the threshold they are.
  - Justification: analysis
  - Paper: none
  - Claim: none
  - Analysis: analysis/v4_contingent_threshold.py, metric: punishment_contingency_score, value: 0.37
  - Expected: The rule should achieve a higher common pool than the base version by incentivizing higher contributions through targeted punishment.
