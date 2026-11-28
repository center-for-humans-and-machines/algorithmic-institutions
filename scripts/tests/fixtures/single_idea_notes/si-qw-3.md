# pfsmoke-20261127-1-v2

## Explorations

1. **Understanding the base rule.** The base version (`pfsmoke-20261127-1`) never punishes: every player gets 0 punishment points regardless of their contribution. Its final score against `ah` is ~55.3 in common pool. Since the manager's goal is a large common pool, and punishment reduces the pool (1.6*sum(c) - sum(p)), not punishing seems beneficial. But the artificial human managers (`ah`) DO punish, suggesting there's a strategic reason.

2. **Analyzing human punishment patterns.** Looked at `experiments/2group_8agent_50ep.csv` to understand how human managers punish. Key findings:
   - Mean punishment is 1.71, but high initially (round 0: 3.72), then drops to ~1-2
   - Free riders (contribution=0) get mean punishment 3.34, while contributors get 1.49
   - Negative correlation between contribution and punishment (-0.17 to -0.27 across rounds)
   - This suggests human managers punish low contributors more, not high contributors

3. **Question: What would happen if we punish free riders?** The human data shows free riders are punished more, but this still reduces the common pool. However, if free riders are punished enough, they might learn to contribute in future rounds (if the game had learning across games, but it doesn't in this setup). In a single repeated game, punishing free riders just reduces the pool without improving future contributions. But perhaps the artificial humans respond to punishment by contributing more in later rounds? Need to check the reference simulations.

## Key findings

1. Base version never punishes, gets ~55.3 pool against `ah`.

2. Human managers punish free riders (contribution=0) more than contributors (mean 3.34 vs 1.49), with negative correlation (-0.17 to -0.27 across rounds).

3. The reference sim (`ah` vs `zero`) shows that `ah` punishes (mean ~2-3 per round) while `zero` never punishes, and `ah`'s group has higher pool.

## Hypothesis

A manager should punish free riders (those who contribute 0) because:
- Human managers do this, and `ah` (trained on human data) also punishes
- Punishing free riders may signal to artificial human players that free-riding is unacceptable, potentially increasing future contributions
- Even if it doesn't increase contributions, the signaling effect may be worth the small pool reduction

The rule will punish free riders with a tunable intensity, decreasing punishment for those who contribute more.

### Justification

Key findings 1 and 2 support that human managers punish free riders. The base version's score of 55.3 is lower than what `ah` achieves against `zero` (which must be higher since `zero` never punishes but still gets ~55). If `ah` punishes and achieves a higher pool, this suggests that punishment has a positive effect—likely by influencing the artificial humans' future contributions.

## Change

- Base version: `pfsmoke-20261127-1`
- Operation: modify
- Idea: Instead of never punishing, punish free riders (those who contribute 0) with tunable intensity, and reduce punishment for contributors
- Justification: Analysis of human data shows free riders are punished more, and `ah` (trained on human data) punishes successfully
- Paper: none
- Claim: none
- Analysis: none
- Expected: The rule should outperform the base version by punishing free riders, which may signal to artificial humans that free-riding is unacceptable and increase future contributions or improve coordination
