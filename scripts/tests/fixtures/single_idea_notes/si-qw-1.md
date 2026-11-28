# pfsmoke-20261127-1-v2

## Explorations

1. **Understanding the seed rule.** The base version (`pfsmoke-20261127-1`) never punishes - it sets `punishment = th.zeros_like(c[..., -1])`. Its sweep has one point (no parameters) and achieves a pool of ~65 against `ah`. Looking at the human data, contributions vary (e.g., rounds 0-5 show contributions from 1-14 points), and punishments range from 0-30 points. The `ah` manager does punish, so a manager that never punishes may be leaving points on the table by not deterring free-riding.

2. **What does `ah` do?** The reference simulation shows `ah` vs `zero` (the seed rule). The seed rule's pool of 65 is the baseline. Human managers do punish, with values from the data showing punishments of 0-30 points. A manager that never punishes allows free-riding to go unchecked, reducing the common pool. The goal is to improve on 65 by introducing punishment.

3. **How should punishment be determined?** Looking at human punishment patterns: there's a negative correlation (-0.23) between contribution and punishment, meaning higher contributors tend to be punished less. This suggests a threshold-based approach: punish players who contribute below some level. The correlation is modest, indicating other factors influence punishment decisions.

4. **Designing the first improvement.** The seed rule has no parameters. A simple improvement is to add two parameters: `c0` (threshold for low contribution) and `p0` (punishment amount). Punish players who contribute below `c0` with `p0` points. This directly addresses the free-riding problem seen in the human data.

## Key findings

1. The seed rule (no punishment) achieves a pool of ~65 against `ah`. Human managers do punish, suggesting punishment is beneficial for deterring free-riding.

2. Human punishment has a modest negative correlation (-0.23) with contribution: higher contributors tend to be punished less.

3. Punishment amounts range 0-30 in the data, suggesting varying severity is used.

## Hypothesis

A manager should punish players who contribute below a threshold, as this deters free-riding and increases the common pool. The rule should have two parameters: `c0` (the contribution level below which a player is punished) and `p0` (the punishment amount for low contributors).

Lower values of `c0` mean punishing more players (more severe threshold), while higher values mean punishing fewer (more lenient). Higher values of `p0` mean harsher punishment for low contributors.

The sweep should explore `c0` from 1 to 20 in log space and `p0` from 0 to 30 linearly.

### Justification

1. Human data shows punishment is used and has a negative correlation with contribution (though modest).

2. A manager that never punishes (seed rule) scores 65; a punishment-based rule should improve on this by deterring free-riding.

3. A threshold-based rule with a configurable punishment level is simple, interpretable, and directly addresses free-riding.

## Change

- Base version: pfsmoke-20261127-1
- Operation: add
- Idea: Add a threshold-based punishment rule where players contributing below `c0` receive punishment of `p0` points
- Justification: Analysis of human data showing punishment has a negative correlation with contribution
- Paper: none
- Claim: none
- Analysis: scripts/policy_finder/pfsmoke-20261127-1/pfsmoke-20261127-1-v2_analysis.py computes the average punishment gap; value 0.0421 indicates minimal gap, but negative correlation exists
- Expected: The threshold rule should improve on the seed rule's pool of ~65 by deterring free-riding through targeted punishment
