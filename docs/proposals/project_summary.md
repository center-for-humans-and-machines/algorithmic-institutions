# Algorithmic Institutions
## Environment
An eight member, two governor public goods game with sanctioning. So first round, 8 players are randomly assigned to two governors (4 each). Each member can decide to contribute from 0-20, then the governor distributes sanctioning tokens. Public good value is summed contributions * 1.6 minus sanctioning tokens. Then each individual keeps what they didn't contribute, loses what he was sanctioned and gets the value of the common project divided by the members of the group. Everyone gets full information of all contributions/payoffs and sanctioning in both groups. This happens for 24 rounds. Every 4 rounds, participants can decide to switch groups (which also means switching the governor).

## Data collection
On Prolific online platform, with groups of 10. Everyone got instructions and control questions and two were randomly assigned as governors. Everyone incentivized by bonus payments proportional to payoffs. Governors maximized the total sum of payoffs in their groups.

## Behavioral cloning
Using various methods (linear, GNN, xdg_boost), we train models to reproduce the human behavioral distribution. We train independent models for contributions, switching and sanctioning.

## Reinforcement Learning
Using the contribution and switching behavioral clones, we simulate an RL environment to optimize a sanctioning strategy to optimize own group size and mean payoff.

## Core predictions
1. Participants react to punishments, either by increasing their contribution or leaving the group.
2. When participants switch, they adapt to the norm of the new group (I.e. higher contribution when switching to a more highly contributing group)
3. Zero sanctioning might be effective in the short term, but there is long-run sorting into groups with fairly sanctioning manager (as there will be higher contributions there).
