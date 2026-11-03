# The game, its actors and its data

The rules and the setup only: how people played is left to the data.

## The game

Eight players form two groups of four, each group with its own manager, and
play 24 rounds (`round_number` 0-23).

Each round:

1. **Endowment.** Every player receives 20 points into their private account.
2. **Contribution.** Every player pays 0-20 of them, in whole points, into
   their group's common pool.
3. **Punishment.** Each group's manager deals every member of its group a
   punishment of 0-30 points, after seeing that round's contributions. A
   punishment is deducted from the player's private account and from the
   group's common pool, so it is costly to the whole group.
4. **Payout.** The common pool is the group's contributions times 1.6, minus
   its punishments:

   ```
   pool = 1.6 * sum(contributions) - sum(punishments)
   ```

   It is split equally among the group's members who contributed that round,
   whatever each gave. A member's round payoff is

   ```
   payoff = 20 - contribution - punishment + pool / members who contributed
   ```

**Switching.** After rounds 3, 7, 11, 15 and 19 (the 4th, 8th, ... 20th round;
5 decisions in all) every player may move to the other group, and plays in
their new group from the next round on. Groups start at four players each but
can then hold anything from 0 to 8; an empty group has no pool.

**Timeouts.** A player who submits no contribution in time contributes 0, is
not punished and gets no share of the pool that round. A manager who submits
no punishments in time punishes no one that round. A player can also miss a
switch decision.

**The manager's aim** is a large common pool in its group, over the whole
game.

## The human experiments

`experiments/2group_8agent_50ep.csv`: 50 games of the game above, in 6
sessions, played by human participants, both as contributors and as the two
groups' managers. The file stores every game twice, the second copy with the
group labels swapped: keep one copy per game (deduplicate on `session`,
`episode`, `round_number` and `participant_code`).

One row per player and round:

| Column | Meaning |
|---|---|
| `session`, `episode` | the game (a session plays several episodes) |
| `round_number` | 0-23 |
| `group_id` | the player's group this round, 0 or 1 |
| `participant_code`, `player_id` | the player (`player_id` 0-7 within the game) |
| `contribution` | 0-20; 0 if the player timed out |
| `punishment` | 0-30, dealt by the group's manager this round |
| `payoff` | the player's round payoff |
| `common_good` | the group's common pool this round |
| `player_no_input` | 1 if the player timed out this round |
| `manager_no_input` | 1 if the group's manager timed out this round |
| `selection_timeout` | 1 if the player missed the switch decision before this round |

## The simulation

A manager rule is tested in simulation, not with people:

- **Artificial humans** play the eight players. They are models trained on
  the human experiments: one decides each player's contribution, one whether
  the player switches groups at a decision.
- **`ah`**, the artificial human manager, is a model trained on the human
  managers' punishments. In a test, the rule manages one group and `ah` the
  other.
- **The score** is the rule's group's common pool, averaged over rounds and
  episodes (an empty group counts 0), compared with `ah`'s group's.

In the simulation a timed-out player's contribution is recorded as 0, and a
punishment aimed at a timed-out player is charged 0.
