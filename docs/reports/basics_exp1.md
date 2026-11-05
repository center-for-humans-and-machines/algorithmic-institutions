# Experiment 1: the game, the data and the simulation

What a rule-based manager for experiment 1 faces. Experiment 2's game, with two groups and
switching, is in `basics.md`. That document is not about this game.

## The game

- **Group and rounds.** One group of 4 players, managed by one manager, plays 24 rounds.
  Nobody joins or leaves.
- **Contributions.** Each round every player gets 20 points and contributes 0–20 of them to
  the group's pool.
- **Timeouts.** A player who does not move the slider in time times out. They contribute
  nothing, the manager cannot punish them, and they take no part in that round's pool:
  they pay nothing into it and get nothing from it.
- **Punishment.** After seeing the round's contributions, the manager deals each valid
  player a punishment of 0–30 points. It is deducted from that player's payoff, and also
  from the pool.
- **The pool.** It is `1.6 · Σc − Σp` over the valid players, shared equally among them.
  A valid player's payoff for the round is

  ```
  payoff = 20 − c − p + (1.6 · Σc − Σp) / n
  ```

  where the sums and `n` run over the round's valid players.

**The manager's goal** is the players' average payoff:
- per round, the mean payoff of the valid players (0 for a round in which everyone timed
  out);
- summed over the 24 rounds;
- averaged over many games.

Punishing costs the group twice, once from the punished player and once from the pool. It
pays only if it raises later contributions by more than it costs.

## The human data

`experiments/pilot_random1_player_round_slim.csv` holds the pilot sessions, one row per
player and round. Groups of 4 played games of 8 or 16 rounds (`round_number` 0–15). There
are two kinds of pilot:

- `trail_rounds_2` (43 groups): a human manager.
- `random_1` (49 groups): a rule-based manager with randomly drawn parameters, so the
  punishments vary more than a human's would.

| Column | Meaning |
|---|---|
| `session`, `global_group_id`, `group_id` | the session, the group within it |
| `episode`, `episode_id` | the game the group played (a group may play more than one) |
| `experiment_name` | `trail_rounds_2` or `random_1` |
| `round_number` | 0-based round |
| `participant_code`, `player_id` | the player |
| `contribution` | 0–20; 0 when the player timed out |
| `player_no_input` | 1 when the player timed out that round |
| `punishment` | the punishment the player got that round, 0–30 |
| `manager_no_input` | 1 when the human manager timed out (then nobody was punished) |
| `common_good` | the group's pool that round, `1.6 · Σc − Σp` |
| `payoff` | the player's payoff that round, `20 − c − p + common_good / 4` |

## The simulation

Rules are tested in simulation against artificial humans, models trained on these pilots:
- a contribution model that predicts each player's contribution from the group's previous
  rounds (contributions, punishments and timeouts);
- a timeout model that predicts who times out.

Every game is one group of 4 over 24 rounds: longer than the pilots, so the models play
some rounds past what they were trained on.

The reference sims, `plots/simulation/29_EXP1_reference/per_round.parquet`, play the
models under two managers, 1000 games each:
- `zero`, which never punishes;
- `humanlike`, a model trained to punish like the pilots' human managers.

One row per player and round:

| Column | Meaning |
|---|---|
| `run` | `ah exp1 managed by <manager>` |
| `episode` | the game, from 0 |
| `participant_code` | the player, `<index>_<episode>` |
| `round_number` | 0–23 |
| `contribution` | 0–20. **Not meaningful when `contribution_valid` is False:** a timed-out player contributed nothing, and the column holds the model's placeholder input. |
| `contribution_valid` | False where the player timed out |
| `punishment` | 0–30, 0 for a timed-out player |
| `common_good` | each valid player's share of the pool, `(1.6 · Σc − Σp) / n` |
| `payoff` | `20 − contribution − punishment + common_good`. Meaningful on valid rows only. |

`aggregates.csv` beside the parquet holds the per-round means by manager.
