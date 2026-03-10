# Pseudo Group Matching: Data Processing Flow

This document describes every step of
`scripts/data_creation/pilot_pseudo_group_matching.py`, which transforms
old 4-player pilot experiment data into the new 8-player format used by
the artificial human training pipeline.

## Usage Modes

The script has two subcommands: `transform` and `combine`.

### Mode 1: `transform` -- Convert 4-player data to 8-player format

Reads an old 4-player pilot CSV, randomly pairs episodes with matching
round counts, and writes an 8-player CSV.

```
python scripts/data_creation/pilot_pseudo_group_matching.py transform \
    --in_path <input_csv> \
    --out_path <output_csv> \
    --seed <int, default=42>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--in_path` | Yes | Path to old 4-player CSV |
| `--out_path` | Yes | Path for the transformed 8-player CSV |
| `--seed` | No | Random seed for pairing (default: 42) |

### Mode 2: `combine` -- Merge transformed data with new data

Concatenates a transformed old-data CSV with the new group-switching
CSV into a single file for training.

```
python scripts/data_creation/pilot_pseudo_group_matching.py combine \
    --transformed_path <transformed_csv> \
    --new_data_path <new_data_csv> \
    --out_path <output_csv>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--transformed_path` | Yes | Path to the transformed 8-player CSV (output of `transform`) |
| `--new_data_path` | Yes | Path to the new group-switching CSV |
| `--out_path` | Yes | Path for the combined output CSV |

### Example: Full pipeline

```bash
# Step 1: Transform old pilot data to 8-player format
python scripts/data_creation/pilot_pseudo_group_matching.py transform \
    --in_path experiments/pilot_random1_player_round_slim.csv \
    --out_path experiments/pilot_random1_pseudo_8_agents.csv

# Step 2: Combine with new group-switching data
python scripts/data_creation/pilot_pseudo_group_matching.py combine \
    --transformed_path experiments/pilot_random1_pseudo_8_agents.csv \
    --new_data_path experiments/group_switching_ah_data_8_agents.csv \
    --out_path experiments/combined_pilot_group_switching.csv
```

## Input Format

The script reads `pilot_random1_player_round_slim.csv`, which has
one row per player per round:

| Column | Type | Description |
|--------|------|-------------|
| session | str | Session identifier (e.g., `mqqsqhav`) |
| global_group_id | str | Group within session (e.g., `mqqsqhav #3`) |
| episode | int | Episode number within the session |
| episode_id | int | Dense integer identifying the episode globally |
| experiment_name | str | Experiment label (e.g., `trail_rounds_2`, `random_1`) |
| round_number | int | 0-based round index within the episode |
| participant_code | str | Unique participant identifier |
| player_no_input | int | 1 if the player did not submit input, 0 otherwise |
| manager_no_input | float | 1.0 if the manager did not submit input, 0.0 otherwise; may be NaN |
| player_id | int | Player index within the group (0-3) |
| contribution | float | Player's contribution in this round |
| punishment | float | Punishment received by the player |
| payoff | float | Player's payoff |
| common_good | float | Sum of contributions of the 4 players in this group |

Each episode consists of a single group of 4 players playing multiple
rounds together. A `(global_group_id, episode_id)` pair uniquely
identifies one such episode.

## Output Format

The output CSV matches `group_switching_ah_data_8_agents.csv`, which
the downstream training pipeline (`src/aimanager/generic/data.py`)
expects:

| Column | Type | Description |
|--------|------|-------------|
| session | str | Carried over from whichever input group is being written |
| global_group_id | str | Synthetic pair identifier: `pair_0`, `pair_1`, ... |
| group_id | int | Sub-group membership: 0 for the first group, 1 for the second |
| episode | int | Carried over from the original episode |
| episode_id | int | New dense integer for the merged pair (0, 1, 2, ...) |
| experiment_name | str | Preserved from the original row |
| round_number | int | Preserved from the original row |
| participant_code | str | Preserved from the original row |
| player_no_input | int | Preserved, cast to int |
| manager_no_input | int | Preserved, cast to int; NaN replaced with 0 |
| player_id | int | 0-3 for group_id=0; 4-7 for group_id=1 |
| contribution | float | Preserved from the original row |
| punishment | float | Preserved from the original row |
| payoff | float | Preserved from the original row |
| common_good | float | Recomputed: sum of all 8 players' contributions in the round |

Each row in the output represents one player in one round.
A single paired episode produces 8 rows per round (4 from each group).

## Processing Steps

### Step 1: Load CSV

```python
df = pd.read_csv(in_path)
```

The entire input CSV is loaded into a pandas DataFrame.

### Step 2: Identify Episodes and Count Rounds

```python
episodes = (
    df.groupby(["global_group_id", "episode_id"])
    .agg(n_rounds=("round_number", "nunique"))
    .reset_index()
)
```

Groups the data by `(global_group_id, episode_id)` -- each combination
identifies one 4-player episode. Counts the number of distinct
`round_number` values in each episode. The result is a table like:

| global_group_id | episode_id | n_rounds |
|-----------------|------------|----------|
| mqqsqhav #3    | 112        | 8        |
| mqqsqhav #9    | 124        | 16       |
| ...             | ...        | ...      |

### Step 3: Bucket by Round Count, Shuffle, and Pair

```python
rng = random.Random(seed)
pairs = []

for n_rounds, bucket in episodes.groupby("n_rounds"):
    keys = list(zip(bucket["global_group_id"], bucket["episode_id"]))
    rng.shuffle(keys)
    if len(keys) % 2 != 0:
        dropped = keys.pop()
        warnings.warn(...)
    for i in range(0, len(keys), 2):
        pairs.append((keys[i], keys[i + 1]))
```

This is the core pairing logic:

1. **Group episodes by round count.** Only episodes with the same number
   of rounds can be paired, because their round sequences must align
   temporally (round 0 with round 0, round 1 with round 1, etc.).

2. **Shuffle within each bucket** using a seeded `random.Random` instance
   for reproducibility. The seed defaults to 42 and can be overridden
   via `--seed`.

3. **Handle odd counts.** If a bucket has an odd number of episodes, the
   last episode after shuffling is dropped and a warning is emitted
   (e.g., `Dropping unpaired episode ('dbcrara4 #21', 63) (6 rounds)`).

4. **Pair sequentially.** After shuffling, episodes are paired in order:
   index 0 with 1, index 2 with 3, etc. Each pair is stored as a tuple
   `((global_group_id_a, episode_id_a), (global_group_id_b, episode_id_b))`.

With `pilot_random1_player_round_slim.csv` and seed=42, this produces
66 pairs from 135 episodes (3 episodes dropped from odd-count buckets).

### Step 4: Build 8-Player Rows for Each Pair

For each pair `(ep_a, ep_b)`:

#### 4a. Extract the two episode DataFrames

```python
df_a = df[
    (df["global_group_id"] == ep_a[0]) & (df["episode_id"] == ep_a[1])
].copy()
df_b = df[
    (df["global_group_id"] == ep_b[0]) & (df["episode_id"] == ep_b[1])
].copy()
```

Filters the full DataFrame to get all rows belonging to each episode.

#### 4b. Assign a synthetic global_group_id

```python
global_gid = f"pair_{pair_episode_id}"
```

Each pair gets a new identifier (`pair_0`, `pair_1`, ...) that replaces
the original `global_group_id` values from both episodes.

#### 4c. Compute combined common_good per round

```python
cg_a = df_a.groupby("round_number")["contribution"].sum().to_dict()
cg_b = df_b.groupby("round_number")["contribution"].sum().to_dict()

for round_num in sorted(cg_a.keys()):
    combined_cg = cg_a.get(round_num, 0.0) + cg_b.get(round_num, 0.0)
```

For each round, the common_good is recomputed as the sum of all 8
players' contributions (sum of group A's contributions + sum of group
B's contributions). The original per-group common_good values from the
input are discarded.

#### 4d. Emit rows for Group A (group_id=0)

```python
for _, r in df_a[df_a["round_number"] == round_num].iterrows():
    rows.append(
        _make_row(r, global_gid, group_id=0,
                  player_id=int(r["player_id"]),
                  episode_id=pair_episode_id,
                  common_good=combined_cg)
    )
```

For the first episode in the pair:
- `group_id` is set to **0**
- `player_id` is **preserved as-is** (0, 1, 2, 3)
- `global_group_id` is replaced with the synthetic pair identifier
- `episode_id` is replaced with the pair's dense index
- `common_good` is replaced with the combined value
- All other fields (`session`, `episode`, `experiment_name`,
  `round_number`, `participant_code`, `contribution`, `punishment`,
  `payoff`, `player_no_input`) are preserved from the original row
- `manager_no_input` is cast to int; NaN values become 0

#### 4e. Emit rows for Group B (group_id=1)

```python
for _, r in df_b[df_b["round_number"] == round_num].iterrows():
    rows.append(
        _make_row(r, global_gid, group_id=1,
                  player_id=int(r["player_id"]) + 4,
                  episode_id=pair_episode_id,
                  common_good=combined_cg)
    )
```

For the second episode in the pair:
- `group_id` is set to **1**
- `player_id` is **remapped by adding 4** (0->4, 1->5, 2->6, 3->7)
- Everything else follows the same logic as Group A

#### 4f. Increment pair counter

```python
pair_episode_id += 1
```

The `pair_episode_id` serves as both the dense `episode_id` in the
output and the suffix for the synthetic `global_group_id`.

### Step 5: Assemble and Write Output

```python
out_df = pd.DataFrame(rows)
cols = [
    "session", "global_group_id", "group_id", "episode", "episode_id",
    "experiment_name", "round_number", "participant_code",
    "player_no_input", "manager_no_input", "player_id",
    "contribution", "punishment", "payoff", "common_good",
]
out_df = out_df[cols]
out_df.to_csv(out_path, index=False)
```

All accumulated rows are assembled into a DataFrame, columns are
reordered to match the target format exactly, and the result is written
to the output CSV path.

## Example

Given two 4-player episodes with 8 rounds each:

**Episode A** (`mqqsqhav #3`, episode_id=112):
```
round 0: player 0 contributes 15, player 1 contributes 3, player 2 contributes 15, player 3 contributes 10
         group common_good = 43
```

**Episode B** (`dbcrara4 #7`, episode_id=55):
```
round 0: player 0 contributes 8, player 1 contributes 12, player 2 contributes 5, player 3 contributes 17
         group common_good = 42
```

**After pairing** (pair_0):
```
round 0, group_id=0: player 0 (15), player 1 (3), player 2 (15), player 3 (10)
round 0, group_id=1: player 4 (8), player 5 (12), player 6 (5), player 7 (17)
combined common_good = 43 + 42 = 85
global_group_id = "pair_0", episode_id = 0
```

## Run Output

### Transform

With `pilot_random1_player_round_slim.csv` and default seed:

```
$ python scripts/data_creation/pilot_pseudo_group_matching.py transform \
    --in_path experiments/pilot_random1_player_round_slim.csv \
    --out_path experiments/pilot_random1_pseudo_8_agents.csv

UserWarning: Dropping unpaired episode ('dbcrara4 #21', 63) (6 rounds)
UserWarning: Dropping unpaired episode ('mqqsqhav #18', 100) (7 rounds)
UserWarning: Dropping unpaired episode ('yaxdn9bp #4', 129) (16 rounds)
Wrote 5728 rows (66 pairs) to experiments/pilot_random1_pseudo_8_agents.csv
```

- 135 input episodes -> 3 dropped (odd buckets) -> 66 pairs
- 5728 output rows (66 pairs x variable rounds x 8 players)

### Combine

```
$ python scripts/data_creation/pilot_pseudo_group_matching.py combine \
    --transformed_path experiments/pilot_random1_pseudo_8_agents.csv \
    --new_data_path experiments/group_switching_ah_data_8_agents.csv \
    --out_path experiments/combined_pilot_group_switching.csv

Combined 5728 + 2720 = 8448 rows to experiments/combined_pilot_group_switching.csv
```

- 5728 rows from transformed old data + 2720 rows from new data = 8448 total
- 3 experiment names in combined output: `trail_rounds_2`, `random_1`, `ah_group_switching`
