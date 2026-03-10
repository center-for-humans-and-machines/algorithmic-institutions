# [DRAFT] Pseudo group matching and Group IDs for old experiment data

## Goal

Enable old 4-player pilot experiment data (`pilot1_player_round_slim.csv`,
`pilot_random1_player_round_slim.csv`) to be used alongside the new 8-player
group-switching data for training artificial humans. This requires transforming
old data into the new 8-player format by randomly pairing two 4-player groups,
assigning sub-group IDs, remapping player IDs, and recomputing common good.

## Plan

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | Preprocessing script | New script to transform old 4-player CSVs into 8-player format | No |
| 2 | Training config | New config (or updated config) that references combined data | No |
| 3 | Data concatenation | Merge old (transformed) and new data into a single training CSV | No |

### 1. Preprocessing script

Create `scripts/data_creation/pilot_pseudo_group_matching.py` that:

- Reads an old-format CSV (no `group_id` column, 4 players per group, player_id 0-3).
- Groups rows by `(global_group_id, episode_id)` to identify distinct 4-player game episodes.
- Counts rounds per episode; only pairs episodes with the same number of rounds.
- Randomly pairs two 4-player episodes (within same round-count bucket) using a configurable seed.
- For each pair, produces 8-player rows:
  - First group: `group_id=0`, `player_id` unchanged (0-3).
  - Second group: `group_id=1`, `player_id` remapped to 4-7.
  - `global_group_id` is set to a new synthetic identifier for the pair.
  - `common_good` is recomputed as the sum of contributions from both groups for each round.
  - `episode_id` is reassigned as a new dense integer for the merged pair.
- If there is an odd number of episodes in a round-count bucket, the unpaired episode is dropped (logged with a warning).
- Outputs a CSV with the same columns and order as `group_switching_ah_data_8_agents.csv`.
- Accepts CLI args: `--in_path`, `--out_path`, `--seed` (default: a fixed integer for reproducibility).

### 2. Training config

Create a new training config YAML (e.g., `configs/training/artificial_humans/combined_old_new.yml`) that:

- Points `data_file` to the combined CSV produced in section 3.
- Uses `experiment_names` that includes both the old experiment names (e.g., `trail_rounds_2`) and `ah_group_switching`.
- Keeps the same model architecture and hyperparameters as `group_switching_21.yml` (`n_player: 8`, `n_groups: 2`, `agent_group` encoder with 2 levels).

### 3. Data concatenation

Add a small script or a step in the preprocessing script that concatenates the
transformed old data CSV(s) with `group_switching_ah_data_8_agents.csv` into a
single combined CSV. This can be as simple as `pd.concat` followed by
`to_csv`. The combined file should go in `experiments/` (e.g.,
`experiments/combined_pilot_group_switching.csv`).

## Implementation notes

- The pairing logic should shuffle episodes within each round-count bucket, then
  pair them sequentially (index 0 with 1, 2 with 3, etc.). This is simpler than
  random sampling and still produces random pairs given the shuffle.
- The old data has `manager_no_input` as float (`1.0`/`0.0`); cast to int in the
  output to match the new format.
- The old data contains multiple `experiment_name` values (e.g., `trail_rounds_2`);
  preserve them so the training config can filter as needed.
- `parse_agent_rounds()` in `src/aimanager/generic/data.py` requires a `group_id`
  column (used to set `agent_group`). The output CSV must include this column --
  no changes to `data.py` are needed.
- The `pilot1_player_round_slim.csv` file is stored in Git LFS. The script
  should handle it like any other CSV (LFS is transparent to `pd.read_csv` when
  checked out).

## Next Actions

- [ ] Implement `scripts/data_creation/pilot_pseudo_group_matching.py`
- [ ] Run the script on both `pilot1_player_round_slim.csv` and `pilot_random1_player_round_slim.csv`
- [ ] Concatenate outputs with `group_switching_ah_data_8_agents.csv` into a combined CSV
- [ ] Create training config `configs/training/artificial_humans/combined_old_new.yml`
- [ ] Verify end-to-end: train artificial humans on combined data on the Raven cluster
