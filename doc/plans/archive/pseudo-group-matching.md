# [DONE] Pseudo group matching and Group IDs for old experiment data

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
  - `common_good` is preserved per-group (each group keeps its original common good value).
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

- [x] Implement `scripts/data_creation/pilot_pseudo_group_matching.py`
- [x] Run the script on `pilot_random1_player_round_slim.csv`
- [x] Concatenate outputs with `group_switching_ah_data_8_agents.csv` into a combined CSV
- [x] Create training configs (`pseudo_group.yml`, `pseudo_group_combined.yml`)
- [x] Verify end-to-end: train artificial humans on combined data on the Raven cluster

## Post-implementation: simulation mismatch debugging

After training, simulations revealed that AH model contributions did not match
pilot data. Investigation uncovered a bug in training
(`src/aimanager/artificial_humans/train.py`) and two bugs in the simulation
environment (`src/aimanager/manager/environment.py`):

### Bug 0: Training used fully-connected edges across groups

`create_fully_connected` in `train.py` built edges between all agents
regardless of group membership. This meant the GNN message-passing during
training mixed signals across groups that should be independent.

**Fix**: Added `n_agent_groups` parameter to `create_fully_connected`. Edges
are now restricted to within-group connections (`i // agents_per_grp ==
j // agents_per_grp`). Training calls pass `n_groups` from the config:

### Bug 1: Simulation agent groups all assigned to group 0

`ArtificialHumanEnv.__init__` created `agent_groups` as
`th.zeros((batch_size, n_agents))`, assigning every agent to group 0. The
model was trained with proper group assignments (0 and 1), so at simulation
time the `agent_group` feature was wrong for half the agents.

**Fix**: Compute agent groups by evenly distributing agents across groups:
```python
agent_groups = th.arange(n_agents).div(
    n_agents // n_groups, rounding_mode="floor"
).clamp(max=n_groups - 1)
```

### Bug 2: Group assignments lost on state reset

`reset_state()` was called before `update_groups()` in `__init__`, and
`reset_state()` itself zeroed out the `agent_group` tensor. This meant
group assignments were overwritten every time the environment reset.

**Fix**: Call `update_groups()` before `reset_state()`, and preserve
`self.agent_groups` in `reset_state()` instead of zeroing:
```python
"agent_group": self.agent_groups.clone()
if hasattr(self, "agent_groups")
else th.zeros(size, dtype=th.int64, device=self.device),
```

### Outcome

With both fixes applied, simulation contributions match pilot data
distributions, confirming the trained models are correct. The combined
model (pilot pseudo + group switching) is slightly less accurate because
pseudo-grouped data lacks real inter-group dynamics and the real group
switching dataset is small.
