# [DRAFT] Retrain AH stack on 50-episode dataset and validate

Tracks issue #85.

## Goal

Retrain the existing AH stack (contribution, punishment-autoregressive, switch
predictor) on the new 50-episode group-switching dataset and check that the
AH-vs-pilot simulation still tracks pilot behaviour. Hyperparameters and
architectures are frozen at the #78/#82 values; only the dataset changes.

## Status of prior work

Data analysis (issue tasks 1–2) was carried out on a personal laptop and
documented in the comment on issue #85. Findings:

- 50ep is a clean shape-preserving extension of 35ep (+15 episodes, +43% rows,
  same 24 rounds × 8 agents). Distributional summaries barely move.
- Feature-importance rankings for contribution, punishment, and switch targets
  are stable from 35ep → 50ep. No retuning is justified by the new data.

The preprocessed long-format CSV produced on that laptop was **not pushed**, so
preprocessing must be re-run on this machine before any training can be
launched on the cluster.

Raw data: `experiments/group_switching/final/session_filtered.csv` (in repo).

## Plan

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | Preprocess 50ep raw → long | Run `group_switching_preprocess.py` to produce `experiments/2group_8agent_50ep.csv` | No |
| 2 | 50ep training configs | Add `_50ep` copies of contribution / punishment-autoregressive / switch configs | No |
| 3 | Train AH stack on cluster | Submit three training jobs on Raven | No |
| 4 | 50ep simulation config | Add `group_switching_ah_punishment_50ep.yml` mirroring the 35ep version | No |
| 5 | Run simulation on cluster | Launch sim against 50ep pilot | No |
| 6 | Compare sim vs pilot | Generate the standard #78 plot set; report deltas vs the 35ep baseline | No |

### 1. Preprocess

- Run `python scripts/data_creation/group_switching_preprocess.py` against
  `experiments/group_switching/final/session_filtered.csv` to produce
  `experiments/2group_8agent_50ep.csv` (24 rounds × 8 agents, 50 episodes).
- Sanity-check row count (~9,600) and that `agent_group`, `does_switch`,
  `prev_common_good`, `switch_mask` columns derive correctly downstream
  (`src/aimanager/generic/data.py`).

### 2. 50ep training configs

Three configs, each a minimal copy of its 35ep ancestor with only `data_file`,
`labels.dataset`, and `output_dir` changed.

- `configs/training/artificial_humans/group_switching_contribution_50ep.yml`
  - base: `group_switching_contribution_35ep.yml`
  - `data_file: experiments/2group_8agent_50ep.csv`
  - `labels.dataset: 50ep`
  - `output_dir: artifacts/artificial_humans/group_switching_contribution_50ep`

- `configs/training/artificial_humans/punishment/autoregressive_50ep.yml`
  - base: `punishment/autoregressive.yml`
  - `data_file: experiments/2group_8agent_50ep.csv`
  - `labels.dataset: 50ep`
  - `output_dir: artifacts/artificial_humans/punishment_autoregressive_50ep`

- `configs/training/artificial_humans/switch_predictor/opt_50ep.yml`
  - base: `switch_predictor/opt.yml`
  - `data_file: experiments/2group_8agent_50ep.csv`
  - `labels.dataset: 50ep`
  - `output_dir: artifacts/artificial_humans/switch_pred_opt_50ep`

Architecture / optimiser / training args (`epochs`, `batch_size`,
`hidden_size`, `lr`, `weight_decay`, etc.) are unchanged. `n_cross_val: 5`
stays — 50 episodes still divides cleanly (10 per fold).

### 3. Train on cluster

- Submit each via `scripts/train_cluster.sh` / `python -m aimanager train-ah`,
  matching how the 35ep configs are launched.
- Confirm SSH ControlMaster to Raven is up before submitting.
- Three independent jobs; can run in parallel.

### 4. Simulation config

- `configs/simulation/ah_testing/group_switching_ah_punishment_50ep.yml`
- Base: `group_switching_ah_punishment_35ep.yml`
- Update model paths to the 50ep artifacts:
  - `contribution_model:` → `..._contribution_50ep/.../...__dataset_50ep.pt`
  - `switch_model:` → `..._switch_pred_opt_50ep/.../...__dataset_50ep.pt`
  - `path:` (punishment human manager) → `..._punishment_autoregressive_50ep/.../...__dataset_50ep.pt`
  - `valid_model:` keeps the existing `raven_script_22` artifact (unchanged in
    issue scope)
- Update pilot-side fields:
  - `pilot_data_file: experiments/2group_8agent_50ep.csv`
  - `output_dir: plots/simulation/ah_group_switching_punishment_50ep`
  - `figure_name: group_switching_punishment_50ep`

### 5. Run simulation

- `python -m aimanager simulate configs/simulation/ah_testing/group_switching_ah_punishment_50ep.yml`
  on Raven (mirrors how the 35ep simulation was run for #82/#84).

### 6. Compare sim vs pilot

- Generate the standard plot set from #78 against the 50ep pilot:
  `comparison_pilot_vs_sim`, `switch_count_per_round`,
  `group_size_evolution_global`.
- Report key trajectory and switching deltas vs the 35ep baseline (#82) and
  vs pilot. Flag any qualitative change in fidelity (e.g. trajectory mean,
  late-round switching, group-size variance).

## Out of scope

- Hyperparameter or architecture changes — frozen at #78/#82 values.
- Re-running feature-importance analysis on 50ep — already done and
  documented in the issue comment.
- RL manager training on 50ep AHs — separate follow-up (see project memory:
  RL manager training next).

## Open questions

- Are the 35ep contribution/punishment/switch artifacts kept in place, or do
  the 50ep models replace them as the canonical AH stack? (Default
  assumption: keep both side-by-side; the 35ep configs remain reproducible.)
