# Punisher current-contribution re-baseline: rerun cases

Companion to `punisher-current-contribution.md` (the experiment log). Every AH punisher so far conditioned on the previous round's contribution while the human manager punishes the current one (`punisher_lag_check.md`, scratchpad). The punishers are retrained on the current contribution (stage C); this note fixes which simulations are rerun with them (stage D), where their "before" data lives, and how to launch and score the reruns. Branch `auto/punisher-current-contribution-sims`, to be merged into `auto/punisher-current-contribution`.

## Artifact contract (retrained punishers)

| slot | before | after |
|---|---|---|
| linear, plain | `artifacts/baselines/punishment_multinomial_best_with_contr.joblib` | `artifacts/baselines/punishment_multinomial_current_contr.joblib` |
| linear, severity-copula-stamped (the PR stacks) | `artifacts/baselines/punishment_multinomial_severity_copula.joblib` | `artifacts/baselines/punishment_multinomial_current_contr_severity_copula.joblib` |
| GNN | `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt` | `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt` |

The GNN "after" path follows the stage contract (`artifacts/artificial_humans/punishment/<name>/`, same `model/<grid labels>.pt` layout as the existing artifact); note the existing artifact sits one level up, at `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/`. If stage C's training config writes elsewhere, only the `gnn` path in `23_2g8a_self_gnn_contr_gnn_switch_curpun.yml` changes.

## Cases

All configs live in `configs/simulation/manager_testing/`, carry `save_per_round: true`, and are byte-identical to their source except for the punisher path, `output_dir` and `figure_name` (case e also drops the ridge and gaussian pairings). Sim dirs are `plots/simulation/<config basename>`.

| case | source branch (PR) | source sim config | before sim dir (`per_round.parquet`) | punisher before | punisher after | curpun config |
|---|---|---|---|---|---|---|
| a | `origin/auto/contribution-group-vnode` (#179, mean 1.099) | `23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml` | `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch` | `punishment_multinomial_severity_copula.joblib` | `punishment_multinomial_current_contr_severity_copula.joblib` | `23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch_curpun.yml` |
| b | `origin/auto/contribution-punishment-response` (#181, RCE 0.91) | `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch.yml` | `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch` | `punishment_multinomial_severity_copula.joblib` | `punishment_multinomial_current_contr_severity_copula.joblib` | `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun.yml` |
| c | `origin/auto/contribution-inflated-gmlp` (#177, RCB 1.46) | `23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml` | `plots/simulation/23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch` | `punishment_multinomial_severity_copula.joblib` | `punishment_multinomial_current_contr_severity_copula.joblib` | `23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun.yml` |
| d | `origin/auto/switch-exodus-k-onehot` (#174, RCE 0.83; artifacts copied from its child `origin/auto/contribution-inflated-gmlp`) | `23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml` | `plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch` | `punishment_multinomial_severity_copula.joblib` | `punishment_multinomial_current_contr_severity_copula.joblib` | `23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun.yml` |
| e (lin) | `main` (sweep top stack, gnn x gnn) | `23_2g8a_self_gnn_contr_gnn_switch.yml` | `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch` (run `lin_multinomial_self`) | `punishment_multinomial_best_with_contr.joblib` | `punishment_multinomial_current_contr.joblib` | `23_2g8a_self_gnn_contr_gnn_switch_curpun.yml` (run `lin_multinomial_self`) |
| e (gnn) | `main` (sweep top stack, gnn x gnn) | `23_2g8a_self_gnn_contr_gnn_switch.yml` | `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch` (run `gnn_self`) | `punishment_rnn_edge_50ep_doubled/.../architecture_node+edge+rnn__dataset_50ep_doubled.pt` | `punishment/rnn_edge_50ep_doubled_current_contr/.../architecture_node+edge+rnn__dataset_50ep_doubled.pt` | `23_2g8a_self_gnn_contr_gnn_switch_curpun.yml` (run `gnn_self`) |

Run names inside `per_round.parquet` / `scores.csv` are `ah group_switching managed by <pairing>`; the pairing names are kept from the source configs so the before and after rows pair by name.

Copied onto this branch for cases c and d (git-tracked; the `.pt` / `.parquet` / `.csv` are LFS objects that already exist upstream): `artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.{joblib,params.json}`, `artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.{joblib,params.json}`, `artifacts/artificial_humans/switch_exodus_k_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`, and the two before sim dirs (`per_round.parquet`, `evaluation/{metrics,scores}.csv`). Everything else the configs reference is already on the PR 181 branch.

## Code-lineage caveat (cases c and d)

The gaussian_mlp contribution bundles unpickle `InflatedGaussianMLPRegressor` / `GaussianMLPRegressor` from `scripts/baselines/gaussian_regressor.py` and sample through the group-copula adapter in `linear_ah.py`; both exist only on the gmlp lineage (`origin/auto/contribution-inflated-gmlp`, 17 files / 6224 lines of `src` and `scripts/baselines` diverged from the PR 181 lineage). Verified locally: `joblib.load` of either bundle fails on this branch (`Can't get attribute 'InflatedGaussianMLPRegressor'`) and succeeds on the gmlp lineage. So stage D runs c and d from a second local checkout of that lineage carrying the punisher fix -- `git worktree add ../curpun-gmlp origin/auto/contribution-inflated-gmlp`, cherry-pick stage A's code commits (the `linear_ah.py` / feature-legality change that lets a punisher bundle read the current `contribution`), and point `CURPUN_GMLP_DIR` at it. The runner copies the two configs and the retrained copula punisher into that checkout and syncs it to its own remote dir (`${CURPUN_REMOTE_DIR}-gmlp`). Cases a, b and e run from the merged `auto/punisher-current-contribution` checkout. If the cherry-pick is not clean, run a, b, e first (`--skip-gmlp`) and add c, d when the second checkout is ready.

## Launch

From the merged `auto/punisher-current-contribution` checkout, with the SSH ControlMaster to Raven open and the retrained artifacts in place:

```bash
export CURPUN_GMLP_DIR=/path/to/curpun-gmlp          # gmlp lineage + punisher fix (cases c, d)
scripts/simulation/run_curpun_reruns.sh --dry-run submit   # prints the commands
scripts/simulation/run_curpun_reruns.sh submit             # sync + sbatch, one job per config
ssh raven squeue -u certuer                                # wait for the five jobs
scripts/simulation/run_curpun_reruns.sh fetch              # sim dirs -> plots/simulation/*_curpun
PYTHON="uv run python" scripts/simulation/run_curpun_reruns.sh evaluate   # 22-row scores + table
```

`submit` expands, per case, to (`AI_REMOTE_DIR=~/repros/ai-runs/punisher-current-contr`; the gmlp cases use the `-gmlp` suffix and `$CURPUN_GMLP_DIR/scripts/simulate_cluster.sh`):

```bash
env AI_REMOTE_DIR=~/repros/ai-runs/punisher-current-contr scripts/simulate_cluster.sh configs/simulation/manager_testing/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch_curpun.yml
env AI_REMOTE_DIR=~/repros/ai-runs/punisher-current-contr scripts/simulate_cluster.sh --no-sync configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun.yml
env AI_REMOTE_DIR=~/repros/ai-runs/punisher-current-contr-gmlp $CURPUN_GMLP_DIR/scripts/simulate_cluster.sh configs/simulation/manager_testing/23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun.yml
env AI_REMOTE_DIR=~/repros/ai-runs/punisher-current-contr-gmlp $CURPUN_GMLP_DIR/scripts/simulate_cluster.sh --no-sync configs/simulation/manager_testing/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun.yml
env AI_REMOTE_DIR=~/repros/ai-runs/punisher-current-contr scripts/simulate_cluster.sh --no-sync configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch_curpun.yml
```

`evaluate` runs `python -m aimanager evaluate <config>` per case (writes `plots/simulation/<dir>_curpun/evaluation/{metrics,scores}.csv` + visuals) and then `python scripts/data_analysis/curpun_rebaseline.py`, which writes `plots/data_analysis/evaluation/punisher_current_contr/rebaseline_table.{csv,md}` and `rce_bands.csv`. The table script tolerates missing after sims (before column only), so it can be run after any subset has landed.

## Before column (old punisher, 22-row suite with RCE)

The source sims' committed `scores.csv` have 21 rows; the RCE row did not exist when they were scored. All six before runs were rescored from their `per_round.parquet` with the merged suite (`auto/punisher-current-contribution` at 091b887, 500 repeats, seed 42; scores only, no visuals) into `plots/data_analysis/evaluation/punisher_current_contr/before/<case>/{metrics,scores}.csv`. The 21 pre-existing rows reproduce the committed values exactly (max abs diff 0.0 on every case), since the resampling plan is drawn once per evaluation and shared by all rows.

| row | a_vnode | b_skip | c_infl | d_kexo | e_lin | e_gnn |
|---|---|---|---|---|---|---|
| CA | 0.961 | 0.848 | 1.442 (1-2) | 1.609 (1-2) | 0.772 | 0.842 |
| CB | 0.953 | 0.830 | 1.017 (1-2) | 0.936 | 0.691 | 0.685 |
| CC | 0.920 | 0.821 | 0.860 | 1.036 (1-2) | 1.606 (1-2) | 1.712 (1-2) |
| CD | 0.922 | 0.796 | 0.961 | 1.116 (1-2) | 0.650 | 0.670 |
| CE | 1.111 (1-2) | 0.910 | 0.803 | 0.947 | 1.332 (1-2) | 1.356 (1-2) |
| CF | 1.076 (1-2) | 0.887 | 0.855 | 1.345 (1-2) | 0.814 | 0.852 |
| CG | 0.899 | 1.310 (1-2) | 1.842 (1-2) | 2.079 (2-5) | 9.850 (> 5) | 10.138 (> 5) |
| SA | 0.864 | 0.784 | 0.916 | 0.810 | 0.721 | 0.659 |
| SB | 1.111 (1-2) | 1.040 (1-2) | 0.965 | 0.891 | 0.754 | 0.744 |
| SC | 0.977 | 1.023 (1-2) | 1.233 (1-2) | 0.980 | 3.270 (2-5) | 3.455 (2-5) |
| PA | 0.582 | 0.630 | 0.621 | 0.619 | 0.634 | 1.267 (1-2) |
| PB | 0.919 | 0.901 | 0.952 | 0.960 | 0.878 | 1.114 (1-2) |
| PC | 0.865 | 0.877 | 0.891 | 0.888 | 0.778 | 1.160 (1-2) |
| PD | 0.775 | 0.854 | 0.919 | 0.865 | 2.935 (2-5) | 2.823 (2-5) |
| RCA | 1.400 (1-2) | 1.469 (1-2) | 1.862 (1-2) | 3.507 (2-5) | 2.035 (2-5) | 2.367 (2-5) |
| RCB | 2.315 (2-5) | 2.087 (2-5) | 1.462 (1-2) | 1.910 (1-2) | 1.928 (1-2) | 1.886 (1-2) |
| RCC | 1.660 (1-2) | 1.613 (1-2) | 1.327 (1-2) | 1.088 (1-2) | 1.539 (1-2) | 1.407 (1-2) |
| RCD | 1.340 (1-2) | 2.205 (2-5) | 1.353 (1-2) | 0.732 | 2.772 (2-5) | 2.893 (2-5) |
| RCE | 1.100 (1-2) | 0.906 | 0.989 | 0.832 | 1.092 (1-2) | 0.986 |
| RSA | 1.355 (1-2) | 1.236 (1-2) | 1.335 (1-2) | 1.317 (1-2) | 0.909 | 1.137 (1-2) |
| RPA | 1.311 (1-2) | 1.227 (1-2) | 1.275 (1-2) | 1.314 (1-2) | 1.268 (1-2) | 1.555 (1-2) |
| RPB | 0.758 | 0.847 | 0.731 | 0.727 | 0.814 | 1.343 (1-2) |
| mean | **1.099** | **1.096** | **1.119** | **1.205** | **1.729** | **1.866** |
| rows <= 1 | 12/22 | 13/22 | 12/22 | 12/22 | 11/22 | 7/22 |

Per-band RCE slopes (OLS of next-round change on punishment received, RCB population; human ++--):

| case | 0-4 | 5-9 | 10-14 | 15-19 | signs vs human |
|---|---|---|---|---|---|
| human | +0.140 | +0.104 | -0.077 | -0.161 | ++-- |
| a_vnode | +0.062 | +0.012 | -0.008 | -0.037 | ++-- (4/4) |
| b_skip | +0.073 | +0.054 | -0.047 | -0.025 | ++-- (4/4) |
| c_infl | +0.033 | +0.061 | -0.051 | -0.231 | ++-- (4/4) |
| d_kexo | +0.103 | +0.037 | -0.116 | -0.174 | ++-- (4/4) |
| e_lin | +0.062 | +0.043 | +0.047 | -0.042 | +++- (3/4) |
| e_gnn | +0.090 | +0.021 | -0.005 | -0.000 | ++-- (4/4) |

## Deviations from the brief

1. Cases c and d cannot run on the PR 181 lineage (see the caveat above); the runner takes a second checkout instead of a code merge, which keeps this branch to configs, artifacts, scripts and this note.
2. The `_curpun` suffix sits after `_switch`, so `evaluation_sweep.py`'s `DIR_PATTERN` does not parse these dirs; they are consumed by `curpun_rebaseline.py` only. The suffix was the brief's naming; rename to `..._curpun_self_...` if the sweep should ever include them.
3. The GNN "after" path uses the brief's `artifacts/artificial_humans/punishment/` directory, one level below where the existing GNN punisher lives.
4. The before scores were written to a separate `before/` location rather than overwriting the source sims' committed `evaluation/scores.csv`, so the parent PRs' outputs stay untouched.
