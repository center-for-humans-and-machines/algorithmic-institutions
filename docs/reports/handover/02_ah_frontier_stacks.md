# AH frontier stacks: model code, training and simulation entrypoints

Handover sub-report 2. Scope: the code, configs and artifacts behind the two
frontier clone stacks (GNN tip #179, Gaussian-MLP tip #177) and Levin's
stack, plus how they are trained and simulated. Campaign machinery and metric
definitions are in other sub-reports. Refs are to `yannik/handover` at
`bb414f5a`. Its model code (`generic/`, `artificial_humans/train.py`,
`simulation/linear_ah.py`) is identical to `origin/autoresearch-optimized-stack`
and `origin/policy-finder-base`. `origin/main` has none of it: no `copula.py`,
`group_vnode.py`, `joint_exodus.py` or `conditional_bernoulli.py`. The gmlp
tip is `origin/auto/contribution-inflated-gmlp` (`c91f9bbf`).

## TL;DR

- The GNN stack runs on this branch, and all of its artifacts are present. Each frontier GNN is the user's original config plus one or two `model_args` flags. The copula parameters are stamped onto the trained `.pt` afterwards by Raven sbatch scripts, not learned by `train-ah`.
- The gmlp stack (#177) does **not** run on this branch. Its contributor class, its group-copula sampler and its one-hot joint-exodus head exist only on `origin/auto/contribution-inflated-gmlp`. Its two artifacts are missing here, and the switch `.pt` there is an LFS pointer with no content in this clone.
- `24_FRONTIER_*` is not the #179 tip as scored by the Atlas. It is the #179 contributor and switch with the #224 punisher (current contribution plus timeout flag, rho 0.48) and `timeout_contribution: 0`. The #179 tip itself is `23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml`. Levin's stack adds the stimulus skip (#181, which the Atlas marks FAIL) and the #192/#196 punisher.
- **Blocker for the proxy path:** `python -m aimanager train-ah <repo AH config>` fails with `KeyError: 'job_id'` (`artificial_humans/train.py:263`; I reproduced it locally). The CLI unwraps `params:` but never adds the `job_id` that `run.py` used to inject. This is the exact example in the `run_on_cluster` skill.
- **Version skew:** every `.joblib` bundle was pickled with scikit-learn 1.6.1, and `pyproject.toml:16` requires `>=1.6.1`. But `algoinst.sif` pins `scikit-learn==1.5.2` (`requirements.raven.txt:10`). Every frontier sim loads a `.joblib` punisher.
- The `.pt` files pickle whole modules (`aimanager.generic.*`, `torch_geometric.nn.models.meta.MetaLayer`). They were written by the Raven `.venv` (Python 3.9, torch < 2.1, numpy < 2), so they should load in the sif. All 106 Python files parse under the 3.9 grammar, and I found no torch API newer than 1.11.

## Role × stack

| Role | GNN tip #179 (Atlas) | Gmlp tip #177 (Atlas) | `24_FRONTIER` | `24_LEVIN` (manager sims, RL, policy finder) |
|---|---|---|---|---|
| Contribution | `GraphNetwork` + `group_vnode` + herding copula (rho 0.0436, phi 1.0). Train cfg `contribution/group_switching_contribution_50ep_group_vnode.yml`. Artifact `group_switching_contribution_50ep_group_vnode_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` (present) | `InflatedGaussianMLPRegressor` (gmlp branch `scripts/baselines/gaussian_regressor.py`) + group copula (rho_p 0.0484, rho_t 0). Cfg `baselines/contribution/gaussian_mlp_inflated.yml` (gmlp branch). Artifact `artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.joblib` (**missing here**, plain blob on gmlp branch) | same as #179 | `GraphNetwork` + `group_vnode` + `stimulus_skip` + herding copula (rho 0.0395, phi 1.0). Cfg `contribution/group_switching_contribution_50ep_vnode_stimulus_skip.yml`. Artifact `group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/...epochs_575.pt` (present) |
| Switch | `GraphNetwork` mlp+rnn+edge + `JointExodusHead` (numeric k/8) + conditional Bernoulli. Cfg `switch_predictor/joint_exodus.yml`. Artifact `switch_joint_exodus/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` (present) | same with `joint_exodus_size_encoding: onehot`. Cfg `switch_predictor/joint_exodus_k_onehot.yml` (gmlp branch). Artifact `switch_exodus_k_onehot/model/...doubled.pt` (**missing here**, LFS pointer on gmlp branch) | as #179 | as #179 |
| Valid (timeout) | `raven_script_22/model/rnn_False__dataset_full.pt` in every stack (present) | same | same | same |
| Punisher (manager `type: linear`) | `punishment_multinomial_severity_copula.joblib`: `[prev_contribution, prev_punishment, round_number, is_first]`, rho 0.351, cfg `baselines/punishment/handcrafted_grid_cat.yml` | same | `punishment_multinomial_current_contr_severity_copula.joblib`: adds `contribution`, `contribution_valid`, rho 0.480, cfg `.../multinomial_current_contr.yml` | `punishment_multinomial_timeout_severity_copula.joblib`: adds `contribution_max` too, rho 0.4273, cfg `.../multinomial_timeout.yml` |
| Sim config | `23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml` (`timeout_contribution: default`) | `23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml` (header: "REFERENCE ONLY ... not on this branch") | `24_FRONTIER_vnode_curpun_..._switch.yml` (t=0) | `24_LEVIN_vnode_skip_timeoutpun_..._switch.yml` (t=0); also `25_LEVIN_run1_*`, `training/rl_manager/04_2g8a_levin.yml` |

All AH paths sit under `artifacts/artificial_humans/`, all bundles under `artifacts/baselines/`. Bundle features and rho come from loading the bundles locally. The copula values come from the `*.copula.json` sidecars, whose sha256 values match the files in the tree.

## 1. Mechanisms: code and the flag that switches each on

- **GNN trunk**: `GraphNetwork` in `src/aimanager/generic/graph.py:130`, registered as `AH_MODELS["graph"]` in `artificial_humans/__init__.py`. The pipeline is `op1` (`MetaLayer` edge and node models) → per-agent `GRU rnn_n` → `op2` readout. The frontier GNNs are the original configs plus flags: `group_switching_contribution_50ep.yml` + `group_vnode: True`, and `opt_50ep_doubled_reanchored.yml` + `joint_exodus: True, joint_exodus_switch_every: 4` (checked by diff).
- **Group virtual node** (`model_args.group_vnode`): `GroupVirtualNode` in `generic/group_vnode.py:68`. It is built at `graph.py:421`, called at `:480` on the post-`op1` embedding, and concatenated before `op2`. It is a GRU over the mean-pooled members plus occupancy k/8, carried across rounds like `rnn_n_h0`.
- **Stimulus skip** (`model_args.stimulus_skip`, Levin only): `graph.py:287` and `:492`. It feeds the post-`op1` embedding straight to `op2` and requires `add_rnn`.
- **Herding copula** (artifact fields `copula_rho`, `copula_phi`, `copula_switch_every`; not a training flag): `generic/copula.py:48 sample_correlated_levels`. Sampling dispatches at `graph.py:786` to `_predict_encoded_copula` (`:611`), with one latent per (episode, agent_group). With phi 1.0 the latent is static for the episode. Teacher-forced probabilities are unchanged (`sample=False` bypasses the copula).
- **Joint exodus + conditional Bernoulli** (`model_args.joint_exodus`, `joint_exodus_switch_every`): `JointExodusHead` in `generic/joint_exodus.py:168` (sizes numeric, `SIZE_NORM` at `:55`), built at `graph.py:404`. Its loss is `train.py:157 joint_exodus_loss`, added at `:209 compute_batch_loss`, with `DROP_INCOMPLETE_PAIRS = True` at `:118`. Sampling dispatches at `graph.py:782` to `_predict_encoded_joint_exodus` (`:660`), then `sample_conditional_bernoulli` (`generic/conditional_bernoulli.py:158`). The copula and the joint head cannot both be on a switch model (assert at `graph.py:220`).
- **Severity copula punisher** (bundle key `copula_rho`): `LinearAHAdapter._sample_levels_copula` at `simulation/linear_ah.py:286`, reached through `get_punishments` (`:343`). There is one shared latent per (round, group) through each agent's multinomial CDF.
- **Gmlp only** (gmlp branch, absent on HEAD): `InflatedGaussianMLPRegressor` (a Gaussian body plus atoms at prev/0/20), bundle keys `copula_rho_p`/`copula_rho_t`, and `model_type == "gaussian_mlp_inflated"` in that branch's `linear_ah.py`. There is also `joint_exodus_size_encoding: onehot` in that branch's `graph.py`/`joint_exodus.py`. HEAD's `scripts/baselines/gaussian_regressor.py` defines only `GaussianRegressor`. The gmlp branch also lacks HEAD's `group_vnode`, `stimulus_skip` and `timeout_contribution`. The two stacks live on divergent code.

## 2. Training entrypoints

### 2a. GNN models (`train-ah`, Raven GPU)

The CLI path is `python -m aimanager train-ah <cfg>` → `cli.py:88 load_and_validate` → `:62 unwrap_orchestrator_config` (returns `params:` when the required keys are not top-level, and exits on `grid`/`include`) → `:124 dispatch_train_ah` → `artificial_humans/train.py:245 main`. `main` reads the CSV (`:288`), calls `create_torch_data` (`:293`), runs `n_cross_val` folds plus a final all-data fit (`generic/data.py:226`, final yield at `:293-294`), and **saves only the final fit**: `model.save` at `train.py:631` to `<output_dir>/model/<job_id>.pt`. W&B runs only if `WANDB_API_KEY` is set (`:297`). `device: cuda` is hard-coded in every AH config.

**The `job_id` gap.** Every config under `configs/training/artificial_humans/` is an orchestrator YAML (`exec:` + `params_only: true` + `params:`), and none has a `job_id` (grep). The SSH orchestrator `artificial_humans/run.py:93-99` builds `job_id` from `labels` (`architecture_node+edge+rnn__dataset_50ep__epochs_575`). `queue_job` writes it into `.log/.../job.yml` and sbatches `scripts/artificial_humans/run_training.sh`. The CLI unwrap (commit `77803906`, "takeover") does not add it. Running `env PYTHONPATH=src python3 -m aimanager train-ah configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode.yml` gives `KeyError: 'job_id'` at `train.py:263`, before any file is written. The untracked `src/aimanager/tests/test_cli_orchestrator_config.py` tests only the unwrap, not training. Fix options (not done): default `job_id` in `train.py` the way `run.py:97` builds it, or submit a flattened config with `job_id:` set.

Original commands (SSH path, for reference): `scripts/train_cluster.sh` → `python src/aimanager/artificial_humans/run.py <cfg>` → sbatch → `python -m aimanager train-ah .log/<...>/job.yml`.

### 2b. GNN contributor copula: calibrate, then stamp (Raven sbatch, CPU)

These scripts imported `graph.py`, so they ran on Raven via sbatch with `$HOME/algorithmic-institutions/.venv/bin/python`. They are not CLI entrypoints.

```
sbatch scripts/artificial_humans/calibrate_copula.slurm [BASE.pt] [PARAMS.json]
  # -> contribution_copula_rho.py --model BASE --roundtrip --preflight --write-params PARAMS
python scripts/artificial_humans/freeze_phi_in_params.py --shipped OLD.json --params NEW.json  # refit of rho only
sbatch scripts/artificial_humans/stamp_copula.slurm [BASE.pt] [PARAMS.json] [OUT.pt]
  # -> make_contribution_copula_artifact.py: copies the base dict, adds the 3 copula fields,
  #    verifies bit-identity and the teacher-forced probabilities, writes OUT + OUT.copula.json
```

Calibration uses the 40 single-copy train episodes (`experiments/baseline/2group_8agent_50ep_bline_train.csv`, per `copula_params.json`). phi was fixed to 1.0 by a ruling. The estimate (0.62 and 0.81) is kept in the JSON. Logs show the jobs ran as `certuer` in `/raven/u/certuer/autoresearch/<branch>` with the `/u/certuer/algorithmic-institutions/.venv` interpreter (`stamp_copula_30257600.log`). Under the proxy, these would need `launcher: python` with a script path as `program_call`. The skill documents only `-m aimanager ...`, so it is unverified whether the server accepts that.

### 2c. Linear / logistic punisher (local, CPU, no PyG)

From the config headers (`multinomial_current_contr.yml:11-24`, `multinomial_timeout.yml`):

```
.venv/bin/python scripts/baselines/run_baseline_cv.py configs/training/baselines/punishment/<cfg>.yml
.venv/bin/python scripts/baselines/inspect_best_model.py data/baselines/punishment_cv_<cfg>.csv \
    --config configs/training/baselines/punishment/<cfg>.yml --save-best --name punishment_<cfg>.joblib
.venv/bin/python scripts/baselines/punishment_copula_rho.py --roundtrip \
    --bundle artifacts/baselines/punishment_<cfg>.joblib --out artifacts/baselines/punishment_<cfg>_severity_copula.joblib \
    [--stamp-rho 0.4273]   # timeout variant: rho carried over by protocol
```

These steps use sklearn `LogisticRegression(C=1.0)` over `handcrafted_grid.build_feature_pool` features, with 4-fold CV on the single-copy train file. `data/` is gitignored. The two force-added CV CSVs (`data/baselines/punishment_cv_multinomial_{ceiling,current_contr}.csv`) are deleted in this working tree, so `--save-best` must start with a fresh `run_baseline_cv.py`.

### 2d. Gaussian-MLP contributor (local, gmlp branch only)

Per `notes/autoresearch_log/contribution-inflated-gmlp.md` steps 4-7 on that branch ("trains locally in ~7 s per fit; nothing ... runs on Raven except the two sims"):
`run_baseline_cv.py configs/training/baselines/contribution/gaussian_mlp_inflated.yml` →
`inspect_best_model.py data/baselines/gaussian_mlp_inflated_cv.csv --config <cfg> --save-best --name contribution_gaussian_mlp_inflated_best.joblib` →
`scripts/baselines/contribution_gmlp_copula_rho.py --bundle ..._best.joblib --config <cfg> --out ..._group_copula.params.json --write-params` →
`scripts/baselines/stamp_contribution_group_copula.py --base ... --params ... --out artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.joblib`.
The k-onehot switch was trained with `train-ah` on `configs/training/artificial_humans/switch_predictor/joint_exodus_k_onehot.yml` (gmlp branch).

## 3. Assembling a stack for simulation

`python -m aimanager simulate <cfg>` → `simulation/simulate.py:804 run_cli` → `run_simulation`, with the device set to cuda if available (`:341`). `load_humans` (`:158-176`) reads `artificial_humans.<name>.{contribution_model, valid_model, switch_model}` and calls `linear_ah.load_ah_model` (`:362`). That function dispatches on extension: `.joblib` → `joblib.load` → `LinearAHAdapter` (`:370`), anything else → `GraphNetwork.load` (`graph.py:893`, a plain `th.load` of the pickled dict). Managers with `type: linear` go to `manager/api_manager.py:243 LinearManager` → `LinearAHAdapter(...).get_punishments`. The `_self` pairings put the punisher on both groups. In `manager/environment.py`, `update_contribution` (`:330`) calls the contributor, then the valid model, and overwrites timed-out contributions with `timeout_contribution` (`:346`). `punish` (`:365`) charges 0 to timed-out players. `step` (`:394`) runs the switch model every round and applies it on decision rounds. `simulate.py:37-39` aliases `torch_geometric.nn.meta` for old pickles.

Existence in the working tree: all 8 artifacts named by `24_FRONTIER`, `24_LEVIN` and the #179 tip config exist and are tracked. The `.pt` and `.csv` content matches the LFS pointer hashes for all 260 "modified" files, so those diffs are LFS materialization only. Both gmlp artifacts are missing (`ls` confirms).

## 4. Data

- `experiments/2group_8agent_50ep.csv`: 19,200 rows = 100 episodes × 24 rounds × 8 players. That is 50 games, each twice with group labels mirrored, sharing a `pair_id`. All rows are `experiment_name: ah_group_switching`. Timeouts: `player_no_input` 560, `manager_no_input` 814, `selection_timeout` 218. It was built by `scripts/data_creation/group_switching_preprocess.py <wide csvs> --n_agents 8` (doubling in `#103`, label timing and timeout fixes in `#105`, June 2026, before the campaign).
- `experiments/baseline/2group_8agent_50ep_bline_{train,test}.csv`: 40 and 10 single-copy games from `scripts/data_creation/split_baseline_holdout.py`. The linear models and every copula calibration use the train file. The GNNs train on the full doubled file (5 pair-aware folds plus a final fit on all 100 episodes), so the final GNN artifacts have seen the 10 "locked" test games (inference from `train.py` and `data.py`).
- Timeouts in training: `contribution_valid` masks the contribution loss. `switch_valid` drops timed-out decisions (`data.py:73-81`). The joint-exodus loss drops incomplete pairs (`train.py:99-118`, keeping 334 of 500 pairs). The punishers use `punishment_valid`, and the #196/#224 punishers also take `contribution_valid` as a feature. `contribution_max` was added in `data.py:204`.
- Compared with the user's original setup: the dataset, the doubling and the pair-aware CV are unchanged. The GNN hyperparameters are unchanged (only flags added). What changed: the GNN punisher (`type: human`, `punishment_rnn_edge_50ep_doubled`) was replaced by a linear multinomial manager, sampling gained copulas, and timeouts gained `timeout_contribution`. The valid model is still the legacy `raven_script_22` artifact (`26a5bf93`, Feb 2026). By the untracked `run/behavioral_cloning/22_contribution_valid_model_v4_raven.yml` it is a 4-player pilot-data model (`pilot_random1_player_round_slim.csv`, input `prev_contribution_valid` only). That attribution is an inference from the artifact name.

## 5. Data-flow sketch

```
50ep.csv (doubled) --data.py--> train.py (train-ah, Raven GPU) --> base .pt
   bline_train.csv --contribution_copula_rho.py (sbatch)--> copula_params.json
                    --make_contribution_copula_artifact.py (sbatch)--> *_herding_copula/model/*.pt
bline_train.csv --handcrafted_grid--> run_baseline_cv --> inspect_best_model --save-best --> X.joblib
                                       --punishment_copula_rho.py--> X_severity_copula.joblib
sim cfg.artificial_humans --load_ah_model(.pt|.joblib)--> ArtificialHumanEnv
sim cfg.managers.linear  --LinearManager/LinearAHAdapter.get_punishments--> env.punish
  --> plots/simulation/<name>/per_round.parquet --> evaluate (local)
```

## 6. Portability risks for `python -m aimanager ...` in `algoinst.sif`

1. **`train-ah` cannot run any repo AH config**: `KeyError: 'job_id'` at `src/aimanager/artificial_humans/train.py:263` (repro above). The skill example at `/opt/algoinst_proxy/.claude/skills/run_on_cluster/SKILL.md:70` uses exactly such a config.
2. **sklearn downgrade**: bundles are 1.6.1 (`_sklearn_version` in every `artifacts/baselines/*.joblib`), `pyproject.toml:16` requires `>=1.6.1`, and `uv.lock` resolves 1.6.1 on py3.9. The sif pins `scikit-learn==1.5.2` (`container/CC_env/requirements.raven.txt:10`). Unpickling `LogisticRegression`/`StandardScaler` across a downgrade probably works with an `InconsistentVersionWarning`, but that is not guaranteed. This hits every sim's punisher (`linear_ah.py:370`, `api_manager.py:250`). sklearn 1.6.1 supports Python 3.9 and numpy < 1.24, so the pin can be raised.
3. **Other sif ≠ lock skews** (lower risk): pandas 2.2.3 vs `>=2.3.3` (`pyproject.toml:12`), pandera `<0.23` vs `>=0.26.1` (`:13`; `data.py` uses only `DataFrameModel`/`Series`), numpy `<1.24` vs lock 1.26.4, fastparquet `<2024.11` vs `>=2024.11`. Pickles reference `numpy.core.*`, which is fine on numpy 1.x.
4. **Whole-module pickles**: the `.pt` files store `aimanager.generic.graph.{EdgeModel,NodeModel}`, `group_vnode.GroupVirtualNode`, `joint_exodus.JointExodusHead` and PyG `MetaLayer` by import path, as dicts (`graph.py:864-890`). Any edit to those classes changes what the loaded models do. The zip layout has no `byteorder`/serialization id, which points to torch < 2.1 (inference). They load in the sif's torch 1.11. They also load in this proxy image (torch 2.5, PyG 2.6, checked), but torch ≥ 2.6 (`weights_only=True` default) would break `graph.py:894`.
5. **`sys.path` imports of scripts**: `linear_ah.py:44` puts `scripts/baselines` on `sys.path` and imports `handcrafted_grid`. Gaussian bundles unpickle `gaussian_regressor.<Class>` by bare module name. This works because the full checkout is bind-mounted. A gmlp bundle would fail to unpickle on HEAD (`InflatedGaussianMLPRegressor` is absent).
6. **Artifacts do not travel**: `POST /sync_to_cluster` excludes `artifacts/` and `plots/` (`SKILL.md:54`). The frontier `.pt`/`.joblib` must already exist in `/u/ykeller/algorithmic-institutions/artifacts/`. They were produced in `/raven/u/certuer/...`. Unverified.
7. **Implicit dependencies**: `pydantic` (imported top-level at `manager/api_manager.py:3`, so on every sim) comes only through `wandb`. `tqdm` (`train.py:19`) comes through `torch_geometric`. `joblib` comes through sklearn. None is in `pyproject.toml`. The sif lists `tqdm` explicitly but not pydantic.
8. **Syntax and torch APIs: clean.** `ast.parse(..., feature_version=(3, 9))` passes on 106 files in `src/`, `scripts/{baselines,artificial_humans,data_creation,policy_finder}`. There is no `match`, no PEP 604 unions, no parenthesised `with` and no `from __future__ import annotations`. PEP 585 generics are 3.9-safe. Torch APIs used: `th.special.ndtr`/`ndtri` (`linear_ah.py:307`, `manager/linear_opponent.py:155`, `punishment_copula_rho.py:62-65`), `searchsorted`, `div(rounding_mode=)`, `log_softmax`, `index_add_`. All of these already ran on the Raven torch-1.11 `.venv` (commit `4badba0c` cites Raven job 30748735 for `24_FRONTIER`).
9. **Local tests, as a weak proxy**: in this image, 472 of 474 frontier-mechanism tests pass. `test_group_vnode.py::test_flip_augmentation_permutes_group_states_output_unchanged` (bit-exact `th.equal` on GRU rows) and `test_punishment_copula.py::test_real_bundle_group_spread_beats_independent` (2.29 vs 2.5 ± 0.125) fail here. Both look stack-dependent and are not verified on Raven.

## 7. Cheap Raven tests (proxy path)

1. `/run_in_container` (login node, CPU, about 1-2 min): `cd /opt/algorithmic-institutions && python -m pytest -q src/aimanager/tests/test_group_vnode_graph.py src/aimanager/tests/test_joint_exodus_graph.py src/aimanager/tests/test_joint_exodus_sampling.py src/aimanager/tests/test_joint_exodus_train_sim_parity.py src/aimanager/tests/test_contribution_copula_graph.py src/aimanager/tests/test_stimulus_skip.py src/aimanager/tests/test_timeout_contribution.py src/aimanager/tests/test_punishment_copula.py src/aimanager/tests/test_linear_opponent.py`. The last two load real `.joblib` bundles, so they exercise risk 2.
2. Sim smoke: copy `24_LEVIN_vnode_skip_timeoutpun_self_gnncopar1_contr_gnn_switch.yml` with `n_episodes: 4` and a new `output_dir` (e.g. `plots/simulation/_smoke_levin`), sync, then submit `-m aimanager simulate <copy>` (`time 00:15:00`). This loads all four real stack artifacts in the sif. Repeat with `24_FRONTIER`.
3. Train smoke (after fixing or working around `job_id`): flatten `switch_predictor/joint_exodus.yml` (`params:` at top level), add `job_id: smoke`, `train_args.epochs: 2`, `n_cross_val: null`, `output_dir: artifacts/artificial_humans/_smoke_joint_exodus`, then submit `-m aimanager train-ah <copy>`. This exercises the joint-exodus loss and `save`. A vnode variant does the same for `group_vnode`.

## Open questions / could not verify

- Whether `/u/ykeller/algorithmic-institutions/artifacts/` on Raven holds the frontier artifacts, and how to get them there if not (sync excludes `artifacts/`; the image has no git-lfs).
- Whether the built `algoinst.sif` really has sklearn 1.5.2 (I read only the recipe), and whether 1.6.1 bundles load in it.
- Whether the command server accepts `program_call` values other than `-m aimanager ...` (needed for the copula and baseline scripts).
- The exact torch version that wrote the `.pt` files. The zip-layout inference and the `.venv/lib/python3.9` log path are the only evidence.
- The content of `switch_exodus_k_onehot/...pt`. It is an LFS pointer on the gmlp branch and possibly still on Raven under `certuer`.
- Whether the two local test failures reproduce on Raven.
- The valid model's training provenance (pilot-data run yml 22) comes from naming only.
