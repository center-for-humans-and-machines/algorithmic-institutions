# The contribution and switch models and the timed-out player (a serving-path correctness fix)

## 1. Declaration

**Slot:** shared simulation code -- a bug fix under §4's "a bug fix in shared code is legal but is its own experiment: fix only, before/after scores for the top-ranked stack". No model is retrained, no model's weights, features or training config change.

**Parent:** PR #194 (`auto/punisher-timeout-feature`, `[FAIL]`, at `7d86efd`), which carries the punisher half of this same fix plus the accepted ceiling fix beneath it. Branch `auto/sim-timeout-imputation` is created from it and the PR opens with `--base auto/punisher-timeout-feature`. Isolated remote dir `~/repros/ai-runs/sim-timeout-imputation` (delete when this PR closes).

**The defect.** When a player times out, the real game charged 0, paid out on 0 and showed everyone 0; the training data stores 0 (`data.py:46`, `fillna(0)`), and `auto/punisher-timeout-feature`'s step 0 settled it with the accounting identity (`common_good == 1.6 * sum(c) - sum(p)` holds to 1.4e-14 on all 516 human group-rounds containing a timeout under the recorded 0, and fails on every one of them under the imputed 9). But at **simulation** time `environment.update_contribution` (line 332) samples who times out with the configured `valid_model` and then overwrites those players' contributions with `default_values["contribution"]` -- 9 -- before the state is handed on. The env's own common-good and payoff accounting zeroes invalid contributions separately and is correct; the defect is purely in what the models are shown. The punisher's two serving paths were corrected on the parent. Two consumers remain: the **switch** model, which would read a timed-out player as having contributed 9 *that* round, and the **contribution** model, which would read 9 *the round before*, through `prev_contribution` and whatever group-level aggregate is built from it.

**Base models (unchanged, not retrained).** The frontier stack exactly as the parent left it: contributor `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`, switch `artifacts/artificial_humans/switch_joint_exodus/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`, punisher `artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib`, validity model `artifacts/artificial_humans/raven_script_22/model/rnn_False__dataset_full.pt`.

**Evaluation stack (§3 under the parent rule of §9).** The frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout`, re-run with the 23-family protocol (seed 42, 100 episodes, 24 rounds) from the parent's config with only `output_dir` and `figure_name` changed (`_simtimeout` suffix). Nothing else differs -- same seed, same episode count, same artifacts.

**Baseline (the parent's confirmed `_timeout` frontier scores).**

| row | score | band |
|---|---|---|
| mean over 22 rows | **0.9824** | |
| gate-2 ceiling (mean x 1.10) | **1.0806** | |
| rows <= 1 | 15/22 | |
| RCC | 1.0769 | 1-2 |
| RCE (protected) | 0.8719 | <= 1 |
| RCB | 1.1797 | 1-2 |
| CG | 1.2975 | 1-2 |
| RSA | 1.2595 | 1-2 |

### Hypothesis

**Behavioural rationale (one sentence, §5):** a player who gave no input contributed nothing, and that is what every other player and the manager saw, so the contribution and switch models must be served the recorded 0 rather than an imputed 9 they were never trained on; the rows that should move are the S family (the switch model's own decision) and CG (the contributor's group-level dispersion), with the contribution family (CA-CF) as watch rows.

**The change.** One coherent serving-path correction, no retraining:

1. `ArtificialHumanEnv.served_state()` -- the state handed to a model reads `MISSING_CONTRIBUTION` (0) on `contribution` and `prev_contribution` wherever the matching validity flag is False. `self.state` itself, the env's dynamics, its common-good and payoff accounting and the recorded simulation output are untouched.
2. `LinearAHAdapter` -- the env-driven path records the realised validity flags and passes them into `build_feature_pool`, exactly as the punisher's rounds-driven path already does.

**Not changed, deliberately: no `contribution_valid` flag for either model.** Reasons in note 3.

### Artifact naming contract

| what | path |
|---|---|
| probe | `scripts/data_analysis/sim_timeout_serving_probe.py`; output `plots/data_analysis/evaluation/sim_timeout_imputation/probe_{before,after}.json` |
| sim config (gated) | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout.yml` |
| sim config (noise-off) | `configs/simulation/manager_testing/23_2g8a_sim_timeout_rho0.yml` |
| sim config (reference) | `configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch_simtimeout.yml` |
| tests | `src/aimanager/tests/test_sim_timeout_serving.py` |
| tables | `plots/data_analysis/evaluation/sim_timeout_imputation/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Confirm the defect on the unchanged tree: which value each model is actually served at a timed-out cell, which state keys each model reads, and the realised timeout rate in simulation. | pending |
| 2 | Fix both serving paths (`environment.served_state`, `linear_ah` validity threading); verify nothing else depends on the substituted value. | pending |
| 3 | Decide, with a reason, whether either model should also receive a timeout flag. | pending |
| 4 | Unit tests mirroring the punisher's; graph paths on Raven. | pending |
| 5 | Noise-off run + `copula_closed_loop_variance.py`: `Var(E[c | history])` over visited states against the human 27.9. | pending |
| 6 | Re-run the frontier stack, evaluate all 22 rows, table against the PR #195 seed floor. | pending |
| 7 | Judge: correctness outcome and gate outcome reported separately; log; PR against the parent. | pending |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| | | | | | |

## 4. Notes

## 5. For a successor
