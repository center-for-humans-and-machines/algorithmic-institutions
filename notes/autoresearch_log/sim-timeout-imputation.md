# The contribution and switch models and the timed-out player (a serving-path correctness fix)

## 1. Declaration

**Slot:** shared simulation code -- a bug fix under §4's "a bug fix in shared code is legal but is its own experiment: fix only, before/after scores for the top-ranked stack". **No model is retrained.** No weights, features or training configs change; every slot in every run below loads the same file on disk the parent's runs loaded.

**Parent:** PR #194 (`auto/punisher-timeout-feature`, `[FAIL]`, at `7d86efd`), which carries the punisher half of this same fix plus the accepted ceiling fix beneath it. Branch `auto/sim-timeout-imputation` is created from it and the PR opens with `--base auto/punisher-timeout-feature`. Isolated remote dir `~/repros/ai-runs/sim-timeout-imputation` (plus `~/repros/ai-runs/sim-timeout-imputation-base` for the before half of the noise-off arm; delete both when this PR closes).

**The defect.** When a player times out, the real game charged 0, paid out on 0 and showed everyone 0; the training data stores 0 (`data.py:46`, `fillna(0)`), and the parent's step 0 settled it with the accounting identity (`common_good == 1.6 * sum(c) - sum(p)` holds to 1.4e-14 on all 516 human group-rounds containing a timeout under the recorded 0 and fails on every one of them under the imputed 9). At **simulation** time, though, `environment.update_contribution` (line 332) samples who times out with the configured `valid_model` and then overwrites those players' contributions with `default_values["contribution"]` -- 9 -- before the state is handed on. The env's own common-good and payoff accounting zeroes invalid cells separately and is correct; the defect is purely in what the models are shown. The punisher's two serving paths were corrected on the parent; the contribution and switch models' were not.

**Base models (unchanged, not retrained).** The frontier stack exactly as the parent left it: contributor `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`, switch `artifacts/artificial_humans/switch_joint_exodus/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`, punisher `artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib`, validity model `artifacts/artificial_humans/raven_script_22/model/rnn_False__dataset_full.pt`.

**Evaluation stack (§3 under the parent rule of §9).** The frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout`, re-run from a config that differs from the parent's in `output_dir` and `figure_name` and in nothing else (`_simtimeout`). The reference stack `23_2g8a_self_gnn_contr_gnn_switch` is re-run too and reported, not gated.

**Baseline (the parent's confirmed `_timeout` frontier scores).**

| row | score | band |
|---|---|---|
| mean over 22 rows | **0.9824** | |
| gate-2 ceiling (mean x 1.10) | **1.0806** | |
| rows <= 1 | 15/22 | |
| SA / SB / SC (declared targets) | 0.8744 / 0.9598 / 1.5278 | <= 1 / <= 1 / 1-2 |
| CG (declared target) | 1.2975 | 1-2 |
| RCE (protected) | 0.8719 | <= 1 |
| RCC | 1.0769 | 1-2 |

### Hypothesis

**Behavioural rationale (one sentence, §5):** a player who gave no input contributed nothing, and that is what every other player and the manager saw, so the contribution and switch models must be served the recorded 0 rather than an imputed 9 they were never trained on; the rows that should move are the S family (the switch model's own decision) and CG (the contributor's group-level dispersion), with the contribution family (CA-CF) as watch rows.

**The change.** One coherent serving-path correction, no retraining:

1. `ArtificialHumanEnv.served_state()` -- the state handed to a model reads `MISSING_CONTRIBUTION` (0) on `contribution` and `prev_contribution` wherever the matching validity flag is False. `self.state` itself, the env's dynamics, its common-good and payoff accounting and the recorded simulation output are untouched.
2. `LinearAHAdapter` -- the env-driven path records the realised validity flags and passes them into `build_feature_pool`, exactly as the punisher's rounds-driven path already does.

**Not changed, deliberately: no `contribution_valid` flag for either model.** Reasons in section 4, note 3.

### Artifact naming contract

| what | path |
|---|---|
| probe | `scripts/data_analysis/sim_timeout_serving_probe.py`; output `plots/data_analysis/evaluation/sim_timeout_imputation/probe_{before,after}.json` |
| sim config (gated) | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout.yml` |
| sim config (noise-off, after / before) | `.../23_2g8a_sim_timeout_rho0.yml`, `.../23_2g8a_sim_timeout_rho0_base.yml` |
| sim config (reference) | `.../23_2g8a_self_gnn_contr_gnn_switch_simtimeout.yml` |
| tests | `src/aimanager/tests/test_sim_timeout_serving.py` |
| tables | `plots/data_analysis/evaluation/sim_timeout_imputation/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Confirm the defect on the unchanged tree: which value each model is actually served at a timed-out cell, which state keys each model reads, and the realised timeout rate in simulation. | **done** (probe, unchanged tree) |
| 2 | Fix both serving paths (`environment.served_state`, `linear_ah` validity threading); verify nothing else depends on the substituted value. | **done** |
| 3 | Decide, with a reason, whether either model should also receive a timeout flag. | **done** -- no; note 3 |
| 4 | Unit tests mirroring the punisher's; graph paths on Raven. | **done** -- 21 pass on Raven |
| 5 | Noise-off run + `copula_closed_loop_variance.py`: `Var(E[c \| history])` over visited states against the human 27.9. | **done** -- section 3, step 5 |
| 6 | Re-run the frontier stack, evaluate all 22 rows, table against the PR #195 seed floor. | **done** -- jobs 30325363 / 30325396 |
| 7 | Judge: correctness outcome and gate outcome reported separately; log; PR against the parent. | **done** -- section 5 |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | contribution and switch models are served the recorded 0 for a timed-out player (nothing retrained) | **SA 0.8744 -> 0.7741** (-0.1003, 0.62 seed sd, not legible); **SB 0.9598 -> 1.0695** (+0.1096, 2.40 sd, band `<= 1` -> `1-2`); **SC 1.5278 -> 1.8295** (+0.3017, 2.23 sd); **CG 1.2975 -> 1.8449** (+0.5475, 1.82 sd). No band upgrade. | 15 -> **14/22** (-1, 0.32 seed sd, **not** legible) | 0.9824 -> **1.0393** (gate-2 ceiling 1.0806, **pass**; +1.20 seed sd) | **[FAIL]** on the gates; the correctness outcome is separate and positive -- section 5 |

### Step 1: what each model is actually served (measured, on the unchanged tree)

`scripts/data_analysis/sim_timeout_serving_probe.py` runs the real simulation machinery -- same env, same artifacts, same protocol, 20 episodes instead of 100 -- with every model's `predict` wrapped, and reports the state keys each model's encoder consumes together with the value it was handed at every timed-out cell it reads. Output: `plots/data_analysis/evaluation/sim_timeout_imputation/probe_before.json`.

**How often the path fires.** 3,840 agent-rounds, **86 timeouts = 2.24%**. The human rate is 2.9% (560 / 19,200); the simulated rate is measured here rather than assumed, and it is a little lower.

| model | keys its encoder reads | value at timed-out cells |
|---|---|---|
| contribution (`vnode_stimulus_skip_herding_copula`) | `agent_group`, `prev_contribution`, `prev_punishment` | `prev_contribution` = **9.0 on all 249** invalid lag cells |
| switch (`switch_joint_exodus`) | `agent_group`, `common_good`, `punishment`, `round_number` | served `contribution` = 9.0 on all 86, **but the key is never read** |
| validity (`raven_script_22`) | `prev_contribution_valid` | reads no contribution key at all |
| punisher (`lin_multinomial` + severity copula) | via round dicts | 9.0 in the round dict; both punisher paths put the recorded 0 back downstream (the parent's fix) |

**The defect is confirmed for the contribution model and refuted for this stack's switch model.** Of the 249 invalid lag cells, 168 are round-0 cells (no previous round exists; `prev_contribution_valid` is False there by construction and `prev_contribution` legitimately carries the dataset default, exactly as `create_torch_data`'s `shift()` puts it there in training) and **81 are real timeouts, every one of them served 9.0**. The parent's successor note expected the switch model to read a timed-out player as having contributed 9 that round; that is true of a *linear* switch bundle (block B1 reads `contribution`), but the frontier stack's switch model is the joint-exodus GNN, whose `x_encoding` is `common_good / punishment / agent_group / round_number` and contains no contribution term. It therefore sees the defect only indirectly, through `common_good` -- which the env already computes with invalid contributions zeroed and is correct. The fix still covers that path, because correctness should not depend on which model happens to occupy the slot, but **the measured effect in this stack comes from the contribution model alone.**

The validity model is the other pleasant surprise: it reads **only** `prev_contribution_valid`, so the substitution never reached it and serving it the corrected state is a measured no-op. That is why the timeout process itself is not disturbed by the fix.

### Step 1 (repeat): the same probe after the fix (measured)

`probe_after.json`, same config, same 20 episodes:

| quantity | before | after |
|---|---|---|
| `prev_contribution` at the 80-81 real timed-out lag cells | 9.0 on all of them | **0.0 on all of them** |
| `prev_contribution` at the 168 round-0 cells | 9.0 | **9.0** (unchanged, as training has it) |
| `contribution` served at invalid cells | 0.0 x 168 (round 0) + 9.0 x 81 | **0.0 x 248** |
| env's own recorded contribution at timeouts | 9.0 | **9.0 (unchanged)** |
| realised timeout rate | 86 / 3840 = 2.24% | 83 / 3840 = 2.16% |

The recorded value is unchanged, which is the point of the chosen form (step 2). The timeout rate moves by three cells out of 3,840: the valid model's inputs are untouched, so this is the closed loop visiting different states, not a change to the timeout process.

### Step 2: the change, and what depends on the substituted value

The cleanest form was considered and **rejected**: the environment keeps substituting, and the two serving points are intercepted instead. The reason is a real dependency.

`self.state["contribution"]` is what `Memory.add` records, what `mem_to_df` turns into `per_round.parquet`, and therefore what the evaluation suite scores. And the suite treats the two sources differently: `evaluation_suite/convert.py::load_human` does `df["contribution"].where(df["player_no_input"] == 0)`, so a human timeout becomes **NaN** and every metric drops it (`dropna(subset=["contribution"])`); `load_sim` has no validity column at all, so a simulated timeout is **scored at whatever the env recorded**. Recording 0 instead of 9 would push ~2.2% of the scored sim rows to a hard zero -- about -0.2 on the mean contribution and a matching distortion in every C and R row -- against human rows that are not there at all. The imputed 9 sits next to the dataset median and is close to neutral in that role; 0 is not. Making the sim mark them missing instead would need a validity column in `per_round.parquet` **and** a change to `convert.py`, which is frozen surface (§8).

So the substitution is load-bearing for the recorded output and wrong only for the served input, and the fix separates exactly those two.

**Every consumer of the substituted value, and what happens to it.** Because `self.state` is not mutated, all six are untouched *by construction*; they were checked rather than assumed:

1. `compute_common_good_per_group` -- zeroes invalid contributions itself (`th.where(contribution_valid, ...)`). Unchanged.
2. `compute_payoff_per_group` -- zeroes the invalid contributor's payoff itself. Unchanged.
3. `update_own_grp_prev_mean_contr` -- already excludes invalid cells from both the sum and the count via `prev_contribution_valid`. Unchanged (and it was already correct).
4. `Memory` -> `mem_to_df` -> `per_round.parquet` -> the evaluation suite. Unchanged -- confirmed by the probe, which still reads 9.0 there after the fix.
5. `simulate.make_round` -> the punisher. Unchanged; the punisher's own two paths correct it downstream, and the parent's tests pin that.
6. `rl_manager`'s observations. Unchanged; `served_state()` is opt-in and the RL call sites were not touched.

**Files.**

- `src/aimanager/manager/environment.py` -- new `served_state()`, used by `update_contribution` (for both the contribution and the validity model) and `_run_switch_predictor`. Round 0 is excluded from the previous-round substitution.
- `src/aimanager/simulation/linear_ah.py` -- `_record` stores the realised `contribution_valid` / `prev_contribution_valid`, `_build_pool` passes the mask to `_pool_from_arrays` instead of `cv=None`. **Inert on both stacks run here** (every contribution, validity and switch slot in them is a `.pt` GNN; the only `.joblib` is the punisher, which goes through the unchanged `_pool_from_rounds`), and inert for any bundle that does not select `contribution_valid` -- which is illegal for the contribution target anyway. It is there so a linear switch bundle cannot hit the same defect.

### Step 4: unit tests

`src/aimanager/tests/test_sim_timeout_serving.py`, nine tests, mirroring the punisher's file:

- `test_switch_model_sees_zero_for_a_timeout` / `test_contribution_model_sees_zero_for_the_previous_round_timeout` -- the timed-out agent reaches each model as 0 and the genuine zero next to it is unchanged, with nobody else disturbed;
- `test_round_zero_lag_keeps_the_dataset_default` -- the fix must not touch round 0;
- `test_env_state_and_recorded_output_keep_the_imputed_value` and `test_common_good_is_unchanged_by_the_fix` -- the guards on step 2's dependency argument;
- two linear-adapter tests (contribution lag, switch value + validity flag);
- two GNN `Encoder` tests (the frontier contributor's own `prev_contribution` encoding: 0.0 served against 9/20 in the raw state).

`scripts/remote_test.sh --test-only` on Raven over this file and the parent's: **21 passed**, 0 failed. Locally 7 of the 9 pass and the two `Encoder` tests fail with a bare `ModuleNotFoundError: torch_scatter`, the repo's convention. *(A whole-suite local run makes them pass spuriously: `test_joint_exodus_train_sim_parity.py` installs a `torch_scatter` stub into `sys.modules` that leaks into later modules. The Raven run is the authoritative one.)*

`black --check` and `flake8 --max-line-length=88 --extend-ignore=E203,W503` are clean over `src/`; the two files already unformatted on the parent (`artificial_humans/train.py`, `rl_manager.py`) were left alone.

### Step 5: the state-spread diagnostic with the shared-noise machinery off (measured)

`scripts/data_analysis/copula_closed_loop_variance.py`, ported from PR #188 with only its arms retargeted, teacher-forces the **bare** stimulus-skip trunk over each arm's own realised history and reports `Var(E[c | history])` -- how much of the contribution variance the model explains from the states it actually visits. The arms are a 2x2 of (serving fix off / on) x (copula off / on) around the frontier stack; nothing is retrained in any of them.

| arm | serving fix | copula | `Var(E[c \| hist])` | retention vs human | `Var(c)` | `Var(resid)` | CG ratio |
|---|---|---|---|---|---|---|---|
| human histories | -- | -- | **27.933** | 1.000 | 39.916 | 11.794 | 0.8480 |
| A (parent) | off | on | 24.211 | 0.867 | 36.353 | 11.507 | 0.8177 |
| B (candidate) | **on** | on | **25.275** | 0.905 | 38.789 | 12.333 | 0.8005 |
| D (parent) | off | **OFF** | 18.924 | 0.678 | 29.805 | 11.651 | 0.7837 |
| **C (candidate)** | **on** | **OFF** | **21.360** | **0.765** | 33.800 | 12.604 | 0.7815 |

**The protocol number: with the shared-noise machinery disabled the candidate's `Var(E[c | history])` over visited states is 21.36 against the human 27.93** (retention 0.765), up from **18.92** (0.678) for the same stack without the fix. The fix adds **+2.44** of state spread, and it does not buy it with shared noise, because the copula is off on both sides of that comparison. With the copula on the same move is +1.06 (24.21 -> 25.27). The control that the port measures the same quantity as PR #186 and PR #188: the human pass returns **27.9331** against their quoted 27.93, and `Var(resid)` 11.794 against 11.79.

The CG column explains the score regression and is worth reading next to it. The sim's group-spread ratio was already **below** the human 0.8480, and the fix lowers it further, 0.8177 -> 0.8005: **the mis-served 9s were an accidental source of between-group dispersion** that the closed loop was living on. With the copula off the same comparison is flat (0.7837 -> 0.7815), so the CG cost is specific to the copula-on stack rather than a property of the fix itself. One run each side; both statements are single draws.

Full tables and the per-round figure: `plots/data_analysis/evaluation/sim_timeout_imputation/copula_closed_loop/`.

### Step 6: the 22 rows, frontier stack (measured, gated)

Every "before" number reproduces the parent's frontier scores exactly. The seed floor beside every movement is PR #195's (`auto/seed-spread-noise-floor`): a typical row moves 0.138 on the training draw alone, the 22-row mean 0.047, the rows <= 1 count 3.16, and ten of the 22 rows cannot be gated on a single run (`ungateable` below). **My own run-to-run variation is smaller than that floor** -- nothing was retrained, both runs use the same artifacts under seed 42, so the before/after difference contains no training noise at all. The floor is still the right scale for reading a row, because it is the scale on which the *model* is uncertain; but this comparison is cleaner than a retrained candidate's, and a movement of 1.2 seed sd here is a firmer statement than the same number would be after a retrain.

| row | before | after | delta | band | seed sd | in seed sd | legible | ungateable |
|---|---|---|---|---|---|---|---|---|
| CA | 0.8588 | 0.8422 | -0.0165 | <= 1 | 0.1878 | 0.09 | no | yes |
| CB | 0.7850 | 0.8164 | +0.0314 | <= 1 | 0.2153 | 0.15 | no | yes |
| CC | 0.8387 | 0.8154 | -0.0233 | <= 1 | 0.1333 | 0.17 | no | yes |
| CD | 0.7939 | 0.7665 | -0.0274 | <= 1 | 0.1878 | 0.15 | no | yes |
| CE | 0.9880 | 0.9675 | -0.0205 | <= 1 | 0.0581 | 0.35 | no | no |
| CF | 0.8414 | 0.8152 | -0.0262 | <= 1 | 0.1413 | 0.19 | no | yes |
| **CG** (target) | 1.2975 | **1.8449** | +0.5475 | 1-2 | 0.3014 | **1.82** | **yes** | yes |
| **SA** (target) | 0.8744 | 0.7741 | -0.1003 | <= 1 | 0.1618 | 0.62 | no | yes |
| **SB** (target) | 0.9598 | **1.0695** | +0.1096 | **<= 1 -> 1-2** | 0.0456 | **2.40** | **yes** | yes |
| **SC** (target) | 1.5278 | **1.8295** | +0.3017 | 1-2 | 0.1355 | **2.23** | **yes** | no |
| PA | 0.6375 | 0.6408 | +0.0033 | <= 1 | 0.0404 | 0.08 | no | no |
| PB | 0.8578 | 0.8995 | +0.0417 | <= 1 | 0.0231 | **1.80** | yes | no |
| PC | 0.8727 | 0.9147 | +0.0420 | <= 1 | 0.0359 | **1.17** | yes | no |
| PD | 0.8854 | **0.6860** | -0.1993 | <= 1 | 0.0593 | **3.36** | **yes** | no |
| RCA | 1.6715 | 1.7761 | +0.1046 | 1-2 | 0.1412 | 0.74 | no | no |
| RCB | 1.1797 | 1.0668 | -0.1129 | 1-2 | 0.1425 | 0.79 | no | no |
| RCC | 1.0769 | **1.4983** | +0.4214 | 1-2 | 0.1631 | **2.58** | **yes** | no |
| RCD | 1.1313 | 1.3368 | +0.2055 | 1-2 | 0.2698 | 0.76 | no | no |
| RCE (protected) | 0.8719 | 0.9474 | +0.0755 | <= 1 | 0.1063 | 0.71 | no | yes |
| RSA | 1.2595 | 1.1529 | -0.1066 | 1-2 | 0.1555 | 0.69 | no | yes |
| RPA | 0.6375 | 0.6426 | +0.0051 | <= 1 | 0.0175 | 0.29 | no | no |
| RPB | 0.7653 | 0.7624 | -0.0029 | <= 1 | 0.0283 | 0.10 | no | no |
| **mean** | **0.9824** | **1.0393** | +0.0570 | | 0.0473 | **1.20** | yes | |
| rows <= 1 | 15 | **14** | -1 | | 3.1623 | 0.32 | **no** | |

Fifteen of the 22 rows move by less than one seed sd and are **not distinguishable from a retrain of an unchanged model**: CA, CB, CC, CD, CE, CF, SA, PA, RCA, RCB, RCD, RCE, RSA, RPA, RPB. Seven are legible: CG (+1.82), SB (+2.40), SC (+2.23), PB (+1.80), PC (+1.17), PD (-3.36), RCC (+2.58); so is the mean (+1.20). The rows <= 1 count is not (0.32).

**The protected row RCE, with every band slope, its standard error and its row count:**

| band | human | before | after | Δ in pooled SE | Δ in seed sd |
|---|---|---|---|---|---|
| 0-4 | +0.140 +- 0.018 (n 965) | +0.092 +- 0.015 (n 2002) | **+0.102 +- 0.014 (n 2045)** | 0.50 | 0.55 |
| 5-9 | +0.104 +- 0.024 (n 929) | +0.034 +- 0.014 (n 1919) | **+0.019 +- 0.015 (n 1906)** | 0.72 | 0.67 |
| 10-14 | -0.077 +- 0.035 (n 560) | -0.048 +- 0.021 (n 1394) | **+0.000 +- 0.022 (n 1364)** | **1.57** | **1.83** |
| 15-19 | -0.161 +- 0.079 (n 206) | -0.097 +- 0.055 (n 436) | **-0.085 +- 0.054 (n 474)** | 0.16 | 0.21 |

RCE's own score 0.8719 -> 0.9474 is +0.71 seed sd, **not** legible, and the band `<= 1` holds. But the band-sign pattern goes `++--` -> `+++-`: **the 10-14 slope loses the human's negative sign**, and the amended magnitude clause fires on the same band (`magnitude_eroded: ['10-14']`). That is a **failure of the protected-row rule** and is reported as one. The context, not an excuse: the move is 1.83 seed sd, i.e. above the floor, but PR #195 measured this exact band running from +0.037 to -0.043 across six retrains of an unchanged model (six-arm mean **-0.0043**, sd 0.0263) and lists RCE among the ten rows that cannot be gated on a single run. The after value is +0.0004 -- the slope crossed zero rather than reversing. Whether the sign is "lost" is decided by the draw on any single run of this stack.

### Step 6 (reported, not gated): the reference stack

`23_2g8a_self_gnn_contr_gnn_switch`, 44 rows over two pairings -- a different contributor (the plain 50ep GNN) and a different switch model (the reanchored one) in the two slots this fix serves. It reads the other way.

| pairing | mean before -> after | Δ in seed sd | rows <= 1 | notable |
|---|---|---|---|---|
| `lin_multinomial_self` | 1.6355 -> **1.5831** | **1.11 (a legible improvement)** | 13 -> 14 | CA -0.267 (1.42), CD -0.246 (1.31), CC -0.172 (1.29), SC -0.392 (2.90), PD -0.080 (1.34) all better; **RCE band upgrade `1-2` -> `<= 1`** (1.0408 -> 0.9412) and its protection holds (no sign lost, the halved 10-14 band moves *toward* the human); RCA band downgrade `1-2` -> `2-5` (+0.188, 1.33) |
| `gnn_self` | 1.8229 -> 1.8403 | 0.37 (**not** legible) | 8 -> 6 | SC -0.415 (3.06) and PD -0.403 (6.80) much better, CE -0.134 (2.30) better; the punishment family worse (PA +2.82, PB +2.78, RPB +3.10, RPA +2.02); RCE band `<= 1` -> `1-2` and the 10-14 sign lost, as on the frontier |

**The same fix improves the reference stack's linear-punisher pairing by a legible margin and is a wash on its GNN-punisher pairing, while costing the frontier stack.** That asymmetry is the most informative thing in the run and is discussed in note 4.

## 4. Notes

1. **Step 1 relocated the defect within the stack, and a successor must not re-derive it.** The parent's successor note said the switch model reads a timed-out player as having contributed 9 that round. Measured: the frontier's switch model (`switch_joint_exodus`) does not read `contribution` at all -- its `x_encoding` is `common_good / punishment / agent_group / round_number`. The claim is true of a *linear* switch bundle and false of this one. The whole measured effect on the gated stack therefore comes from the contribution model's lagged input, one cell in 46 per round, spread to the rest of the group through the edge model and the group virtual node.

2. **The env's substitution was kept, not removed, and that was a judgement with evidence behind it.** See step 2. The one-line version: the same value is both a model input (where it is wrong) and the recorded simulation output the frozen evaluation suite scores (where it is the lesser of two evils, because `convert.load_human` drops human timeouts to NaN and `load_sim` cannot). Removing the substitution would have silently changed 2.2% of the evaluated rows to a hard zero and confounded the whole comparison. Making the suite drop them instead is the right long-run answer and needs the maintainer, because `evaluation_suite/` is frozen (§8). This is the single most important thing in this log for a successor.

3. **No timeout flag for either model, and the reason is not timidity.** Three independent ones. (a) It is a *training-side* change: both models would have to be retrained, which forfeits the one thing that makes this experiment unusually clean -- that nothing was retrained and the before/after therefore carries no training noise -- and it is a second change in one experiment (§4). (b) The only evidence anyone has on this flag is the parent's, and it points the wrong way here: the flag helped the *linear* punisher (CV 1.3446 -> 1.3271, locked test 1.2234 -> 1.1934) and hurt the *graph* one (CV -0.0024 at a paired t of -0.46, mechanism slope moving away from the human, `gnn_self` mean worsening by 3.44 seed sd), with the stated reading that a graph architecture reconstructs the timeout from context and an explicit channel displaces capacity. Both slots here are graph models. (c) It would not even be the same feature: the punisher's flag is the *current* round's input status, the contributor would need the *previous* round's, and this stack's switch model reads no contribution key for a flag to disambiguate. **It is a separate, declarable hypothesis and is left to a successor**, who should note that the first thing to try is the flag on a *linear* contributor, where the parent's evidence actually applies.

4. **The fix is correct and it costs the frontier stack score, and both halves of that sentence are real.** The state-spread diagnostic (step 5) says the model's conditional expectation tracks the visited states substantially better with the fix (`Var(E[c|hist])` 18.92 -> 21.36 against the human 27.93, with the shared-noise machinery off on both sides), which is a property of the model rather than of the metric. The 22 rows say the frontier stack's group-level rows get worse, and the CG column says why: the sim's group-spread ratio was already below the human and the mis-served 9s were topping it up. The frontier's contributor carries a herding copula calibrated to supply exactly that missing between-group dispersion, and with the copula off the CG difference between the two arms disappears (0.7837 vs 0.7815). The reasonable reading is that the copula-on stack had absorbed part of the defect into its calibration; the reference stack, whose contributor has no copula, improves instead. **Nobody should re-run the gated stack hoping for a better draw: there is no draw here. The runs are deterministic given the artifacts and the seed.**

5. **RCC is the loudest single casualty and it is not this branch's row.** 1.0769 -> 1.4983 (+2.58 seed sd), undoing most of the parent's -0.2200 move on its declared target. RCC is the contributor's response to punishment at the ceiling; serving the true 0 changes the contributor's lag on 2.2% of cells and the group aggregate on rather more. Whoever next declares on RCC should know that the parent's 1.0769 was measured on a stack that was being fed a wrong lag.

6. **PD is the loudest gain and it is consistent across all three stacks.** Frontier 0.8854 -> 0.6860 (3.36 seed sd), `ref_lin` 2.7419 -> 2.6623 (1.34), `ref_gnn` 3.2712 -> 2.8678 (6.80). PD is the punishment group-spread ratio; the three stacks agree, which is unusual enough to be worth a successor's attention.

7. **Housekeeping.** No copula was recalibrated: the noise-off arms use the bare, unstamped trunk that already exists on disk, exactly as PR #188's arm B did, and the punisher's severity copula carries rho 0.4273 unchanged everywhere. The probe and the mechanism-style scripts were given distinct labels per model so nothing overwrote a row. The remote dirs `~/repros/ai-runs/sim-timeout-imputation` and `~/repros/ai-runs/sim-timeout-imputation-base` can be deleted when this PR closes.

## 5. The verdict: [FAIL] on the gates, with a confirmed defect and a real correctness gain

### Correctness (reported separately, as it must be)

**The defect was real, it was measured on the unchanged tree, and it is fixed.** In simulation the `valid_model` fires on **2.24%** of agent-rounds, and on every one of them the contribution model's lagged input read **9.0** -- a value the game never used and one that never occurs in its training data, where the same cell is 0. After the fix it reads 0.0 on all of them, round 0 keeps the dataset default as training does, and the env's recorded output is untouched. Nine unit tests pin it and 21 pass on Raven. On the state-spread diagnostic with the shared-noise machinery off, the fix moves `Var(E[c | history])` over visited states from **18.92 to 21.36** against the human **27.93** -- the model explains more of what it does from the states it visits, and not by adding noise.

**Nothing was retrained.** Every slot in every run loads the same file on disk as the parent's runs. That makes this before/after cleaner than any retrained candidate's: the seed floor of 0.138 per row is a *training* floor, and this comparison contains none of it.

### Gates: [FAIL]

**Gate 1 -- a band upgrade on a declared target (SA / SB / SC / CG): FAIL.** No target upgrades. SA improves by 0.62 seed sd (not legible); SB, SC and CG all get worse by legible margins, and SB **downgrades** `<= 1` -> `1-2`.

**Gate 2 -- the 22-row mean: PASS.** 0.9824 -> 1.0393 against a ceiling of 1.0806; the move is +1.20 seed sd. `rows <= 1` goes 15 -> 14 (-1 = 0.32 sd of 3.16, **not** legible and never concluded from).

**The protected row RCE: the rule fires.** The score itself moves 0.71 seed sd and holds its band, but the 10-14 band slope loses the human's negative sign (-0.048 -> +0.000, 1.57 pooled SE / 1.83 seed sd) and the amended magnitude clause fires with it. Reported as a failure of the protection, with PR #195's finding beside it that this band's sign is decided by the draw (six-arm mean -0.0043, sd 0.0263, both signs present).

### Verdict tag

**[FAIL]** -- no band upgrade on a declared target, SB downgrades, and the protected row's sign clause fires. Gate 2 passes. The change is nevertheless a **correct** fix to a **confirmed** defect, with an independent state-spread gain and a legible improvement on the reference stack's linear-punisher pairing; the cost is concentrated in the group-spread rows of the one stack whose contributor carries a copula calibrated to supply that spread.

## 6. For a successor

1. **Take the fix, and take it with the frontier stack's copula recalibrated.** The evidence says the fix is right (step 1, step 5) and that the frontier stack was absorbing part of the defect into its herding copula: with the copula off, the fix costs nothing on CG (0.7837 -> 0.7815) and gains 2.44 of state spread; with the copula on, CG falls further below the human and SB, SC and RCC follow. The obvious next experiment is **the contribution copula recalibrated on top of this fix** -- one change, a declared target of CG, and a baseline that already has the serving path right. This branch deliberately did not recalibrate anything, because a copula recalibration and a serving fix in one run would be inseparable.

2. **The evaluation suite still scores a simulated timeout and drops a human one.** `convert.load_human` NaNs the human's timed-out contributions and `load_sim` has no validity column, so ~2.2% of simulated agent-rounds enter every C and R row at the imputed 9 with no human counterpart. This is now the last place in the pipeline where the imputed value is doing work, it is the reason the env's substitution was kept, and it cannot be fixed by an agent: `evaluation_suite/` is frozen surface (§8). **Escalate it to the maintainer.** The fix is a `contribution_valid` column in `per_round.parquet` plus one `where` in `load_sim`.

3. **The switch slot never had this defect in the frontier stack, and a successor should not go looking for it there.** Measured: `switch_joint_exodus` reads `common_good / punishment / agent_group / round_number` and no contribution key. The parent's successor note said otherwise; it was written about the linear switch bundle, which does read `contribution`, and that path is now fixed too. If the S rows are the target, the mechanism is `common_good`, not the contribution channel.

4. **RCC's 1.0769 on the parent was measured with a wrong lag.** It is 1.4983 here. PR #194's successor note put RCC "within one noise floor of its band"; on a correctly-served stack it is not. Anyone continuing that line should re-baseline first.

5. **The flag question is open and should be asked on a linear contributor.** Note 3. The parent's evidence is that the explicit `contribution_valid` channel helps a linear model and hurts a graph one; every slot on this branch is a graph model, so the flag was not tried. A linear contribution or switch bundle is where it would be expected to pay, and `linear_ah` now carries the mask through to `build_feature_pool` so such a bundle can simply select the feature.

6. **PD improved on all three stacks** (3.36 / 1.34 / 6.80 seed sd). It is the punishment group-spread ratio and it is the one row where every context agrees. Worth a look.
