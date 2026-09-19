# The contributor at the contribution ceiling (RCC)

## 1. Declaration

**Slot:** contribution -- the stimulus-skip GNN trunk the frontier stack runs, one input feature added.

**Parent:** PR #192 (`auto/punisher-ceiling-fix`, `[FAIL]` on its own gate, at `e2306292eb667f60ae2e9fd1b7189975f8b3efa2`). The maintainer has accepted that branch's punisher as the new baseline despite the failed band gate, so it is this experiment's parent under §9: branch and worktree are created from it, the PR opens with `--base auto/punisher-ceiling-fix`, and the parent's confirmed scores are the baseline both gates are judged against. Isolated remote dir `~/repros/ai-runs/contributor-ceiling` (delete when this PR closes).

**Base model.** Contributor: `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` (node+edge+rnn, `group_vnode: True`, `stimulus_skip: True`, hidden 20, lr 3e-4, 575 epochs, batch 4, 5-fold CV, seed 38381, `x_encoding = prev_contribution (numeric), prev_punishment (numeric), agent_group (onehot)`), and its copula-stamped copy `..._stimulus_skip_herding_copula` (rho 0.03949863621805423, phi 1.0, `copula_switch_every` 1) which is the artifact the frontier stack actually runs. Punisher and switch slots untouched, held at the parent's artifacts.

**Evaluation stack (§3 under the parent rule of §9).** The parent's frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling` -- this contributor x joint-exodus GNN switch x the parent's ceiling-indicator copula-stamped `lin_multinomial`. Re-run with the 23-family protocol (seed 42, 100 episodes, 24 rounds) from the parent's config with only the contributor path, `output_dir` and `figure_name` changed.

**Baseline (the parent's stage-7 frontier scores at full precision, `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling/evaluation/scores.csv`; both gates are judged against these).**

| row | score | band |
|---|---|---|
| **RCC** (declared target) | **1.2969** | 1-2 |
| RCE (protected; bands 0-4 / 5-9 / 10-14 / 15-19 slopes +0.087 / +0.038 / -0.043 / -0.130, signs ++--) | 0.8823 | <= 1 |
| RCB | 1.6591 | 1-2 |
| RCA | 1.6526 | 1-2 |
| mean over 22 rows | **1.0331** | |
| gate-2 ceiling (mean x 1.10) | **1.1364** | |
| rows <= 1 | 14/22 | |

**Target row:** RCC (gate 1: a band improvement, 1-2 -> <= 1, i.e. RCC < 1.0). Watch rows: RCB (the non-ceiling half of the same response, which the parent left worse at 1.6591), RCA, CG and CE (contributor-slot rows that move whenever the trunk changes), and RCE (protected).

### Hypothesis

**The defect (established by the parent, not re-derived).** The parent fixed the punisher's half of RCC completely: the fabricated population of punished full contributors is gone (punished share at the ceiling 12.4% -> 4.0% against the human 3.9%) and the dose is right (E[p | p>0] at 20 8.02 against the human 7.00). What is left is the other half of the contrast, and it is the contributor's:

| | contrast | dc, punished | n punished | dc, unpunished | n unpunished | punished share |
|---|---|---|---|---|---|---|
| human | **-7.035** | **-8.659** | 44 | -1.624 | 1096 | 3.9% |
| frontier before the parent | -0.973 | -2.704 | 277 | -1.731 | 1964 | 12.4% |
| frontier after the parent (this branch's baseline) | **-1.978** | **-3.747** | 91 | -1.770 | 2178 | 4.0% |

A simulated player who gave the maximum and was then punished cuts back 3.75 the next round where a real person cuts back 8.66 -- an under-reaction of about 2.3x. RCC is the only row in the suite that measures the ceiling dose-response: RCE's population is the punished *non-full* contributors by construction, so this defect has nowhere else to show up. The parent's log declares it a contributor-slot defect and leaves the decomposition above as the baseline for exactly this experiment.

**The diagnosis.** The contributor reads its own previous contribution as one number scaled to the unit interval (`prev_contribution`, `encoding: numeric`, `map = linspace(0, 1, 21)`). Detecting the exact value 20 with a smooth encoder is hard -- which is precisely why giving the punisher an explicit maximum indicator moved its ceiling behaviour onto the human number (P(p>0 | c_t = 20) 0.122 -> 0.040 against the human 0.038). The same limitation plausibly explains why the contributor under-reacts specifically at the ceiling.

**Behavioural rationale (one sentence, §5):** a real player treats "I gave everything and was punished anyway" as its own situation and withdraws hard, not as the top of a smooth scale, so the contributor gets an explicit indicator that it contributed the maximum last round; the row that should move is RCC, with RCB and RCA as watch items.

**The change.** One derived feature, `prev_contribution_max = I(c_{t-1} = 20)`, added to the contributor's `x_encoding` as `etype: bool`. It is the lagged counterpart of the `contribution_max` the parent added on the punisher side, and it does not yet exist: `create_torch_data_new` derives `contribution_max` *after* the `prev_` shift comprehension and never lags it, so the lag has to be added alongside it. Everything else -- data, split, seed, architecture, hidden size, lr, epochs, batch size, `shuffle_features`, copula rho and phi -- is identical to the parent's artifact.

### Legality (§5), verified rather than assumed

- **Observable at decision time.** The feature is a deterministic function of `prev_contribution`, which the contributor already reads; a real player plainly knows what they themselves gave last round. Legal under §5's first bullet.
- **No target leakage.** For the contribution target, `apply_mask_pattern` -> `mask_data` masks `contribution` (round t) for the agents being predicted. `prev_contribution_max` is derived from `prev_contribution` (round t-1), which is never masked -- exactly like `prev_contribution` itself. Asserted in `test_contributor_ceiling_indicator.py`. The current-round `contribution_max` stays out of the contributor's encoding; it remains the punisher's feature only.
- **Not engineered at a metric's definition.** 20 is the per-round endowment (`reports/basics.md`), the physical maximum a player can give, not a bin edge the evaluation suite invented. The causation runs the other way: RCC exists as its own row *because* RCB's rate `p / (20 - c)` is undefined at the behavioural category "gave everything". The maintainer already accepted the unlagged form as a legal feature on this reasoning (`notes/baseline_feature_defs.md`).
- **Iteration budget.** One extra input dimension on an unchanged architecture; the parent's contributor trains in ~8-11 min on one A100 and this does not move that. Well inside the 3x rule.

### Indicator alone, not the punished-at-the-ceiling interaction

The decomposition says the damage is concentrated in `dc | punished, c_{t-1} = 20`, so the interaction `I(c_{t-1} = 20) x p_{t-1}` is the term that literally names the defect. It is deliberately **not** what is added, for three reasons, and only one change is made:

1. **The trunk can already form the product once the indicator exists.** `op1`/`op2` are MLPs over the concatenated node encoding, so `I(c_{t-1} = 20) x p_{t-1}` is inside the hypothesis class as soon as the indicator is an input. What the network provably could not do was detect the exact level 20 through a `linspace(0, 1, 21)` scalar -- that is a representation gap, the product is not. This is the parent's own finding transposed: its GNN punisher, whose MLP head could already approximate a bend, still gained from the bare bool and needed no product term.
2. **One change per experiment (§4), ties to the simpler model (§5).** Shipping indicator + interaction as one diff makes the result uninterpretable: a move could not be attributed.
3. **The product sits closer to the §5 line.** `punished x at-the-ceiling` is RCC's own cell written as a feature; "I gave everything last round" is a behavioural category. If the indicator alone does not install the response, the interaction is the obvious successor experiment -- declared separately, with this branch's numbers as its baseline.

### Artifact naming contract

| what | path |
|---|---|
| training config | `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_vnode_stimulus_skip_ceilind.yml` |
| bare artifact | `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_ceilind/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` |
| copula-stamped artifact (the one the stack runs) | `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_ceilind_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`, rho and phi carried frozen from the parent's calibration |
| frontier sim | `configs/simulation/manager_testing/23_2g8a_contr_ceilind_self_gnncopar1_contr_gnn_switch_ceiling.yml`; dir `plots/simulation/<same>` |
| copula-off sims (state-spread diagnostic) | `23_2g8a_contr_skip_copoff_self_gnncopar1_contr_gnn_switch_ceiling` (baseline trunk) and `23_2g8a_contr_ceilind_copoff_self_gnncopar1_contr_gnn_switch_ceiling` (candidate trunk) |
| screens and tables | `plots/data_analysis/evaluation/contributor_ceiling_indicator/` |

## 2. Plan

| # | step | implementer | status |
|---|---|---|---|
| 1 | Add the lagged indicator to the training data path (`generic/data.py`, one line beside `contribution_max`) and to the simulation state (`manager/environment.py`: `contribution_max` maintained in the state so `step()`'s existing `prev_` shift produces `prev_contribution_max`, and both seeded in `reset_state`). Unit test `src/aimanager/tests/test_contributor_ceiling_indicator.py`: the lag is correct, round 0 reads False, the sim path matches the training path round by round, and masking the contribution target leaves the feature untouched. | Opus | done |
| 2 | Training config `..._stimulus_skip_ceilind.yml`: the parent's config with `- etype: bool, name: prev_contribution_max` after `prev_contribution`, and the output dir renamed. Nothing else moves. | Opus | done |
| 3 | Train on Raven in the isolated dir; fetch the artifact; report 5-fold CV log loss against the parent contributor's. | Opus | done |
| 4 | Teacher-forced ceiling screen (`scripts/data_analysis/contributor_ceiling_tf.py`, built on `rcb_teacher_forced.py`'s loaders and stimulus frame): the punished-minus-unpunished contrast among full contributors on the 50 human games, candidate against baseline against the observed human row. Gate before spending a simulation. | Opus | done |
| 5 | Stamp the frozen copula (rho and phi carried unchanged, never recalibrated) onto the candidate with `make_contribution_copula_artifact.py`. | Opus | done |
| 6 | Simulate the frontier stack with the candidate contributor; fetch; evaluate all 22 rows with `PYTHONPATH=<worktree>/src`. | Opus | done |
| 7 | Reproduce the ceiling decomposition table line for line against the parent's, and the RCE band slopes with standard errors and row counts (`scripts/data_analysis/contributor_ceiling_table.py`). | Opus | done |
| 8 | Copula-off runs of both trunks; state-spread diagnostic `Var(E[c \| history])` over visited states (`scripts/data_analysis/contributor_state_spread.py`), calibrated against PR #186's human 27.93. | Opus | done |
| 9 | Judge under the gates with RCE protected and the amended magnitude clause; log; PR against the parent. | Opus | done |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | (baseline) the parent's frontier stack, `_ceiling` | RCC 1.2969, RCB 1.6591, RCA 1.6526, RCE 0.8823 | 14/22 | 1.0331 | baseline |

## 4. Notes

1. **The decomposition tooling was validated against the parent before anything was built.** Recomputing the parent's ceiling table from `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling/per_round.parquet` and the human file reproduces its published row exactly -- human contrast -7.035 (dc punished -8.659, n 44; unpunished -1.624, n 1096; share 3.86%), frontier after -1.978 (-3.747, n 91; -1.770, n 2178; share 4.01%). Every candidate number below comes out of the same function.
