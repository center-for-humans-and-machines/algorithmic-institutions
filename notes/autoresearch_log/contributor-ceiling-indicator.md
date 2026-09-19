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
| 2026-09-19 | **the lagged ceiling indicator on the stimulus-skip contributor**, same stack (`_ceilind`) | RCC **1.0857**, RCB 1.4736, RCA 1.5487, RCE **1.1058** | 9/22 | 1.0983 | **FAIL** (gate 1: RCC stays in band 1-2; protected row RCE drops a band) |

### Step 3: the contributor retrained (measured)

`configs/training/artificial_humans/contribution/group_switching_contribution_50ep_vnode_stimulus_skip_ceilind.yml` on Raven, job **30318205**, 10 min 49 s on one A100 for the 5-fold CV plus the full fit (the parent's trunk takes the same; the added bool costs nothing, and the 3x iteration budget is not in play). wandb `ccj4eaft`.

| | best-epoch CV log loss, mean +- sd over 5 folds | globally best epoch | log loss at the shipped epoch 575 |
|---|---|---|---|
| baseline `..._stimulus_skip` | 2.0028 +- 0.0646 | 450 | 2.0206 |
| candidate `..._stimulus_skip_ceilind` | **1.9855 +- 0.0687** | 350 | **2.0227** |

Read this honestly, both ways. At each fold's own best epoch the candidate is better by 0.0173 -- about a quarter of the between-fold standard deviation, so not a result on its own -- and it gets there 100 epochs earlier. At **epoch 575, which is the epoch the shipped artifact is saved at**, the candidate is 0.0021 *worse*. The feature is not bought with fit; the CV is a tie, and a tie is exactly what a 21-class log loss should show for a bool that only changes behaviour on the 12.7% of rows where the previous contribution was 20. As on the punisher side, it is the mechanism table and not the CV that carries the claim. Per-fold best epochs [550, 350, 350, 250, 300] against the baseline's [550, 450, 400, 350, 350]; per-fold best losses [1.9836, 2.0677, 1.9872, 2.0107, 1.8784] against [2.0304, 2.0391, 1.9823, 2.0624, 1.8996] -- the candidate wins 4 folds of 5.

Artifact `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_ceilind/` (model, metrics, confusion matrix; LFS). Its loaded `x_encoding` is `['prev_contribution', 'prev_contribution_max', 'prev_punishment', 'agent_group']` and no current-round feature, verified by the screen's own alignment check below.

### Step 4: the teacher-forced ceiling screen (measured), before any simulation

`scripts/data_analysis/contributor_ceiling_tf.py` on Raven's login node, 50.8 s, replaying the 50 single-copy human games: each trunk sees the human history and never its own draws, under the human default values, through `rcb_teacher_forced.py`'s own frame and alignment assertion. The population is RCC's -- full contributors (`c_t = 20`) with a valid punishment and a valid next contribution, 1,140 rows, 44 of them punished. Table: `plots/data_analysis/evaluation/contributor_ceiling_indicator/teacher_forced_ceiling.csv`.

| | contrast | dc, punished | n punished | dc, unpunished | n unpunished | OLS slope of dc on p, punished rows |
|---|---|---|---|---|---|---|
| observed (C), the same rows | **-7.035** | **-8.659** | 44 | -1.624 | 1096 | -0.697 |
| baseline trunk | -3.902 | -5.565 | 44 | -1.663 | 1096 | -0.289 |
| candidate trunk | **-4.172** | **-6.010** | 44 | -1.838 | 1096 | **-0.343** |

The `(C)` row reproduces the canonical human RCC to the last digit (-7.035003317850032, gap 0.0), which is the proof that this is the evaluation suite's own population rebuilt from the same frame.

**The mechanism installs, in the right place, and it is small.** The whole move is on the punished rows: -5.565 -> -6.010, which is 14% of the 3.09 that separated the baseline from the human -8.659. The dose-response slope at the ceiling goes -0.289 -> -0.343 against the human -0.697, from 41% to 49% of the human magnitude. The unpunished rows move by 0.175 in the same direction, which is why the contrast gains only 0.270 of the 3.13 gap, 9%.

**The prediction recorded before spending the simulation.** Teacher-forced, the baseline sits at -3.902 and its own closed loop at -1.978: closed-loop drift keeps about 51% of the teacher-forced contrast. At the same retention the candidate would land near -2.13, which against the human -7.035 would leave RCC roughly where it is. **The screen therefore predicted a small improvement and no band upgrade.** The simulation was spent anyway, because the screen measures the conditional and the gate is on the closed loop, and because the one thing the screen cannot see is whether the sharper ceiling detector also changes which states the loop visits. It did -- by more than the screen implied, in both the intended direction and several unintended ones (step 6).

### Step 5: the copula, carried rather than refit (measured)

`scripts/artificial_humans/carry_contribution_copula_params.py` (new on this branch, the contributor-side counterpart of `punishment_copula_rho.py --stamp-rho`) writes `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_ceilind_herding_copula/calibration/copula_params.json`, carrying **rho = 0.03949863621805423 and phi = 1.0 unchanged** from the parent contributor's calibration with `copula_switch_every = 1`, and repointing only `source_model` / `source_model_sha256` at the retrained base. Nothing is estimated: the protocol freezes a copula's parameters per model family when only the marginal is retrained, and **no copula was recalibrated on this branch**. The file carries a `carried_note` saying in so many words that every other key in it -- `rho_ci`, `rho_se`, the bootstrap counts, the preflight, the round-trip bias -- describes the *source* calibration and is provenance, not a measurement on this model.

`make_contribution_copula_artifact.py` then stamps it: 14 tensors compared bit-identically against the base, modules `op1 / op2 / rnn_n / group_vnode_module` unchanged, and the honesty check passes -- **7,457 teacher-forced train-split probability rows bit-identical to the base model's**, so the copula touches sampling and nothing else. Base sha256 `d8327fd3...fae94`, stamped sha256 `6b23e0ae...a8341`.

### Step 6: the closed loop and the 22 rows (measured)

Three simulations from the isolated remote dir `~/repros/ai-runs/contributor-ceiling`, 23-family protocol unchanged (seed 42, 100 episodes, 24 rounds, `save_per_round: true`), one A100 each, 1 min 50 s each: Raven **30318409** (the gated frontier run), **30318410** and **30318411** (the two copula-off arms for step 8). Evaluated locally with the merged 22-row suite (500 repeats, seed 42) under `PYTHONPATH=<worktree>/src`, so the protected row is scored and the suite returns 22 rows, not 21. Tables built by `scripts/data_analysis/contributor_ceiling_table.py` into `plots/data_analysis/evaluation/contributor_ceiling_indicator/`.

That the simulations ran at all is itself the end-to-end check that the feature reaches the model in the closed loop: `BoolEncoder.forward` indexes `state["prev_contribution_max"]` directly and would raise `KeyError`, not silently substitute a default, if the environment had not produced it. Its *value* is pinned separately by `test_simulation_state_matches_the_training_lag`, which walks a scripted three-round episode and asserts the tensor the contributor is shown equals the training tensor round by round.

**The 22 rows, frontier stack (the gated one).**

| row | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.8563 | 1.2053 | +0.3491 | <= 1 -> 1-2 |
| CB | 0.7941 | 1.1477 | +0.3536 | <= 1 -> 1-2 |
| CC | 0.8958 | 1.0751 | +0.1793 | <= 1 -> 1-2 |
| CD | 0.7949 | 1.1736 | +0.3786 | <= 1 -> 1-2 |
| CE | 1.0579 | 1.1177 | +0.0598 | 1-2 |
| CF | 0.8169 | 0.9908 | +0.1739 | <= 1 |
| CG | 1.7588 | 0.9123 | -0.8465 | 1-2 -> <= 1 |
| SA | 0.7687 | 0.9932 | +0.2244 | <= 1 |
| SB | 1.0105 | 1.0993 | +0.0887 | 1-2 |
| SC | 1.4632 | 1.3997 | -0.0635 | 1-2 |
| PA | 0.6526 | 0.8011 | +0.1486 | <= 1 |
| PB | 0.9558 | 0.9743 | +0.0185 | <= 1 |
| PC | 0.9349 | 0.9105 | -0.0243 | <= 1 |
| PD | 0.7598 | 1.0933 | +0.3335 | <= 1 -> 1-2 |
| RCA | 1.6526 | 1.5487 | -0.1039 | 1-2 |
| RCB | 1.6591 | 1.4736 | -0.1854 | 1-2 |
| **RCC** (target) | **1.2969** | **1.0857** | **-0.2112** | 1-2 (no upgrade) |
| RCD | 1.2515 | 1.6050 | +0.3534 | 1-2 |
| **RCE** (protected) | **0.8823** | **1.1058** | **+0.2235** | **<= 1 -> 1-2** |
| RSA | 0.9653 | 0.9687 | +0.0034 | <= 1 |
| RPA | 0.6620 | 0.6657 | +0.0037 | <= 1 |
| RPB | 0.8380 | 0.8157 | -0.0223 | <= 1 |
| **mean** | **1.0331** | **1.0983** | +0.0652 | |
| rows <= 1 | 14 | 9 | -5 | |

### Step 7: the ceiling decomposition, line for line against the parent's (measured)

Same function that reproduced the parent's published table exactly (note 1).

| | contrast | dc, punished | n punished | dc, unpunished | n unpunished | punished share |
|---|---|---|---|---|---|---|
| human | **-7.035** | **-8.659** | 44 | -1.624 | 1096 | 3.86% |
| frontier before (the parent) | -1.978 | -3.747 | 91 | -1.770 | 2178 | 4.01% |
| frontier after (this branch) | **-2.930** | **-4.707** | 99 | -1.777 | 2124 | 4.45% |

**The hypothesis' own mechanism is confirmed, and by more than the screen predicted.** The whole move is where it was aimed: the punished full contributor's next-round change goes -3.747 -> **-4.707**, closing 19.5% of the 4.91 that separated the baseline from the human -8.659, while the unpunished mean moves by 0.008. The contrast gains 0.952 of the 5.06 gap, 18.8% -- more than double the 9% the teacher-forced screen showed and well past the -2.13 the 51%-retention extrapolation predicted, so the closed loop *amplified* the feature rather than washing it out. The punisher's half stays where the parent left it (punished share 4.45% against the human 3.86%, still the right order of magnitude, drifting up a little because the contributors now sit higher).

RCC's own arithmetic, for a reader who wants to see how close it came: the scored numerator falls 5.7632 -> **4.8247** against a noise-ceiling denominator of 4.4439. The row needed 4.4439 and reached 4.8247 -- **8.6% short of the band**.

### Step 8: the state-spread diagnostic, noise machinery off (measured)

`scripts/data_analysis/contributor_state_spread.py`, both trunks bare (`copula_rho = 0`), each teacher-forced over the 50 single-copy human games and over **its own** copula-off simulation (`23_2g8a_contr_skip_copoff...` and `23_2g8a_contr_ceilind_copoff...`, Raven 30318410 / 30318411). `Var(c) = Var(E[c | history]) + Var(residual)`; `retention` is each arm's closed-loop `Var(E)` over its **own** human-history value, per PR #191's rule that a worse fit must not be charged as a contraction.

| arm | states | Var(c) | **Var(E[c \| hist])** | Var(resid) | mean predictive SD | CG ratio | retention |
|---|---|---|---|---|---|---|---|
| baseline | human histories | 39.916 | **27.933** | 11.794 | 3.094 | 0.848 | 1.000 |
| baseline | closed loop | 30.025 | **19.342** | 11.446 | 3.105 | 0.784 | 0.692 |
| candidate | human histories | 39.916 | **27.544** | 11.791 | 3.096 | 0.848 | 1.000 |
| candidate | closed loop | 28.992 | **18.829** | 10.538 | 2.966 | 0.768 | 0.684 |

**Calibration: the baseline trunk on human histories returns `Var(E[c | hist])` = 27.9331 against PR #186's 27.93** -- this tree measures the same quantity that branch did, so the numbers are comparable to its.

**The answer is the boring one the protocol expected, and it is worth having.** Against the human 27.9 and the quoted current 18.9, the candidate's closed-loop `Var(E)` is **18.83** where this branch's own baseline arm measures **19.34** on the same punisher stack (PR #186's 18.88 was measured on the `_curpun` punisher, which is why it sits between the two). The move is -0.51, 2.6%, and retention goes 0.692 -> 0.684. **No surprise here**: a single bool at one level of one lagged feature neither restores state spread nor destroys it, and the diagnostic correctly declines to explain step 6's damage. Two smaller readings for the record: the candidate explains slightly *less* of the human variance on human histories (27.54 against 27.93), consistent with its marginally worse log loss at the shipped epoch; and its closed-loop residual variance drops more than its state variance (11.45 -> 10.54), so the loop it runs is a little quieter overall, which is the direction the CG row moved.

### Step 9: verdict -- `[FAIL]`

| gate | criterion | baseline | result | outcome |
|---|---|---|---|---|
| 1 (declared target) | RCC improves a band, 1-2 -> <= 1, i.e. RCC < 1.0 | 1.2969 | **1.0857**, still band 1-2 | **FAIL** |
| 2 | 22-row mean at or under 1.1364 | 1.0331 | 1.0983 | pass |
| protected | RCE: no band drop, no lost human sign, no eroded slope magnitude | 0.8823 (<= 1), slopes +0.087 / +0.038 / -0.043 / -0.130, signs `++--` | **1.1058 (1-2)**, +0.048 / +0.014 / -0.057 / -0.037, signs `++--` | **FAIL (band drop)** |

**RCE band slopes, each with its standard error and row count**, and the amended magnitude clause. `change_in_se` is the before-to-after change over the pooled standard error of the two slopes; `closer_to_human` says whether the after slope sits nearer the human value than the before slope did. The amended clause fires only when a slope's magnitude at most halves **and** it did not move closer to the human value **and** the change exceeds one pooled standard error.

| band | human | before | after | change_in_se | closer to human | halved | **eroded (amended)** |
|---|---|---|---|---|---|---|---|
| 0-4 | +0.140 +- 0.018 (n 965) | +0.087 +- 0.014 (n 1918) | +0.048 +- 0.014 (n 1511) | **1.96** | no | no | no |
| 5-9 | +0.104 +- 0.024 (n 929) | +0.038 +- 0.013 (n 2098) | +0.014 +- 0.016 (n 1733) | 1.17 | no | yes | **yes** |
| 10-14 | -0.077 +- 0.035 (n 560) | -0.043 +- 0.023 (n 1307) | -0.057 +- 0.021 (n 1400) | 0.46 | yes | no | no |
| 15-19 | -0.161 +- 0.079 (n 206) | -0.130 +- 0.067 (n 448) | -0.037 +- 0.033 (n 716) | 1.24 | no | yes | **yes** |

Protected-row checks, verbatim from the script: `band_downgrade=True, sign_lost=[], magnitude_halved_raw=['5-9', '15-19'], magnitude_eroded=['5-9', '15-19'], signs ++-- -> ++--`.

**The branch fails, on both counts, and the second one matters more than the first.** Gate 1 needed RCC under 1.0 and it reached 1.0857 -- the second-largest move that row has ever had on this stack, and 8.6% of noise-ceiling short. Gate 2 passes. But **RCE, the protected row, drops a band**, 0.8823 -> 1.1058, and this is not the rule misfiring: the band drop is the hard clause, not the amended magnitude one, and even the amended clause fires on two of four bands independently. The 0-4 band, which carries the most rows and the largest human slope, loses 45% of its slope at **1.96 pooled standard errors** -- the one band movement on this branch that is genuinely outside one seed's sampling error, and it fails the erosion clause only on the halving threshold, not on significance. Three of four bands moved away from the human value. No arithmetic rescues either gate.

## 4. Notes

1. **The decomposition tooling was validated against the parent before anything was built.** Recomputing the parent's ceiling table from `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling/per_round.parquet` and the human file reproduces its published row exactly -- human contrast -7.035 (dc punished -8.659, n 44; unpunished -1.624, n 1096; share 3.86%), frontier after -1.978 (-3.747, n 91; -1.770, n 2178; share 4.01%). Every candidate number below comes out of the same function.

2. **The indicator alone was chosen over the punished-at-the-ceiling interaction, and the result does not overturn that choice.** The reasoning is in the declaration: the trunk's MLP head can form the product once the indicator exists, the protocol is one change per experiment, and `p_{t-1} x I(c_{t-1} = 20)` is RCC's own cell written as a feature. The measurement is consistent with it -- the ceiling response *did* strengthen from the bare bool, teacher-forced and closed-loop, which is what "the missing ingredient was the detector, not the product" predicts. A successor who wants the interaction now has a reason to try it, but should read note 4 first: the problem with this branch is not that the ceiling response moved too little, it is what moved with it.

3. **What actually broke: the trunk did not add a ceiling response, it reallocated a contribution response.** Five rows left band `<= 1` and the contributor's whole level shifted. Over the frontier sims' 19,200 agent-rounds against the human 8,914:

   | | mean c | sd c | share c = 20 | share c = 0 | mean p | share p > 0 |
   |---|---|---|---|---|---|---|
   | human | 9.457 | 6.318 | 13.4% | 9.4% | 1.791 | 30.6% |
   | before | 9.324 | 6.068 | 12.5% | 8.2% | 1.801 | 31.3% |
   | after | **10.355** | 5.893 | 12.3% | **5.4%** | 1.624 | 29.2% |

   The candidate contributes a full point more on average than the human and nearly two-fifths fewer of its rounds at zero, where the baseline sat within 0.14 and 1.2 points of the human on both. That is CA, CB, CC, CD and PD moving out of band in one line: the level rows were at the noise ceiling and are now a fifth above it. CG improving by 0.85 into band `<= 1` is the same event read from the other end and is the known trap -- §6 records CG as anti-correlated with the individual-fit rows at r ~ -0.7 to -0.9, and buying group spread with worse individual behaviour is named there as the failure mode. **This branch is a clean instance of it.** RCE's erosion belongs to the same story: the punished non-full contributor's dose-response flattens in three of four bands while the ceiling response sharpens, which is what a fixed-capacity model does when a new categorical input takes over part of the punishment channel.

4. **How much of that is the feature and how much is one retrain, honestly.** Both, and this branch cannot separate them. Against the feature being the whole story: the candidate's globally best CV epoch moves 450 -> 350 and at the shipped epoch 575 its log loss is 0.0021 *worse* than the baseline's, so the artifact that was simulated is a slightly further-trained model than the baseline is at the same epoch count, and a 1.0-point level shift is larger than a bool on 12.7% of rows obviously buys. For the feature: the shift is not random noise in the level -- the collapse is concentrated at `c = 0` (8.2% -> 5.4%) while `c = 20` barely moves, which is a shape change, not a scale change, and the state-spread diagnostic shows the conditional is essentially unchanged on human histories (27.93 -> 27.54), so what moved is the closed loop, not the fit. **The protocol runs one training per candidate and the verdict is judged on it; that is what is reported.** A successor who wants the attribution should run the same config at two more seeds before concluding the feature causes the level shift, and should not assume this branch settled it.

5. **The screen was directionally right and quantitatively too pessimistic, which is useful to know about the screen.** It predicted a closed-loop contrast near -2.13 from 51% retention of the teacher-forced -4.172; the measured value is **-2.930**, so closed-loop retention rose to 70% for the candidate against 51% for the baseline. A feature that sharpens a categorical boundary is amplified by a loop that keeps revisiting that boundary. The screen is still worth its minute -- it costs 1/3 of a simulation and it correctly said "improvement, no band" -- but its extrapolation should be read as a floor on the closed-loop effect, not an estimate of it.

6. **The state-spread diagnostic was run as the protocol requires and found nothing, which is the informative outcome.** 19.34 -> 18.83 against the human 27.9, retention 0.692 -> 0.684. The damage in note 3 is therefore not a collapse of state spread; the loop visits a *differently centred* set of states, not a narrower one. The calibration against PR #186 passes exactly (27.9331 vs 27.93), so this is a real null and not a broken measurement.

7. **No copula was recalibrated.** rho 0.03949863621805423 and phi 1.0 were carried onto the retrained marginal unchanged, the stamped artifact is bit-identical to its base outside the three copula fields, and its teacher-forced probabilities are bit-identical on 7,457 rows. The carried params JSON says in the file which of its keys are measurements and which are provenance, because a successor reading `rho_ci` out of it would otherwise be reading the parent's interval as if it were this model's.

8. **Housekeeping.** The isolated remote dir `~/repros/ai-runs/contributor-ceiling` can be deleted when this PR closes. The test suite in an isolated dir needs `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet` shipped by hand (`rsync` excludes `plots/`), or nine evaluation-suite tests fail on the missing fixture; with it shipped, 133 of 134 passed and the one failure was this branch's own test asserting the masked target under the wrong name, fixed in `ab43aea`. Black and flake8 clean under `src/`.

## 5. For a successor

1. **Do not re-run the bare lagged indicator on this trunk.** It is measured: RCC 1.2969 -> 1.0857, ceiling contrast -1.978 -> -2.930, and it costs the protected row a band and four level rows their `<= 1`. The mechanism half of the hypothesis is *confirmed* -- the contributor really was unable to detect the exact level 20 through a `linspace(0, 1, 21)` scalar, and giving it the detector really does strengthen the ceiling response in both the conditional and the closed loop. What is refuted is that installing the detector is free.

2. **RCC is now 8.6% of one noise ceiling away from its band, and the remaining distance is still the same defect.** `dc | punished, c_{t-1} = 20` is -4.71 against the human -8.66 on a population of the right size. A successor targeting RCC should assume the row is reachable and that the binding constraint is collateral, not the target.

3. **The obvious next variant is the interaction, and it should be declared with the level rows as watch items, not just RCE.** `p_{t-1} x I(c_{t-1} = 20)` concentrates the new capacity on the punished ceiling rows instead of letting a free-standing category reshape the whole contribution distribution, which is precisely this branch's failure mode. It is a single change and legal on the same reading as the indicator (see the declaration's legality section), but it sails closer to §5's metric-definition line and its PR must argue that explicitly.

4. **A cheaper variant worth one experiment before the interaction: keep the indicator and freeze what broke.** The level collapse is concentrated at `c = 0`, a region the indicator says nothing about. An encoding change that adds the detector *without* giving the head a new free direction -- for example encoding `prev_contribution` as ordinal rather than numeric, which makes every level including 20 separately addressable -- tests the same representation hypothesis with no extra categorical capacity at the ceiling alone.

5. **Extend the screen before the next candidate.** `contributor_ceiling_tf.py` costs 51 seconds and predicts the target row's direction correctly. It does not predict the collateral, which is what actually decided this branch. Adding the teacher-forced contribution *level* and the RCE band slopes to the same pass would have flagged this candidate before the simulation, and both quantities are already computed by `rcb_teacher_forced.py`'s frame.

6. **The protected-row rule's amended magnitude clause worked as intended and was not what failed this branch.** It suppressed nothing that should have fired and fired on two bands that genuinely moved away from the human value by more than one pooled standard error. The band-drop clause is what failed the row. The rule needs no further amendment on this evidence; PR #192's suggestion of an absolute floor is still open but this branch gives no new case for it.
