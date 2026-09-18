# Autoresearch log: diagnostic — emission head and closed-loop state spread

Branch `auto/head-state-spread-diagnostic` (worktree `.claude/worktrees/agent-a5a141fc7c7452f71`), created from `origin/auto/punisher-current-contribution-gmlp`. The PR opens with `--base auto/punisher-current-contribution`. Remote isolation dir `~/repros/ai-runs/head-diagnostic`.

This is a **diagnostic**, not a candidate: it trains nothing and ships no model. It exists to decide whether a combined architecture — graph body with per-group node and direct punishment path, plus an ordinal location-scale emission with explicit inflation at both corners and at repeat-previous — is worth building. The answer had to be clean either way, so the write-up separates §3 (measured) from §4 (inference) strictly.

## 1. Declaration

- **Question.** Does a location-scale (Gaussian) emission head resist closed-loop contraction better than a categorical head over 21 free logits?
- **Hypothesis under test.** A categorical head has no structure tying its 21 levels together, so off the human manifold it relaxes toward the training marginal, while a location-scale head moves a scalar and keeps extrapolating a monotone shift.
- **Prior result being extended.** PR #186 (`auto/copula-closed-loop-variance`) established on the categorical stimulus-skip trunk that the closed loop keeps the per-round noise but loses state spread, using the decomposition `Var(c) = Var(E[c | history]) + Var(residual)` over a free-running simulation. Its four rows are **quoted, not recomputed** (§3.4).
- **Counter-evidence declared in advance.** The Gaussian stacks are not better on the evaluation suite's group-spread row (1.85 and 1.67 against the categorical 1.55), though their switch models differ, so that row is confounded. This diagnostic was expected to be able to *confirm* the counter-evidence rather than overturn it.
- **Arms.** Case c `contribution_gaussian_mlp_inflated_group_copula` (discrete mixture of a binned Gaussian body with status-quo and corner atoms) and case d `contribution_gaussian_mlp_v2_group_copula` (rounded heteroscedastic Gaussian), each with its group copula ON (the committed `_curpun` sims) and OFF (rho-zero stamps).
- **Hygiene.** Nothing retrained, no copula recalibrated; zeroing the copula for the ablation is the only permitted change and the only one made.

## 2. Plan

1. Verify the inherited inputs (rho-zero bundles weight-identical to their parents; configs identical to their `_curpun` parents but for the bundle).
2. Run the two copula-off simulations on Raven in the isolated dir and fetch them.
3. Teacher-force each bundle over its own arm's realised trajectories and over the 50 human games; decompose `Var(c)`.
4. Probe the off-manifold gain on human states shifted by ±2, 4, 6 points, for all three heads.
5. Verdict, stated in the direction the evidence points.

## 3. Results (measured)

### 3.1 The inherited inputs are what they claim to be

The rho-zero bundles differ from their parents in exactly one substantive key. Independent check (not the stamper's own asserts): the estimator net is weight-identical after reload (4 tensors, 109 params inflated / 82 params v2), the scaler's `mean_` and `scale_` are identical, and the only differing key is `copula_rho_p` — `0.048443521435665396 -> 0.0` (inflated) and `0.04378520865574197 -> 0.0` (v2). `copula_rho_t` was already `0.0` in both parents. Two provenance keys are added (`copula_rho0_source`, `copula_rho0_source_sha256`); the recorded source sha256 matches the parent file on disk in both cases.

The two sim configs differ from their `_curpun` parents only in `contribution_model`, `output_dir` and `figure_name`. Seed 42, 100 episodes, 24 rounds, `switch_every: 4`, punisher `punishment_multinomial_current_contr_severity_copula`, switch model `switch_exodus_k_onehot`, `valid_model` `raven_script_22` are all preserved.

### 3.2 The two copula-off simulations

Jobs **30317175** (`c_infl_rho0`) and **30317176** (`d_kexo_rho0`), one A100 each, 2:07 elapsed, exit code `0:0`. Both logs carry the PROVENANCE line resolving `aimanager` to `~/repros/ai-runs/head-diagnostic/src/`, i.e. the branch's code and not the shared checkout's. Turning the copula off moves the group-mean spread in the direction PR #186's arm B did: sd of group means 4.91 -> 3.98 (inflated), 4.48 -> 3.66 (v2).

### 3.3 The teacher-forced reconstruction is faithful

Each bundle is teacher-forced over its arm's realised trajectories through `LinearAHAdapter._pool_from_arrays`, which is the same function the closed loop itself calls via `_build_pool` — the features are the simulation's own, not a re-derivation. The parquet is reshaped directly to `[A, T]` arrays per episode, so PR #186's `parse_agent_rounds` lexicographic-episode hazard does not arise: no dense-rank join is performed anywhere in this path.

The reconstruction is validated by a property that wrong features could not produce — the realised residual variance lands on the model's own mean predictive variance in every arm:

| arm | Var(residual) | mean predictive variance | mean residual |
|---|---|---|---|
| c_infl | 13.38 | 14.06 | +0.087 |
| c_infl_rho0 | 14.58 | 14.73 | +0.032 |
| d_kexo | 11.49 | 11.59 | +0.090 |
| d_kexo_rho0 | 12.49 | 12.44 | +0.027 |
| human (infl) | 14.67 | 14.22 | +0.067 |
| human (v2) | 14.71 | 11.36 | +0.050 |

The one shortfall (c_infl at 0.95) is the copula's within-group correlation, which makes residuals non-independent; it is absent in the copula-off arm (0.99). The human row under v2 at 1.29 is the v2 head being underdispersed on human histories, a property of that head and not of the reconstruction. As a cross-check on the human data being the same 50 games PR #186 used, `Var(c) = 39.916` here against its quoted 39.92.

### 3.4 State spread (headline)

The first four rows are PR #186's, quoted. `var_c`, `var_cond_mean`, `var_resid` are over teacher-forced agent-rounds; `sd_group_mean` and `sd_individual` are the CG ingredients over the canonical frame.

| arm | model | Var(c) | Var(E[c\|hist]) | Var(resid) | sd group mean | sd individual | ratio | retention |
|---|---|---|---|---|---|---|---|---|
| human histories | skip_categorical | 39.92 | 27.93 | 11.79 | 5.36 | 6.32 | 0.848 | 1.000 |
| skip A (rho .0395, phi 1) | skip_categorical | 35.97 | 23.88 | 11.40 | 4.86 | 6.00 | 0.811 | 0.855 |
| **skip B (no copula)** | skip_categorical | 29.48 | **18.88** | 11.36 | 4.26 | 5.43 | 0.785 | **0.676** |
| skip C (rho .0395, phi 0) | skip_categorical | 29.57 | 18.86 | 11.18 | 4.33 | 5.44 | 0.796 | 0.675 |
| human histories | infl | 39.92 | 24.70 | 14.67 | 5.36 | 6.32 | 0.848 | 1.000 |
| human histories | v2 | 39.92 | 23.34 | 14.71 | 5.36 | 6.32 | 0.848 | 1.000 |
| c_infl (copula on) | infl | 37.63 | 22.87 | 13.38 | 4.91 | 6.14 | 0.800 | 0.926 |
| **c_infl_rho0 (copula off)** | infl | 30.70 | **16.51** | 14.58 | 3.98 | 5.54 | 0.719 | **0.668** |
| d_kexo (copula on) | v2 | 30.98 | 18.51 | 11.49 | 4.48 | 5.57 | 0.804 | 0.793 |
| **d_kexo_rho0 (copula off)** | v2 | 25.29 | **13.30** | 12.49 | 3.66 | 5.03 | 0.728 | **0.570** |

`Var(E[c | hist])` is a property of the model as well as of the states: the three heads explain different shares of the *same* human data — 27.93 categorical, 24.70 inflated, 23.34 v2, all out of 39.92. Dividing every arm by the categorical 27.93 would charge the Gaussian heads for being worse fits rather than for contracting, so `retention` is each arm's `Var(E)` over its **own** human-history value.

Episode-cluster bootstrap (2000 draws; episodes resampled whole in both the sim arm and the human reference, independently, because agents within a game share a trajectory):

| arm | Var(E[c\|hist]) | 95% CI | retention | 95% CI |
|---|---|---|---|---|
| c_infl | 22.87 | [21.30, 24.42] | 0.937 | [0.790, 1.106] |
| c_infl_rho0 | 16.51 | [15.57, 17.51] | 0.678 | [0.575, 0.802] |
| d_kexo | 18.51 | [17.15, 19.77] | 0.800 | [0.674, 0.952] |
| d_kexo_rho0 | 13.30 | [12.69, 13.87] | 0.576 | [0.490, 0.683] |

### 3.5 Per-round blocks

| arm | rounds | sd group mean | sd individual | ratio | predictive SD |
|---|---|---|---|---|---|
| human | 1-8 | 4.19 | 5.79 | 0.719 | 4.10 (infl) / 3.77 (v2) |
| human | 9-16 | 5.54 | 6.32 | 0.876 | 3.35 / 2.95 |
| human | 17-24 | 6.09 | 6.77 | 0.900 | 3.16 / 2.66 |
| c_infl | 1-8 | 4.02 | 5.73 | 0.698 | 4.13 |
| c_infl | 9-16 | 5.16 | 6.23 | 0.828 | 3.36 |
| c_infl | 17-24 | 5.44 | 6.37 | 0.854 | 3.10 |
| c_infl_rho0 | 1-8 | 3.39 | 5.46 | 0.616 | 4.16 |
| c_infl_rho0 | 9-16 | 4.33 | 5.62 | 0.770 | 3.45 |
| c_infl_rho0 | 17-24 | 4.17 | 5.50 | 0.758 | 3.25 |
| d_kexo | 1-8 | 3.84 | 5.41 | 0.707 | 3.85 |
| d_kexo | 9-16 | 4.56 | 5.49 | 0.830 | 3.00 |
| d_kexo | 17-24 | 4.94 | 5.72 | 0.863 | 2.69 |
| d_kexo_rho0 | 1-8 | 3.35 | 5.24 | 0.636 | 3.91 |
| d_kexo_rho0 | 9-16 | 3.76 | 4.94 | 0.761 | 3.14 |
| d_kexo_rho0 | 17-24 | 3.82 | 4.85 | 0.788 | 2.93 |

The human group-mean spread rises monotonically across the three blocks (4.19 -> 5.54 -> 6.09). Both copula-on arms rise too (4.02 -> 5.16 -> 5.44; 3.84 -> 4.56 -> 4.94). Both copula-off arms **stall or reverse** in the last block (4.33 -> 4.17 inflated; 3.76 -> 3.82 v2), i.e. the contraction is not a level offset but a failure of late divergence.

### 3.6 Off-manifold gain

Every realised contribution in a human game is shifted by delta, clipped to the grid, `prev_contribution` rebuilt by the loader's own t-1 roll, and the conditional expectation recomputed; `gain(delta) = mean(E_delta - E_0) / delta`. **This probe touches the contribution model alone** — no switch model, no punisher, no closed loop — so unlike §3.4 it is not confounded (§4.1). All three models are scored on identical row sets (`n` = 6023 / 6948 / 7628 / 7702 / 7436 / 6718 for `own_in_grid`, 3807 for the common set), verified equal cell by cell.

Gain on the common set (own previous contribution in [6, 14], the rows that stay in-grid at every delta):

| delta | skip_categorical | infl | v2 |
|---|---|---|---|
| -6 | 0.957 | 0.897 | 0.852 |
| -4 | 0.970 | 0.927 | 0.889 |
| -2 | 0.999 | 0.945 | 0.914 |
| +2 | 0.951 | 0.957 | 0.922 |
| +4 | 1.010 | 0.942 | 0.912 |
| +6 | 1.023 | 0.910 | 0.885 |

Grid clipping means the group-mean feature moves less than delta (realised shift 0.838 to 0.896, a property of the data and identical for both Gaussian bundles, so it normalises all three the same way). Normalised:

| delta | skip_categorical | infl | v2 |
|---|---|---|---|
| -6 | 1.142 | 1.070 | 1.017 |
| -4 | 1.116 | 1.066 | 1.023 |
| -2 | 1.115 | 1.055 | 1.019 |
| +2 | 1.092 | 1.099 | 1.059 |
| +4 | 1.174 | 1.095 | 1.060 |
| +6 | 1.219 | 1.083 | 1.054 |

On the wider `own_in_grid` set the same ordering holds (categorical 0.910-1.013, infl 0.850-0.874, v2 0.814-0.840). The Gaussian body's location `mu` has gain 0.903-0.947 (infl) and 0.888-0.940 (v2), so the mixture's atoms are not what limits it — the body itself moves less than one-for-one.

**No head's gain decays with |delta|.** The categorical head is flat-to-rising (0.957 at -6, 1.023 at +6 on the common set); both Gaussian heads are mildly hump-shaped, peaking near delta 0 and falling ~0.05 at |delta| = 6. Categorical is above infl in 10 of 12 cells and above v2 in 12 of 12.

## 4. Notes (inference)

1. **The hypothesis is falsified in its mechanism.** The premise was that a categorical head, having no structure across its 21 levels, would relax toward the training marginal off the human manifold. It does not. Its gain is the flattest and the highest of the three, and it is the only head whose gain *rises* at the extremes rather than falling. Whatever the 21 free logits fail to do, they do not fail to extrapolate a monotone shift in the conditioning state. The location-scale parameterisation is not buying extrapolation, because extrapolation was not missing.

2. **And it is falsified in its consequence.** On the headline copula-off comparison the Gaussian heads are *worse*, not better: `Var(E[c|hist])` 16.51 (inflated) and 13.30 (v2) against the categorical 18.88, all against human 27.9 / 24.7 / 23.3. On the ratio row, 0.719 and 0.728 against 0.785, human 0.848.

3. **The fit-normalised reading is the fairest one to the Gaussian heads, and it is a tie at best.** Retention: categorical 0.676, inflated 0.678 (CI [0.575, 0.802]), v2 0.576 (CI [0.490, 0.683]). The inflated head is a statistical dead heat with the categorical; the v2 head is lower, with the categorical value sitting just at its upper CI bound. So the honest summary is: **one Gaussian head ties, one is worse, neither is better.** No reading of these numbers supports "location-scale resists contraction better".

4. **The confound is real but it does not rescue the hypothesis, and the clean probe agrees with the confounded one.** The categorical arms ran with `switch_joint_exodus`, the Gaussian arms with `switch_exodus_k_onehot`; punisher, valid model, seed, episodes and cadence are identical, so the switch model is the single cross-lineage difference in §3.4. That confound is entirely absent from §3.6, which runs the three contribution models over the same human states with no switch model in the loop. Both measurements order the heads the same way (categorical >= inflated > v2). A confound that flipped the true ordering in §3.4 would have to be coincidentally reproduced by an independent probe that cannot see it.

5. **What the copula contrast says, and why it matters more than the head.** Within each stack the copula is worth far more than the head choice: retention 0.668 -> 0.926 (inflated) and 0.570 -> 0.793 (v2), against PR #186's 0.676 -> 0.855. The spread between the best and worst *head* at fixed copula setting is ~0.10 in retention; the spread between copula off and on within a single head is ~0.22-0.26. The emission head is the smaller lever by a factor of two.

6. **The inflated head is nonetheless the better of the two Gaussians, and for a legible reason: calibration of the noise, not of the state.** On human histories the inflated head's mean predictive variance is 14.22 against a realised residual variance of 14.67 — calibrated to 3%. The v2 head's is 11.36 against 14.71: the realised residual variance exceeds what the model thinks it emits by **29%**, i.e. v2 is substantially underdispersed. Per round block the same gap shows as predictive SD against realised residual SD — inflated 4.10/3.35/3.16 against 4.32/3.57/3.48 (5-9% under), v2 3.77/2.95/2.66 against 4.33/3.59/3.47 (13-23% under). The atoms at the corners and at repeat-previous are what close that gap, and they are a property of the *mixture*, not of the location-scale body. If anything here is worth carrying into a future head, it is the inflation.

7. **Contraction is a late-episode failure, not a level offset** (§3.5). Both copula-off arms track the human group-mean spread reasonably in rounds 1-8 (3.39 and 3.35 against 4.19) but stop diverging after round ~16 while the human keeps spreading to 6.09. Whatever generates persistent between-group divergence in the human data is a slow, accumulating process, and the copula supplies a crude version of it (a per-episode group latent) while no per-round emission head supplies any of it. This is consistent with Note 5: the missing ingredient is *persistence across rounds*, which is a property of the latent, not of the shape of the per-round conditional.

## 5. Verdict

**This kills the combined head as motivated.** The specific argument for building it — that an ordinal location-scale emission would hold state spread where 21 free logits collapse — is contradicted by both measurements available. The categorical head extrapolates better off-manifold (gain 0.95-1.02 against 0.85-0.96, and it is the only one that does not sag at the extremes), and it retains at least as much closed-loop state spread (retention 0.676 against 0.678 and 0.576). The evidence is not mixed on the question as posed; it is one-directional.

It does not kill every element of the proposed architecture, and the distinction matters. The **inflation at the corners and at repeat-previous** is the one component that demonstrably earns its place (Note 6): it is what takes the v2 head's underdispersion (residual variance 29% above predictive) to the inflated head's near-exact match with the human predictive-SD trajectory. The graph body with per-group node and direct punishment path was never tested here and is untouched by this result. What is dead is the *location-scale emission* as a contraction remedy.

If someone wants to argue the combined head anyway, the thing that would settle it is not another emission head. It is a **persistent latent**: Note 5 shows the copula (a per-episode group latent, rho ~0.04, phi 1) is worth twice what any head choice is worth, and Note 7 shows the residual failure is specifically the absence of late-episode divergence. A learned, state-dependent persistent group effect — as opposed to the copula's fixed, state-independent one — is the lever with the remaining headroom.

## 6. What this leaves for a successor

- **Do not re-run the head comparison.** Both the confounded closed-loop measurement and the unconfounded gain probe are in this branch with their scripts; `scripts/data_analysis/head_state_spread.py` regenerates every number from committed inputs in four stages (`--teacher-force`, `--gain`, `--gnn-gain` on Raven, `--analyse`).
- **The one loose thread that would tighten §3.4** is a categorical arm run with `switch_exodus_k_onehot` instead of `switch_joint_exodus`, which would remove the last cross-lineage difference and make the closed-loop row exactly like-for-like. It costs two ~2-minute Raven sims. It is *not* needed for the verdict (Note 4), but a successor who wants to quote the retention numbers as a clean head contrast should run it first.
- **The live question is the persistent latent, not the head** (Note 5, Note 7). The copula's phi=1 per-episode group draw is state-independent by construction; PR #186 showed phi=0 buys nothing (retention 0.675 against 0.676) while phi=1 buys 0.18. So the value is entirely in the *persistence*, not in the correlation. A head or body that can carry a slow group-level state across rounds — an RNN group node whose hidden state is not reset, or an explicit learned group effect — is the untested family with the largest indicated headroom.
- **Carry the inflation forward** if a new contributor is built on the Gaussian lineage (Note 6); drop the plain `gaussian_mlp_v2` body, which is underdispersed on human histories and the weakest arm on every row measured here.
- **`gain_skip.csv` must be regenerated on Raven** if the categorical trunk changes: the stimulus-skip artifact is not on this lineage, so `--gnn-gain` reads PR #186's tree read-only via `HEAD_DIAG_TREE=~/repros/ai-runs/copula-cl-variance`. Everything else in the script runs locally on CPU.
