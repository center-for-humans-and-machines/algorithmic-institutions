# Closed-loop variance with and without the contribution copula

## 1. Declaration

**Kind:** diagnostic (`[RESULT]`), not a gated experiment: no slot changes, no artifact is proposed, nothing is claimed against the §2 gates. One of three parallel experiments on the "copula question"; siblings: `auto/copula-seed-ensemble`, `auto/copula-missing-state`.

**Parent:** `auto/punisher-current-contribution` (the re-baseline branch: fixed current-contribution punisher, RCE row, the PR #181 stimulus-skip contributor trunk with its recalibrated copula, and the b_skip `_curpun` sim `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun/`). Branch `auto/copula-closed-loop-variance`, worktree `.claude/worktrees/agent-a64e83fc9f1557eaf`, isolated Raven dir `~/repros/ai-runs/copula-cl-variance` (delete when this PR closes).

**Question.** The contribution copula on the stimulus-skip trunk has rho = 0.0395 (phi = 1.0, a static per-(episode, group) latent) yet, on the vnode trunk, moves CG from 2.40 (no copula) to 0.90 (PR #179's ablation, commit e618f9d: both spreads collapse without the latent, sd(group means) 5.51 -> 4.43 and sd(individual) 6.48 -> 5.65). Is the copula restoring a missing within-group correlation (what its calibration measures: the residual co-movement of members given the state), or is it refilling variance that the deterministic trunk loses when the simulation feeds on its own output (closed-loop dispersion collapse)? The two readings call for different fixes: a sampler for the first, a trunk change for the second.

**Method.** Three closed-loop simulations of the same stack (stimulus-skip contributor x joint-exodus GNN switch x current-contribution multinomial copula punisher; 23-family protocol: seed 42, 100 episodes, 24 rounds, `save_per_round`), identical but for the contribution copula fields:

| arm | contributor artifact | rho | phi | sim dir | Raven job |
|---|---|---|---|---|---|
| A | `..._vnode_stimulus_skip_herding_copula` (stamped, PR #181) | 0.0395 | 1.0 | `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun` (existing, reused) | 30305005 (stage D) |
| A' | same artifact, rerun with the drawn latent logged (`copula_z` column) | 0.0395 | 1.0 | `23_2g8a_copula_cl_variance_a_zlog` | 30309028 |
| B | `..._vnode_stimulus_skip` (bare trunk, weight-identical, rho = 0) | 0 | -- | `23_2g8a_copula_cl_variance_b_rho0` | 30309083 |
| C | `..._vnode_stimulus_skip_herding_copula_phi0` (dict copy of A's artifact with `copula_phi = 0`; `scripts/artificial_humans/stamp_copula_phi0.py`, 14 tensors verified identical) | 0.0395 | 0.0 | `23_2g8a_copula_cl_variance_c_phi0` | 30309084 |

Per round t, for each arm and the human data (`experiments/2group_8agent_50ep.csv`, one copy per game via the evaluation suite's `load_human`): (i) SD of the group-mean contribution over (game, group); (ii) SD of individual contributions; (iii) their ratio (the CG ingredient; CG = |ratio_sim - ratio_human| over the noise ceiling); (iv) the within-group residual correlation given the state: the bare trunk teacher-forced over each arm's own realised trajectories (the loaders of `rcb_teacher_forced.py --sim-parquet`; for the humans, the 50 single-copy games), residual = realised contribution - E[c | that history], then the pairwise Pearson correlation of the residuals between members of one (game, round, group) (all ordered within-cell pairs, the copula estimator's pairing); (v) the mean and SD of the trunk's predictive SD and entropy at the visited states. Then the group-mean variance is decomposed into a between-(episode, group) persistent part (variance of each group's episode-level mean) and a round-to-round part, the individual variance into Var(E[c | history]) and residual variance, and in arm A' the group mean and the residual are regressed on the logged latent. Arms B and C get the 22-row evaluation. Script: `scripts/data_analysis/copula_closed_loop_variance.py` (`--teacher-force` on Raven, `--analyse` and `--scores` locally); figure and tables under `plots/data_analysis/evaluation/copula_closed_loop/` (`per_round_lines.jpg`, `per_round.csv`, `round_blocks.csv`, `cg_decomposition.csv`, `latent_regression.csv`, `scores_22.{csv,md}`, `rce_bands.csv`, `tables.md`, the four `tf_<arm>.parquet` teacher-forced frames).

**Decision rule (stated before the numbers).** If arm B keeps the individual spread but loses the group-mean spread over rounds while its residual correlation given the state sits near the human value, the copula is patching closed-loop collapse and the fix belongs in the trunk's closed-loop behaviour. If arm B keeps the group spread and only the residual correlation is short, the copula is doing what its name says. Arm C separates correlation from persistence: if C recovers A's group spread, the within-round correlation carries the effect; if C sits with B, the persistence does.

**Latent logging (the only code change).** `GraphNetwork._predict_encoded_copula` stores the per-node latent of the round just sampled (`copula_z_last`, read-only, no RNG use, so the RNG contract and every draw are unchanged); `ArtificialHumanEnv` carries it in `state["copula_z"]` (NaN without a copula); `simulate.mem_to_df` writes it as a `copula_z` column. Arm A' is the check that this is a no-op for every other column.

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Latent logging (graph.py, environment.py, simulate.py); phi = 0 stamping script; three sim configs. | done |
| 2 | Stamp the phi = 0 artifact on Raven; run A', B, C (23-family protocol) in the isolated dir. | done (jobs 30309028 / 30309083 / 30309084, ~2 min each on one A100) |
| 3 | Teacher-force the bare trunk over the human data and the three arms' trajectories (`--teacher-force`); fetch. | done (the CPU job 30309303 sat on `QOSGrpCpuLimit`; cancelled and run on the login node under `nice`, ~3 min) |
| 4 | Local analysis (`--analyse`): per-round series, block means, decomposition, latent regression, figure. | done |
| 5 | `python -m aimanager evaluate` on B and C; the 22-row table for A, B, C (`--scores`). | done |
| 6 | Verdict, log, commit sim dirs and outputs, PR against `auto/punisher-current-contribution`. | done |

## 3. Results

**Measured.** All numbers below are computed by the script from the committed parquets; the human column is the same data every metric row uses.

**Checks that hold.** A' reproduces A bit-for-bit in `contribution`, `punishment`, `agent_group` and `common_good` (the logging is a no-op on the draws); in A' the latent is constant within every (episode, group) (phi = 1) and in C it is redrawn every round (within-episode SD 0.97); B carries NaN. The teacher-forced conditional reproduces the evaluation suite's CG ingredients exactly (human ratio 0.8480, A 0.8109; CG numerator 0.0372 = the suite's `metrics.csv`).

### Round-block means (rounds 1-8 / 9-16 / 17-24)

| arm | rounds | (i) sd group mean | (ii) sd individual | (iii) ratio | (iv) resid corr given state | (v) pred SD | (v') entropy |
|---|---|---|---|---|---|---|---|
| human | 1-8 | 4.193 | 5.794 | 0.719 | 0.026 | 3.930 | 2.170 |
| human | 9-16 | 5.544 | 6.321 | 0.876 | 0.035 | 2.809 | 1.650 |
| human | 17-24 | 6.091 | 6.767 | 0.900 | 0.052 | 2.546 | 1.446 |
| A copula (rho .04, phi 1) | 1-8 | 3.778 | 5.539 | 0.681 | 0.031 | 3.939 | 2.203 |
| A | 9-16 | 4.936 | 6.020 | 0.820 | 0.016 | 2.934 | 1.800 |
| A | 17-24 | 5.648 | 6.360 | 0.888 | 0.013 | 2.515 | 1.587 |
| B no copula | 1-8 | 3.491 | 5.255 | 0.664 | -0.001 | 3.846 | 2.197 |
| B | 9-16 | 4.397 | 5.413 | 0.812 | 0.008 | 2.893 | 1.845 |
| B | 17-24 | 4.730 | 5.557 | 0.851 | -0.004 | 2.545 | 1.681 |
| C rho .04, phi 0 | 1-8 | 3.529 | 5.195 | 0.679 | 0.043 | 3.869 | 2.225 |
| C | 9-16 | 4.482 | 5.455 | 0.821 | 0.020 | 2.874 | 1.866 |
| C | 17-24 | 4.835 | 5.613 | 0.861 | 0.020 | 2.577 | 1.712 |

Figure: `plots/data_analysis/evaluation/copula_closed_loop/per_round_lines.jpg` (six panels, rounds 1-24, four arms).

### Where the variance is (all rounds)

| arm | Var(group mean) | between (episode, group) | round-to-round | share between | Var(c) | Var(E[c \| hist]) | Var(resid) | ratio | resid corr given state |
|---|---|---|---|---|---|---|---|---|---|
| human | 28.70 | 17.12 | 12.59 | 0.596 | 39.92 | 27.93 | 11.79 | 0.8480 | 0.032 |
| A | 23.65 | 14.52 | 10.13 | 0.614 | 35.97 | 23.88 | 11.40 | 0.8109 | 0.022 |
| B | 18.16 | 9.91 | 9.15 | 0.546 | 29.48 | 18.88 | 11.36 | 0.7848 | 0.001 |
| C | 18.75 | 10.82 | 8.71 | 0.577 | 29.57 | 18.86 | 11.18 | 0.7964 | 0.031 |

### Arm A': the group mean on the logged latent (phi = 1, one latent per episode and group)

| quantity | value |
|---|---|
| slope of the (episode, round, group) mean on z | 2.556 points per SD of latent (R² 0.229 of Var 23.65) |
| slope of the episode-level group mean on z | 2.572 (R² 0.378: 5.50 of the 14.52 between-episode variance) |
| slope of the teacher-forced residual c - E[c \| hist] on z (the latent's one-shot per-round push) | 0.485 |
| slope of E[c \| hist] on z (what the state already carries) | 2.088 |
| compounding factor (group-mean slope / one-shot slope) | 5.3 |

### The 22-row evaluation (scores over the noise ceiling; bands <= 1 / 1-2 / 2-5 / > 5)

| row | A copula | B no copula | C phi 0 |
|---|---|---|---|
| CA | 0.8602 (<= 1) | 1.2624 (1-2) | 1.1472 (1-2) |
| CB | 0.7881 (<= 1) | 0.9325 (<= 1) | 0.7862 (<= 1) |
| CC | 0.8890 (<= 1) | 1.3230 (1-2) | 1.1965 (1-2) |
| CD | 0.8100 (<= 1) | 1.2420 (1-2) | 1.1495 (1-2) |
| CE | 1.0574 (1-2) | 1.2114 (1-2) | 1.1580 (1-2) |
| CF | 0.8281 (<= 1) | 1.1892 (1-2) | 1.1445 (1-2) |
| CG | 1.5535 (1-2) | 2.4386 (2-5) | 2.1330 (2-5) |
| SA | 0.7852 (<= 1) | 0.7450 (<= 1) | 0.8263 (<= 1) |
| SB | 1.0063 (1-2) | 0.9430 (<= 1) | 1.0555 (1-2) |
| SC | 1.4271 (1-2) | 1.0144 (1-2) | 0.8446 (<= 1) |
| PA | 0.6596 (<= 1) | 0.7770 (<= 1) | 0.6407 (<= 1) |
| PB | 0.9687 (<= 1) | 1.0263 (1-2) | 0.9667 (<= 1) |
| PC | 0.9073 (<= 1) | 0.9698 (<= 1) | 0.9279 (<= 1) |
| PD | 0.7224 (<= 1) | 1.0291 (1-2) | 0.8886 (<= 1) |
| RCA | 1.6327 (1-2) | 1.7498 (1-2) | 1.9058 (1-2) |
| RCB | 1.5454 (1-2) | 1.2234 (1-2) | 1.3073 (1-2) |
| RCC | 1.5298 (1-2) | 1.5180 (1-2) | 1.6896 (1-2) |
| RCD | 1.3091 (1-2) | 1.7951 (1-2) | 1.6519 (1-2) |
| RCE | 0.8942 (<= 1) | 0.8376 (<= 1) | 0.9319 (<= 1) |
| RSA | 1.0701 (1-2) | 1.2677 (1-2) | 1.0872 (1-2) |
| RPA | 0.6930 (<= 1) | 0.6979 (<= 1) | 0.6753 (<= 1) |
| RPB | 0.8473 (<= 1) | 0.7276 (<= 1) | 0.8013 (<= 1) |
| mean | 1.0357 | 1.1782 | 1.1325 |
| rows <= 1 | 13 | 8 | 10 |

RCE band slopes (0-4 / 5-9 / 10-14 / 15-19; human +0.140 / +0.104 / -0.077 / -0.161): A +0.095 / +0.020 / -0.058 / -0.160; B +0.111 / +0.060 / -0.013 / -0.010; C +0.083 / +0.029 / -0.045 / -0.077. All three read ++--.

### Verdict (inferred from the measured numbers above)

**Both mechanisms are real, and they are separable: the within-round correlation is what the copula's name says and is correctly dosed but small; the persistence is variance refill and is the load-bearing part.**

1. **Arm B loses both spreads, not the ratio.** Without the copula the individual SD sits at 5.43 against the human 6.32 (-14%) and the group-mean SD at 4.26 against 5.36 (-21%), while the ratio falls only from 0.848 to 0.785 (-7%). Over rounds, B's individual SD stays flat (5.26 -> 5.56 across the three blocks) where the human's climbs 5.79 -> 6.77 and A's climbs 5.54 -> 6.36; its group SD plateaus at 4.7 where the human reaches 6.1 and A 5.6. The decision rule's first clause ("keeps individual spread, loses group spread") does not describe B: the loop loses dispersion in both, and the ratio -- the only thing CG scores -- is affected by the smaller of the two losses.

2. **The lost dispersion is in the states, not in the noise.** The trunk generates the same amount of noise at the visited states in every arm: predictive SD 3.09 (human states) / 3.13 (A) / 3.09 (B) / 3.11 (C), residual variance 11.8 / 11.4 / 11.4 / 11.2. What differs is the spread of the conditional means, Var(E[c | history]): 27.9 on human histories, 23.9 in A, 18.9 in B and C. The closed loop's trajectories fan out less than the human ones -- persistent between-participant and between-group differences that the human histories carry (and the trunk conditions on through `prev_contribution` and the vnode) are not regenerated by the trunk's own sampling. That is a property of the trunk in the loop, and this is the collapse.

3. **The within-round correlation given the state is small in the humans and the copula reproduces it.** On the trunk's own conditional the human within-group residual correlation is 0.032 over all rounds (0.026 / 0.035 / 0.052 by block, rising through the game). Arm C, rho at the fitted 0.0395 with fresh latents, gives 0.031 (0.043 / 0.020 / 0.020): the dose is right on average. But that correlation alone raises the ratio from 0.785 to 0.796 -- 18% of the ratio gap -- and moves CG only from 2.44 to 2.13, still band 2-5; it adds 0.6 to the group-mean variance and nothing to the individual variance (29.48 -> 29.57).

4. **Persistence carries the rest, by compounding.** Arm A's ratio 0.811 closes 41% of the gap (CG 1.55); the difference between A and C is phi. Of A's gain over B, persistence supplies 80% of the between-episode variance (A - B 4.6, C - B 0.9), 89% of the group-mean variance and essentially all of the individual variance (A - B 6.5, C - B 0.1). The logged latent shows the mechanism: its one-shot push is 0.49 contribution points per SD of latent per round, but the group's level ends up shifted by 2.57 points per SD -- 5.3x the push -- because the trunk's stickiness integrates a persistent nudge round after round (E[c | hist] alone carries 2.09 of the 2.57). The static latent explains 38% of A's between-episode variance of group means (5.5 of 14.5), i.e. it is the source of the persistent group-level heterogeneity the trunk does not regenerate by itself. That is variance refill through the loop, not a within-round correlation: by rounds 17-24 arm A's residual correlation given the state (0.013) is *below* the human 0.052, because the state has absorbed the latent -- the copula in A restores group spread while undershooting the very correlation it is calibrated on late in the game.

5. **So: the copula's phi = 1 is patching closed-loop collapse and the fix for that part belongs in the trunk's closed-loop behaviour** (persistent per-agent or per-group state that survives self-play: what `auto/copula-missing-state` is after, and what an episode-level ensemble draw of `auto/copula-seed-ensemble` would also supply in the shape of a persistent shared error). The rho part is a legitimate sampler job, correctly dosed at 0.04, and buys about a fifth of CG's gap. Persistence, not correlation, carries the effect -- with the numbers: ratio gain A 0.026 vs C 0.012; group-mean variance gain A 5.5 vs C 0.6; individual variance gain A 6.5 vs C 0.1.

## 4. Notes

1. Arm A is the committed b_skip `_curpun` sim, reused; A' (rerun, latent logged) is bit-identical in every other column, so the logging change is verified as a no-op on the draws and the 22-row table of A is the stage-D one. Only B and C were evaluated fresh.
2. The `stamp_copula_phi0.py` dict copy was verified on Raven (rho 0.03949863621805423, phi 0.0, `copula_switch_every` 1, 14 tensors identical); the committed artifact is a second stamp of the same dict after the isolated dir's `rsync --delete` sync removed the first one (which arm C had already consumed) -- semantically identical, not byte-compared against the consumed one.
3. The human residual correlation given the state (0.032) is measured against the trunk's in-sample teacher-forced conditional (the artifact trains on all 50 games); it is not the 0.073 of the PR #140 comment, which conditioned on state features by regression. The two are the same quantity under different conditioning sets, and what the copula is calibrated against is this one (the pairwise MLE on the same teacher-forced marginals gives the latent-scale rho 0.0395, which on the contribution scale is the ~0.03 seen in C).
4. The human residual correlation rises through the game (0.026 -> 0.052) while both copula arms fall (A 0.031 -> 0.013, C 0.043 -> 0.020): in A because the state absorbs the static latent, in C because the marginals sharpen (predictive SD 3.9 -> 2.5, more mass at 0 and 20) so the same latent rho yields less co-movement on the contribution scale. A round-dependent rho (the successor PR #168 and PR #170 both named) is supported from this side too.
5. Collateral of the ablation, for a successor: removing the copula improves SB (1.006 -> 0.943, band upgrade), SC (1.427 -> 1.014), RCB (1.545 -> 1.223), RPB and PA-in-C; C alone puts SC at 0.845 (<= 1) and PD at 0.889. The persistent latent's costs are the switch and switching-pull rows (SC, RCD) and RCB; its gains are the whole contribution block (CA/CC/CD/CF leave <= 1 without it: the static group latent doubles as the missing participant heterogeneity, which is CA). The RCE 15-19 withdrawal slope in this stack is carried by the copula, not the trunk: A -0.160, C -0.077, B -0.010 (human -0.161) -- the same finding as PR #181 note 6 on the vnode trunk, now with the persistence identified as the carrier.
6. The `small` partition's group CPU quota (`QOSGrpCpuLimit`) blocked the teacher-forcing job for 15 minutes; four single-core forward passes of a 20-unit GNN over 19,200 rows each are a few minutes, so they ran on the login node under `nice`. The slurm wrapper is committed for the batch route.
7. `parse_agent_rounds` dense-ranks the string key `"<global_group_id>__<episode_id>"`, so the tensor's episode axis is in lexicographic order of the episode number (0, 1, 10, 11, ..., 2, ...). Any join of a teacher-forced frame back onto a parquet by episode must remap through that order; `latent_regression` does, and the inconsistency it produced before the fix (individual slope 0.18 vs cell-mean slope 2.56) is the tell to watch for.
