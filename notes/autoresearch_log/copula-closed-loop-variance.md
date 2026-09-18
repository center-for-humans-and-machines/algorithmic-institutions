# Closed-loop variance with and without the contribution copula

## 1. Declaration

**Kind:** diagnostic (`[RESULT]`), not a gated experiment: no slot changes, no artifact is proposed, nothing is claimed against the §2 gates. One of three parallel experiments on the "copula question"; siblings: `auto/copula-seed-ensemble`, `auto/copula-missing-state`.

**Parent:** `auto/punisher-current-contribution` (the re-baseline branch: fixed current-contribution punisher, RCE row, the PR #181 stimulus-skip contributor trunk with its recalibrated copula, and the b_skip `_curpun` sim `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun/`). Branch `auto/copula-closed-loop-variance`, worktree `.claude/worktrees/agent-a64e83fc9f1557eaf`, isolated Raven dir `~/repros/ai-runs/copula-cl-variance`.

**Question.** The contribution copula on the stimulus-skip trunk has rho = 0.0395 (phi = 1.0, a static per-(episode, group) latent) yet, on the vnode trunk, moves CG from 2.40 (no copula) to 0.90 (PR #179's ablation, commit e618f9d: both spreads collapse without the latent, sd(group means) 5.51 -> 4.43 and sd(individual) 6.48 -> 5.65). Is the copula restoring a missing within-group correlation (what its calibration measures: the residual co-movement of members given the state, human ~0.07 after removing state per the PR #140 comment), or is it refilling variance that the deterministic trunk loses when the simulation feeds on its own output (closed-loop dispersion collapse: a conditional expectation fed back on itself contracts toward its fixed point, and any shared noise re-inflates the spread of group means)? The two readings call for different fixes: a sampler for the first, a trunk change for the second.

**Method.** Three closed-loop simulations of the same stack (stimulus-skip contributor x joint-exodus GNN switch x current-contribution multinomial copula punisher; 23-family protocol: seed 42, 100 episodes, 24 rounds, `save_per_round`), identical but for the contribution copula fields:

| arm | contributor artifact | rho | phi | sim dir |
|---|---|---|---|---|
| A | `..._vnode_stimulus_skip_herding_copula` (stamped, PR #181) | 0.0395 | 1.0 | `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun` (existing, reused) |
| A' | same artifact, rerun with the drawn latent logged (`copula_z` column) | 0.0395 | 1.0 | `23_2g8a_copula_cl_variance_a_zlog` |
| B | `..._vnode_stimulus_skip` (bare trunk, weight-identical, rho = 0) | 0 | -- | `23_2g8a_copula_cl_variance_b_rho0` |
| C | `..._vnode_stimulus_skip_herding_copula_phi0` (dict copy of A's artifact with `copula_phi = 0`; `scripts/artificial_humans/stamp_copula_phi0.py`) | 0.0395 | 0.0 | `23_2g8a_copula_cl_variance_c_phi0` |

Per round t, for each arm and the human data (`experiments/2group_8agent_50ep.csv`, one copy per game via the evaluation suite's `load_human`): (i) SD of the group-mean contribution over (game, group); (ii) SD of individual contributions; (iii) their ratio (the CG ingredient; CG = |ratio_sim - ratio_human| over the noise ceiling); (iv) the within-group residual correlation given the state: the bare trunk teacher-forced over each arm's own realised trajectories (the machinery of `rcb_teacher_forced.py --sim-parquet`), residual = realised contribution - E[c | that history], then the pairwise Pearson correlation of the residuals between members of one (game, round, group); (v) the mean and SD of the trunk's predictive SD and entropy at the visited states. Then the group-mean variance is decomposed into a between-(episode, group) persistent part and a round-to-round part, and in arm A' the group mean is regressed on the logged latent. Arms B and C get the 22-row evaluation. Script: `scripts/data_analysis/copula_closed_loop_variance.py` (`--teacher-force` on Raven, `--analyse` and `--scores` locally); figures and tables under `plots/data_analysis/evaluation/copula_closed_loop/`.

**Decision rule (stated before the numbers).** If arm B keeps the individual spread but loses the group-mean spread over rounds while its residual correlation given the state sits near the human value, the copula is patching closed-loop collapse and the fix belongs in the trunk's closed-loop behaviour. If arm B keeps the group spread and only the residual correlation is short, the copula is doing what its name says. Arm C separates correlation from persistence: if C recovers A's group spread, the within-round correlation carries the effect; if C sits with B, the persistence does.

**Latent logging (the only code change).** `GraphNetwork._predict_encoded_copula` stores the per-node latent of the round just sampled (`copula_z_last`, read-only, no RNG use, so the RNG contract and every draw are unchanged); `ArtificialHumanEnv` carries it in `state["copula_z"]` (NaN without a copula); `simulate.mem_to_df` writes it as a `copula_z` column. Arm A' is the check that this is a no-op for every other column.

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Latent logging (graph.py, environment.py, simulate.py); phi = 0 stamping script; three sim configs. | done |
| 2 | Stamp the phi = 0 artifact on Raven; run A', B, C (23-family protocol) in the isolated dir. | running |
| 3 | Teacher-force the bare trunk over the human data and the three arms' trajectories (`--teacher-force`, CPU job); fetch. | |
| 4 | Local analysis (`--analyse`): per-round series, block means, decomposition, latent regression, figure. | |
| 5 | `python -m aimanager evaluate` on B and C; the 22-row table for A, B, C (`--scores`). | |
| 6 | Verdict, log, commit sim dirs and outputs, PR against `auto/punisher-current-contribution`. | |

## 3. Results

(filled in below)

## 4. Notes

