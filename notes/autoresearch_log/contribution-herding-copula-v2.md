# Autoresearch log: contribution-herding-copula-v2

AR(1)-persistent herding copula for the GNN contributor — the third leg of
the copula triptych (punisher PR #160, switch PR #150) — stacked on the
maintainer-designated parent PR #160. Supersedes the un-PRed branch
`auto/contribution-herding-copula` (same mechanism, parented on PR #162):
the maintainer redirected the parent to #160 (2026-08-27, "work on the
contribution model on top of PR #160") before that branch's calibration
ever ran, so no result is discarded — its tested code is ported, its
baselines are re-anchored here.

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gnn` contributor
  (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`),
  unchanged — the change is at the free-running sampler only.
- **Parent PR (§9):** #160, branch `auto/punisher-severity-copula-v2`
  (`[SUCCESS]`, PD 2.93 -> 1.53). Evaluation stack: the parent's config
  `configs/simulation/manager_testing/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml`
  — `gnn` contribution x plain `gnn` switch x severity-copula
  `lin_multinomial` punisher (rho = 0.3507588625344979).
- **Baseline** (the parent's confirmed run,
  `plots/simulation/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch/evaluation/scores.csv`,
  single `lin_multinomial_copula_self` pairing): rows <= 1 **10/21**,
  mean **1.6879978841849728**. Contribution-slot rows >= 2 (§6 target
  candidates): CG 9.808514112722413 (band > 5; raw ratio gap
  0.2594222471221652, noise ceiling 0.026448679600274437),
  RCD 2.941928428442498 (band 2-5), RCA 2.0829074791966917 (band 2-5).
- **Target rows:** **CG** (primary; gate 1 needs CG < 5, i.e. the
  spread-ratio gap halves to < 0.13224339800137218) and **RCD**
  (secondary; gate 1 needs RCD < 2, i.e. the pull-coefficient gap below
  0.16021334822457584). Gate 2 requires the 21-row mean below
  1.6879978841849728. RCA (2.0829) and RCB (1.9881) are watch items, not
  targets — the mechanism makes no clean claim on them.
- **Hypothesis:** groups develop a persistent shared culture —
  group-mates' contributions co-move beyond what observed history
  explains, and the shared component persists across the episode. The
  prior record triangulates this as the one unclaimed piece of the CG
  deficit: within-round residual dependence is real but small (PR #149:
  rho ~ 0.07, ~14% of the teacher-forced gap, growing by round thirds
  0.035 / 0.074 / 0.119 — lock-in), exposure bias is real but small
  (PR #163: schedsamp closes ~7% free-running, family vetoed on cost),
  and an episode-persistent shared latent is the only mechanism that has
  moved CG without taxing the marginals (PR #159: spread ratio
  0.586 -> 0.662, tax-free, dose capped by teacher-forced MLE). The
  switch slot's arm comparison (PR #150) showed the same structure: a
  fresh per-round latent mean-reverts through the dynamics and regresses,
  an AR(1)-persistent one wins. An AR(1) shared per-(episode, group)
  Gaussian latent at the contribution sampler injects the shared
  component every round; because each member's draw feeds back through
  their own `prev_contribution` (this artifact's `x_encoding` is
  `[prev_contribution, prev_punishment, agent_group]` with a group-blind
  fully-connected EdgeModel — no explicit same-group conformity feature,
  the mechanism correction inherited from the prior branch's note 3), a
  *persistent* shift compounds through the closed loop instead of
  washing out, widening the per-(game, round, group) mean spread CG
  measures. **RCD:** a switcher's copula cell is their *arrival* group
  (`apply_switch` updates `agent_group` before `update_contribution`),
  so from the switch round on their draws share the receiving group's
  latent — coherent assimilation toward the new group's level, which is
  exactly the switching pull RCD regresses. Marginals are preserved by
  construction (each agent's draw passes through their own predicted
  CDF), so the C-block/CG anti-correlation tax (§6) is avoided the same
  way the severity and herding copulas avoided it, and no retraining.
- **Planned change:** Gaussian-copula sampling for the GNN contribution
  head. Port the head-agnostic AR(1) copula machinery from the switch
  precedent (PR #150/#162 branches: `src/aimanager/generic/copula.py`,
  the `GraphNetwork` copula dispatch) and the contribution-specific work
  from the superseded branch (head-gate relaxation to
  `y_name in {"does_switch", "contribution"}`, `copula_switch_every = 1`
  since contributions are decided every round, tests, the calibration
  script `scripts/artificial_humans/contribution_copula_rho.py` with the
  AR(1) lag-1 extension, artifact stamping). rho and phi estimated on
  the teacher-forced residual dependence of the frozen contribution GNN
  on the human train split only (pairwise-likelihood MLE, the #146/#149
  estimator; PRIMARY phi from cross-player-only lag-1 pairs — the
  orchestrator amendment inherited from the prior branch, recorded
  before any calibration number was seen). rho must reproduce #149's
  0.06958238086256316 exactly or the data path moved. Parameters stored
  as fields on a copy of the contribution artifact (artifacts without
  the fields sample independently, bit-identical legacy path); the plain
  `gnn` switch artifact carries no copula fields, so the switch slot's
  behavior is untouched (§4). One simulation in the parent's stack, one
  evaluation, verdict from the §2 gates.
- **Iteration budget (§5):** no retraining — one CPU calibration job and
  one standard-protocol GPU simulation; far under the 3x bound.

## 2. Plan

(to be written by the planning subagent and validated by the
orchestrator)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Re-parenting decision (2026-08-27): the maintainer pointed this
   experiment at PR #160 after the prior branch had declared against
   PR #162 (a titled `[FAIL]` with a config-dependent passing draw).
   #160's stack is also the cleaner claim: every slot in it is either a
   confirmed `[SUCCESS]` or the reference model, and the CG baseline
   here (9.8085) matches the re-baselined #149/ar1-copula declarations,
   so the whole prior CG record reads directly against this run.
