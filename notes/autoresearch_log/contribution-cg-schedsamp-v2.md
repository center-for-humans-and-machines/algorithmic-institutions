# Autoresearch log: contribution-cg-schedsamp-v2

Resume of the `[ABANDONED]` `auto/contribution-cg-schedsamp` experiment
(abandoned 2026-08-13 by the maintainer as a cost/time call, before any arm
finished training; implementation complete and fully tested there), rebased
onto the `[SUCCESS]` parent PR #160 (`auto/punisher-severity-copula-v2`)
under the current single-stage, two-gate protocol. Maintainer-requested
addition: a cheap two-arm pilot (with / without scheduled sampling) that
must show a CG effect and yield a wall-clock estimate before the full-size
training runs are submitted; if the pilot does not separate, stop and ask
the maintainer.

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gnn`
  (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`)
- **Parent PR (§9):** #160, branch `auto/punisher-severity-copula-v2`.
  Branch and worktree created from its head (`b7dabfc`); the PR targets
  that branch so the diff shows only this experiment.
- **Evaluation stack (§3, inherited from the parent):**
  `gnn x gnn x lin_multinomial+copula`, sim config
  `configs/simulation/manager_testing/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml`.
  Baseline = the parent's confirmed scores
  (`plots/simulation/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch/evaluation/scores.csv`):
  **CG 9.808514112722413**, RCD 2.941928428442498, RCA 2.0829074791966917,
  rows <= 1 **10/21**, mean **1.6879978841849728**.
- **Target rows:** **CG** 9.808514 (band > 5; gate 1 needs < 5) — the worst
  row of the entire score matrix, concordant across all GNN-contribution
  contexts. Secondary declared targets: **RCD** 2.941928 and **RCA**
  2.082907 (band 2-5; gate 1 satisfied by 1-2 or better) — the
  punishment-response rows are scored on free-running histories, so
  training under self-generated histories should sharpen the conditional
  response they measure.
- **Hypothesis:** the CG deficit is free-running state-tracking loss
  (exposure bias), not sampling structure: PR #149 measured the same
  model + independent sampler at spread ratio 0.784 teacher-forced on real
  histories (human 0.847) vs ~0.59 free-running (floor 0.583), with the
  within-round sampler worth only ~14% of the gap. Trained exclusively on
  human histories, the model mean-reverts when conditioned on its own
  slightly-off outputs and the drift compounds over 24 rounds. Training it
  under its own rollouts should teach it to hold a group norm through
  self-generated histories, recovering group-mean spread (CG) and the
  free-running punishment response (RCD/RCA).
- **Planned change:** the scheduled-sampling curriculum from
  `auto/contribution-cg-schedsamp`, restored unchanged (train.py steps
  2-4 + `test_scheduled_sampling.py`; 98 remote + 89 local tests were green
  there, p=0 reproduces the legacy path bit-identically at the input level,
  logits to ~1e-5): training unrolls episodes round by round; with
  per-agent probability p (0 below ramp_start, linear to p_max at
  ramp_end, hold after), `prev_contribution` at round t is replaced by the
  model's own detached sample from round t-1. Targets remain the true
  human actions; no architecture change, same feature set. p_max selected
  between arms by evaluation score (legal per §5).
- **Pilot (maintainer-requested, precedes the full runs):** two ~150-epoch
  arms — control p_max=0 and schedsamp p_max=0.5 with the ramp scaled
  proportionally — each simulated under the standard 23-family protocol
  and evaluated; compare CG and the free-running spread ratio, and measure
  per-epoch wall time of the unrolled path to price the 575-epoch runs.
  **Stop-and-ask gate:** if the pilot arms do not separate on CG, escalate
  to the maintainer instead of submitting the full-size jobs. The pilot's
  fold-loss trajectory in the hold phase also answers whether 575 epochs
  suffice for the real run or need extending.
- **Guardrail watch (not targets):** the marginal C block (CA/CB/CD/CF,
  all <= 1) — the failure mode that killed #154/#155/#156; and the known
  risk that held-out teacher-forced log-loss ticks up while free-running
  behavior improves (declared on the original branch; the evaluation suite
  is the judge, log-loss is a diagnostic only).

## 2. Plan

(to be filled by the validated step list)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-25 | (baseline) parent stack, severity-copula punisher (PR #160) | — | CG 9.808514112722413, RCD 2.941928428442498, RCA 2.0829074791966917 | 10/21 | 1.6879978841849728 | baseline |

## 4. Notes

1. Direction re-affirmed from the abandoned branch's note 1: PR #149's
   decomposition makes the training regime the measured lever; #157 and
   #159 both independently name scheduled sampling as their successor
   (#157: the conformity gate is pinned at w~0.08 by teacher forcing;
   #159: the group latent delivered 38% of the required spread move and
   its remainder is the free-running state-tracking loss).
2. The maintainer's pilot requirement replaces the original plan's
   optional teacher-forced preflight (step 10 there) with a stronger
   free-running one: same-protocol sims of short-trained arms measure the
   CG effect directly and price the full runs before they are bought.
