# Autoresearch log: punisher-ar-gnn

## 1. Declaration

- **Slot:** punisher
- **Base model:** GNN punisher (`architecture node+edge+rnn`,
  `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`),
  candidate judged against the reference punisher
  (`lin_multinomial` + severity copula, PR #146) inside the reference stack.
- **Target rows:** PD — reference stack 1.5324969616723312 (band 1-2; success
  needs <= 1). Secondary target RPA — reference stack 1.1845929635267691
  (band 1-2; success needs <= 1). Deficit evidence: the GNN punisher's own
  slot-attributable deficit is PD (slot avg 2.634, concordant 2.01-2.88 in
  8/8 contexts of `23_stack_sweep_severity_copula`); the severity copula
  closed only part of the same gap (PD 2.93 -> 1.53 in the reference stack,
  48% above the ceiling).
- **Hypothesis:** a group's punishments in a round are one human manager's
  joint decision. The severity copula proved round-level dependence is real
  but also that an exchangeable Gaussian latent is not the whole story: the
  likelihood-implied rho (0.3508, CI [0.278, 0.423]) and the spread-implied
  rho (~0.52) disagree (punisher-severity-copula log, note 8), leaving PD at
  1.53. An autoregressive factorization of the joint punishment distribution
  — each agent's punishment conditions on round t-1 observables AND on
  groupmates' already-decided punishments of the same round (within-decision
  information the manager trivially has; never current-round contributions)
  — can represent non-exchangeable, non-Gaussian dependence (severity plus
  targeting concentration), moving the group-spread row PD to the ceiling
  and plausibly the punishment-response row RPA with it.
- **Planned change:** one change — a GNN punisher trained (flip-doubled data,
  per convention) with teacher-forced autoregressive conditioning on
  same-round groupmate punishments under a random decision order, and
  sequential (agent-by-agent) sampling in simulation. Marginal-quality risk
  is explicit: the current GNN punisher's marginals trail the multinomial's
  (PA slot avg 1.20 vs 0.65), so the P-family guards below gate the keep.
- **Stack guards (must not regress):** rows <= 1 baseline 10/21; mean
  baseline 1.687998 (reference stack, sim
  `23_2g8a_severity_copula_self_gnn_contr_gnn_switch`). P-family marginal
  guards: PA 0.5924374705499833, PB 0.9512984216948207,
  PC 0.9051112565676434, RPB 0.7705194340881973; RSA sits razor-edge at
  1.0010089590803533.
- **Slug:** `ar_gnn` (configs, artifacts, sim output dirs; slug before
  `_self_` so `evaluation_sweep.py`'s DIR_PATTERN parses).

## 2. Plan

(to be filled by the validated step list)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Deficit profile fetched from
   `plots/data_analysis/evaluation/23_stack_sweep_severity_copula/score_matrix.csv`:
   for punisher = gnn, PD is the only P-family row >= 2 (slot avg 2.634,
   concordant 2.01-2.88 across all 8 contexts); CG/SC/RCA/RCB are high in
   every punisher context and belong to the other slots. For the reference
   punisher (multinomial_copula) PD is already 1.07 slot avg but 1.53 in the
   reference stack — the residual is the non-exchangeable dependence this
   experiment targets.
2. Prior work read: `punisher-severity-copula.md` (merged log; note 8 is the
   motivating finding), PR #150 (switch herding copula — persistence of the
   latent was essential there), PR #151 (contribution gaussian_mlp — better
   marginals bought with independent sampling explode CG; a warning that
   joint structure and marginal fit must be judged together).
