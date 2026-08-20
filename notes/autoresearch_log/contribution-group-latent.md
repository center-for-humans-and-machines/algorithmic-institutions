# Autoresearch log: contribution-group-latent

## 1. Declaration

- **Slot:** contribution
- **Base model:** reference GNN contributor
  (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`)
- **Target rows:** **CG** (9.850261, band > 5 — primary; band upgrade
  requires < 5). Watch: CC (1.606), RCA (2.035), RCD (2.772) as plausible
  co-movers; the C marginals (CA 0.772, CB 0.691, CD 0.650) as the known
  anti-correlated collateral (§6: r ~ -0.7 to -0.9).
- **Hypothesis:** Groups develop persistent shared cultures: group-mates'
  contributions co-move beyond what observed history explains, and the
  shared component persists across the episode rather than resetting each
  round (PR #149: residual rho grows 0.035 -> 0.074 -> 0.119 over round
  thirds; human early-late segregation correlation 0.38 vs sim 0.03).
  Independent per-agent sampling erases this factor, so simulated group
  means regress toward one attractor and the group-mean spread collapses
  (CG 9.85). An episode-persistent shared group latent — #149's direction
  (ii), PR #140 Option 1 — injects a small persistent push per group that
  compounds through the closed-loop peer dynamics, restoring the spread
  CG measures.
- **Planned change (one change):** a 1D per-(group, episode) latent
  `z_g ~ N(0,1)` added to the contribution GNN, entering the emission
  logits through a learned loading vector (`logits += z_g * v`,
  `v` in R^21). Trained by **exact marginal likelihood**: the integral
  over z is a 20-node Gauss-Hermite sum — ~20 forward passes per group
  trajectory, blended by log-sum-exp — so there is no encoder network, no
  ELBO, and no posterior-collapse failure mode. Two training phases,
  selected between by Stage-1 score (legal, §5): (A) base frozen at the
  reference artifact, only the loading learned; (B) short joint finetune
  starting from A. Simulation draws `z_g` once per (group, episode) and
  holds it fixed for all 24 rounds.
- **Relation to `contribution-type-latent` (in flight):** that experiment
  gives each *agent* a persistent disposition (per-agent z, dim 4,
  VAE/ELBO); this one gives each *group* a shared factor (per-group z,
  dim 1, exact quadrature likelihood). Different behavioral claims —
  individual identity vs group culture — and different estimation
  machinery. Siblings, not a retry.
- **Evidence base:** #149 — within-round rho ~ 0.07 closes only ~14% of
  the spread gap, the deficit is trajectory-persistent plus free-running
  drift, and an episode-persistent shared latent is its named direction
  (ii); #151/#154 — a trajectory-level latent is the repeatedly named
  missing piece; #157 — the teacher-forced conditional caps *gated*
  mechanisms, but a marginalized latent is not a conditioning feature, so
  the MLE assigns it exactly the weight the data likelihood supports;
  #146/#150 — a shared normal factor is the mechanism that already works
  post-hoc; this experiment learns it inside the model.
- **Legality note:** z is an architecture change (latent-variable model),
  not an information feature — no observed data column feeds it, so the
  §5 decision-time-information clause does not apply.
- **Guards (§2):** rows <= 1 must not fall below 11/21; mean score must
  not rise above 1.759557.

### Method, at a glance

z is not a computed feature: nothing in the data feeds it. At training
time it is *integrated out* — each (group, episode) trajectory's
likelihood is the prior-weighted average of the model's likelihood
evaluated at 20 fixed z values, so gradient descent makes different z
values specialize to high- and low-culture groups exactly when the data
demands it (the log-sum-exp gradient is soft-EM). At simulation time it
is a *random draw*, shared by the four group members and persistent for
the episode — the in-model analogue of the #146/#150 copula's shared
normal, with the network free to learn during training what the draw
means. If the persistent shared factor does not exist in the data, the
MLE leaves the loading at zero and the experiment fails honestly.

### Config sketch

New training config, identical to the reference
(`group_switching_contribution_50ep.yml`: seed 38381, flip-doubled data,
575-epoch budget, hidden 20) plus:

```yaml
model_args:
  group_latent:
    dim: 1                # z_g ~ N(0,1), one draw per (group, episode)
    n_quadrature: 20      # Gauss-Hermite nodes: exact marginal likelihood
    pathway: logit_skip   # logits += z_g * v, v a learned 21-vector
    loading_init: <from step-1 calibration>
train_phases:
  - freeze_base: true     # phase A: learn v only, base weights frozen
  - freeze_base: false    # phase B: short joint finetune from A
```

Simulation side: when the artifact declares `group_latent`, the simulator
draws `z_g` per (group, episode) at reset and holds it across rounds. No
change to the simulation protocol (seeds, episode counts) or the frozen
surface.

## 2. Plan

Validated by the orchestrator (§9): targets per §2, every step legal per
§5, frozen surface untouched — the simulation change is gated on the
artifact declaring `group_latent`, so legacy artifacts behave identically.

- [x] 1. **Gate moved into training (plan revision, 2026-08-20).** The
  standalone persistence diagnostic was dropped by the maintainer: with
  the base frozen and z entering as `logits += z * v`, phase-A training
  computes base logits once and the 20 quadrature variants are logit
  shifts — phase A *is* the persistence measurement (same MLE as the
  #149-style calibration, on the full joint likelihood), at ~1x forward
  cost. **Gate (now at step 7):** phase-A loading ||v|| materially
  non-zero AND held-out marginal log-likelihood beats the frozen base;
  else stop -> [FAIL] PR. The residual-export machinery written for the
  original step 1 (`scripts/artificial_humans/contribution_latent_phi.py`
  + copula imports, locally validated: same-round estimator reproduces
  #149's estimand to 4e-16; false-positive gate phi_hat 0.0013 on
  rho=0.07/phi=0 synthetics) is kept as optional analysis tooling. Its
  first cluster run was cancelled: BLAS thread oversubscription (51 min
  user vs 2h52m system time), lesson — pin OMP/torch threads to 1 in
  cluster stats jobs.
- [x] 2. `graph.py`: optional `group_latent` on `GraphNetwork` — z as a
  per-(group, episode) scalar input, loading vector v added to the
  emission logits; legacy artifacts load with v = 0 and produce identical
  logits.
- [x] 3. `train.py`: quadrature marginal-likelihood loss (log-sum-exp
  over 20 Gauss-Hermite nodes per (group, episode) trajectory);
  `freeze_base` support; disabled path byte-identical to current training.
- [x] 4. Simulation free-running path: draw z_g at episode reset, hold
  fixed across rounds (the type-latent step-4 caching pattern).
- [x] 5. Raven unit tests: disabled parity; quadrature loss equals a
  brute-force integral on a toy model; z persistence across sim rounds;
  save/load round-trip plus legacy-artifact load.
- [x] 6. black + flake8 over touched files (one batched pass).
- [ ] 7. Training config per the sketch; train phase A then B on Raven;
  fetch artifacts and metrics. **Gate (was step 1):** phase-A ||v||
  materially non-zero and held-out marginal log-likelihood beats the
  frozen base; else stop -> [FAIL] PR. Phase A is cheap (frozen base:
  one forward per batch + 20 logit shifts); only phase B pays the ~20x
  quadrature forward cost and stays short.
- [ ] 8. Stage-1 sim config
  `23_2g8a_group_latent_self_gnn_contr_gnn_switch.yml` — exact copy of
  the reference except the contribution artifact and output paths;
  simulate both phase arms on Raven; fetch `per_round.parquet`.
- [ ] 9. Sanity: between-group contribution spread in the sim exceeds the
  reference run (z reaches behavior) before evaluating.
- [ ] 10. Evaluate locally; append results rows; decision per §2 —
  Stage 2 sweep only on a CG band upgrade; PR either way.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-20 | (baseline) reference stack, lin_multinomial punisher | 1 | CG 9.850261 | 11/21 | 1.759557 | baseline |

## 4. Notes

1. Declared after the maintainer distinguished this branch from
   `contribution-type-latent`: that one owns the per-agent disposition
   latent, this one owns the group-level shared factor. Both descend from
   #149's post-mortem; they can run in parallel.
2. Estimation choice (quadrature over ELBO) removes the posterior-collapse
   risk entirely: with dim 1 the marginal likelihood is exact, no encoder
   is trained, and the latent is used iff it raises the data likelihood —
   the honest in-model counterpart of the #146/#150 two-stage copula fits.
3. Known headwind, stated upfront: #149 attributes most of the CG gap to
   free-running state-tracking loss (teacher-forced spread ratio 0.784 vs
   free-running ~0.59, human 0.847). The bet is that an episode-persistent
   push compounds under free-running dynamics in a way #149's within-round
   copula could not; the phase-A gate and the step-9 spread sanity check
   are the cheap points to falsify this before burning a full evaluation
   cycle.
4. Plan revision (2026-08-20, maintainer): the standalone step-1
   diagnostic was dropped after its first run had to be cancelled (BLAS
   thread oversubscription on the CPU partition; also a coordination
   scare with the parallel type-latent session's training job on the
   shared Raven checkout). Rationale: frozen-base phase-A training
   measures the same persistence signal at ~1x forward cost and inside
   the pipeline that decides the experiment anyway. Diagnostic code kept
   as analysis tooling; estimator validations recorded in step 1.
