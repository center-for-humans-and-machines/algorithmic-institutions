# Autoresearch log: switch-herding-copula

## 1. Declaration

- **Slot:** switch
- **Base model:** `gnn` switch predictor
  (`artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`)
- **Target rows:** SC — reference stack 3.270439 (band 2-5); slot average
  2.65 over the 16 gnn-switch contexts, >= 2 in 15/16 (range 1.77-3.45),
  concordant. The only S-family deficit: SA 0.721349, SB 0.753990,
  RSA 0.908837 in the reference stack — rates match, the group aggregate
  does not. Success requires SC < 2 in the reference stack (band upgrade
  2-5 -> 1-2), Stage-2 confirmed.
- **Hypothesis:** eligible players in a group see the same situation and
  their switch decisions co-move beyond what shared observable features
  explain (~40% of within-group co-movement survives conditioning on state —
  the motivating comment on PR #140). The simulation draws every agent's
  switch independently, which pins group-level switching to the independence
  floor: an independent null handed the *exact* human switch rates
  reproduces our simulated SC almost exactly (mean larger-group size 5.28
  null / 5.46 sim / 6.09 human; P(fully segregated) 0.018 / 0.030 / 0.144),
  human switching is ~1.8x more variable than independent draws allow, and
  episode memory is absent (first-half/second-half segregation correlation
  0.38 human vs 0.03 sim). Capturing the shared component should widen the
  larger-group-size distribution toward the human one and move SC.
- **Planned change:** Gaussian-copula sampling for the GNN switch predictor —
  a shared standard-normal latent per (episode, switch-round, group), mixed
  with per-agent noise at weight rho, pushed through each agent's own
  predicted Bernoulli marginal; marginals preserved by construction, so
  SA/SB/RSA should not move; no retraining of the marginal model. Two arms,
  selected by Stage-1 score per §5: (A) rho only; (B) rho plus an AR(1)
  persistence phi on the group latent across an episode's switch rounds, to
  carry the episode-memory effect. rho (and phi) are estimated from the
  human train split only, by pairwise-likelihood MLE against the GNN's own
  predicted marginals (the PR #146 estimator, adapted to binary outcomes);
  stored as a field on a copy of the switch artifact; artifacts without the
  field sample independently, bit-identical to today.
- **Stack guards (must not regress):** rows <= 1 baseline 11/21; mean
  baseline 1.759557 (reference stack `gnn x gnn x lin_multinomial`);
  marginal guards SA 0.721349, SB 0.753990, RSA 0.908837 — an S-rate move
  signals a sampler bug.

## 2. Plan

(to be filled by the validated step list)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Deficit profile fetched from
   `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`:
   SC is the gnn switch's only slot-attributable row >= 2 and is concordant
   (>= 2 in 15/16 gnn-switch contexts). CG/RCA/PC/PA are high in every
   switch context and belong to the other slots.
2. Prior switch experiment PR #145 `[FAIL]` (same_group edge feature):
   SC 3.2704 -> 2.8003 Stage 1, slot mean 2.651 -> 2.307 in Stage 2 —
   real but within-band. Its key lesson: contribution-based input features
   made SC *worse* (3.35-3.37) despite better CV loss; do not retry feature
   routes toward SC.
3. Precedent PR #146 `[SUCCESS]` (punisher severity copula): the identical
   mechanism (shared per-round group latent, pairwise-likelihood MLE rho,
   marginals preserved, artifact-gated dispatch) took PD 2.934892 ->
   1.532497 and won 8/8 contexts; its collateral moved SC -0.454 in the
   reference stack — direct evidence the shared-latent mechanism reaches SC.
4. The base switch model has an RNN (`mlp+rnn+edge`), and `predict_autoreg`
   asserts RNN and autoregression cannot combine — the PR #140 comment's
   Option 2 (autoregressive cross-agent sampling) is structurally
   unavailable for this artifact; Option-1-style correlated sampling at the
   sampler is the remaining route, and the copula is its parameter-free-at-
   simulation-time form.
