# Autoresearch log: contribution-prev-onehot

## 1. Declaration

- **Slot:** contribution
- **Base model:** `cat` (categorical linear,
  `artifacts/baselines/contribution_categorical_best.joblib` — sklearn
  multinomial LogisticRegression over 21 levels on 6 scalar features).
- **Target rows:** RCA — Stage-1 cell (cat contr x gnn switch x
  lin_multinomial punisher) 5.730682422372414; slot average 5.735; fully
  concordant (5.54–5.89 in all 8 contexts, essentially constant across the
  other slots — purely slot-attributable). Band > 5; a band upgrade
  requires RCA < 5 (into 2–5), confirmed by Stage 2.
- **Hypothesis:** humans are sticky — 43.8% of human round-to-round
  transitions repeat the previous contribution exactly, 62.1% lie within
  +/-1 (mean |change| 2.29). The cat sim produces 19.3% exact repeats and
  mean |change| 3.30, over-dispersed in all four RCA round types (the RCA
  visual shows human change IQR ~[-1, +1] vs cat [-3, +3] everywhere). The
  cause is representational: with a *scalar* `prev_contribution`, each
  class logit is linear in the previous level — the model can shift mass
  up or down but cannot concentrate it *at* the previous value. Encoding
  the previous own contribution categorically (one-hot, 21 levels) gives
  every previous level its own per-class intercepts — a full 21x21
  transition structure whose diagonal ridge expresses the human repeat
  spike — which should compress the sampled change distribution in every
  round type and move RCA. (Reference points: gnn contributor repeats
  32.8% and scores RCA 2.85 — repeat-rate fidelity and RCA move together.)
- **Planned change:** in the categorical contribution baseline only,
  replace the scalar `prev_contribution` feature with a one-hot encoding
  of the previous own contribution (21 dummy features in the shared
  feature pool); keep the other 5 features; refit with the existing
  training pipeline and its CV convention (single-copy training data,
  log-loss metric), C searched as a hyperparameter. One change; the
  gaussian/ridge baselines and other slots untouched.
- **Stack guards (must not regress), Stage-1 cell
  `23_2g8a_self_cat_contr_gnn_switch` / lin_multinomial run:** rows <= 1
  baseline **7/21**; mean baseline **1.7806142974986534**.

## 2. Plan

(to be filled by the validated plan)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Deficit profile fetched from
   `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`:
   RCA is the cat contributor's worst row (slot avg 5.74) and the most
   concordant deficit in the whole matrix (range 5.54–5.89 over all 8
   contexts). PC/PA/RPB averages are punisher-attributable (they collapse
   to <= 1 in the multinomial/gnn punisher contexts); PD/SC are the known
   shared independence-floor rows; RCD (2.05) is a possible secondary but
   not declared.
2. Prior contribution experiments (PRs #144, #147) are GNN-slot: self-history
   dropout worsened RCA monotonically (masking the self-anchor noises the
   change statistic) — evidence that RCA wants a *sharper* self-anchor, not
   a weaker one, which is exactly what the one-hot transition structure
   provides. No prior experiment has touched the cat baseline.
3. Empirical basis for the hypothesis (human file, one copy per game;
   change = contribution[t] - contribution[t-1] within episode/participant):
   human repeat rate 0.438, |change|<=1 0.621, mean |change| 2.29 (n=18400
   doubled transitions); cat sim (Stage-1 cell, lin_multinomial run) repeat
   0.193, |change|<=1 0.331, mean |change| 3.30; gnn sim repeat 0.328,
   |change|<=1 0.565, mean |change| 2.43.
4. Reference stack per `notes/autoresearch.md` §3 is still
   gnn x gnn x lin_multinomial (the merged severity-copula punisher has not
   been promoted into the reference definition by the maintainer), so
   Stage 1 swaps the candidate into the cat x gnn x lin_multinomial cell
   and compares against that cell's existing scores.
