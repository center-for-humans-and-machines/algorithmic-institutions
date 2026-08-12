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
  `23_2g8a_severity_copula_self_cat_contr_gnn_switch` /
  lin_multinomial_copula run (maintainer ruling 2026-08-12: the reference
  punisher is now `punishment_multinomial_severity_copula.joblib`, same
  multinomial class):** RCA baseline **5.744514189776296**; rows <= 1
  baseline **8/21**; mean baseline **1.560720754938168**. (Superseded
  original baselines from the pre-copula cell: RCA 5.730682422372414,
  rows <= 1 7/21, mean 1.7806142974986534.)

## 2. Plan

Validated by the orchestrator 2026-08-12 (targets per §2, legality per §5,
frozen surface per §8). Slug: `prev_onehot`; contr label: `cat_onehot`.

- [x] 1. Worktree interpreter check: `uv sync` in the worktree; assert
      `aimanager.__file__` and `handcrafted_grid.__file__` resolve inside
      this worktree; `pytest tests/baselines` green pre-change.
- [x] 2. `handcrafted_grid.py`: emit `prev_contribution_onehot_00..20`
      (N_CONTRIBUTION_LEVELS = 21) from
      `clip(rint(prev_contribution), 0, 20)` in the prev family; NOT in
      CURRENT_VALUED; document in `notes/baseline_feature_defs.md`.
- [x] 3. `tests/baselines/test_baseline_features.py`: add the dummies to
      PREV_FAMILY with an independent pandas reference; explicit test that
      round 0 puts onehot_09 = 1 in both the pipeline and adapter paths and
      rows sum to 1.
- [x] 4. Run `pytest tests/baselines` + eval-suite tests (frozen surface
      untouched proof).
- [x] 5. New config `configs/training/baselines/contribution/
      cat_prev_onehot.yml`: data block identical to `cat.yml` (single-copy
      train file, exclude_flipped, 21 levels, switch_every 4), cv seed
      38381 / 4 folds, C grid [0.001..3.0], set s0 = 5 retained scalars +
      21 dummies (scalar prev_contribution replaced), set s1 = incumbent's
      6 features as matched-search diagnostic only.
- [x] 6. `mkdir -p data/baselines`; run `run_baseline_cv.py`; report s0/s1
      C-curves; check convergence at the selected C with warnings on.
- [x] 7. Save `artifacts/baselines/contribution_categorical_prev_onehot.joblib`
      via `inspect_best_model.py` — selection by CV log loss within s0 rows
      only; verify bundle (26 features, coef shape (21, 26), defaults);
      incumbent joblib untouched. Exactly one variant goes to Stage 1.
- [x] 8. Offline pre-flight (go/no-go, never tuning): teacher-forced
      sampling on train-split human features through LinearAHAdapter;
      report repeat rate / |change|<=1 / mean |change| vs human
      0.438 / 0.621 / 2.29 and cat sim 0.193 / 0.331 / 3.30. Escalate if
      the repeat spike does not appear.
- [x] 9. Stage-1 config `configs/simulation/manager_testing/
      23_2g8a_prev_onehot_self_cat_onehot_contr_gnn_switch.yml`: copy of
      `23_2g8a_severity_copula_self_cat_contr_gnn_switch.yml` (the new
      reference-punisher cell, single lin_multinomial_copula pairing),
      only the contribution artifact swapped and output slugged; protocol
      and RNG stream shape byte-identical to the baseline cell. (Supersedes
      D2's 4-pairing design — the new baseline dir is single-pairing.)
- [x] 10. `squeue -u certuer` (no PENDING jobs), push the joblib to Raven
      explicitly (artifacts/ excluded from sync), `simulate_cluster.sh`.
- [x] 11. Poll; confirm remote per_round.parquet; `fetch_cluster.sh`.
- [x] 12. `python -m aimanager evaluate` the Stage-1 config.
- [x] 13. Keep gate (lin_multinomial_copula run): RCA < 5.744514189776296,
      rows<=1 >= 8, mean <= 1.560720754938168; band upgrade needs RCA < 5.
      Log unrounded + collateral C/RC rows.
- [ ] 14. Not kept, or kept within-band: complete log, `[FAIL]` PR, stop.
- [x] 15. Band upgrade: two 4-pairing Stage-2 configs (gaussian / gnn /
      lin_multinomial_copula / ridge punishers; one gnn-switch, one
      lin-switch; exact names decided at the step) covering all 8 contexts
      with the candidate contribution artifact.
- [ ] 16. squeue check, simulate, fetch, evaluate both Stage-2 configs.
- [ ] 17. Add `cat_onehot` to CONTR_ORDER/CONTR_MARKERS in
      `evaluation_sweep.py` (analysis layer); sweep the candidate dirs
      against the per-context baselines (original 23_* dirs for
      gaussian/gnn/ridge cells, severity_copula dirs for the multinomial
      cells) into `23_stack_sweep_prev_onehot`.
- [ ] 18. Confirm slot claim: candidate beats cat on RCA in (nearly) all
      8 contexts; concordance panel + Kendall's W; guards netted per
      context; unrounded scores.csv for all boundary judgments.
- [ ] 19. Complete log; PR `[SUCCESS]`/`[FAIL]` per §9.7; commits map to
      steps.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-12 | (baseline) copula reference cell, cat contr unchanged | 1 | RCA 5.744514189776296 | 8/21 | 1.560720754938168 | baseline |
| 2026-08-12 | prev-contribution one-hot (C=0.03), same 5 other features | 1 | RCA 2.972152984061371 | 11/21 | 1.4242459459240815 | kept — band upgrade (>5 -> 2-5), run Stage 2 |

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
5. Planner discrepancies, orchestrator rulings: (D1) new training config
   `cat_prev_onehot.yml` instead of editing `cat.yml` — the incumbent
   bundle records its config path and cv output; editing in place destroys
   provenance. (D2) Stage 1 keeps all 4 punisher pairings, unlike the
   copula's single pairing: the sim seed is consumed before MultiManager
   builds the GNN punisher, so a 1-pairing config would shift the torch RNG
   stream relative to the baseline dir; 4 pairings keep pre-run consumption
   identical, and the dir doubles as the gnn-switch half of Stage 2.
   (D3) Stage 2 therefore needs only 1 extra config (lin-switch); the 8
   contexts = 2 switch configs x 4 punisher pairings. (D4) sweep needs the
   `cat_onehot` label registered or it KeyErrors; the slugged dir name
   parses as contr=cat_onehot, never silently averaged into `cat`.
   (D6, ruled (a)) round-0 rows impute prev_contribution = 9 (train
   median), so onehot_09 mixes "repeated 9" with "first round" (320 of 537
   level-9 train rows) — accepted: the scalar had the same contamination,
   and one change per experiment; an `is_first` feature is a candidate
   follow-up. (D5) the locked test split's median default is 10, so the
   reported test_metric is mildly pessimistic — affects neither selection
   nor the sim. (D8) StandardScaler inflates rare dummies (level 19 scale
   ~0.095) — handled by the C grid reaching 1e-3; selection by CV log
   loss. (D9) feasibility: 7457 valid train rows vs 567 parameters, all
   21 prev levels populated (min 67 at level 19). (D11) RNG consumption
   per contribution call is one multinomial draw regardless of feature
   count, so switch/punisher streams are unperturbed.
6. Maintainer ruling (2026-08-12, mid-experiment, before any sim): the
   reference punisher is now the severity-copula bundle
   `artifacts/baselines/punishment_multinomial_severity_copula.joblib`
   (same multinomial class, not a separate slot option; the autoresearch
   doc will be updated by the maintainer). Re-anchored: Stage-1 baseline
   cell is `23_2g8a_severity_copula_self_cat_contr_gnn_switch` (RCA
   5.744514189776296, rows<=1 8/21, mean 1.560720754938168); the Stage-1
   config copies that cell's config (single pairing — supersedes ruling
   D2, whose 4-pairing rationale was RNG parity with the old 4-pairing
   baseline dir); Stage 2 covers the 8 contexts with two 4-pairing
   configs where the multinomial slot points at the copula bundle.
   Training steps 5-7 are unaffected (the punisher does not enter
   contribution training).
7. CV (steps 5-7): 24 rows (floor / s0 / s1 x 8 C values). Selected
   s0 (5 scalars + 21 dummies) at C = 0.03: CV log loss
   2.1099111203770766 +/- 0.034361 vs the incumbent search s1 best
   2.445486104561887 at C = 1.0 (bit-identical to the incumbent's
   recorded cv_metric — search reproduced exactly). Every s0 row at
   C >= 0.003 beats every s1 row; C-curve flat over 0.01-3.0. Saved
   bundle train_metric 2.0259675054010176, test_metric
   1.8823689047046444 (incumbent 2.3858 / 2.2726; same test_floor
   2.717958377177829) — test gain exceeds train gain, no overfit
   signal. Fit converges in 25 iterations (MAX_ITER 1000 untouched);
   incumbent joblib checksum unchanged. Adapter smoke: [1, 8, 1]
   int64 predictions, one-hot rows sum to 1, argmax recovers the
   previous level.
8. Step-8 pre-flight (teacher-forced on the 7149 train rows with a real
   previous round, R=20 replicates, seed 42; train log loss reproduces
   both bundles' recorded train_metric exactly): repeat rate — human
   0.414, incumbent 0.182, new bundle 0.401 (94.4% of the gap closed);
   mean |change| — human 2.22, incumbent 3.19, new 2.52 (68.8% closed);
   |change|<=1 — 0.612 / 0.336 / 0.586. Transition corner at prev in
   {0,5,10,15,20}: the new bundle's diagonal lands within ~0.01 of the
   human empirical values (in-sample caveat; out-of-sample support is
   the test log loss 1.882 vs 2.273). Residual over-dispersion remains
   in the tails, so RCA should improve substantially but need not reach
   the ceiling. GO.
9. Stage 1 (job 29301475): RCA 5.744514189776296 -> 2.972152984061371 —
   a band upgrade (>5 into 2-5); rows<=1 8 -> 11, mean 1.560720754938168
   -> 1.4242459459240815; all three gates pass. Collateral +: the whole
   marginal C block (CA 2.0752 -> 1.7155, CB 1.7107 -> 0.8468, CC 1.5555
   -> 1.0602, CD 1.5852 -> 0.7341), RCD 1.9968 -> 1.3452, SC 2.5628 ->
   2.2672, RSA 0.9886 -> 0.7923. Collateral -: CG 1.7604 -> 4.8242 (the
   sticky self-anchor weakens group-mean convergence — the §6
   anti-correlation trade, now from the opposite direction), RCB 1.7931
   -> 2.3193, PD 1.0300 -> 1.3913. CG/RCB concordance is the thing to
   watch in Stage 2.
10. Stage-2 configs: `23_2g8a_prev_onehot_full_self_cat_onehot_contr_
    {gnn,lin}_switch.yml` — copies of the original 4-pairing cat configs
    with the contribution artifact swapped and the multinomial manager
    renamed/pointed to the copula bundle (label `multinomial_copula`,
    matching the severity_copula baseline dirs). The gnn-switch copula
    cell re-runs Stage 1's context inside the 4-pairing config — a free
    robustness replication under different RNG pre-consumption. Jobs
    29301963 (gnn) and 29301965 (lin).
