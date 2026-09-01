# Autoresearch log: contribution — gaussian MLP v2 (group-conditioned features)

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gaussian` (heteroscedastic Gaussian linear,
  `scripts/baselines/gaussian_regressor.py`; artifact
  `artifacts/baselines/contribution_gaussian_best.joblib`)
- **Evaluation stack (§3):** `gaussian x gnn x multinomial` — the
  top-ranked stack containing the base model (rows <= 1: 8/21, mean 1.6405
  in `23_stack_sweep_updated`), **with the punisher slot swapped to the
  PR #160 severity-copula multinomial bundle by maintainer directive**
  (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`).
  The baseline for both §2 gates is re-established by a fresh sim + eval of
  this exact stack before the candidate runs (maintainer: "update the eval
  metrics before going into improvement"). Expected values from the
  pre-revert severity-copula sweep (PR #151 baseline cell): RCA 5.208,
  CA 2.170, RCB 2.347, CG 3.978, rows <= 1 8/21, mean 1.6308 — the fresh
  run's exact numbers replace these.
- **Maintainer directives (2026-09-01):** branch from `main` (not stacked on
  a parent PR); integrate the PR #160 copula punisher into the stack; update
  the baseline metrics first; then a simple, locally-trainable 2-layer MLP
  attempt on the gaussian contribution model whose feature set is chosen to
  exploit the hidden layer's nonlinearity. The copula integration is stack
  plumbing shared by baseline and candidate runs — both gates measure only
  the contribution-slot delta.
- **Target rows** — pinned to the step-4 baseline run (exact, unrounded):
  - **RCA 5.20832162095758** (band > 5) — primary; declared upgrade
    > 5 -> 2-5.
  - **RCB 2.3471849734654717** and **CA 2.1698859787106297** (both band
    2-5) — secondary candidates for 2-5 -> 1-2.
  - **CG 3.9780560538984258** is *excluded* as a target but is the named
    guard: it is what killed PR #151 through gate 2 (mean rose on
    CG 3.98 -> 6.58).
  - Gate 2 threshold: the 21-row mean must fall strictly below
    **1.6308337314805847** (rows <= 1: 8/21, context only).
- **Hypothesis:** Human contribution behavior is nonlinear *and
  group-conditioned*: the punishment response saturates and depends on the
  player's own level, persistence differs at the 0/20 boundaries, and —
  decisive here — players are conditional cooperators who move toward their
  group's recent contribution level. PR #151 proved a 2-layer MLP on the
  heteroscedastic Gaussian heads buys the response rows (RCA 5.21 -> 4.53,
  RCB, RCC) but its CV-shopped 5-feature set carried only round-level group
  context, and sharper individual persistence stopped group means from
  converging (CG exploded). Its post-mortem names this experiment's path:
  keep the MLP's response gains but restore group convergence through the
  features. Giving the hidden layer own-state *and* group-state inputs —
  round-level and window-level group means, so the net can form the
  own-vs-group deviation and a group-trajectory signal internally — lets one
  model express both the state-dependent response structure (RCA/RCB) and
  the conformity pull that co-moves group members (CG guard). Group-mean
  features are also a poor man's shared group state: agents in one group see
  the same values, so their sampled means co-move even under independent
  per-agent sampling (the mechanism PR #159/#165 proved lives at the
  trajectory level).
- **Planned change (one change, contribution slot only):** train
  `gaussian_mlp` (PR #151's estimator machinery, ported) with a
  behaviorally-declared feature core forced in — B1 self
  (`prev_contribution`, `prev_punishment`) + B2 own-group
  (`prev_contribution_mean_group`, `prev_punishment_mean_group`) + B5 window
  (`prev_win_contribution_mean_group`) — plus a small structural candidate
  set (`round_number`, `rounds_since_switch`, `switched_last_choice`), a
  compact hyperparameter grid covering PR #151's unexplored lr/wd/epochs
  edges, and variant selection by evaluation score per §5 (bounded, every
  run logged). Swap the winning bundle into the evaluation stack; verdict
  per §2 against the freshly-established baseline.
- **Iteration budget check (§5):** the incumbent trains in < 1 s per fit
  locally (PR #151 note 2: 0.35-0.91 s at h=8-32); the planned CV is
  minutes of local wall clock, far under 3x. Sims are ~2.5 min on Raven.

## 2. Plan

Validated by the orchestrator, 2026-09-01: targets are §2-shaped (band
upgrades on declared rows + mean), every step is legal per §5 (all features
are prev-family, i.e. observable at decision time — `handcrafted_grid.py`'s
`validate_feature_legality` hard-errors otherwise; hyperparameter/variant
selection is bounded and logged), and nothing touches the frozen surface
(§8: `src/aimanager/evaluation_suite/`, the two `notes/` definition files,
`experiments/`, protocol/scoring parameters). Three revisions were made to
Fable's draft, all recorded in Notes 4-6: the `make_shortlist_config.py`
port is dropped as unused (step 5), the "hand-merge" of `linear_ah.py` is a
verified file-level restore rather than a manual merge (step 5), and the
forced feature core needs an explicit mechanism because
`enumerate_feature_sets` gives every block an implicit OFF (step 6).
Implementer tags per §9 Roles.

- [x] 1. **Worktree env prep** — [Sonnet] in
  `.claude/worktrees/contribution-gaussian-mlp-v2`: `uv sync`; `mkdir -p
  data/baselines` (gitignored CV output dir); verify
  `experiments/baseline/2group_8agent_50ep_bline_{train,test}.csv` and
  `artifacts/baselines/contribution_gaussian_best.joblib` are real files, not
  LFS pointers. No tracked-file changes.
- [x] 2. **Integrate the PR #160 severity-copula punisher** — [Sonnet] verify
  `origin/main` has not touched `src/aimanager/simulation/linear_ah.py`
  since `git merge-base origin/main origin/auto/punisher-severity-copula-v2`,
  then `git checkout origin/auto/punisher-severity-copula-v2 --`
  `src/aimanager/simulation/linear_ah.py`
  `scripts/baselines/punishment_copula_rho.py`
  `tests/baselines/test_punishment_copula.py`
  `artifacts/baselines/punishment_multinomial_severity_copula.joblib`.
  Run `pytest tests/baselines/test_punishment_copula.py` locally (green
  required). One commit, message crediting PR #160.
- [x] 3. **Updated-baseline sim config** — [Sonnet] new file
  `configs/simulation/manager_testing/23_2g8a_gmlp2_base_self_gaussian_contr_gnn_switch.yml`:
  copy of `23_2g8a_self_gaussian_contr_gnn_switch.yml` with managers reduced
  to a single `lin_multinomial_copula` entry
  (path `artifacts/baselines/punishment_multinomial_severity_copula.joblib`),
  one self pairing, `output_dir`/`figure_name` renamed to match the file
  (slug before `_self_` so `evaluation_sweep.py`'s DIR_PATTERN still parses,
  same convention as PR #160's config). Protocol fields byte-identical to
  the 23 family (seed 42, 100 episodes, 24 rounds, `save_per_round: true`).
- [x] 4a. **Unblock the isolated remote dir** (inserted 2026-09-01 after
  step 4 failed at launch; Note 8) — [Opus] `scripts/simulate_cluster.sh`:
  add `SBATCH_EXPORT=ALL` to the exports of the isolated-dir branch. Raven's
  login shells set `SBATCH_EXPORT=NONE`, so `AIMANAGER_VENV` and
  `PYTHONPATH` never reached the sbatch job. Verified by an sbatch probe, not
  by inspection: both variables arrive, the interpreter is the shared venv
  and `aimanager.__file__` resolves to the isolated dir's `src/`.
- [x] 4. **Baseline sim + eval (the metrics update)** — [Opus] submit with
  `AI_REMOTE_DIR='~/autoresearch/contribution-gaussian-mlp-v2'
  scripts/simulate_cluster.sh <config>` (check `squeue` for PENDING jobs
  before any sync; login node is orchestration only), fetch with the same
  `AI_REMOTE_DIR`, then locally `python -m aimanager evaluate <config>`.
  Record the baseline row in §3 Results and pin the Declaration's target
  scores to these exact numbers. Sanity-check against the expected
  pre-revert values (RCA ~5.208, mean ~1.6308, PD in 1-2); a large
  unexplained discrepancy is a stop-and-escalate.
- [x] 5. **Port the gaussian_mlp machinery from PR #151's branch**
  (`origin/auto/contribution-gaussian-mlp`) — [Opus] `GaussianMLPRegressor` in
  `scripts/baselines/gaussian_regressor.py`, registry entry in
  `scripts/baselines/baseline_models.py`, `run_baseline_cv.py` show_ce gate,
  `inspect_best_model.py` save-best branch,
  `tests/baselines/test_gaussian_mlp.py`, and the teacher-forced preflight
  script `scripts/baselines/gaussian_mlp_preflight.py`. The one overlap,
  `src/aimanager/simulation/linear_ah.py`, needs no manual merge: PR #151's
  copy *is* the copula version plus the two `gaussian_mlp` lines (verified —
  the only diff between the two branches' copies is the
  `model_type in ("gaussian", "gaussian_mlp")` branch and two comments), so a
  file-level restore from #151 is the merge, checked by re-running
  `tests/baselines/test_punishment_copula.py`. `make_shortlist_config.py` is
  *not* ported: it only generates configs from a ridge/gaussian CV shortlist
  and step 6 writes the config by hand from the behavioral declaration
  (Note 4). Local `pytest tests/baselines/` + eval-suite tests green;
  `scripts/remote_test.sh` for the PyG suite.
- [x] 6. **Training config** — [Opus]
  `configs/training/baselines/contribution/gaussian_mlp_v2.yml`
  (data conventions identical to `gaussian.yml`: single-copy train file,
  `exclude_flipped: True`, seed 38381, 4 folds, `show_ce: true`,
  `model: gaussian_mlp`, `cv.output: data/baselines/gaussian_mlp_v2_cv.csv`).
  Two blocks only, because `enumerate_feature_sets` gives every block an
  implicit OFF and the declared core must survive it (Note 6):
  - `B_core` — 5 nested sets, each containing the minimum core
    `prev_contribution` + `prev_contribution_mean_group` (so own-vs-group
    deviation is always formable): s0 = `[prev_contribution,
    prev_punishment, prev_contribution_mean_group]`; s1 = s0 +
    `prev_punishment_mean_group`; s2 = s0 +
    `prev_win_contribution_mean_group`; s3 = s0 + both (the full declared
    B1+B2+B5 core); s4 = s3 + `prev_win_punishment_mean_group`.
  - `B7_structural` — 3 sets over the declared structural candidates:
    `[round_number, switched_last_choice, rounds_since_switch]`,
    `[round_number]`, `[switched_last_choice, rounds_since_switch]`.

  6 x 4 = 24 feature sets, of which the 4 with `B_core` OFF (structural-only
  / floor) are excluded from selection by the step-7 filter. Grid: hidden
  [8, 16, 32], lr [0.003, 0.01, 0.03], wd [1e-4, 3e-4, 1e-3], epochs
  [500, 1000] = 54 settings — covers PR #151's unexplored edges (its note 4).
  24 x 54 x 4 folds = 5,184 fits, ~10-20 min local at #151's measured
  0.35-0.91 s/fit on ~9 workers.
- [x] 7. **CV gate** — [Sonnet] `run_baseline_cv.py` locally ->
  `data/baselines/gaussian_mlp_v2_cv.csv`. Selection is the best `ce` row
  among rows whose `features` contain both `prev_contribution` and
  `prev_contribution_mean_group` (the declared core; a deterministic filter
  fixed before the run, not a post-hoc choice). Gate: that row's binned CE
  < 2.445040254938614 (incumbent gaussian CV CE; binned CE gates per
  PR #151's revised protocol — continuous NLL is reported, not gating).
  Fail -> stop, `[FAIL]` PR without a sim.
- [ ] 8. **Save-best + preflight** — [Opus] `inspect_best_model.py --save-best` ->
  `artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib`; TEST
  binned CE reported against incumbent 2.351383364066987. Teacher-forced
  preflight (PR #151's script + one addition): sigma(x) state-dependence,
  implied exact-repeat mass, and the **conformity read** — d mu / d
  (own-vs-group-mean deviation) must be negative (mean pulls toward the
  group mean); a no-pull fit predicts PR #151's CG explosion and is a
  documented stop.
- [ ] 9. **Candidate sim config** — [Sonnet]
  `configs/simulation/manager_testing/23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch.yml`:
  copy of the step-3 baseline config, only `contribution_model` (the new
  bundle) and `output_dir`/`figure_name` change.
- [ ] 10. **Candidate sim + eval** — [Opus] same Raven isolation + local evaluate
  as step 4; log the Results row against the step-4 baseline.
- [ ] 11. **Bounded variant selection (§5)** — [Opus] only if gate 1 passes but
  gate 2 fails on the known CG mode (or RCA misses narrowly): up to two
  further variants — (a) alternative declared feature set from the step-6
  config, (b) tighter capacity on the sigma path (smaller hidden / larger
  wd) trading response sharpness against CG. Each is retrain + one sim +
  one eval, logged. Hard stop after three candidate evaluations total.
- [ ] 12. **Close out** — [Opus] verdict per §2 (band upgrade on a declared target
  AND mean below the step-4 baseline); PR to `main` titled
  `[SUCCESS]`/`[FAIL]` with Hypothesis / Results / Collateral, noting up
  front that the diff carries the PR #160 copula integration by maintainer
  directive; delete `~/autoresearch/contribution-gaussian-mlp-v2` on Raven.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-01 | (baseline) gaussian contr x gnn switch x PR #160 severity-copula punisher, contribution slot unchanged | RCA 5.20832162095758, CA 2.1698859787106297, RCB 2.3471849734654717 (guard CG 3.9780560538984258) | 8/21 | 1.6308337314805847 | baseline — reproduces PR #151's cell bit-for-bit on all 21 rows |

## 4. Notes

1. Ranking per §3 over `23_stack_sweep_updated`: gaussian-contr stacks rank
   `gaussian x gnn x multinomial` 8/21 (mean 1.6405) > `gaussian x gnn x
   gnn` 7/21 > `gaussian x lin x multinomial` 6/21 — the multinomial-punisher
   cell is the evaluation stack, and the maintainer's copula directive swaps
   in the PR #160 bundle (same estimator, `copula_rho` at sampling time).
2. Differentiation from PR #151 (`[FAIL]`, same estimator family): its
   feature set was CV-shopped (rank-1 = 5 features, B1 s0 + B2 s0, no
   window/trajectory group signal), its lr/wd/epochs sat at unexplored grid
   edges, and it had no conformity check before simulating. This experiment
   is the post-mortem's path (b) with the feature core declared from
   behavior, the grid edges covered, and a preflight that tests the CG
   mechanism directly.
3. PR #156 (`[FAIL]`) pins the emission axis: a nonlinear mean under
   *homoscedastic* sampling fails RCA (5.37) while heteroscedastic sigma
   buys it (PR #151: 4.53) — so the heteroscedastic heads stay; the open
   question this experiment answers is whether group-conditioned features
   can hold CG while sigma sharpens.
4. Plan validation (orchestrator, 2026-09-01), revision 1: the port in
   step 5 drops `scripts/baselines/make_shortlist_config.py`. Its PR #151
   diff only adds `--model`/`--hidden` so a config can be *generated* from a
   ridge/gaussian CV shortlist; this experiment's whole point is a feature
   core declared from behavior rather than CV-shopped, so the config is
   hand-written and the CLI change would be dead diff.
5. Revision 2: step 5's "hand-merge" of `src/aimanager/simulation/linear_ah.py`
   is unnecessary. `git diff origin/auto/punisher-severity-copula-v2
   origin/auto/contribution-gaussian-mlp -- src/aimanager/simulation/linear_ah.py`
   is exactly the `model_type in ("gaussian", "gaussian_mlp")` sampling
   branch plus two comment lines — PR #151 was cut when the copula was still
   on `main`, so its copy already *is* copula + gaussian_mlp, and the copula
   scripts/tests/artifact are byte-identical across the two branches (blobs
   `8180c7c`, `bbf30f1`, `c267898`). Step 2 therefore restores the copula
   state, step 5 restores #151's file on top, and
   `test_punishment_copula.py` re-verifies the copula half. Also verified for
   step 2: `origin/main`'s last commit touching `linear_ah.py` is the PR #146
   revert `ab246f4`, an ancestor of the `main` <-> copula-v2 merge base, so
   the restore reintroduces exactly the original diff with no lost `main`
   work.
6. Revision 3: the "forced core" of step 6 has no config-level support —
   `enumerate_feature_sets` (`scripts/baselines/run_baseline_cv.py`) always
   offers each block an implicit OFF, so a core placed in its own block
   disappears in a quarter of the enumerated sets. Mechanism adopted: every
   set of `B_core` repeats the minimum core (`prev_contribution`,
   `prev_contribution_mean_group`), and selection is filtered to rows whose
   `features` string contains both — a rule fixed before the run, so the
   4 core-less combinations (structural-only, floor) simply cannot be
   selected. Grid sized to 24 sets x 54 settings x 4 folds = 5,184 fits.
7. Steps 1-2 confirmed (commit `97f6a3c`): the restore touched exactly the
   four intended paths and is byte-identical to
   `origin/auto/punisher-severity-copula-v2`; `copula_rho` reloads as
   0.3507588625344979 and the 31 copula tests pass locally. Phase A can
   therefore be checked hard: PR #151's own baseline config
   (`23_2g8a_severity_copula_self_gaussian_contr_gnn_switch.yml`) is this
   exact stack with the same seed and the same artifacts, so the refreshed
   baseline must reproduce its 21 rows bit-for-bit (RCA 5.20832162095758,
   CA 2.1698859787106297, RCB 2.3471849734654717, CG 3.9780560538984258,
   8/21, mean 1.6308337314805847). Anything else means the restored code or
   an artifact is not what it claims, and is a stop-and-escalate.
8. Step 4 failed at launch, and the cause is a real bug in `main`, not in
   this experiment: job 29855185 died in 1 s on
   `.venv/bin/activate: No such file or directory`. Raven's login shells set
   `SBATCH_EXPORT=NONE`, so the `AIMANAGER_VENV` / `PYTHONPATH` exports that
   `scripts/simulate_cluster.sh` sets for an isolated remote dir (commit
   `8680db9`, whose comment claims "both propagate into sbatch jobs") never
   reach the job; `run_simulation.sh` then falls back to the isolated dir's
   `.venv`, which by design does not exist. The isolation mechanism had
   evidently never been exercised — `~/autoresearch/` was empty. The second,
   quieter half matters more: with `PYTHONPATH` dropped, the shared venv's
   editable install would have made the job import the SHARED checkout's
   `src/`, so a job that *appeared* to run would have silently simulated
   `main`'s code instead of the branch's. Fix: `SBATCH_EXPORT=ALL` alongside
   the two exports, verified by an sbatch probe printing both variables plus
   `sys.executable` and `aimanager.__file__` (interpreter = shared venv,
   `aimanager` = isolated dir). Kept as its own commit, and it cannot move a
   score: without it no job runs at all. `scripts/train_cluster.sh` carries
   the identical unfixed idiom at lines 108-109 — out of scope here (this
   experiment submits no training jobs, so a fix could not be verified), but
   the next agent to train a GNN from a worktree will hit it.
9. Step 4 (job 29855233, seed 42, 100 episodes, repeats_used 500/500): the
   refreshed baseline reproduces PR #151's severity-copula cell **bit-for-bit
   on all 21 rows** (max |delta| exactly 0.0), mean 1.6308337314805847,
   rows <= 1 8/21. That is the intended outcome and it does double duty: the
   maintainer's requested metrics refresh is done, and it independently
   confirms the restored copula code, the copula bundle and the incumbent
   gaussian artifact are exactly what PR #151 ran — the provenance the
   step-4a fix exists to protect. The §2 gates are now pinned to these
   numbers: gate 1 needs RCA out of > 5 (or RCB/CA out of 2-5), gate 2 needs
   the mean strictly below 1.6308337314805847.
10. Step 7 CV gate PASSES, and the declared core won on its own merits.
    Selected row (lowest `ce` among the 1080 core-containing rows of 1296):
    `B_core:s3 + B7_structural:s2`, hidden 8, wd 3e-4, lr 0.01, 1000 epochs,
    7 features — binned CE **2.3991710817157537** < the incumbent gaussian's
    2.445040254938614, and also below PR #151's CV CE 2.421791; context NLL
    2.692550840548696 < incumbent 2.713452599085718 (and < #151's 2.693316).
    Two things worth recording. First, the pre-fixed core filter turned out
    to be non-binding: the best `ce` row over the *whole* grid is the same
    row, so nothing was excluded by insisting on the core — the declared
    feature set is the CV winner outright. Second, `B_core:s3` is exactly the
    declared B1+B2+B5 core (own state + group contribution + group
    punishment + group contribution trajectory), and the structural block
    that survives is `switched_last_choice + rounds_since_switch` (tenure,
    not `round_number`). Grid edges: hidden 8 and epochs 1000 both sit at an
    edge, so capacity and training length are the axes a follow-up would
    extend; lr and wd are interior, which closes PR #151's Note 4 gap.
