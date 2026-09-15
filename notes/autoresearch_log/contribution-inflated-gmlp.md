# Autoresearch log: contribution — status-quo- and corner-inflated emission on the gaussian_mlp_v2 group-copula contributor

Branch `auto/contribution-inflated-gmlp` (worktree
`.claude/worktrees/contribution-inflated-gmlp`), created from
`origin/auto/switch-exodus-k-onehot` at `dd35443` — the head of the
maintainer-designated parent PR #174, per §9 "Building on a `[SUCCESS]` PR".
The PR opens with `--base auto/switch-exodus-k-onehot`. Remote isolation dir
`~/autoresearch/contribution-inflated-gmlp`.

## 1. Declaration

- **Slot:** contribution.
- **Parent PR:** **#174** `[SUCCESS] One-hot the joint head's group sizes`
  (`auto/switch-exodus-k-onehot`, stacked on #172 on #170 on #167). Its log
  is `notes/autoresearch_log/switch-exodus-k-onehot.md`; its Declaration
  pre-declared CG's return to `2-5` as the price of its SC move (the
  fully-merged share fell 0.214 -> 0.186), and its notes 20-21 hand the
  next move to the contribution slot. The sibling that took that hand-off
  literally — PR #175 `[FAIL]`, `auto/contribution-size-onehot`, own-group
  size as `1/n` on this same stack, 2026-09-15 — is read and not retried
  (Notes 1, 9).
- **Base model:** the parent stack's contributor, PR #170's candidate
  `artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib`
  (sha256 `da42031ab0ca5bc2ea355036f2e07f5dd37b369d49bf0bf6999c9ebe68d0ea7c`):
  the gaussian_mlp_v2 trunk (`contribution_gaussian_mlp_v2_best.joblib`,
  sha256 `2f0b02e2588dbd8b2c4860ca3918d670095a6eb32851bcec931392c2d6a02e75`
  — 2-layer heteroscedastic Gaussian MLP, hidden 8, wd 3e-4, lr 0.01,
  1000 epochs, seed 38381, 7 features `prev_contribution, prev_punishment,
  prev_contribution_mean_group, prev_punishment_mean_group,
  prev_win_contribution_mean_group, switched_last_choice,
  rounds_since_switch`, trained by
  `configs/training/baselines/contribution/gaussian_mlp_v2.yml` on the
  single-copy 40-episode train split, continuous Gaussian NLL objective)
  **plus** #170's stamped group copula (`copula_rho_p = 0.04378520865574197`,
  `copula_rho_t = 0.0`, interval-censored pairwise MLE on the train split).
  Sampler: `c_i = clip(rint(mu_i + sigma_i z_i), 0, 20)` with
  `z_i = sqrt(rho_p) u_g + sqrt(1 - rho_p) e_i`, `u_g` per (episode, group)
  (`src/aimanager/simulation/linear_ah.py::_sample_levels_gaussian_copula`).
  The base is "trunk + stamped copula" and so is the candidate: the rho is
  model-conditional (it is fitted against the bundle's own teacher-forced
  21-bin marginal), so it is re-estimated on the new trunk with #170's
  estimator and stamper — the recipe's derived step, as PRs #173 and #175
  ruled, not a second change. This model trains **locally in ~7 s per fit**
  (Note 12); nothing in this experiment runs on Raven except the two sims.
- **Evaluation stack (§3, parent rule of §9):** the parent's candidate config
  `configs/simulation/manager_testing/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
  — this contributor x the one-hot-k joint-exodus GNN switch
  (`artifacts/artificial_humans/switch_exodus_k_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`,
  sha256 `28dd4b40…c820d`) x PR #160 severity-copula multinomial punisher
  (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`,
  sha256 `9e3cf677…8cc2f`), `valid_model`
  `raven_script_22/model/rnn_False__dataset_full.pt`, single pairing
  `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
  `switch_every: 4`, `save_per_round: true` — with **only
  `contribution_model` swapped** (plus `output_dir` / `figure_name`), so the
  RNG context and the single pairing are preserved.
- **Baseline for BOTH §2 gates**, read at full precision from
  `plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch/evaluation/scores.csv`
  in this worktree (21 rows, run `ah group_switching managed by
  lin_multinomial_copula_self`, 500/500 repeats, seed 42; the parent's
  `per_round.parquet` hashes
  `3cb8b3d72784afe09464e0048d27d7cb05f7a3b157780f296e99d168407fef3f`, parent
  note 14):

  | row | score | band | numerator | denominator |
  |---|---|---|---|---|
  | CA | 1.6090078134256451 | 1-2 | 1.6794894304111205 | 1.0438043969689725 |
  | CB | 0.9355125181513249 | <= 1 | | |
  | CC | 1.0355592574724157 | 1-2 | | |
  | CD | 1.115667249665237 | 1-2 | | |
  | CE | 0.9474137306581857 | <= 1 | | |
  | CF | 1.344653071211141 | 1-2 | 0.05621951948267676 | 0.041809683617529195 |
  | **CG** | **2.079263618744627** | 2-5 | 0.05499377725668383 | 0.026448679600274437 |
  | SA | 0.8104274621473664 | <= 1 | | |
  | SB | 0.8905141825966464 | <= 1 | | |
  | SC | 0.9798486593189664 | <= 1 | 0.19060799999999936 | 0.19452799999999945 |
  | PA | 0.6190657898701699 | <= 1 | | |
  | PB | 0.960438980131689 | <= 1 | | |
  | PC | 0.8883018644753072 | <= 1 | | |
  | PD | 0.8653952317181541 | <= 1 | | |
  | **RCA** | **3.50675152179351** | 2-5 | 0.9818661982020965 | 0.2799930910702011 |
  | RCB | 1.9103192887378266 | 1-2 | 0.6645145087480038 | 0.3478552054965939 |
  | RCC | 1.0879967153785937 | 1-2 | | |
  | RCD | 0.7316138678546696 | <= 1 | | |
  | RSA | 1.3172344098473787 | 1-2 | | |
  | RPA | 1.3140842885539825 | 1-2 | | |
  | RPB | 0.7272136595819294 | <= 1 | | |

  mean **1.2226801514921317**, rows <= 1 **11/21** (context, not a
  criterion), **gate-2 ceiling 1.3449481666413449** (1.1 x the mean).

- **Target rows (two, each with its own mechanism claim; pre-declaration
  binds, nothing else is claimed):**
  - **CG 2.079263618744627** (`2-5`) — **the maintainer's assigned target.**
    CG = |ratio_sim - ratio_human| / 0.026448679600274437 with ratio =
    SD(group-mean contribution over (episode, round, group) cells) /
    SD(individual contribution); human ratio 0.8480163543652899, the parent
    sits **below** it at 0.79336 (SD group means 4.4807 / SD individual
    5.6478; confirmed from `per_round.parquet`, Note 2). Band `1-2` needs the
    500-repeat numerator under **0.052897359200548874**, i.e. the sim ratio
    above **0.795118995164741** (+0.0018 on the point estimate, +0.0021 on
    the scored one); `<= 1` needs ratio > 0.8215676747650155 (+0.0286).
    **Pre-registered expectation from the schedule-matched closed-loop proxy
    (Note 8): CG moves in the right direction by ~0.08 of score, to ~2.00 —
    the band edge itself. A crossing is a coin flip, and it is claimed only
    if the mechanism's own CG signature (U3 below) moved with it.** Every
    other contribution-slot mechanism I measured moves CG the wrong way
    (Notes 5-8); this is the only one with the right sign.
  - **RCA 3.50675152179351** (`2-5`) — **the mechanism's home row, the stack's
    worst row, and the declared primary.** RCA is the EMD of `dc = c_{t+1} -
    c_t` per round type (four types, human-weighted; `no_switch_allowed`
    carries 0.79 of the weight). Band `1-2` needs the numerator under
    **0.5599861821404022** (2 x the denominator; -42.97 %). The deficit is
    the exact-repeat spike: human P(c_{t+1} = c_t) = **0.4395** against the
    parent's **0.1684**, human P(20 | prev 20) 0.789 vs 0.468, P(0 | prev 0)
    0.610 vs 0.364 (Note 4) — a Gaussian with sigma ~3.4 puts ~0.15 on
    dc = 0 whatever mu does (PR #175 note 9, reconfirmed). **Pre-registered
    expectation (Note 8): RCA 3.41 -> 1.97 in the proxy, i.e. ~2.06 applied
    to the real parent — at the band edge; a crossing is a coin flip.**
- **Gate 2:** 21-row mean must stay <= **1.3449481666413449**. Predicted
  (Note 8): the mean *falls* ~0.07 to ~1.15, so gate 2 is expected to pass
  with a wide margin; the experiment's robust value lies there and in the
  likelihood (Note 7), not in either band crossing.
- **Guards (declared, non-gating, reading fixed in advance; predicted
  deltas from Note 8's scored proxy, applied to the parent's real scores):**
  - **RCB 1.9103192887378266 (`1-2`)**: predicted -0.33 (-> ~1.58); a `<= 1`
    crossing is not expected and not claimed.
  - **CA 1.6090078134256451 (`1-2`)**: -0.22 (-> ~1.39); **CF
    1.344653071211141 (`1-2`)**: -0.10 (-> ~1.25); **CD 1.115667249665237**:
    -0.05. Improvements within band, not claimed.
  - **RCC 1.0879967153785937 (`1-2`)**: **+0.21** (-> ~1.30) — the corner
    atoms change who sits at 20 and how they leave it; the largest
    predicted cost. **RCD 0.7316138678546696 (`<= 1`)**: +0.13 (-> ~0.86),
    holds its band on the prediction; PR #175 lost RCD's band on a
    contribution change, so it is watched. **CB 0.9355125181513249 (`<= 1`)**:
    +0.07 (-> ~1.01) and **PB 0.960438980131689 (`<= 1`)**: +0.04 (-> ~1.00)
    — both may leave `<= 1` by noise-sized amounts; **PA 0.6190657898701699**:
    +0.07. The 0-atom's closed-loop overshoot (P(c = 0) 0.136 against human
    0.094 in the proxy) is the mechanism behind CB/PA/PB; U4 below watches
    it.
  - **SC 0.9798486593189664 (`<= 1`, margin 0.02), SA, SB**: the switch trunk
    reads per-capita common good, a function of contributions (PR #173's
    propagation channel; PR #175 moved SC 0.98 -> 0.82 through it). The
    proxy holds the parent's regrouping schedule fixed, so it cannot predict
    these; reported, not claimed in either direction.
  - **RSA 1.3172344098473787, RPA 1.3140842885539825**: inherited; no
    mechanism claim.
- **Behavioral claim (§5, one sentence):** people treat "give the same as
  last round", "give nothing" and "give everything" as discrete choices,
  not as the peak and tails of a bell — a human repeats last round's
  contribution exactly in 44 % of rounds (79 % at 20, 69 % at 0) where the
  Gaussian contributor manages 17 % (47 % / 36 %) — and because whole groups
  lock into those states and stay there, the human group-mean distribution
  carries three times the sim's mass at >= 18 (0.093 vs 0.031) and its lone
  players are bimodal (sd 8.0 vs 5.9); an emission with explicit probability
  atoms at the status quo and at the two corners, its Gaussian body kept for
  everything else, is what lets the contributor hold a level, and a group
  hold a corner, from one round to the next — which is what RCA scores
  directly and what CG scores through the group-mean tails.
- **Planned change (one change, contribution slot only):** replace the
  emission head of the gaussian_mlp_v2 trunk by a **status-quo- and
  corner-inflated binned Gaussian**: the same `Linear(7 -> 8) -> tanh`
  hidden layer feeding `Linear(8 -> 5)` = (mu, log sigma, three atom
  logits), with

      P(c | x) = pi_body(x) BinnedN(c; mu(x), sigma(x))
               + pi_rep(x) 1[c = prev_contribution]
               + pi_0(x) 1[c = 0] + pi_20(x) 1[c = 20],

  the four weights a softmax (body logit fixed at 0), `BinnedN` the 21-level
  discretisation the sampler already realises (`bin_probs`: unit bins, tails
  folded into 0 and 20), fitted by the **21-way cross-entropy** (the
  quantity this lineage already gates on as `test_logloss_binned`) at the
  incumbent's exact setting — hidden 8, wd 3e-4, lr 0.01, 1000 epochs, seed
  38381, the same 7 features, same train split, same scaler convention —
  with the atom set chosen among `{prev}`, `{prev, 20}`, `{prev, 0, 20}` by
  the lowest 4-fold CV cross-entropy on the train split (a likelihood
  choice, pre-specified, ties to the fewer atoms; the scratch fit puts the
  three-atom set first by a wide margin, Note 7). No new input feature:
  `prev_contribution` is already the trunk's first feature; the atoms are
  the emission's shape, not conditioning information. Sampling: the
  adapter's existing persistent group latent `z_i = sqrt(rho_p) u_g +
  sqrt(1 - rho_p) e_i` (same 3n draws, same order, same arrival-group rule)
  is pushed through the discrete CDF, `c_i = F_i^{-1}(Phi(z_i))` — the
  inversion `_sample_levels_copula` already performs for the multinomial
  punisher — so marginals are preserved exactly for any rho and the
  dependence structure of PR #170 carries over unchanged in kind. Then
  #170's estimator re-fits `rho_total` against the new trunk's own 21-bin
  marginal (expected ~0.043, Note 7) and #170's stamper writes the
  candidate bundle; swap it into the parent's config; one sim, one
  evaluation.
- **Expected effect sizes and their uncertainty (Note 8; the proxy is the
  closed-loop rollout of the real contribution and punisher bundles along
  the parent parquet's own per-episode regrouping schedule, scored through
  the evaluation suite's `score_all` at 500 repeats, seed 42).** Its
  incumbent arm reproduces the real parent to 0.10 on RCA (3.41 vs 3.51),
  0.29 on CG (1.79 vs 2.08), 0.28 on CA, 0.13 on CF, 0.33 on RCB; it cannot
  see the S rows or RSA (schedule fixed) and it lacks the validity model.
  Deltas, candidate arm minus incumbent arm: **RCA -1.45, RCB -0.33, CA
  -0.22, CF -0.10, CG -0.08, CD -0.05; RCC +0.21, RCD +0.13, CB +0.07, PA
  +0.07, PB +0.04; mean -0.071.** Applied to the real parent: RCA ~2.06,
  CG ~2.00, mean ~1.15. Both declared crossings sit at their band edges;
  the proxy's own miscalibration on the incumbent arm (0.1-0.3 per row) is
  the uncertainty, and it is of the same size as the distance to either
  edge. Stated plainly: **P(gate 1) is roughly a coin flip on each of two
  rows; P(gate 2) is high.** The maintainer asked for headroom over the
  0.0021 CG crossing; this experiment has none on CG (Note 9 says why
  nothing in this slot does) and its headroom is on RCA's likelihood side
  — test 21-way CE 2.3101 -> 1.8855, -18 %, the largest fit gain in the
  linear lineage — rather than on RCA's band edge.
- **Pre-registered unsoundness criteria (decided from the candidate's
  `per_round.parquet`, whatever the gates say):**
  - **U1 (the mechanism did what it claims):** closed-loop P(c_{t+1} = c_t)
    rises from **0.1684** to **>= 0.30** (human 0.4395; proxy 0.395), and
    P(20 | prev 20) from **0.468** to **>= 0.65** (human 0.789). If RCA
    crosses while U1 fails, RCA moved for a reason other than the declared
    one; the title stays honest under §2 but the log says so.
  - **U2 (RCA moved where the mechanism lives):** the `no_switch_allowed`
    round type's EMD (parent 0.9065, 0.79 of the weight) must carry at least
    half of RCA's numerator reduction.
  - **U3 (the CG signature):** the share of (episode, round, group) cells
    with group-mean contribution >= 18 rises from **0.0307** to **>= 0.05**
    (human 0.0925; proxy 0.070), and the ratio's rise comes with
    SD(group means) rising at least as fast as SD(individual) — if CG
    crosses `1-2` while U3 fails, the crossing is noise, not the declared
    cause, and it is **not claimed**.
  - **U4 (the overshoot watch):** closed-loop P(c = 0) must stay **<= 0.15**
    (human 0.094, parent 0.054, proxy 0.136) and P(c = 20) <= 0.17 (human
    0.134, proxy 0.126). An excess past these means the corner atoms
    self-reinforce beyond the data, which is the likely cause of any CB /
    PA / PB regression.
- **Iteration budget (§5):** one fit of the incumbent setting takes 0.71 s
  (PR #175), the inflated head **6.9 s** for 1000 epochs (measured, Note
  12); the CV is 3 atom sets x 4 folds + 1 refit = 13 fits, ~1.5 min; the
  copula estimator with its 200-resample bootstrap and round-trip ~5 min;
  the scored proxy ~6 min; two Raven sims of ~2.5 min. Everything is far
  under 3x anything in the lineage, and nothing trains on Raven (this
  slot's model is not a GNN; the assignment's `train-ah` log check does
  not apply).
- **Legality and frozen surface.** An emission-head architecture change
  (§5 legal) on the existing feature set; no new observable; the two
  corners are the game's own endowment bounds and the status-quo atom is a
  behavior (exact repetition, 44 % of human rounds), not a term keyed to a
  metric's bin — RCA is an EMD over `dc` with no bins, CF's shares are at
  the same 0/20 the game defines. The copula dose is re-estimated on human
  training data by the established estimator, never tuned. No seed,
  episode count, scoring parameter or protocol field changes. Nothing under
  `src/aimanager/evaluation_suite/`, `notes/evaluation_metric_defs.md`,
  `notes/eval_scoring_schema.md` or `experiments/` is touched; the proxy and
  the diagnostics import `convert`, `metrics` and `scoring` read-only. No
  other branch's log file is written.

### Candidate mechanisms considered, and why three were dropped (measured)

All measurements are from read-only scratchpad diagnostics on the parent's
`per_round.parquet`, the human reference through `convert.load_human`, the
train split through `gmlp_group_copula_diagnostic.build_rows`, and two
closed-loop proxies: **B** (PR #170's rollout — real contribution and
punisher bundles, fixed 4/4 groups, no switching, 100 x 24, seed 42) and
**C** (the same rollout along the parent parquet's own regrouping schedule,
so singletons, size composition and regroupings are present with the
sorting held fixed). Proxy C's incumbent arm lands at ratio 0.8020 against
the real 0.7934; proxy B's at 0.8275. Full numbers in Notes 2-8.

| # | mechanism | behavioral claim | evidence for | evidence against | verdict |
|---|---|---|---|---|---|
| 1 | Trained episode-persistent group latent (PR #159's mechanism ported to the gmlp: `z_g` into mu / log sigma via learned loadings, exact marginal likelihood) | groups carry a shared culture | #159 was tax-free; the maintainer's hint | the within-cell dose is **already delivered** in the stack — sim residual MLE 0.0449 vs stamped 0.0438 (Note 3), so #166's dilution worry does not bite; a teacher-forced loading estimates the same estimand the copula MLE already gave; the two pathway variants a trained latent could add were tested as sampling-time proxies and are CG-inert: round-growing rho (thirds MLE 0.027/0.036/0.077) **-0.009** (B) / **-0.003** (C), shared-scale t-copula (nu 4) **-0.006** (B) | dropped — no headroom |
| 2 | Per-agent persistent disposition latent in the same copula (`rho_a`; joint censored MLE (rho_g, rho_a) = (0.04, **0.11**), same-agent lags >= 2 MLE 0.159, LR 4480) | a player's unexplained deviations persist all game | the strongest dependence in the residuals; CA 1.33 -> **0.87**, CF 1.21 -> 0.92, RCA -0.5 in proxy C (scored) | CG **+0.42** in proxy C (ratio -0.011 at 0.11), flat in B; RCD 0.83 -> 1.27; #158's algebra (independent dispositions scale both SDs) holds unless the switch rule sorts them, which it barely does (Note 9) | dropped as a CG mechanism; **the natural next experiment if CA is ever the target** |
| 3 | Corner-inflated emission (atoms at 0 and 20 only) | all-or-nothing is a discrete choice | test CE 2.3101 -> 2.0542; RCA -1.18 (-> ~2.3), RCB -0.42, CA -0.31 in proxy C | CG -0.004 (C) / -0.007 (B): neutral; RCA lands short of the edge; RCD +0.60 | superseded by 4 |
| 4 | **Status-quo + corner-inflated emission (atoms at prev, 0, 20)** | repetition and the corners are discrete choices | test CE 2.3101 -> **1.8855**; RCA **-1.45**, RCB -0.33, CA -0.22, CF -0.10, **CG -0.08** (the only right-signed CG move measured), mean -0.071, closed-loop P(c = prev) 0.167 -> 0.395 (human 0.44) | both declared landings at the edge (RCA ~2.06, CG ~2.00); RCC +0.21, RCD +0.13; P(c = 0) overshoots to 0.136 | **declared** |
| — | Own-group size (`1/n`, one-hot) | payoff-correct singleton level | — | **PR #175 `[FAIL]` today on this stack**: RCA 3.51 -> 3.78, CG +0.06, RCD lost its band | not retried |

## 2. Plan

Steps for the orchestrator to validate (§2 targets, §5 legality, §8 frozen
surface) and tag. Paths are relative to the worktree
`.claude/worktrees/contribution-inflated-gmlp`. Local Python is the main
checkout's venv with the worktree's source first on the path
(`PYTHONPATH=$PWD/src /Users/ertuerkan/Desktop/algorithmic-institutions/.venv/bin/python …`;
`uv sync` cannot run in a worktree, PR #174 note 9); `mkdir -p data/baselines`
first (gitignored CV output dir). `AI_REMOTE_DIR='~/autoresearch/contribution-inflated-gmlp'`
on every `simulate_cluster.sh` / `fetch_cluster.sh` / `remote_test.sh` call,
and `squeue -u certuer` over a live SSH tunnel before any syncing call. Lint
(`black`, `flake8 --max-line-length=88 --extend-ignore=E203,W503` on `src/`)
once per step before staging. Commits map to steps.

1. **The inflated estimator** `[Opus]` — `scripts/baselines/gaussian_regressor.py`
   (existing), new class `InflatedGaussianMLPRegressor(GaussianMLPRegressor)`.
   Constructor adds `atoms` (tuple over `{"prev", "0", "20"}`, default all
   three) and `prev_index` (position of `prev_contribution` in the feature
   list); `_make_net` builds `Linear(d, hidden) -> tanh -> Linear(hidden,
   2 + len(atoms))` with the incumbent's warm start (zero output weights,
   mu bias = mean(y), log-sigma bias = log std(y)) and atom-logit biases at
   -3. `log_probs(Z)` returns `[N, 21]`: the binned Gaussian body
   (`binned_logloss`'s convention — unit bins, tails folded into 0 and 20;
   factor it out of `binned_logloss` into a shared torch helper) weighted by
   `pi_body`, plus the atoms; the `prev` atom sits at
   `clip(rint(Z[:, prev_index] * scale + mean), 0, 20)` where `(mean,
   scale)` is the standardiser's affine map for that column, recovered
   exactly at fit time from the raw column passed as `fit(Z, y,
   prev=None)` (a `None` raises: the estimator refuses to fit without it).
   `fit` minimises the mean 21-way cross-entropy with Adam at the same
   (lr, wd, epochs, seed) knobs; `predict_proba(Z)` returns the
   probabilities, `classes_ = np.arange(21)` (so the adapter's
   `_class_probs` works unchanged), `predict` / `predict_std` return the
   body's mu / sigma (diagnostics only), `nll(Z, y)` returns the cross-
   entropy (the model's primary metric). New tests
   `tests/baselines/test_inflated_gmlp.py`: rows sum to one; the prev atom
   moves with the prev column; `atoms=()` reproduces a binned Gaussian
   fitted by the same objective; a synthetic panel with a planted 40 %
   exact-repeat mass is recovered to >= 0.30 by the fitted `pi_rep`; joblib
   round-trip gives `np.array_equal` `predict_proba`; fitting without `prev`
   raises.

2. **Registry and CV plumbing** `[Opus]` — `scripts/baselines/baseline_models.py`
   (existing): `_SPEC["gaussian_mlp_inflated"] = {hidden, weight_decay, lr,
   epochs, atoms}` with `atoms` a griddable string (`"prev"`, `"prev,20"`,
   `"prev,0,20"`; default `"prev,0,20"`), `_METRIC` `"ce"`,
   `resolve_model` accepting it for continuous targets, `build_model`
   constructing it with `prev_index`, and a `NEEDS_PREV = ("gaussian_mlp_inflated",)`
   constant; `predict_scores` returns `(ce, ce)` for it from
   `predict_proba` (its primary metric *is* the 21-way CE, so `show_ce`
   reports the same number), `floor_score` the marginal 21-way histogram
   floor (the multinomial branch). `scripts/baselines/run_baseline_cv.py`
   (existing): `_init` additionally receives the pool column index of
   `prev_contribution` (`prep["col_of"]["prev_contribution"]`), `_score`
   passes `prev=Xtr[:, pos]` to `fit` for models in `NEEDS_PREV`, where
   `pos` is that column's position inside the task's `cols` (a task without
   `prev_contribution` is a config error for this model — assert).
   `scripts/baselines/inspect_best_model.py` (existing): `_fit` passes the
   raw prev column the same way; `save_best` writes `model:
   gaussian_mlp_inflated`, the `atoms` string, `prev_index`, and
   `test_logloss_binned` = the test CE (the same key the incumbent carries,
   so the comparison is one number against one number). Existing
   `tests/baselines/test_gaussian_mlp.py` stays green (no behaviour change
   for the existing models — verify by running it); add to
   `test_inflated_gmlp.py` a registry test (settings expand to the three
   atom sets; an unknown atom string is rejected).

3. **Training config** `[Sonnet]` — new
   `configs/training/baselines/contribution/gaussian_mlp_inflated.yml`:
   `data` block byte-identical to `gaussian_mlp_v2.yml` (train file,
   `exclude_flipped: True`, `mask: contribution_valid`, `switch_every: 4`)
   except `model: gaussian_mlp_inflated`; `cv` identical except `output:
   data/baselines/gaussian_mlp_inflated_cv.csv` (`show_ce: true` kept);
   `setting` pinned to scalars `hidden: 8`, `weight_decay: 0.0003`, `lr:
   0.01`, `epochs: 1000` plus the one griddable knob `atoms: ["prev",
   "prev,20", "prev,0,20"]`; `blocks` reduced to one block `B_declared`
   with one set — the incumbent's seven features verbatim (so
   `enumerate_feature_sets` yields the floor and the set). Header comment:
   the Declaration's one-sentence rationale and the scratch numbers (test CE
   2.3101 / 2.0542 / 1.8855 for incumbent / corner-only / three-atom).

4. **Fit, save, F1 (stop-gate)** `[Sonnet]` — `run_baseline_cv.py <step-3 config>`
   (expected: the three-atom row ranks first on CV CE, then `prev,20`, then
   `prev`; the floor last); `inspect_best_model.py
   data/baselines/gaussian_mlp_inflated_cv.csv --config <step-3 config>
   --save-best --name contribution_gaussian_mlp_inflated_best.joblib`.
   Assert the saved bundle's `features` are the seven (a floor win is a
   stop, PR #175's amendment), record the selected `atoms`. **F1:** TEST
   21-way CE **< 2.3101098745482482** (the incumbent's
   `test_logloss_binned`; the scratch fit gives 1.8855, so a failure is a
   pipeline bug, not a result — stop and debug). Also print, on the test
   rows, the implied P(c = prev) (scratch 0.398 vs realised 0.503),
   P(20 | prev 20) (0.677 vs 0.851), P(c = 0) and P(c = 20) (0.063 / 0.168
   vs 0.057 / 0.220), and the mean fitted `pi_rep`, `pi_0`, `pi_20`. Commit
   the bundle (plain ~5 KB binary — `.gitattributes` does not LFS-track
   `*.joblib`) and its sha256 in the Notes.

5. **Adapter: the discrete emission under the group copula** `[Opus]` —
   `src/aimanager/simulation/linear_ah.py` (existing). (a) The
   `copula_rho_p / copula_rho_t` gate in `__init__` (line ~125) accepts
   `model_type in ("gaussian", "gaussian_mlp", "gaussian_mlp_inflated")`.
   (b) `_sample_levels`: for `gaussian_mlp_inflated` take the multinomial
   branch (`_class_probs` -> `th.multinomial`; `sample=False` -> argmax) —
   `_class_probs` works because `classes_` is `arange(21)` and
   `temperature` is 1. (c) `_sample_levels_gaussian_copula`: build `z`
   exactly as today — the 3n float64 draws in the fixed order `zu, zv,
   eps`, the first-member `pick` map, `setdefault` on `_copula_z`, the
   arrival-group rule — then branch on the model type: Gaussian models keep
   `clip(rint(mu + sd * z))` verbatim; the inflated model computes `P =
   self._class_probs(Xs, n_levels)`, `u = ndtr(z)` and inverts the row CDF
   with `th.searchsorted(cumsum(P))` clamped to `[0, n_levels - 1]` (the
   `_sample_levels_copula` idiom), `sample=False` -> argmax with no RNG.
   Docstring gains the discrete-inversion convention. **Bit-identity
   requirement:** every existing Gaussian bundle produces byte-identical
   output and RNG consumption — `tests/baselines/test_contribution_group_copula.py`
   and `test_punishment_copula.py` stay green unmodified. New tests in
   `test_contribution_group_copula.py` with an inflated toy bundle
   (`classes_`, `predict_proba`, atoms present): marginal preservation
   under the copula against the independent sampler's histogram within the
   in-test noise floor (PR #170 note 21's calibration); persistent latent
   constant within an episode and redrawn at `t == 0`; 3n draws whatever
   the weights (the `rho_t == 0.0` stream check); replayed-`z` correlation
   recovery of `rho_p` from the levels' normal scores; a switcher draws
   from the receiving group's latent; and PR #170 note 22's two mutations
   (`setdefault` -> assignment; conditional `zv` draw) must each fail at
   least one test. Raven: `AI_REMOTE_DIR=… scripts/remote_test.sh -- src/ -v
   --tb=short` for `test_linear_manager` (the only PyG test touching
   `linear_ah.py`); the `test_eval_*` failures on an isolated dir are the
   known `plots/` exclusion, not regressions.

6. **Dose re-estimation on the new trunk (F2 stop-gate)** `[Opus]` —
   `scripts/baselines/contribution_gmlp_copula_rho.py` (existing):
   generalise the marginal — `score_bundle` returns `P =
   estimator.predict_proba(Xs)` when `bundle["model"] == "gaussian_mlp_inflated"`
   and `bin_probs(mu, sigma)` otherwise (the estimator, bootstrap,
   round-trip and power arms consume `P` and are untouched); add `--bundle`,
   `--config`, `--out` with defaults equal to today's constants (PR #175's
   step 4 did exactly this on its branch — port it if byte-compatible, else
   re-implement; **never run the script with defaults**, that would
   overwrite the parent's committed sidecar); the round-trip panels must be
   sampled through the adapter's own step-5 inversion path (they call
   `LinearAHAdapter` already). Run with `--bundle
   artifacts/baselines/contribution_gaussian_mlp_inflated_best.joblib
   --config <step-3 config> --out
   artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.params.json
   --write-params`. Record `rho_total`, CI, SE, LR, `rho_lag1` (provenance
   only), rows **7457** and pairs **15090** (both must reproduce), censored
   share, round-trip verdict (tol 0.02) and the power arm. **STOP-GATE
   (F2):** a CI including 0 ends the experiment as a calibration-only
   `[FAIL]` (no stamp, no sim). Expected `rho_total` ~0.043 (scratch
   0.0433). Commit the sidecar.

7. **Stamp** `[Sonnet]` — `scripts/baselines/stamp_contribution_group_copula.py`
   (existing): add `--base`, `--params`, `--out` (defaults = today's
   constants), make bundle-vs-sidecar sha256 the primary assert (the
   literal `EXPECTED_BASE_SHA256` applies only under the default `--base`),
   replace the literal `rho_p` assert by `params["rho_total"]`, and make
   `check_predict_bit_identical` compare `predict_proba` for the inflated
   model (it compares `predict` / `predict_std` today). Run to write
   `artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.joblib`;
   all checks PASS (identical objects, exact `NEW_KEYS` manifest, reload
   bit-identity, `sample=False` adapter equivalence over the fixed 6-round
   switching sequence, rho read-back, sha256). Commit the bundle; sha256 in
   the Notes (re-checked on Raven in step 10).

8. **Pre-sim diagnostic — the scored schedule-matched proxy (never a
   gate)** `[Opus]` — new `scripts/baselines/gmlp_inflated_preflight.py`: roll the
   stamped candidate and the parent's stamped incumbent through the real
   punisher along the parent parquet's per-episode `agent_group` schedule
   (`plots/simulation/23_2g8a_kexo_…/per_round.parquet`; 100 x 24, seed 42,
   env round order contribution -> punishment -> per-group common good ->
   `prev_*` shift, every agent valid), write each as a per-round frame,
   load through `convert.load_sim` and score both arms against
   `convert.load_human` with `scoring.score_all(n_repeats=500, seed=42)`
   (read-only imports of the frozen suite). Print the 21-row table (real
   parent | incumbent arm | candidate arm | delta | predicted real), the
   arm means, and the closed-loop U1-U4 quantities. Record the table as a
   Note **before** step 10 is submitted. The Declaration's expectations
   (RCA ~2.06, CG ~2.00, mean ~1.15) are what this step re-derives from the
   committed bundle; a materially different table is recorded, not acted
   on. **The experiment proceeds to the simulation whatever this prints.**

9. **Sim configs, control and candidate** `[Sonnet]` — two new files under
   `configs/simulation/manager_testing/`, each a byte-copy of the parent's
   `23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`:
   (a) **control**
   `23_2g8a_inflctl_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
   with exactly two edits (`output_dir`, `figure_name` -> the `inflctl`
   name) — the parent's own stack re-simulated on this branch's code, which
   must reproduce the parent's `per_round.parquet` **bit for bit**
   (`3cb8b3d7…fef3f`): the proof that step 5's adapter branch is inert for a
   Gaussian bundle and what licenses judging against the parent's committed
   `scores.csv`; (b) **candidate**
   `23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
   with exactly three edits (`contribution_model` -> the step-7 bundle,
   `output_dir`, `figure_name`). Both parse under `evaluation_sweep.py`'s
   `_self_(\w+?)_contr_(\w+?)_switch$`. Verify by `diff` (2 / 3 changed
   lines; seed 42, 100 episodes, 24 rounds, `switch_every 4`, single
   pairing, `save_per_round: true` byte-identical).

10. **Raven: control then candidate, isolated** `[Sonnet]` — `squeue -u certuer`; one
    syncing call `AI_REMOTE_DIR='~/autoresearch/contribution-inflated-gmlp'
    scripts/simulate_cluster.sh <control config>` (ships `artifacts/`
    including the committed step-7 bundle; nothing exists remotely before
    this call, so PR #173's `rsync --delete` hazard does not arise), then
    `--no-sync` for the candidate; wait with one blocking `ssh raven 'while
    squeue -j <ids> -h | grep -q .; do sleep 30; done; sacct -j <ids>
    --format=JobID,State,ExitCode -n'`. From each log: the PROVENANCE line
    names the shared venv's interpreter and `aimanager.__file__` under
    `~/autoresearch/contribution-inflated-gmlp/src/`;
    `algorithmic-institutions/src` absent; remote sha256 of the three slot
    artifacts (switch `28dd4b40…`, punisher `9e3cf677…` shared; contribution
    `da42031a…` for the control, step 7's hash for the candidate).
    **Activation check:** the candidate parquet must **differ** from
    `3cb8b3d7…fef3f`. One draw each, seed 42, no re-runs.

11. **Fetch, evaluate, diagnose** `[Opus]` — `AI_REMOTE_DIR=… scripts/fetch_cluster.sh
    plots/simulation/<control dir>` and `<candidate dir>`; the control
    reproducing `3cb8b3d7…fef3f` is a **gate**: if it does not, stop and
    escalate — no verdict may be issued from this run (PR #175's
    amendment). Then `python -m aimanager evaluate <candidate config>`.
    Fill the Results row with RCA, CG, the mean, rows <= 1, and the guards
    RCB / CA / CF / CD / RCC / RCD / CB / PA / PB / SC / SA / SB / RSA
    exactly as computed. Then the read-only diagnostic (an uncommitted
    scratchpad script importing `convert.load_human` / `load_sim` and
    `metrics.ResponseMetrics` only) on human, parent and candidate: U1-U4
    in numbers; RCA's per-round-type EMDs against the parent's (0.9065 /
    1.0920 / 1.2959 / 1.0590, PR #175's table); the CG decomposition of Note
    2 (ratio, SDs, group-mean tails, by cell size, by round third, by
    tenure since regrouping) — so the CG reading can be tied to the
    mechanism or to noise; contribution by group size and per-capita common
    good by size (the PR #175 knock-on set, reported, never claimed); and
    the step-8 proxy table against the realised scores (the calibration
    record for the next agent). Commit sim outputs and evaluation only.

12. **Verdict, log, PR, clean-up** `[Opus]` (orchestrator) — §2 on the single
    evaluation, no second stage: `[SUCCESS]` iff (RCA < 2.0 **or** CG < 2.0)
    **and** mean <= 1.3449481666413449; otherwise `[FAIL]`. State U1-U4 in
    plain words whatever the title; a CG crossing with U3 failing is
    reported as unclaimed noise and the verdict rests on RCA alone.
    Complete Results and Notes (the atom set selected, the F1/F2 numbers,
    the proxy-vs-realised table, the guard outcomes, collateral `+` / `-`).
    `gh pr create --base auto/switch-exodus-k-onehot`, body Hypothesis /
    Results / Collateral (§9.7), stating up front that the diff shows only
    this experiment's change over PR #174 and that the maintainer's CG
    target was measured to have no headroom in this slot (Note 9). Delete
    `~/autoresearch/contribution-inflated-gmlp` on Raven when the PR
    closes, not at open.


### Orchestrator validation (Opus, 2026-09-15)

The plan is **validated and released for implementation**, with the rulings
and the one amendment below. Implementers are tagged on each step: `[Opus]`
where the step is mathematical, RNG-sensitive or analytical, `[Sonnet]` where
it is mechanical execution against an explicit spec (§9 Roles).

**Ruling 1 — the status-quo atom is legal (§5).** Fable asked for an explicit
ruling on whether probability mass at `c = prev_contribution` is "a feature
engineered at a metric's definition rather than at behavior", given that RCA
measures the distribution of exactly that change. It is legal, on three
grounds, and the decisive one is the third:

- RCA is an EMD over the change distribution with **no bins and no strata
  boundaries**, so the §5 example (a term keyed to a bin edge) does not apply;
  and the atom is unconditional on RCA's four round types, so it is not keyed
  to its stratification either.
- The quantity it encodes is measured in the human data independently of any
  metric — exact repetition in 44 % of rounds (79 % at 20, 69 % at 0) against
  the emission's 17 % — and a binned Gaussian **structurally cannot** place a
  point mass. This is a deficiency of the emission family, not of its
  conditioning.
- **It is selected and dosed by held-out likelihood, never by the evaluation.**
  That is the test that separates a modelling fix from a gamed one: a
  metric-keyed term buys score without buying fit, whereas this buys 18 % of
  test 21-way cross-entropy (2.3101 -> 1.8855), the best fit in the linear
  lineage. F1 (step 4) enforces that ordering before any simulation exists.

**Ruling 2 — two declared targets, both binding.** CG is the maintainer's
designated row and stays declared; RCA is declared alongside it as the
mechanism's home row. §2 requires a band upgrade on *a* row the hypothesis
declares, so either crossing satisfies gate 1. Pre-declaration binds in both
directions: no third row may be claimed afterwards, and **U3 governs CG** — a
CG crossing while the group-mean tail `P(mean >= 18)` has not risen from
0.0307 toward the human 0.0925 is reported as unclaimed noise, with the
verdict resting on RCA alone (step 12 already states this; it is upheld).

**Ruling 3 — the change is one change (§4).** The emission family is the
change. Re-estimating the copula dose on the new trunk (step 6) is not a
second one: a dose fitted to the old trunk's residuals is simply the wrong
number for a new marginal, and carrying it over is the incoherence PR #166
diagnosed. Selecting the atom subset by CV likelihood (step 4) is the
hyperparameter search §5 explicitly permits.

**Ruling 4 — shared-code edit approved (§4).** Step 5 edits
`src/aimanager/simulation/linear_ah.py`, which is shared simulation code, but
additively: a new branch for a new bundle type in the contribution sampler,
with the existing Gaussian path required to be bit-identical in output *and*
RNG consumption. That requirement is proven twice — by the mutation-checked
unit tests in step 5 and, decisively, by step 9/10's control run having to
reproduce the parent's `per_round.parquet` hash `3cb8b3d7…fef3f`. Step 11's
control gate stands: no verdict may be issued if the control does not
reproduce.

**Ruling 5 — frozen surface clear (§8).** Steps 8 and 11 import
`evaluation_suite` read-only and write nothing into it; step 11's diagnostic
stays an uncommitted scratchpad script. Scoring parameters (500 repeats,
master seed 42) and the 23-family protocol (100 episodes, 24 rounds, seed 42,
`switch_every: 4`, single pairing) are untouched, enforced by step 9's
byte-copy discipline and its 2-line / 3-line diffs.

**Amendment A — the atom set freezes at step 4.** The selected `atoms` value
is fixed by CV likelihood at step 4 and **may not be revisited after step 8**.
Step 8 scores the candidate against the evaluation suite; allowing it to feed
back into variant selection would turn a declared diagnostic into a back-door
selector against the metrics, which §5 forbids. Step 8 remains what it says
it is: recorded, never acted on, and **the simulation runs whatever it
prints**.

**Amendment B — step 6's overwrite hazard is a hard requirement.** The dose
script must never be run with its default `--out`: that would overwrite the
parent's committed sidecar. Every invocation passes `--bundle`, `--config` and
`--out` explicitly.

**Recorded weakness, accepted.** The modal outcome of this experiment is a
`[FAIL]` with RCA in the low 2s, a −18 % likelihood gain and the mean around
1.15 — the strongest contribution result since PR #170 but not a crossing.
Both declared rows sit a coin flip from their edges. That is accepted going
in, and the log will report it as measured either way.

**The finding that outlives the verdict** (Fable's Note 9, upheld here): CG's
residual in this stack is **not a contribution-slot deficit**. It is set at
regrouping — rounds 0–3 already match the human (gap −0.0035) while rounds
>= 4 carry the whole 0.055 — and it is composition plus sorting, with the
human sorting gain at arrival +0.034 against the sim's +0.010 and
`P(move | far above own group)` 0.349 against 0.291. The lever is the switch
slot's leaving signal, not the contributor's emission. Every contribution-side
mechanism measured is CG-inert except this one, at roughly +0.002 of ratio.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|

## 4. Notes

1. (Fable, opening) **Two experiments already stand on this exact stack and
   both are read.** PR #174's log pre-declared CG's return to `2-5` as the
   price of SC (fully-merged share 0.214 -> 0.186) and its notes 20-21 point
   the successor at the contribution slot's inverted size -> per-capita
   common-good map. PR #175 (`auto/contribution-size-onehot`, 2026-09-15)
   ran that successor as `1/n`: the mechanism installed (singleton mean 7.29
   -> 11.61, the CG map flipped sign, SC 0.98 -> 0.82 unclaimed) but RCA
   3.507 -> 3.779, CA flat, RCD 0.73 -> 1.70 and CG 2.079 -> 2.143; its
   note 9 names "a distributional head that can place an atom at dc = 0" as
   the only thing in view for RCA. This experiment is that head, and it is
   not a size feature.
2. (Fable, research) **Where the CG deficit lives, from
   `per_round.parquet` and the human frame (`convert`, one copy per game).**
   Human SD(group means) 5.3577 / SD(individual) 6.3179 = 0.84802; parent
   4.4807 / 5.6478 = **0.79336** (point-estimate score 2.067 against the
   scored 2.079). By cell size (share of cells, var of group means, human
   vs parent): n = 1 0.109 / 0.107, **64.1 vs 34.3** (sd 8.0 vs 5.9; human
   singletons 23 % at 0 and 31 % at 20, the sim's 16 % / 5 %); n = 2 30.4
   vs 24.2; n = 3 26.3 vs 19.9; n = 4 20.6 vs 14.2; n = 5 **30.9 vs 13.9**;
   n = 6-8 match or exceed (23.1 / 17.4, 20.2 / 17.5, 14.5 / 15.9).
   Excluding singleton cells the ratio gap is 0.0397 of the 0.0547; the
   size composition itself is irrelevant (reweighting the parent's per-size
   variances to the human size shares moves the ratio 0.7934 -> 0.7940).
   Group-mean tails: P(mean >= 18) human **0.0925** vs sim **0.0307**,
   P(mean <= 2) 0.079 vs 0.043, while the mean within-cell sd (n >= 2) is
   identical (3.914 vs 3.913) — the sim's groups scatter internally like
   human groups but never reach the corners together. By round third the
   ratio is human 0.730 / 0.878 / 0.897 vs sim 0.710 / 0.820 / 0.847. The
   decisive split: **rounds 0-3, before any regrouping, human 0.6145 vs
   sim 0.6180 (gap -0.0035)**; rounds >= 4, 0.8817 vs 0.8268; by tenure
   since the group's composition last changed, 0-1: 0.840 / 0.793, 2-3:
   0.898 / 0.847, >= 4: 0.987 / 0.904. The contributor's group-level
   dynamics match human groups exactly until the first regrouping.
3. (Fable, research) **The stamped dose is delivered, so PR #166's
   composition lesson does not apply as dilution.** Teacher-forcing the
   stamped bundle on the sim's own histories (features rebuilt with the
   adapter's `_pool_from_arrays`): within-cell censored pairwise MLE
   **0.0449** against the stamped 0.0438 (moment 0.033), by thirds 0.044 /
   0.051 / 0.040 — flat; lag-1 cross-member MLE 0.026. On all 50 human
   episodes the same estimator gives 0.088 (thirds 0.025 / 0.016 / 0.103;
   the train-split value the dose came from is 0.044 with thirds 0.027 /
   0.036 / 0.077 — the test episodes carry much more dependence, and late
   rounds carry most of it), co-censoring both-at-20 **3.69x** independence
   and both-at-0 2.50x against the sim's 1.70x / 1.11x, and a shared-scale
   signal the sim lacks (within-cell corr of squared residuals +0.018 /
   corr of |r| +0.048 vs -0.004 / +0.001; dispersion of within-cell sd 1.16x
   an independent-shuffle null vs the sim's 0.95x). The dependence is
   right in size and wrong in shape — it lives at the corners.
4. (Fable, research) **The corner and status-quo behaviour, human vs the
   model's own conditional.** P(c = 20 | prev 20): human realised 0.789,
   model-implied 0.400 (sim realised 0.468); with the group's prior mean
   >= 18 it is 0.950 vs implied 0.472; P(c = 0 | prev 0) 0.610 vs 0.362;
   with the group <= 2, 0.809 vs 0.433. Overall P(c = 20) human 0.134 vs
   implied 0.069, P(c = 0) 0.094 vs 0.075. Exact repeats: human P(c = prev)
   **0.4395**, parent sim **0.1684**. Teacher-forced on human histories the
   model's independent one-shot spread ratio is 0.8105 vs the human 0.8481
   on the same rows (copula 0.0438 adds +0.008; forcing the human corner
   repeat rates onto the model's draws adds +0.011) — the conditional
   itself under-disperses group means, and the shortfall is at the corners
   where mu shrinks toward the middle (mu 18.0 at prev 20 where humans give
   20 with 0.79).
5. (Fable, research) **Proxy B (PR #170's rollout: real bundles, fixed 4/4
   groups, no switching, seed 42) — the dependence-shape family is
   CG-inert.** Incumbent rho 0: 0.7216; rho 0.0438: **0.8275** (thirds
   0.698 / 0.861 / 0.898 against human 0.730 / 0.878 / 0.897 — with fixed
   groups the stack reproduces the human spread trajectory, and sits above
   the `<= 1` edge 0.8216); round-growing rho at the thirds MLE (0.0271 /
   0.0356 / 0.0769): 0.8186 (**-0.009**); t-copula with a shared per-group
   chi-square scale, nu = 4: 0.8220 (**-0.006**); corner atoms at rho
   0.0438 / re-estimated 0.0473: 0.8156 / 0.8209 (-0.012 / -0.007); a
   per-agent persistent component `rho_a` 0.03 / 0.06 / 0.10 on top of the
   group latent: 0.8292 / 0.8289 / 0.8252 (flat), `rho_a` 0.06 alone
   0.7380. Every arm raises SD(group means) and SD(individual) together.
6. (Fable, research) **The per-agent disposition is real and large, and it
   is a CA mechanism, not a CG one.** Censored pairwise MLE on same-agent
   cross-round pairs of the incumbent's residuals (train split): lag 1
   0.026 (the corner bounce), lags 2-12 0.09-0.29, **lags >= 2 pooled
   0.159 (LR 4480, n 76,994)**; a joint grid over (rho_g, rho_a) with
   within-cell, same-agent-same-group and same-agent-other-group pair sets
   gives **(0.04, 0.11)**. PR #170 set this aside as "cannot move CG"
   (#158); measured here in proxy C (Note 8's machinery): CA 1.334 ->
   **0.872**, CF 1.212 -> 0.915, RCA -0.50, but CG 1.790 -> **2.206** and
   RCD 0.828 -> 1.274; ratio -0.011 at `rho_a` 0.11. #158's algebra holds:
   independent dispositions scale both SDs, and they only raise the ratio
   if the switch rule sorts them into groups, which it barely does (Note
   9). Recorded as the natural next experiment for CA (1.609, `std_diff`
   -1.59), not run here.
7. (Fable, research) **The inflated emission, prototyped at the incumbent's
   setting on the train split (7 s per fit), scored on the locked test
   split.** TEST 21-way CE: incumbent (continuous-NLL fit) **2.3101**; the
   same Gaussian fitted on the binned objective 2.1626; corner atoms (0,
   20) **2.0542**; corner + status-quo atoms (0, 20, prev) **1.8855** —
   -18 %, against the reference GNN contributor's ~1.99 and the xgb's 1.834
   (PR #154), the best fit this lineage has produced. Implied on the test
   rows: P(c = prev) 0.398 (realised 0.503 on that split), P(20 | prev 20)
   0.677 (0.851), P(c = 0) 0.063 (0.057), P(c = 20) 0.168 (0.220); the
   corner-only model's P(20 | prev 20) was 0.766 with mean `pi_20 | prev 20`
   0.618. Within-cell censored MLE against the new trunk's own marginal:
   **0.0433** (three-atom) / 0.0473 (corner-only), against the incumbent's
   0.0438 — the shared dependence survives the emission change essentially
   unchanged, as it should if the atoms explain the corners' marginal and
   the copula their co-occurrence.
8. (Fable, research) **Proxy C — the parent's own regrouping schedule,
   scored through the suite.** Rolling the real contribution and punisher
   bundles along each parent episode's actual `agent_group` timeline (so
   singletons, sizes and regroupings are present; sorting fixed), then
   `score_all` at 500 repeats, seed 42, incumbent arm vs candidate arms;
   the incumbent arm lands at ratio 0.8020 (real 0.7934) and scores CA 1.334
   / CG 1.790 / CF 1.212 / RCA 3.412 / RCB 2.237 / RCD 0.828 (real 1.609 /
   2.079 / 1.345 / 3.507 / 1.910 / 0.732; S rows identical by construction;
   RSA 2.71 vs 1.32 is the fixed-schedule artefact). Three-atom candidate
   (rho 0.0433): **RCA 1.966 (-1.446), RCB 1.910 (-0.326), CA 1.113
   (-0.221), CF 1.116 (-0.096), CG 1.707 (-0.082), CD 0.774 (-0.051); RCC
   1.354 (+0.208), RCD 0.954 (+0.126), CB 0.840 (+0.073), PA 0.733
   (+0.070), PB 1.014 (+0.042); mean 1.2568 -> 1.1854 (-0.0714)**;
   closed-loop P(c = prev) 0.167 -> 0.395, P(c = 20) 0.071 -> 0.126, P(c =
   0) 0.074 -> 0.136, P(group mean >= 18) 0.038 -> 0.070. Corner-only arm:
   RCA -1.175, RCB -0.417, CA -0.313, CG +0.151, RCD +0.595, mean -0.037.
   `rho(t)` arm: nothing moves (CG +0.108). Ratios only, for the record:
   incumbent 0.8020; + `rho_a` 0.06 / 0.11 / 0.16: 0.7977 / 0.7908 /
   0.7858; corner atoms 0.7978; `rho(t)` 0.7989; corner atoms + `rho_a`
   0.11: 0.7842. **Calibration for the reader:** PR #170's no-switching
   proxy over-predicted its delta by 30 %; proxy C is schedule-matched and
   its incumbent arm sits 0.1-0.3 from the real scores per row, which is
   the honest uncertainty on every delta above — the same size as both
   declared rows' distance to their edges.
9. (Fable, research) **Why CG has no headroom in this slot — the residual
   is composition and sorting after regrouping, and that is the switch
   rule's.** At each arrival round the spread ratio of members' *previous*
   contributions under the new grouping is human 0.798 against a
   random-movers null (same mover counts) of 0.764 — a sorting gain of
   **+0.034**; the sim's is 0.740 against 0.730, **+0.010**. Who leaves:
   human P(move | own prev minus own group's mean in (-30,-5] / (-5,-1] /
   (-1,1] / (1,5] / (5,30]) = 0.333 / 0.315 / 0.239 / 0.295 / **0.349** —
   both kinds of misfit leave more; the sim 0.340 / 0.293 / 0.279 / 0.281 /
   **0.291** — only the punished low misfit does, the exploited high
   contributor does not (the switch trunk reads common good and punishment,
   not own contribution against the group). Half of the human group-mean
   variance is composition: var(group mean) 32.1 of which var(mean of
   members' episode means) **21.8**; the sim 21.9 of which **11.4**; sd of
   participant means 5.37 vs 3.85. After arrival the two stacks re-diverge
   in parallel (|g0 - g1| by rounds since arrival human 5.11 / 5.83 / 6.20 /
   6.51, sim 4.03 / 4.63 / 5.00 / 5.32): the gap is set at arrival, not
   grown afterwards. Taken with Note 2 (gap zero before round 4) and Notes
   5 and 8 (every dependence-shape and emission change is CG-neutral or
   negative with the schedule fixed), the contribution slot cannot deliver
   the CG band on its own; the lever is in the switch slot — own
   contribution relative to the group as a leaving signal — with the
   contribution slot's persistent dispositions (Note 6) as the thing to
   sort. This is stated for the maintainer, not acted on here.
10. (Fable, opening) **Why the trained group latent (PR #159's mechanism)
    was not ported despite the hint.** Its estimand under teacher-forced
    marginal likelihood is the within-(episode, group) latent correlation,
    which the copula MLE already fixes at 0.044 and which Note 3 shows is
    delivered in the stack; what a trained latent could add over the copula
    is a state-dependent or scale pathway, and both proxies for those
    (round-growing rho, shared-scale t-copula) are CG-inert (Note 5). #159's
    38 % structural ceiling and its leaked likelihood gate would be inherited
    for no measured gain. The corner behaviour the human dependence actually
    shows (Note 3: co-censoring 3.7x) is an emission-shape fact, and the
    emission is where this experiment goes.
11. (Fable, opening) **What the maintainer should weigh before validation.**
    (a) Both declared crossings are coin flips by the proxy — RCA ~2.06 and
    CG ~2.00 against edges at 2.0 — so a `[FAIL]` with a -18 % likelihood
    gain, a mean around 1.15 and RCA in the low 2s is the modal outcome and
    would still be the strongest contribution-slot result on record after
    #170. (b) The maintainer's CG target is declared per assignment, with
    U3 as the honesty condition; the mechanism's home is RCA, and if the
    maintainer prefers a single declared row it should be RCA. (c) The
    emission changes marginals, so the R- and C-block guards are trade-offs,
    not bug detectors as in #170; RCC (+0.2) and the 0-atom's overshoot
    (CB / PA / PB) are the predicted costs. (d) The proxy holds the parent's
    regrouping fixed; the switch trunk will react to the changed
    contributions through per-capita common good (PR #175 moved SC by 0.16
    through that channel), and nothing here predicts the S rows.
12. (Fable, opening) **Timings measured today, local.** Incumbent-setting
    fit 0.71 s (PR #175); two-atom and three-atom prototype fits 6.9-10 s
    for 1000 epochs at hidden 8 on 7457 rows; proxy B arm ~30 s; proxy C
    arm ~50 s; `score_all` over 21 rows x 4 arms at 500 repeats 253 s;
    the parent's sims 2m39s each.
13. (Step 1, confirmed) **The estimator is in and green; one deliberate
    deviation from the step text, accepted.** The step said to factor the
    binned body out of `binned_logloss` into a shared *torch* helper, but
    `torch.special.ndtr` and `scipy.stats.norm.cdf` differ by 2.2e-16, so
    routing the existing loss through torch would have changed the
    incumbent's numbers. The convention is instead one documented pair —
    numpy `binned_probs` (which `binned_logloss` now delegates to) and its
    torch twin `binned_log_probs` — pinned together by a test. Verified by
    the orchestrator independently of the implementer: `binned_logloss` is
    **bit-identical to HEAD** (`==`, not `approx`) over 480 cases at
    `k_levels` 5 / 21 / 41 including the scalar-sigma path, with identical
    warning behaviour. `tests/baselines/` is **298 passed**, which includes
    all 31 pre-existing `test_gaussian_mlp.py` tests and the 18 new ones.
    Planted-mass recovery: fitted `pi_rep` **0.4003** against a planted
    0.40. Row sums max `|sum - 1|` **4.44e-16**. `atoms=()` degeneracy:
    `nll` 2.2781915897 vs `binned_logloss` 2.2781915904. Two hazard tests
    beyond the specified list — colliding atoms at a corner must **add**
    rather than overwrite, and the standardiser's affine map must
    round-trip to the raw integers (a wrong `prev_index` now hard-errors
    instead of silently misplacing the mass).
14. (Step 1, findings that bind later steps) Four, all recorded before
    step 2 was dispatched. (a) **Fit cost is ~14x the declared figure:**
    9.5 s at 10 threads, 15.5 s single-threaded at n=7000 against the
    incumbent's 0.68 s. `run_baseline_cv._score` pins `th.set_num_threads(1)`,
    so step 4's CV is ~3-7 min of core time, not Note 12's ~1.5 min — still
    far inside the §5 3x rule, but the declaration's figure was optimistic.
    (b) **`prev_index` is task-dependent** (the position of
    `prev_contribution` inside the task's `cols`, which changes per feature
    set) so it cannot live in `_SPEC`, which expands one grid for all
    tasks: step 2 threads it through `build_model` as a parameter, not a
    setting key. `predict_proba` needs only `Z` after fit. (c) **Step 6's
    dose estimator is only half emission-agnostic:** `score_bundle` returns
    `(Xs, mu, sigma)` and the caller builds `P = bin_probs(mu, sigma, K)`,
    which for an inflated bundle is the **body only** and silently wrong —
    it must come from `predict_proba`. The pairwise MLE consumes only `P`
    and carries over unchanged, but the moment diagnostic
    `r = (c - mu) / sigma` has no meaning under a mixture and must be
    replaced by the probit of the discrete CDF interval or dropped for this
    bundle. (d) **Step 5's adapter would silently take the Gaussian
    branch:** `linear_ah.py` dispatches on
    `model_type in ("gaussian", "gaussian_mlp")` and samples `mu + z*sigma`
    from `predict` / `predict_std`, which on this class are the *body's*
    parameters — a bundle reaching the sampler without the new branch runs
    the incumbent emission and raises no error. (e) Step 7's stamper
    verification asserts `predict` / `predict_std` bit-identity after
    reload, which still passes but no longer verifies the emission;
    `predict_proba` must join that check.
15. (Step 2, confirmed) **Registry and CV plumbing in; `tests/baselines/` is
    311 passed** (298 after step 1 + 13 new), `test_gaussian_mlp.py` 31
    passed unchanged, flake8 per-file counts identical to HEAD. The
    `prev_index` hazard of Note 14b is closed by one helper,
    `prev_position(model, ordering, key)`, used by both call sites so they
    cannot drift: the **pool** index `prep["col_of"]["prev_contribution"]`
    travels to the worker and never reaches the estimator, while the **task
    position** `cols.index(prev_col)` is what goes to
    `build_model(..., prev_index=pos)` and indexes the same matrix as the
    raw `Xtr[:, prev_pos]` handed to `fit`. In the real run the position is
    0; the smoke deliberately placed it at pool 3 / position 2 and it
    resolved correctly, and a mismatched index raises rather than fitting.
    The `_SPEC` defaults (hidden 32, wd 0.0, lr 0.05, epochs 500) mirror
    `gaussian_mlp`'s house defaults, **not** the incumbent's knobs, so
    step 3's config must pin all four explicitly and step 4 must assert
    them back off the saved bundle — an omitted key would silently train at
    different hyperparameters and make the F1 comparison a confound rather
    than a like-for-like test of the emission.
16. (Step 2, caught before it broke step 4) **A duplicate `ce` column.**
    `_METRIC` for this model is `"ce"`, and `main` both renames `mean_loss`
    to the metric name and separately writes a `ce` column under
    `show_ce` — so the CV CSV carried two columns named `ce` and `df[order]`
    listed it twice. Fixed at the row dict (skip the duplicate when
    `metric == "ce"`) and at the `order` list (key off `"ce_se"`). Gaussian
    CSVs are byte-unchanged. The floor is a real number, not a crash,
    because `floor_score` uses `ce_levels` (21) rather than `n_levels` (0
    for this continuous config): **floor 2.9214 CV / 2.7180 test**, well
    above the incumbent's 2.3101, so a floor win at step 4 would be an
    unambiguous stop. A 20-epoch end-to-end smoke on the real train split
    (rows 7457, matching the declaration) already orders the settings
    `prev,0,20 < prev,20 < prev < floor`.
17. (Step 3, confirmed) **Training config in, one file, nothing else
    touched.** `configs/training/baselines/contribution/gaussian_mlp_inflated.yml`
    differs from `gaussian_mlp_v2.yml` only in the header comment, the
    model name, the CV output path, a reworded `show_ce` comment, the
    `setting` block (the incumbent's grid pinned to its exact scalars 8 /
    0.0003 / 0.01 / 1000 plus the one griddable `atoms`), and `blocks`
    (the 15-set feature grid reduced to the single declared seven, in
    order). Dry expansion, no fitting: 3 settings, 2 feature sets (floor +
    `B_declared:s0`), **7457 rows**, `prev_contribution` at pool index 18
    and task position 0. Note 15's confound is closed — all four
    hyperparameters are pinned, so the emission is the only thing that
    differs from the incumbent.
18. (Step 3, a reported hazard that is not one) The implementer found that
    `prev_position` **raises** on the floor feature set and read it as a
    guard. It is neither a guard nor a crash: the runner never calls it
    there. `run_baseline_cv.py:134` guards with `if cols else None`, and
    `_score:110` returns the floor before any prev handling — which is why
    step 2's end-to-end smoke produced floor rows normally. Verified by the
    orchestrator directly in the code, not by re-running. Step 4's floor
    check therefore stands as the plan writes it: read `features` off the
    saved bundle, do not rely on an exception that will not be raised.
19. (Step 4, confirmed) **F1 passes decisively and the atom set is frozen
    at `prev,0,20`.** CV cross-entropy on the training split ranks exactly
    as the declaration predicted — `prev,0,20` **2.161016**, `prev,20`
    2.168084, `prev` 2.186107, floor 2.921355 (the three floor rows tie by
    construction) — so the selection is by likelihood alone and, per
    Amendment A, is not revisited again. Bundle
    `artifacts/baselines/contribution_gaussian_mlp_inflated_best.joblib`,
    sha256 **687755b6c9f746f03844dbeec991a3f437036d3cd29f6f0a0505fd7130580b0a**,
    4175 bytes, not LFS-tracked. **Test 21-way CE 1.8817206133537128**
    against the incumbent's 2.3101098745482482 (**-18.5 %**) and a test
    floor of 2.717958377177829. Read back off both bundles by the
    orchestrator independently of the implementer: same seven features in
    the same order, same `test_logloss_binned` key, and the candidate's
    stored knobs are hidden 8 / wd 0.0003 / lr 0.01 / epochs 1000 with
    `atoms ('prev','0','20')` and `prev_index 0` — so Note 15's confound is
    closed on the artifact itself, not merely in the config, and the
    emission is the only difference.
20. (Step 4, the mechanism's signature at fit time) Mean fitted mixture
    weights on the 1863 test rows: `pi_body` 0.6957, **`pi_prev` 0.2493**,
    `pi_0` 0.0108, `pi_20` 0.0442. Implied vs realised on those rows:
    P(c = prev) **0.398359 vs 0.502952**, P(20 | prev 20) 0.676935 vs
    0.850785 (n 382), P(c = 0) 0.063567 vs 0.057434, P(c = 20) 0.168887 vs
    0.219538 — every one within 0.001 of the scratch numbers the
    declaration pre-registered. The repeat mass moved from the incumbent's
    ~0.17 (Note 4) to 0.398 against the human 0.4395, so the mechanism is
    real but **still under-delivers the observed repeat rate by about
    0.10**. Recorded as a teacher-forced likelihood diagnostic only: it is
    not U1, which is a closed-loop quantity measured on the rollout at
    steps 8 and 11, and the two must not be conflated when the verdict is
    written.
21. (Orchestrator, checked while verifying step 4) The bundle pickles its
    estimator under the bare module name `gaussian_regressor`, so it
    unpickles only with `scripts/baselines` on `sys.path`. This is not a
    new hazard for the cluster run: `linear_ah.py:42` already inserts that
    directory, which is how every existing gaussian bundle loads.
22. (Step 5, confirmed) **The discrete inversion is in, and the silent-
    fallthrough hazard of Note 14d is closed by construction.** The three
    call sites now dispatch on four class-level family tuples
    (`_INFLATED`, `_CATEGORICAL`, `_GAUSSIAN`, `_HOMOSCEDASTIC`), and both
    `_sample_levels` and `_sample_levels_gaussian_copula` **raise** on an
    unregistered model type instead of falling through to `predict` /
    `predict_std` — which on this estimator are the mixture's body and
    would have run the incumbent emission without erroring. The inflated
    path is `c_i = F_i^{-1}(Phi(z_i))` over the row CDF of `_class_probs`,
    the multinomial punisher's own idiom; `sample=False` returns the modal
    level and draws nothing.
23. (Step 5, bit-identity — the property that licenses the verdict)
    Verified twice, independently. **Measured** by the implementer against
    `git show HEAD:...linear_ah.py` loaded as a second module: every
    `.joblib` in `artifacts/baselines/` x 3 seeds (42, 7, 12345) = 30
    case/seed pairs, over a 6-round episode whose schedule contains a
    switch and a collapse to one group, comparing the level array per
    round, the `_copula_z` store, and RNG **position** (a trailing
    `randn(1)` plus a `randn(3, float64)`, so a wrong draw count or dtype
    would show) — all identical, including the parent's own
    `contribution_gaussian_mlp_v2_group_copula` bundle at ρ_p = 0.0438.
    **Structurally** by the orchestrator, reading the diff: for a Gaussian
    bundle the only changes are membership tests over identical literal
    tuples, `mu`/`sd` moved inside an `elif` that runs no RNG, and a new
    branch that cannot fire — no RNG call moved, became conditional, or
    changed dtype. Step 10's control run should therefore reproduce
    `3cb8b3d7…fef3f`.
24. (Step 5, mutation testing) PR #170 Note 22's two mutations were each
    applied to the source, the suite run, and the file restored (verified
    by `diff -q`). **(a)** drawing `zv` only when `rho_t > 0`: 5 tests
    failed, including the candidate's own `rho_t = 0.0` shape. **(b)**
    `setdefault` -> plain assignment: 11 failed, including the
    switcher/receiving-group test. Both mutations are caught by new
    inflated tests, not only by the inherited Gaussian ones — the
    `rho_t = 0.0` configuration PR #170 left untested is now parametrised
    in every invariant test. `tests/baselines/` **326 passed** (311 + 15),
    `tests/` overall 510 passed, `test_punishment_copula.py` and
    `test_gaussian_mlp.py` green unmodified.
25. (Step 5, marginal preservation) In-test noise floor 0.01575, ~2.4x the
    Gaussian module's because the atoms concentrate mass and raise the
    binomial noise; 85 bins clear the >= 20-count filter. Worst analytic
    gaps 3.45 / 3.19 / 2.05 SE at (ρ_p, ρ_t) = (0.3, 0) / (0, 0.3) /
    (0.2, 0.2), i.e. 1.4-1.7x the floor. Replayed-z recovery is exact
    (`np.array_equal(levels, invert(P, z))`), and the **levels' normal
    scores recover only 0.60-0.65 of the latent correlation** against the
    Gaussian path's ~0.92 (note 21's ~8 % attenuation) — the atoms widen
    the CDF intervals, so the same stamped ρ_p delivers **less realised
    within-group correlation through a discrete emission than through a
    continuous one**. That is a prediction about the sim, recorded before
    it runs: the dose re-estimated at step 6 is fitted on the same
    emission, so it is the right number, but CG's realised lift may be
    smaller than #170's at an equal ρ. On the real artifact: max |bin freq
    − `predict_proba`| 0.00388 against a binomial SE of 0.00346 over
    20 000 draws, and P(level == prev) = 0.2497.
26. (Step 5, cluster and tooling) Raven `test_linear_manager` **1 passed**
    under `AI_REMOTE_DIR='~/autoresearch/contribution-inflated-gmlp'`;
    `squeue` showed one PENDING job in a *different* experiment's isolated
    dir, disjoint from the sync target, so no `rsync --delete` race. Two
    tooling facts for later steps: (a) `remote_test.sh` excludes
    `artifacts/` as well as `plots/`, so on a fresh isolated dir
    `test_linear_manager` fails with `FileNotFoundError` on a punisher
    bundle until `artifacts/baselines/` is rsynced in (done, additive, no
    `--delete`) — the known-benign list for this branch reads "`plots/`
    **and `artifacts/`** exclusions"; (b) `black src/` reformats two files
    unrelated to this experiment (`artificial_humans/train.py`,
    `rl_manager.py`) that are already non-Black at HEAD, so lint stays
    scoped to the files a step touches.
27. (Step 5, binding on step 6) The dose script's round-trip arm **must
    stamp `copula_rho_p` onto the bundle it hands `LinearAHAdapter`**.
    Without it the adapter takes `_sample_levels`, which is still the
    correct categorical law but consumes **one `th.multinomial` call
    instead of 3n `randn`** — so the inflated bundle's independent path is
    not RNG-comparable with a Gaussian bundle's. Either path samples the
    right distribution; only the stream differs.
28. (Step 6, confirmed) **F2 passes: the dose is
    `rho_total = 0.048443521435665396`**, 95 % CI
    [0.02675017796883691, 0.06748487570258777], SE 0.010272335023860522 —
    the CI excludes 0, so the experiment continues. LR 23.046568982113968.
    **rows 7457 and within-cell pairs 15090 both reproduce** (asserted),
    lag-1 pairs 28714; censored share 0.21577041705779804 (0.1026 at 0,
    0.1132 at 20); 1814 cells, 1608 with >= 2 members. Round trip **PASS**,
    max |bias| 0.007274625058280623 against tol 0.02. Power arm (0.03, 0)
    recovers lag-1 0.03262316498716052, so the falsifier reading is live.
    Falsifier `rho_lag1` 0.024927591372935834, CI
    [0.006301366612174943, 0.043892804851301225] — provenance only, never
    stamped. Sidecar
    `artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.params.json`,
    sha256 `5ff6f324b30b31ac5c6a7775269ce2d7fe4799a5b5af2f83923ef9c2a0326176`,
    carrying `rho_p = rho_total`, `rho_t = 0.0` and
    `base_bundle_sha256 687755b6…580b0a` (Note 19's bundle). Amendment B
    held: every invocation passed `--bundle`, `--config` and `--out`
    explicitly, and the parent's sidecar hashes
    `6f042d88abe27e8dcc2ad39a4f3009408bd2ef0fe89644a2a7ee8a5e44ff18e4`,
    identical to `git show HEAD:` of the same file (checked by the
    orchestrator, not only reported).
29. (Step 6, a pre-registered number that moved) **The measured dose is
    0.0484, not the declaration's ~0.043** (scratch 0.0433) — +0.0051,
    about 0.5 SE, and above the parent's stamped 0.04378520865574197.
    Nothing was tuned: it is the single MLE on the committed bundle, whose
    fit differs slightly from the prototype's (test CE 1.8817 vs 1.8855),
    and §5's rule is that the dose is estimated once and used as-is.
    Step 7 stamps **0.048443521435665396**; the declaration's ~0.043 is
    superseded by this measurement, not by a choice.
30. (Step 6, the marginal and the moment diagnostic) `score_bundle` now
    returns `P = predict_proba` for the inflated bundle and
    `bin_probs(mu, sigma)` otherwise, so a default run still reproduces
    PR #170 exactly; the inflated `P` is floored at 1e-12 and renormalised
    on `bin_probs`'s own convention, differing from the adapter's
    `_class_probs` by at most 2.33e-12, with no observed level in a
    floored bin (min p at the realised y is 2.53e-08). The moment
    diagnostic was **replaced, not dropped**: the probit mid-PIT residual
    `Phi^-1((F(c-1)+F(c))/2)` for the inflated bundle, the unchanged
    `(c - mu)/sigma` for Gaussian ones, labelled everywhere it is printed
    and recorded in the sidecar as `moment_residual`. **Every moment number
    in this run is on the new residual and is not level-comparable with
    PR #170's** — step 11 must not put them in one column.
31. (Step 6, correcting Note 25's stated reason — the claim itself stands)
    On the emission-matched mid-PIT residual the round trip recovers
    0.0913-0.0932 at a true 0.10, i.e. the same ~8 % attenuation as the
    Gaussian path. So the 0.60-0.65 recovery Note 25 measured is a property
    of **scoring levels' normal scores**, not evidence that the estimator
    is mis-dosed: the dose is fitted on this emission and is the right
    number. The caution Note 25 raised about the **simulation** is
    unaffected and still stands, but its reason is now sharper — what CG
    feels is the correlation of the realised *levels*, and the atoms
    concentrate mass so that a shared latent can move `u` a long way inside
    one wide CDF interval without moving the level at all. Which way that
    nets out for group-mean spread is genuinely open; it is measured at
    steps 8 and 11, not argued here.
32. (Step 6, two stale narratives for step 11 not to quote) (a) The
    script's hard-coded closing CAVEAT asserts the human pairwise LR per
    pair sits far above the arm at the same fitted rho — PR #170's finding
    on the Gaussian marginal. On this marginal the human within-cell
    LR/pair is **0.00153** against the (0.03, 0) arm's **0.00130**, i.e.
    broadly consistent with an exchangeable-Gaussian shape. The narrative
    was deliberately left untouched (rewriting it would be interpreting
    this run's result), so step 11 must not cite it. (b) Corner clustering:
    both-0 ratio **1.249** and both-20 ratio **1.496**, far below the
    3.69x / 2.50x the declaration's Note 3 recorded on the old marginal —
    **the atoms have absorbed most of the corner co-censoring excess**,
    which is direct marginal-level evidence the emission does what it
    claims, independent of any simulation. Round-thirds MLEs 0.0244 /
    0.0362 / 0.0899 reproduce PR #170's late-round concentration.
33. (Step 7, confirmed) **Stamped bundle
    `artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.joblib`,
    sha256 `7bfe4d9bff96d012996f8b3edafe36ccb02ce950407ef5314870763760199ed0`**,
    4728 bytes, not LFS-tracked. Read back by the orchestrator:
    `copula_rho_p 0.048443521435665396`, `copula_rho_t 0.0`, `atoms
    prev,0,20`, `test_logloss_binned 1.8817206133537128` — the dose comes
    from the sidecar's `rho_total`, never from a typed literal. All six
    checks PASS: 22 pre-existing keys identical by `is`, the 12-key
    `NEW_KEYS` manifest exact with nothing removed, reload bit-identity on
    7457 rows for `predict`, `predict_std` **and now `predict_proba`**
    (shape (7457, 21) — Note 14e's gap closed, since the first two are the
    mixture's body and verified nothing about the emission that is actually
    sampled), `sample=False` adapter equivalence over 6 rounds x 8 agents,
    and the rho read-back. The parent's bundle and sidecar are byte-
    identical to `git show HEAD:`.
34. (Step 7, the sha256 assert binds — demonstrated, not asserted) Pointed
    at the wrong trunk (`contribution_gaussian_mlp_v2_best.joblib`) with
    the inflated sidecar, the script **refused**: "params sidecar's own
    base_bundle_sha256 disagrees with the --base bundle on disk … stop, do
    not stamp", exit 1, **before any file was written**. The literal
    `EXPECTED_BASE_SHA256` is now a secondary check gated on the default
    `--base`, so a default run still reproduces PR #170 while a
    non-default one is bound by the sidecar instead.
35. (Step 7, a correction to the orchestrator's own brief) I asked that
    verification 4's `sample=False` equivalence exercise
    `_sample_levels_gaussian_copula`. **It cannot, and the implementer was
    right to say so rather than make the wording fit.** That function is
    gated on `self.sample` (`linear_ah.py:490`), so under `sample=False` it
    is called 0 times; `_sample_levels` is called 6 times, its CATEGORICAL
    branch returns `P.argmax(1)`, and the torch RNG state is unchanged —
    confirmed by monkey-patching both samplers and reading the RNG state.
    So verification 4 exercises the CATEGORICAL registration that step 5
    gave `_sample_levels`, which is itself worth having; the inversion
    function's own correctness rests on step 5's suite at `sample=True`
    (Notes 23-25) and on step 8's rollout, which runs the stamped bundle
    through the adapter at `sample=True`.
36. (Step 9, confirmed) **Both sim configs in.** Control
    `23_2g8a_inflctl_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
    differs from the parent's config in exactly **two** functional lines
    (`output_dir`, `figure_name`); candidate
    `23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
    in exactly **three** (those plus `contribution_model` ->
    `artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.joblib`,
    verified present at 4728 bytes and sha256
    `7bfe4d9b…99ed0` — the **stamped** bundle, not the 4175-byte unstamped
    `..._best.joblib`, which would have run the candidate emission with no
    group copula at all and looked superficially fine). Seed 42,
    `n_episodes: 100`, `n_rounds: 24`, `switch_every: 4`, the single
    `lin_multinomial_copula_self` pairing, `save_per_round: true`, the
    switch and punisher artifacts and `valid_model` are byte-identical to
    the parent's. Both output-dir names were run through
    `evaluation_sweep.py`'s `DIR_PATTERN` rather than checked by eye, and
    both parse (`contr` `gaussian_mlp_v2_group_copula` /
    `gaussian_mlp_inflated_group_copula`, `switch`
    `gnn_joint_exodus_k_onehot`).
37. (Step 9, a plan revision the orchestrator made and is recording)
    The byte-copy inherited a **header comment describing the wrong
    experiment** — PR #170's gmlp group copula, naming its rho 0.0438 and
    PR #167's stack — which on the candidate file actively misdescribes the
    contributor it invokes. The plan's "exactly two / exactly three edits"
    exists to protect the **protocol and the RNG context**, not to preserve
    a stale comment, and YAML comments are not parsed into behaviour. Both
    headers were therefore rewritten to describe what each config actually
    is. Verified behaviour-neutral by comparing `yaml.safe_load` output
    before and after (**identical** for both files) and by re-diffing
    comment-stripped against the parent: still exactly 2 and 3 changed
    lines.
38. (Step 8, confirmed — the scored schedule-matched proxy, recorded before
    step 10 is submitted, never a gate) Both stamped bundles rolled through
    the real severity-copula punisher along the parent parquet's
    `agent_group` schedule (100 x 24, torch seed 42), scored by
    `score_all(n_repeats=500, seed=42)` against `convert.load_human`.
    Rollout 26 s per arm, scoring 142 s. **The two declared rows split:**

    | row | real parent | incumbent arm | candidate arm | delta | offset (inc − real) | predicted real |
    |---|---|---|---|---|---|---|
    | **RCA** | 3.50675152179351 | 3.412020623525348 | 1.8160716616190906 | **−1.5959489619062572** | −0.0947308982681623 | **1.910802559887253** |
    | **CG** | 2.079263618744627 | 1.7895995114892012 | 1.9636526447857692 | **+0.1740531332965680** | −0.2896641072554260 | **2.2533167520411954** |

    Means: real parent 1.2226801514921317, incumbent arm 1.2567619508793844,
    candidate arm 1.1936925145764061, **predicted real 1.1596107151891535**
    — 0.185 under the gate-2 ceiling 1.3449481666413449.
39. (Step 8, calibration — what the proxy is and is not entitled to say)
    SA / SB / SC are **bit-identical across arms**: the schedule is the
    parent's and the switch model never reacts, so their zeros are
    structural, not predictions. Over the 18 non-structural rows the
    incumbent arm's mean |offset| is 0.2055 (median 0.1144); excluding RSA,
    mean 0.1358, median 0.0959, max 0.3263. **Only 7 rows have
    |delta| > |offset|** — PA, PB, RCA, RCB, RCC, RCD, RPA. **RSA must not
    be read off this table at all**: its offset is 1.3895, ten times the
    typical one, because it conditions fixed switch decisions on
    received-punishment bins and the proxy's switches do not respond to
    punishment.
40. (Step 8, U1-U4 closed-loop on the rollout — not step 4's teacher-forced
    numbers) **U1 PASS on both clauses:** P(c_t = c_{t−1}) 0.174402 →
    **0.413804** (threshold >= 0.30; human 0.439973) and P(20 | prev 20)
    0.459191 → **0.678290** (threshold >= 0.65; human 0.788975); the
    incumbent arm reproduces the parent's declared 0.1684 / 0.468 to 0.006 /
    0.009, so U1 is the tightest-calibrated quantity in the step. **U2
    PASS:** full-sample human-weighted RCA d 0.931264 → 0.468841, and
    `no_switch_allowed` (weight 6902) carries **0.8633** of the reduction
    against a >= 0.5 threshold — the mechanism moved RCA where it claims to
    live. `switched` is essentially unmoved (1.699 → 1.673). **U4 PASS at
    the top of the band:** P(c = 0) 0.0740 → 0.1380 (limit 0.15, human
    0.0936), P(c = 20) 0.0713 → 0.1305 (limit 0.17, human 0.1344).
41. (Step 8, **U3's second clause fails, and it explains CG's sign flip**)
    P(group-mean >= 18) does rise, 0.037715 → 0.051942 (threshold >= 0.05;
    human 0.092452) — the corner atoms do put whole groups at the top
    together. But SD(group means) rises 4.735185 → 5.251063 (+10.9 %) while
    SD(individual) rises 5.905059 → 6.583996 (**+11.5 %**): the individual
    spread rises *faster*, so the ratio CG actually scores falls
    0.801886 → 0.797550, **away** from the human 0.848872. P(group-mean
    <= 2) overshoots badly, 0.056233 → 0.113369 against human 0.078607.
    This is Notes 25 / 31 realised: a shared latent can move `u` a long way
    inside a wide CDF interval without moving the level, so the same dose
    buys less realised within-group correlation through a discrete
    emission — and the atoms lift both spreads, the individual one more.
42. (Orchestrator's read before the sim runs, so the verdict cannot be
    written backwards) **RCA is the call the proxy is entitled to make:**
    delta −1.596 against a 0.095 offset is 17x the row's own error, the
    largest signal-to-offset ratio in the table, and the incumbent arm
    reproduces the real RCA to 0.095. Predicted real 1.911 — **0.089 inside
    the 2.0 edge, a margin slightly smaller than the row's own offset**, so
    the direction is not in doubt but the crossing is still near a coin
    flip, tilted to crossing; one adverse offset puts it at 2.006.
    **CG is predicted to move the wrong way** (+0.174, against the
    declaration's pre-registered −0.08), to 2.253. Formally that predicts
    nothing — |delta| 0.174 < |offset| 0.290 — but the sign flip is
    mechanistic rather than noise-shaped, and U3 gives the mechanism. So
    the maintainer's assigned row is now expected to *regress*, and the
    experiment's case rests on RCA, which is exactly why Ruling 2 declared
    both. **Amendment A holds: nothing is re-fitted, re-dosed or varied on
    the strength of this table, and the simulation runs as configured.**
43. (Step 8, binding on step 11) (a) If the real sim crosses CG anyway,
    step 11 must check the **SD decomposition**, not just the group-mean
    tail, before anything is claimed — here the tail rose while the ratio
    fell, so the tail alone would have licensed a false claim; on this
    evidence a CG crossing would most likely be noise. (b) Three guards sit
    on a band boundary in the prediction and must be reported as measured
    without treating the proxy as having called them: RCD 0.994268 (holds
    `<= 1` by 0.006 — PR #175 lost RCD's band on a contribution change),
    CB 1.011036 (leaves `<= 1` by 0.011), PB 0.997619 (holds by 0.002).
    (c) RCD's cost is twice its pre-registration (+0.263 against +0.13);
    with CG's sign flip these are the only two guard predictions that moved
    materially, the other nine reproducing within ~0.1.
44. (Step 10, confirmed — verified by the orchestrator on Raven, not only
    reported) Both sims `COMPLETED 0:0` in **00:02:45** each, jobs
    **30254229** (control) and **30254230** (candidate), submitted into
    `~/autoresearch/contribution-inflated-gmlp` under
    `AI_REMOTE_DIR`. **The control reproduces PR #174's
    `per_round.parquet` bit for bit:
    `3cb8b3d72784afe09464e0048d27d7cb05f7a3b157780f296e99d168407fef3f`** —
    so step 5's adapter edit is provably inert for a Gaussian bundle, and
    the candidate may be judged against the parent's committed
    `scores.csv`. **Activation confirmed:** the candidate's parquet is
    `f0296f86478d98d49d452c71ca44c89d1351244636578b45839bfa8178159312`,
    different, so the inflated branch was taken. Remote slot artifacts all
    match: contribution `da42031a…0ea7c` (control) /
    `7bfe4d9b…99ed0` (candidate, the stamped bundle), punisher
    `9e3cf677…8cc2f`, switch `28dd4b40…c820d`. PROVENANCE in both logs:
    `/raven/u/certuer/algorithmic-institutions/.venv/bin/python` — the
    shared venv, as §9 intends — with `aimanager` resolving under
    `/u/certuer/autoresearch/contribution-inflated-gmlp/src/`, and
    `algorithmic-institutions/src` appears **zero** times in either log, so
    no job imported the shared checkout's code. One draw each, seed 42, no
    re-runs. A `squeue` check before syncing showed one PENDING job in a
    different experiment's isolated dir
    (`~/autoresearch/contribution-arrival-tenure`), disjoint from the sync
    target, so no `rsync --delete` race.
