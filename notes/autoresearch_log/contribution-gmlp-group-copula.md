# Autoresearch log: contribution — shared group latent (Gaussian copula) at the gaussian_mlp_v2 sampler

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gaussian_mlp_v2` — the PR #167 candidate (2-layer
  heteroscedastic Gaussian MLP, hidden 8, forced group-conditioned core, 7
  features; bundle `artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib`).
- **Parent PR:** **#167** (`[SUCCESS]`, branch `auto/contribution-gaussian-mlp-v2`,
  RCA 5.21 -> 4.00). This experiment is stacked on it per §9 "Building on a
  `[SUCCESS]` PR": branch `auto/contribution-gmlp-group-copula` off
  `origin/auto/contribution-gaussian-mlp-v2`, PR opened with
  `--base auto/contribution-gaussian-mlp-v2`.
- **Evaluation stack (§3, parent-directed):** exactly PR #167's —
  `gaussian_mlp_v2 contr x gnn switch x PR #160 severity-copula multinomial
  punisher` (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`,
  `copula_rho` 0.3507588625344979), single self pairing, seed 42, 100 episodes,
  24 rounds, `save_per_round: true`. Baseline for BOTH §2 gates = the parent's
  own candidate run, at full precision from
  `plots/simulation/23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch/evaluation/scores.csv`
  (in this worktree; 21 rows, run
  `ah group_switching managed by lin_multinomial_copula_self`).
- **Target rows** (exact, unrounded, from that file):
  - **CG 5.91060457046713** (band `> 5`) — primary; declared upgrade
    `> 5` -> `2-5`. Band arithmetic: CG = |ratio_sim - ratio_human| / denominator
    with ratio = SD(group-mean contribution over (episode, round, group) cells)
    / SD(individual contribution). Human ratio 0.8480163543652899, denominator
    0.026448679600274437. The sim ratio must therefore rise above
    **0.7157729563639177** for `2-5` (above 0.795118995164741 for `1-2`, above
    0.8215676747650155 for `<= 1`). The parent's candidate sits at
    0.6915274426731914 (gap 0.15648891169209844, identical to the metrics.csv
    `d`), so the band upgrade needs **+0.0243** in the ratio, i.e. SD(group
    means) 3.4969 -> at least 3.62 at unchanged SD(individual) 5.0568.
  - **CA 2.1657982689458293** (band `2-5`) — secondary, declared upgrade
    `2-5` -> `1-2`. Its diagnostic `std_diff` is **-2.621590294686563**: the
    sim's SD of participant means is 2.62 below the human's. A *persistent*
    shared component raises between-participant spread (members of a
    high-culture group all sit high for the whole episode); a transient one
    does not — so under the fully persistent candidate CA is expected to
    move (Note 9: SD of participant means 2.98 -> 3.85 in the closed-loop
    proxy at rho_p 0.026).
  - **SC 2.8665076492844292** (band `2-5`) — secondary, declared upgrade
    `2-5` -> `1-2`. SC shares CG's root cause (§6: CG, PD, SC all sit on the
    independence floor; PR #165's contribution-only copula moved SC 2.82 ->
    2.19 as collateral).
  - **Named guard: RCA 3.9995310319882194** must stay out of `> 5`. Every
    agent's marginal N(mu_i, sigma_i) is preserved *by construction*, so RCA
    (and CA/CD/CF/RCB, which are marginal or per-agent rows) should not move
    beyond the free-running feedback of the changed trajectories; a large RCA
    move is evidence of an implementation bug (marginals not preserved), not
    of a trade-off, and is a stop.
  - **Second guard: RCD 1.1913081291097947** (switching pull, band `1-2`).
    PR #168 found a shared latent in the *switch* slot cost RCD, while PR
    #165's contribution latent improved it (2.94 -> 1.96); with the
    arrival-group rule a switcher assimilates toward the receiving group's
    latent, so RCD is the row that tells the sign of the trajectory-level
    side effect. Watched, not gating.
  - **Overshoot is not a risk at this dose:** every proxy puts the fitted
    persistent dose at a ratio of ~0.775-0.79 against the human 0.848 —
    undershooting, on the near side of the `2-5` edge, not past `<= 1`.
  - **Gate 2:** the 21-row mean must fall strictly below
    **1.6145149045441503** (rows <= 1 at 8/21 is context only).
- **The finding that motivates the experiment** (orchestrator-measured,
  reproduced here with `evaluation_suite.convert` + the CG cell definition,
  Notes 1-2): human ratio 0.8480163543652899; PR #167 baseline (plain
  gaussian) 0.7427252620205091 (SD group means 3.788970987242582, SD
  individual 5.101443536382576); PR #167 candidate (gaussian_mlp_v2)
  0.6915274426731914 (3.4968874221622444 / 5.056758714657166). SD(group
  means) fell 3.789 -> 3.497 while SD(individual) barely moved 5.10 -> 5.06:
  the sim is **under-dispersed at the group level**, and the candidate moved
  *further* below the human ratio. **This falsifies the mechanism the parent
  proposed for its own successor** (parent Notes 13/15: "hold a+b near
  0.77"). a+b is an open-loop AR(1) argument about group-mean drift; the
  measured group-mean spread went DOWN when a+b went UP (0.7681 -> 0.9331),
  because the loop is closed by the punisher's feedback and by the 0/20
  bounds. Persistence of the mean is not the knob. Group-level dispersion
  has to come from *shared* variance, which independent per-agent sampling
  cannot produce at any a+b.
- **Hypothesis (behavioral, one sentence):** members of one group read the
  same situation the same way and act alike without observing each other, so
  their unexplained deviations from the model's conditional mean are
  positively correlated within the group rather than independent — the
  between-participant correlation §6 names as the shared root cause of CG,
  PD and SC (and which `notes/evaluation_metric_defs.md` states in its
  independence-floor paragraph). The row this should move is CG; CA moves
  iff the shared component has a persistent part.
- **Planned change (one change, contribution slot only, no retraining):** a
  **marginal-preserving shared group latent (Gaussian copula) at the
  gaussian_mlp_v2 contribution sampler**. Today
  `src/aimanager/simulation/linear_ah.py` draws
  `c_i = clip(rint(mu_i + sigma_i * z_i), 0, 20)` with independent standard
  normals `z_i`. The change replaces the `z_i` of the members of one group by
  equicorrelated standard normals with two shared components,

      z_i = sqrt(rho_p) * u_g[episode] + sqrt(rho_t) * v_g[round]
            + sqrt(1 - rho_p - rho_t) * e_i,

  `u_g` drawn once per (episode, group) and held for the episode, `v_g`
  fresh per (episode, round, group), `e_i` idiosyncratic. Each `z_i` is
  exactly N(0, 1), so every agent's marginal N(mu_i, sigma_i) is unchanged
  whatever (rho_p, rho_t); only the within-group dependence changes. The
  implementation keeps this general two-component sampler (it costs
  nothing, degenerates cleanly, and lets the preflight run every diagnostic
  arm from one code path), but **the single declared candidate is the fully
  persistent one: `rho_p = rho_total`, `rho_t = 0`** — one per-(episode,
  group) latent held for the whole episode, PR #165's structure at a
  Gaussian emission. One candidate, one sim, one evaluation. A
  Gaussian emission needs no inverse-CDF inversion and no
  `generic/copula.py` (it exists only on PR #165's branch and is not ported):
  the correlated normal goes straight into `mu + sigma * z`. The group array
  is the adapter's post-arrival membership `self._group[t]`, so a switcher
  picks up the *receiving* group's latent (PR #165's arrival-group rule).
- **The dose is estimated, not chosen — and not tuned.** `rho_total`
  comes from one estimator, fitted once on the **training split**
  (`experiments/baseline/2group_8agent_50ep_bline_train.csv`, single copy,
  `exclude_flipped: True`, mask `contribution_valid`, 40 episodes, 7457 rows)
  before any simulation, and is used **as-is** as `rho_p`:
  - `rho_total` = the repo's **interval-censored pairwise Gaussian-copula
    MLE** (`scripts/baselines/punishment_copula_rho.py`: `rho_mle` /
    `pair_nll` / `bvn_cdf` / `rect_points` / `cdf_bounds`, reused unmodified)
    over all cross-member pairs inside one (episode, round, group) cell,
    applied to gaussian_mlp_v2's 21-bin marginals `bin_probs(mu_i, sigma_i)`
    (`scripts/baselines/gaussian_mlp_preflight.py`), which is exactly the
    discrete marginal `clip(rint(mu + sigma z))` produces. Statistical
    justification, in one sentence: the emission is censored at 0 and 20
    on 21.6 % of rows (10.3 % at 0, 11.3 % at 20) and rounded on all of
    them, and the rectangle-probability MLE is consistent for the latent
    correlation under exactly that censoring where a moment correlation of
    standardised residuals is attenuated — it is the estimator PRs #149,
    #160 and #165 established and it is reused unmodified. Cluster-bootstrap
    CI over episodes reported alongside.
  - **No grid over rho_p, no arm selection by score, no boundary ruling.**
    The dose-response table in Note 8 is a *prediction*, and the fitted
    dose is simulated whatever that table implies about a larger or smaller
    one. No score enters the estimate.
  - The **cross-member lag-1 refit** (member i at round r vs member j != i of
    the same (episode, group) at r + 1 — PR #165's `cross_pairs` minus
    self-pairs) is computed and reported prominently, **as the
    pre-registered falsifier, never as the dose** (next bullets).
- **Persistence is a structural choice, pre-registered on grounds
  independent of any score** — not an estimate, not a boundary ruling.
  (An earlier draft of this Declaration carried the orchestrator's
  position that persistence was a *liability* here; that rested on a
  mislabelled probe table and is retracted — Note 8 records the
  correction. What stands from it is the mechanism: a+b = 0.9331 is a
  large closed-loop amplifier of a constant group offset, which is *why* a
  persistent latent pays here; the clipping-compression claim was the
  artefact.) The grounds:
  1. **PR #150's controlled A/B**, the only direct experiment in the repo
     on this question: same rho, the transient arm *regressed* SC (2.905
     against a 2.816 baseline), the persistent arm band-upgraded it
     (1.914). Prior evidence, unrelated to CG.
  2. **The teacher-forced vs free-running decomposition**, a property of
     the model rather than of a metric (Note 10): the one-step
     teacher-forced ratio is 0.812 against human 0.847 (gap 0.035), while
     free-running is 0.6915 against 0.848 (gap 0.157). The one-step
     conditional correlation is therefore nearly right already; the
     missing variance is *generated*, not conditional. A dose read off
     one-step residuals targets the 0.035 gap, and CG scores the 0.157
     one — PR #149's own closing finding restated, and why its one-step
     preflight killed it.
  3. **The lag-1 residual estimator targets the wrong quantity for the
     split, and is attenuated by construction in this model.** The mean
     function carries `prev_contribution_mean_group` and
     `prev_win_contribution_mean_group`, which absorb the persistent group
     level into mu *given observed group history*. A near-zero lag-1
     cross-member residual correlation is therefore evidence that the
     persistent component is represented conditionally — not evidence
     that the free-running loop generates it. The exact mirror of #165's
     argument that phi_hat was biased upward by the `prev_contribution`
     feedback; here the same reasoning points down.
  4. Minor, §5's tie-break: one estimated parameter beats two.

  Both proxies agree on sign and magnitude (Notes 8-9): the orchestrator's
  reduced-form MC and Fable's closed-loop rollout of the real model both
  put a persistent share at ~5x the transient share per unit, and both
  land a persistent 0.07 at 0.844 — two independently built proxies
  agreeing is the strongest evidence either has.
- **The preflight is a diagnostic, never a gate — pre-committed.** PR #149
  was killed by a one-step teacher-forced preflight that #150's arm A/B
  comparison later proved structurally blind to multi-round accumulation;
  #165 pre-committed to simulating regardless and was right to. Step 8
  reports predicted *deltas* against each proxy's own rho = 0 arm (the
  proxies are ~0.03-0.04 off in absolute ratio, Notes 8-9), and **the
  experiment proceeds to simulation at the fitted (rho_p, rho_t) whatever
  the preflight says.** Only an implementation failure (marginals not
  preserved, round-trip not recovered) stops it.
- **Pre-registered predictions and the falsifier (Notes 6-10).** The
  within-cell share is small — moment share 0.0263 (95 % CI [0.0157,
  0.0376]), orchestrator's all-rows moment 0.02776 — so `rho_total` from
  the MLE is expected in the ~0.03-0.05 range; the lag-1 cross-member share
  is ~0.006 (CI spanning 0). Two readings of the same residuals, both
  stated without softening:
  - **The honest two-component reading** (rho_p ~ 0.006 from lag-1,
    rho_t ~ 0.021 the rest) predicts ratio ~0.707, **CG ~5.3, a FAIL**
    (Note 8, mixed lag-1 arm); a purely transient 0.028 predicts CG ~6.6.
  - **The persistent reading** (rho_p = rho_total ~ 0.028, rho_t = 0)
    predicts ratio 0.775 / **CG ~2.8** on the orchestrator's MC and ratio
    delta +0.068-0.07 / **CG ~3.4** on Fable's real-model rollout — a band
    upgrade in both.
  - **The measured CG discriminates between them.** If it lands near 6.6
    the transient reading was right and the structural choice was wrong;
    if it lands in `2-5` the persistent reading was right. That is the
    scientific content of the run, and it is recorded here so the verdict
    cannot be re-read after the fact.
- **Iteration budget check (§5):** no GNN training. One refit of the
  incumbent setting (hidden 8, lr 0.01, wd 3e-4, 1000 epochs, 7457 rows)
  takes **0.70 s** locally, and nothing here refits at all: the estimator is
  a pairwise MLE over 15,090 within-cell pairs plus ~28,700 lag-1 pairs
  (seconds, plus a 200-resample cluster bootstrap in low minutes), the
  closed-loop proxy is ~30 s per arm locally, and the single Raven sim is
  ~2.5 min. Total well under 3x anything.

## 2. Plan

Steps are written for the orchestrator to validate and tag (§9 Roles). Every
step is legal per §5 (sampling-time dependence structure estimated from the
training split; no feature, seed, episode-count or scoring change), and
nothing touches the frozen surface (§8). Paths are relative to the worktree
`.claude/worktrees/contribution-gmlp-group-copula`.

- [x] 1. **Worktree env prep** — [Sonnet] in the worktree: `uv sync` (the worktree has no
   `.venv`; the parent's worktree does, this one must build its own);
   `mkdir -p data/baselines` (gitignored); set the §9 commit identity
   (`git config extensions.worktreeConfig true`, `--worktree user.name
   "Claude"`, `--worktree user.email "noreply@anthropic.com"`); verify
   `artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib`,
   `artifacts/baselines/punishment_multinomial_severity_copula.joblib`,
   `experiments/baseline/2group_8agent_50ep_bline_{train,test}.csv` and the
   parent's `plots/simulation/23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch/per_round.parquet`
   are real files, not LFS pointers (`git lfs pull` otherwise). Run the local
   suites once to pin the starting point: `pytest tests/baselines/`
   (test_gaussian_mlp + test_punishment_copula + features) and
   `pytest src/aimanager/tests/test_eval_*.py`. No tracked-file changes.

- [x] 2. **Reproduce the CG direction + residual ICC diagnostic** — [Sonnet] new file
   `scripts/baselines/gmlp_group_copula_diagnostic.py`. Part A: load the
   human CSV via `aimanager.evaluation_suite.convert.load_human` and the two
   parent sims (`23_2g8a_gmlp2_base_self_gaussian_contr_gnn_switch`,
   `23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch`) via `load_sim`
   (read-only import of the frozen suite; `GROUP_CELL = [episode_id,
   round_number, group_id]` from `metrics.py`), print SD(group means),
   SD(individual), ratio, |gap| and gap/denominator for all three, and the
   band edges of the ratio (expected: 0.8480 / 0.7427 / 0.6915; Note 1).
   Part B: build the train-split rows with `create_torch_data` +
   `build_feature_pool` (mask `contribution_valid`; `prepare_data()` flattens
   the (episode, agent, round) indices away and cannot be used — keep
   `agent_group` yourself, as `punishment_copula_rho.build_rows` does),
   compute the teacher-forced standardised residuals
   `r = (c - mu) / sigma` of the incumbent bundle, and print: the one-way
   ICC(1) per (episode, round, group) cell (`icc_oneway` from
   `punishment_copula_rho`), the cross-member within-cell moment share, the
   cross-member lag-1 share, the same-agent cross-round share (individual
   persistence, for contrast), the cross-group same-round share (episode
   level), by round thirds, and the same on the test split as an
   out-of-sample check only. Expected values in Note 6. Labelled throughout
   as *attenuated moment diagnostics*; the estimate that matters is step 6.

- [x] 3. **Open the adapter's gate to Gaussian contribution copulas (its own
   step, so the revision is visible in the diff)** — [Sonnet]
   `src/aimanager/simulation/linear_ah.py`, `LinearAHAdapter.__init__`
   (currently lines 88-101): keep `copula_rho` as the multinomial-punisher
   field with its existing asserts unchanged; add `self.copula_rho_p =
   float(bundle.get("copula_rho_p", 0.0) or 0.0)` and `self.copula_rho_t`
   likewise, assert `0 <= rho_p`, `0 <= rho_t`, `rho_p + rho_t < 1`, and
   assert the pair is zero unless `self.target == "contribution" and
   self.model_type in ("gaussian", "gaussian_mlp")` (message naming the
   Gaussian contribution sampler). `tests/baselines/test_gaussian_mlp.py::test_copula_rho_rejected_on_gaussian_mlp`
   (line 323) currently asserts the punisher field is rejected on gaussian
   bundles — keep that assertion (it still holds) and extend the test: the
   new fields are accepted on `gaussian` / `gaussian_mlp` contribution
   bundles, rejected on a multinomial punishment bundle and on a `ridge`
   bundle, and `rho_p + rho_t >= 1` is rejected. Local
   `pytest tests/baselines/` green.

- [x] 4. **The Gaussian group-copula sampler** — [Opus] `src/aimanager/simulation/linear_ah.py`:
   (a) `_reset_history()` (line 128) additionally sets `self._copula_z = {}`
   (group id -> persistent latent `u_g`, episode state). (b) New method
   `_sample_levels_gaussian_copula(self, Xs, n_levels, groups)`, following
   PR #160's `_sample_levels_copula` template (lines 274-298): `mu =
   predict`, `sd = predict_std`; **exactly 3n float64 torch draws per call
   in fixed order** — `zu = randn(n)` (persistent innovations), `zv =
   randn(n)` (round latents), `eps = randn(n)` — whatever (rho_p, rho_t) and
   whatever the group composition, so the RNG stream is composition-stable;
   a `first`-index `pick` map (first member of each group id in row order)
   selects each group's slot in `zu`/`zv`; a group id not yet in
   `self._copula_z` gets `u_g = zu[pick]` stored there (persistent for the
   episode; a group id that empties and re-forms keeps its stored `u_g`);
   `z = sqrt(rho_p) * u_g(i) + sqrt(rho_t) * zv[pick[i]] + sqrt(1 - rho_p -
   rho_t) * eps[i]`; return `clip(rint(mu + sd * z), 0, n_levels - 1)` as
   int64. `sample=False` returns the rounded mean and consumes no RNG (as
   the legacy path does). (c) Call-site gate in `predict()` (line 325):
   `if self.sample and (self.copula_rho_p + self.copula_rho_t) > 0.0:` use
   the new method with `groups = self._group[t]` (post-arrival membership,
   recorded by `_record` just before), else the existing `_sample_levels` —
   so a bundle without the fields keeps a **byte-identical RNG stream and
   output**. `n_levels` is `self.n_contributions` (21), never the bundle's
   `n_levels` (0 for continuous bundles). Docstring records the conventions
   the code cannot show (draw count, first-member rule, arrival-group
   semantics, persistent-latent lifetime).

- [x] 5. **Unit tests** — [Opus] new file `tests/baselines/test_contribution_group_copula.py`
   (local, CPU torch, `toy_bundle` pattern from `test_gaussian_mlp.py` with
   `copula_rho_p` / `copula_rho_t` added). The two that matter: **(i)
   marginal preservation** — over many draws on fixed feature rows, each
   agent's level histogram under the copula sampler matches the independent
   sampler's to sampling tolerance (chi-square or max abs frequency
   difference), for (rho_p, rho_t) in {(0.3, 0), (0, 0.3), (0.2, 0.2)}; and
   **(ii) correlation recovery** — generate an episode-shaped synthetic
   panel through the adapter itself at a known (rho_p, rho_t) and recover
   both from the pre-clip `z` (monkeypatch `predict_std`/`predict` to
   constants so `z` is observable, or read `z` back from levels on
   wide-sigma rows): within-cell correlation ~ rho_p + rho_t, cross-member
   lag-1 ~ rho_p, cross-group ~ 0, tolerance 0.02. Plus: legacy bundles
   (fields absent / both 0.0) bit-identical to `_sample_levels` over three
   consecutive calls including the next RNG draw (PR #160's
   `test_legacy_bundles_bit_identical` pattern); fixed RNG consumption
   regardless of composition (one-group vs two-group round leaves the
   stream at the same point); the persistent latent is constant across
   rounds within an episode and re-drawn on `reset_rnn` / `t == 0`; a
   switcher (group array change between rounds) draws from the receiving
   group's `u_g`; `sample=False` ignores both fields and consumes no RNG;
   determinism under `th.manual_seed`. Existing suites stay green:
   `pytest tests/baselines/` (31 copula + gaussian_mlp tests) and the
   eval-suite tests locally; `scripts/remote_test.sh` for the PyG suite on
   Raven (`test_linear_manager` touches `linear_ah.py`) — note the parent's
   Note 16: `test_eval_*` fail on an isolated remote dir because `plots/` is
   not synced; not a regression.

- [x] 6. **The estimator** — [Opus] new file `scripts/baselines/contribution_gmlp_copula_rho.py`
   (local, CPU torch, no PyG). Imports `punishment_copula_rho` as a module
   for `rho_mle`, `pair_nll`, `bvn_cdf`, `rect_points`, `cdf_bounds`,
   `pair_index`, `blocks`, `check_bvn` (unmodified) and
   `gaussian_mlp_preflight.bin_probs`. Rows: the step-2 builder (train
   split, `contribution_valid`, indices kept; the bundle's own 7 features and
   scaler; bundles unpickle only with `scripts/baselines` on `sys.path`).
   `P = bin_probs(mu, sigma)` (21 bins, tails folded — the marginal the
   sampler realises), `y = int(c)`, `z_lo, z_hi = cdf_bounds(P, y)`.
   (a) `rho_total`: `pair_index` over the cell `(episode, round, group)`,
   `rho_mle` -> point estimate, pairwise LR, `bootstrap_mle` cluster CI (200
   resamples over episodes, seed 38381). **This is the dose: `rho_p =
   rho_total`, `rho_t = 0`, used as-is.** (b) The **falsifier**: a local
   `lag1_cross_pairs(episode, round, group, agent)` (PR #165's `cross_pairs`
   at step 1, keyed by each row's group at its own round, **self-pairs
   excluded**), same `rect_points` + `rho_mle`, and a cluster bootstrap over
   episodes of the same fit for its CI — printed prominently together with
   the implied two-component reading `(rho_p_lag1, rho_total - rho_p_lag1)`,
   and **never stamped as the dose**. (c) Diagnostics printed but never
   used: the attenuated moment shares (step 2), round-thirds splits, the
   out-of-sample test-split MLEs. (d) **Round-trip acceptance gate**: sample
   synthetic panels through the *adapter's own* sampler (step 4) on the real
   feature rows at (rho_p, rho_t) in {(0.1, 0), (0, 0.1), (0.05, 0.05),
   (0.03, 0)}, re-estimate `rho_total` and the lag-1 `rho_p` with (a)-(b);
   max |bias| <= 0.02 on each is PASS, else stop (implementation bug).
   The `(0.03, 0)` arm carries extra weight and its result is reported
   separately: it is the direct **power test of the falsifier estimator** at
   the fitted dose. If the lag-1 fit recovers ~0.03 from a purely persistent
   panel, the ~0.006 it returns on human data is informative and the
   falsifier reading of the Declaration is live. If it recovers ~0, the lag-1
   estimator cannot see the persistence it is meant to falsify, which
   confirms ground 3 empirically rather than by argument. Either outcome is
   recorded in the Notes before step 8; neither changes the dose.
   (e) `--write-params`: JSON sidecar
   `artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.params.json`
   with rho_total + CI (the dose), rho_lag1 + CI and the two-component
   reading (the falsifier), n_pairs (within / lag-1), estimator tag,
   `structure = "persistent_episode_group"`, data file, base bundle sha256,
   git sha, timestamp. Record all numbers in Notes before step 7.

- [x] 7. **The stamper** — [Sonnet] new file `scripts/baselines/stamp_contribution_group_copula.py`
   (precedent: PR #160's `save_bundle`, PR #165's
   `make_contribution_copula_artifact.py`; **never** via
   `inspect_best_model.py`). Loads the base bundle and the step-6 JSON,
   writes `artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib`
   = base dict + **`copula_rho_p = rho_total`, `copula_rho_t = 0.0`**,
   `copula_rho_total_ci`, `copula_structure = "persistent_episode_group"`,
   `copula_estimator = "pairwise_mle_censored_gaussian"`, `copula_cell_key =
   "episode_round_group"`, `copula_data_file`, `copula_n_pairs`, and, as
   provenance only, `copula_rho_lag1`, `copula_rho_lag1_ci`,
   `copula_n_pairs_lag1` (the falsifier, not consumed by the sampler).
   Verifies: every pre-existing key is the identical
   object (`new[k] is base[k]`), reload -> `predict` and `predict_std` on
   the train rows bit-identical to the base, and
   `LinearAHAdapter(new, sample=False)` returns the same levels as the base
   adapter on a fixed state sequence. Prints the new file's sha256 (checked
   again on Raven in step 10). The stamped bundle is LFS-tracked like its
   siblings.

- [x] 8. **Preflight (diagnostic, not a gate)** — [Opus] new file
   `scripts/baselines/gmlp_group_copula_preflight.py`, running four arms
   from the one two-component code path, reporting **deltas** against the
   rho = 0 arm: (i) rho = 0; (ii) **the candidate**, `(rho_p, rho_t) =
   (rho_total, 0)` from the stamped bundle; (iii) the **falsifier
   reading**, `(rho_lag1, rho_total - rho_lag1)` from the step-6 JSON;
   (iv) transient-only `(0, rho_total)` for completeness. Two proxies,
   both local: (A) the
   orchestrator's reduced-form Monte Carlo — `c_it = clip(rint(a * c_i,t-1
   + b * m_g,t-1 + k + sigma * z_it), 0, 20)` at the parent's measured a =
   0.7316, b = 0.2015, sigma = 3.410, n = 8, 4000 episodes, 4-round
   burn-in — whose rho = 0 arm sits at 0.6551 (Note 8); (B) the closed-loop
   rollout of the actual `LinearAHAdapter` contribution bundle against the
   actual PR #160 punisher bundle with fixed 4/4 groups and no switching,
   100 episodes x 24 rounds, seed 42, mirroring the env's round order
   (contribution -> punishment -> common good `(1.6 * sum c - sum p) /
   n_valid` per group -> `prev_*` shift), whose rho = 0 arm sits at ~0.72
   (Note 9). Print for each: SD(group means), SD(individual), ratio, the
   delta vs rho = 0, the implied CG at the parent's baseline ratio +
   delta, and SD(participant means) (the CA diagnostic). Record the
   predictions as a Note *before* step 10 is submitted — arm (ii) is the
   prediction the run tests, arm (iii) is what a FAIL would vindicate.
   **The experiment proceeds to simulation at the fitted persistent dose
   whatever these print, and whatever a larger or smaller dose would
   imply.**

- [x] 9. **Candidate sim config** — [Sonnet] new file
   `configs/simulation/manager_testing/23_2g8a_gmlpcop_self_gaussian_mlp_v2_group_copula_contr_gnn_switch.yml`:
   a copy of the parent's
   `23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch.yml` in which only
   `contribution_model` (-> the step-7 bundle), `output_dir`
   (`plots/simulation/23_2g8a_gmlpcop_self_gaussian_mlp_v2_group_copula_contr_gnn_switch`)
   and `figure_name` change. Slug before `_self_` so `evaluation_sweep.py`'s
   DIR_PATTERN still parses; seed 42, 100 episodes, 24 rounds, single
   `lin_multinomial_copula_self` pairing, `save_per_round: true`,
   byte-identical protocol fields.

- [x] 10. **Job-log provenance line, Raven sim, fetch, local evaluate** — [Opus] (a)
    `scripts/run_simulation.sh` (the SLURM template): add one line before
    the `python -m aimanager simulate` call, `python -c "import sys,
    aimanager; print('PROVENANCE', sys.executable, aimanager.__file__)"`,
    so every job log states which interpreter and which `src/` ran (the
    parent's Note 8 bug: an isolated-dir job can silently import the shared
    checkout's editable install; the parent verified the fix with a side
    probe, this makes it permanent and it cannot move a score). Note the file
    is a Python `.format()` template (`{{AIMANAGER_VENV:-.venv}}`,
    `{config_path}`), so the added line must contain **no single braces**,
    and the rendered script must be inspected once before submission. Its
    own commit. (b) Check
    `squeue -u certuer` for PENDING jobs before any sync, then
    `AI_REMOTE_DIR='~/autoresearch/contribution-gmlp-group-copula'
    scripts/simulate_cluster.sh <step-9 config>`. (c) When the job finishes,
    confirm from `.log/simulation/<config>/<job>/log.log` that the
    PROVENANCE line names the shared venv's interpreter
    (`~/algorithmic-institutions/.venv/bin/python`) and
    `aimanager.__file__` under `~/autoresearch/contribution-gmlp-group-copula/src/`,
    and `ssh raven sha256sum
    ~/autoresearch/contribution-gmlp-group-copula/artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib`
    equals the step-7 hash. Any mismatch: the run is void, fix and resubmit.
    (c2) **Activation check:** compare the fetched `per_round.parquet`
    against the parent's
    (`plots/simulation/23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch/per_round.parquet`)
    and require them to **differ**. Same seed, same marginals and the same
    code on the independent path would make an inactive copula reproduce the
    parent bit-for-bit, so bit-identity is the unmistakable signature that
    the sampler branch was never taken — the failure mode that voided four
    of PR #168's runs. Identical files: the run is void.
    (d) `AI_REMOTE_DIR=... scripts/fetch_cluster.sh
    plots/simulation/23_2g8a_gmlpcop_self_gaussian_mlp_v2_group_copula_contr_gnn_switch`,
    then locally `python -m aimanager evaluate <config>`. Log the Results
    row (all 21 scores available in `evaluation/scores.csv`; report CG, CA,
    SC, RCA, rows <= 1, mean, exactly as computed) against the parent's
    baseline numbers in the Declaration.

- [x] 11. **Close out** — [Opus] verdict per §2 from that single evaluation: a band
    upgrade on CG, CA or SC **and** mean strictly below 1.6145149045441503
    is `[SUCCESS]`, anything less `[FAIL]`; no second stage, no re-run, no
    dose change. Update this log (Results, Notes: the mechanism reading —
    did marginals hold (RCA), which reading the measured CG supports
    (persistent ~2.8-3.4 vs two-component ~5.3 vs transient ~6.6), what
    RCD and the collateral say). Open the PR with `gh pr
    create --base auto/contribution-gaussian-mlp-v2`, titled `[SUCCESS] ...`
    / `[FAIL] ...`, body Hypothesis / Results / Collateral (§9.7), noting up
    front that the diff shows only this experiment's change over PR #167.
    Delete `~/autoresearch/contribution-gmlp-group-copula` on Raven.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-02 | group copula at the gaussian_mlp_v2 sampler, `rho_p = 0.04378520865574197` (censored pairwise MLE), `rho_t = 0.0` | CG 2.8292447... (**band upgrade > 5 -> 2-5**), CA 1.6416427... (**band upgrade 2-5 -> 1-2**), SC 2.6692710... (within band); guards RCA 3.4934320... (2-5, held), RCD 0.6701004... (1-2 -> <= 1) | 10/21 | 1.3138927788530981 | **SUCCESS -- gate 1 twice and gate 2 both pass** |

## 4. Notes

1. **CG direction reproduced** (`evaluation_suite.convert` +
   `GROUP_CELL`, one copy per human game, `contribution` NaN rows dropped):
   human SD(group means) 5.35768983626258 / SD(individual)
   6.317908621317357 = ratio 0.8480163543652899 (2239 cells, 9320 rows);
   PR #167 baseline 3.788970987242582 / 5.101443536382576 = 0.7427252620205091
   (4700 cells, 19200 rows); PR #167 candidate 3.4968874221622444 /
   5.056758714657166 = 0.6915274426731914 (4704 cells). Gaps 0.10529 and
   0.15649 (the latter equals the metrics.csv `d` exactly); over the
   denominator 3.98096 and 5.91670 — within 0.006 of the resampled scores
   3.97806 / 5.91060, as expected for a point estimate vs a 500-repeat mean.
2. Band edges on the sim ratio, from the parent's exact numbers: `2-5`
   needs > 0.7157729563639177 (+0.0243 from 0.6915), `1-2` > 0.795118995164741,
   `<= 1` > 0.8215676747650155. The independence floor on the human cells is
   E[1/sqrt(n_cell)] = 0.553 (1/sqrt(mean n) = 0.490); the sim at 0.69 is
   above the floor only because of the state feedback through the group-mean
   features, not because of any shared sampling.
3. **Why the parent's a+b successor is not pursued.** Parent Note 15 reads
   a+b = 0.93 vs 0.77 as "the right knob" and proposes a successor that holds
   a+b at 0.77 with the v2 feature core. But the measured SD(group means)
   *fell* (3.789 -> 3.497) as a+b *rose*; the open-loop AR(1) story predicts
   the opposite. The loop is closed by the punisher (low contributors are
   punished back up) and by the 0/20 clip, so persistence of the conditional
   mean does not translate into group-mean dispersion. Dispersion needs
   shared variance, which no a+b can create from independent draws — hence a
   copula, not a re-tuned mean.
4. **What the prior shared-variance work establishes.** PR #149 (`[FAIL]`,
   GNN): within-round copula rho 0.0696 by pairwise MLE closes ~14 % of the
   teacher-forced gap and was stopped at a one-step preflight; its key
   finding is that the CG deficit is overwhelmingly *free-running* loss (the
   GNN's teacher-forced independent ratio is already 0.78). PR #158
   (`[FAIL]`): an independent per-agent type latent leaves the ratio pinned
   (0.451 -> 0.470) — individual heterogeneity scales both SDs alike. PR #159
   (`[FAIL]`, best CG on record then, 8.01): a trained per-(episode, group)
   latent moves CG tax-free but is dose-capped by teacher-forced MLE (38 % of
   the required move). PR #165 (`[SUCCESS]`): the same rho 0.0696 applied
   as a *static* per-episode group latent (phi = 1.0 by boundary ruling after
   phi_hat saturated at 1.16) took CG 9.81 -> 4.16 plus RCD and RCA band
   upgrades — ~15x the one-step preflight's prediction, through multi-round
   accumulation. PR #150 (`[SUCCESS]`, switch): rho-only *regressed* SC,
   rho + AR(1) phi 0.70 band-upgraded it — persistence was essential there.
   Common thread: doses of 0.06-0.12 estimated by the pairwise MLE, applied
   once, never tuned; and the one-step preflight underpredicts every time.
5. **Sampler semantics vs those precedents.** #165's copula had to invert a
   categorical CDF (`generic/copula.py`); the Gaussian emission here needs
   only correlated normals into `mu + sigma * z`, so marginals are preserved
   exactly (not "to the CDF's resolution"), and both components can enter the
   same `z` additively — which is why a two-component (persistent +
   transient) split is available here without a second mechanism.
6. **Fable's residual decomposition (moment estimators, train split;
   attenuated diagnostics, not the estimate).** gaussian_mlp_v2
   teacher-forced standardised residuals: mean +0.0056, var 0.9975
   (calibrated). Cross-member within-(episode, round, group) share
   **+0.0263** (15,090 pairs; cluster-bootstrap 95 % CI [+0.0157, +0.0376],
   SE 0.0057); ICC(1) one-way +0.0433 over 1608 cells; by round thirds
   +0.0227 / +0.0195 / +0.0364 (mild late-game growth, flat enough for a
   constant rho). Cross-member same-group other-round share averaged over
   all lags **-0.0006** (325,764 pairs; CI [-0.0084, +0.0066]); lag 1
   +0.006, lag 2 +0.003, no lag above +0.013. Cross-group same-round
   (episode-level) -0.0041. Same-agent cross-round +0.0662 (individual
   persistence — real, but §6/#158 say it cannot move CG). Test split
   (10 episodes, check only): within +0.0353, other-round +0.0110, lag 1
   +0.055 — noisier and a little higher, train is the convention.
7. **Censoring attenuation measured.** Feeding known (rho_p, rho_t)
   through `clip(rint(mu + sigma z))` at *fixed* teacher-forced (mu, sigma)
   and re-reading the moment shares: (0.10, 0) -> (+0.094, +0.104 within);
   (0.10, 0.10) -> within +0.195, persistent +0.092; (0.20, 0.10) -> +0.282 /
   +0.177; at (0, 0) the moment reads +0.005 / +0.004. So the moment shares
   are attenuated 3-12 % relative to the latent correlation, more at higher
   doses — small, but exactly the bias the interval-censored MLE removes, and
   the reason the MLE (not the moment) sets the dose (C2).
8. **Orchestrator-supplied probes (scratchpad, nothing in the repo), and
   an orchestrator correction recorded honestly.** Probe 1, residual ICC on
   the train split (7457 rows, 15,090 within pairs — the same pair count as
   PR #165, so the data path agrees): all valid rows rho_within 0.02776
   (se ~0.0081), cross-member lag-1 0.00637 (se ~0.0059); interior rows only
   (y not 0/20, 78.4 %): rho_within 0.04060, lag-1 0.01684. Probe 2,
   reduced-form Monte Carlo `c_it = clip(rint(a c_i,t-1 + b m_g,t-1 + k +
   sigma z_it), 0, 20)` at a = 0.7316, b = 0.2015, sigma = 3.410, n = 8,
   4000 episodes, 4-round burn-in, two-component z, arms labelled by the
   non-zero weight:

   | rho_p | rho_t | ratio | SD gmean | SD ind | CG | band | kind |
   |---|---|---|---|---|---|---|---|
   | 0 | 0 | 0.6551 | 3.413 | 5.209 | 7.293 | > 5 | independent |
   | 0.00637 | 0.02138 | 0.7071 | 3.822 | 5.405 | 5.329 | > 5 | mixed (lag-1 split) — the falsifier |
   | 0.01684 | 0.02375 | 0.7506 | 4.233 | 5.640 | 3.684 | 2-5 | mixed (interior split) |
   | 0.02776 | 0 | 0.7747 | 4.516 | 5.829 | 2.771 | 2-5 | **persistent only — the candidate** |
   | 0 | 0.02776 | 0.6733 | 3.528 | 5.240 | 6.604 | > 5 | transient only |
   | 0.04 | 0 | 0.8013 | 4.821 | 6.016 | 1.765 | 1-2 | persistent only |
   | 0.07 | 0 | 0.8443 | 5.382 | 6.375 | 0.139 | <= 1 | persistent only |

   The first version of this table sent to Fable had the two single-component
   arms' *tags* swapped (numbers right, labels wrong), and the orchestrator
   read off the tags a conclusion that persistence was a liability
   ("clipping compresses a persistent offset back out"). Fable's real-model
   rollout (Note 9) found the opposite sign independently; the re-labelled
   table agrees with it. The mislabelled conclusion is retracted; what
   stands is the mechanism (a+b = 0.9331 amplifies a constant group offset
   in the closed loop, which is why a persistent latent pays) and the
   table itself, which is a *prediction*: the independent arm sits 0.036
   below the real sim (0.6551 vs 0.6915), so it is a delta predictor, and
   the fitted dose is simulated whatever the table implies.
9. **Fable's closed-loop proxy with the real model agrees with the
   corrected table.**
   Rolling out the actual gaussian_mlp_v2 adapter against the actual PR #160
   punisher adapter (fixed 4/4 groups, no switching, 100 episodes x 24
   rounds, env round order mirrored), with the two-component z
   monkeypatched onto `_sample_levels` in the scratchpad: rho = 0 -> ratio
   0.7216 at seed 42 (seeds 43-45: 0.7366 / 0.7261 / 0.7361, mean 0.7329,
   sd 0.006) vs the real sim's 0.6915 — switching/regrouping costs ~0.03-0.04
   of ratio, so this too is a delta predictor. Transient 0.026 -> 0.7338
   (+0.012; seeds mean +0.013, sd 0.002). **Persistent 0.026 -> 0.7894
   (+0.068; seeds mean +0.068)**, SD(group means) 3.74 -> 4.44, SD(indiv)
   5.18 -> 5.63, SD(participant means) 2.98 -> 3.85 (the CA diagnostic
   moves only here), mean contribution unchanged 9.7. Transient 0.07 ->
   +0.030; persistent 0.07 -> +0.123 (0.8442 — within 0.0001 of the MC's
   0.8443 at the same dose, and nowhere near the bounds). So in the real
   closed loop the persistent share is worth ~5x the transient one per unit
   and is not clipped away; the MLP's nonlinearity and the punisher's
   response damp the reduced form's open-loop gain of 15 to something the
   0/20 bounds never see. Applied to the real sim's 0.6915 with the
   rollout's deltas: persistent ~0.028 -> ~0.76 (CG ~3.4); the two-component
   falsifier reading -> ~0.70 (CG ~5.5). Two independently built proxies
   agreeing on sign and magnitude is the basis for the structural choice
   in the Declaration.
10. **Free-running loss reproduced for this model, and the identification
    caveat.** Teacher-forced one-shot sampling on the human train rows gives
    ratio 0.8121 (independent) vs the human 0.8473 on the same rows —
    gaussian_mlp_v2 reproduces 96 % of the human ratio when fed real
    histories and only 0.69 free-running, #149's finding on a second model
    family; a within-round share of 0.026 moves the one-shot ratio just
    +0.003, which is why a one-step preflight cannot gate this (C4). Second,
    the group-mean features absorb a persistent group culture into the
    fitted dynamics (the parent's a+b went 0.77 -> 0.93 when they were
    added), so a residual-based lag-1 estimate is pulled toward zero
    relative to the culture's true persistence, while #165 recorded the
    `prev_contribution` feedback pulling phi upward. These two observations
    are grounds 2 and 3 of the Declaration's structural choice: the lag-1
    estimate is still computed and reported as the falsifier, but it is
    not what sets the structure.
11. **Pitfalls the plan must respect.** (a) The joblib bundles unpickle only
    with `scripts/baselines` on `sys.path` (`gaussian_regressor` module);
    `linear_ah.py` inserts it, standalone scripts must too. (b)
    `_sample_levels(Xs, n_levels)` has no group argument — the group array
    is `self._group[t]`, recorded by `_record` from `state["agent_group"]`
    (post-arrival), and `n_levels` must be `self.n_contributions` (21),
    since continuous bundles carry `n_levels = 0`. (c) The legacy Gaussian
    path draws `th.randn(len(mu))` in float32 once per call; the new path
    must be gated at the call site so that stream is untouched when the
    fields are absent (the PR #160 tests treat this as an invariant). (d)
    `prepare_data()` drops the (episode, agent, round) indices — estimator
    and diagnostic build rows from `create_torch_data` + `build_feature_pool`
    directly. (e) `test_copula_rho_rejected_on_gaussian_mlp` and the
    `__init__` assert both currently forbid Gaussian copula fields — revised
    in their own step (3). (f) The worktree has no `.venv`; the SLURM
    template prints no provenance, so step 10 adds the PROVENANCE line
    rather than relying on a side probe. (g) `scripts/remote_test.sh` on an
    isolated dir fails the `test_eval_*` tests for lack of `plots/` (parent
    Note 16) — known, not a regression.
12. **Iteration budget.** No training anywhere: one refit of the incumbent
    setting is 0.70 s (measured), the estimator is seconds plus a low-minute
    bootstrap, each proxy arm ~30 s (measured), the single Raven sim ~2.5
    min (parent). Far under §5's 3x rule.
13. **Revision round (before validation).** The orchestrator retracted the
    "persistence is a liability" constraint after finding the Probe 2
    labels swapped (Note 8) and replaced the two-component estimated split
    with a single fully persistent per-(episode, group) latent at
    `rho_p = rho_total`, `rho_t = 0`, on the four grounds now in the
    Declaration. Changed as a result: the dose bullet (no boundary rule, no
    split), the persistence bullet, the pre-registered predictions (the
    two-component reading kept as the falsifier), RCD added as a second
    guard, step 6 (lag-1 is reported, not stamped), step 7 (stamps
    `rho_total` / 0.0), step 8 (four arms from one code path), step 11 (the
    reading the verdict adjudicates). The sampler (step 4) and tests (step
    5) are unchanged: the general two-component code path stays.
14. **Plan validation (orchestrator, 2026-09-02).** Validated against §2
    (targets are band upgrades on declared rows plus the mean), §5 (the
    change is a sampling-time dependence structure whose one parameter is
    estimated on the training split; no feature, seed, episode-count or
    scoring change) and §8 (nothing on the frozen surface — step 2's only
    contact with `src/aimanager/evaluation_suite/` is a read-only import of
    `load_human` / `load_sim` / `GROUP_CELL`). Implementer tags attached per
    §9 Roles: steps 4, 5, 6, 8, 10 and 11 to Opus — the sampler, its tests,
    the estimator, the pre-registered prediction and the cluster run are
    where a silent error would be unrecoverable — and steps 1, 2, 3, 7, 9 to
    Sonnet. All 16 names step 6 imports from `punishment_copula_rho` were
    checked to exist in this branch's copy (`build_rows`, `blocks`,
    `pair_index`, `cdf_bounds`, `rect_points`, `pair_nll`, `rho_mle`,
    `mle_on_rows`, `bootstrap_mle`, `icc_oneway`, `roundtrip`,
    `spread_ratio`, `preflight`, `bvn_cdf`, `check_bvn`, `save_bundle`);
    `cross_pairs` does not exist here (it is on PR #165's branch), hence the
    local `lag1_cross_pairs`. The band arithmetic was re-derived
    independently and reconciles with `evaluation/metrics.csv`: human ratio
    0.8480163543652899, parent 0.6915274426731914, gap
    0.15648891169209844 = the metrics file's `d`.
15. **Three additions made at validation**, each recorded in the step it
    changed. (a) Step 6's `(0.03, 0)` round-trip arm is promoted to an
    explicit power test of the falsifier estimator: whether the lag-1 fit
    can recover a persistence it is shown is the difference between the
    falsifier being live and ground 3 being confirmed empirically. (b) Step
    10's provenance line must respect `run_simulation.sh` being a
    `.format()` template — no single braces, rendered output inspected once
    — and lands as its own commit. (c) Step 10 gains an **activation
    check**: the fetched `per_round.parquet` must differ from the parent's.
    An inactive copula would reproduce the parent bit-for-bit under the same
    seed and marginals, so bit-identity is the signature of the failure that
    voided four of PR #168's runs, and it is cheaper to test for than to
    discover afterwards.
16. **Step 1 confirmed (orchestrator, 2026-09-02).** Venv built; all five
    inputs are real files, not LFS pointers (bundles 3934 / 3816 bytes, the
    two baseline CSVs, the parent's `per_round.parquet` 112609 bytes).
    Starting point pinned: `pytest tests/baselines/` **244 passed**, the five
    `test_eval_*` modules **70 passed**. `pytest src/aimanager/tests/` as a
    whole cannot be collected locally — `test_encoder` and `test_edge_encoder`
    import PyG, which per CLAUDE.md is Raven-only — so the eval-suite modules
    are named explicitly rather than selected with `-k eval`.
17. **Step 3 confirmed (orchestrator, 2026-09-02).** The gate is open and
    behaviour is provably unchanged: `copula_rho_p` / `copula_rho_t` are read
    tolerantly, restricted to `target == "contribution"` with a
    `gaussian`/`gaussian_mlp` model, and bounded by `rho_p + rho_t < 1` so the
    idiosyncratic weight `sqrt(1 - rho_p - rho_t)` stays real and non-zero.
    PR #160's `copula_rho` field and both of its asserts are byte-for-byte
    untouched, so the punisher path is unaffected. 244 -> 255 tests, black and
    flake8 clean. The load-bearing one is
    `test_copula_rho_p_t_no_behaviour_change_yet`: with a non-zero pair set,
    output and RNG consumption are still bit-identical to the fields being
    absent. That is a step-3 invariant by design and step 4 must update it
    when the sampler starts honouring the fields — a test that keeps passing
    after step 4 would mean the sampler is never reached.
18. **Step 2 confirmed (orchestrator, 2026-09-02).** Part A reproduces the
    direction exactly: human ratio 0.8480163543652899 (SD group means
    5.35768983626258 / SD individual 6.317908621317357), base sim
    0.7427252620205091, candidate sim 0.6915274426731914 with gap
    0.15648891169209844 — equal to the CG row's `d` in the candidate's
    `evaluation/metrics.csv` to 1e-12. The deficit is under-dispersion at the
    group level in both sims, and the candidate is *further* below human than
    the base, which is the finding this experiment is built on.
19. **The ratio band edges are indicative to ~0.1%, not exact.** The
    diagnostic's `d` is a point estimate on the full data, while
    `scores.csv`'s numerator is the mean over the 500 resampling repeats:
    0.15648891169209844 vs 0.15632768652820284 for the candidate (0.10%) and
    0.10529109234478073 vs 0.10521433000149152 for the base (0.07%). So the
    quoted thresholds (ratio > 0.7157729563639177 for `2-5`, > 0.795118995164741
    for `1-2`) are accurate to about +-0.0002 in ratio units. Immaterial at
    the margins this experiment works with, but the verdict comes from
    `evaluation/scores.csv`, never from a ratio computed against these edges.
20. **Two things in the moment diagnostics that shape what step 6 can claim.**
    First, *individual* persistence dominates *group* persistence: same-agent
    cross-round residual correlation +0.06618 (n 84031) against cross-member
    lag-1 +0.00620 (n 28714), while the cross-group same-round share is
    -0.00437 (n 10271) — so there is no episode-level common shock, and the
    shared structure really is group-scoped. Second, the lag-1 estimate is
    the *unstable* one: on the test split it comes out at +0.05139, larger
    than that split's own within-cell +0.03301, on 10 episodes. Combined with
    the 21.58% censored share on train (27.70% on test), that is direct
    evidence the falsifier estimator is low-powered, which is exactly what
    step 6's `(0.03, 0)` round-trip arm is there to quantify rather than
    assume. Within-cell by round thirds: +0.02271 / +0.01942 / +0.03493 — a
    mild rise in the last third, the weak echo of PR #149's round-growing rho.
21. **Step 4 confirmed (orchestrator, 2026-09-02).** `_sample_levels_gaussian_copula`
    is in and the independent path is provably untouched — the only line
    removed from the file is the old call site, now the `else` branch of the
    gate. Verified by the implementer against the pre-change file loaded from
    `git show HEAD`, not by inspection: the fields-absent adapter reproduces
    the old levels bit-for-bit on four rounds at seed 42 and leaves the RNG at
    the same position (next draw 0.13314196467399597). Marginal preservation
    was measured *against its own noise floor*, which is the right calibration:
    max abs per-agent bin-frequency deviation 0.0063 over 40,000 draws, where
    two independent runs of the *independent* sampler at different seeds
    differ by 0.0065 — no detectable marginal change. Dependence structure at
    `rho_p = 0.3`: within-group pair correlation +0.2770 [0.2560, 0.2991]
    against +0.0010 independent, cross-group +0.0013 — the 0.277 against a
    latent 0.3 is `rint`/`clip` attenuation, so **level-based correlation
    recovery runs ~8% low and a 0.02 tolerance on it would be too tight**.
22. **Mutation testing found the step-4 tests have no teeth on the two
    invariants that matter, and both gaps sit on the candidate's own
    configuration.** I broke the sampler two ways and the full suite stayed
    green at 256 passed both times: (a) drawing `zv` only when `rho_t > 0`,
    which destroys stream invariance, and (b) replacing the
    `_copula_z.setdefault` with an assignment, which redraws the persistent
    latent every round and destroys persistence altogether. The reason is
    visible in the test: the only behavioural case uses
    `copula_rho_p=0.2, copula_rho_t=0.1`, both non-zero, whereas **the
    declared candidate runs `rho_t = 0.0`** — the one configuration no test
    exercises. Mutation (b) matters most: it would silently convert the
    experiment into the transient-only arm, which the Declaration predicts
    lands at CG ~6.6, and nothing would have caught it. Both mutations are
    handed to step 5 as acceptance criteria — its test module must fail under
    each. Worth stating why (a) is a real defect even though `sqrt(0) * v`
    contributes nothing to the candidate: the arms in step 8 are only
    comparable if they share an RNG stream, so a stream that depends on the
    weights confounds the very comparison the preflight exists to make.
23. **`z` is replayable, which step 5 should use instead of a monkeypatch.**
    The draw order is fixed and documented, so a test can seed, call
    `predict`, re-seed, regenerate `zu, zv, eps` in order, apply the `pick`
    map and reconstruct `z` bit-for-bit; the implementer verified the replay
    reproduces `predict`'s levels exactly and that the reconstructed `zu`
    entries equal the stored `u_g`. Recovering the two weights separately
    needs multi-round episodes: within-round within-group correlation targets
    `rho_p + rho_t`, cross-round cross-member targets `rho_p` alone.
24. **Step 5 confirmed (orchestrator, 2026-09-02).** 24 tests in
    `tests/baselines/test_contribution_group_copula.py`; `tests/baselines/`
    256 -> **280 passed**, the five `test_eval_*` modules still 70. Both
    Note-22 mutations are now caught, and I re-ran the dangerous one myself
    rather than taking it on report: mutation B (`setdefault` -> assignment)
    fails 6 tests — `test_correlation_recovery_from_replayed_z[0.1-0.1]` and
    `[0.15-0.0]`, `test_persistent_latent_constant_within_episode[0.15-0.0]`
    and `[0.2-0.1]`, `test_group_that_empties_and_reforms_resumes_its_latent`,
    `test_switcher_draws_from_the_receiving_group` — with the source restored
    to the identical md5 (`da1903d0...`) and an empty `git diff`. Mutation A
    fails 2, including the `rho_t == 0.0` stream check that is the candidate's
    own configuration.
25. **The sampler is now independently validated by replay.** Reconstructing
    the pre-clip `z` from the documented draw order recovers the two weights
    separately: at `(0.10, 0.10)` within-round within-group 0.2075 against a
    target 0.20 and cross-round cross-member 0.1106 against 0.10; at
    `(0.15, 0.00)` 0.1552 and 0.1535 against 0.15 and 0.15; cross-group
    -0.0074 / +0.0044 against 0. Replay bit-identity with `predict`'s levels
    is asserted first, so those numbers cannot be an artefact of a wrong
    replay. Marginal preservation was tested twice over: a binomial test
    against the *exact analytic* marginal (worst 2.51 / 3.37 / 2.24 SEs over
    98 bins, where four independent-sampler runs score 2.38-2.67 on the same
    bins) and the Note-21 noise-floor comparison with the floor recomputed
    in-test.
26. **A real trap step 5 caught for step 6, relayed mid-flight.** Marginals
    are preserved *across* episodes, not within one: `_copula_z` freezes a
    group's `u_g` until `_reset_history()` clears it, so driving the sampler
    in a plain repeated-call loop generates **one giant episode** with a
    single frozen offset per group. Step 6's round-trip panels must reset per
    synthetic episode and match the episode structure the estimator then keys
    cells on, or the persistent arms are misconstructed and the `(0.03, 0)`
    power test — the arm that decides whether the declared falsifier is live —
    measures the wrong thing. Sent to the step-6 implementer while it was
    still running.
27. **Step 6 confirmed: the dose is `rho_total = 0.04378520865574197`**, used
    as-is, so `rho_p = 0.04378520865574197` and `rho_t = 0.0`. Cluster
    bootstrap over the 40 episodes, 200 resamples, seed 38381: 95% CI
    [0.032281937767127324, 0.06507099556615693], SE 0.008769066352832287;
    pairwise LR against rho = 0 is 46.506379232683685. The two hard
    provenance checks pass: within-cell **n_pairs = 15090**, identical to
    PR #165 and step 2, and `check_bvn` max deviation against scipy's mvn is
    3.3306690738754696e-16. Censored share 0.21577041705779804; residual mean
    +0.005617, var 0.997453. The dose sits inside the pre-registered
    0.03-0.05 expectation.
28. **The declared falsifier inverted, and the power test settled it against
    my own ground 3.** The censored MLE of the lag-1 cross-member pairs is
    **0.03634366202384774** (n 28714, CI [0.009304553955279456,
    0.061219727158390254], excluding 0) — 83% of `rho_total`, i.e. the
    dependence is almost entirely *persistent*. The moment version of the
    same pair set is +0.00620, an order of magnitude smaller. The `(0.03, 0)`
    power test decides which to believe: driven through the real sampler on a
    purely persistent panel, the lag-1 fit recovers **0.03410270756524159**
    (6 panels, sd 0.00715; panel-0 CI [0.020882239028975094,
    0.05450848823201269], excluding 0). **The estimator has power.** So
    Declaration **ground 3 is retracted** — the lag-1 estimate is *not*
    attenuated by construction, it simply disagreed with its own moment
    version, and the censored fit is the one with the better properties.
    Grounds 1 (PR #150's A/B), 2 (teacher-forced 0.812 vs free-running
    0.6915) and 4 (§5 tie-break) stand unchanged, and the data now supports
    the persistent structure independently of them.
29. **The honest cost of Note 28: this experiment has lost its falsifier.**
    The Declaration's discriminating test was that the two-component reading
    predicts CG ~5.3 while the persistent reading predicts ~2.8, so a
    measured CG near 6.6 would convict the structural choice. But the
    two-component reading *as the censored MLE actually estimates it* is
    (0.03634366202384774, 0.007441546631894225) — 83% persistent, which is
    essentially the candidate. Both readings now predict the same move, so
    CG can no longer discriminate between them. The remaining ways this run
    fails are real but different: the closed-loop dynamics not delivering
    what the open-loop proxies predict, or a collateral cost that keeps the
    mean above 1.6145149045441503. Recorded here rather than quietly dropped,
    because the pre-registration is weaker than it was when written.
30. **The prediction therefore sharpens upward, to a two-band upgrade.** At
    0.0438 the dose sits by Note 8's 0.040 row — ratio 0.8013, CG 1.765,
    band `1-2` — not the 0.0278 row that gave the Declaration's headline
    CG ~2.8. Step 8 must predict against 0.0438, not 0.028. Still on the
    undershoot side (`<= 1` needs ~0.07 by the same table), so overshoot
    remains a non-risk.
31. **The one finding that limits what the dose means: the dependence is not
    exchangeable-Gaussian in shape.** Same-group pairs co-occupy the
    censoring bounds far more often than the fitted copula allows —
    within-cell both-at-0 2.50x and both-at-20 3.49x independence, lag-1
    2.58x and 3.54x, against 0.99x / 1.06x on synthetic panels at the same
    fitted rho. Per-pair likelihood ratio on human data vs a synthetic arm at
    the same rho: within-cell 0.00308 vs 0.00173 (1.8x), lag-1 0.04677 vs
    0.00116 (**40x**). So the lag-1 "persistence" is substantially *persistent
    same-group boundary clustering* — groups locking together at 0 or at 20
    across consecutive rounds — rather than a clean Gaussian latent, and that
    is why its MLE and its moment estimate point in opposite directions while
    agreeing on every synthetic panel. The within-cell dose's own excess is
    mild, so the dose is the sounder of the two numbers, which is fortunate
    given it is the one being used. A latent that reproduces boundary
    clustering directly, rather than through a Gaussian copula, is the
    successor this points at.
32. **Two methodological points worth carrying forward.** (a) The
    per-episode-reset warning relayed from step 5 was material, not
    hypothetical: re-run without it, the `(0.03, 0)` arm collapses to 0.0071
    total / 0.0121 lag-1 (bias -0.023 / -0.018) and would have **failed** the
    round-trip gate, which would have read as a sampler bug. (b) Each arm is
    the mean of 6 independent panels with common random numbers, because a
    single panel's own sd is 0.006-0.019 — the first single-panel `(0.1, 0)`
    draw came out at +0.0215 bias, which alone would have failed a correct
    implementation. Round-trip max |bias| over the four arms is
    0.008091217083174201, PASS against the 0.02 tolerance. The panels re-draw
    contributions on fixed human feature rows, so this validates the
    estimator, not the closed-loop dynamics.
33. **Constant rho is an approximation, and the round profile says so.**
    Round-thirds MLE 0.0271281091298599 / 0.03559872... / 0.07688 — a 2.8x
    rise across the game, the sharper version of step 2's mild moment rise
    and of PR #149's round-growing rho. Interior-only within-cell MLE 0.0647
    against all-rows 0.0438. Test split (10 episodes) within-cell MLE 0.0927
    and lag-1 0.0309, with the Note-20 moment impossibility reproduced
    exactly; noise indicators only, not estimates. A round-dependent rho is a
    seeded follow-up, not a change to this experiment.
34. **Step 9 confirmed (orchestrator, 2026-09-02).** Written in front rather
    than dispatched, because it gates the run and the check is a diff. Against
    the parent config, exactly three lines differ — `contribution_model`,
    `output_dir`, `figure_name` — and every protocol field is byte-identical
    (seed 42, `n_episodes` 100, `n_rounds` 24, `n_episode_steps` 24,
    `switch_every` 4, `agent_groups` [0,0,0,0,1,1,1,1], `n_contributions` 21,
    `n_punishments` 31, `save_per_round: true`), including the shared
    `valid_model` and the single `lin_multinomial_copula_self` pairing.
    `evaluation_sweep.py`'s `DIR_PATTERN` parses the new directory name to
    `contr = gaussian_mlp_v2_group_copula`, `switch = gnn`, so the sweep
    convention still holds.
35. **Step 7 confirmed (orchestrator, 2026-09-02).** `contribution_gaussian_mlp_v2_group_copula.joblib`,
    sha256 **da42031ab0ca5bc2ea355036f2e07f5dd37b369d49bf0bf6999c9ebe68d0ea7c**
    (re-checked here, and to be re-checked again on Raven in step 10). Base
    bundle sha256 matched the value the params sidecar recorded. 21 keys in,
    33 out: 12 added, none removed, and the 19 directly comparable shared keys
    are value-equal. The two that are not comparable by `==` are the
    `estimator` and `scaler` objects, so I checked those the way that
    actually matters — on all 7457 train rows the stamped bundle's scaler
    output, `predict` and `predict_std` are **bit-identical** to the base
    (`np.array_equal`). The model is numerically untouched; only the
    dependence structure differs. Read back through `LinearAHAdapter`:
    `copula_rho_p = 0.04378520865574197`, `copula_rho_t = 0.0`, accepted by
    the step-3 gate; the deterministic (`sample=False`) path returns identical
    levels to the base adapter over a 6-round sequence including a switch.
36. **A correction to the plan's own text:** step 7 called the stamped bundle
    "LFS-tracked like its siblings". It is not — `.gitattributes` tracks only
    `*.csv`, `*.parquet` and `*.pt`, so none of the joblib bundles are, and
    this one commits as a plain ~4 KB binary exactly like
    `contribution_gaussian_mlp_v2_best.joblib`. `.gitattributes` was not
    touched.
37. **One check of mine was malformed and is recorded so the log is not
    misleading.** I first tried to verify "every pre-existing key is the
    identical object" by loading both bundles separately and testing `is`.
    That cannot work: two independent `joblib.load` calls unpickle to distinct
    objects, so the test reports False regardless of correctness. The
    identity claim is only meaningful in-process at construction time, where
    the stamper asserts it; across a reload the right check is numerical
    bit-identity, which is what Note 35 reports.
38. **THE PRE-REGISTERED PREDICTION, recorded before the simulation is
    submitted.** At the fitted dose (`rho_p = 0.04378520865574197`,
    `rho_t = 0.0`, read from the stamped bundle, not retyped):
    **CG ~ 1.91, band `1-2`** — a two-band upgrade on the parent's
    5.91060457046713 — corresponding to a predicted sim ratio of 0.7975
    (0.6915274426731914 + 0.1059). That is proxy B's number, the closed-loop
    rollout of the real contribution and punisher bundles, and it is the one
    I stand behind. **The two proxies span CG 0.22 (`<= 1`) to 1.91 (`1-2`)**;
    both are two-band upgrades. Secondary: CA is predicted to improve, its
    `std_diff` moving from -2.621590294686563 toward about -1.24, because
    every persistent arm raises SD(participant means) (2.9787 -> 4.3600 in
    proxy B) while the transient-only arm barely does (+0.0715) — exactly the
    asymmetry the Declaration pre-registered. Gate 2 has a wide margin on
    these numbers: CG alone would take the 21-row mean from
    1.6145149045441503 to roughly 1.42.
39. **Both proxies calibrate, in opposite directions, which is why only
    deltas are used.** Proxy A's rho = 0 arm gives 0.6558 against Note 8's
    0.6551 and reproduces its other arms to ~0.001; proxy B's gives 0.7216,
    matching Note 9's 0.7216 to four decimals despite being independently
    rebuilt. The real sim sits at 0.6915, *between* them — A is 0.036 low, B
    is 0.030 high — so every prediction above applies the proxy's delta to
    the parent's real baseline ratio rather than the proxy's own level. Both
    remain open-loop approximations: A has a linear mean and no punisher, B
    fixes group membership, disables switching and treats every contribution
    as valid where the real stack has a GNN validity model. Neither carries
    the switch-slot interaction. One incidental confirmation: Note 8's "n = 8"
    means two groups of four; a single group of eight gives 0.502 and does not
    reproduce it.
40. **Dose-response around the fitted dose (proxy B, `rho_t = 0`):** ratio
    0.7216 / 0.7876 / 0.8277 / 0.8472 / 0.8644 at `rho_p` = 0 / 0.02 / 0.0438
    / 0.06 / 0.08, i.e. implied CG 5.92 / 3.42 / 1.91 / 1.17 / 0.52. Local
    slope ~1.49 of ratio per unit of `rho_p` (about 0.56 of CG score per 0.01
    of dose), clearly concave. So the verdict is not knife-edge in the dose:
    anything from 0.02 upward band-upgrades CG, and the fitted dose's CI
    [0.03228, 0.06507] maps entirely inside `1-2` to `2-5`.
41. **An honest discrepancy with Note 30, not reconciled away.** Note 30 read
    proxy A's *absolute* level off its 0.040 row (0.8013 -> CG 1.765). Step 8's
    delta rule instead applies proxy A's delta (+0.1507, 42% larger than
    proxy B's) to the real baseline, which pushes A's prediction to CG 0.22 —
    materially more optimistic. Proxy B lands at 1.91, within 0.15 of Note
    30's expectation, so the faithful proxy confirms Note 30 and proxy A
    brackets it from the optimistic side. Overshoot past `<= 1` is possible on
    A's reading, not on B's. Nothing here changed the dose.
42. **One implementation point that makes the arms comparable.** The (0,0)
    reference is forced through `_sample_levels_gaussian_copula` by a
    script-local adapter subclass, because `predict`'s gate would otherwise
    route it to the independent sampler at 1n draws instead of 3n and the arms
    would sit on different RNG streams — the deltas would then carry that
    difference. This is exactly what the unconditional-3n-draw invariant
    (Note 22) exists to permit; at (0,0) the copula sampler is algebraically
    the independent one. `linear_ah.py` was not modified. Also worth noting
    for the guards: SD(individual) rises too (5.1800 -> 5.9287 in proxy B).
    That is free-running feedback on the trajectories, not a marginal change —
    the per-agent conditional law is preserved by construction — but it means
    the C-family rows will move, and the under-dispersed ones (CA -2.62,
    CC -1.86, CE -2.21 std_diff) should move toward human.
43. **Step 10 confirmed, and the run's provenance is clean.** Job 29869716
    (dir `8fca66b7`), ExitCode 0:0, 2:56 elapsed on ravg1005. All three
    validity checks pass. The PROVENANCE line reads
    `/raven/u/certuer/algorithmic-institutions/.venv/bin/python` and
    `/u/certuer/autoresearch/contribution-gmlp-group-copula/src/aimanager/__init__.py`
    — shared venv, **isolated dir's source**, which is exactly the failure
    mode PR #167 note 8 and PR #168 were bitten by. The bundle's sha256 on
    Raven is `da42031ab0ca5bc2ea355036f2e07f5dd37b369d49bf0bf6999c9ebe68d0ea7c`,
    equal to the local stamp. And the activation check passes: the fetched
    `per_round.parquet` (sha `72599716...`) differs from the parent's
    (`dbb4343f...`), so the copula branch was genuinely taken — bit-identity
    there would have meant an inactive sampler and a void run.
44. **VERDICT: SUCCESS, on both gates, with two target band upgrades.**
    Gate 1 twice over: **CG 5.91060457046713 -> 2.8292447** (`> 5` -> `2-5`,
    the declared primary) and **CA 2.1657982689458293 -> 1.6416427**
    (`2-5` -> `1-2`, a declared secondary). Gate 2: mean
    **1.6145149045441503 -> 1.3138927788530981**, down 0.3006221256910522.
    Rows <= 1 go 8/21 -> 10/21 (context only). Five rows change band and
    **every one improves; none regresses**: the two targets plus PD
    1.051227 -> 0.836562 (`1-2` -> `<= 1`), RCB 2.056138 -> 1.910008
    (`2-5` -> `1-2`) and RCD 1.191308 -> 0.670100 (`1-2` -> `<= 1`). SC
    2.866508 -> 2.669271 improved but stayed in band, so it is not one of the
    upgrades.
45. **The pre-registered prediction was one band optimistic, and that is the
    most useful number in this log.** Note 38 predicted CG ~1.91 (`1-2`) from
    a delta of +0.1059; the realised delta is **+0.081718** (ratio
    0.691527 -> **0.773246**), giving CG 2.827. So the closed-loop proxy
    **over-predicted the delta by 30%**. The likely mechanism is what proxy B
    left out: it fixes group membership and disables switching, but the real
    simulation reshuffles groups every 4 rounds, so an agent carries the
    *arrival* group's latent and a per-(episode, group) latent's coherence is
    partially scrambled at the agent level; the real stack also runs a GNN
    validity model the proxy ignores. Both dilute the numerator relative to
    the denominator. The same 30% shows up in CA, predicted at `std_diff`
    ~-1.24 and realised at **-1.5958** — consistent over-prediction, not two
    unrelated misses. **Calibration for future preflights: scale a
    no-switching closed-loop rollout's ratio delta by ~0.77 before quoting a
    prediction.**
46. **The mechanism did what it claimed, on the diagnostics that are not
    scores.** All three under-dispersion `std_diff` values move toward human
    together: CA -2.621590 -> -1.595834, CC -1.860802 -> -0.968395, CE
    -2.205269 -> -1.552298. That is the signature of shared group variance
    rather than of a wider marginal, and it is why the C block improves as a
    block (CC 1.666 -> 1.115, CD 1.309 -> 1.140, CE 1.298 -> 1.059, CF 1.722
    -> 1.300) instead of trading off against CG. The §6 anti-correlation tax
    did not materialise — as in PR #159, and unlike the mean-function
    approaches.
47. **The RCA guard behaved as designed and reported no bug.** RCA
    3.999531 -> 3.493432, an improvement inside `2-5`. Marginal preservation
    predicted no large move, and there was none; the improvement comes from
    the trajectories, not the marginals. RCD is the more informative guard:
    1.191308 -> 0.670100, a band upgrade, siding with PR #165 (contribution
    latent improves RCD) against PR #168 (switch latent costs it) — a
    switcher assimilating toward the receiving group's latent is real
    switching pull, not diluted selection.
48. **Everything that got worse, in full.** Only two rows move by more than
    0.05 in the wrong direction and both stay at ceiling: CB 0.724767 ->
    0.983765 (round mean contributions — the shared latent adds
    round-to-round wobble in the group means) and RSA 0.774043 -> 0.956174
    (switching after punishment), the latter being exactly the "RSA dilution"
    PR #150 recorded for a shared latent: co-switching that is not
    punishment-driven. Both remain `<= 1`. The rest is noise: SA +0.0038,
    PA +0.0084, PB +0.0181. No row changed band for the worse.
49. **Where this leaves the slot.** CG at 2.829 is still the largest deficit
    in the stack, and Note 40's dose-response says the remaining gap is dose,
    not mechanism: the fitted 0.0438 delivered 52% of the ratio gap, and the
    same rollout puts `<= 1` near `rho_p` 0.08 — roughly twice the MLE. Two
    honest routes, neither of which is raising the dose by hand: Note 31's
    finding that the dependence is **not exchangeable-Gaussian** (same-group
    co-censoring at 2.5-3.5x independence, lag-1 per-pair LR 40x synthetic),
    which points at a latent that reproduces boundary clustering directly;
    and Note 33's **round-growing rho** (thirds MLE 0.0271 / 0.0356 / 0.0769,
    a 2.8x rise), which points at a round-dependent or lock-in latent and
    echoes PR #149's seed. SC 2.669 remains the switch slot's problem.
50. **The governing gate-2 rule changed after this experiment closed, and the
    verdict is unaffected.** `main` commit `b174f90` relaxes §2 gate 2 from
    "the mean score improves" to "the mean may not rise more than 10% above
    the evaluation stack's baseline mean", so the ceiling for this experiment
    becomes 1.7759663949985653 rather than the Declaration's pinned
    1.6145149045441503. The candidate's mean is 1.3138927788530981, which
    passes under both readings, and gate 1 is untouched by the change. The
    Declaration is left as pre-registered — it committed to the stricter of
    the two rules and met it — rather than rewritten to the looser one.
    Merged into this branch so the log and the protocol it cites agree; the
    commit is documentation only and touches no code the experiment depends
    on.
