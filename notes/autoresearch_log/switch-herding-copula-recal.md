# Autoresearch log: switch-herding-copula-recal

Branch: `auto/switch-herding-copula-recal` (worktree
`.claude/worktrees/switch-herding-copula-recal`), based on the parent
`[SUCCESS]` PR #165 (`auto/contribution-herding-copula-v2`), per §9 of
`notes/autoresearch.md`. Successor to the `[FAIL]` PR #166
(`auto/switch-herding-copula-v3`): same mechanism, recalibrated for the
stack it runs in.

## 1. Declaration

- **Slot**: switch.
- **Base model**: GNN switch predictor
  `artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`
  — the switch model of the parent stack.
- **Parent PR**: #165, `auto/contribution-herding-copula-v2`
  (maintainer-designated). Evaluation stack and baseline are the parent's
  own confirmed run,
  `plots/simulation/23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch/evaluation/scores.csv`:
  **mean 1.2893632310269196, rows <= 1: 11/21**.
- **Target row**: **SC 2.1867905905576634** (band 2-5; a band upgrade
  means < 2.0, into 1-2). Gate 2: 21-row mean < 1.2893632310269196.
- **Prior evidence**: PR #166 ported #162's parameters
  (rho = 0.116482333585783, phi = 0.70366020589033, calibrated against
  *independently sampled* contributions) unchanged into this stack and
  the effect **reversed sign** — SC 2.187 -> 2.967, CG 4.16 -> 9.73,
  PD 1.00 -> 2.51, mean 1.289 -> 1.723 — while the switch marginals held
  (SA/SB), i.e. the mechanism executed as built but the dose/persistence
  calibrated for one input process is wrong for another. Separately
  calibrated latents do not compose.
- **Hypothesis**: the switch herding latent is still the right mechanism
  for SC (founding exodus, human mean net flow 2.42 vs sim ~1.5 — the
  one confirmed S deficit), but its parameters must be set *conditional
  on the contribution copula being active*: with contributions already
  co-moving, the switch model's inputs carry part of the herding that
  #162's residual rho was calibrated to add, so the correct residual
  dose is different (and plausibly smaller or differently persistent).
  A recalibrated (rho, phi) should band-upgrade SC without the CG/PD
  collateral #166 measured.
- **Planned change** (one change: the switch-slot copula parameters,
  selected by declared hyperparameter search — legal per §5, "selecting
  between variants by their evaluation score"): first a diagnostic on
  #166's committed run to characterize *how* the latent damped net flow,
  then a small grid (at most 4 variants) of (rho, phi) values fixed by
  that diagnostic — each within the mechanism as built (shared
  per-(episode, group) Gaussian latent, AR(1) across decision rounds,
  agents' own Bernoulli marginals preserved) and within #162's
  calibration CIs (rho CI [0.035037571771319845, 0.21502476987895425],
  phi CI [0.11702191363043744, 2.0829306438837505] truncated to [0, 1])
  unless the diagnostic justifies a boundary value (as the parent's
  phi = 1.0 ruling did). Every variant simulated once in the parent's
  exact single-pairing config (seed 42, 100 episodes — never touched),
  every variant's full scores reported; the best variant's single
  evaluation is the claim. No retraining anywhere.
- **Honesty constraints, declared upfront**: the grid is fixed and
  committed *before* any of its simulations run — no extending the grid
  after seeing scores, no re-draws of any variant, no seed or episode
  changes, all variants reported in the Results table including the
  losers. If no variant passes both gates, the experiment is a `[FAIL]`
  with the dose-response curve as its finding.
- **Wall-clock**: zero training; the grid simulations run in parallel on
  Raven (distinct output dirs, one isolated remote dir), so total
  wall-clock is roughly one simulation plus queue time (~1-2 h) —
  within budget.

## 2. Plan

Validated and implementer-tagged by the orchestrator per §9. Baseline
figures in §1 were re-verified against the parent's `scores.csv` before
tagging (21 rows, mean 1.2893632310269196, rows <= 1: 11/21,
SC 2.1867905905576634 — all exact). Three validation findings are applied
in the steps below and argued in Notes 2:

- (a) The draft's step 2 committed #166's `plots/simulation/` outputs into
  this branch. Dropped: the local worktree
  `.claude/worktrees/switch-herding-copula-v3` already holds that run's
  real (LFS-smudged) `per_round.parquet`, so the diagnostic reads it in
  place; committing another experiment's outputs would put them in this
  PR's diff, which §9 wants restricted to this experiment's own change.
  Step 2 becomes an input check instead.
- (b) The draft's step 4 allowed "a plain local torch check ... if PyG
  blocks local load". That is not possible: the base switch `.pt`
  unpickles `torch_geometric` modules, so even `th.load` fails on macOS
  (the ported script's own header says RUNS ON RAVEN ONLY). Stamping
  therefore runs on Raven via sbatch in the isolated dir, and the
  artifacts are fetched back and committed locally — which also protects
  them from the sim sync's `rsync --delete`.
- (c) The frozen grid is committed as machine-readable JSON and the
  stamping script reads only that file, so the freeze is auditable rather
  than narrative.

1. [ ] **[Sonnet] Port the isolation tooling and calibration machinery** — in
   the worktree, `git checkout auto/switch-herding-copula-v3 --` the six
   AI_REMOTE_DIR-aware scripts (`scripts/fetch_cluster.sh`,
   `scripts/remote_test.sh`, `scripts/simulate_cluster.sh`,
   `scripts/run_simulation.sh`, `scripts/artificial_humans/run_training.sh`,
   `scripts/manager/run_training.sh` — the script half of `main`'s 8680db9,
   byte-identical port; confirm with
   `git diff HEAD 8680db9^ -- <the six>` being empty before the checkout),
   the calibration provenance
   (`artifacts/artificial_humans/switch_pred_herding_copula/calibration/`,
   all three files) and the three scripts
   (`scripts/artificial_humans/dump_switch_probs.py`,
   `scripts/artificial_humans/make_switch_copula_artifact.py`,
   `scripts/baselines/switch_copula_rho.py`). Verify LFS content is real
   (working-copy sha256 == the pointer oid on the source branch) and that
   `copula_params.json` reads rho = 0.116482333585783,
   phi = 0.70366020589033, rho_ci
   [0.035037571771319845, 0.21502476987895425], phi_ci
   [0.11702191363043744, 2.0829306438837505]. Nothing outside `scripts/`
   and `artifacts/` is touched. Commit. **Until this commit lands, no
   cluster script may be run** — without it `AI_REMOTE_DIR` is silently
   ignored and the syncs would `rsync --delete` into the shared checkout.

2. [ ] **[Sonnet] Check the diagnostic's three inputs, in place** (replaces the
   draft's plots port) — confirm all three frames load locally and report
   their shapes / key columns: the human reference
   `experiments/2group_8agent_50ep.csv` (single copy after de-duplicating
   the flip augmentation), the parent's run
   `plots/simulation/23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch/per_round.parquet`
   (this worktree), and #166's run
   `../switch-herding-copula-v3/plots/simulation/23_2g8a_full_copula_v3_self_gnncopar1_contr_herdcopar1_switch/per_round.parquet`
   (the sibling worktree — read-only, never written). Record the three
   paths' sha256 in the log so the diagnostic is reproducible without the
   sibling worktree. No files added; log entry only.

3. [ ] **[Opus] Diagnostic: how did the latent damp net flow?** — local analysis
   (pandas, no cluster) over the three frames of step 2, on the switch
   decision rounds (3, 7, 11, 15, 19) and the SC support (rounds >= 4).
   Quantify per (episode, decision round): gross out-flow per group,
   **net** flow and its sign relative to the group mean contribution gap,
   within-group co-switching (same-direction pairs, and the full
   distribution of "k of the group's eligible members switched"), the
   realized larger-group-size distribution and where its EMD mass sits,
   and the split of SC's deficit between the first decision round
   (founding exodus) and rounds 7-19. The deliverable is a Notes entry
   answering, with numbers: does rho > 0 damp *net* flow monotonically in
   this stack (i.e. intra-group herding is symmetric and cancels in the
   difference), or is the damage in phi (a persistent latent re-exporting
   from the group it just filled, undoing exodus across rounds), or in the
   round-1 draw specifically? Write the analysis as a committed script
   under `scripts/data_analysis/`. From the answer, fix a grid of **at
   most 4** (rho, phi) variants, each with a one-sentence behavioral
   rationale, each inside #162's CIs (rho in
   [0.035037571771319845, 0.21502476987895425], phi in
   [0.11702191363043744, 1.0] after truncation) unless the diagnostic
   explicitly justifies a boundary value. Emit the grid as
   `artifacts/artificial_humans/switch_pred_herding_copula_recal/grid.json`
   (a list of `{"k", "rho", "phi", "rationale"}`). Submit grid + rationale
   as a plan revision for orchestrator validation; the revision and
   `grid.json` are committed BEFORE any stamping or simulation.

4. [ ] **[Sonnet] Stamp the grid artifacts on Raven** — a new script
   `scripts/artificial_humans/make_switch_copula_recal_artifacts.py`,
   modelled on the ported `make_switch_copula_artifact.py` but reading its
   arms **only** from the committed `grid.json` (no inline parameters), and
   a `stamp_copula_recal.slurm` following the existing
   `stamp_copula.slurm` pattern (shared venv,
   `PYTHONPATH=$PWD/src`). For each variant k: base
   `switch_pred_opt_50ep_doubled_reanchored`, output
   `artifacts/artificial_humans/switch_pred_herding_copula_recal_<k>/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`
   with (`copula_rho`, `copula_phi`, `copula_switch_every = 4`) stamped;
   every module state_dict `th.equal`-identical to the base and every
   pre-existing non-module key unchanged; `GraphNetwork.load` round-trips
   the three fields. Runs in the isolated remote dir via `sbatch` (never
   the login node). Then `fetch_cluster.sh` the stamped artifacts back and
   commit them locally, so the step-6 sync ships identical content and
   cannot delete them. Commit scripts + artifacts.

5. [ ] **[Sonnet] Grid sim configs** — one per variant,
   `configs/simulation/manager_testing/23_2g8a_swrecal<k>_self_gnncopar1_contr_herdcoprc<k>_switch.yml`,
   each a byte-copy of the parent's
   `23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch.yml`
   with EXACTLY three edits (`switch_model`, `output_dir`, `figure_name`).
   The manager list (single `lin_multinomial_copula_self` pairing), seed 42,
   `n_episodes: 100`, `switch_every: 4`, `save_per_round: true` and every
   other key stay untouched, so each variant's RNG context equals the
   baseline's (§3; #162's finding that the manager list shifts the draw).
   The inherited header comment is left stale by design rather than spend a
   fourth edit. Variant tokens carry no extra `_self_`/`_contr_`/`_switch`
   and no underscores, so the dir names parse under the sweep's
   `_self_(\w+?)_contr_(\w+?)_switch$`. Commit.

6. [ ] **[Opus] Isolated parallel simulations** — `AI_REMOTE_DIR='~/autoresearch/switch-herding-copula-recal'`
   on every remote call. Order: `squeue` PENDING check (other experiments'
   isolated dirs are disjoint, but verify) -> create the dir -> `ln -s
   ~/algorithmic-institutions/.venv` inside it (v3 Notes 9-10: isolated
   dirs have no venv and the first job dies in 0 s at
   `.venv/bin/activate`) -> one syncing `simulate_cluster.sh` call, which
   ships `artifacts/` -> **before submitting**, in the isolated dir:
   `remote_test.sh --test-only` for
   `test_switch_copula_graph.py` and `test_contribution_copula_graph.py`
   (`--test-only` is mandatory: a syncing re-run would `rsync --delete` the
   artifacts back out, v3 Note 7) plus a `GraphNetwork.load` round-trip on
   each stamped variant. No `src/` file changes in this experiment, so
   #166's fully green suites already cover identical code; this is a smoke
   check on the new artifacts. Then submit all variants in parallel
   (`--no-sync`, distinct output dirs), and wait with ONE blocking
   foreground call,
   `ssh raven 'while squeue -j <ids> -h | grep -q .; do sleep 30; done;
   sacct -j <ids> --format=JobID,State,ExitCode -n'`, generous timeout,
   re-issued if it times out while jobs still run. The template prints no
   provenance, so verify it afterwards: remote sha256 of all three slot
   artifacts per job and `aimanager.__file__` resolving inside the isolated
   `src/`. One draw per variant, seed 42, no re-draws.

7. [ ] **[Sonnet] Fetch + evaluate all variants** — `AI_REMOTE_DIR=... scripts/fetch_cluster.sh`
   each output dir, then `python -m aimanager evaluate <config>` locally per
   variant. Fill one Results row per variant, scores exact, losers included.
   **Selection rule, declared here before any score is seen**: among
   variants passing BOTH gates (SC < 2.0 and mean < 1.2893632310269196),
   the winner is the lowest mean; ties broken by lower SC, then by smaller
   rho (§5, ties go to the simpler model). If no variant passes both gates
   the experiment is a `[FAIL]` and the dose-response curve is the finding.
   Commit sim outputs + evaluations.

8. [ ] **[Opus] Verdict, log, PR** — §2 on the selected variant (gate 1:
   SC < 2.0; gate 2: mean < 1.2893632310269196), exact values both ways.
   Push, PR `--base auto/contribution-herding-copula-v2`, title
   `[SUCCESS]`/`[FAIL]`, body per §9.7 (hypothesis, results table with ALL
   grid rows, collateral +/-). Delete the remote isolated dir after the PR
   opens.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|

## 4. Notes

1. (Fable, opening) #166 established: mechanism executes correctly in
   this stack, marginals preserved, loading contract sound, isolation
   tooling works — the open variable is purely (rho, phi). The
   diagnostic (step 3) exists because the failure was a *sign reversal*,
   not an overdose: if rho > 0 turns out to damp net flow monotonically
   in this stack, no positive dose can pass and the grid should probe
   phi and the smallest defensible rho instead; the grid must encode
   what the diagnostic finds, not hope.

2. (Opus, plan validation) Validated against §2/§5/§8 and tagged. Legality
   is clean: only the switch slot's two sampling parameters change, no
   retraining, protocol/seed/episode count untouched, grid frozen and
   committed before its simulations, all variants reported — §5 names
   "selecting between variants by their evaluation score" as legal
   explicitly. Two defects fixed rather than discovered later: the draft
   would have committed #166's simulation outputs into this branch (the
   sibling worktree already has them, so the diagnostic reads in place),
   and it allowed a *local* torch round-trip of the stamped artifact,
   which cannot work — the base `.pt` unpickles `torch_geometric`
   modules, so stamping and its verification both belong on Raven, with
   the artifacts fetched back and committed so the sim sync's
   `rsync --delete` cannot remove them. Added an auditable freeze: the
   grid lives in `grid.json` and the stamping script accepts no inline
   parameters.

3. (Opus, plan validation) Read the mechanism before tagging, so the
   diagnostic has a sharp question. The latent's cell is
   `batch_index * 2 + agent_group` (`GraphNetwork._predict_encoded_copula`)
   — one latent per (episode, current group), advanced by the AR(1) only
   on decision rounds, and switching is the latent's UPPER tail
   (`levels_from_u`). With 4 agents per group, per-group herding raises
   Var(out_A) and Var(out_B) while leaving both means intact, and the two
   latents are independent, so Var(net flow) should rise and E|net| with
   it — naively *more* segregation, yet #166 measured less. The candidate
   explanations are therefore dynamic, not static: symmetric mass
   exchange (both groups export together, high gross flow and zero net),
   persistence re-exporting from the group just filled, or the
   size-boundedness of a group that has already emptied. Step 3 has to
   separate these with numbers, not pick one.
