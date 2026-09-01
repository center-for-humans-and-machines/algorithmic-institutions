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

1. [x] **[Sonnet] Port the isolation tooling and calibration machinery** — in
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

2. [x] **[Sonnet] Check the diagnostic's three inputs, in place** (replaces the
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

3. [x] **[Opus] Diagnostic: how did the latent damp net flow?** — local analysis
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

4. [x] **[Sonnet] Stamp the grid artifacts on Raven** — a new script
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

5. [x] **[Sonnet] Grid sim configs** — one per variant,
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

6. [x] **[Opus] Isolated parallel simulations** — `AI_REMOTE_DIR='~/autoresearch/switch-herding-copula-recal'`
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

7. [x] **[Sonnet] Fetch + evaluate all variants** — `AI_REMOTE_DIR=... scripts/fetch_cluster.sh`
   each output dir, then `python -m aimanager evaluate <config>` locally per
   variant. Fill one Results row per variant, scores exact, losers included.
   **Selection rule, declared here before any score is seen**: among
   variants passing BOTH gates (SC < 2.0 and mean < 1.2893632310269196),
   the winner is the lowest mean; ties broken by lower SC, then by smaller
   rho (§5, ties go to the simpler model). If no variant passes both gates
   the experiment is a `[FAIL]` and the dose-response curve is the finding.
   Commit sim outputs + evaluations.

8. [x] **[Opus] Verdict, log, PR** — §2 on the selected variant (gate 1:
   SC < 2.0; gate 2: mean < 1.2893632310269196), exact values both ways.
   Push, PR `--base auto/contribution-herding-copula-v2`, title
   `[SUCCESS]`/`[FAIL]`, body per §9.7 (hypothesis, results table with ALL
   grid rows, collateral +/-). Delete the remote isolated dir after the PR
   opens.

## 2a. Plan revision after step 3: the frozen grid

Validated by the orchestrator and committed before any stamping or
simulation. The grid is fixed at four arms; `copula_switch_every = 4`
throughout; nothing but `copula_rho` and `copula_phi` varies.

| k | rho | phi | one-line rationale |
|---|---|---|---|
| 1 | 0.035037571771319845 | 1.0 | minimum dose, static latent: one founding group becomes the episode's persistent exporter and the other a stable core, the human pattern on exporter concentration (0.204 vs the parent's 0.153) and co-membership (0.635 vs 0.575) |
| 2 | 0.116482333585783 | 1.0 | #166's exact rho with only phi moved to the boundary — the strict one-parameter test that exporter-role *rotation*, not dose, caused #166's failure |
| 3 | 0.21502476987895425 | 1.0 | top of #162's CI, static: the largest defensible dose toward the human round-3 bimodality (P(k = 4 of 4) 0.19 vs the parent's 0.125) |
| 4 | 0.21502476987895425 | 0.11702191363043744 | max dose, memoryless: tests whether the right tail is buyable round-locally, with no group-level memory |

Design: phi = 1.0 is the primary lever, with a three-point rho
dose-response at phi = 1.0 (arms 1-2-3) and two clean one-parameter phi
contrasts — arm 2 against #166's already-measured (0.116, 0.704), and arm 4
against arm 3 — which together give the phi ladder 0.117 / 0.704 / 1.0.

Orchestrator validation:

- **Bounds.** rho takes #162's CI lower bound, its point estimate and its
  CI upper bound exactly; phi takes the CI lower bound and the static
  boundary 1.0. Every value is inside the Declaration's envelope, and
  phi = 1.0 is the boundary the Declaration pre-authorised (the parent's
  own contribution copula runs there). Nothing needed an out-of-CI
  justification, and the diagnostic explicitly declined to chase the
  rho ~ 0.294 that its round-3 algebra points at, on the grounds that the
  contribution-spread tax scales with the latent weight — the required
  dose is logged as a finding instead of dosed into a near-certain
  gate-2 failure. I agree with that call.
- **Legality (§5).** Sampling parameters only; no retraining, no code
  change, no new mechanism; seeds, episode count and the protocol
  untouched; the grid is frozen here, before any of its simulations run,
  and all four arms will be reported whatever they score.
- **Semantics of phi = 1.0 verified in code, not assumed.**
  `environment.py` calls the switch predictor with
  `reset_rnn = (round_number == 0)`, so `GraphNetwork._copula_z` is
  cleared once per episode and carried across rounds; the AR(1) advances
  only on decision rounds. phi = 1.0 therefore means one latent per
  (episode, group label), frozen for the whole episode — which is the
  mechanism the grid's headline relies on.
- **Each arm is falsifiable.** Every arm carries a prediction in
  `grid.json` that makes a null result informative: if arm 2 leaves CG
  near 0.59 the tax is rho-driven and no positive dose can pass gate 2;
  if arm 4 matches arm 3, phi is irrelevant and the successor needs a
  round-dependent rho.
- **Diagnostic reproduced independently** before accepting it (Notes 9).

## 2b. Plan revision after step 6: the isolation was never real

The four grid simulations ran to COMPLETED 0:0 and produced four
**byte-identical** `per_round.parquet` files, identical also to PR #166's
run — five different (rho, phi) settings, one set of bytes. The cause is
not the model and not the grid; it is the job launcher.

**The bug.** `scripts/run_simulation.sh` (the SLURM template, reproduced
verbatim into every generated `run.sh`) activates a venv but never sets
`PYTHONPATH`. An isolated experiment dir has no venv of its own and
borrows the canonical checkout's, whose *editable install* resolves
`aimanager` to `~/algorithmic-institutions/src/aimanager`. So a job
submitted from an isolated dir runs the **shared checkout's** code, not
the experiment's. `simulate_cluster.sh` does export the right variables at
submit time, but they are lost between the local shell and the job (the
same loss #166 hit with `AIMANAGER_VENV`, which the `.venv` symlink
papered over — there is no symlink trick for `PYTHONPATH`).

At the time of these runs the shared checkout was on an unrelated
`punisher-ar-copula` tree, which has **no copula code at all**:
`aimanager.generic.copula` does not exist there, `predict_independent`
has no copula branch, and `GraphNetwork.__init__` takes `**kwargs`, so
`copula_phi` and `copula_switch_every` were silently swallowed and
`copula_rho` was consulted only on a path these artifacts never take.
Hence four clean exits and total inertness of every stamp.

**Evidence** (probe job 29858573, reported in Notes 13): re-running arm 1
with no `PYTHONPATH` — byte-equal to the real jobs' environment —
reproduces the collision hash `a9a446e4…` exactly, with **0** calls to
`sample_correlated_levels`; re-running arms 1 and 3 with
`PYTHONPATH=$PWD/src` gives **4801** sampler calls each and two different
outputs. Saturation (the innocent explanation) is ruled out: the switch
probabilities have mean 0.296, max 0.949, and *zero* mass above 0.99 or
below 0.01.

**The fix**, one line in `scripts/run_simulation.sh`:
`export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"` after the venv
activation, mirroring the preamble the stamping SLURM templates already
use. `--chdir=.` puts the job in the submitting dir, so `$PWD/src` is the
experiment's own source; in the canonical checkout the two coincide, so
behaviour there is unchanged. This is launcher plumbing, not model or
scoring code: it changes *which code runs*, which §9's isolation mandate
already requires, and it is what makes the experiment runnable at all.
The parent-control run below is the before/after evidence §4 asks of a
bug fix.

**Revised step 6**, six simulations in parallel instead of four:

- the four frozen grid arms, unchanged;
- **a baseline control**, `23_2g8a_swrecalctl_self_gnncopar1_contr_gnn_switch.yml`
  — the parent's config with only `output_dir`/`figure_name` changed. Under
  the fix it must reproduce the parent's `per_round.parquet` **bit for
  bit** (sha256 4f64fc42…). If it does not, the baseline the gates are
  judged against is not reproducible on this tooling and I escalate
  instead of reporting scores. This is the control that licenses every
  comparison in the Results table;
- **a repair run of PR #166's exact setting** (rho 0.116482333585783,
  phi 0.70366020589033; artifact sha256 9b89f3c7…, ported from the v3
  branch), because #166's published numbers describe the copula-free
  stack and not the mechanism it claimed to test. Reported for the record
  and to complete the phi ladder 0.117 / 0.704 / 1.0; **not eligible for
  selection** — the grid is frozen at four arms and that setting belongs
  to #166.

**The grid stays frozen.** No valid score has been observed: the four void
runs carry no information about any arm's performance, so nothing has been
seen that could bias a re-specification, and the arms are re-run exactly
as committed in `grid.json`.

**What this voids in step 3.** The diagnostic's frame C was #166's run,
i.e. the copula-free stack — so every conclusion drawn from B-vs-C is
void, including the headline z = -6.8 "contribution-spread collapse"
(that is simply the parent's contribution copula being absent, which is
PR #165's effect measured backwards) and the claim that the switch latent
"never delivered co-movement". The H1/H2/H3 verdicts fall with it.
What survives untouched is everything derived from human-vs-parent, since
both are valid runs: the SC deficit being a missing right tail, absorption
0.300 vs 0.091, reaching size 8 from balance 0.219 vs 0.069, the round-3
concentration of human co-switching (r 0.424 vs 0.237) with rounds 7-19
already matched, the between-label anti-correlation accounting for half
the human full-merge rate, and the rho ~ 0.294 implied by the round-3 gap.
The grid's rationale rests on those, plus the parent's own phi = 1.0
precedent — so it stands, but Notes 7 and 8 must be read with Note 14.

**Escalation for the maintainer.** This bug is in `main` (commit 8680db9)
and silently affects **any** experiment that ran a simulation from an
isolated `AI_REMOTE_DIR` while keeping its change in `src/`. PR #166 is
certainly affected: its `[FAIL]` verdict measured the copula-free stack.
Other isolated runs should be re-checked before their results are trusted.

## 3. Results

All runs are the parent's stack with only the switch slot's sampling
parameters changed, in the parent's exact single-pairing config (seed 42,
100 episodes). Gates: SC < 2.0 (band upgrade out of 2-5) and mean <
1.2893632310269196.

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-01 | baseline: parent PR #165 stack (gnn-copula contr x gnn switch x severity-copula punisher) | SC 2.1867905905576634 | 11/21 | 1.2893632310269196 | reference |
| 2026-09-01 | CTL: parent's config re-simulated under the fixed launcher, output bit-identical | SC 2.1867905905576634 | 11/21 | 1.2893632310269196 | control, exact |
| 2026-09-01 | arm 1: switch herding copula, rho 0.035037571771319845, phi 1.0 | SC 1.4531172890278043 | 7/21 | 1.3207710955319005 | **[FAIL]** — gate 1 passed (2-5 -> 1-2), gate 2 failed |
| 2026-09-01 | arm 2: switch herding copula, rho 0.116482333585783, phi 1.0 | SC 1.4170916269123233 | 6/21 | 1.358350400404427 | **[FAIL]** — gate 1 passed (2-5 -> 1-2), gate 2 failed |
| 2026-09-01 | arm 3: switch herding copula, rho 0.21502476987895425, phi 1.0 | SC 0.9587925645665427 | 6/21 | 1.3616236192643985 | **[FAIL]** — gate 1 passed (2-5 -> <= 1, two bands), gate 2 failed |
| 2026-09-01 | arm 4: switch herding copula, rho 0.21502476987895425, phi 0.11702191363043744 | SC 1.6062674782036555 | 11/21 | 1.3377431871380137 | **[FAIL]** — gate 1 passed (2-5 -> 1-2), gate 2 failed |
| 2026-09-01 | REF (not selectable): PR #166's setting, rho 0.116482333585783, phi 0.70366020589033, correctly executed | SC 1.3193781872018437 | 8/21 | 1.3064125900381773 | repair run — gate 1 passed, gate 2 failed |

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

4. (Opus, step 1) The port is exact and the AI_REMOTE_DIR gap is closed:
   the six scripts were byte-identical on HEAD to 8680db9's parent before
   the checkout and byte-identical to 8680db9 after it, so this applies
   that commit's script half with zero drift. `fetch_cluster.sh`,
   `simulate_cluster.sh` and `remote_test.sh` now read
   `REMOTE_PROJECT_DIR="${AI_REMOTE_DIR:-...}"`; the three SLURM templates
   contribute only `source "${AIMANAGER_VENV:-.venv}/bin/activate"` — the
   exact line that killed #166's first job, which is why step 6 symlinks
   `.venv` into the isolated dir rather than trusting the variable to
   survive `sbatch`. Calibration provenance verified: rho
   0.116482333585783, phi 0.70366020589033, rho_ci
   [0.035037571771319845, 0.21502476987895425], phi_ci
   [0.11702191363043744, 2.0829306438837505]; both calibration parquets'
   working-copy sha256 equal the source-branch LFS oids (train
   b0fe6892..., test d61deeee...).

5. (Opus, step 2) All three diagnostic inputs load through the evaluation
   suite's own loaders, so the diagnostic sees exactly the frame SC is
   computed on. Provenance, for reproducibility without the sibling
   worktree: human `experiments/2group_8agent_50ep.csv` sha256
   27c6bdbe417d487461ce1fe2db6e81c5df8bff50a5c8d8f260c3e9b2daf0517a
   (9600 rows / 50 episodes after `load_human` collapses the flip
   augmentation by `pair_id`); parent run `per_round.parquet` sha256
   4f64fc42fe23cad1cf14423f79b8496adff9874e2e546c05a0d7683055960be2;
   #166's run `per_round.parquet` sha256
   a9a446e49a5afca89f5543b200501e0edc234d77affed59ca2bda047de2bcd82.
   Membership is `group_id` (the loader never reads `agent_group`).
   A worktree-local `.venv` was created with `uv sync --frozen`; `uv.lock`
   unchanged.

6. (Opus, step 2) The SC deficit is now localised, before the diagnostic
   proper. Larger-group-size counts (rounds >= 4; human n = 1000, sims
   n = 2000), as shares:

   | size | 4 | 5 | 6 | 7 | 8 | mean |
   |---|---|---|---|---|---|---|
   | human | 9.6% | 24.4% | 28.0% | 23.6% | 14.4% | 6.088 |
   | parent (SC 2.187) | 16.2% | 32.0% | 25.4% | 20.8% | 5.6% | 5.676 |
   | #166 (SC 2.967) | 19.0% | 34.0% | 27.4% | 15.8% | 3.8% | 5.514 |

   The stack's whole SC deficit is a missing right tail: humans sit at
   size 8 (one group emptied) in 14.4% of observations, the parent in
   5.6%. #166's latent moved mass the wrong way at exactly those two
   extremes (8: 5.6% -> 3.8%, 7: 20.8% -> 15.8%) and piled it onto 4-6.
   So the grid is not chasing a mean — it has to produce more *complete*
   exoduses, and the diagnostic must explain why intra-group herding
   produced fewer of them.

7. (Opus, step 3, the diagnostic's answer) **None of the three
   hypotheses was the cause: at rho = 0.116 the switch copula barely
   touched switching at all, and #166's damage was done to the
   contribution slot.** Gross flow is statistically identical across
   human / parent / #166 (2.288 / 2.390 / 2.316) and out-flow variance
   *fell* rather than rose (Var(out_0) 1.706 / 1.364 / 1.238), so the
   latent never delivered the co-movement it exists to add; every
   switch-side difference between the parent and #166 sits at |z| <= 2.0
   under a 400-episode bootstrap. Meanwhile the between-group
   contribution spread ratio collapsed 0.742 -> 0.592 at **z = -6.8**.
   The channel is *who* moves, not how many: the shared latent partly
   overrides individual propensity in selecting the movers, so
   assortativity (own contribution vs destination group mean) fell
   0.278 -> 0.110, the exporter role rotated between labels each
   decision round (concentration 0.153 -> 0.136), group composition
   mixed (pair co-membership 0.575 -> 0.552), and the sorting that
   produces between-group contribution spread was destroyed. That is
   what drove #166's mean from 1.289 to 1.723, and it also removes the
   pressure SC's right tail feeds on. H1 (size-coupled restoring force)
   is real but second-order (restoring slope -0.365 -> -0.445, |z| <= 2);
   H2 (direction flipping) is refuted outright — humans oscillate *more*
   than either sim (lag-1 corr -0.349 vs -0.283 / -0.297) and cancel
   more, so human segregation is not built by directional persistence;
   H3 holds for the human-vs-parent gap (co-switching r at round 3:
   0.424 human vs 0.237 parent, matched from round 7 on) but not for
   #166's own regression, which accumulates evenly across the episode.

8. (Opus, step 3, why the grid targets phi) Two human signatures the
   parent misses are both about *persistence of roles*, not dose:
   humans hold a merged group once formed (P(stay at size 8) 0.300 vs
   the parent's 0.091) and reach it far more often from balance
   (0.219 vs 0.069) — and under label independence the human marginals
   predict only 0.110 of that 0.219, so half the human full-merge rate
   needs between-label anti-correlation. Within the mechanism as built,
   the only lever on role persistence is phi, and #166 itself is
   evidence it works: at phi = 0.704 absorption already rose
   0.091 -> 0.125 even as the overall merge rate fell. phi = 1.0 freezes
   each label's latent for the episode, which is both the absorption
   mechanism and the thing whose rotation measurably destroyed the
   contribution spread. Hence a static-latent grid rather than a pure
   dose ladder.

9. (Opus, validation of step 3) Re-derived the diagnostic's load-bearing
   numbers independently before fixing the grid, from the same three
   frames but my own code: larger-group size 8 shares 0.1440 / 0.0560 /
   0.0380 and size >= 7 shares 0.3800 / 0.2640 / 0.1960, and absorption
   at size 8 exactly 9/30 = 0.300 (human), 2/22 = 0.091 (parent),
   2/16 = 0.125 (#166) — all reproduced. On the contribution channel my
   simpler spread ratio (sd of group means over sd of individual
   contributions) gives 0.536 / 0.374 / 0.331 with sd(group means)
   falling 1.831 -> 1.768 and sd(individual) rising 4.901 -> 5.350: a
   different normalisation from the suite's CG row, same direction and
   same decomposition. The diagnostic is sound.

10. (Opus, findings for a successor, all needing code changes and
   therefore out of scope here) (i) Half the human full-merge rate
   requires **between-label anti-correlation** — one latent shared across
   both labels with opposite signs, so a migration wave has a direction,
   rather than two independent per-label latents. (ii) Human co-switching
   is confined to the founding exodus (r 0.424 at round 3 vs 0.05-0.11
   later, where the parent already matches), so a **constant rho over all
   decision rounds necessarily overdoses rounds 7-19**. (iii) Human
   switching concentrates in persistent individuals (top-2 movers 49.7%
   vs 43.8%), pointing at an **agent-level** rather than group-level
   latent. (iv) Closing the round-3 consensus gap within the current
   mechanism would need rho ~ 0.294, above #162's CI.

11. (Opus, step 4) All four arms stamped on Raven (job 29858081, COMPLETED
   0:0, 31 s) inside the isolated dir — `aimanager.__file__` resolved to
   `/raven/u/certuer/autoresearch/switch-herding-copula-recal/src/aimanager/__init__.py`,
   so the job used this branch's code. The base artifact arrived with
   sha256 184f7f5c8ed326d49983fe455ef6478715fcac79c8161f08fa685b9cfb25d037,
   equal to the `base_artifact_sha256` in #162's calibration provenance.
   Control: the unmodified base loads as (0.0, 0.0, None) through
   `GraphNetwork.load`, so each arm's round-tripped values provably come
   from the artifact and not from `__init__` defaults. Round-trips, all
   with `copula_switch_every = 4` as a genuine int, and all module
   state_dicts `th.equal`-identical to the base:

   | k | copula_rho | copula_phi | sha256 (remote == local) |
   |---|---|---|---|
   | 1 | 0.035037571771319845 | 1.0 | f172cfc84e99db5a... |
   | 2 | 0.116482333585783 | 1.0 | a69316ae8e32712a... |
   | 3 | 0.21502476987895425 | 1.0 | 4d335b3b59a25105... |
   | 4 | 0.21502476987895425 | 0.11702191363043744 | 4ece9b0add0d298e... |

   The stamping script takes no inline parameters — rho and phi are read
   only from the committed `grid.json` — so the freeze is enforced by the
   tooling, not just by the log. Artifacts were fetched back and committed
   locally so the step-6 sync ships identical bytes and `rsync --delete`
   cannot remove them.

12. (Opus, step 5) Four configs, each a byte-copy of the parent's with
   exactly three changed lines (`switch_model`, `output_dir`,
   `figure_name`) — verified by `diff` showing 3 removed / 3 added lines
   and nothing else, and by a recursive YAML dict comparison finding
   exactly those three keys differing. All five configs (parent plus the
   four arms) agree on `seed: 42`, `n_episodes: 100`, `switch_every: 4`,
   `save_per_round: true` and the single `lin_multinomial_copula_self`
   pairing, so every arm's RNG context equals the baseline's — the
   condition #162 showed the manager list controls. The inherited header
   comment, which describes the punisher-severity experiment, is left
   stale deliberately rather than spend a fourth edit. Dir names parse
   under the sweep's DIR_PATTERN as contr `gnncopar1`, switch
   `herdcoprc<k>`.

13. (Opus, step 6, the anomaly) All four arms COMPLETED 0:0 in ~83 s and
   produced one set of bytes (`a9a446e4…`), equal to #166's run and
   different from the parent's. Provenance had looked perfect: correct
   artifact sha256s remotely, correct (rho, phi, 4) round-tripping through
   `GraphNetwork.load` on the cluster minutes before submission, correct
   config path in every job log, the two copula suites 23/23 green. All of
   that was true and all of it was beside the point — those checks ran
   through `PYTHONPATH=$PWD/src` on the login node, while the jobs ran
   without it. The lesson for the protocol: verifying an artifact loads
   correctly *next to* a job is not evidence that the job imported the
   same code. Provenance has to be printed **from inside the job**, which
   the simulation template still does not do.

14. (Opus, step 6, what it costs us) PR #166's published `[FAIL]` numbers
   (SC 2.967, mean 1.723, CG collapse) are not measurements of the switch
   herding copula — they are the copula-free stack, with #165's
   contribution copula and #160's punisher copula equally inert. That also
   voids the parts of this experiment's own step-3 diagnostic that leaned
   on frame C (§2b). It is worth being blunt that the diagnostic's most
   confident, most quantitative claim — a 6.8 sigma channel — was an
   artifact, and that it looked *more* convincing than the human-vs-parent
   findings that actually survived. The tell was there in hindsight: a
   z = -6.8 effect on the contribution slot from a switch-slot parameter
   was too large for its cause, and I accepted it because it explained the
   sign reversal so neatly.

15. (Opus, step 6, the fix and its control) One line in the SLURM template
   exports `PYTHONPATH="$PWD/src"`. Rather than assert that this restores
   the intended behaviour, the re-run includes a control that re-simulates
   the parent's own config under the fixed tooling: it must reproduce the
   parent's `per_round.parquet` bit for bit. This branch changes no file
   under `src/` (`git diff` against the parent over `src/` is empty), so a
   bit-identical control is exactly the right expectation, and it is what
   licenses judging the arms against the parent's `scores.csv`.

16. (Opus, step 6 re-run) Six jobs (29858758/59/60/66/67/68), all
   COMPLETED 0:0, every generated `run.sh` carrying the PYTHONPATH export,
   no traceback in any log. **The control passes: CTL's
   `per_round.parquet` is
   4f64fc42fe23cad1cf14423f79b8496adff9874e2e546c05a0d7683055960be2,
   bit-identical to the parent's** — so the fixed tooling reproduces the
   baseline exactly and the gate comparisons are licensed. All six digests
   are mutually distinct and the old collision hash appears nowhere:
   arm1 aaee236c…, arm2 1e504971…, arm3 ad0756e9…, arm4 983c6392…,
   CTL 4f64fc42…, REF 0bdd714a…. Determinism cross-check for free: arms 1
   and 3 had already been run once by the diagnostic probe under the same
   corrected import path, and both reproduced their digests exactly here.

17. (Opus, step 7-8, the verdict) **[FAIL], but on gate 2 only — the
   mechanism does work on its target.** Every one of the four arms
   band-upgrades SC, and arm 3 upgrades it by *two* bands, 2.1867905905576634
   -> 0.9587925645665427, i.e. from "clear deviation" to at or below the
   human-vs-human noise ceiling. That is the largest SC move the campaign
   has produced. Not one arm improves the 21-row mean: 1.3207710955319005 /
   1.358350400404427 / 1.3616236192643985 / 1.3377431871380137 against the
   parent's 1.2893632310269196. Per the selection rule declared in step 7
   before any score was seen — winner is the lowest mean *among arms
   passing both gates* — there is no winner and the experiment is a
   `[FAIL]`. Nothing here is a near miss to be talked up: the closest arm
   is 0.031 of mean above the bar, and the bar is what stops a target row
   being bought with the rest of the stack.

18. (Opus, the dose-response, which is the finding) At phi = 1.0 the
   trade-off is clean and monotone in rho: SC 1.4531 -> 1.4171 -> 0.9588
   as rho goes 0.035 -> 0.116 -> 0.215, while the mean moves the other way,
   1.3208 -> 1.3584 -> 1.3616. So segregation is buyable, at a
   quantified and steadily rising price. The phi contrasts show
   persistence is genuinely the lever the grid bet on: at rho = 0.215,
   phi = 1.0 buys SC 0.9588 where phi = 0.117 only reaches 1.6063. But the
   ladder is not monotone in phi — at rho = 0.116, the intermediate
   phi = 0.704 (REF) beats the static phi = 1.0 (arm 2) on *both* axes,
   SC 1.3194 vs 1.4171 and mean 1.3064 vs 1.3584. The best mean of all five
   copula runs is REF's 1.3064, and it still misses gate 2. A successor
   should read this as: the (rho, phi) plane has an interior optimum near
   rho ~ 0.12, phi ~ 0.7, and no point in it pays for itself under gate 2.

19. (Opus, the correction to PR #166) With the launcher fixed, #166's exact
   setting scores **SC 1.3193781872018437** — a band upgrade — not the
   2.967099851949342 regression that PR reports, and mean 1.3064125900381773
   rather than 1.7229430164446164. #166's `[FAIL]` title happens to survive,
   because gate 2 still fails, but every substantive claim in it is wrong:
   the switch copula did not reverse sign, did not damp segregation, and did
   not collapse the contribution slot. Its Notes 12 and 13 — "the two latents
   do not compose", "the net effect on segregation reverses sign" — describe
   an artifact of running the copula-free stack, and should not be carried
   forward. The premise this experiment was commissioned on (recalibrate
   because the ported parameters reverse sign) was therefore false; what was
   actually needed was to run them at all.

20. (Opus, collateral) The tax is concentrated and interpretable, not
   diffuse. **RCD, switching pull** — how strongly a switcher's contribution
   adapts toward the gap with their new group — is the one row that changes
   band for the worse, 1.9647336396755046 -> 2.78 / 2.68 / 3.03 / 2.63
   (1-2 into 2-5) in all four arms. That is the same channel the step-3
   diagnostic guessed at from a void comparison, now measured on valid runs:
   a shared latent decides *who* moves partly by group draw rather than by
   individual propensity, so switchers are less well matched to the group
   they join and the pull coefficient drifts from the human value. Second,
   the C-family individual-fit rows drift up together (CA/CB/CC/CD all cross
   from <= 1 into 1-2 in arms 1-3), which is why rows <= 1 falls from 11 to
   7/6/6 even as SC improves — the known CG-vs-individual-fit anti-correlation
   of §6, appearing on the switch slot. Arm 4 is the exception that proves
   the reading: with persistence nearly off it holds rows <= 1 at 11/21 and
   buys the least SC. Movements toward human worth noting: CG improves in
   arm 1 (4.1635 -> 3.9756), PD improves in arm 3 (1.0029 -> 0.8363) and in
   REF (-> 0.7908), and CE improves in every arm.

21. (Opus, for the successor) The switch herding copula is now proven to be
   a real and strong lever on SC, so the open question is no longer whether
   group-level co-movement helps segregation but how to stop it taxing RCD
   and the C rows. The step-3 findings that survived the void comparison
   point the same way as the RCD damage does: human co-switching is
   concentrated in the founding exodus (pairwise r 0.424 at round 3 versus
   0.05-0.11 later, where the parent already matches), so a constant rho
   across all five decision rounds necessarily overdoses rounds 7-19 —
   a round-dependent rho, high at round 3 and near zero after, is the
   obvious next experiment and it targets exactly the rows that broke.
   Beyond that, half the human full-merge rate needs between-label
   anti-correlation (0.219 observed vs 0.110 predicted from independent
   labels), which needs a cross-group latent and hence a code change.
