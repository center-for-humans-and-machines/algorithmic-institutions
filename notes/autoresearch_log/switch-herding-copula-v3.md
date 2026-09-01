# Autoresearch log: switch-herding-copula-v3

Branch: `auto/switch-herding-copula-v3` (worktree
`.claude/worktrees/switch-herding-copula-v3`), based on the parent
`[SUCCESS]` PR #165 (`auto/contribution-herding-copula-v2`), per §9 of
`notes/autoresearch.md`. Completes the copula triptych in one stack:
punisher severity copula (PR #160) + contribution herding copula (PR #165)
+ switch herding copula (this experiment, the mechanism of PR #150/#162).

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
- **Target row**: **SC 2.1867905905576634** (band 2-5, the stack's worst
  S row; a band upgrade means < 2.0, into 1-2). Success additionally
  requires the 21-row mean to drop below 1.2893632310269196 (§2 gate 2).
- **Hypothesis**: eligible players' switch decisions co-move beyond what
  shared observable features explain (founding exodus: human mean net
  flow 2.42 vs sim ~1.5), and the segregation they build persists across
  an episode's decision rounds; independent per-agent switch sampling
  pins SC to the independence floor. A shared per-(episode, group)
  Gaussian latent at weight rho, carried across decision rounds by an
  AR(1) at phi, pushed through each agent's own Bernoulli marginal,
  should band-upgrade SC — marginals (SA/SB) preserved by construction.
- **Planned change**: sampling-time only — no retraining, no
  recalibration. Port PR #162's calibrated, stamped switch artifact
  (`switch_pred_herding_copula_ar1`, **rho = 0.116482333585783,
  phi = 0.70366020589033**, pairwise-likelihood MLE on the human train
  split, reproduced bit-for-bit in #162) plus its calibration provenance
  into this branch, and swap it into the parent stack's switch slot. The
  sampler code (`src/aimanager/generic/copula.py`,
  `GraphNetwork` copula path in `src/aimanager/generic/graph.py`) is
  already on this branch — the parent generalized #162's machinery to
  both heads; this branch's assertion explicitly admits
  `y_name == "does_switch"` with phi in [0, 1].
- **Known risk, declared upfront**: PR #162 found the SC draw straddles
  the 2.0 band edge and is RNG-context-dependent (the config's manager
  list shifts every draw — `MultiManager.get_punishments` consumes RNG
  for all configured managers). The evaluation context is therefore
  pinned to the parent's exact single-pairing config, per §3; whatever
  SC that draw produces is the verdict. In #162 the mechanism moved SC
  2.816 -> 2.188 from a non-copula-contribution stack; here it starts
  from 2.187 on a stack whose contribution slot already co-moves, so the
  two latents interact — that interaction is what this experiment
  measures.

## 2. Plan

Validated and implementer-tagged by the orchestrator per §9. Validation
findings, applied below: (a) the original step 3 (compatibility check)
forward-referenced the original step 4 (isolated remote setup) for the
environment it needs, so the two are swapped — remote setup and tests run
first, the compatibility check second; (b) the original step 4 named local
test paths that do not exist (`tests/switch`, `tests/copula`) — the switch
and contribution copula suites are `src/aimanager/tests/test_*_copula_graph.py`
and are PyG-dependent, so they run on Raven; local pytest covers the
eval-suite suites only. Baseline figures in §1 were re-verified against the
parent's `scores.csv` before tagging (mean 1.2893632310269196, rows <= 1:
11/21, SC 2.1867905905576634 — all exact).

1. [x] **[Sonnet] Port the switch-copula calibration provenance** — in the experiment
   worktree, `git checkout origin/auto/switch-herding-copula-v2 --`
   `artifacts/artificial_humans/switch_pred_herding_copula/calibration/copula_params.json`,
   `.../calibration/switch_probs_train.parquet`,
   `.../calibration/switch_probs_test.parquet`,
   `scripts/artificial_humans/dump_switch_probs.py`,
   `scripts/artificial_humans/make_switch_copula_artifact.py`,
   `scripts/baselines/switch_copula_rho.py`. No edits, no re-derivation:
   the parameters carry over unchanged from #162 (calibration there
   reproduced #150 bit-for-bit). Verify `copula_params.json` still reads
   rho = 0.116482333585783, phi = 0.70366020589033. Commit.

2. [x] **[Sonnet] Port the stamped switch artifact** — `git checkout
   origin/auto/switch-herding-copula-v2 --
   artifacts/artificial_humans/switch_pred_herding_copula_ar1/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`
   (Git LFS; `git lfs pull` as needed). Integrity gate: the working-copy
   file's sha256 must equal the LFS pointer oid recorded on
   `origin/auto/switch-herding-copula-v2` for that path. Commit.

2b. [x] **[Opus] Port the AI_REMOTE_DIR isolation tooling** *(plan revision, added
   after step 1 — see the "Plan revision" section below)* — this branch
   predates `main`'s commit 8680db9, so `simulate_cluster.sh`,
   `fetch_cluster.sh` and `remote_test.sh` still hardcode the shared remote
   checkout and ignore `AI_REMOTE_DIR`; steps 3/6/7 would silently sync into
   `~/algorithmic-institutions`. Port the script half of 8680db9 only:
   `git checkout 8680db9 -- scripts/fetch_cluster.sh scripts/remote_test.sh
   scripts/simulate_cluster.sh scripts/run_simulation.sh
   scripts/artificial_humans/run_training.sh
   scripts/manager/run_training.sh`. All six files on this branch are
   byte-identical to 8680db9's parent, so this applies the commit's script
   changes with no drift and no conflict. Its `notes/autoresearch.md` change
   is deliberately NOT ported (that file has diverged on this branch and is
   the maintainer's process doc). No behavior change without
   `AI_REMOTE_DIR` set. Commit.

3. [x] **[Sonnet] Isolated remote setup + tests** *(was step 4; moved ahead of the
   compatibility check, which needs this environment)* — create the isolated
   dir via `AI_REMOTE_DIR='~/autoresearch/switch-herding-copula-v3'` (all
   `train_cluster.sh`/`simulate_cluster.sh`/`fetch_cluster.sh` calls from
   this worktree export it; the shared checkout is never touched — check
   `squeue` for PENDING jobs before any sync anyway). Run the PyG suites
   there on the login node: `src/aimanager/tests/test_switch_copula_graph.py`,
   `test_contribution_copula_graph.py`, plus the standard src suites
   (`test_encoder`, `test_edge_encoder`, `test_environment`,
   `test_linear_manager`). Locally, run the eval-suite suites
   (`src/aimanager/tests/test_eval_*.py`) and `scripts/tests/`. Green before
   anything runs; log entry only.

4. **[Opus] Compatibility check: the #162 artifact under the #165 code** *(was
   step 3)* — the stamped `.pt` was written by #162's
   `make_switch_copula_artifact.py`; this branch's `GraphNetwork` is the
   parent's generalized version. In the isolated remote dir (step 3), load the
   artifact via `GraphNetwork.load` and assert the three stamped fields
   round-trip (`copula_rho = 0.116482333585783`,
   `copula_phi = 0.70366020589033`, `copula_switch_every = 4`) and
   construction passes the `does_switch` head gate. Failure here means the
   stamping/loading contract drifted between the branches — stop, report, and
   re-stamp from the base artifact with this branch's script conventions
   instead (a plan revision, not an improvisation). Notes entry either way.

5. [x] **[Sonnet] Sim config** —
   `configs/simulation/manager_testing/23_2g8a_full_copula_v3_self_gnncopar1_contr_herdcopar1_switch.yml`:
   byte-copy of the parent's
   `23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch.yml`
   with exactly three edits — `switch_model` ->
   `artifacts/artificial_humans/switch_pred_herding_copula_ar1/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`,
   `output_dir` ->
   `plots/simulation/23_2g8a_full_copula_v3_self_gnncopar1_contr_herdcopar1_switch`,
   `figure_name` to match. The manager list (single
   `lin_multinomial_copula_self` pairing), seed 42, 100 episodes,
   `switch_every: 4`, `save_per_round: true` all untouched — the RNG
   context must equal the parent baseline's exactly (#162's finding).
   Dir name parses under the sweep DIR_PATTERN: contr token `gnncopar1`,
   switch token `herdcopar1` (verified against
   `_self_(\w+?)_contr_(\w+?)_switch$`). Exactly three edits means exactly
   three: the inherited header comment (which describes the punisher-severity
   experiment) is left stale by design rather than spend a fourth edit, since
   this config's provenance is recorded here. Commit.

6. **[Opus] Isolated simulation** — from the worktree,
   `AI_REMOTE_DIR='~/autoresearch/switch-herding-copula-v3'
   scripts/simulate_cluster.sh <config>` (sbatch, never login-node
   compute). The job prints provenance first: `aimanager.__file__`,
   sha256 of all three slot artifacts (contribution copula `.pt`, switch
   copula `.pt`, punisher `.joblib`). One run, seed 42 — no re-draws.

7. **[Sonnet] Fetch + evaluate** — `AI_REMOTE_DIR=... scripts/fetch_cluster.sh`
   the sim output dir; locally `python -m aimanager evaluate
   configs/simulation/manager_testing/23_2g8a_full_copula_v3_self_gnncopar1_contr_herdcopar1_switch.yml`.
   Commit sim outputs + evaluation.

8. **[Opus] Verdict, log, PR** — §2 against the parent baseline: gate 1 = SC
   finishes < 2.0; gate 2 = mean < 1.2893632310269196. Scores reported
   exactly as computed. Fill the Results row and Notes, commit, push,
   open the PR with `--base auto/contribution-herding-copula-v2`, titled
   `[SUCCESS] ...` or `[FAIL] ...`, body per §9.7 (hypothesis, results
   table, collateral +/-). Delete the remote isolated dir after the PR
   is open.

## 2a. Plan revision (after step 1)

The plan assumed `AI_REMOTE_DIR` isolation was available, as the maintainer's
brief states. It is not on this branch: `main`'s commit 8680db9 ("Isolate
parallel experiments on Raven via AI_REMOTE_DIR") is not an ancestor of
`auto/switch-herding-copula-v3`, nor of the parent
`auto/contribution-herding-copula-v2` — PR #165's branch was cut before that
commit landed. On this branch `scripts/simulate_cluster.sh` still reads
`REMOTE_PROJECT_DIR="~/algorithmic-institutions"` unconditionally, so
exporting `AI_REMOTE_DIR` would have been silently ignored and steps 3, 6 and
7 would have synced (with `rsync --delete`) into the shared remote checkout —
precisely the race the brief forbids, and it would have gone unnoticed
because the runs would still have succeeded.

Rationale for the fix chosen (new step 2b): port only the six script files
from 8680db9. All six are byte-identical on this branch to that commit's
parent (`git diff HEAD 8680db9^ --` over them is empty), so the port is
exact — no conflict, no unrelated drift, and no behavior change unless
`AI_REMOTE_DIR` is set. The commit's `notes/autoresearch.md` hunk is not
ported: that file has diverged on this branch and is the maintainer's process
document, not this experiment's to edit. Nothing in the frozen surface (§8)
is touched, and nothing about the model, protocol, seeds or RNG stream
changes — this is tooling only.

**Second revision (during step 3).** Step 3 as written assumed a fresh
isolated remote dir could run the whole `src/` suite. It cannot:
`remote_test.sh` excludes `artifacts/` and `plots/`, so artifact-reading tests
fail on absent files. Resolution, folded into step 3 rather than added as a
new step: run `AI_REMOTE_DIR=... scripts/simulate_cluster.sh --sync-only`
first (it ships artifacts to isolated dirs, mirroring what step 6 does
anyway), then re-run any artifact-dependent suite with `--test-only`. The
eval-suite suites stay local per the plan and are non-gating remotely. No code
change; no frozen surface touched.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|

## 4. Notes

1. (Fable, opening) The parent branch already carries the generalized
   copula sampler and both test suites (parent plan steps 2-3); #162's
   artifact, calibration outputs, scripts, and config were explicitly
   not ported there. This experiment is therefore a pure port-and-wire:
   the only open technical question is step 3's stamping/loading
   contract, since the artifact was stamped by #162's script against
   #162's `GraphNetwork` and will be loaded by the parent's.

2. (Opus, plan validation) Read the contract statically before tagging and
   it looks sound: `GraphNetwork.load` is `cls(**th.load(...))`, this
   branch's `__init__` takes `copula_rho`/`copula_phi`/`copula_switch_every`
   and its head gate admits `does_switch`, and #162's stamped `.pt` adds
   exactly those three keys to the base dict — so step 4 should be a
   confirmation, not a fork. Kept it as a hard gate anyway since the
   unpickle only happens on Raven.

3. (Opus, step 1) The port is exact: `copula_params.json` reads
   rho = 0.116482333585783 / phi = 0.70366020589033, and both calibration
   parquets' working-copy sha256 equal the LFS pointer oids on
   `origin/auto/switch-herding-copula-v2` (train b0fe6892…, test d61deeee…).
   Six paths, no edits.

4. (Opus, after step 1) Caught the plan's one wrong assumption before it
   could do damage: `AI_REMOTE_DIR` isolation does not exist on this branch
   (commit 8680db9 postdates the parent PR's branch point), so the remote
   steps would have rsync --deleted into the shared checkout while appearing
   to work. Added step 2b to port that commit's six script files, which are
   byte-identical to its parent here and so apply exactly — see §2a.

5. (Opus, step 3) The pre-sync `squeue` check tripped: a PENDING job
   (29855233) appeared between the plan check and the sync. `scontrol` put
   its WorkDir at `/u/certuer/autoresearch/contribution-gaussian-mlp-v2/.` —
   a different experiment's isolated dir, disjoint from our
   `~/autoresearch/switch-herding-copula-v3` target, so `rsync --delete`
   could not reach it. Proceeded rather than waiting; this is the case
   remote isolation is designed to make safe, and step 2b is what made the
   guarantee real here.

6. (Opus, step 3) Remote suites: 103 passed, 6 failed, 4 errors — but every
   failure is a missing input file, not a logic failure (9 `FileNotFoundError`
   plus one `SystemExit: 1`, no other error class in the log). Cause:
   `remote_test.sh` excludes `artifacts/` and `plots/` from its rsync
   unconditionally, and the isolated dir was fresh, so anything reading a
   `.joblib` or an existing `per_round.parquet` had nothing to read.
   Decisive evidence the mechanism is intact: `test_switch_copula_graph` and
   `test_contribution_copula_graph` pass 23/23, including
   `test_save_load_round_trips_copula_fields` and `test_phi_requires_switch_every`.

7. (Opus, step 3 resolution) Fixed environmentally, not in code, and without a
   hand-rolled rsync: `simulate_cluster.sh` already ships artifacts to isolated
   dirs (`--exclude='artifacts/manager/'` in place of all of `artifacts/`), so
   `--sync-only` through it is exactly what step 6 does. After that sync
   `test_linear_manager::test_multimanager_linear_side` passes (1 passed), and
   the ported `.pt`'s sha256 on the cluster still reads
   9b89f3c7b29f05e75df2601034c27a404906146872ca717d3dffbcc5e69c6f65 —
   integrity holds source branch -> worktree -> Raven. The remote eval-suite
   failures are explicitly **non-gating**: those suites are designated local
   (they need `plots/`, which no isolated sync ships) and they pass locally,
   89 passed / 3 skipped. Re-running them remotely would only re-fail on
   absent data. Note for later: re-runs of `remote_test.sh` in an isolated dir
   must use `--test-only`, since a syncing re-run would `rsync --delete` the
   artifacts back out.
