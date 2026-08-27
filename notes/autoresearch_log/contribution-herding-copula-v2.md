# Autoresearch log: contribution-herding-copula-v2

AR(1)-persistent herding copula for the GNN contributor — the third leg of
the copula triptych (punisher PR #160, switch PR #150) — stacked on the
maintainer-designated parent PR #160. Supersedes the un-PRed branch
`auto/contribution-herding-copula` (same mechanism, parented on PR #162):
the maintainer redirected the parent to #160 (2026-08-27, "work on the
contribution model on top of PR #160") before that branch's calibration
ever ran, so no result is discarded — its tested code is ported, its
baselines are re-anchored here.

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gnn` contributor
  (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`),
  unchanged — the change is at the free-running sampler only.
- **Parent PR (§9):** #160, branch `auto/punisher-severity-copula-v2`
  (`[SUCCESS]`, PD 2.93 -> 1.53). Evaluation stack: the parent's config
  `configs/simulation/manager_testing/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml`
  — `gnn` contribution x plain `gnn` switch x severity-copula
  `lin_multinomial` punisher (rho = 0.3507588625344979).
- **Baseline** (the parent's confirmed run,
  `plots/simulation/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch/evaluation/scores.csv`,
  single `lin_multinomial_copula_self` pairing): rows <= 1 **10/21**,
  mean **1.6879978841849728**. Contribution-slot rows >= 2 (§6 target
  candidates): CG 9.808514112722413 (band > 5; raw ratio gap
  0.2594222471221652, noise ceiling 0.026448679600274437),
  RCD 2.941928428442498 (band 2-5), RCA 2.0829074791966917 (band 2-5).
- **Target rows:** **CG** (primary; gate 1 needs CG < 5, i.e. the
  spread-ratio gap halves to < 0.13224339800137218) and **RCD**
  (secondary; gate 1 needs RCD < 2, i.e. the pull-coefficient gap below
  0.16021334822457584). Gate 2 requires the 21-row mean below
  1.6879978841849728. RCA (2.0829) and RCB (1.9881) are watch items, not
  targets — the mechanism makes no clean claim on them.
- **Hypothesis:** groups develop a persistent shared culture —
  group-mates' contributions co-move beyond what observed history
  explains, and the shared component persists across the episode. The
  prior record triangulates this as the one unclaimed piece of the CG
  deficit: within-round residual dependence is real but small (PR #149:
  rho ~ 0.07, ~14% of the teacher-forced gap, growing by round thirds
  0.035 / 0.074 / 0.119 — lock-in), exposure bias is real but small
  (PR #163: schedsamp closes ~7% free-running, family vetoed on cost),
  and an episode-persistent shared latent is the only mechanism that has
  moved CG without taxing the marginals (PR #159: spread ratio
  0.586 -> 0.662, tax-free, dose capped by teacher-forced MLE). The
  switch slot's arm comparison (PR #150) showed the same structure: a
  fresh per-round latent mean-reverts through the dynamics and regresses,
  an AR(1)-persistent one wins. An AR(1) shared per-(episode, group)
  Gaussian latent at the contribution sampler injects the shared
  component every round; because each member's draw feeds back through
  their own `prev_contribution` (this artifact's `x_encoding` is
  `[prev_contribution, prev_punishment, agent_group]` with a group-blind
  fully-connected EdgeModel — no explicit same-group conformity feature,
  the mechanism correction inherited from the prior branch's note 3), a
  *persistent* shift compounds through the closed loop instead of
  washing out, widening the per-(game, round, group) mean spread CG
  measures. **RCD:** a switcher's copula cell is their *arrival* group
  (`apply_switch` updates `agent_group` before `update_contribution`),
  so from the switch round on their draws share the receiving group's
  latent — coherent assimilation toward the new group's level, which is
  exactly the switching pull RCD regresses. Marginals are preserved by
  construction (each agent's draw passes through their own predicted
  CDF), so the C-block/CG anti-correlation tax (§6) is avoided the same
  way the severity and herding copulas avoided it, and no retraining.
- **Planned change:** Gaussian-copula sampling for the GNN contribution
  head. Port the head-agnostic AR(1) copula machinery from the switch
  precedent (PR #150/#162 branches: `src/aimanager/generic/copula.py`,
  the `GraphNetwork` copula dispatch) and the contribution-specific work
  from the superseded branch (head-gate relaxation to
  `y_name in {"does_switch", "contribution"}`, `copula_switch_every = 1`
  since contributions are decided every round, tests, the calibration
  script `scripts/artificial_humans/contribution_copula_rho.py` with the
  AR(1) lag-1 extension, artifact stamping). rho and phi estimated on
  the teacher-forced residual dependence of the frozen contribution GNN
  on the human train split only (pairwise-likelihood MLE, the #146/#149
  estimator; PRIMARY phi from cross-player-only lag-1 pairs — the
  orchestrator amendment inherited from the prior branch, recorded
  before any calibration number was seen). rho must reproduce #149's
  0.06958238086256316 exactly or the data path moved. Parameters stored
  as fields on a copy of the contribution artifact (artifacts without
  the fields sample independently, bit-identical legacy path); the plain
  `gnn` switch artifact carries no copula fields, so the switch slot's
  behavior is untouched (§4). One simulation in the parent's stack, one
  evaluation, verdict from the §2 gates.
- **Iteration budget (§5):** no retraining — one CPU calibration job and
  one standard-protocol GPU simulation; far under the 3x bound.

## 2. Plan

Written by the planning subagent (Opus), validated by the orchestrator
2026-08-27: targets per §2, every step legal per §5, frozen surface
untouched per §8. Orchestrator amendments at validation, before anything
ran: (i) the remote PyG test suites run via login-node pytest **inside
the isolated dir** (step 6 before step 7) — `scripts/remote_test.sh`'s
shared-checkout `--delete` sync is skipped entirely, the race that
voided the superseded branch's first run; (ii) port set A is minimal —
`dump_switch_probs.py`, `switch_copula_rho.py`,
`make_switch_copula_artifact.py` are not ported (docstring-only
references); (iii) only doc pointers to the superseded
`contribution-herding-copula.md` log are repointed to this file; switch-
slot pointers stay as historical PR references.

- [x] 1. Provenance survey, no code. Verified at planning: `b7dabfc` is
      an ancestor of both donor branches, so both ports are clean path
      checkouts; `scripts/baselines/punishment_copula_rho.py` is
      byte-identical between base and donor (the estimator
      `contribution_copula_rho.py` imports is unchanged, #149's rho
      reproducible); port set A is self-contained (`copula.py` imports
      only `math` + `torch`). Record in Notes; no commit of its own.
- [x] 2. Port the machinery (commit): `git checkout
      auto/switch-herding-copula-v2 -- src/aimanager/generic/copula.py
      src/aimanager/generic/graph.py
      src/aimanager/tests/test_switch_copula_graph.py
      tests/switch/test_switch_copula.py`. Nothing else; no edits.
- [x] 3. Port the contribution work (commit): `git checkout
      auto/contribution-herding-copula -- src/aimanager/generic/graph.py
      src/aimanager/tests/test_switch_copula_graph.py
      src/aimanager/tests/test_contribution_copula_graph.py
      tests/copula/test_contribution_copula.py` — head gate relaxed to
      `y_name in ("does_switch", "contribution")`, rejection case moved
      to `contribution_valid`, plus the two new suites. Repoint dangling
      references to the superseded log file to this one. NOT ported: the
      superseded branch's sim config and log, every #162
      artifact/config/log.
- [x] 4. Re-verify the mechanism points once in the new tree, by
      reading, into Notes (commit with step 3 or alone): head-agnostic
      sampler; 21-level ordering == encoder; per-round sim calls with
      `n_rounds=1`; `reset_rnn` clears `_copula_z` at round 0;
      `copula_switch_every=1` advances the AR(1) every round;
      `apply_switch` before `update_contribution` (arrival-group cells);
      invalid-agent overwrite after the draw; `agent_group` and
      `round_number` present in `env.state`.
- [x] 5. Port the calibration tooling (commit): `git checkout
      auto/contribution-herding-copula --
      scripts/artificial_humans/contribution_copula_rho.py
      scripts/artificial_humans/calibrate_copula.slurm`, then rename the
      artifact dir in both from `..._herding_copula` to
      `..._herding_copula_v2`. Estimator, data path, train split (40
      single-copy episodes), teacher-forced `sample=False`, cell
      `(episode, round, agent_group)`, pairwise MLE, round-trip gate,
      200-episode-cluster bootstrap, PRIMARY cross-player-only lag-1 phi
      all carried over unchanged.
- [x] 6. Create the isolated remote dir `~/iso_contr_herdcopar1_v2`
      (squeue PENDING check first) by direct rsync of the branch
      including `artifacts/`; md5-verify `copula.py`, `graph.py`,
      `environment.py`, `contribution_copula_rho.py` against the local
      branch. Everything remote for this experiment runs here, never in
      the shared checkout.
- [x] 7. Tests. Raven (PyG): login-node pytest inside the isolated dir
      for `test_switch_copula_graph.py` +
      `test_contribution_copula_graph.py` (+ the other src suites).
      Local (torch-only, main checkout's interpreter with `PYTHONPATH`
      at this worktree's `src`): `pytest tests/copula tests/switch` +
      the eval-suite tests. Log entry only if green.
- [ ] 8. `sbatch scripts/artificial_humans/calibrate_copula.slurm` in
      the isolated dir (CPU job; no compute on login nodes).
      Acceptance: rho must reproduce #149's 0.06958238086256316 exactly
      (SE 0.010418260898762315, CI [0.04592661278794028,
      0.0854596547235886], round-trip max |bias| 0.009462259703284237)
      — a mismatch means the data path moved: stop and report, never
      tune. STOP-GATE: rho CI includes 0, or phi_hat <= 0, or phi CI
      includes 0, or phi_hat >= 1 -> no artifact, no simulation,
      calibration-only `[FAIL]` PR. Otherwise continue regardless of
      preflight magnitude.
- [ ] 9. Commit the calibration output: job log +
      `artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2/calibration/copula_params.json`;
      log rho, phi, both CIs, the all-pairs phi diagnostic, job id,
      unrounded.
- [ ] 10. Port `scripts/artificial_humans/make_contribution_copula_artifact.py`
      with the `PARAMS`/`OUT` dirs renamed to `..._herding_copula_v2`;
      run in the isolated dir (stamps `copula_rho`, `copula_phi`,
      `copula_switch_every=1`; bit-identical check on every pre-existing
      key; three-field round-trip through `GraphNetwork.load`; honesty
      check: teacher-forced train-split probabilities bit-identical to
      the base). Fetch and commit script + stamped artifact
      (`.../group_switching_contribution_50ep_herding_copula_v2/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`,
      LFS) + `.copula.json` sidecar.
- [ ] 11. Sim config `configs/simulation/manager_testing/23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch.yml`:
      byte-copy of the parent's
      `23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml` with
      exactly three edits — contribution artifact path, `output_dir`,
      figure name. Manager list (single `lin_multinomial_copula_self`
      pairing), seed 42, episode count, `save_per_round: true` all
      untouched; dir name parses under the sweep DIR_PATTERN with switch
      token `gnn`.
- [ ] 12. Isolated simulation: rsync config + artifact into
      `~/iso_contr_herdcopar1_v2`, re-md5 the code, submit one job that
      prints provenance first (`aimanager.__file__`, md5 of
      `copula.py`/`graph.py`/`environment.py`, sha256 of all four
      artifacts, torch/CUDA versions, loaded
      `copula_rho`/`copula_phi`/`copula_switch_every`). squeue PENDING
      check before the rsync; one job, seed 42; confirm
      `per_round.parquet`.
- [ ] 13. Fetch into
      `plots/simulation/23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch/`;
      `python -m aimanager evaluate <config>`; commit sim outputs +
      evaluation, scores unrounded.
- [ ] 14. Verdict per §2 against the declaration's baseline (gate 1:
      CG < 5 from 9.808514112722413, or RCD < 2 from 2.941928428442498;
      gate 2: mean < 1.6879978841849728; rows <= 1 vs 10/21 as context).
      Fill results table + Notes; PR with
      `--base auto/punisher-severity-copula-v2`, titled
      `[SUCCESS]`/`[FAIL]`, body Hypothesis / Results / Collateral.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Re-parenting decision (2026-08-27): the maintainer pointed this
   experiment at PR #160 after the prior branch had declared against
   PR #162 (a titled `[FAIL]` with a config-dependent passing draw).
   #160's stack is also the cleaner claim: every slot in it is either a
   confirmed `[SUCCESS]` or the reference model, and the CG baseline
   here (9.8085) matches the re-baselined #149/ar1-copula declarations,
   so the whole prior CG record reads directly against this run.

2. The copula cell computation `cells = batch_index * 2 + agent_group`
   hard-codes `n_groups = 2` — correct for the whole 23 family, silently
   wrong if a future config changes group count (planning-stage flag (d),
   recorded for the next reader).

3. Steps 1-4 done in the worktree branch
   `auto/contribution-herding-copula-v2`. **Provenance (step 1), re-run
   here:** `git merge-base b7dabfc auto/switch-herding-copula-v2` ==
   `b7dabfc259965d8300ee01ef7cc5749cfeb80b3a`, and the same for
   `auto/contribution-herding-copula` — `b7dabfc` is an ancestor of both
   donors, so both ports are clean path checkouts with no merge.
   `git diff b7dabfc <donor> -- scripts/baselines/punishment_copula_rho.py`
   is empty for both donors, so the estimator #149's rho came from is
   byte-identical and step 8's acceptance number is reproducible. Port set
   A is self-contained: `copula.py` on the donor imports only `math`
   (`copula.py:15`) and `torch` (`copula.py:17`). **Ports:** step 2 is the
   four donor files verbatim (`copula.py` new, `graph.py` +78/-1,
   `test_switch_copula_graph.py` +361, `tests/switch/test_switch_copula.py`
   +316); step 3 adds only the head-gate relaxation in `graph.py`, the
   rejection-case move in `test_switch_copula_graph.py`, and the two new
   contribution suites — no unexpected hunks in `graph.py` in either
   direction. Four dangling pointers to the superseded log were repointed
   to this file (`graph.py:177`, `test_switch_copula_graph.py:332`,
   `test_contribution_copula_graph.py:2,16`,
   `tests/copula/test_contribution_copula.py:2,12`), their stale plan-step
   numbers renumbered to this plan's step 3; switch-slot log and PR
   references were left as historical pointers per plan amendment (iii).
   **Mechanism re-verification (step 4), by reading this tree:**
   (a) head-agnostic sampler — `copula.py:48` `sample_correlated_levels`
   takes `proba` as `(N, L)` with the only shape constraints
   `proba.dim() == 2` (`copula.py:74`) and `n_levels >= 2`
   (`copula.py:76`), and the level comes from a generic row-cumsum
   `searchsorted` (`copula.py:43-45`, called at `copula.py:101`); nothing
   assumes L == 2. (b) 21-level ordering == encoder — the head's decoder is
   `IntEncoder(encoding="onehot", n_levels=y_levels)`
   (`graph.py:156`, `y_levels=21` default at `graph.py:137`), whose onehot
   map has row `i` hot at position `i` (`encoder.py:21-25`, indexed at
   `encoder.py:47`) and whose sampling decode returns the column index
   itself (`encoder.py:55-58`); so softmax column `i` is contribution value
   `i` and the copula's returned level is the contribution directly (the
   artifact's stored `y_levels` is re-checked at load in step 10).
   (c) one predictor call per round with `n_rounds=1` — `env.reset()`
   (`simulate.py:258`) and `env.step()` (`simulate.py:300`) each call
   `update_contribution` once (`environment.py:346`, `environment.py:407`),
   which calls `artifical_humans.predict(self.state, ...)`
   (`environment.py:318-322`); `predict` dispatches to
   `predict_independent` for the non-autoregressive head
   (`graph.py:497-501`), which reads `n_rounds` off the state tensor shape
   (`graph.py:419`), and the env's state tensors are
   `(batch_size, n_agents, 1)` (`environment.py:136`) — so the copula loop
   `for r in range(n_rounds)` (`graph.py:391`) runs exactly once per round.
   (d) `reset_rnn` clears the copula state at round 0 —
   `_predict_encoded_copula` sets `self._copula_z = None` when `reset_rnn`
   (`graph.py:377-378`), and the env passes
   `reset_rnn=self.round_number[0, 0, 0] == 0` (`environment.py:320`), i.e.
   True only on the round-0 call; the field is initialised to `None` at
   construction (`graph.py:194`). (e) `copula_switch_every=1` advances the
   AR(1) every round — the existing gate is
   `(int(rounds[r]) + 1) % self.copula_switch_every == 0`
   (`graph.py:408`), which is unconditionally true at `k=1`, with `rounds`
   taken from the state's `round_number` (`graph.py:386-390`); no code
   change was needed for the every-round case. (f) arrival-group cells —
   `step` calls `apply_switch(pending_switch)` (`environment.py:405-406`)
   before `update_contribution()` (`environment.py:407`), and
   `apply_switch` rewrites `state["agent_group"]`
   (`environment.py:368-373`), so a switcher's cell id
   (`graph.py:381-382`) is their receiving group from the switch round on —
   the RCD claim in §1 holds as written. (g) invalid-agent overwrite after
   the draw — the copula draw happens at `environment.py:318-322` and the
   `contribution[~contribution_valid] = default` overwrite only at
   `environment.py:331-334`, so the validity model never perturbs the
   copula's RNG consumption or its latent. (h) required state keys present
   — `round_number` (`environment.py:140`) and `agent_group`
   (`environment.py:150-154`) are both in the dict `reset_state` builds,
   and `round_number` is advanced at `environment.py:396`; the copula
   sampler reads them off the raw state, which `predict_independent`
   forwards untouched (`graph.py:426-428`). All eight points OK; nothing in
   the diffs was unexpected. Not run at these steps: any Python, hence no
   `pre-commit`/flake8 pass — the only hand edits are comments and
   docstrings, all within 88 characters (checked with `awk`).

4. Steps 5-7 (calibration tooling ported, isolated remote setup, tests).
   **Port (step 5):** `contribution_copula_rho.py` (613 lines) and
   `calibrate_copula.slurm` checked out from
   `auto/contribution-herding-copula`. The artifact dir
   `group_switching_contribution_50ep_herding_copula` occurred exactly ONCE
   across both files (`calibrate_copula.slurm:25`, the `PARAMS` path) and is
   now `..._herding_copula_v2`; a post-edit grep for the old name returns
   nothing. Estimator specifics re-confirmed by reading the whole file:
   train split `experiments/baseline/2group_8agent_50ep_bline_train.csv`
   with `N_TRAIN_EP = 40` asserted in `select_split`; teacher-forced
   `predict_independent(..., sample=False, reset_rnn=True)`; cell
   `(episode * n_rounds + round) * N_GROUPS + agent_group` (meta
   `cell_key="episode_round_agent_group"`); pairwise-likelihood MLE via
   `pc.rho_mle`; round-trip gate at tolerance 0.03 behind `--roundtrip`;
   episode-cluster bootstrap at `pc.N_BOOT = 200`; PRIMARY phi from
   `cross_pairs` filtered to `agent[i] != agent[j]` with the all-pairs refit
   kept as `phi_allpairs` ("never used"); `--write-params` dumps estimates
   plus `source_model_sha256` and `git_head`. `pc` is
   `scripts/baselines/punishment_copula_rho.py`, still byte-identical to
   base (`git diff b7dabfc` empty), so step 8's acceptance number is
   reproducible; `SEED = 38381`, `RHO_GRID = 0.0..0.90` non-negative.
   **Slurm adaptation (flagged):** the donor script's only environment
   assumption was the relative `source .venv/bin/activate`, and the
   isolated dir carries no venv. Replaced with
   `PY="$HOME/algorithmic-institutions/.venv/bin/python"` plus
   `export PYTHONPATH="$PWD/src:..."`, so the shared checkout supplies only
   the interpreter while THIS tree supplies the code (the script's own
   `sys.path.insert(0, ROOT / "src")` off `__file__` is the second guard);
   the job now echoes `cwd` and `aimanager.__file__` before running. No
   absolute `~/algorithmic-institutions` path was ever hard-coded in either
   file, and `#SBATCH --chdir=.` already resolves to the submission dir.
   **Isolated setup (step 6):** `~/iso_contr_herdcopar1_v2`, populated by
   direct `rsync` from the worktree (`src/`, `scripts/`, `configs/`,
   `pyproject.toml`, the three human CSVs, the base contribution artifact
   dir, and `artifacts/baselines/`); the shared checkout was never synced,
   read, or run from beyond borrowing its interpreter. squeue at the start:
   one unrelated RUNNING gpu job (29666118 `architec`), no PENDING, so no
   sync race. All nine md5s match local vs remote:
   `copula.py 96393213cfd3244aa4ef26ef22c2e11e`,
   `graph.py c277589c2e9471d8eeafb48294d1face`,
   `environment.py e0f14eeefc088c53250709e6a8a9bf13`,
   `contribution_copula_rho.py 140723dd30a95bcdb3a93dde5fd60ffd`, plus the
   slurm, the three CSVs and the base `.pt`
   (`c70309eac20b48ad18d96aa5c5bf7725`). No shipped file is an LFS pointer:
   scanned locally before the rsync and again remotely
   (`grep -rl 'version https://git-lfs.github.com/spec/v1'` returns
   nothing), and the CSVs/`.pt` were byte-sniffed remotely (CSV header, `PK`
   zip magic).
   **Tests (step 7), both import paths verified before trusting a result:**
   local `aimanager.__file__` ==
   `<worktree>/src/aimanager/__init__.py`, remote ==
   `/u/certuer/iso_contr_herdcopar1_v2/src/aimanager/__init__.py` — neither
   fell through to an editable install. Local (main checkout's interpreter,
   `PYTHONPATH` at this worktree's `src`): `tests/copula`, `tests/switch`
   and the five eval-suite suites, **101 passed**. Raven login-node pytest
   inside the isolated dir: `test_switch_copula_graph.py`,
   `test_contribution_copula_graph.py`, `test_encoder.py`,
   `test_edge_encoder.py`, `test_environment.py`,
   `test_linear_manager.py`, **41 passed**. The remote run first showed
   `1 failed` — `test_linear_manager.py::test_multimanager_linear_side`
   raised a bare `FileNotFoundError` for
   `artifacts/baselines/punishment_multinomial_best_with_contr.joblib`,
   i.e. a gap in what I had shipped, not a code failure; shipping
   `artifacts/baselines/` (needed for the parent stack's
   `lin_multinomial_copula_self` punisher at step 12 regardless) made it
   pass with no source edit. Lint, one batched pass over
   `git diff --name-only b7dabfc -- 'src/**/*.py' 'scripts/**/*.py'`
   (5 files): flake8 (88, `E203,W503`) clean and `black --check` reports all
   5 unchanged; the two branch-added `tests/` suites are clean too. No fix
   was needed, so nothing was edited for lint.
