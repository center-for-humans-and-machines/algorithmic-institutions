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

- [ ] 1. Provenance survey, no code. Verified at planning: `b7dabfc` is
      an ancestor of both donor branches, so both ports are clean path
      checkouts; `scripts/baselines/punishment_copula_rho.py` is
      byte-identical between base and donor (the estimator
      `contribution_copula_rho.py` imports is unchanged, #149's rho
      reproducible); port set A is self-contained (`copula.py` imports
      only `math` + `torch`). Record in Notes; no commit of its own.
- [ ] 2. Port the machinery (commit): `git checkout
      auto/switch-herding-copula-v2 -- src/aimanager/generic/copula.py
      src/aimanager/generic/graph.py
      src/aimanager/tests/test_switch_copula_graph.py
      tests/switch/test_switch_copula.py`. Nothing else; no edits.
- [ ] 3. Port the contribution work (commit): `git checkout
      auto/contribution-herding-copula -- src/aimanager/generic/graph.py
      src/aimanager/tests/test_switch_copula_graph.py
      src/aimanager/tests/test_contribution_copula_graph.py
      tests/copula/test_contribution_copula.py` — head gate relaxed to
      `y_name in ("does_switch", "contribution")`, rejection case moved
      to `contribution_valid`, plus the two new suites. Repoint dangling
      references to the superseded log file to this one. NOT ported: the
      superseded branch's sim config and log, every #162
      artifact/config/log.
- [ ] 4. Re-verify the mechanism points once in the new tree, by
      reading, into Notes (commit with step 3 or alone): head-agnostic
      sampler; 21-level ordering == encoder; per-round sim calls with
      `n_rounds=1`; `reset_rnn` clears `_copula_z` at round 0;
      `copula_switch_every=1` advances the AR(1) every round;
      `apply_switch` before `update_contribution` (arrival-group cells);
      invalid-agent overwrite after the draw; `agent_group` and
      `round_number` present in `env.state`.
- [ ] 5. Port the calibration tooling (commit): `git checkout
      auto/contribution-herding-copula --
      scripts/artificial_humans/contribution_copula_rho.py
      scripts/artificial_humans/calibrate_copula.slurm`, then rename the
      artifact dir in both from `..._herding_copula` to
      `..._herding_copula_v2`. Estimator, data path, train split (40
      single-copy episodes), teacher-forced `sample=False`, cell
      `(episode, round, agent_group)`, pairwise MLE, round-trip gate,
      200-episode-cluster bootstrap, PRIMARY cross-player-only lag-1 phi
      all carried over unchanged.
- [ ] 6. Create the isolated remote dir `~/iso_contr_herdcopar1_v2`
      (squeue PENDING check first) by direct rsync of the branch
      including `artifacts/`; md5-verify `copula.py`, `graph.py`,
      `environment.py`, `contribution_copula_rho.py` against the local
      branch. Everything remote for this experiment runs here, never in
      the shared checkout.
- [ ] 7. Tests. Raven (PyG): login-node pytest inside the isolated dir
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
