# Autoresearch log: switch — one-hot group-size encoding in the joint exodus head

Branch `auto/switch-exodus-k-onehot` (worktree
`.claude/worktrees/switch-exodus-k-onehot`), created from
`origin/auto/switch-joint-exodus-gmlp` at `bc77932` — the head of the
maintainer-designated parent PR #172, per §9 "Building on a `[SUCCESS]` PR".
The PR opens with `--base auto/switch-joint-exodus-gmlp`.

## 1. Declaration

- **Slot:** switch.
- **Parent PR:** **#172** `[SUCCESS] Joint exodus head on the gmlp
  group-copula stack` (`auto/switch-joint-exodus-gmlp`, itself stacked on
  PR #170). Its log is `notes/autoresearch_log/switch-joint-exodus-gmlp.md`;
  its verdict passed both §2 gates **with its own pre-registered unsoundness
  criterion fired** (parent note 34: SC bought through excess full-group
  exodus, full-exodus cell share 0.1463 against human 0.1079), and the
  maintainer's review comment on #172 names the successor this experiment
  runs. This is the "fix the dose" experiment the parent said a successor
  must not stack without.
- **Base model:** the parent's candidate switch artifact
  `artifacts/artificial_humans/switch_joint_exodus_gmlp/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`
  (sha256
  `ecd231f423c3550d22eba5a9574b53e68026227441a818890dddf016b1131f33`,
  equals parent note 29), trained by
  `configs/training/artificial_humans/switch_predictor/joint_exodus.yml` —
  the stock switch config (`x_encoding = common_good, punishment,
  agent_group, round_number`, `edge_encoding = []`, `y_levels = 2`, hidden
  10, 375 epochs, batch 10, lr 5e-4, wd 1e-3, 5-fold, seed 38381,
  flip-doubled data) plus `model_args.joint_exodus: True` and
  `joint_exodus_switch_every: 4`. Its head is `Linear(23 -> 10) -> Tanh ->
  Linear(10 -> 81)`, 1,131 parameters, whose 23 inputs are the two
  group-pooled post-RNN embeddings (2 x 10), the two valid-decider counts as
  `k / 8`, and the round as `r / 23`.
- **Evaluation stack (§3, parent rule of §9):** the parent's candidate
  config
  `configs/simulation/manager_testing/23_2g8a_jexogmlp_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_switch.yml`
  — `gaussian_mlp_v2 + group copula` contributor
  (`artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib`),
  PR #160 severity-copula multinomial punisher
  (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`),
  `valid_model` `raven_script_22/model/rnn_False__dataset_full.pt`, single
  pairing `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
  `switch_every: 4`, `save_per_round: true` — with **only `switch_model`
  swapped** to this experiment's retrained artifact (plus
  `output_dir`/`figure_name`). Everything else byte-identical.
- **Baseline for BOTH §2 gates**, re-verified from
  `plots/simulation/23_2g8a_jexogmlp_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_switch/evaluation/scores.csv`
  (21 rows, run `ah group_switching managed by lin_multinomial_copula_self`;
  the parent's `per_round.parquet` hashes to
  `f3c3136cf6b2241c8b3c942a4b746c54c6d0946c80a7dc48a2e6aff4487aa2fa`, equal
  to parent note 33; SC re-scored through the suite's own `score_row` gives
  1.301036 / numerator 0.253088 / denominator 0.194528, Note 2):

  | row | score | band |
  |---|---|---|
  | CA | 1.6334161555537756 | 1-2 |
  | CB | 0.9521466564786460 | <= 1 |
  | CC | 1.0516855807783450 | 1-2 |
  | CD | 1.1453909344406925 | 1-2 |
  | CE | 0.9801160413212853 | <= 1 |
  | CF | 1.3437393668769062 | 1-2 |
  | CG | 1.9676786688812167 | 1-2 |
  | SA | 1.0355693521727887 | 1-2 |
  | SB | 0.9461647801631368 | <= 1 |
  | **SC** | **1.3010363546635990** | 1-2 |
  | PA | 0.6340478465984272 | <= 1 |
  | PB | 0.9366487166608538 | <= 1 |
  | PC | 0.8948662648672548 | <= 1 |
  | PD | 0.7756377769460460 | <= 1 |
  | RCA | 3.4941490452010120 | 2-5 |
  | RCB | 2.1039340633795360 | 2-5 |
  | RCC | 1.1135232381843163 | 1-2 |
  | RCD | 0.7154547676074160 | <= 1 |
  | RSA | 1.3893213082295712 | 1-2 |
  | RPA | 1.2753141878504728 | 1-2 |
  | RPB | 0.8046502469403721 | <= 1 |

  mean **1.2616424454188417**, **gate-2 ceiling (+10%) 1.387806689960726**,
  rows <= 1 **9/21** (context, not a criterion).

- **Target row (the only one claimed): SC 1.3010363546635990**, band 1-2 →
  requires **<= 1.0** (band `<= 1`), i.e. the 500-repeat mean 25-vs-25 EMD
  numerator must fall from **0.253088** to at most **0.19452799999999945**
  (the human-vs-human ceiling itself; a drop of >= 0.0586). Nothing else is
  a target: pre-declaration binds, and a band upgrade on any other row is
  collateral.
- **Gate 2:** 21-row mean must stay at or below **1.387806689960726**
  (0.126 of headroom above the parent's 1.2616424454188417).
- **Guards (declared, non-gating, with the reading fixed in advance).**
  - **CG 1.9676786688812167 (1-2, 0.032 under the `2-5` edge).** Its band
    upgrade on the parent was collateral of the group-size distribution:
    the spread ratio's independence floor is `1/sqrt(size)`, so a
    fully-merged share of 0.214 (against human 0.144) inflated the group
    spread. This experiment intends to *lower* that share; CG returning to
    `2-5` is the expected price of an honest SC and costs ~0.04 on the mean
    per 0.8 of CG movement. Reported, not gating.
  - **CE 0.9801160413212853 (<= 1, 0.020 under the edge).** Same
    provenance (collateral of the merged share on the parent); the same
    reading.
  - **RCD 0.7154547676074160 (<= 1).** The parent's guard, inherited: the
    conditional-Bernoulli selection is untouched here, so any RCD movement
    is through the count distribution alone.
  - **SB 0.9461647801631368 (<= 1)** and **SA 1.0355693521727887 (1-2,
    0.036 over).** Fewer singleton and pair exits mean fewer movers at
    rounds 7-19; the candidate's per-round switch rates (0.409 / 0.274 /
    0.245 / 0.250 / 0.223) already sit under the human's (0.442 / 0.299 /
    0.245 / 0.241 / 0.251) at three of five rounds, so SA is expected to
    move slightly *against* us and SB may leave `<= 1`. Neither is claimed
    in either direction.
  - **RSA 1.3893213082295712 (1-2)** and **RCB 2.1039340633795360 (2-5)**:
    both entered their bands on the parent through the joint draw itself.
    They are not expected to move much (the joint draw stays); if RCB
    returns to `1-2` it is collateral, not claimed.
- **Behavioral claim (§5, one sentence):** humans leave a group en masse at
  a rate that rises with group size only up to four and then hits a hard
  floor — no human group of five or more ever emptied (0 of 119 complete-pair
  cells) while singletons emptied in 16% of cells, not the 47% this stack
  produces — so a hazard with a free intercept per group size, instead of
  one smooth curve in `k/8`, is what stops the simulation from over-emptying
  singletons and pairs after the founding round, which is where its excess
  merges come from (post-founding formation 0.190 against human 0.082,
  entirely from pre-sizes 6 and 7) and hence its surplus at larger-group
  size 8 (0.214 against 0.144) that SC measures.
- **Planned change (one change, switch slot only): the joint head's two
  per-group valid-decider counts enter as one-hot codes over `k in {0..8}`
  (2 x 9 = 18 inputs) instead of two numeric scalars `k/8`.** The head's
  input width goes 23 → 39 at the stack's `embed_size = hidden_size = 10`
  and its parameter count 1,131 → **1,291** (+160). Implemented as a
  `size_encoding` option on `JointExodusHead` whose default is the parent's
  `"numeric"`, so the parent's pickled head still loads and samples
  bit-identically (the control run depends on it), and selected by one new
  `model_args` key in the training config. **Everything else is
  byte-identical to the parent**: the detached readout (the one-hot of an
  integer count carries no gradient, so the cut is unchanged), the
  label-order pooling, the masked joint over the 9 x 9 grid,
  `DROP_INCOMPLETE_PAIRS`, the conditional-Bernoulli selection, the firing
  schedule (`(round + 1) % 4 == 0`, all five decision rounds — no schedule
  parameter exists or is added, parent note 5), every hyperparameter, seed
  38381, the flip-doubled data. No new observable enters the model; `k` is
  a feature the head already has. Under the drop ruling every training
  pair sums to eight, so the two one-hots are perfectly collinear
  (`k_1 = 8 - k_0`) and the encoding amounts to nine free intercept vectors
  for the nine size configurations the simulation can actually produce
  (human doubled training pairs per configuration: (4,4) 96, (3,5)+(5,3)
  70, (2,6)+(6,2) 68, (1,7)+(7,1) 62, (0,8)+(8,0) 38; Note 3).
- **Measured deficit profile (Notes 3-5 carry the full numbers).** On
  complete pairs, P(full exodus | k) is human 0.161 / 0.147 / 0.200 / 0.177
  for k = 1..4 (n 31 / 34 / 35 / 96) and **0 of 119** cells for k >= 5;
  the candidate is 0.465 / 0.239 / 0.180 / 0.159 (n 101 / 92 / 78 / 290)
  and 5 of 355 for k >= 5. The singleton factor is **2.9x** on complete
  pairs (0.465 / 0.161) and 1.7x against all human cells (0.465 / 0.269);
  the parent-of-parent stock GNN has the same singleton rate (0.476), so
  the joint head inherited it from the per-agent model rather than
  creating it. Formation P(post = 8 | pre < 8) by pre-size 4 / 5 / 6 / 7:
  human 0.219 / 0.157 / 0.018 / 0.082, candidate 0.200 / 0.167 / **0.141 /
  0.267** — the excess is *only* where a group of two or one empties.
  Founding-round formation is right (0.220 vs 0.260, if anything short);
  post-founding (rounds 7-19) it is 0.190 vs 0.082. Persistence of the
  merged state is right (P(stay 8) 0.298 vs 0.300). The fully-merged share
  by four-round block is human 0.26 / 0.18 / 0.06 / 0.10 / 0.12 (decaying
  from the founding burst) against candidate 0.22 / 0.24 / 0.16 / 0.22 /
  0.23 (flat).
- **Pre-registered prediction (Note 6, a Markov oracle on the larger-group
  size scored through the suite's own `score_row`, 20 pools of 100
  episodes each).** The candidate's own kernel reproduces SC 1.36 +- 0.22
  (real 1.301) and the parent's 2.72 +- 0.26 (real 2.669), so the oracle
  is faithful, and the pool-to-pool sd says a single seed-42 draw carries
  roughly +-0.1 to +-0.2 of SC noise at this level. Holding every other
  transition of the candidate's kernel fixed: setting the **pre = 7 row**
  (singleton empties) to the human rate gives SC **1.154 +- 0.167**
  (P(SC <= 1) = 0.15); **pre in {6, 7}** (singleton or pair empties) gives
  **0.981 +- 0.109** (P = 0.60, mean L 6.176, P(L = 8) 0.140); pre in
  {5, 6, 7} gives 0.920 +- 0.100 (P = 0.90); all formation rows at human
  rates 0.974 +- 0.117 (P = 0.55); the human kernel throughout 0.808 +-
  0.053. **So: if the one-hot brings the k = 1 and k = 2 hazards to the
  human 0.16 / 0.15, the expected SC is ~0.98 and the band upgrade is
  close to a coin flip; fixing the singleton alone is predicted
  insufficient.** The experiment proceeds to simulation whatever the
  retrained head's loss looks like; only an implementation failure (Plan
  step 6's activation/detach checks, or a non-bit-identical control) stops
  it.
- **Pre-registered unsoundness criteria — stated now, in numbers, decided
  from the candidate's `per_round.parquet` with the Note-3 diagnostic on
  complete pairs, whatever the gates say:**
  - **(U1) The mechanism did what it claims** only if P(full exodus |
    k = 1) falls from **0.4653** (47/101) to **<= 0.31** (at least half the
    gap to the human 0.1613 closed) **and** P(full exodus | k = 2) falls
    from **0.2391** (22/92) to **<= 0.20**. If SC is band-upgraded while
    P(full | k = 1) > 0.31, SC moved for a reason other than the declared
    one and the result is **mechanistically unsound** — a `[SUCCESS]` title
    would still be honest under §2, but the log and PR body say so plainly
    and a successor must not build on the encoding as if it worked.
  - **(U2) The parent's dose criterion, inherited verbatim.** The
    full-exodus cell share (cells with `k_g > 0` on complete pairs, all
    decision rounds) must not exceed the human **0.1079** (34/315); the
    candidate sits at 0.1463 (134/916). If SC is upgraded with the share
    still above 0.1079, this experiment failed at exactly the thing it
    exists to fix and the unsound reading carries over. (If U1 holds, the
    arithmetic says the share lands near 0.10: 134 - ~31 - ~8 cells.)
  - **(U3) The floor.** Full exodus among k >= 5 cells must not exceed
    **0.03** (candidate 5/355 = 0.014, human 0/119). A rise means the free
    intercepts were spent on the wrong end.
  - **Falsifier in the other direction (a `[FAIL]` that still validates
    the mechanism):** U1 and U2 both hold and SC does **not** reach
    `<= 1`. Then the small-group hazard was correctly identified and
    fixed, and the residual — the candidate is short at sizes 5 and 6
    (0.192 / 0.228 vs 0.244 / 0.280), a dissolution-path shape the oracle's
    40% says formation alone cannot close — is the successor's target, not
    this encoding.
- **Iteration budget (§5):** the parent's retrain ran 4m36s (job 29892851)
  against a ~10.6-min ceiling (3x the 3m32s base). One-hot adds 160
  parameters to a readout MLP on a detached input — no measurable change to
  the epoch time; two sims of ~2 min each. Comfortably inside.
- **Legality and frozen surface.** No new input feature — the one-hot
  re-encodes the valid-decider count the head already receives, and `k` is
  derived from `agent_group` and the decider mask exactly as before. No
  round-keyed term, no term keyed to any metric's bin or stratum (SC is an
  EMD over pooled (game, round) larger-group sizes from round 4 on with no
  strata — nothing here is aimed at a boundary). No seed, episode count,
  scoring parameter or protocol field changes. Nothing under
  `src/aimanager/evaluation_suite/`, `notes/evaluation_metric_defs.md`,
  `notes/eval_scoring_schema.md` or `experiments/` is touched; the two
  scratchpad diagnostics import `convert.load_human` / `load_sim` and
  `scoring.score_row` read-only and live outside the repo. No other
  branch's log file is written.

## 2. Plan

Steps for the orchestrator to validate (§2 targets, §5 legality, §8 frozen
surface) and tag. Paths are relative to the worktree.
`AI_REMOTE_DIR='~/autoresearch/switch-exodus-k-onehot'` on every
`train_cluster.sh` / `simulate_cluster.sh` / `fetch_cluster.sh` /
`remote_test.sh` call, and `squeue -u certuer` over a *live* SSH tunnel
before any syncing call (parent notes 6, 31). The parent's step-1
training-template hardening (in-job `PYTHONPATH` export and PROVENANCE echo)
is already on this branch. Local tests run on the main checkout's venv with
the worktree's source first on the path
(`PYTHONPATH=$PWD/src /Users/ertuerkan/Desktop/algorithmic-institutions/.venv/bin/python -m pytest ...`;
`uv sync` cannot run in the worktree, parent note 9). Lint (`black src/`,
`flake8 src/ --max-line-length=88 --extend-ignore=E203,W503`) once per step
before staging.

**Validated by the orchestrator (Opus), 2026-09-03.** Targets per §2: one
declared row, SC 1.3010363546635990 (`1-2`) crossing to `<= 1`, with the
gate-2 ceiling 1.387806689960726 = 1.1 x the parent's 1.2616424454188417 —
both arithmetic-checked against the parent's committed `scores.csv`. Every
step legal per §5: one change in one slot, a re-encoding of a feature the
head already receives (no new observable, no round-keyed term, and SC has
no strata or bin edges for the one-hot to be keyed to), +160 readout
parameters on a detached input so the §5 iteration budget is untouched.
Nothing in the step list writes to the §8 frozen surface, and the
diagnostics of steps 8 read the evaluation suite without modifying it. The
`**_` sink at `graph.py:152` is real (confirmed), so step 2's explicit
keyword is a correctness requirement, not a style choice.

**One amendment (orchestrator), folded into step 2:** when `load` supplies
*both* a pickled `joint_exodus_head` and a `joint_exodus_size_encoding`
key, `GraphNetwork.__init__` must assert that the head's own
`size_encoding` equals the key. Without it a saved dict could advertise
`onehot` while its pickled head runs `numeric` — which is precisely the
failure the step-6 activation check exists to catch, and the check reads
the same two places, so it would not catch itself.

**Implementer tags** are `[Opus]` where the step is subtle or a silent
failure would void the run (the encoding branch and its pickle
back-compat, the plumbing, the retrain's activation/detach checks, the
sims' bit-identity, the evaluation) and `[Sonnet]` where the step is
mechanical against a precise spec (test parametrisation, the config copy,
running the suites). Step 9 stays with the orchestrator.

1. **[Opus]** **The `size_encoding` option on the head** —
   `src/aimanager/generic/joint_exodus.py`, `JointExodusHead.__init__`
   (existing). Add keyword `size_encoding="numeric"` (allowed values
   `"numeric"`, `"onehot"`; anything else asserts), store it, and compute
   `in_features = n_groups * embed_size + size_features + 1` with
   `size_features = n_groups` for numeric and `n_groups * (max_group_size + 1)`
   (18) for onehot. In `forward`, replace the single `sizes = k / SIZE_NORM`
   line by a branch: numeric keeps that expression verbatim; onehot builds
   `th.nn.functional.one_hot(k, self.grid)` over the label-ordered `(…, 2)`
   count tensor and flattens the last two axes to `(…, 18)` in `x.dtype`,
   with group 0's nine codes first. The feature layout is then
   `[pooled (2F) | onehot(k_0) (9) | onehot(k_1) (9) | round (1)]`. Add
   `__setstate__` that calls the `nn.Module` one and then
   `self.__dict__.setdefault("size_encoding", "numeric")`, so the parent's
   head pickled inside `ecd231f4…` — saved before the attribute existed —
   unpickles as a numeric head and its 23-wide `mlp` keeps working. Nothing
   else in the module changes: `SIZE_NORM`, `ROUND_NORM`, `pool_by_group`,
   `joint_count_mask`, `masked_joint_log_prob`, the detach, the round
   pooling. Module docstring gains one paragraph on why (the measured hump
   and floor of §1, with the numbers). Tests in
   `tests/switch/test_joint_exodus.py` (existing): `make_head` gains a
   `size_encoding` argument; parametrise `test_head_emits_a_valid_masked_joint`,
   `test_head_handles_the_fully_merged_state`,
   `test_head_support_follows_the_decider_mask` and
   `test_head_is_deterministic_and_differentiable` over both encodings;
   keep `test_head_feature_normalisation_convention` as the numeric case
   and add its onehot sibling (shape `(1, 2, 2*5 + 18 + 1)`, columns 10-18
   equal `e_3`, 19-27 equal `e_5`, column 28 the round `/ 23`); add: the
   fully merged `(8, 0)` and `(0, 8)` produce distinct, transposed codes;
   numeric-by-default is `th.equal` to a head built without the keyword
   under the same seed; an unknown encoding is rejected; a head pickled
   with `size_encoding` deleted from its `__dict__` unpickles as numeric
   and runs with the 23-wide MLP.

2. **[Opus]** **The `GraphNetwork` plumbing** — `src/aimanager/generic/graph.py`
   (existing). `__init__` gains an **explicit** keyword
   `joint_exodus_size_encoding=None` — it must not fall into the `**_`
   sink, since `train.py` passes `**model_args` and a swallowed key would
   silently train the parent's numeric head under this experiment's name.
   Asserts: `None` or one of `"numeric"` / `"onehot"`; only with
   `joint_exodus`. Store it; in the `op1 is None` branch pass
   `size_encoding=joint_exodus_size_encoding or "numeric"` into the
   `JointExodusHead(...)` call (still built **last**, after every trunk
   parameter, so the trunk's initial weights under seed 38381 stay the base
   run's); append `"joint_exodus_size_encoding"` to `save()`'s `to_save`.
   An artifact saved without the key loads through `cls(**to_load)` with
   `None` and its pickled head decides its own encoding via step 1's
   `__setstate__`. `forward`, `encode`, `_predict_encoded_joint_exodus`,
   `predict_independent` are untouched. Tests in
   `tests/switch/test_joint_exodus_graph.py` (existing): `make_model`
   accepts the kwarg; parametrise
   `test_head_on_shares_the_trunk_state_dict_and_logits` and
   `test_joint_output_is_a_valid_masked_distribution` over both encodings;
   extend `test_save_load_round_trips_the_head` with an onehot variant that
   also asserts the loaded head's `mlp[0].in_features == 2 * HIDDEN + 19`
   and `size_encoding == "onehot"`; add to
   `test_artifact_without_the_new_keys_loads_and_behaves_as_today`'s
   stripped-key list the new key; add a head-on back-compat test: save a
   numeric head-on model, delete only `joint_exodus_size_encoding` from the
   saved dict, load, and assert `th.equal` joint output and identical
   sampling with RNG state; guards: the key without the head, a bad string.

3. **[Sonnet]** **Detach, loss and sampling under the new encoding — tests only** —
   `tests/switch/test_joint_exodus_detach.py` (existing): parametrise all
   four tests over the encoding; in
   `test_the_cut_is_exactly_the_pooled_embedding` the tail check branches —
   numeric block `th.equal` to `counts.round() / SIZE_NORM` as today, onehot
   block `th.equal` to the flattened `one_hot(counts.round(), 9)` — and the
   `not features.requires_grad` assertion stays for both (a one-hot of an
   integer never carried gradient). `tests/switch/test_joint_exodus_loss.py`
   (existing): parametrise the head-on per-agent-component identity and the
   joint-objective-descends tests over the encoding.
   `tests/switch/test_joint_exodus_sampling.py` (existing): parametrise the
   RNG-cost tests (three draws on a decision round, one otherwise), the
   switcher-counts-equal-the-drawn-pair test and the save/load of
   `switch_every` over the encoding — the sampler is encoding-agnostic and
   these prove it. `src/aimanager/tests/test_joint_exodus_train_sim_parity.py`
   is untouched (it tests `(m, k)` counts, which do not depend on the
   encoding). No source changes in this step.

4. **[Sonnet]** **Training config** — new
   `configs/training/artificial_humans/switch_predictor/joint_exodus_k_onehot.yml`:
   a copy of `joint_exodus.yml` in which **only** `description`,
   `output_dir: artifacts/artificial_humans/switch_exodus_k_onehot`, and one
   added `model_args` key `joint_exodus_size_encoding: onehot` differ. Every
   hyperparameter (375 epochs, batch 10, lr 5e-4, wd 1e-3, hidden 10,
   5-fold, seed 38381, `mask_name: switch_valid`, `switch_every: 4`,
   `joint_exodus: True`, `joint_exodus_switch_every: 4`, flip-doubled
   `experiments/2group_8agent_50ep.csv`, labels `architecture:
   mlp+rnn+edge`, `dataset: 50ep_doubled`) stays verbatim, so the artifact
   file name is `architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` under
   the new dir. Verify with `diff` against `joint_exodus.yml` (description
   hunk, `output_dir` line, one added line — nothing else).

5. **[Sonnet]** **Tests, local then Raven** — locally,
   `PYTHONPATH=$PWD/src <main venv python> -m pytest tests/switch
   src/aimanager/tests/test_joint_exodus_train_sim_parity.py tests/baselines
   src/aimanager/tests/test_eval_*.py -q` all green (parent: 499). On Raven
   from the isolated dir, `squeue` check, then **two** calls —
   `AI_REMOTE_DIR=... scripts/remote_test.sh -- tests/switch -v --tb=short`
   and `AI_REMOTE_DIR=... scripts/remote_test.sh --test-only -- src/ -v
   --tb=short` — never `src/` together with an explicit file under it
   (pytest 8.4.2 then collects only the explicit file; parent note 26).
   Known and not regressions: the eval-suite and linear-manager tests fail
   on an isolated dir for lack of `plots/` / `artifacts/` (parent note 28).
   Also on Raven, in the isolated dir with `device="cpu"`: load the
   parent's artifact `artifacts/artificial_humans/switch_joint_exodus_gmlp/model/…pt`
   through `GraphNetwork.load` and assert `joint_exodus_head.size_encoding
   == "numeric"`, `joint_exodus_head.mlp[0].in_features == 23`,
   `joint_exodus_size_encoding is None` — the back-compat proof on the real
   bytes. Record test counts and the stand-in reports (*not installed* on
   Raven).

6. **[Opus]** **Retrain the switch model on Raven** — `squeue` check, then
   `AI_REMOTE_DIR='~/autoresearch/switch-exodus-k-onehot' scripts/train_cluster.sh ah configs/training/artificial_humans/switch_predictor/joint_exodus_k_onehot.yml`.
   From the job log: the PROVENANCE line names the shared venv's interpreter
   and `aimanager.__file__` under `~/autoresearch/switch-exodus-k-onehot/src/`;
   `algorithmic-institutions/src` absent; five per-fold `joint exodus loss`
   lines present; elapsed against the 10.6-min ceiling (parent 4m36s). On
   Raven, `sha256sum` the artifact and load it (`device="cpu"`):
   **activation check for training** — `joint_exodus_size_encoding ==
   "onehot"`, head `Linear(39 -> 10) -> Tanh -> Linear(10 -> 81)`,
   **1,291 parameters**; a 23-wide head here means the key was swallowed
   and the run is void. Fetch `artifacts/artificial_humans/switch_exodus_k_onehot`
   (model, metrics, confusion_matrix); local sha256 equals the in-job one.
   **Detach check** from `metrics/*.parquet` with the filter stated in Note
   7 (`name == "log_loss"`, `set == "test"`, `shuffle_feature` NaN, epoch
   374, mean of `value` per `cv_split`): base folds 0.649232 / 0.666026 /
   0.625692 / 0.609553 / 0.544544 (mean 0.6190096203), parent candidate
   0.633162 / 0.663601 / 0.601870 / 0.625205 / 0.541792 (mean
   0.6131260537, signs vs base − − − + −); the train-set values are the
   sharper diagnostic (base 0.49851 / 0.48644 / 0.50671 / 0.49425 / 0.51314
   vs parent 0.49851 / 0.49105 / 0.50388 / 0.49225 / 0.51365 — fold 0
   identical to five digits). A same-sign degradation across all five folds
   with |mean delta| > 0.005 means the detach is not in effect — stop and
   fix. Report the joint loss first → last per fold (parent ~3.0 → 1.92-2.12);
   a final joint loss that does not sit below the parent's on the majority
   of folds means the extra intercepts are not being used — recorded, not
   a stop. Commit the artifact (LFS `*.pt`) and metrics before the step-7
   sync.

7. **[Opus]** **Sim configs, control and candidate, isolated parallel simulation** —
   two new files under `configs/simulation/manager_testing/`, each a
   byte-copy of the parent's candidate config
   `23_2g8a_jexogmlp_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_switch.yml`:
   (a) **control**
   `23_2g8a_kexoctl_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_switch.yml`
   with exactly two edits (`output_dir`, `figure_name` → the `kexoctl`
   name): the parent's candidate stack — its `ecd231f4…` head-on artifact —
   re-simulated on this branch's code, which must reproduce the parent's
   `per_round.parquet` **bit for bit** (`f3c3136c…aa2fa`). That is the
   head-on back-compat proof on the real stack and what licenses judging
   the candidate against the parent's `scores.csv`; (b) **candidate**
   `23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
   with exactly three edits (`switch_model` →
   `artifacts/artificial_humans/switch_exodus_k_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`,
   `output_dir`, `figure_name`). Both parse under `evaluation_sweep.py`'s
   `_self_(\w+?)_contr_(\w+?)_switch$` (contr `gaussian_mlp_v2_group_copula`,
   switch `gnn_joint_exodus` / `gnn_joint_exodus_k_onehot`). Verify by
   `diff` (2 / 3 changed lines; seed 42, 100 episodes, 24 rounds,
   `switch_every 4`, single pairing, `save_per_round: true` byte-identical).
   `squeue` check; one syncing `AI_REMOTE_DIR=... scripts/simulate_cluster.sh
   <control>` (ships `artifacts/`), then `--no-sync` for the candidate; wait
   with one blocking `ssh raven 'while squeue -j <ids> -h | grep -q .; do
   sleep 30; done; sacct -j <ids> --format=JobID,State,ExitCode -n'`. From
   each log: PROVENANCE as in step 6; remote sha256 of all three slot
   artifacts (contribution `da42031a…`, punisher `9e3cf677…` shared; switch
   `ecd231f4…` for the control, step 6's hash for the candidate).
   **Activation check:** the candidate parquet must **differ** from
   `f3c3136c…`. One draw each, seed 42, no re-runs.

8. **[Opus]** **Fetch, evaluate, diagnose** — `AI_REMOTE_DIR=... scripts/fetch_cluster.sh
   plots/simulation/<control dir>` and `<candidate dir>`; confirm the
   control's sha256; locally `python -m aimanager evaluate <candidate
   config>`. Fill the Results row with SC, the mean, rows <= 1, and CG /
   CE / RCD / SB / SA / RSA / RCB / RCA / CA exactly as computed. Then run
   the read-only switch diagnostic — an **uncommitted scratchpad script**
   (the Note-3 analysis; imports `convert.load_human` / `load_sim` only) —
   on human, parent candidate and this candidate: P(full exodus | k) with
   cell counts on complete pairs, P(full | round, k), the full-exodus cell
   share and movers-in-full-exodus share, formation by pre-size and by
   decision round, P(stay 8), the larger-group-size distribution / mean /
   five anchors, P(L = 8) by four-round block, per-round switch rates,
   between-group count correlation per decision round. Apply U1-U3 of §1
   and state each outcome in numbers. Commit sim outputs and evaluation
   only.

9. **[Orchestrator] Verdict, log, PR, clean-up** — §2 on the single
   evaluation, no second stage: `[SUCCESS]` iff SC <= 1.0 **and** mean <=
   1.387806689960726; otherwise `[FAIL]`. State the U1-U3 outcomes in plain
   words in Notes and the PR body regardless of the title, and if the
   other-direction falsifier of §1 applies, say that the mechanism is
   validated and name the residual. Complete the Results table and Notes
   (the hazard-by-k profile against the pre-registration, the oracle's
   prediction against the realised SC, the guard outcomes, the collateral
   `+`/`-`). `gh pr create --base auto/switch-joint-exodus-gmlp`, body
   Hypothesis / Results / Collateral (§9.7), noting the diff shows only
   this experiment's change over PR #172. Delete
   `~/autoresearch/switch-exodus-k-onehot` on Raven **when the PR closes**,
   not at open (parent note 39).

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|

## 4. Notes

1. (Fable, opening) Baseline re-verified from the parent's candidate
   `scores.csv` in this worktree: 21 rows, mean 1.2616424454188417, rows
   <= 1 = 9, SC 1.301036354663599 with numerator 0.25308799999999987 over
   denominator 0.19452799999999945, gate-2 ceiling 1.387806689960726.
   Parent candidate `per_round.parquet` sha256 `f3c3136c…aa2fa` (equals
   parent note 33); base switch artifact sha256 `ecd231f4…1133f33` (equals
   parent note 29). Re-scoring SC myself through
   `scoring.make_repeats(…, 500, 42)` + `score_row` on the parent's parquet
   reproduces 1.301036 / 0.253088 / 0.194528 and the parent's own 2.669271
   / 0.519248 — so the oracle of Note 6 is scored by the exact instrument
   that gates.
2. (Fable, research) **How SC's numerator relates to the plain EMD, and
   what `<= 1` therefore demands.** The numerator is the mean over 500
   repeats of the EMD between a 25-episode human half and a 25-episode
   draw from the sim's 100-episode pool; the denominator is the same
   between the two human halves. The plain full-sample EMD of the
   candidate's larger-group distribution against the human's is 0.2220,
   the scored numerator 0.2531 — the gap is finite-sample noise that a sim
   matching the human distribution exactly would also carry, which is why
   such a sim scores ~1, not 0. Band `<= 1` therefore means "not
   distinguishable from a second human sample at n = 25", not "closer than
   the parent".
3. (Fable, research; read-only, `convert.load_human` / `load_sim`, single
   copy per human game, decision rounds 3/7/11/15/19, complete pairs where
   stated) **The exodus hazard by group size — reproducing the review
   comment and correcting its labels.** The comment's human series 0.269 /
   0.327 / 0.388 / 0.406 and candidate series 0.465 / 0.429 / 0.402 /
   0.385 / 0.233 / 0.205 / 0.190 are **the mean leaver fraction E[m/k | k]
   over all k > 0 cells**, which I reproduce exactly (human all-cells
   0.2692 / 0.3273 / 0.3881 / 0.4062 / 0.2586 / 0.2244 / 0.2415 / 0.1447
   for k = 1..8; candidate 0.4653 / 0.4293 / 0.4017 / 0.3853 / 0.2333 /
   0.2047 / 0.1895 / 0.1711) — **not** P(full exodus | k), which the
   comment names. P(full exodus | k) itself is: human, complete pairs,
   0.1613 / 0.1471 / 0.2000 / 0.1771 for k = 1..4 (n 31 / 34 / 35 / 96;
   5 / 5 / 7 / 17 cells) and **0 / 0 / 0 / 0 for k = 5..8 (n 35 / 34 / 31 /
   19, i.e. 0 of 119)**; human, all k > 0 cells, 0.2692 / 0.1455 / 0.1940 /
   0.1750 and 0 of 171 for k >= 5; candidate 0.4653 / 0.2391 / 0.1795 /
   0.1586 (47/101, 22/92, 14/78, 46/290) and **5 of 355 for k >= 5** (3 at
   k = 5, 2 at k = 7); the parent-of-parent stock GNN 0.4762 / 0.1650 /
   0.0677 / 0.0710 and 0 of 324. So the hump the comment describes is
   real in the leaver fraction (rising to k = 4, collapsing at k >= 5) and
   the hard floor is real, but P(full | k) for humans is roughly flat at
   0.15-0.20 over k = 1..4 rather than peaking at 4; the candidate's
   error is a **2.9x singleton excess on the training support** (1.7x
   against the all-cells human figure the comment used) and a 1.6x pair
   excess, with k = 3 and 4 already at human levels. I cannot reproduce
   the comment's "0 of 141" denominator (complete pairs give 119, all
   cells 171). Human k distribution on complete pairs (the head's training
   support): 31 / 34 / 35 / 96 / 35 / 34 / 31 / 19 cells for k = 1..8.
   Full-exodus cell share and movers-in-full-exodus share: human 0.1079
   (34/315) / 0.2600, candidate 0.1463 (134/916) / 0.3089, stock 0.0831 /
   0.1588 — all three equal the parent's notes 3 and 34.
4. (Fable, research) **Formation, dissolution and the shape of the merged
   share — the comment's numbers located.** Formation P(post = 8 | pre <
   8) over decision transitions: human 0.1227 (27/220) all rounds, **0.260
   at round 3 and 0.0824 (14/170) over rounds 7-19**; candidate 0.1971
   (82/416), **0.220 at round 3 and 0.190 (60/316) over rounds 7-19** —
   the comment's "0.190 against 0.082" is the post-founding rate. Its
   "dissolution 0.298 against 0.300" is P(stay 8), the persistence of the
   merged state (candidate 25/84 = 0.2976, human 9/30 = 0.300); dissolution
   proper is 0.702 vs 0.700. Either way the merged state's exit rate is
   right and its entry rate after founding is 2.3x. By pre-size 4 / 5 / 6 /
   7: human 0.219 / 0.157 / 0.018 / 0.082 (n 64 / 51 / 56 / 49), candidate
   0.200 / 0.167 / 0.141 / 0.267 (n 145 / 78 / 92 / 101), stock 0.068 /
   0.023 / 0.058 / 0.079. The candidate's excess is confined to pre-sizes 6
   and 7 — a pair or a singleton emptying — which is the comment's
   mechanism exactly. Fully-merged share by four-round block from round 4:
   human 0.26 / 0.18 / 0.06 / 0.10 / 0.12, candidate 0.22 / 0.24 / 0.16 /
   0.22 / 0.23, stock 0.10 / 0.02 / 0.05 / 0.08 / 0.04. The comment's
   "climbs to a stationary merged share" is half right: the candidate is
   flat at ~0.22 from the founding round on and never climbs — its founding
   share is already under the human 0.26; what it lacks is the human
   **decay** to 0.06 by rounds 12-15. Larger-group distribution over sizes
   4..8, rounds >= 4: human 0.096 / 0.244 / 0.280 / 0.236 / 0.144 (mean
   6.088), candidate 0.100 / 0.192 / 0.228 / 0.266 / 0.214 (mean 6.302),
   stock 0.190 / 0.330 / 0.258 / 0.164 / 0.058 (5.570); anchors 4/8/12/16/20
   human 6.44 / 6.20 / 6.04 / 5.92 / 5.84, candidate 6.27 / 6.26 / 6.12 /
   6.36 / 6.50 — all equal to parent notes 3, 35 and 36. Between-group
   count correlation by decision round: human −0.694 / −0.515 / −0.357 /
   −0.337 / −0.185 (pooled −0.3667), candidate −0.520 / −0.321 / −0.311 /
   −0.236 / −0.375 (pooled −0.3142). Per-round switch rates human 0.4419 /
   0.2989 / 0.2453 / 0.2414 / 0.2513, candidate 0.4088 / 0.2738 / 0.2450 /
   0.2500 / 0.2225.
5. (Fable, research) **The singleton rate is inherited, not created.** The
   stock per-agent GNN empties singletons at 0.476 and the joint head at
   0.465: the head, given `k/8 = 0.125` and a single agent's pooled
   embedding, reproduced the per-agent model's propensity for the lone
   player rather than the human 0.16. Its leaver fraction is a nearly
   constant ~0.40-0.47 over k = 1..4 and steps down to ~0.2 at k >= 5 — one
   step, where the human profile rises 0.16 → 0.43 and then steps down. A
   10-unit Tanh layer on a scalar can in principle represent a hump, but
   under weight decay and with 96 of 315 training cells at k = 4 the smooth
   fit wins; a free intercept per size configuration removes the contest.
   That is the whole hypothesis, and U1 is the number that tests it.
6. (Fable, research) **A Markov oracle on the larger-group size, scored
   through the suite.** Kernels P(L_post | L_pre, decision round) estimated
   from the candidate's 100 games (cells with n < 10 pooled over rounds),
   trajectories from L = 4 at round 0 written into a minimal canonical
   frame (8 participants, group ids realising L) and scored by
   `score_row(SC)` on the master-seed-42 plan — 20 pools of 100 episodes.
   Candidate kernel: SC 1.3625 +- 0.2221 (min 0.946, max 1.993; real
   1.301); parent-of-parent kernel 2.7197 +- 0.2598 (real 2.669); human
   kernel 0.8077 +- 0.0534 (mean L 6.097, P(L = 8) 0.147). Interventions on
   the candidate kernel, moving only the pre → 8 mass of the named rows to
   the human rate and the excess into "stay": pre = 7 alone → 1.1540 +-
   0.1668, P(SC <= 1) = 0.15, mean L 6.246; pre in {6, 7} → **0.9811 +-
   0.1089, P = 0.60**, mean L 6.176, P(L = 8) 0.140; pre in {5, 6, 7} →
   0.9201 +- 0.0999, P = 0.90; all four formation rows → 0.9737 +- 0.1169,
   P = 0.55. Two readings. First, the pool-to-pool sd of 0.1-0.2 is the
   noise of the one seed-42 draw the verdict rests on — an SC anywhere in
   0.85-1.15 after a working fix is the same outcome drawn twice. Second,
   even the full fix of the small-group hazard leaves the candidate short
   at sizes 5 and 6, which the oracle says formation cannot close; that is
   the other-direction falsifier of §1 and the successor's seed if it
   fires.
7. (Fable, research) **A discrepancy with the parent log's detach
   reference.** Parent note 29 states the 5-fold held-out per-agent
   log-loss at epoch 374 as base 0.5163464160282509 and candidate
   0.5169947301762930. Recomputing from the committed `metrics/*.parquet`
   with `name == "log_loss"`, `set == "test"`, `shuffle_feature` NaN, epoch
   374, mean of `value` per `cv_split` (five rows per fold and epoch, all
   `mask == 0`, `n_pred == 8`, `strategy` None), I get base **0.6190096203**
   (0.649232 / 0.666026 / 0.625692 / 0.609553 / 0.544544) and parent
   **0.6131260537** (0.633162 / 0.663601 / 0.601870 / 0.625205 / 0.541792),
   deltas −0.0161 / −0.0024 / −0.0238 / +0.0157 / −0.0028 — the parent's
   sign pattern (− − − + −) but a mean delta of −0.0059, not +0.00065, and
   a different level. The only rows near 0.5163 are train-set rows at
   epochs 145-155. I could not identify the parent's filter; the values
   quoted in step 6 are the ones this experiment's detach check compares
   against, with the filter stated so it is reproducible. The train-set
   log-loss is the sharper trunk-identity diagnostic anyway: base 0.49851 /
   0.48644 / 0.50671 / 0.49425 / 0.51314 vs parent 0.49851 / 0.49105 /
   0.50388 / 0.49225 / 0.51365, fold 0 identical to five digits. The parent's
   joint loss first → last by fold: 3.006 → 1.999, 2.971 → 1.949, 3.036 →
   1.957, 2.951 → 1.924, 3.034 → 2.117.
8. (Fable, opening) **Why an option and not an unconditional change.** The
   head is pickled whole inside the artifact (`save()` stores the module;
   `load()` does `cls(**to_load)`, so `__init__` never runs for a loaded
   head and its 23-wide `mlp` is restored as-is). An unconditional one-hot
   `forward` would hand that MLP a 39-wide input and the parent's artifact
   could no longer run — which would make the step-7 control (the parent's
   candidate stack re-simulated bit-identically on this branch) impossible,
   and that control is what licenses judging against the parent's
   `scores.csv` instead of re-scoring it. A `size_encoding` attribute with
   the parent's behaviour as the default, plus a `__setstate__` default for
   heads pickled before the attribute existed, is the smallest change that
   keeps both. The same reasoning makes step 2's kwarg explicit rather than
   letting `**_` swallow it: `train.py` passes `**model_args`, and a
   swallowed key would train the parent's head under this experiment's
   name with no error anywhere.
9. (Fable, opening) **Test surface the encoding change touches, read from
   the files.** Directly broken by a one-hot default or an unconditional
   change: `tests/switch/test_joint_exodus.py::test_head_feature_normalisation_convention`
   (asserts `features.shape == (1, 2, 2*5 + 2 + 1)`, columns 10 and 11
   equal `3/SIZE_NORM` and `5/SIZE_NORM`, column 12 the round, and
   `SIZE_NORM == 8.0`) and
   `tests/switch/test_joint_exodus_detach.py::test_the_cut_is_exactly_the_pooled_embedding`
   (asserts the block after the pooled features `th.equal`
   `counts.round() / SIZE_NORM`, and imports `SIZE_NORM` at line 124).
   With the numeric default both keep passing unchanged and gain onehot
   siblings (Plan steps 1 and 3). Unaffected by construction:
   `test_joint_exodus_graph.py::test_the_head_reads_the_post_rnn_width`
   (asserts `embed_size == HIDDEN`, not the input width),
   `test_joint_exodus_detach.py:242` (`len(on_head) == 4` — still two
   `Linear` layers), the loss, sampling and parity suites (counts and RNG
   discipline, not features). No test asserts the 1,131 parameter count;
   it lives in the parent's log, and 1,291 is this experiment's expected
   figure.
10. (Orchestrator, steps 1-5 confirmed) **The numeric path is bit-identical,
    proved twice and on the real bytes.** Locally I loaded
    `HEAD:src/aimanager/generic/joint_exodus.py` as a second module in one
    process: every parameter `th.equal` under the same seed with and without
    the explicit keyword, and `forward`'s `log_prob` and `k` `th.equal`
    including a batch carrying an invalid decider. The step-2 agent did the
    same for `graph.py` against the parent head `bc77932` — identical
    `state_dict`, `predict_independent` output, probabilities and post-call
    RNG state, head-off and head-on, at rounds 0 and 7, with the round-7
    head-on draw differing from the independent path so the identity is not
    vacuous. On Raven the parent's artifact `ecd231f4…1f33` (sha re-verified
    after transfer) loads through `GraphNetwork.load` against **real PyG**
    with `joint_exodus_size_encoding is None`, head `size_encoding
    == "numeric"`, `in_features == 23`, 1,131 head parameters; a fresh
    onehot model in the same process gives 39 and 1,291. `aimanager`
    resolved at `~/autoresearch/switch-exodus-k-onehot/src/aimanager/`, the
    shared venv's interpreter, `algorithmic-institutions/src` absent.
11. (Orchestrator, steps 1-5 confirmed) **Test counts, and what the new
    coverage actually binds.** `tests/switch` 141 → 150 → 168 → 184 over
    steps 1-3; full local set 542 passed (parent's comparable figure 499,
    the difference being this experiment's additions). On Raven with real
    PyG, `tests/switch` collected 184 and gave **182 passed, 2 skipped** —
    the two skips are `needs_baseline`-marked git-recovery tests that cannot
    build their baseline module because `remote_test.sh` excludes `.git/`
    from the rsync, which is the fallback their own docstring describes, and
    the git-free sibling `test_head_off_matches_the_pre_change_expression`
    passed. `src/` collected 98 items: the PyG unit suites (edge encoder 8,
    encoder 4, environment 7) and the train/sim parity suite 8/8 all passed;
    the 10 failures/errors are all the parent's note-28 category, missing
    `plots/` and `artifacts/` in an isolated dir. Two negative controls
    establish the coverage is not decorative: undeclaring the plumbing
    keyword so it falls into `graph.py`'s `**_` sink fails 13 tests while
    every head-off and `None` case still passes, and swapping the two
    groups' one-hot codes fails 3 tests including the detach feature-tail
    check — so the label-order convention the flip-doubling depends on is
    now enforced rather than merely documented.
