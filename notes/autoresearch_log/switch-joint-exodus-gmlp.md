# Autoresearch log: switch — joint exodus head on the gmlp group-copula stack

Branch `auto/switch-joint-exodus-gmlp` (worktree
`.claude/worktrees/switch-joint-exodus-gmlp`), created from
`origin/auto/contribution-gmlp-group-copula` at `5337473` — the head of the
maintainer-designated parent PR #170, per §9 "Building on a `[SUCCESS]` PR".
The PR opens with `--base auto/contribution-gmlp-group-copula`.

## 1. Declaration

- **Slot:** switch.
- **Parent PR:** **#170** `[SUCCESS] contribution: marginal-preserving group
  copula on gaussian_mlp_v2` (`auto/contribution-gmlp-group-copula`). Its
  log is `notes/autoresearch_log/contribution-gmlp-group-copula.md`.
- **Base model:** the stock GNN switch predictor as it sits in #170's stack,
  `artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`
  (sha256 `184f7f5c8ed326d49983fe455ef6478715fcac79c8161f08fa685b9cfb25d037`,
  the same bytes PR #168 note 11 recorded), trained by
  `configs/training/artificial_humans/switch_predictor/opt_50ep_doubled_reanchored.yml`
  — `x_encoding = common_good, punishment, agent_group, round_number`,
  `edge_encoding = []`, `y_levels = 2`, hidden 10, 375 epochs, batch 10,
  lr 5e-4, 5-fold, seed 38381, flip-doubled data. Held-out per-agent
  log-loss over the 5 folds: **0.5163464160282509** (from its
  `metrics/*.parquet`, epoch 374, unperturbed) — the detach reference of
  step 10.
- **Mechanism source:** PR #171 `[SUCCESS] Joint exodus head for the switch
  slot` (`auto/switch-joint-exodus`, on the PR #165 lineage). **Not ported,
  not cherry-picked**: reimplemented fresh on this tree using #171's code as
  direct inspiration, in the same files, with the lineage-A copula surface
  (which does not exist here) simply absent. #171's `.pt` is not reusable
  (`GraphNetwork.load` does `cls(**to_load)` and its dict carries
  `copula_rho/phi/switch_every`), so the switch model is retrained here.
- **Evaluation stack (§3, parent rule of §9):** #170's own config
  `configs/simulation/manager_testing/23_2g8a_gmlpcop_self_gaussian_mlp_v2_group_copula_contr_gnn_switch.yml`
  — `gaussian_mlp_v2 + group copula` contributor
  (`contribution_gaussian_mlp_v2_group_copula.joblib`), PR #160
  severity-copula multinomial punisher, single pairing
  `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
  `switch_every: 4`, `save_per_round: true` — with **only `switch_model`
  swapped** to the candidate (plus `output_dir`/`figure_name`). Everything
  else byte-identical.
- **Baseline for BOTH §2 gates**, re-verified from
  `plots/simulation/23_2g8a_gmlpcop_self_gaussian_mlp_v2_group_copula_contr_gnn_switch/evaluation/scores.csv`
  (21 rows; `per_round.parquet` sha256
  `72599716d3e93f37a604e7b4a893090100e080e5ee5da6fe6d2e7f9928d0d429`):

  | quantity | value | band |
  |---|---|---|
  | **SC** (target) | **2.669271261720684** (EMD 0.5192479999999997 / ceiling 0.19452799999999945) | 2-5 |
  | mean over 21 rows | 1.3138927788530981 | — |
  | **gate-2 ceiling (+10%)** | **1.4452820567384081** | — |
  | rows <= 1 | 10/21 | context |
  | RCD (guard) | 0.6700998716392149 | <= 1 |
  | RCB (guard) | 1.910007740580805 | 1-2 |
  | SB (watch) | 0.8440272761008656 | <= 1 |
  | SA (watch) | 0.7165907575706952 | <= 1 |
  | RSA (watch) | 0.9561739624425271 | <= 1 |
  | CG · RCA · CA | 2.82924493473058 · 3.49343199120526 · 1.641643032985992 | 2-5 · 2-5 · 1-2 |

- **Target row:** **SC 2.669271261720684**, band 2-5 → requires < 2.0
  (band 1-2), i.e. the EMD numerator must fall below **0.389056**.
- **Gate 2:** 21-row mean must stay below **1.4452820567384081**.
- **Guards (declared, non-gating, with the reading fixed in advance):**
  - **RCD 0.6700998716392149 (<= 1).** The row PR #171's
    conditional-Bernoulli half was built to protect and instead degraded a
    full band (1.9647 → 2.7649), because a drawn count `m = k` leaves
    exactly one subset and no propensity selection happens at all. On this
    stack the same absolute movement (+0.80) would land at ~1.47 (`1-2`).
    **The unsoundness criterion**, decided now: the candidate's
    full-exodus cell share (cells with `k_g > 0` on complete pairs, all
    decision rounds) is compared with the human **0.1079** and its share of
    movers inside full-exodus cells with the human **0.2600** (both
    re-measured in Note 3; parent 0.0831 / 0.1588). If SC is band-upgraded
    while the full-exodus share **exceeds** the human 0.1079, SC was bought
    through excess full-group exodus — the #171 failure mode — and the
    mechanism is ruled **mechanistically unsound** whatever the gates say
    (a `[SUCCESS]` title would still be honest under §2, but the PR body
    and this log say so in plain words, and a successor must not stack on
    it without fixing the dose). If the share is at or below 0.1079 and
    RCD still leaves `<= 1`, the conditional-Bernoulli *selection* in
    partial cells is what fails and the defence is falsified for good.
  - **RCB 1.910007740580805 (1-2).** #171 pushed it into `2-5` (+0.286).
    Named so it is not read as noise if it moves.
  - **SB 0.8440272761008656 (watch).** PR #169 note 11 forecast and #171
    confirmed (0.859 → 1.065) that driving the simulation into more
    unbalanced states raises the late-round switch rate. Parent per-round
    rates 0.439 / 0.311 / 0.275 / 0.251 / 0.228 against human 0.442 /
    0.299 / 0.245 / 0.241 / 0.251 (Note 3).
  - **RSA 0.9561739624425271** sits 0.044 under the ceiling and moved
    +0.42 on #171; it will leave `<= 1` on any co-switching that is not
    punishment-driven. Reported, not gating.
- **Behavioral claim (§5, one sentence):** at the *founding* of the groups a
  migration wave has a direction — when the minority side of a split
  empties, the majority side holds rather than moving too — so the two
  groups' leaver counts in the same decision round are opposed rather than
  independent, and reproducing that opposition restores the full-merge
  transitions that build SC's missing right tail (larger-group size 8 in
  14.4% of human rounds against 5.8% here).
- **Planned change (one change, switch slot only):** the PR #171 mechanism,
  reimplemented — a **round-level joint head over the pair of leaver counts
  `(m_0, m_1)`**, read off the two group-pooled post-RNN node embeddings
  plus both valid-decider counts and the round, fitted by cross-entropy
  alongside the *unchanged* per-agent loss as a **detached readout** (the
  trunk's objective is bit-identically the base run's), with
  **conditional-Bernoulli** selection of *which* members leave given the
  drawn counts. **Exactly as #171 built it, no modification**: the joint
  draw replaces the independent per-agent draw on every decision round
  (`(round + 1) % joint_exodus_switch_every == 0`, i.e. rounds 3, 7, 11, 15,
  19), the head trains on every decision round's count pair (334 of 500
  complete pairs, incomplete pairs dropped per #171 note 3) and conditions
  on the round. No firing-schedule parameter exists on this branch
  (maintainer ruling, Note 5).
- **Measured deficit profile of this stack (Note 3 has the full numbers;
  recorded as findings, not as the basis for any parameter).** The
  starting deficit is larger than #171's (mean larger-group size 5.570 vs
  its parent's 5.676; human 6.088) and sits at the founding of the groups:
  the per-round larger-group line is 0.73 short at anchor round 4 and 0.27
  short at round 20. Gross flow at round 3 already matches (3.51 movers per
  game vs human 3.58); the arrangement does not — between-group
  correlation of the count pair −0.356 vs human −0.694, "one side empties
  while the other holds" 0.100 vs 0.289. By decision round the human
  correlation is −0.694 / −0.515 / −0.357 / −0.337 / −0.185 against the
  parent's −0.356 / −0.370 / −0.108 / −0.345 / −0.172, and the full-exodus
  cell share human 0.211 / 0.109 / 0.057 / 0.066 / 0.066 against parent
  0.110 / 0.084 / 0.091 / 0.067 / 0.062 — the parent is short at rounds 3
  and 7 and already at the human rate at rounds 11-19. These are the
  per-round facts against which the candidate's own per-round profile is
  to be read, and a seed for whoever comes after.
- **Pre-registered prediction (Note 4).** A Markov oracle on the
  larger-group size that reproduces the parent's own SC (2.673 vs 2.669)
  puts the all-rounds human kernel — the in-sample ceiling for this
  mechanism — at mean **6.09 against the human 6.088** (EMD ~0.002). That
  is the same oracle #169 note 9 reported (6.1078) and #171 then
  **overshot in the closed loop to 6.286**, from a parent at 5.676. Here the
  parent starts at 5.570, so the honest expectation sits **between the
  oracle and #171's realisation**, and the named risk under this schedule
  is the same overshoot of the fully-merged mass (#171: 0.214 vs human
  0.144; parent here 0.058) with its collateral (RCD out of `<= 1`, RCB
  into `2-5`, SB out of `<= 1`), which would pass §2 on this stack's 0.131
  of gate-2 headroom and be ruled on by the unsoundness criterion above.
  **The experiment proceeds to simulation whatever the retrained head's
  loss curve looks like**; only an implementation failure (parity or detach
  test failing, trunk log-loss diverging from the base by more than the
  RNG-realisation band #171 measured, control run not bit-identical) stops
  it.
- **Falsifiers, stated before the run.** (a) SC not leaving `2-5`: the
  between-group opposition the head represents is not what this stack is
  missing. (b) SC upgraded with the full-exodus share above the human
  0.1079: the unsoundness call — SC bought through excess full-group
  exodus. (c) RCD leaving `<= 1` with the share at or below 0.1079: the
  conditional-Bernoulli defence is dead as a design element and should be
  dropped by a successor, not re-argued. (d) The late-game anchors (16, 20)
  rising where the human line declines (5.92 → 5.84): under all-rounds
  firing this is a **prediction to test**, not a design flaw to avoid — the
  parent already matches the human full-exodus rate at rounds 11-19 (Note
  3), so if the candidate's late anchors rise, the head is adding exodus
  where the independent draw was already right, and that per-round profile
  is the strongest thing this run can hand a successor.
- **Iteration budget (§5):** base training 3m32s on one A100; #171's head
  added 21% (4m17s, job 29870374). Ceiling 3x = ~10.6 min. Two sims of
  ~3 min each. Comfortably inside.
- **Legality and frozen surface.** No new input feature — the head
  refactorises the *label* distribution and reads `agent_group` and
  `round_number`, both of which the base model already encodes. No seed,
  episode count, scoring parameter or protocol field changes. Nothing under
  `src/aimanager/evaluation_suite/`, `notes/evaluation_metric_defs.md`,
  `notes/eval_scoring_schema.md` or `experiments/` is touched; the two
  diagnostics of Note 3 import `convert.load_human` / `load_sim` read-only.
  No other branch's log file is written.

## 2. Plan

Steps for the orchestrator to validate (§2 targets, §5 legality, §8 frozen
surface) and tag. Paths are relative to the worktree. `AI_REMOTE_DIR='~/autoresearch/switch-joint-exodus-gmlp'`
on every `train_cluster.sh` / `simulate_cluster.sh` / `fetch_cluster.sh` /
`remote_test.sh` call, and `squeue -u certuer` (over a *live* SSH tunnel —
an expired ControlMaster returns an empty queue, #171 note 26) before any
syncing call. Reference sources for every port are #171's files under the
orchestrator's `scratchpad/ref171/`; nothing from there is copied into the
worktree verbatim, and nothing mentioning `copula` exists on this tree.
Lint (`black src/`, `flake8 src/ --max-line-length=88 --extend-ignore=E203,W503`)
runs once per step before staging, not after every edit.

1. **[Sonnet] Harden the training SLURM template** — `scripts/artificial_humans/run_training.sh`
   (existing; a Python `.format()` template rendered by
   `src/aimanager/artificial_humans/run.py::create_script`, so every literal
   brace must be doubled). Directly after the
   `source "${{AIMANAGER_VENV:-.venv}}/bin/activate"` line add
   `export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"`, and directly
   before `{command}` add the same PROVENANCE line `scripts/run_simulation.sh`
   already carries: `python -c "import sys, aimanager; print('PROVENANCE', sys.executable, aimanager.__file__)"`
   (no single braces in it). Render the template once locally through
   `create_script` with dummy kwargs and inspect the output. Why: this
   experiment retrains on Raven from an isolated dir, and `main`'s
   isolation for training rests on `SBATCH_EXPORT=ALL` alone with no in-job
   fallback and no provenance echo (#171 note 8; the silent-wrong-code
   hazard that voided PR #166). Own commit; nothing else in this step.

2. **[Opus] The joint exodus head module** — new `src/aimanager/generic/joint_exodus.py`
   (torch only, no PyG import). Port #171's design: constants
   `MAX_GROUP_SIZE = 8`, `N_GROUPS = 2`, `ROUND_NORM = 23.0`, `SIZE_NORM = 8.0`
   (the `IntEncoder(encoding="numeric")` convention `v / (n_levels - 1)` the
   model's own `round_number` feature uses); `pool_by_group(x, agent_group,
   batch, *, n_batch, mask, n_groups)` — masked mean-pool per (graph, round,
   group **label**), group the fastest axis, empty cells pool to zero not NaN;
   `joint_count_mask(k)` and `masked_joint_log_prob(logits, k)` — `-inf` on
   `m_g > k_g` before one `log_softmax` over the flattened 9 x 9 grid, `(0,0)`
   always valid; `JointExodusHead(embed_size, hidden_size)` — `Lin(2*F + 2 + 1,
   hidden) → Tanh → Lin(hidden, 81)`, whose `forward(x, *, agent_group,
   round_number, batch, n_batch, decider_mask)` pools, **detaches the pooled
   embedding** (the step-2b cut of #171: the joint loss is ~2-3 nats against
   ~0.5 for the per-agent term and would otherwise re-fit the trunk), appends
   `k / 8` and `r / 23`, and returns `(log_prob (n_batch, R, 9, 9), k
   (n_batch, R, 2))`. Docstrings explain label-order pooling (the
   flip-doubled data makes the two copies transposes on the grid), the
   `agent_group`-not-edge-index mask (the graph is complete over all 8), and
   that `k` counts valid *deciders*. Drop every reference to `generic/copula.py`.
   Tests: new `tests/switch/test_joint_exodus.py` (plain pytest, local):
   masked softmax sums to one over valid cells and is exactly zero elsewhere,
   degenerate `(8,0)` / `(0,8)`, batching over leading dims, finite gradients
   through the mask, pooling equals a plain Python group mean, pooling is
   canonical in label not size, empty group → zero without NaN, decider mask
   respected, batch elements separated, head emits a valid masked joint, the
   fully-merged state, the feature normalisation convention, determinism and
   differentiability. `tests/switch/` is a new directory beside
   `tests/baselines/`.

3. **[Opus] The conditional-Bernoulli sampler** — new
   `src/aimanager/generic/conditional_bernoulli.py` (torch only). Port #171's
   design: fixed row width `MAX_GROUP_SIZE = 8` with a boolean `mask` marking
   real members (`k = mask.sum(-1)`, `k == 0` an all-False row);
   `conditional_bernoulli_log_prob(p, m, *, mask, max_group_size)` enumerates
   all 256 subsets, accumulates each subset's weight as a **sum of log-odds**
   (`log p - log1p(-p)`, clamp `1e-12`, padded slots at a neutral finite
   logit so `0 * -inf` never appears), masks to size `m` and no padded slot,
   one `log_softmax`; `sample_conditional_bernoulli(...)` makes **exactly one**
   `th.multinomial` over the batch from the global RNG and returns the bool
   `(B, 8)` selection. Tests: new `tests/switch/test_conditional_bernoulli.py`
   (local): marginals recovered when `m` is drawn from the Poisson-binomial of
   `p`; propensity ordering preserved; exactly `m` selected for every `(k, m)`
   including `m = 0` and `m = k`; `k = 1`; equal `p` → uniform; extreme `p`
   (`1e-9`, `1 - 1e-9`) finite; determinism under seed; rejects `m > k`,
   negative `m`, wrong width, mismatched lengths.

4. **[Opus] The `GraphNetwork` gate** — `src/aimanager/generic/graph.py` (existing;
   byte-identical to `main`, no copula surface). (a) Imports of
   `JointExodusHead` and `sample_conditional_bernoulli`. (b) `__init__` gains
   keyword args `joint_exodus=False`, `joint_exodus_head=None`,
   `joint_exodus_switch_every=None` — #171's three, nothing more; asserts:
   `joint_exodus` is a `bool`; only with `y_name == "does_switch"`;
   `joint_exodus_switch_every` None or positive non-bool `int`, only with the
   head; store both scalars. **No mutual-exclusion assert
   against a copula** — there is none here. (c) In the `op1 is None` branch
   the head is built **last** (`JointExodusHead(embed_size=x_features,
   hidden_size=hidden_size)` where `x_features` is the post-RNN width), so
   every pre-existing parameter is initialised from exactly the RNG state it
   saw before; in the `else` (load) branch `self.joint_exodus_head =
   joint_exodus_head`; after both, assert `(head is not None) ==
   joint_exodus`. (d) `forward(self, data, reset_rnn=True, return_joint=False,
   decider_mask=None)`: after the RNNs and before `op2`, if `return_joint`
   and the head exists, `joint = head(x, agent_group=data["agent_group"],
   round_number=data["round_number"], batch=batch, decider_mask=data.get("mask")
   if decider_mask is None else decider_mask)`; return `(x, joint)` when
   `return_joint` else `x` — every existing caller keeps its signature.
   (e) `encode()`: when the head exists, assert `agent_group` and
   `round_number` are in `data` and carry both, flattened `(N, R)`, into
   `encoded`; a model without the head encodes exactly today's keys. (f)
   `save()`: append `"joint_exodus"`, `"joint_exodus_head"`,
   `"joint_exodus_switch_every"` to `to_save`.
   An artifact without those keys loads through `cls(**to_load)` with the
   head absent and behaves as today. Tests: new
   `tests/switch/test_joint_exodus_graph.py` (PyG stand-ins installed only
   when `torch_scatter` / `torch_geometric.nn` are absent, #171's discipline;
   every assertion is an invariance so a stand-in cannot manufacture a pass):
   head off by default and nothing extra encoded; head-off sampling bitwise
   identical to the pre-change expression *including the global RNG state*;
   head-on trunk `state_dict` equals head-off under the same seed and the
   per-agent logits are bit-identical; `return_joint` is None with the head
   off; valid masked joint through the real encode path for `(3,5)`, `(8,0)`,
   `(0,8)`; decider mask shrinks the support; encode requires membership when
   on; save/load round-trips the head and `switch_every`; an artifact
   stripped of the new keys loads and samples bit-identically; guards
   (non-switch head, non-bool gate, gate/head disagreement, bad
   `switch_every`, `switch_every` without the head). Runs locally and on
   Raven.

5. **[Opus] The detached joint training objective** — `src/aimanager/artificial_humans/train.py`
   (existing; byte-identical to `main`). Port #171's design: module constant
   `DROP_INCOMPLETE_PAIRS = True` with the ruling written above it (109 of
   2,000 human decision rows fail `switch_valid`, leaving 84 of 465 group
   cells short and 83 of 250 pairs with `k_0 + k_1 < 8`; 18 of 112 apparent
   full-exodus cells are timeouts and carrying them inflates P(full exodus)
   0.1079 → 0.1204; the between-group correlation is indifferent, −0.3660 vs
   −0.3658; dropping keeps 334 of 500 doubled pairs and matches the object
   the predecessor's oracle resampled); `joint_exodus_counts(y, mask,
   agent_group, batch, *, n_batch)` → `(m, k)` int64 `(n_batch, R, 2)` via
   `pool_by_group`'s count channel — which returns **floats** (it
   accumulates mask weights in `x.dtype`), so this function must do its
   own `.round().to(th.int64)` exactly as `JointExodusHead.forward`
   does; the raw return cannot be used as an index (found in step 2) on `decides = mask` and `leaves = mask &
   y`, so the loss scores the quantity the head emits by construction;
   `joint_exodus_loss(joint, batch_data)` → gathers `-log_prob` at
   `m_0 * 9 + m_1`, asserts `k == k_head` and `m <= k`, selects cells with
   `k.sum(-1) > 0` (and `== n_player` under the drop ruling) by boolean
   indexing — never a 0/1 multiply, since dropped cells can sit on `-inf` —
   and returns `(mean nll, n_cells)` (a differentiable zero when empty);
   `compute_batch_loss(model, batch_data, loss_fn, l1_entropy)` → with the
   head off, *exactly* the pre-existing per-agent expression (same forward,
   same value); with it on, `model(batch_data, return_joint=True)` and
   `agent_loss + joint_loss`, returning the components. The training loop
   calls `compute_batch_loss`, accumulates `sum_loss` from the **agent**
   component only (so the recorded `loss` curve stays comparable with
   head-off runs), records `joint_exodus_loss` through `rec.rec(...,
   name="joint_exodus_loss", set="train")` and the wandb log, shows it in the
   tqdm postfix, and prints one per-fold line at the end: agent loss, joint
   loss first → last, count pairs per batch. `eval_model` /
   `create_confusion_matrix` are untouched (they call `predict_encoded`, the
   legacy per-agent path), so the recorded held-out log-loss is directly
   comparable with the base artifact's 0.5163464160282509. Tests: new
   `tests/switch/test_joint_exodus_loss.py` (hand-worked count pair; indexed
   by label not size; an invalid decider; a fully merged round; several
   graphs per batch; leavers never exceed deciders; decision rounds follow
   `switch_every`; the loss selects exactly the decision rounds; incomplete
   pairs dropped by default and a fully merged complete pair kept; a batch
   with no usable pair yields a finite zero; the observed pair is never on a
   masked cell; training counts equal the head's own pooling; a mismatched
   pair is caught; **head off reproduces the legacy loss bit for bit, with
   and without entropy regularisation**; head on keeps the per-agent
   component identical; the joint objective descends on a synthetic batch)
   and new `tests/switch/test_joint_exodus_detach.py` (trunk gradients
   bitwise identical head-on vs head-off; the trunk gradient is the
   per-agent gradient alone; the joint term alone moves the head and nothing
   above it; the cut is exactly the pooled embedding). Both use the stand-in
   discipline of step 4 and run locally and on Raven.

6. **[Opus] The two-stage joint draw in the simulation path** —
   `src/aimanager/generic/graph.py` (existing). New method
   `_predict_encoded_joint_exodus(self, encoded, shape, reset_rnn=True)`:
   asserts the head, `joint_exodus_switch_every`, `y_levels == 2` and
   `n_nodes <= 8`; `self.eval()`; `y_logit, joint = self(encoded, reset_rnn,
   True)`; softmax; **the legacy `self.y_encoder.decode(y_pred_proba, True)`
   draw taken verbatim** (off a firing round it is the only draw, so a
   non-firing round leaves the global RNG exactly where the independent path
   would — the switch model runs every round to keep the GRU warm,
   `manager/environment.py::step`); then, per round `r` with
   `(round + 1) % joint_exodus_switch_every == 0` — **the only firing
   condition**, so with `switch_every 4` every decision round 3, 7, 11, 15,
   19 fires, #171's behaviour: one `th.multinomial` over the flattened
   masked joint gives the pair `(m_0, m_1)` per batch element (row-major, the
   same flattening the loss gathers with); membership is read **pre-switch**
   from `encoded["agent_group"][:, :, r]`, asserted equal to the head's own
   `k[:, r]`; one batched `sample_conditional_bernoulli` call over both groups
   (rows group-major, `p` tiled, `mask = member`, `max_group_size = n_nodes`)
   selects who leaves; the two disjoint selections are OR-ed into `y_pred[:, r]`.
   Three categorical draws on a firing round, one otherwise. Dispatch in
   `predict_independent`: `if sample and self.joint_exodus_head is not None:
   ... _predict_encoded_joint_exodus(...)` else the existing
   `predict_encoded(...)` — the only new branch. `predict_encoded` itself is
   untouched. **No change to `manager/environment.py` or
   `simulation/simulate.py`**: `environment.py:361` already calls
   `.predict(state, reset_rnn=..., edge_index=...)` once for all 8 agents of
   both groups with `state["agent_group"]` and `state["round_number"]`
   present, `apply_switch` consumes the returned bool vector, and
   `simulate.py::load_ah_model` dispatches `.pt` to `GraphNetwork.load`,
   whose `cls(**to_load)` accepts the new keys — verified by reading, and
   re-verified by step 7's parity test and step 11's activation check.
   Tests: new `tests/switch/test_joint_exodus_sampling.py` (stand-in
   discipline): head off bitwise identical including RNG at several rounds;
   head off costs exactly one categorical; a legacy artifact without
   `switch_every` loads; a non-decision round consumes no extra RNG and
   costs one categorical; a decision round costs three; switcher counts
   equal the drawn pair; the leavers are members of the group the count was
   drawn for (a transposition is caught); no switchers on `(0,0)`; fully
   merged rounds `(8,0)` / `(0,8)` sample without error; the whole group can
   leave; determinism; `switch_every` round-trips through save/load;
   sampling without `switch_every` fails loudly; `sample=False` never
   reaches the joint path; `switch_every` requires the head and must be a
   positive int. #171's mutual-exclusion-with-the-copula test is **not**
   ported, and nothing is added in its place.

7. **[Opus] Train/sim parity test** — new
   `src/aimanager/tests/test_joint_exodus_train_sim_parity.py` (local; stand-ins
   only to satisfy `train.py`'s import of `AH_MODELS`). Port #171's design:
   four synthetic scenarios (unbalanced one-sided 5-3, fully merged 8-0
   un-merging, unbalanced both sides, and a label-symmetric 4-4 with a 2-2
   exchange whose swap is invisible to totals) built as a raw human-shaped
   frame through `generic/data.py::parse_agent_rounds` / `create_torch_data`
   on the training side and driven through a real `ArtificialHumanEnv.step()`
   with a scripted switch predictor on the simulation side; assert the
   **pre-switch** membership captured at decision time equals `data.py`'s
   `agent_group` at the decision round agent for agent, that pre != post,
   that `joint_exodus_counts` gives identical `(m, k)` on both sides per
   scenario, that every declared case is hit, and that a label swap is caught
   by the per-agent check where a totals-only check is blind. Closes the
   pandas-vs-torch invariant the mechanism depends on (#169 note 3a).

8. **[Sonnet] Training config** — new
   `configs/training/artificial_humans/switch_predictor/joint_exodus.yml`: a
   copy of `opt_50ep_doubled_reanchored.yml` in which **only** `description`,
   `output_dir: artifacts/artificial_humans/switch_joint_exodus_gmlp`, and
   two `model_args` keys differ — `joint_exodus: True`,
   `joint_exodus_switch_every: 4`. Every
   hyperparameter (375 epochs, batch 10, lr 5e-4, wd 1e-3, hidden 10, 5-fold,
   seed 38381, `mask_name: switch_valid`, `switch_every: 4`, flip-doubled
   `experiments/2group_8agent_50ep.csv`, labels `architecture: mlp+rnn+edge`,
   `dataset: 50ep_doubled`) stays verbatim, so the artifact file name is the
   base's `architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` under the new
   dir. `switch_every` does not enter training; it is persisted into the
   artifact for step 6's sampler. Verify with `diff` against the base config
   and against #171's `joint_exodus_train.yml` (only `description` and
   `output_dir` may differ).

9. **[Sonnet] Tests, local then Raven** — locally, with `PYTHONPATH=$PWD/src` and the
   **main checkout's** interpreter,
   `/Users/ertuerkan/Desktop/algorithmic-institutions/.venv/bin/python`
   (`uv sync` in the worktree is not available — see note 8): `pytest
   tests/switch src/aimanager/tests/test_joint_exodus_train_sim_parity.py
   tests/baselines src/aimanager/tests/test_eval_*.py -q` all green; each
   stand-in module reports whether stand-ins were installed. Then on Raven
   from the isolated dir (`squeue` check, then `AI_REMOTE_DIR=...
   scripts/remote_test.sh -- tests/switch
   src/aimanager/tests/test_joint_exodus_train_sim_parity.py src/ -v
   --tb=short`) against real `torch_scatter` / `torch_geometric`, with the
   stand-in reports confirming *not installed*. Known and not a regression:
   `test_eval_*` and `test_linear_manager` fail on an isolated dir for lack
   of `plots/` / `artifacts/` (#170 note 11g). Record test counts.

10. **[Opus] Retrain the switch model on Raven** — `squeue` check, then
    `AI_REMOTE_DIR='~/autoresearch/switch-joint-exodus-gmlp' scripts/train_cluster.sh ah configs/training/artificial_humans/switch_predictor/joint_exodus.yml`.
    From the job's own log: the PROVENANCE line must name the shared venv's
    interpreter and `aimanager.__file__` under
    `~/autoresearch/switch-joint-exodus-gmlp/src/`; the string
    `algorithmic-institutions/src` must not appear; the five per-fold
    `joint exodus loss` lines must be present (the shared tree cannot print
    them); elapsed time recorded against the 10.6-min ceiling. On Raven,
    `sha256sum` the artifact and load it through `GraphNetwork.load` in the
    isolated dir: `joint_exodus True`, `joint_exodus_switch_every 4`, head
    parameter count (#171: 1,131). Fetch
    `artifacts/artificial_humans/switch_joint_exodus_gmlp` (model, metrics,
    confusion_matrix); local sha256 equals the in-job one. **Detach check**
    from `metrics/*.parquet`: the 5-fold held-out per-agent log-loss at the
    last epoch against the base's **0.5163464160282509**; #171 measured
    +0.00065 with mixed signs across folds (a different RNG realisation, the
    head's construction consumes RNG) against +0.0042 systematic when
    attached. A systematic same-sign degradation across folds means the
    detach is not in effect — stop and fix. Report the joint loss first → last
    per fold. Commit the artifact (LFS-tracked `*.pt`) and metrics so the
    step-11 sync ships identical bytes and `rsync --delete` cannot remove
    them.

11. **[Sonnet] Sim configs, control and candidate, isolated parallel simulation** —
    two new files under `configs/simulation/manager_testing/`, each a
    byte-copy of #170's
    `23_2g8a_gmlpcop_self_gaussian_mlp_v2_group_copula_contr_gnn_switch.yml`:
    (a) **control** `23_2g8a_jexoctl_self_gaussian_mlp_v2_group_copula_contr_gnn_switch.yml`
    with exactly two edits (`output_dir`, `figure_name` → the `jexoctl`
    name), the parent's own stack re-simulated on this branch's code from the
    isolated dir — it must reproduce the parent's `per_round.parquet`
    **bit for bit** (sha256 `72599716d3e9…d429`), which is what licenses
    judging the candidate against the parent's `scores.csv`; (b)
    **candidate** `23_2g8a_jexogmlp_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_switch.yml`
    with exactly three edits (`switch_model` →
    `artifacts/artificial_humans/switch_joint_exodus_gmlp/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`,
    `output_dir`, `figure_name`). Both parse under
    `evaluation_sweep.py`'s `_self_(\w+?)_contr_(\w+?)_switch$` (contr
    `gaussian_mlp_v2_group_copula`, switch `gnn` / `gnn_joint_exodus`).
    Verify each by `diff` (2 / 3 changed lines, protocol fields byte-identical:
    seed 42, 100 episodes, 24 rounds, `switch_every 4`, single pairing,
    `save_per_round: true`). `squeue` check; one syncing
    `AI_REMOTE_DIR=... scripts/simulate_cluster.sh <control>` (ships
    `artifacts/`), then `--no-sync` for the candidate; wait with one blocking
    `ssh raven 'while squeue -j <ids> -h | grep -q .; do sleep 30; done; sacct
    -j <ids> --format=JobID,State,ExitCode -n'`. From each job log: the
    PROVENANCE line as in step 10; remote sha256 of all three slot artifacts
    (switch candidate equals step 10's). **Activation check:** the candidate
    `per_round.parquet` must **differ** from the parent's — bit-identity means
    the joint branch was never taken and the run is void. One draw each, seed
    42, no re-runs.

12. **[Opus] Fetch, evaluate, diagnose** — `AI_REMOTE_DIR=... scripts/fetch_cluster.sh plots/simulation/<control dir>`
    and `<candidate dir>`; confirm the control's sha256; locally `python -m
    aimanager evaluate <candidate config>`. Fill the Results row with SC, the
    mean, rows <= 1, and RCD / RCB / SB / SA / RSA / CG / RCA / CA exactly as
    computed. Then run the read-only switch diagnostic — an **uncommitted
    scratchpad script** (the two Note-3 analyses folded into one file,
    importing `convert.load_human` / `load_sim` only; this branch's diff is
    the mechanism and nothing else) — on human, parent and candidate:
    larger-group-size shares and mean (rounds >= 4) and the five anchor-round
    means; full-exodus cell share and movers-in-full-exodus share on complete
    pairs (human 0.1079 / 0.2600); per-decision-round switch rates;
    between-group count correlation and P(one side empties, other holds) per
    decision round; P(post = 8 | pre = L) and P(stay 8). Report the numbers
    in Notes and the PR body as measured values. These decide the guard
    reading of §1, not the gates. Commit sim outputs and evaluation only.

13. **[Orchestrator] Verdict, log, PR, clean-up** — §2 on the single evaluation, no second
    stage: `[SUCCESS]` iff SC < 2.0 **and** mean < 1.4452820567384081;
    otherwise `[FAIL]`. Apply the §1 unsoundness criterion and state its
    outcome in plain words in Notes and the PR body regardless of the title.
    Complete the Results table and Notes (mechanism reading, the guard
    outcomes, the late-anchor shape, the collateral `+`/`-`). `gh pr create
    --base auto/contribution-gmlp-group-copula`, body Hypothesis / Results /
    Collateral (§9.7), noting the diff shows only this experiment's change
    over PR #170. Delete `~/autoresearch/switch-joint-exodus-gmlp` on Raven.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-03 | baseline: parent PR #170 stack (gaussian_mlp_v2 group-copula contr x gnn switch x severity-copula punisher) | SC 2.669271261720684 | 10/21 | 1.3138927788530981 | reference |

## 4. Notes

1. (Fable, opening) Baseline re-verified from the parent's `scores.csv` in
   this worktree: 21 rows, mean 1.3138927788530981 (recomputed
   1.313892778853098, the same double), rows <= 1 = 10, SC 2.669271261720684
   with EMD 0.5192479999999997 over ceiling 0.19452799999999945, gate-2
   ceiling 1.4452820567384081. Parent `per_round.parquet` sha256
   `72599716d3e93f37a604e7b4a893090100e080e5ee5da6fe6d2e7f9928d0d429` (equals
   #170 note 43). Base switch artifact sha256 `184f7f5c…5037` (equals #168
   note 11). Base 5-fold held-out log-loss 0.5163464160282509 (equals #171
   note 10). `graph.py`, `train.py`, `environment.py` are byte-identical to
   `main` and carry no copula surface (`grep -c copula` is 0), as the
   maintainer stated.
2. (Fable, opening) Three things read off #171's code that shape the port and
   are easy to get wrong. (a) The head must be constructed *last* in
   `__init__` so the trunk's initial weights under seed 38381 are exactly the
   base run's — the detach argument (the candidate is "the base trunk plus a
   head") depends on it. (b) `predict_encoded` — the path `eval_model` and the
   confusion matrix use — is left untouched; only `predict_independent`
   dispatches to the joint draw, so training-time metrics stay comparable with
   the base and the head cannot leak into the recorded log-loss. (c) On a
   firing round the legacy per-agent draw is made and then overwritten, one
   wasted categorical, so the non-firing path is the pre-existing expression
   itself rather than a re-derivation — that is what makes the step-11 control
   and the round-11/15/19 RNG-identity tests meaningful.
3. (Fable, research; read-only, `convert.load_human` / `load_sim`, single copy
   per human game, complete pairs where stated) **Where this stack's SC
   deficit sits.** Larger-group size, rounds >= 4 — human 9.6 / 24.4 / 28.0 /
   23.6 / 14.4 %, mean 6.088; parent 19.0 / 33.0 / 25.8 / 16.4 / 5.8 %, mean
   5.570 (#171's parent: 5.676). Anchor-round means human 6.44 / 6.20 / 6.04 /
   5.92 / 5.84 (declining), parent 5.71 / 5.60 / 5.40 / 5.57 / 5.57 (flat):
   gap 0.73 at round 4, 0.27 at round 20. Of the 21 human games that reach
   size 8, 13 do so at the first decision (round 3), 5 at the second, 3
   later; parent 22 of 100 games, 10 / 1 / 3 / 5 / 3. **Gross flow at round 3
   is already right — 3.58 movers per game human, 3.51 parent — the
   arrangement is wrong:** between-group correlation of `(m_0, m_1)` human
   −0.694 vs parent −0.356; "one side empties while the other holds" 0.289 vs
   0.100; P(reach 8 | 4-4) 0.219 vs 0.068. By decision round: correlation
   human −0.694 / −0.515 / −0.357 / −0.337 / −0.185, parent −0.356 / −0.370 /
   −0.108 / −0.345 / −0.172 (pooled −0.3667 vs −0.1987 — the parent already
   carries the share of the dependence that observables explain, cf. #169
   note 8's −0.4676 → −0.3038); full-exodus cell share human 0.211 / 0.109 /
   0.057 / 0.066 / 0.066 (n 76 / 64 / 53 / 61 / 61), parent 0.110 / 0.084 /
   0.091 / 0.067 / 0.062; the directional event human 0.289 / 0.111 / 0.000 /
   0.097 / 0.094, parent 0.100 / 0.010 / 0.050 / 0.070 / 0.030. Overall
   full-exodus share human **0.1079** (315 cells), parent 0.0831; movers inside
   full-exodus cells human **0.2600**, parent 0.1588 (#171's candidate: 0.1424
   / 0.3043 — over human on both). 47 % of human full-exodus cells and 47 % of
   human round-3 movers sit at round 3 (parent 27 % / 25 %). Two rows the
   mechanism does not address and that cap it: P(stay 8) human 0.300 vs parent
   0.120 (n 25), and P(reach 8 | 5) human 0.157 vs parent 0.023. Per-round
   switch rate human 0.442 / 0.299 / 0.245 / 0.241 / 0.251, parent 0.439 /
   0.311 / 0.275 / 0.251 / 0.228 (SB's raw material).
4. (Fable, research) **A Markov oracle on the larger-group size, for the
   pre-registration.** Transition kernels P(L_{r+1} | L_r, decision round)
   estimated from the human games (n = 50 per round) and from the parent's
   (n = 100), sparse cells pooled over rounds; 200,000 replicates from L = 4.
   Parent kernel throughout reproduces the parent (mean 5.568, EMD 0.520,
   SC ~2.673 vs the real 2.669), so the approximation is faithful. **Human
   kernel throughout — the ceiling for a head fired on every decision round
   — gives anchors 6.44 / 6.20 / 6.04 / 5.92 / 5.84, mean 6.090, EMD 0.002**,
   the in-sample joint oracle #169 note 9 reported as 6.1078. It is in-sample
   (50 games, sparse cells at sizes 7-8) and cannot see the head's covariate
   shift in a closed loop that visits more lopsided states than the human
   games did — which is how #171 realised 6.286 above its own 6.1078 ceiling.
   For the record: oracles for restricted firing schedules were also
   computed during the opening and were the basis of a calibration proposal;
   the maintainer ruled against any such parameter (Note 5), so they are
   not part of this experiment and are not restated here.
5. (Fable, opening; maintainer ruling relayed by the orchestrator,
   2026-09-03) **The head goes in exactly as #171 built it.** The opening
   draft of this declaration proposed a firing-schedule calibration
   (restricting the joint draw to the decision rounds where Note 3 finds the
   parent short), argued from the per-round deficit profile and #171's late
   upturn. The maintainer's ruling is that the exodus head is implemented on
   this branch unmodified, with no improvements and no selective sampling.
   The parameter does not exist here: the firing condition is
   `(round + 1) % joint_exodus_switch_every == 0` alone, all five decision
   rounds fire, and the Note-3 per-round measurements stand as the deficit
   profile the candidate's own profile is read against. What that ruling
   turns into a prediction: #171's late-anchor upturn is now falsifier (d)
   of §1, and the expected failure mode is the fully-merged overshoot
   recorded in the prediction bullet. (Orchestrator, same date) Checked
   `notes/evaluation_metric_defs.md`: SC is an EMD over pooled (game, round)
   larger-group sizes from round 4 on with no round strata, so nothing about
   this experiment's firing is keyed to a metric boundary; with the knob
   gone the SB-strata question is moot, since all five of SB's decision
   rounds fire.
6. (Fable, opening) Hazards inherited and designed around, none needing a
   feature. `Environment.default_values` comes from the *contribution*
   artifact (#169 note 3b) — irrelevant here, no new state key. Train-time
   features are pandas, sim-time torch (#169 note 3a) — step 7's parity test.
   `remote_test.sh` on an isolated dir lacks `plots/` and `artifacts/` (#170
   note 11g) — the eval-suite and linear-manager failures there are known.
   `run_training.sh` is a `.format()` template — step 1's braces are doubled.
   The trained `.pt` lives in the isolated dir's `artifacts/` after step 10;
   it is fetched and committed *before* the step-11 sync, whose
   `rsync --delete` would otherwise remove it. A dead SSH tunnel makes
   `squeue` return empty (#171 note 26b) — the sync-race check must confirm
   the connection first.
7. (Orchestrator, validation) Plan validated and tagged. §2: target SC
   2.669271261720684 (`2-5`, needs < 2.0) and gate-2 ceiling
   1.4452820567384081 both trace to the parent's own `scores.csv`,
   recomputed here as 21 rows, mean 1.3138927788530981, rows <= 1 10/21.
   §5: no new input feature (`agent_group` and `round_number` are already
   in the base model's `x_encoding`); no seed, episode-count, scoring or
   protocol change; iteration budget 4m17s measured from `sacct` job
   29870374 against a ~10.6-min ceiling; the firing schedule that would
   have been the one tunable is gone, so nothing in the mechanism is
   selected by a score. §8: nothing under `evaluation_suite/`,
   `notes/evaluation_metric_defs.md`, `notes/eval_scoring_schema.md` or
   `experiments/` is written, and no other branch's log file is touched.
   Implementer tags follow §9 Roles: Opus on steps 2-7, 10 and 12 — the
   mechanism itself (masked joint algebra, the 256-subset log-odds
   accumulation where `0 * -inf` is a live hazard, the shared
   `GraphNetwork` surface, the detached objective, the RNG discipline of
   the sampler, the parity invariant, and the guard measurement the
   unsoundness call rests on) — and Sonnet on steps 1, 8, 9 and 11, which
   are mechanical against a fixed spec and verified by `diff`, a template
   render, or a sha. Step 13 is the orchestrator's. Step 4 is the one
   step that touches code shared with another slot: `GraphNetwork` also
   loads this stack's `valid_model`, so its head-off bitwise-identity
   tests are not ceremony and a failure there halts the run.

8. (Orchestrator, step 1 confirmed) Step 1 verified independently of the
   implementer's report: both added lines render correctly through
   `create_script`'s two-step `.format()`, and a brace audit of the whole
   template leaves only the six real placeholders (`log_file`, `job_id`,
   `cores`, `memory`, `experiment_name`, `command`) single-braced. The
   comment's appeal to `--chdir=.` is accurate — it is on line 3 of the
   template — and under the file's `set -e` a failed provenance import
   stops the job rather than letting it train on shared-tree code, which
   is the behaviour we want.
9. (Orchestrator, environment correction to step 9) `uv sync` cannot run
   in this worktree: it resolves `--find-links https://data.pyg.org/whl/
   torch-1.11.0+cu113.html` and the sandbox has no DNS for that host, so
   it creates an empty `.venv` and fails. The empty venv was removed —
   left in place it would be picked up by anything defaulting to
   `.venv`. Local tests instead run on the **main checkout's** venv with
   the worktree's source on the path:
   `PYTHONPATH=$PWD/src /Users/ertuerkan/Desktop/algorithmic-institutions/.venv/bin/python -m pytest ...`.
   Verified: `aimanager` resolves to the worktree's
   `src/aimanager/__init__.py`, and `test_eval_metrics.py` passes 44/44.
   That venv is torch 1.11.0 / pytest 8.4.2 / numpy 1.26.4 / pandas
   2.3.3 with **no** `torch_scatter` or `torch_geometric` — exactly the
   macOS setup CLAUDE.md describes and the condition the plan's stand-in
   discipline expects, so the stand-ins install locally and must report
   *not installed* on Raven.

10. (Orchestrator, step 2 confirmed) Verified independently: 31 tests pass;
    `-inf` is applied before a single `log_softmax` over the flattened grid
    and `(0, 0)` is always valid, so a fully masked grid — which would be
    NaN — is unreachable; `pool_by_group` takes membership from
    `agent_group` alone, keeps group as the fastest axis so the reshape is
    in label order, and divides by `counts.clamp(min=1.0)` so an emptied
    group pools to zero rather than NaN; the detach is on the pooled
    embedding only, with `k` and the round being integer features that
    never carried gradient. **The head has 1,131 parameters at
    `embed_size = hidden_size = 10`, matching the count #171 recorded** —
    an architecture-level confirmation that the reimplementation is
    faithful, not merely plausible. The implementer added four invariances
    beyond the plan's list, of which the useful one is that logits off the
    support cannot move the surviving cells: that is the only assertion
    that distinguishes `-inf`-before-softmax from a large finite penalty.
    It also dropped three borrowed figures from #171's comments (a
    single-fold log-loss and a nats estimate measured on that branch)
    rather than restate another run's numbers as if measured here, which
    is the right call.

11. (Orchestrator, step 3 confirmed) 21 tests pass. The `0 * -inf` hazard
    is closed structurally rather than by tolerance: real slots get a
    two-sided `clamp(1e-12, 1 - 1e-12)` before `log(p) - log1p(-p)`, and
    padded slots are overwritten with `p = 0.5`, i.e. a logit of exactly
    0, so the subset-weight matmul `logit @ bits.T` can never form
    `0 * ±inf`; subsets touching padding are discarded by `valid`
    regardless. Probed independently with an adversarial batch — exact
    `0.0` and `1.0` on *real* slots, a `k == 0` row, `m == k`, `m == 0`,
    and NaN-poisoned padding — no NaN anywhere, every row's `exp` sums to
    1.0, exactly `m` selected, no padded slot ever chosen.
    `sample_conditional_bernoulli` consumes the global RNG exactly once
    (verified by state replay against a single `th.multinomial` on a
    table built without consuming RNG, with a two-draw sanity check);
    that predictable draw count is what step 6's head-off bitwise
    identity depends on. Marginal recovery over 20,000 draws with `m`
    from the Poisson-binomial DP: max absolute error 0.0045, worst
    deviation 1.30 SE, mean drawn `m` 3.586 against `sum(p) = 3.58` —
    the check that actually validates the mechanism, since it confirms
    per-agent propensities survive the conditioning.
12. (Orchestrator, for step 4) Two surface facts from step 3 that the
    gate's author needs: `conditional_bernoulli_log_prob` upcasts `p` to
    float64 internally regardless of input dtype (the reference's choice,
    kept), though the *returned selection* is `bool`, which is what the
    gate consumes; and the assert message substrings `"exceed"`,
    `"non-negative"`, `"max_group_size"` and `"rows of p"` are now part
    of the tested surface, so they must not be reworded casually.

13. (Orchestrator, step 4 confirmed) 25 tests in the new suite, 77 across
    `tests/switch`, 357 across `tests/`; `graph.py` shows 91 insertions
    and one deletion (the `forward` signature) over nine hunks, with no
    incidental reformatting, no `copula` and no `last_round` anywhere in
    `src/`, `tests/` or `configs/`. The two invariances that protect the
    other slots both hold: head-off `predict_independent` is `th.equal`
    in predictions, probabilities **and** the full post-call global RNG
    state against the pre-change class built out of git (`ad3bc9c`), and
    head-on shares the head-off trunk `state_dict` tensor for tensor
    with bit-identical per-agent logits, which is what building the head
    **last** buys. Confirmed by mutation rather than by a green run:
    injecting a single `th.rand(1)` into `forward` fails the identity
    test on the predictions themselves, and the file restores clean. The
    implementer also verified `embed_size` against the **measured** RNN
    output width by hooking `rnn_n`'s forward, not by reading the
    assignment chain — and noted that post-node-model and post-RNN
    widths coincide here (both `hidden_size`), so "post-RNN" is not what
    distinguishes it.
14. (Orchestrator, ruling on step 4's git-dependent tests) The
    pre-change-class tests `skipif` out where `.git` is absent, which is
    every Raven run, since `remote_test.sh` rsyncs with `--exclude
    '.git/'`. Accepted as-is rather than committing a copy of the old
    module as a fixture. Reason: with the head off, none of the new code
    executes except the changed `forward` signature, so head-off
    identity is a PyG-independent property and a local check settles it;
    what actually needs real `torch_scatter` is the head-**on** path,
    whose assertions do run on Raven at step 9. A duplicated 500-line
    fixture would rot and buy nothing.
15. (Orchestrator, plan correction from step 4) `sample_conditional_bernoulli`
    is imported in step 6, not step 4: nothing at step 4 calls it and
    flake8's F401 is active, so an early import would fail lint and a
    `noqa` would be papering over it. Step 4's inherited test item about
    head-on non-decision-round sampling is not yet meaningful either —
    `predict_independent` is untouched here, so with the head on
    sampling is the legacy path on *every* round; the honest version of
    that test asserts step 4 changes no sampling behaviour at all, and
    the round-dependent claim belongs to step 6.

16. (Orchestrator, step 5 confirmed) 99 tests across `tests/switch`, 379
    across `tests/`; `train.py` +200/-11, the deletions being exactly the
    inlined per-agent expression that `compute_batch_loss` now holds, with
    no incidental reformatting and `src/aimanager/generic/` untouched. The
    two properties that decide whether this experiment can attribute
    anything both hold bitwise: head-off `compute_batch_loss` is
    `th.equal` to a verbatim copy of the pre-change expression in value
    **and** in every parameter gradient after `backward()`, with and
    without entropy regularisation; and with the head on, the trunk
    gradient is `th.equal` to the per-agent-only gradient, the joint term
    alone leaves every trunk gradient `None` or exactly zero, and the cut
    is `th.equal` to the detached pooled embedding. Confirmed by my own
    mutation, not the implementer's: replacing `pooled.detach()` with
    `pooled * 1.0` turns **all four** detach tests red, and restoring
    gives 99/99 with `generic/` clean. The loss selects usable cells by
    boolean indexing and returns `nll.sum()` on an empty selection — a
    differentiable zero — so a `-inf` grid entry can never reach a
    multiply.
17. (Orchestrator, for step 6 — a real coupling, not a nit) `k == k_head`
    inside `joint_exodus_loss` is an identity **only because the call
    site hands the head the same mask**: the loss re-derives `k` from
    `batch_data["mask"]` while the head derived it from `data.get("mask")`
    inside `forward`. Today they are the same tensor. If step 6 passes a
    `decider_mask` other than `data["mask"]`, that assert breaks on any
    training-shaped call. Step 6 must either pass the same mask or leave
    `decider_mask` unset in the training path. The assert bites — there is
    a test proving it — so this would fail loudly rather than silently,
    which is the outcome we want.
18. (Orchestrator, two step-5 spec notes recorded) `joint_exodus_loss`
    carries a `drop_incomplete_pairs` keyword the plan's prose does not
    name; it is the reference's, nothing in the training path passes it,
    so `DROP_INCOMPLETE_PAIRS` stays the only live setting, and the
    keyword exists so the `False` arm can show the drop is doing work.
    And `STAND_INS` cannot honestly report per-file in a shared pytest
    process — whichever suite imports first installs, and the siblings
    then report `[]` because `sys.modules` already carries them, which
    reads exactly like a Raven run. The variable is still sound as "what
    this file installed"; a genuine per-session report would need one
    shared `tests/switch/conftest.py`, which is a step 9 decision.

19. (Orchestrator, step 6 confirmed) 42 new tests, 141 across
    `tests/switch`, 421 across `tests/`; `graph.py` +122/-1, the deletion
    being the `predict_encoded` dispatch line now inside the `else`, with
    steps 2/3/5's files untouched. RNG discipline **measured directly**,
    not inferred: every one of the five decision rounds (3, 7, 11, 15,
    19) costs exactly three `th.multinomial` draws — `(16, 2)` the
    verbatim legacy per-agent decode, `(2, 81)` the pair draw over the
    flattened grid per episode, `(4, 256)` one batched selection over
    `n_groups x n_batch` rows — and every non-firing round costs exactly
    one, leaving the full global RNG state byte-identical to the
    pre-change expression. That all five fire confirms the maintainer's
    ruling is in force in the code, not merely absent from a config.
    Both hazards mutation-tested by me independently, reproducing the
    implementer's counts exactly: transposing the group axis
    (`m.flip(-1)`) turns 11 tests red, including the mirrored-label row
    where positions are identical and only labels move, which is the
    case a totals-only assertion cannot see; replacing the verbatim
    legacy draw with an `argmax` turns 18 red. Restores to 141 green.
    `environment.py` and `simulate.py` confirmed unchanged and
    unnecessary to change, and `state` carries no `"mask"` key, so
    `decider_mask` stays unset in the simulation path and note 17's
    coupling is never exercised.
20. (Orchestrator, correcting my own error) I told the step 4 and step 6
    implementers that PR #171 carried a `joint_exodus_last_round`
    firing-schedule parameter that this branch was deliberately
    dropping. **That is wrong, and step 6's implementer caught it.**
    `last_round` has zero hits in the #171 reference sources, in #171's
    log, and on `origin/auto/switch-joint-exodus` itself. #171 always
    fired on every decision round; `last_round` was *this* experiment's
    proposed calibration in the original declaration, which the
    maintainer then ruled out (Note 5). The effect on the code is nil —
    no such parameter exists anywhere in `src/`, `tests/` or `configs/`,
    which is what both the ruling and #171's own behaviour require — but
    the provenance in those two briefs was mine, not #171's, and the
    record should not carry it.
21. (Orchestrator, for step 7) `sample=False` provably makes zero
    `th.multinomial` calls and returns the per-agent argmax, so
    `eval_model` and the confusion matrix cannot reach the joint path
    even with the head on — which is what keeps step 10's held-out
    log-loss comparable with the base artifact's 0.5163464160282509.
    Step 6's test list was all single-round; the implementer added a
    four-round window asserting the gate is evaluated per round rather
    than per call, which is the only test that catches a
    once-per-forward gate and is among those mutation 2 turns red.

