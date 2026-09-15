# PNA-style multi-aggregator for the GNN contributor: mean, max, min, std at the peer-message aggregation

Autoresearch experiment on the contribution slot, stacked on the
maintainer-designated frontier PR #171 (`auto/switch-joint-exodus`). Branch
`auto/contribution-pna-aggregation`, worktree `.claude/worktrees/pna-aggregation`,
remote dir `~/autoresearch/pna-aggregation`. The PR opens with
`--base auto/switch-joint-exodus`.

## 1. Declaration

- **Slot:** contribution.
- **Base model:** the PR #165 artifact
  `artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`
  — the M0 trunk (`configs/training/artificial_humans/contribution/group_switching_contribution_50ep.yml`:
  `add_global_model: False`, `add_edge_model: True`, `hidden_size: 20`,
  `x_encoding = [prev_contribution numeric, prev_punishment numeric, agent_group
  onehot]`, no `edge_encoding`, 575 epochs, seed 38381, 5-fold CV on the
  flip-doubled data) with `copula_rho = 0.06958238086256316`, `copula_phi = 1.0`,
  `copula_switch_every = 1` stamped on top.
- **Parent PR (§9):** #171, branch `auto/switch-joint-exodus`, itself on PR #165.
  Evaluation stack: the parent's own config
  `configs/simulation/manager_testing/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml`
  — gnn-copula contribution x joint-exodus gnn switch x severity-copula
  `lin_multinomial` punisher, single `lin_multinomial_copula_self` pairing, seed 42,
  100 episodes, 24 rounds, `save_per_round: true`. Only the `contribution_model`
  path changes; switch and punisher artifacts stay byte-identical to the parent's.
- **Baseline** (the parent's confirmed run,
  `plots/simulation/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch/evaluation/scores.csv`):
  rows <= 1 **11/21**, mean **1.3040409569053069**; gate-2 ceiling (mean + 10%,
  §2 as amended by `b174f90`) **1.4344450525958377**. Rows >= 2:
  CG 4.267640451429015 (band 2-5; numerator 0.11287345494901657, noise ceiling
  0.026448679600274437), RCD 2.764919035295771 (2-5), RCB 2.0242478714062093 (2-5).
  Next: RCC 1.5810557625733717, RCA 1.5746350718021775, RSA 1.3222991041595975,
  RPA 1.247405946294638, SC 1.147392663266986, CE 1.105386005744275,
  SB 1.06526788518747. Marginal C block: CA 0.821396113408324, CB 0.8705450772997657,
  CC 0.9207602769543581, CD 0.8018706784949698, CF 0.9329186421936507;
  SA 0.8396005958717294.
- **Target rows.** **CG** (primary): 4.267640451429015, band 2-5 -> needs < 2, i.e.
  the spread-ratio gap `|ratio_sim - ratio_human|` below
  **0.052897359200548874** (from 0.11287345494901657 — the gap must more than
  halve). **RCA** (secondary, with its own mechanism claim below):
  1.5746350718021775, band 1-2 -> needs <= 1, i.e. the round-type-averaged EMD
  below **0.2799930910702011** (from 0.4408869410614397). Gate 2: the 21-row mean
  must stay <= 1.4344450525958377.
- **Not declared, deliberately.** RCD (2.764919035295771) and RCB
  (2.0242478714062093) are >= 2 but this mechanism makes no claim on the switching
  pull or the punishment response; declaring them would be shopping (PR #173's
  ruling: pre-declaration binds in both directions). If either band-upgrades it is
  collateral.
- **Watch items:** the marginal C block (CA/CB/CC/CD/CF — the tax that killed #154,
  #156, #158 and arrived in full in #173), RCC (a sharper reaction to peers can move
  the ceiling contrast), SA/SC (PR #173's propagation channel: a contribution change
  reaches the switch trunk through per-capita common good), and RCD/RCB as above.

### Hypothesis

**Behavioral claim.** Contributors read their group's *dispersion and extremes*,
not only its average: a divided group destabilises its members, a united group
holds them. Measured on the single-copy human data at planning (50 games; 7,543
player-rounds with a valid own and previous contribution, at least two same-group
peers observed in the previous round, and no group change), controlling for own
previous contribution and the own-group peer mean: the peer **std raises the
probability of a large move in either direction** — logit P(dc <= -5): std
+0.124 (p = 0.0045); logit P(dc >= +5): std +0.119 (p = 0.0096); the three
extra statistics (min, max, std) add LR chi^2(3) ~ 26 and ~ 21 nats to the two
tail models. In the high-contributor tail (prev >= 15, peer mean <= 5) a peer at
0 raises P(drop >= 5) from 0.349 to 0.500 and the mean drop from -4.14 to -6.42
(n = 43 / 60). On the *level*, by contrast, the peer mean does almost all the
work (coef 0.27-0.38; min/max/std add R^2 0.001): humans track the mean and are
*destabilised by disagreement*. Consensus groups lock in where they are;
divided groups swing. Across games that is heterogeneous group-level
persistence — group means that stay apart instead of regressing to a common
attractor — which is what CG scores (SD of group means over SD of individuals;
PR #177 found the same row crossing for persistence, not corner-locking). The
same dispersion-conditioned large-move mass is what RCA's per-round-type EMD of
dc distributions sees, hence the secondary target.

**Mechanism.** The trunk aggregates its 7 incoming peer messages with a uniform
`scatter_mean` (`NodeModel.forward`, `src/aimanager/generic/graph.py`), the one
live aggregation site in this artifact (`GlobalModel` is off and `op2` is fed an
empty `edge_attr` — verified by reading). A mean over a multiset provably cannot
carry its extremes or its spread (Xu et al. 2019, GIN; Corso et al. 2020, PNA):
whatever the edge MLP emits, the receiving node sees one location statistic.
The change replaces that mean with the PNA multi-aggregator
**[mean, max, min, std]** over the same 7 messages, concatenated into the node
MLP's input (4 x 20 = 80 features instead of 20; the node MLP grows from 480 to
1,680 parameters on a ~3.4k-parameter model). Each aggregator has a behavioral
reading and a data footprint above: mean = the level humans track (0.27-0.38),
std = disagreement (both tails, p < 0.01), min = the free-rider trigger
(high-contributor collapse), max = the discounted outlier (significant for
large *rises* with a *negative* sign, -0.096, p = 0.0007: given the mean, a
lone high peer means the rest sit lower, and people follow the majority, not
the outlier — a shape a mean cannot express either). Degree scalers, the other half of
PNA, are **dropped**: this is a fixed fully-connected 8-node room, in-degree is
the constant 7, so a log-degree scaler is a constant multiplier and buys nothing
(§5: ties go to the simpler model).

**Whose spread — decided, not hedged.** Aggregation stays over the mixed 7-peer
room, with group selectivity delegated to the edge MLP rather than built into the
scatter. Reasons: (i) constant degree, no empty partitions (a singleton's own
group has no peers — 2.3% of human rows, far more in the simulation), no
per-round masked scatter, no doubled width; the legacy path is a one-line
fallthrough. (ii) The `agent_group` one-hot of both endpoints is already in the
edge MLP's input, and with a `same_group` edge bit (arm B below) a single tanh
unit with a large weight on that bit saturates other-group messages to a
constant, so the room max/min of that unit *is* the own-group max/min: the
aggregators supply the statistics, the edge model chooses the neighbourhood.
(iii) Explicit own-group conditioning has been tried and bought nothing on fit —
PR #116's `own_group` mean feature 1.991426 and `own_group + same_group`
1.990806 against M0's 1.9897416823554699 — so a per-partition design would spend
its complexity on the half of the information the trunk already has. Rejected:
per-partition (same-group / other-group) aggregation, as the larger change with
its own failure modes and no evidence the mean half needs it.

**Why this is not PR #153.** Attention is a convex re-weighting of the messages
— still a single location statistic between the min and the max, never both,
never a dispersion. #153-sg learned exactly the group-selective weighting its
hypothesis predicted (2.9x same-group mass) and moved CG 15%, which its
post-mortem read as "the deficit is not missing peer-conditioning structure".
This experiment reads the same result differently: the *weighting* was never
the bottleneck, the *statistic* was — and the data above locates the missing
information in the tails, where a re-weighted mean is blind by construction.
Arm B gives the multi-aggregator the identical information set #153-sg had, so
the comparison is clean: same inputs, aggregator instead of attention. The stack
has also changed under it: the trunk now carries the episode-persistent group
latent (CG 9.81 -> 4.27), and a dispersion-conditioned response can amplify a
shared shock through the closed loop where a mean-conditioned one damps it.

**Honest prior and pre-registered risks.** (1) The human level response is
mean-dominated; the tail effect is real but modest, so the expected CV log-loss
gain is small (order 0.005 nats) and CG must be moved by closed-loop dynamics,
not by teacher-forced fit — PR #175 and #177 both record teacher-forced readings
pointing opposite to the simulated ones. (2) The copula is recalibrated on the
retrained trunk (the #166 lesson). If the trunk absorbs within-group co-movement
through dispersion-conditioning, the residual rho **falls** below
0.06958238086256316 — expected, not a data-path bug, and it means the two
mechanisms partly substitute rather than add. rho and its CI are recorded either
way. (3) A sharper peer response is the §6 anti-correlation trade: the marginal C
block and RCC are where it would be paid.

### Planned change

1. `NodeModel` gains an optional `aggregators` list; `forward` concatenates one
   `torch_scatter` reduction per entry over the incoming edges; absent or `None`
   (every existing artifact, whose pickled `NodeModel` has no such attribute)
   takes the literal legacy `scatter_mean` line — bit-identical values and RNG.
2. `GraphNetwork` accepts, validates, persists and reports `aggregators`; only
   `op1`'s `NodeModel` receives it.
3. Two training arms of the M0 config with `aggregators: [mean, max, min, std]`:
   **A** — no edge features (M0's information set); **B** — `edge_encoding:
   [same_group]` (#113's existing `SameGroupEdgeEncoder`, #153-sg's information
   set). Selection by 5-fold CV held-out log-loss at epoch 574 under a rule fixed
   below, before any training runs; **one** arm is calibrated and simulated.
4. PR #165's calibrate -> stamp recipe re-run on the selected trunk
   (`contribution_copula_rho.py` unchanged; stop-gate restated below), the
   parent's sim config with three edits, a bit-identical control of the parent's
   own run, one candidate simulation, one evaluation, verdict per §2.

**Iteration budget (§5).** Recent plain M0 trainings on Raven: 00:08:09, 00:08:52,
00:07:57, 00:08:59 — base ~ 8-9 min, ceiling ~ 25 min per training. The change
adds three scatter reductions per forward and a 1,200-parameter-wider node MLP:
expected ~ 1.1-1.3x, i.e. ~ 10-12 min per arm, two arms -> two GPU trainings
(submitted together). Plus one CPU calibration (~ 12 min, jobs 29666293 /
29892494 precedent), one CPU stamp (~ 1 min), two GPU simulations (control +
candidate, ~ 2.5 min each). Total: 4 GPU jobs, 2 CPU jobs, ~ 45 min of compute
before queueing. Legal under §5 (architecture change with a behavioral sentence
per statistic; variant selection by a pre-declared held-out criterion; no
metric-engineered feature; GNN on the doubled data, copula calibration on the
40-episode single-copy train split, teacher-forced). Nothing on the §8 frozen
surface is touched.

## 2. Plan

Every remote call from this worktree sets
`AI_REMOTE_DIR='~/autoresearch/pna-aggregation'` and checks `squeue` for PENDING
jobs first — after confirming the SSH ControlMaster is live (`ssh -O check
raven`; a dead tunnel makes `squeue` return empty, PR #171 note 26b). No compute
on login nodes: calibration, stamping, training and simulation all go through
`sbatch`. **`rsync --delete` hazard (PR #173 note 11 / #176 plan header):** the
launchers sync `artifacts/` (excluding only `artifacts/manager/`) with
`--delete`, so every artifact a remote step produces is fetched into this
worktree and committed *before* the next launcher call; single files (slurm
wrappers) go by `scp`. Slugs: artifacts under
`artifacts/artificial_humans/group_switching_contribution_50ep_pna_aggregation[_sg]/`,
the stamped model under `..._pna_aggregation_herding_copula/`, sim output
`plots/simulation/23_2g8a_contr_pna_aggregation_self_gnnpnacopar1_contr_gnn_switch`
(slug before `_self_`, so `evaluation_sweep.py`'s `DIR_PATTERN` parses
`contr = gnnpnacopar1`, `switch = gnn`).

- [x] 1. *(Opus)* **Multi-aggregator in `NodeModel`** — `src/aimanager/generic/graph.py`,
      class `NodeModel` (existing). Constructor gains `aggregators=None`; store
      it and size the node MLP as `in_features = x_features +
      n_aggr * edge_features + u_features` with `n_aggr = 1 if aggregators is
      None else len(aggregators)`. Add a module-level table
      `AGGREGATORS = {"mean": scatter_mean, "max": scatter_max[0], "min":
      scatter_min[0], "std": scatter_std(unbiased=False)}` (all four exist in
      `torch_scatter` 2.0.9; `scatter_max`/`scatter_min` return `(out, argmax)`,
      take `[0]`; population std so a degree-1 node yields 0, not NaN). In
      `forward`: `aggregators = getattr(self, "aggregators", None)` — the
      `getattr` is load-bearing, because every existing artifact unpickles a
      `NodeModel` without the attribute — and if `None` run the existing
      `scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))` line verbatim;
      otherwise `th.cat([AGGREGATORS[a](edge_attr, col, dim=0, dim_size=x.size(0))
      for a in aggregators], dim=-1)` in config order. `GlobalModel`, `EdgeModel`
      and `op2` are untouched.

- [x] 2. *(Opus)* **Plumbing in `GraphNetwork`** — `src/aimanager/generic/graph.py`,
      `GraphNetwork.__init__` (existing): new keyword `aggregators=None`;
      validate (a non-empty list of distinct keys of `AGGREGATORS`, or `None`;
      `aggregators is None or add_edge_model` — the aggregators reduce the edge
      model's messages, there is nothing to reduce without one); pass it to
      `op1`'s `NodeModel` only (the `op1 is None` branch); when `op1` is given
      (the load path) assert `getattr(op1.node_model, "aggregators", None) ==
      aggregators` so an artifact cannot claim one aggregation and run another;
      `self.aggregators = aggregators`. `save()` (existing): append
      `"aggregators"` to `to_save` — a legacy artifact lacks the key and loads
      with the default, exactly as `copula_*` and `joint_exodus` do. Degree
      scalers are not implemented (constant in-degree; declaration).

- [x] 3. *(Opus)* **Tests and lint** — new `src/aimanager/tests/test_pna_aggregation.py`
      with the PyG stand-in preamble of
      `src/aimanager/tests/test_joint_exodus_train_sim_parity.py` so it runs
      locally *and* on Raven against real PyG (each test reports which). Cases:
      (a) each aggregator against a brute-force per-node loop over the incoming
      edges of a random 8-node fully-connected batch (mean/max/min/std, several
      rounds, several batch elements); (b) legacy bit-identity — a `NodeModel`
      built without `aggregators`, and one with the attribute deleted to mimic an
      unpickled legacy module, both return exactly `scatter_mean(...)` and build
      the same `in_features`; a `GraphNetwork(aggregators=None)` under a fixed
      seed has a `state_dict` equal to one built by the same call before the
      change (compare against the literal M0-shaped constructor in
      `test_contribution_copula_graph.py::make_model`); (c) `aggregators=
      ["mean"]` equals the legacy output numerically; (d) save -> load round-trips
      `aggregators` and the loaded model's `forward` equals the saved one's on
      the same input; the `op1`/`aggregators` disagreement assert fires;
      (e) `["mean", "max", "min", "std"]` changes the node MLP's input width to
      `4 + 80 = 84` and the forward runs on M0-shaped data with `edge_encoding=[]` and
      with `[same_group]`. **Stand-ins:** `graph.py` now imports `scatter_max`,
      `scatter_min`, `scatter_std` at module top, so every stand-in installer
      that sets `scatter.scatter_mean` must also install the three new functions
      (loop-based, dim=0 only): `src/aimanager/tests/test_joint_exodus_train_sim_parity.py`,
      `tests/switch/test_joint_exodus_detach.py`, `tests/switch/test_joint_exodus_graph.py`,
      `tests/switch/test_joint_exodus_sampling.py`, `tests/switch/test_joint_exodus_loss.py`
      (grep for `scatter.scatter_mean` to catch any other). Local: the new suite,
      `tests/` (copula, switch, baselines), `src/aimanager/tests/test_eval_*.py`.
      Raven: after step 5, login-node pytest *inside the isolated dir* for the
      new suite plus `test_encoder`, `test_edge_encoder`, `test_environment`,
      `test_linear_manager`, `test_contribution_copula_graph`,
      `test_switch_copula_graph`, `test_joint_exodus_train_sim_parity` (eval-suite
      fixture failures there are expected per #165 note 4). One batched `black`
      + `flake8` (88, `E203,W503`) pass over the touched `src/` files before
      staging.

- [ ] 4. *(Sonnet)* **Training configs, two arms** — new
      `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_pna_aggregation.yml`
      (arm A): verbatim copy of `group_switching_contribution_50ep.yml` (575
      epochs, batch 4, lr 3e-4, hidden 20, 5-fold, seed 38381, same
      `shuffle_features`) with exactly three edits — `model_args.aggregators:
      [mean, max, min, std]`, `output_dir:
      artifacts/artificial_humans/group_switching_contribution_50ep_pna_aggregation`,
      a `description` naming this experiment. New
      `..._pna_aggregation_sg.yml` (arm B): arm A plus
      `model_args.edge_encoding: [{name: same_group, etype: bool}]` (the syntax
      of the existing `group_switching_contribution_50ep_same_group.yml`) and
      `output_dir: ..._pna_aggregation_sg`. `labels` unchanged in both, so the
      artifact filename stays
      `architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` for the copula
      recipe.

- [ ] 5. *(Sonnet)* **Isolated remote dir** — `ssh -O check raven`, then `squeue -u certuer`
      (PENDING check), then `AI_REMOTE_DIR='~/autoresearch/pna-aggregation'
      scripts/train_cluster.sh --sync-only ah <arm-A config>`: creates the dir,
      ships `src/`, `scripts/`, `configs/`, the human CSVs and the AH artifacts
      (the M0 trunk, the parent's `switch_joint_exodus`, `raven_script_22` and
      `artifacts/baselines/` are all needed later by the stack). Verify remotely
      that no shipped `.csv`/`.pt`/`.joblib` is an LFS pointer (`grep -rl
      'git-lfs.github.com/spec'`), `md5sum` `graph.py` local vs remote, and that
      `PYTHONPATH=~/autoresearch/pna-aggregation/src python -c 'import
      aimanager; print(aimanager.__file__)'` resolves inside the isolated dir
      (#171 notes 7 / 24). Then run the Raven half of step 3.

- [ ] 6. *(Sonnet)* **Train both arms on Raven** — `AI_REMOTE_DIR=... scripts/train_cluster.sh
      --no-sync ah <arm-A config>` and the same for arm B (squeue check between;
      `--no-sync` after step 5's sync so nothing is deleted). Record per arm:
      SLURM job id, elapsed against the ~25 min ceiling, the in-job provenance
      (a traceback or log line naming `/autoresearch/pna-aggregation/src`; the
      string `algorithmic-institutions/src` must not appear), the artifact
      sha256, and the loaded `aggregators` field. **Immediately** fetch both
      artifact dirs with `AI_REMOTE_DIR=... scripts/fetch_cluster.sh
      artifacts/artificial_humans/group_switching_contribution_50ep_pna_aggregation`
      (and `_sg`) and commit them (LFS `.pt`, plus the `metrics/` and
      `confusion_matrix/` parquets) before any further launcher call.

- [ ] 7. *(Sonnet)* **Held-out fit check and arm selection (pre-declared rule)** — from each
      arm's `metrics/architecture_node+edge+rnn__dataset_50ep__epochs_575.parquet`
      (rows `name == log_loss`, `set == test`, `shuffle_feature` and
      `leave_one_in_shuffle_feature` null, `epoch == 574`): the five fold values
      and their mean, against M0's **1.9897416823554699** (folds, in `cv_split`
      order, 2.0482240744250215 / 2.0359881250603187 / 1.9632594823807232 /
      1.9994286152042022 / 1.9018081147070847 — the fold assignment is a function
      of the seed and the data, so the comparison is paired). **Selection:** arm A
      unless arm B's mean is lower by more than 0.005 *and* lower on at least 4 of
      the 5 folds; ties and everything else go to arm A (§5, simpler). **Fit
      stop-gate:** if the selected arm's mean is worse than M0's by more than
      0.01, stop and report to the orchestrator before anything else runs — extra
      expressivity on the same inputs that fits worse is not being used, and the
      hypothesis needs it to be. Report also the `shuffle_feature` deltas
      (`agent_group`, `prev_contribution`, `prev_punishment`) for both arms
      against M0's (1.990605 / 3.831123 / 2.015537 at epoch 574). Log everything
      in Notes, unrounded.

- [ ] 8. *(Opus)* **Pre-sim mechanism diagnostic, report-only** — a scratch teacher-forced
      script (not committed; PyG stand-ins locally as #176 did, or a CPU `sbatch`
      in the isolated dir — never login-node compute) over the 40-episode
      single-copy train split for M0 and the selected arm: (i) the expected
      contribution regressed on own prev, own-group peer mean / min / max / std
      (human: 0.703 / 0.381 / -0.068 / -0.049 / -0.075); (ii) logit of the
      predicted P(dc <= -5) and P(dc >= +5) on the same regressors (human std
      coefficients +0.124 / +0.119); (iii) predicted P(|dc| >= 5) for
      high-contributors (prev >= 15) facing a peer at 0 vs not, peer mean <= 5
      (human 0.500 vs 0.349). This says whether the aggregators are *used* and
      in the human direction; **whatever it says, the simulation runs** and the
      verdict comes only from step 14 (§2, §6). Numbers into Notes.

- [ ] 9. *(Sonnet)* **Port the stamper precondition fix** — `git checkout
      origin/auto/contribution-arrival-tenure --
      scripts/artificial_humans/make_contribution_copula_artifact.py` (PR #173
      step 7b as ported by #176 step 7: `NEUTRAL_FIELDS` and
      `assert_only_copula_fields_changed`, 32 lines of assertion logic, identical
      on both donor branches). Any trunk trained on this branch carries the three
      copula keys at their neutral defaults, which the parent's stamper refuses
      with `assert k not in base`. Shared code, admitted as a precondition on the
      same §4 ruling #173 and #176 recorded; it cannot alter a stamped value.

- [ ] 10. *(Opus)* **Calibrate the copula on the selected trunk, with the stop-gate
      restated** — new `scripts/artificial_humans/calibrate_copula_pna_aggregation.slurm`,
      a copy of `calibrate_copula.slurm` with `BASE` = the selected step-6
      artifact and `PARAMS` =
      `artifacts/artificial_humans/group_switching_contribution_50ep_pna_aggregation_herding_copula/calibration/copula_params.json`;
      `contribution_copula_rho.py` unchanged (`--roundtrip --preflight
      --write-params`). `scp` the wrapper, `sbatch` from the isolated dir (~ 12
      min CPU). Record rho, SE, the 200-episode-cluster bootstrap CI, the
      round-trip gate, phi_hat and its CI, the preflight ratio triple, the
      round-thirds rho, the holdout rho. **Acceptance is restated, not copied
      from #165 step 8:** rho will *not* reproduce 0.06958238086256316 — the
      trunk changed, so the residual dependence changed; the expected direction
      is downward (declaration risk 2), and the number is recorded whatever it
      is. **STOP-GATE:** rho CI includes 0 -> no artifact, no simulation,
      calibration-only `[FAIL]` PR. **phi rule, fixed here before the number is
      seen:** phi_hat >= 1 or CI spanning 1 -> `phi_final = 1.0` with the #165
      boundary ruling as `phi_final_reason` (the estimate stays in the JSON; the
      job's exit `1:0` on `STOP-ESCALATE` is by design, not a crash); phi_hat in
      (0, 1) with CI excluding 1 -> stamp the estimate as-is (no `phi_final`);
      phi_hat <= 0 or CI including 0 -> stop-gate, as #165's step 8. Fetch and
      commit the params JSON and the job log (`git add -f`, the #165 precedent).

- [ ] 11. *(Sonnet)* **Stamp** — new `scripts/artificial_humans/stamp_copula_pna_aggregation.slurm`
      (copy of `stamp_copula.slurm`) invoking the step-9 stamper with `--params`
      (step 10), `--base` (the selected step-6 trunk) and `--out
      artifacts/artificial_humans/group_switching_contribution_50ep_pna_aggregation_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`.
      Verify from the job log: the three fields round-trip (`copula_rho` = step-10
      rho, `copula_phi` per the step-10 rule, `copula_switch_every` = 1); every
      weight tensor `torch.equal` to the trunk; the honesty check reports the
      7,457 teacher-forced train-split rows bit-identical; the loaded model
      reports `aggregators == ["mean", "max", "min", "std"]` and, for arm B,
      `edge_encoding == [same_group]`. Fetch and commit the artifact (LFS) and
      its `.copula.json` sidecar.

- [ ] 12. *(Sonnet)* **Simulation config** — new
      `configs/simulation/manager_testing/23_2g8a_contr_pna_aggregation_self_gnnpnacopar1_contr_gnn_switch.yml`,
      a byte-copy of the parent's
      `23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml` with
      exactly three edits: `contribution_model` -> the step-11 artifact;
      `output_dir` ->
      `plots/simulation/23_2g8a_contr_pna_aggregation_self_gnnpnacopar1_contr_gnn_switch`;
      `figure_name` likewise. `switch_model`, `valid_model`, the punisher, the
      single pairing, `seed: 42`, `n_episodes: 100`, `n_rounds: 24`,
      `switch_every: 4`, `save_per_round: true` untouched (verify by `diff`: 6
      changed lines).

- [ ] 13. *(Opus)* **Baseline control, then the candidate** — two simulations via
      `AI_REMOTE_DIR=... scripts/simulate_cluster.sh` (squeue check first; all
      remote artifacts already fetched and committed, so the `--delete` sync is
      safe; ~ 2.5 min GPU each). First the parent's own config unchanged: its
      `per_round.parquet` must reproduce sha256
      `0a34f8280bccb98a75fe002eb3669827358117ce56c44f7c10268f312904b7ab` **bit for
      bit** — the licence to compare against the parent's `scores.csv`, and the
      end-to-end proof that steps 1-2 are inert for an artifact without
      `aggregators` (the flag-off path). Then the step-12 candidate; its parquet
      must *differ*. Both job logs must show `aimanager` resolving inside
      `~/autoresearch/pna-aggregation/src` and the loaded contribution model's
      `aggregators` / `copula_rho` / `copula_phi` / `edge_encoding`. Fetch both
      output dirs with `fetch_cluster.sh` from the isolated dir.

- [ ] 14. *(Opus)* **Evaluate** — `python -m aimanager evaluate <step-12 config>` locally,
      with `aimanager.__file__` confirmed at this worktree's `src`. One
      simulation, one evaluation, no second stage (§3). Commit
      `per_round.parquet`, `evaluation/metrics.csv`, `evaluation/scores.csv`,
      `evaluation/visuals/`. Record the results row with CG and RCA at full
      precision, rows <= 1, the mean; every watch item explicitly; and, from the
      candidate parquet, the realised spread ratio and its two SDs, the
      closed-loop P(|dc| >= 5) by own-group peer std tercile against the
      human, so the mechanism is read directly and not only through the score.

- [ ] 15. *(Opus)* **Verdict and PR** — `[SUCCESS]` only if (CG < 2 **or** RCA <= 1) **and**
      the 21-row mean <= 1.4344450525958377; otherwise `[FAIL]`. Fill the
      results table and Notes (scores exactly as computed). Open the PR with
      `--base auto/switch-joint-exodus`, body per §9 step 7 (Hypothesis /
      Results / Collateral grouped + / -), naming any undeclared band movement
      as collateral, not claimed. Delete `~/autoresearch/pna-aggregation` when
      the PR closes.

### Plan validation (orchestrator, 2026-09-15)

Validated per §9 loop step 3: the two declared targets are legal §6 candidates
read off the parent's own `scores.csv`; every step is legal under §5; nothing on
the §8 frozen surface is touched; the slot discipline of §4 holds (switch and
punisher artifacts byte-identical to the parent's). Implementer tags attached
inline above (Opus where the step is subtle or where a past experiment has been
voided by getting it wrong; Sonnet otherwise).

Claims checked directly by the orchestrator before approval, so no step rests on
an unverified number:

1. **The control's sha256 is right.** The parent run's committed LFS pointer for
   `plots/simulation/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch/per_round.parquet`
   is `0a34f8280bccb98a75fe002eb3669827358117ce56c44f7c10268f312904b7ab`, exactly
   the value step 13 gates on. (It is *not* #171's `4f64fc42…`, which was that
   experiment's own control reproducing PR #165's run — the parent here is #171's
   candidate.)
2. **All four aggregators exist.** Raven's `torch_scatter` 2.0.9 exports
   `scatter_mean`, `scatter_max`, `scatter_min`, `scatter_std`, and
   `scatter_std(src, index, dim, out, dim_size, unbiased=True)` takes the
   `unbiased=False` step 1 specifies.
3. **M0's fit baseline reproduces.** The committed
   `.../group_switching_contribution_50ep/metrics/architecture_node+edge+rnn__dataset_50ep__epochs_575.parquet`
   gives test log_loss at epoch 574 of 2.048224 / 2.035988 / 1.963259 / 1.999429 /
   1.901808 by `cv_split`, mean **1.9897416823554699** — the step 7 comparison
   stands as written.
4. **The step 9 port is precondition-only.** The diff on
   `origin/auto/contribution-arrival-tenure` is 26 insertions / 6 deletions, all
   assertion logic: it admits a base carrying the three copula keys at their
   neutral defaults and *tightens* the refusal to a base carrying an active
   copula. It cannot alter a stamped value. Admitted as a precondition under the
   §4 ruling #173 and #176 recorded.
5. **Gate-2 ceiling.** 1.3040409569053069 x 1.10 = **1.4344450525958377** in
   IEEE double; the log file's value is the correct one.

**Amendments, fixed here before any number is seen:**

- **A. Arm B is a two-part change and must be reported as one.** Arm B alters the
  aggregator *and* adds the `same_group` edge bit. §4's one-change rule is
  satisfied the way PR #153 satisfied it — both arms declared before training,
  selection by a pre-declared held-out rule — but if arm B is selected, the PR
  title and hypothesis must claim "multi-aggregator + `same_group` edge bit", not
  the aggregator alone. Arm A remains the default on any tie (§5, simpler).
- **B. The step 7 fit stop-gate escalates; it does not abort.** Its purpose is to
  catch a *broken or under-optimised training* (a wider node MLP fitting worse on
  the same inputs is an optimisation symptom), never to predict CG — the
  declaration's own risk 1 records that teacher-forced readings have pointed the
  wrong way in #175 and #177, and §2 says the verdict comes from the simulation.
  Ruling, fixed now: if the gate fires, the step reports the training curves and
  stops. If there is a diagnosable optimisation fault, the orchestrator rules on
  **one** remedy inside the §5 budget and the arm is retrained once. If the
  training is clean and the arm merely fits marginally worse, **the simulation
  runs anyway** and the verdict comes from step 14.
- **C. Step 8 is report-only and cannot cancel the simulation** — as the step
  already states. Recorded here so it cannot be reinterpreted later.
- **D. Undeclared rows are collateral.** CG and RCA are the only rows that can
  satisfy gate 1. A band upgrade on RCD, RCB or anything else is reported under
  Collateral and never claimed as the success (PR #173's pre-declaration ruling).

## 4. Notes

1. **Steps 1-2 confirmed** (`79bdb39`, `67a6b42`), 90 lines in
   `src/aimanager/generic/graph.py` and nothing else. `AGGREGATORS` holds three
   thin wrappers rather than the raw `torch_scatter` functions, because
   `scatter_max`/`scatter_min` return `(values, argmax)` and `scatter_std`
   defaults to the unbiased estimator; the validator requires a `list` strictly
   rather than accepting tuples, since the load-path check is an `==` comparison
   where a tuple/list mismatch would fire a confusing assert (YAML always yields
   lists, so no config is affected).
2. **The legacy path is pinned two independent ways.** Structurally:
   `aggregators=None` gives `n_aggr == 1`, so `in_features` is the pre-change
   expression, the single `Lin` draws the same values in the same order, and no
   module is created, deleted or reordered — the RNG stream is untouched.
   Empirically: the M0-shaped trunk built under `th.manual_seed(38381)` hashes
   to the same `state_dict` sha256
   (`b5bd4985d0313218976c851cc2113a833a23342c76714db7bd84560e4916764d`) on the
   pre-change and post-change code. This is what step 13's bit-identical control
   will confirm end to end.
3. **Orchestrator check, empty neighbourhoods.** `create_fully_connected`
   (`graph.py:842`, `train.py:74`) is the only edge builder on both the training
   and the simulation path, so every node always has exactly `n_nodes - 1 = 7`
   incoming edges and `scatter_max`/`min`'s empty-neighbourhood fill can never
   surface here. No runtime handling was added; step 3 documents the fill in a
   test instead, so a future agent changing the topology finds the answer
   written down. Measured on Raven's torch_scatter 2.0.9: the fill is `0.0`,
   argmax `src.size(0)`.
4. **Step 3 confirmed** (`6146b71`, `aaffde0`): 25 new cases, and the five PyG
   stand-in installers taught the three new reductions with their *real*
   signatures (2-tuple returns, the `unbiased` keyword) — without which the
   suites error at collection, which is exactly what steps 1-2 left behind
   between `67a6b42` and `6146b71`. Local: **491 passed, 3 skipped**, zero
   failures (orchestrator re-ran a 397-test subset independently: green).
   `black` + `flake8` + `pre-commit` clean over all seven touched files. The
   suite was mutation-checked: breaking the max aggregator, the concat order, or
   the `getattr` each fails tests.
5. **Plan correction, step 3(e).** The node MLP's input width under four
   aggregators is **84**, not the plan's `3 + 80`: M0's `x_encoding` is two
   numerics plus `agent_group` **onehot at `n_levels: 2`**, which `IntEncoder`
   sizes as 2, so `x_features == 4`. Corrected in the step above; step 11's
   verification wording inherits the corrected number.
6. **Recorded because it was reported rather than hidden:** the step-3
   implementer ran two read-only probes on the Raven login node (a `sed` of
   `torch_scatter/composite/std.py` and a ~3 s `python -c` scattering a 4x2
   tensor) to establish note 3's fill value instead of shipping an unverified
   assertion. No sync, no job, no `aimanager` import. Orchestrator ruling:
   inside the "login node is orchestration only" line, which bars *compute*, not
   a three-second API probe — and the alternative was an assertion nobody had
   checked.
