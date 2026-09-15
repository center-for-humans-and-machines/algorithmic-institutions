# Per-group virtual node for the contributor: a learned, persistent group state

## 1. Declaration

**Slot:** contribution.

**Parent:** PR #171 (`auto/switch-joint-exodus`, `[SUCCESS]`), the maintainer-designated
frontier, at `6ba366c`. Branch `auto/contribution-group-vnode`, worktree
`.claude/worktrees/contribution-group-vnode`, created from `origin/auto/switch-joint-exodus`;
the PR opens with `--base auto/switch-joint-exodus`. Siblings already stacked on this
parent and read before planning: PR #173 (`contribution-group-size`, `[FAIL]`, CE
1.105 -> 1.050; its step 7b fixes the copula stamper this recipe reuses, its note 18
names the common-good channel into the S rows) and PR #176
(`contribution-arrival-tenure`, `[FAIL]`, RCD 2.765 -> 2.053 by 0.053 of score; the
procedural template this plan follows: retrain the trunk, recalibrate the copula on
it, control sim + candidate sim).

**Base model:** the parent stack's contributor -- the M0 GNN trunk
`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`
(`x_encoding = prev_contribution (numeric, 21), prev_punishment (numeric, 31),
agent_group (onehot, 2)`; no `edge_encoding`; hidden 20; `add_global_model: False`;
575 epochs, batch 4, lr 3e-4, seed 38381, flip-doubled data) **plus** PR #165's stamped
copula (`copula_rho = 0.06958238086256316`, `copula_phi = 1.0`, `copula_switch_every = 1`),
shipped as
`artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`.
The base is "trunk + stamped copula"; so is the candidate.

**Evaluation stack (§3 under the parent rule of §9):** the parent's own config
`configs/simulation/manager_testing/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml`
-- this contributor x the joint-exodus GNN switch predictor
(`artifacts/artificial_humans/switch_joint_exodus/...`) x the severity-copula
`lin_multinomial` punisher (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`),
single pairing `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
`save_per_round: true`.

**Baseline (the parent's confirmed scores; both §2 gates are judged against these).**
Source: `plots/simulation/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch/evaluation/scores.csv`
in this worktree (`per_round.parquet` sha256
`0a34f8280bccb98a75fe002eb3669827358117ce56c44f7c10268f312904b7ab`).

| row | score | band | numerator (500-repeat mean) | noise ceiling |
|---|---|---|---|---|
| **CG** (primary target) | **4.267640451429015** | 2-5 | 0.11287345494901657 | 0.026448679600274437 |
| **RCD** (secondary target) | **2.764919035295771** | 2-5 | 0.22148846810729983 | 0.08010667411228792 |
| RCB (watch) | 2.0242478714062093 | 2-5 | 0.7041451592840497 | 0.3478552054965939 |
| RCA | 1.5746350718021775 | 1-2 | | |
| RCC | 1.5810557625733717 | 1-2 | | |
| RSA | 1.3222991041595975 | 1-2 | | |
| RPA | 1.247405946294638 | 1-2 | | |
| SC (the parent's success row) | 1.147392663266986 | 1-2 | | |
| CE | 1.105386005744275 | 1-2 | | |
| SB | 1.06526788518747 | 1-2 | | |
| mean over 21 rows | **1.3040409569053069** | | | |
| gate-2 ceiling (mean x 1.10) | **1.4344450525958377** | | | |
| rows <= 1 | 11/21 (context only) | | | |

**Target rows and their band-upgrade thresholds.**

- **CG, primary.** Band 2-5 -> 1-2 requires the resampled ratio gap below
  **0.0528973592005489** (2 x ceiling). On the canonical evaluation-suite frames the
  human spread ratio is **0.848016** (SD of per-(game, round, group) means 5.3577 over
  SD of individual contributions 6.3179) and the parent's is **0.738087** (4.8322 /
  6.5469; raw gap 0.10992965817635214 in `metrics.csv`, resampled 0.11287). The
  simulated ratio must therefore reach **> 0.7951**, a **+0.0570** move -- the gap has
  to halve. For scale: PR #165's copula moved it +0.152 (0.586 -> 0.738), PR #170's
  +0.082, PR #159's latent +0.076, while the two structural peer-channel changes on
  the pre-copula stack (PR #153 attention, PR #157 conformity mixture) moved CG by
  ~15% of a much larger gap. Honest reading: this is a large ask, at the upper end of
  what any single mechanism has delivered, and the band edge is not near.
- **RCD, secondary.** Band 2-5 -> 1-2 requires the pull-slope gap below
  **0.16021334822457584**: human slope 0.4302, parent 0.2062, so the simulated slope
  must exceed **0.2700**. PR #176's tenure counter reached 0.2638 alone and reported
  (note 14) that its arrivals "let go of the anchor but had nothing new to move
  toward" -- the receiving group's state is exactly what this mechanism hands them
  (preflight below: human arrivals weight the receiving group's recent *history* at
  0.28, the parent's at 0.15).

Gate 1 is met by either row leaving its band; gate 2 requires the 21-row mean
<= 1.4344450525958377.

**Watch items (reported whatever the verdict, never claimed):** **RCB** (2.0242, 0.024
from the edge; the group's punishment climate does enter the virtual node, but RCB
conditions on the player's *own* punishment rate and no clean one-sentence claim
exists -- declaring a row 1.2% from an edge without one is shopping, PR #173 note 16);
the marginal C block **CA / CB / CD / CF / CC / CE** (a retrained trunk re-randomises
everything the contributor does, and PR #173 note 17 shows a 1% likelihood cost
propagating to every C row); **SA / SB / SC** through the common-good channel (PR #173
note 18; SC is the parent's success row and the one most exposed to any
contribution-slot change); **RCA / RCC** (retrain wobble); **PD** (PR #166 showed the
punisher's copula row is sensitive to the contribution slot's dependence structure,
which the recalibration changes).

### Hypothesis

**The behaviour.** A player reads *their own group* -- its level, its punishment
climate, and, on arrival, what kind of group it has been -- and moves toward it.
Measured on the human data (canonical frames, rounds >= 4, n = 7,137 rows with all
lags valid; OLS, episode-cluster bootstrap CIs):

| regressors (human contribution at t) | own c(t-1) | all 7 others' mean c(t-1) | own group's others' mean c(t-1) | R² |
|---|---|---|---|---|
| own + group-blind peer mean (what M0's `scatter_mean` sees) | 0.745 | 0.229 | -- | 0.6954 |
| + own-group mean | 0.695 | **-0.0005** | **0.278** [0.222, 0.327] | 0.7124 |
| + own c(t-2), c(t-3) | 0.469 | -0.050 | **0.264** | 0.7394 |

The human peer response is *entirely own-group-specific*: once the own-group mean
enters, the group-blind seven-peer mean drops to zero weight, and the other group's
mean carries nothing (-0.014). The weight survives controlling for the player's own
history (0.264 with two own lags), so it is not redundant with own trajectory. Two
further own-group channels: group-mates' previous punishment raises a human's
contribution (+0.165 [0.103, 0.238] against own previous punishment +0.02 -- seeing
others punished is a deterrent), and **arrivals weight the receiving group's recent
history** (mean of its cell means over t-2..t-4) at **0.280 [0.061, 0.503]**, above its
last-round mean (0.118 [-0.028, 0.260]; n = 513) -- newcomers read what the group
*has been*, not just its last move. For settled members that history term is ~0.02:
the group's past matters at arrival, its present matters always.

**The parent's sim has the shared-variance half and lacks the response half.** Same
regressions on the parent's parquet (n = 15,283): own-group mean **0.060** [0.036,
0.087] against the human 0.278 -- one fifth -- with the group-blind mean at 0.114
(M0's channel, not the human one); group punishment climate **-0.003** [-0.023,
0.019]; arrivals own 0.771 / receiving-group history 0.147 (human 0.465 / 0.280).
Meanwhile the sim's *stayers* carry a group-history weight of 0.108 against the
human 0.020 -- that is PR #165's static episode latent showing up as "my group's past
predicts me", a persistent shared *offset* standing in for a shared *response*. The
spread ratio by round thirds says the same: human 0.719 / 0.876 / 0.900, parent
0.658 / 0.738 / 0.791 -- both grow (lock-in), the parent from the latent, the human
from members converging on their group and the groups then drifting apart; the gap is
widest in the middle third (0.138) where conformity compounds.

**Why the trunk cannot express it.** M0 pools its seven incoming edge messages with
a group-blind `scatter_mean`, the edge index is complete over all 8 agents regardless
of membership (`graph.py: create_fully_connected`, `train.py` likewise), and
`agent_group` enters only as a onehot on the node. PR #176 note 9 measured the
consequence directly: shuffling `agent_group` costs the trunk **nothing**
(-0.000156 / +0.000863 held-out log-loss) -- the frontier contributor is group-blind
in fact, not just in architecture. There is no object in the model that *is* the
group.

**Planned change: a learned per-group virtual node with its own recurrent state.**
Each round, the members of each group are mean-pooled (post-`op1` node embeddings,
plus the group's occupancy k / 8) into a group input; a **group GRU** -- one per
group, weights shared, hidden state carried across the episode -- turns it into a
persistent group state; that state is **broadcast back to the group's members and
concatenated to their post-`rnn_n` embedding at the `op2` readout**. Per group rather
than per graph (Gilmer et al. 2017 / OGB virtual node), because the human response is
own-group-specific and the two groups are the two cultures CG measures. Membership is
time-varying, so a switcher reads their *arrival* group's state from the switch round
on (`apply_switch` runs before `update_contribution`, the same fact PR #165's copula
cells rest on) -- which is the RCD claim. An emptied group pools to a zero vector with
occupancy 0 and its GRU keeps stepping; no member reads it until someone arrives. The
per-agent path `op1 -> rnn_n` is untouched and the node is trained *attached* by the
per-agent loss -- this is a trunk change, not a readout head, so the PR #171 detach
does not apply. Legality: everything the node reads -- group-mates' previous
contributions and punishments, membership and size -- is what a real player sees on
their screen at decision time (`notes/baseline_feature_defs.md`: membership-derived
features are legal for the contribution target; the prev-anchored group means are the
linear family's own `contribution_mean_group` / `punishment_mean_group` features).
Nothing is keyed to a bin or stratum. The base is trunk + calibrated copula, so the
candidate is completed by **re-running PR #165's calibrate -> stamp recipe on the
retrained trunk**: rho is model-conditional (PR #166: separately calibrated latents do
not compose), and a trunk that absorbs part of the within-group residual dependence
into its conditional should *lower* the recalibrated rho -- that number is itself a
reading of the mechanism. **One change, no knob, one evaluation.**

**Why this is not a retry.** PR #158 (per-agent type latent) showed CG needs *shared*
variance; PR #159 (episode-persistent group latent, noise) and PR #165 / #170 (group
copulas) supplied it at the sampler and moved CG further than anything else -- but a
latent is an offset, and the regressions above show the remaining gap is a
*response*: the sim's own-group weight is 0.06 against 0.28. PR #153 (peer attention)
sharpened the edge weighting (2.9x on same-group peers) but kept a stateless,
per-graph aggregation and moved CG ~15% -- "attention learned real group structure
but CG is free-running drift". PR #157 (conformity mixture) hand-built a bell around
the group mean and found the gate pinned at w ~ 0.08 by teacher-forced MLE because
own history already explained the peer-correlated variance; the human table above
says the own-group weight survives two own lags (0.264), so on this data the channel
is not redundant. PR #167's group-conditioned feature core lifted the peer weight on
the Gaussian head and PR #170 then had to add the shared latent on top; here the
shared latent is already in the base and the learned group state is the missing
half. What is new: a *stateful*, *learned*, *per-group* object -- not a noise latent,
not an edge weight, not a hand-built kernel, not a per-graph global -- carrying the
group's level, its punishment climate and its history in one recurrent state the
readout can use. **What the record predicts will go wrong**, stated before anything
runs: (i) PR #157's MLE-dose problem may recur in learned form -- the linear R² gain
of the own-group channel (0.017) is real but modest, and PR #157 records that the
explicit own-group-mean feature (#116, M3) did *not* buy CV log-loss on this trunk;
if the virtual node's held-out log-loss lands flat against M0's 1.989742, the
response was not learned and the sim will show it; (ii) a retrained, larger trunk
(the group GRU roughly doubles the parameter count, ~3.0k on ~3.6k) at M0's 575-epoch
budget risks the C-block tax that PR #173 paid (all C rows up on a 1% likelihood
loss) and gate 2; (iii) SC, the parent's success row, is exposed through the
common-good channel. **Probability, stated before anything runs:** gate 1 on CG
~0.3, on RCD ~0.3, either ~0.45; gate 2 given a band upgrade ~0.7.

**Iteration budget (§5).** Recent plain contribution trains on Raven: 07:57 (job
29891768), 08:52 (job 29898154), 08:09 (job 30252677) -> 3x ceiling ~24-27 min. The
group GRU adds one recurrence over 24 rounds on 8 sequences per batch of 4 episodes
(the node GRU already runs 32) plus one `index_add_` pooling per forward; estimate
**~1.2-1.5x, ~10-13 min**, inside the ceiling with margin. Whole experiment: ~12 min
GPU training + ~11-12 min CPU calibration + <1 min stamp + 2 x ~2.5 min simulations,
~35 min of cluster wall-clock; every step is a same-day step.

## 2. Plan

To be validated by the orchestrator against §2 (targets), §5 (legality, budget) and §8
(frozen surface) before any step runs; implementer tags are the orchestrator's. Nothing
under `src/aimanager/evaluation_suite/`, `notes/evaluation_metric_defs.md`,
`notes/eval_scoring_schema.md` or `experiments/` is touched; simulation protocol, seeds
and episode count are the parent's. Every remote call sets
`AI_REMOTE_DIR='~/autoresearch/contribution-group-vnode'` (the launchers in this
worktree honour it: `train_cluster.sh` / `simulate_cluster.sh` / `fetch_cluster.sh`
read it, and `run_training.sh` / `run_simulation.sh` carry the union template --
`SBATCH_EXPORT=ALL` plus the in-job `PYTHONPATH` export, `fbec309`), checks `squeue`
for PENDING jobs first, and confirms the SSH tunnel is live before reading an empty
queue as safe (PR #171 note 26). **`rsync --delete` hazard (PR #173 note 11):** the
launchers sync `artifacts/` with `--delete`, so every artifact a remote step produces
is fetched into this worktree *before* the next launcher call; single new files
(slurm wrappers) go by `scp`. Heavy compute -- training, calibration, stamping,
simulation, the pre-sim diagnostic -- goes through `sbatch`, never the login node;
login-node work is pytest and orchestration only.

### Orchestrator validation (Opus, 2026-09-15) -- before any step ran

Plan validated against §2 (targets are rows with score >= 2 in the baseline, per §6's
own rule), §5 (every input the node reads is lagged, player-observable state; one
change; budget estimated against a measured ceiling) and §8 (nothing frozen is
touched). Implementers attached below: Opus where the step is structurally risky
(bit-identity, RNG discipline, calibration rulings, the verdict), Sonnet otherwise.
Commits map to steps (§9 step 4), one per confirmed step. Five amendments, all ruled
here, before anything ran:

- **A0 -- RCB is decided by measurement, not by argument (new step 0).** The
  declaration makes RCB a watch item because the measured group-punishment-climate
  effect (+0.165) is on the contribution *level*, while RCB stratifies the *change*
  by the player's own punishment rate -- a different quantity, and §5 wants the claim
  measured rather than told. That is correct as reasoning, and it is also a question
  the human data answers for free. Step 0 measures the interaction directly. If the
  climate genuinely moves the punishment response, RCB is declared as a third target
  with that sentence as its rationale, and the log states plainly that its threshold
  is only 0.024 away so a crossing there is weak evidence next to CG or RCD. If the
  interaction is absent or indistinguishable from zero, RCB stays a watch item and
  cannot be claimed later whatever it does. The measurement is binding either way and
  is committed before step 1, so the declaration is fixed before any candidate exists.
- **A1 -- step 10's departure from the #165 stop-gate is confirmed, with the
  discretion removed.** Fable is right that a vanishing residual rho does not end this
  experiment: the copula is inherited plumbing that §9/#166 obliges us to recalibrate,
  not the mechanism under test, so the MLE landing at zero is the trunk having
  absorbed the shared component -- the hypothesis working. But the "CI includes 0"
  form of the rule creates a discontinuity (rho_hat 0.04 stamped or discarded on which
  side of zero a CI endpoint falls) and a judgment call at exactly the moment the
  number becomes visible. Replaced by a rule with no discretion in it: **stamp rho_hat
  as measured whenever rho_hat > 0; take the bare trunk only when rho_hat <= 0**,
  which is also the only case the sampler cannot represent. The CI is recorded and
  interpreted either way, and exactly one candidate is simulated on either branch.
- **A2 -- the phi rulings stand as pre-declared**, with one change: the
  phi-CI-includes-0 case escalates to the orchestrator and is not self-resolved by the
  implementer.
- **A3 -- no retuning.** The training recipe is M0's, byte-identical apart from step
  5's three declared edits: 575 epochs, batch 4, lr 3e-4, hidden 20, 5-fold, seed
  38381. The node roughly doubles the parameter count at a fixed epoch budget and that
  is a known risk to gate 2; a worse held-out log-loss at step 7 is a **finding to
  report**, not a licence to adjust epochs, learning rate or width. An implementer that
  wants to tune stops and escalates instead.
- **A4 -- step 9's stamper port is an enabler, not a second experiment**, under the
  precedent PR #173 and PR #176 set: the 67-line diff is assertion logic that cannot
  alter a stamped value, and without it the stamper refuses any trunk trained after
  copula support landed. Recorded here as the §4 ruling.

0. *(implementer: Opus)* **Measure the punishment-climate interaction, no code shipped
   and nothing on the cluster** -- local, on the canonical human frames the declaration
   already used (`evaluation_suite/convert.py` loader, read-only; the suite itself is
   frozen and is not modified). Regress the human contribution *change*
   `c(t) - c(t-1)` for punished non-full contributors on their own punishment rate
   `p / (20 - c)`, the own-group climate (group-mates' mean punishment rate at t-1,
   leave-one-out) and **their interaction**, with episode-cluster bootstrap CIs, and
   report the same fit split by RCB's four rate bins so it is visible *where* any
   deficit sits. Run the identical fit on the parent's `per_round.parquet` for the
   contrast. **Binding rule, fixed before the numbers are seen:** RCB becomes a
   declared target iff the interaction coefficient's 95% CI excludes zero on the human
   data **and** the parent's sim coefficient lies outside that CI -- i.e. the response
   depends on the climate in humans and the sim gets it wrong. Otherwise RCB stays a
   watch item. Amend §1 with the numbers and the resulting declaration, commit, and
   report; no other step is affected by the outcome.

1. *(implementer: Opus)* **The virtual-node module** -- new `src/aimanager/generic/group_vnode.py`, torch-only
   (no `torch_scatter` / `torch_geometric`, so it imports and tests on macOS like
   `joint_exodus.py`). `class GroupVirtualNode(th.nn.Module)` with
   `__init__(embed_size, hidden_size, *, n_groups=2, size_norm=8.0)`: one
   `GRU(input_size=embed_size + 1, hidden_size=hidden_size, num_layers=1,
   batch_first=True)`. `forward(x, *, agent_group, batch, h0=None, n_batch=None)` with
   `x` float `(N, R, F)` post-`op1` node embeddings, `agent_group` int `(N, R)`, `batch`
   int `(N,)`: (a) `pooled, counts = pool_by_group(x, agent_group, batch, n_batch=...)`
   (reuse `joint_exodus.pool_by_group` unchanged -- every member counts, membership is
   not validity, PR #173 step 1's ruling); (b) group input
   `cat([pooled, counts / size_norm], -1)` -> `(n_batch, R, G, F + 1)`, permuted and
   reshaped to `(n_batch * G, R, F + 1)` with the group index *fastest* so the hidden
   layout is `(1, n_batch * G, H)` and stable across per-round calls; (c)
   `g, h = self.gru(seq, h0)`; (d) broadcast `g_node[n, r] = g[batch[n] * G +
   agent_group[n, r], r]` -> `(N, R, H)` via one `gather`. Returns `(g_node, h)`. An
   empty cell pools to zeros with occupancy 0 (no NaN; asserted). Module docstring
   states the design points above (post-`op1` pooling, readout injection, arrival
   pickup, empty groups, time-varying membership).

2. *(implementer: Opus)* **Wire it into `GraphNetwork`** -- `src/aimanager/generic/graph.py` (existing). Constructor
   keywords `group_vnode=False, group_vnode_module=None, group_vnode_hidden=None`
   (all persisted). Asserts: `group_vnode` is a bool; `group_vnode_hidden` is `None` or
   a positive int (excluding `bool`, the `joint_exodus_switch_every` precedent);
   `(self.group_vnode_module is not None) == self.group_vnode` after construction
   (the joint-head pattern). In the `op1 is None` build branch: `vnode_hidden =
   group_vnode_hidden or hidden_size`; `op2`'s `NodeModel` gets `x_features =
   x_features + (vnode_hidden if group_vnode else 0)`; the module is built **LAST**,
   after the joint-head slot (`GroupVirtualNode(embed_size=hidden_size,
   hidden_size=vnode_hidden)`), so with the flag off every parameter is initialised
   from exactly today's RNG state and the model is bit-identical. `forward`: after
   `op1`, if the node is present, `g_node, self.vnode_h0 = self.group_vnode_module(x,
   agent_group=data["agent_group"], batch=batch, h0=None if reset_rnn else
   self.vnode_h0)`; after `rnn_n`, `x = cat([x, g_node], -1)` before `op2`; `vnode_h0`
   initialised to `None` in both constructor branches. `encode`: the existing
   joint-head block that carries `agent_group` becomes `if self.joint_exodus_head is
   not None or self.group_vnode_module is not None` (still `round_number` only for the
   head), so a model without either encodes exactly the keys it encodes today.
   `save`: append `group_vnode`, `group_vnode_module`, `group_vnode_hidden` to
   `to_save`; `load` is untouched -- an artifact without the keys gets the defaults.
   The copula dispatch (`_predict_encoded_copula`) and the legacy path call
   `self(encoded, reset_rnn)` and need no change; `train.py`, `evaluation.py` and the
   calibration script all go through `encode` -> `forward` and need no change.

3. *(implementer: Sonnet)* **Module tests, local** -- new `tests/vnode/test_group_vnode.py` (torch-only, plain
   pytest): output shapes; every node reads its own group's state and nothing else
   (two groups fed distinguishable inputs); an agent that changes `agent_group`
   mid-sequence reads the new group's state from that round on; an emptied group
   yields a finite zero-input step and no NaN anywhere; **per-round parity** --
   feeding the 24 rounds as 24 `R = 1` calls with the carried `h` reproduces the single
   `R = 24` call to 1e-6 (the train/sim contract for a recurrent module; the
   simulation calls `predict` once per round with `n_rounds = 1`,
   `environment.py: update_contribution`); relabelling the groups (the flip
   augmentation) permutes the group states and leaves every node's output unchanged;
   `pool_by_group`'s counts equal the membership counts.

4. *(implementer: Opus)* **Graph gate tests, local with stand-ins and on Raven with real PyG** -- new
   `tests/vnode/test_group_vnode_graph.py`, modelled on
   `tests/switch/test_joint_exodus_graph.py` (its `make_model` / `make_data` /
   `legacy_predict` / `run_seeded` fixtures, contribution-shaped: `y_levels=21`,
   `y_name="contribution"`). Gates: (a) off by default -- a model built with the flag off
   has every `op1` / `rnn_n` / `op2` tensor `torch.equal` to one built before this
   change under the same seed, and `predict_independent(sample=True)` matches
   `legacy_predict` in values and RNG consumption; (b) an artifact saved and then
   stripped of the three new keys loads with `group_vnode is False`,
   `group_vnode_module is None` and samples bit-identically; (c) save/load round-trip
   with the flag on preserves the module's state dict and `group_vnode_hidden`; (d) on
   != off -- the forward differs, and shuffling `agent_group` across episodes changes
   the on-model's output but not the off-model's (the group-blindness contrast PR #176
   note 9 measured); (e) 24 per-round `predict_independent` calls with `reset_rnn` only
   at round 0 reproduce one 24-round call (the `vnode_h0` carry, including with the
   copula fields stamped so `_predict_encoded_copula` and the node co-exist); (f) with
   `agent_group` absent from the data the on-model raises the assert, the off-model
   does not. Local `pytest tests/vnode` must be green before step 6.

5. *(implementer: Sonnet)* **Training config** -- new
   `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode.yml`,
   a verbatim copy of `group_switching_contribution_50ep.yml` (575 epochs, batch 4,
   lr 3e-4, hidden 20, 5-fold, seed 38381, `x_encoding` and `shuffle_features`
   unchanged -- `agent_group` stays in `shuffle_features`, which now becomes the
   direct readout of how much the model uses group structure) with exactly three
   edits: `model_args.group_vnode: True`; `output_dir:
   artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode`; a
   `description` naming this experiment. `group_vnode_hidden` is *not* set (defaults to
   `hidden_size`, one knob fewer). Labels unchanged so the artifact filename
   `architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` is preserved.

6. *(implementer: Sonnet)* **Isolated remote dir, Raven tests, lint** -- create the dir once with
   `AI_REMOTE_DIR=... scripts/train_cluster.sh --sync-only ah <step-5 config>` (squeue
   PENDING check and live-tunnel check first); confirm remote `aimanager.__file__`
   resolves inside `~/autoresearch/contribution-group-vnode`; run the PyG suites by
   login-node pytest *inside* it -- `src/aimanager/tests/test_encoder.py`,
   `test_edge_encoder.py`, `test_environment.py`, `test_linear_manager.py`,
   `test_contribution_copula_graph.py`, `test_switch_copula_graph.py`, the
   `tests/switch` and `tests/copula` suites and both `tests/vnode` suites (each
   reporting its stand-ins not installed) -- never `scripts/remote_test.sh`, whose
   shared-checkout `--delete` sync is the race that voided an earlier branch. Eval-suite
   fixture failures on Raven are expected (PR #165 note 4) and not this experiment's.
   One batched `black` + `flake8` (88, `E203,W503`) pass over the touched `src/` files
   before staging.

7. *(implementer: Sonnet)* **Train on Raven** -- `AI_REMOTE_DIR='~/autoresearch/contribution-group-vnode'
   scripts/train_cluster.sh ah <step-5 config>`. Record: SLURM job id, elapsed (expected
   ~10-13 min against the ~24-27 min ceiling; **if it exceeds the ceiling the method is
   ruled out per §5 and the experiment ends as a budget `[FAIL]`**), the in-job
   provenance (the log's `Work dir` and `aimanager` resolving inside the isolated dir),
   the artifact sha256 and its loaded fields (`group_vnode True`, `group_vnode_hidden`,
   `copula_rho 0.0`), and from the run's metrics parquet the per-fold held-out log-loss
   at epoch 574 against M0's **1.989742** (folds 2.048224 / 2.035988 / 1.963259 /
   1.999429 / 1.901808) plus the `shuffle_feature = agent_group` log-loss delta against
   M0's ~0 (PR #176 note 9) -- a positive delta is the first evidence the model uses the
   group. Then **immediately fetch**
   `artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/` into this
   worktree with `fetch_cluster.sh` and commit it (LFS) -- before any further launcher
   call.

8. *(implementer: Opus)* **Pre-sim diagnostic, report-only** -- scratch script (not committed), run locally
   with the PyG stand-ins on CPU (the stand-in `scatter_mean` is exact, PR #176 step 6's
   precedent) or via `sbatch` in the isolated dir, never on the login node. Teacher-forced
   on the human single-copy data, regress the model's expected contribution E[c(t)] on
   own c(t-1), the group-blind seven-peer mean and the own-group mean, for the new trunk
   and for M0, and report the own-group weight (human 0.278; M0 expected ~0), the
   group-punishment-climate weight (human +0.165), the arrival-row own / receiving-group
   weights (human 0.460 / 0.280; M0 0.707 / 0.111, PR #176 note 4) and the
   teacher-forced pull on the 513 human switch events (M0 0.186, human 0.430). **Whatever
   it says, the simulation runs and the verdict comes from the single evaluation** (§2,
   §6); this step exists so a failure reads as "not learned" or "learned but did not
   carry".

9. *(implementer: Sonnet)* **Port the stamper precondition fix** -- `git checkout
   origin/auto/contribution-group-size -- scripts/artificial_humans/make_contribution_copula_artifact.py`
   (PR #173 step 7b: `NEUTRAL_FIELDS`, `assert_only_copula_fields_changed`; a 67-line
   diff of assertion logic that cannot alter a stamped value). This branch's stamper
   still carries `assert k not in base` (line 244), which refuses any trunk trained
   after copula support landed, so step 11 cannot run without it. The orchestrator
   records the same §4 ruling PR #173 and PR #176 did. Checked at planning: the ported
   `assert_only_copula_fields_changed` walks *every* non-copula top-level key with
   `same()`, so the new `group_vnode_module` tensors are covered by the bit-identity
   check automatically; `MODULE_KEYS` there is a print line only. Add
   `"group_vnode_module"` to that tuple so the log names the node among the modules
   compared -- cosmetic, but it is what a reviewer reads.

10. *(implementer: Opus)* **Calibrate the copula on the new trunk, with an amended stop-gate** -- copy
    `scripts/artificial_humans/calibrate_copula.slurm` to
    `calibrate_copula_group_vnode.slurm` with `BASE` = the step-7 artifact and `PARAMS` =
    `artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode_herding_copula/calibration/copula_params.json`;
    `contribution_copula_rho.py` unchanged (it drives the model through
    `predict_independent(sample=False)`, so the node runs teacher-forced inside it).
    `scp` the wrapper, `sbatch` from the isolated dir (~11-12 min). The job exits `1:0`
    by design on `phi >= 1` (`STOP-ESCALATE`); not a crash. Record rho, its
    200-episode-cluster bootstrap CI, phi_hat and CI, the round-trip gate, the preflight
    ratio pair, the round-thirds rho -- all against PR #165's (0.06958238086256316,
    [0.0459, 0.0855]) and PR #173's (0.0752) values. **The gate, ruled here rather than
    after the number is seen.** A rho whose CI excludes 0 -> `phi_final = 1.0` by PR
    #165's boundary ruling when the phi CI includes 1 (the estimate stays in the JSON);
    if the phi CI lies entirely below 1 with phi_hat > 0, stamp phi_hat (the sampler
    supports phi in (0, 1] and the estimator has then measured mean reversion rather
    than saturating); if the phi CI includes 0, escalate to the orchestrator before
    stamping. **A rho whose CI includes 0 does NOT end the experiment here, unlike PR
    #165 / #173 / #176:** for this mechanism a vanishing residual within-group
    dependence is the trunk having absorbed the shared component into its conditional
    -- the hypothesis working, not the recipe failing. In that case skip the stamp
    (the stamper asserts `0 < rho`) and simulate the bare step-7 trunk as the
    candidate; the candidate is still "trunk + MLE-calibrated copula" with the MLE at
    its neutral value. Record which branch was taken. Fetch and commit the params JSON
    and the job log (`git add -f`, the #165 precedent).

11. *(implementer: Sonnet)* **Stamp** -- copy `scripts/artificial_humans/stamp_copula.slurm` to
    `stamp_copula_group_vnode.slurm` invoking the step-9 stamper with `--params` (step
    10), `--base` (step 7) and `--out
    artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`.
    Verify: the three copula fields round-trip, every weight tensor -- `op1`, `rnn_n`,
    `op2` **and `group_vnode_module`** -- `torch.equal` to the step-7 trunk, the honesty
    check reports the 7,457 teacher-forced train-split rows bit-identical, and the loaded
    artifact carries `group_vnode True`. Fetch and commit the artifact (LFS) and its
    `.copula.json` sidecar. (Skipped if step 10 took the rho-at-zero branch.)

12. *(implementer: Sonnet)* **Simulation config** -- new
    `configs/simulation/manager_testing/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml`,
    a byte-copy of the parent's
    `23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml` with exactly three
    edits: `contribution_model` -> the step-11 artifact (or the step-7 trunk on the
    rho-at-zero branch); `output_dir` ->
    `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch`;
    `figure_name` likewise (slug before `_self_` so `evaluation_sweep.py`'s
    `DIR_PATTERN` still parses). Switch model, valid model, punisher, pairing list, seed,
    episodes, rounds untouched.

13. *(implementer: Sonnet)* **Baseline control, then the candidate** -- two simulations via
    `scripts/simulate_cluster.sh` with `AI_REMOTE_DIR` (~2.5 min each; all remote
    artifacts already fetched, squeue checked). First the parent's own config unchanged:
    its `per_round.parquet` must reproduce sha256
    `0a34f8280bccb98a75fe002eb3669827358117ce56c44f7c10268f312904b7ab` **bit for bit** --
    the licence to compare anything, and the proof that steps 1-2 are inert for an
    artifact that carries no `group_vnode` key. Then the step-12 candidate; its parquet
    must *differ*. Fetch both with `fetch_cluster.sh` from the isolated dir; verify
    neither is an LFS pointer stub.

14. *(implementer: Opus)* **Evaluate and rule** -- `python -m aimanager evaluate <step-12 config>`, locally,
    with `aimanager.__file__` confirmed at this worktree's `src`. One simulation, one
    evaluation, no second stage (§3). Record the results row: CG and RCD at full
    precision, rows <= 1, mean; every watch item explicitly (RCB, CA / CB / CD / CF / CC /
    CE, SA / SB / SC, RCA / RCC, PD); and, from the candidate parquet with the preflight
    script, the closed-loop own-group weight, group-punishment-climate weight,
    arrival own / receiving-group-history weights, the spread ratio by round thirds and
    the RCD slope, so the mechanism is read directly and not only through the score.
    `[SUCCESS]` only if (CG < 2 or RCD < 2) **and** the 21-row mean <=
    1.4344450525958377; otherwise `[FAIL]`. Fill the results table and Notes; open the PR
    with `--base auto/switch-joint-exodus`, body per §9 step 7 (Hypothesis / Results /
    Collateral grouped +/-); delete `~/autoresearch/contribution-group-vnode` when the
    PR closes.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-15 | (baseline) parent stack, PR #171 joint-exodus switch x PR #165 copula contributor x PR #160 severity-copula punisher | CG 4.267640451429015, RCD 2.764919035295771 | 11/21 | 1.3040409569053069 | baseline |

## 4. Notes
