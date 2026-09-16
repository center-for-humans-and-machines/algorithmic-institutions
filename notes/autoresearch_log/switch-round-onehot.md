# Autoresearch log: switch — one-hot decision-round index in the joint exodus head

Branch `auto/switch-round-onehot` (worktree `.claude/worktrees/switch-round-onehot`),
created from `origin/auto/contribution-group-vnode` at `11fe223` — the head of the
maintainer-designated parent PR #179, per §9 "Building on a `[SUCCESS]` PR". The PR
opens with `--base auto/contribution-group-vnode`.

## 1. Declaration

- **Slot:** switch. **One change:** the encoding of the decision-round index inside
  `JointExodusHead`. The contribution slot (PR #179's group-vnode trunk + its
  recalibrated copula, `copula_rho 0.0435568043640977`, `phi 1.0`) and the punisher
  (PR #160 severity copula) are **untouched, byte-for-byte**.

- **Parent PR:** **#179** `[SUCCESS] Per-group virtual node for the contributor`
  (`auto/contribution-group-vnode`). Its log is
  `notes/autoresearch_log/contribution-group-vnode.md`. Read before planning, together
  with **PR #171** (`switch-joint-exodus`, the head this experiment re-encodes) and
  **PR #174** (`switch-exodus-k-onehot`, `[SUCCESS]`, the same re-encoding applied to
  the head's *group sizes* on the parallel gmlp chain, SC 1.3010 -> 0.9800). #174's
  one-hot is **not** on this chain: this branch's head still carries
  `sizes = k / 8` numeric, and it stays that way — one change per experiment (§4).

- **Base model:** the parent stack's switch artifact
  `artifacts/artificial_humans/switch_joint_exodus/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`,
  sha256 `8a4ae4ade60d5443970255a4265bb7abaf555164ec020a4c77ba28ff364abdc0`, trained by
  `configs/training/artificial_humans/switch_predictor/joint_exodus.yml` (stock switch
  config — `x_encoding = common_good, punishment, agent_group, round_number`, no
  `edge_encoding`, `y_levels 2`, hidden 10, 375 epochs, batch 10, lr 5e-4, wd 1e-3,
  5-fold, seed 38381, flip-doubled data — plus `joint_exodus: True`,
  `joint_exodus_switch_every: 4`). Its head is `Linear(23 -> 10) -> Tanh ->
  Linear(10 -> 81)`, whose 23 inputs are the two group-pooled post-RNN embeddings
  (2 x 10), the two valid-decider counts as `k / 8`, and **the round as `r / 23`**.

- **Evaluation stack (§3 under the parent rule of §9):** the parent's own config
  `configs/simulation/manager_testing/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml`
  — group-vnode + copula GNN contributor, joint-exodus GNN switch, PR #160
  severity-copula multinomial punisher, `valid_model`
  `raven_script_22/model/rnn_False__dataset_full.pt`, single pairing
  `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds, `switch_every: 4`,
  `save_per_round: true` — with **only `switch_model` swapped** to this experiment's
  retrained artifact (plus `output_dir` / `figure_name`).

- **Baseline for the verdict**, read at full precision from
  `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch/evaluation/scores.csv`
  (21 rows, run `ah group_switching managed by lin_multinomial_copula_self`):

  | row | score | band | | row | score | band |
  |---|---|---|---|---|---|---|
  | CA | 0.9613090287218787 | <= 1 | | PB | 0.9193591474504252 | <= 1 |
  | CB | 0.9529251851213593 | <= 1 | | PC | 0.8654365451627818 | <= 1 |
  | CC | 0.9196696964595157 | <= 1 | | PD | 0.7748909506730214 | <= 1 |
  | CD | 0.9218986258528552 | <= 1 | | RCA | 1.4000972794261808 | 1-2 |
  | CE | 1.1106638835612843 | 1-2 | | RCB | 2.3151705700149083 | 2-5 |
  | CF | 1.0762250158942481 | 1-2 | | RCC | 1.6596292609904482 | 1-2 |
  | CG | 0.8989560290763007 | <= 1 | | RCD | 1.3404829181218743 | 1-2 |
  | SA | 0.8644984008518957 | <= 1 | | **RSA** | **1.3546243055859613** | 1-2 |
  | **SB** | **1.1109915195318010** | 1-2 | | RPA | 1.3111568561676463 | 1-2 |
  | **SC** | **0.9774633985852937** | **<= 1** | | RPB | 0.7580933001937700 | <= 1 |
  | PA | 0.5818753718233393 | <= 1 | | | | |

  mean **1.0988293946890038**, rows <= 1 **12/21**. (§2 gate-2 ceiling for reference:
  1.2087123341579042.)

### Verdict rule — maintainer ruling, 2026-09-16

**SC is already at 0.9775, in the `<= 1` floor band.** §2 gate 1 requires a *band
upgrade on a declared target row*, and there is no band above `<= 1`; declaring SC
alone would be a guaranteed `[FAIL]` however well the mechanism worked. Raised with
the maintainer before any code was written. The ruling, verbatim:

> "mean scores as main and check if rows under 1 at least stable and i will look at
> results and update the status if necessary"

Pre-registered accordingly, before the run:

1. **Primary criterion (replaces §2 gate 1 for this experiment): the 21-row mean
   must improve on 1.0988293946890038.**
2. **Stability criterion: rows <= 1 must not fall below the parent's 12/21.**
3. **SC 0.9774633985852937 must not leave the `<= 1` band**, and the mechanism's own
   behavioural signature (§ below) must lose its sign reversal.
4. **Declared secondary targets, which do have band room: SB 1.1109915195318010 and
   RSA 1.3546243055859613** (1-2 -> `<= 1`). Declared now so that an upgrade on
   either is a claimed movement rather than collateral (§10, pre-declaration binds).
   Neither is required by the ruling above.
5. The maintainer rules the final PR status at review.

### Hypothesis

The head's round input is `r / 23`, a single monotone scalar sharing one 10-unit
`Tanh` layer with both pooled group embeddings and both sizes. Measured on the
parent's own `per_round.parquet` (orchestrator diagnostic, note 1), the mean size of
the larger group in the four rounds following each of the five switches is:

| after switch | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| human | 6.44 | 6.20 | 6.04 | 5.92 | 5.84 |
| parent #179 | 6.39 | 6.15 | 5.99 | **6.30** | **6.16** |
| delta | -0.05 | -0.05 | -0.05 | **+0.38** | **+0.32** |

Humans **re-balance monotonically** across the game: each successive decision round
leaves the larger group slightly smaller than the last. The simulation tracks that
decline to a constant -0.05 for three switches and then **reverses sign**, ending the
game more segregated instead of less. A constant offset followed by a break is the
signature of a shared, smooth round trend: the head can afford one global slope in
`r`, and it spends it on the early rounds where most of the mass is.

**The change:** replace the scalar `r / 23` with a **one-hot over the five decision
rounds** (1st..5th). At decision rounds `r / 23` takes exactly five values
(3/23, 7/23, 11/23, 15/23, 19/23), so the one-hot is a **bijective re-encoding of an
input the head already has** — it adds no observable and no information, on the
realised support it is an exact reparametrisation. What changes is only the inductive
bias: five free logit offsets the head can set independently, instead of one direction
in `r` shared through ten tanh units with everything else. This is the same
re-encoding, on the same head, that took SC 1.3010 -> 0.9800 in PR #174.

**Behavioural sentence (§5):** late in the game people re-balance rather than
consolidate, and how strongly they do it is specific to *which* switching opportunity
this is, not a smooth function of the clock — so the round index is a category, not a
number. The rows that should move are **SB** (the switch rate computed separately at
each of the five decision rounds — literally the quantity the five offsets set) and
**SC**'s late-game shape, with **RSA** a weaker downstream possibility.

**Legality (§5), argued explicitly because it is close to a line.** SB's strata *are*
the five decision rounds, so a feature keyed to them must be shown to be behavioural
rather than metric-engineered. Three things carry it: (i) `round_number` is already a
model input (`encoding: numeric, n_levels: 24`) and already reaches this head — no new
observable enters, and on the decision rounds the map is a bijection, so the model
cannot learn anything it could not already have learnt; (ii) the five decision rounds
are set by the experiment's own `switch_every: 4` schedule, a game parameter every
real player lives in and knows the position of, not a boundary the evaluation suite
invented; (iii) PR #174 established the precedent of re-encoding an existing input of
this head, and was accepted. If the maintainer reads it otherwise, the row to drop is
SB — the ruling's primary criterion is the mean and does not depend on it.

### Guards (declared, non-gating, read fixed in advance)

- **G1 — the trunk must be bit-identical to the base model.** The head is detached
  (`joint_exodus.py`, "THE CUT"), so the joint loss sends no gradient to the trunk,
  and the head is constructed *last* so every trunk parameter is initialised from the
  same RNG state. The per-agent switch model must therefore come out **weight-identical**
  to the base artifact, and all score movement is attributable to the joint head. If it
  is not identical, the attribution fails and the run is reported as such.
- **G2 — the joint fit must actually improve.** Held-out joint-exodus cross-entropy
  must fall against the base model's. A re-encoding that buys nothing in-sample has no
  business moving the simulation.
- **G3 — full-exodus dose.** PR #172/#174's known failure mode: SC bought by
  over-producing full-group exodus. Report the share of decision rounds landing in a
  full-exodus cell, candidate vs human, alongside the scores.
- **G4 — iteration budget (§5).** The base model trains in **4m17s** on one A100
  (PR #171, SLURM 29870374); the ceiling is ~12m50s. Four extra input dimensions on a
  10-unit MLP cannot approach it; if training exceeds the ceiling, the method is ruled
  out regardless of result.

## 2. Plan

*Written by the orchestrator, validated against §2 (targets), §5 (legality), §8
(frozen surface: nothing here touches `evaluation_suite/`, `notes/` definitions,
`experiments/`, seeds, or the simulation protocol). Implementer per step per §9.*

1. **[Opus] Round one-hot in the joint head** — `src/aimanager/generic/joint_exodus.py`,
   `JointExodusHead.__init__` and `.forward`. Two new keyword args,
   `round_onehot_slots=None` and `round_onehot_every=None`. With both `None` the class
   is behaviourally unchanged: `in_features = n_groups * embed_size + n_groups + 1` and
   `forward` keeps `rounds = r / round_norm`. With `round_onehot_slots = S`,
   `in_features = n_groups * embed_size + n_groups + S` and `forward` emits, in place of
   the scalar, a one-hot at index `clamp(round // round_onehot_every, 0, S - 1)`, set
   only where `(round + 1) % round_onehot_every == 0` and all-zero elsewhere (rounds the
   loss never selects and the sampler never fires on). Both are read in `forward` through
   `getattr(self, ..., None)` so a head pickled before this change restores and runs the
   numeric path exactly as today.
2. **[Opus] Plumb the flag through the graph model** — `src/aimanager/generic/graph.py`,
   the `elif joint_exodus:` head construction and the `save` list. New `model_args` key
   `joint_exodus_round_onehot` (positive int or `None`), validated alongside the other
   joint args: it requires `joint_exodus` and a non-`None` `joint_exodus_switch_every`,
   which it passes to the head as the cadence. Added to `save` so it round-trips. Note in
   the code that this promotes `joint_exodus_switch_every` from sampler-only to
   training-relevant, and assert it against the training `switch_every` where both are in
   hand.
3. **[Sonnet] Unit tests** — new `src/aimanager/tests/test_joint_exodus_round_onehot.py`
   (torch-only, no PyG, runs locally): (a) with the flag off, outputs are bit-identical
   to the pre-change class under the same seed; (b) the one-hot fires on exactly the
   decision rounds, sums to 1 there and is zero elsewhere; (c) the five realised decision
   rounds map to five distinct slots; (d) `GraphNetwork.save`/`load` round-trips the flag
   and the head's shape; (e) a head object lacking the new attributes still forwards on
   the numeric path.
4. **[Sonnet] Training config** — new
   `configs/training/artificial_humans/switch_predictor/joint_exodus_round_onehot.yml`,
   copied verbatim from `joint_exodus.yml`; only `model_args.joint_exodus_round_onehot: 5`,
   `output_dir: artifacts/artificial_humans/switch_joint_exodus_round_onehot`, and the
   description differ. Every hyperparameter, the seed, the fold count and the data
   handling are unchanged — this tests a mechanism, not a tuning. Verified by diff.
5. **[Sonnet] Simulation config** — new
   `configs/simulation/manager_testing/23_2g8a_switch_round_onehot_self_gnncopar1_contr_gnn_switch.yml`,
   copied from the parent's sim config with only `switch_model`, `output_dir` and
   `figure_name` changed. Verified by diff that no seed, episode count or game parameter
   moves.
6. **[Opus] Train on Raven** — full sync of the branch into
   `AI_REMOTE_DIR='~/autoresearch/switch-round-onehot'` (a commit does not reach the
   cluster — PR #179 note 18), then `scripts/train_cluster.sh ah <config>`. Fetch the
   artifact; record SLURM id, elapsed against the G4 ceiling, and the held-out per-agent
   and joint losses. **Check G1 and G2 here**: every trunk tensor `torch.equal` to the
   base artifact, and the joint cross-entropy lower.
7. **[Opus] Simulate, fetch, evaluate** — `scripts/simulate_cluster.sh` in the same
   isolated dir, `scripts/fetch_cluster.sh`, then `python -m aimanager evaluate` locally.
   Record all 21 rows at full precision, the mean, rows <= 1, the per-switch
   larger-group table against human, and G3.
8. **[Opus] Verdict, log and PR** — results table and notes into this file per §10;
   PR against `auto/contribution-group-vnode` with the maintainer's ruling quoted and
   the verdict read against it.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-16 | (baseline) parent stack, PR #179 group-vnode contributor x PR #171 joint-exodus switch x PR #160 severity-copula punisher | SC 0.9774633985852937, SB 1.1109915195318010, RSA 1.3546243055859613 | 12/21 | 1.0988293946890038 | baseline |

## 4. Notes

1. **The failure signature reproduces exactly, and it is a sign reversal, not an
   offset.** Computed on the parent's own `per_round.parquet` through the evaluation
   suite's `convert.load_human` / `load_sim` (read-only; §8 untouched), mean larger-group
   size over the four rounds following each switch: human 6.44 / 6.20 / 6.04 / 5.92 /
   5.84, monotone; parent 6.39 / 6.15 / 5.99 / 6.30 / 6.16. The first three deltas are
   **-0.05, -0.05, -0.05** — a constant, which is what a single shared slope in `r` buys
   — and then **+0.38, +0.32**. The model does not merely mis-scale the trend, it runs
   the wrong way once the human curve flattens. That is the specific defect a categorical
   round index can fix and a monotone one cannot cheaply express.
2. **Verdict rule raised before implementation.** SC at 0.9775 has no band above it, so
   §2 gate 1 is unreachable on the row the mechanism targets. Escalated to the maintainer
   per §8 rather than quietly substituting a criterion; the ruling is recorded verbatim
   in the declaration and was fixed before any code was written or any number seen.
3. **Training (step 6), SLURM 30266359, 00:04:28, exit 0:0.** Against the base
   model's 00:04:17 (PR #171, SLURM 29870374) and the §5 budget ceiling of
   ~12:50 — **G4 holds** with a 4% increase, which is what four extra input
   dimensions on a 10-unit MLP should cost. Artifact
   `artifacts/artificial_humans/switch_joint_exodus_round_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`;
   `joint_exodus_round_onehot: 5` and `joint_exodus_switch_every: 4` both
   round-trip on load, head `in_features` 23 -> 27 as designed.
4. **G2 holds: the joint fit improves.** Joint-exodus cross-entropy, mean over
   the final 20 epochs across all 5 folds, **1.994373 -> 1.940754**, a fall of
   **0.053619 nats**. The head is recorded on the train split only, so this is
   an in-sample statement; it is nonetheless the statement the mechanism makes
   — five free offsets fit the round-level count distribution better than one
   shared direction in `r`.
5. **G1 FAILS, and the prediction was wrong for a structural reason worth the
   whole campaign's attention.** The guard predicted a bit-identical trunk: the
   head is detached, so the joint loss sends it no gradient, and the head is
   constructed last, so it perturbs no initialisation. **0 of 10 trunk tensors
   are identical**, with max|delta| 0.35-0.95 — a different model, not drift.
   The cause is neither of the two channels the guard considered. The head is
   **40 parameters wider** (1,131 -> 1,171: `Linear(23 -> 10)` becomes
   `Linear(27 -> 10)`), so its own initialisation draws 40 more values from the
   **global** torch RNG. `train.py:381` then draws
   `p_idx = th.randint(0, len(training_mask_pattern), (batch_size,))` from that
   same global stream **once per batch**, to pick each batch's feature-masking
   pattern. A shifted stream means a different masking sequence from the first
   batch onward, hence a different training trajectory. Verified directly
   (scratchpad `verify_rng.py`): with the same seed, trunk parameters are
   identical at init — the "built last" comment is correct — while the next
   `th.randint` draw differs, `[4,5,1,3,0,...]` against `[0,1,4,0,5,...]`.
6. **What that costs, and why the experiment still stands.** The candidate is
   therefore the base model plus the round one-hot **plus one random restart**,
   and the two are not separable by any knob this pipeline has. The restart's
   size is measurable: held-out per-agent log-loss **0.613127 -> 0.618378**
   (+0.005251), accuracy 0.663020 -> 0.662550. The confound is **not specific
   to this experiment** — it applies to every change that alters a module's
   parameter count, including PR #174's own one-hot (`in_features` 23 -> 39)
   and PR #179's virtual node. It is the campaign's standing condition rather
   than something introduced here, and it is the reason a claim should rest on
   a band-width movement rather than a hairline one. Reported rather than
   worked around: no seed was changed, no variant was shopped, and the guard is
   recorded as failed.
