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
| **RCB** (third target, declared by step 0) | **2.0242478714062093** | 2-5 | 0.7041451592840497 | 0.3478552054965939 |
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

- **RCB, third target, declared by step 0's measurement and not by argument (A0).**
  Humans' response to being punished depends on their group's punishment climate
  (interaction -0.5785, CI [-2.2347, -0.1818], excluding zero) and the parent sim gets
  that dependence wrong in sign and magnitude (+0.1399, outside the human CI); the
  virtual node is the object that carries the climate, so the row is declared. Band
  2-5 -> 1-2 requires the resampled per-bin discrepancy below **0.6957104109931878**
  (2 x ceiling) against the parent's 0.7041451592840497. **Stated plainly: that is a
  move of only 0.0084 of raw discrepancy, 0.024 of score, so the row sits 1.2% from
  its band edge and a crossing there is weak evidence beside CG or RCD -- it is a
  declared target so that it *can* be read, not a result that would carry an
  experiment on its own.** Where the real deficit sits: the three upper rate bins hold
  89% of the raw statistic on 53% of the weight (human mean dc climbs 0.89 -> 2.01
  across the bins, the parent's flattens 0.73 -> 0.10).

Gate 1 is met by any of the three target rows leaving its band; gate 2 requires the
21-row mean <= 1.4344450525958377.

**Watch items (reported whatever the verdict, never claimed):** the marginal C block
**CA / CB / CD / CF / CC / CE** (a retrained trunk re-randomises
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

**Punishment-climate interaction (step 0, pre-declaration).** Amendment A0's
measurement, run locally on the same canonical frames as the preflight above
(`convert.load_human`, single copy per game -- 50 games, 9,600 rows -- and
`convert.load_sim` on the parent's `per_round.parquet`, sha256 `0a34f82...`
confirmed, one pairing, 100 episodes), with the same episode-cluster bootstrap
machinery the preflight used at **300 resamples, seed 0** (the preflight's own
count for its human CIs; its main-table CIs used 200). Population: RCB's own,
taken from the frozen `ResponseMetrics._rcb_population` -- punishment > 0,
contribution < 20, `dc = c(t+1) - c(t)` valid: **n = 2,660 human / 4,955 parent**.
`own_rate = p(t) / (20 - c(t))`; `climate` = the leave-one-out mean of the
player's own current-group members' rates in the **same** round, over group-mates
whose rate is defined -- members at contribution 20 (rate undefined, RCC's case)
and no-input rows are dropped from the mean, which empties it for **116 / 2,660 =
4.360902%** of human rows and **170 / 4,955 = 3.430878%** of parent rows; those
rows are dropped from the fit (fitted n = 2,544 / 4,785). Model:
`dc ~ own_rate + climate + own_rate x climate`.

| `dc ~ own_rate + climate + own_rate x climate` | own_rate | climate | **interaction** | R² |
|---|---|---|---|---|
| **human** (n = 2,544, 50 episodes) | 0.36295273816181034 [-0.22099781084615291, 1.4048660681320564] | 0.7483657467520084 [0.23151820237240844, 2.252791474551691] | **-0.5784888432158086** [**-2.2347037710975943, -0.1818489123474606**] | 0.005393401378913865 |
| **parent sim** (n = 4,785, 100 episodes) | -0.4249511850876603 [-0.7199232804156007, -0.23705011836648193] | 0.19390597252588004 [-0.10492511554671208, 0.6543523726468206] | **+0.13987819212243902** [-0.22336228831725252, 0.37652648372334935] | 0.010086447790802588 |

Human intercepts 1.000343879934476 [0.6227436501050642, 1.2413881660886248],
parent 0.7108906010336964 [0.5531551860073343, 0.9056856442867345].

Split by RCB's four rate bins (same fit inside each bin; no bin's interaction CI
excludes zero, so the effect is a pooled one and the bins are reported for
location, not for inference):

| bin | n (human / parent) | own_rate, human | interaction, human | own_rate, parent | interaction, parent |
|---|---|---|---|---|---|
| (0, 0.25] | 1,214 / 2,160 | 2.4564303398445935 [-2.170603400921234, 7.567692081935424] | -14.823696080948478 [-38.350805076241656, 8.23804134961163] | -2.54907092084357 [-5.194791114244655, 0.16661493946396264] | -5.560321028867625 [-11.026173794811175, 1.1713787516553746] |
| (0.25, 0.5] | 653 / 1,087 | 1.9500782533981869 [-3.089272291357104, 9.837485167398535] | 7.9804286616575 [-8.206842880877034, 23.33426300900026] | -3.0012368857769216 [-6.553769653472654, 0.24609650640540182] | -0.43716889234650247 [-9.070850923699156, 8.124689401112727] |
| (0.5, 1] | 435 / 838 | -0.916096169657199 [-6.640969172038876, 4.800813115586446] | -2.1378883059103555 [-12.611789272277967, 5.839843271882242] | -1.4825263525017303 [-4.620182172224934, 1.1480001396160588] | 0.4724967342677841 [-4.7041746149190775, 5.431458086310755] |
| > 1 | 242 / 700 | -1.468810325544845 [-6.542785886912706, -0.38585898448378714] | 0.27751868771400656 [-3.601954517170127, 0.8346490842297946] | -0.5390557789546289 [-1.0966861207960734, -0.24280156479796497] | 0.2767870143705323 [-0.28490633776988655, 0.7893649102461832] |

Context -- RCB's own numerator ingredients, per bin, on the full population
(these reproduce `metrics.csv`'s raw `d` = 0.6887596736481019 exactly when
averaged with RCB's human-frequency weights; the scored numerator 0.7041451592840497
is the 500-repeat resampled mean of the same quantity):

| bin | human mean dc (n) | parent mean dc (n) | abs Δ | weight | share of the raw statistic |
|---|---|---|---|---|---|
| (0, 0.25] | 0.8917609046849758 (1,238) | 0.7284886312973696 (2,243) | 0.1632722733876062 | 0.46541353383458645 | 0.11032748960467824 |
| (0.25, 0.5] | 1.3407738095238095 (672) | 0.5084594835262689 (1,123) | 0.8323143259975406 | 0.25263157894736843 | 0.30528628548120135 |
| (0.5, 1] | 1.6786469344608879 (473) | 0.5362485615650172 (869) | 1.1423983728958707 | 0.17781954887218046 | 0.29493707467612945 |
| > 1 | 2.0144404332129966 (277) | 0.1 (720) | 1.9144404332129965 | 0.10413533834586466 | 0.28944915023799117 |

**Reading.** In humans the punishment response is climate-dependent and the
dependence is *negative*: alone in a calm group a punished player raises their
contribution more the harder they were hit (own_rate +0.363), and that
sensitivity is cancelled as the group's own punishment rate rises (interaction
-0.578, CI excluding zero). The parent sim has neither half -- its own-rate main
effect is *negative* (-0.425, punished harder means moving less) and its
interaction is +0.140, the opposite sign and outside the human CI. The deficit
grows monotonically with the rate bin: humans keep climbing (0.89 -> 2.01 across
the bins) while the parent flattens (0.73 -> 0.10), so the three upper bins carry
89% of RCB's raw discrepancy on 53% of the weight. Caveats stated with the
result: R² is ~0.005 (this is a weak relation in very noisy per-round changes),
the human interaction CI's near endpoint is -0.18 -- close to zero -- and no
single bin's interaction is individually distinguishable from zero.

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

4. *(implementer: Sonnet; retagged from Opus, see note 6)* **Graph gate tests, local with stand-ins and on Raven with real PyG** -- new
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

6a. *(implementer: Sonnet; inserted by the orchestrator at step 6, see note 8)*
   **Make gate (a)'s pre-change fetch degrade to a skip** -- `tests/vnode/test_group_vnode_graph.py`
   (existing). The fetch of the pre-change `graph.py` via `git show 7b440ee^:...`
   happens at import time and raises `CalledProcessError` wherever there is no git
   repo, which takes the whole module down at collection and cost the Raven run all
   ten of its gates. Wrap the fetch so a failure sets a module-level flag and
   `pytest.skip`s only the tests that need the pre-change file, with the reason
   naming the cause; gates (b) to (f) then run unchanged on real PyG. Re-run the
   file on Raven inside the isolated dir and record the per-test outcome.

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
    `[SUCCESS]` only if (CG < 2 or RCD < 2 or RCB < 2) **and** the 21-row mean <=
    1.4344450525958377; otherwise `[FAIL]`. Fill the results table and Notes; open the PR
    with `--base auto/switch-joint-exodus`, body per §9 step 7 (Hypothesis / Results /
    Collateral grouped +/-); delete `~/autoresearch/contribution-group-vnode` when the
    PR closes.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-15 | (baseline) parent stack, PR #171 joint-exodus switch x PR #165 copula contributor x PR #160 severity-copula punisher | CG 4.267640451429015, RCD 2.764919035295771 | 11/21 | 1.3040409569053069 | baseline |

## 4. Notes

1. **Step 0, the A0 measurement.** On RCB's own population (the frozen
   `_rcb_population`; n = 2,660 human / 4,955 parent) the human punishment response
   is climate-dependent -- `dc ~ own_rate + climate + own_rate x climate` gives an
   interaction of **-0.5784888432158086**, 300-resample episode-cluster bootstrap CI
   **[-2.2347037710975943, -0.1818489123474606]**, excluding zero -- while the parent
   sim's is **+0.13987819212243902** [-0.22336228831725252, 0.37652648372334935],
   the opposite sign and outside the human CI. Both halves of the pre-registered rule
   fire, so **RCB is declared a third target**; its threshold and the plain statement
   of how weak a crossing there would be are in §1.
2. **Where RCB's discrepancy actually is.** Reproducing `metrics.csv`'s raw
   `d = 0.6887596736481019` exactly from the per-bin means and RCB's human-frequency
   weights shows the deficit is monotone in the rate bin: human mean dc 0.892 / 1.341 /
   1.679 / 2.014 against the parent's 0.728 / 0.508 / 0.536 / 0.100, so the three upper
   bins carry 89% of the statistic on 53% of the weight. The parent does not merely
   under-react to punishment; it *inverts* the gradient (own-rate main effect -0.425
   against the human +0.363).
3. **Two things the step-0 numbers do not settle.** The relation is weak in absolute
   terms (R² ~ 0.005 human, ~ 0.010 parent) and no single rate bin's interaction CI
   excludes zero -- the effect is a pooled one, and the human CI's near endpoint
   (-0.18) is close to zero; the rule was applied mechanically to the pre-registered
   pooled specification and no variant was run. Separately, §2 step 14's `[SUCCESS]`
   condition still reads "(CG < 2 or RCD < 2)" and was left untouched per this step's
   scope -- the orchestrator has to decide whether the third target enters gate 1
   there, as §1's gate sentence now says it does.

4. **Orchestrator ruling on note 3 (before step 1).** Step 14's gate-1 condition is
   corrected to `(CG < 2 or RCD < 2 or RCB < 2)`: RCB entered by a rule fixed before
   its numbers were visible and §2 makes every declared row a gate-1 row, so excluding
   it from the verdict line would be declaring a target and then not honouring it.
   Two caveats are recorded with it, and bind how any RCB crossing is written up.
   First, RCB's threshold is 0.0084 of raw discrepancy away, so a crossing there is
   weak evidence beside CG or RCD and must be reported as such rather than as the
   headline. Second, and more important, the channel this experiment declared is *not*
   where most of RCB's deficit lives: note 2 shows the deficit is dominated by the
   flattened own-rate gradient (human 0.89 -> 2.01 across the bins, parent 0.73 ->
   0.10), and the virtual node supplies the group's punishment *climate*, not a
   steeper response to the player's own punishment. The declared claim is the
   interaction only. If RCB crosses, the write-up says which of the two moved.
5. **The step-0 fit is also a provenance check.** Reconstructing `metrics.csv`'s raw
   RCB statistic 0.6887596736481019 exactly from independently computed per-bin means
   confirms the bins are weighted by human frequency rather than uniformly (uniform
   would give 1.0131) -- worth knowing for any later reading of a per-bin figure, and
   evidence that this step's population filter matches the frozen suite's.
6. **Step 4 retagged Opus -> Sonnet, orchestrator ruling.** The first step-4
   dispatch died mid-run on an org monthly spend limit, having written nothing
   (tree clean at `0809a81`). The step is re-dispatched to Sonnet: the retag is
   forced by a resource constraint, not a judgement that the step got easier, so
   the orchestrator verifies gate (a) -- the off-flag bit-identity that licenses
   step 13's control comparison -- independently rather than on the subagent's
   report, and the gates are re-run against real PyG on Raven at step 6 either way.
7. **Orchestrator ruling: the squeue gate is about the sync destination, not the
   account.** Step 6's first dispatch stopped without touching Raven because
   `squeue -u certuer` showed four RUNNING jobs. Two are an unrelated `Ekklesia`
   project; two are a *sibling* autoresearch experiment (`~/autoresearch/pna-aggregation`,
   contribution training, submitted ~20:41) in its own isolated dir. Reading
   `scripts/train_cluster.sh` directly: with `AI_REMOTE_DIR` set the `rsync -azP
   --delete` destination is `${REMOTE_PROJECT_DIR}` alone, i.e.
   `~/autoresearch/contribution-group-vnode`, which does not yet exist; the shared
   checkout and every sibling dir lie outside its scope. The hazard the gate exists
   for (PR #173 note 11) is a `--delete` racing a job that needs the files being
   deleted, so the correct predicate is **a job whose working directory is the
   shared checkout or this experiment's own isolated dir** -- not any job on the
   account. Under the account-wide reading no parallel experiment could ever sync,
   which is precisely the situation `AI_REMOTE_DIR` was introduced to support.
   Step 6 re-dispatched with the refined gate; the subagent was right to stop
   rather than reason past a rule it had been given literally.
8. **Orchestrator ruling: gate (a) may skip on Raven, the rest may not.** Step 6
   reported `tests/vnode/test_group_vnode_graph.py` failing at *collection* on
   Raven -- `git show 7b440ee^:src/aimanager/generic/graph.py` exits 128 because
   rsync ships this worktree's `.git` pointer file verbatim (it names a macOS path)
   and `train_cluster.sh` excludes `.git/` regardless. Confirmed not a stand-in
   artefact: `STAND_INS = False` and real PyG imported cleanly before the failure.
   The ruling is that gate (a) does not *need* to run on Raven, while the other
   nine do. Gate (a) is an invariance between the pre-change and post-change code
   evaluated with the **same** message-passing implementation on both sides, so as
   `test_joint_exodus_graph.py`'s docstring puts it, a stand-in cannot manufacture
   a pass; what it compares -- parameter initialisation, RNG consumption, sampled
   values old vs new -- is identical in kind under stand-ins and under real PyG,
   and it was verified bit-exact locally at step 4 and again independently by the
   orchestrator. What is *not* acceptable is the current state, where one import
   error silently costs the Raven run all ten gates including the five that do
   exercise real PyG. Hence step 6a: degrade the fetch to a skip, keep everything
   else. Vendoring a copy of the pre-change file into the repo was rejected -- a
   761-line duplicate that drifts is a worse artefact than a named skip.
9. **Step 7, the candidate trunk (SLURM 30256343, 11m12s, exit 0:0).** 1.37x the
   8:09 base run against the ~24-27 min §5 ceiling, so the budget gate passes and
   the method is affordable -- a real question for an architecture that roughly
   doubles the parameter count and adds a second recurrent path. Provenance: job
   `Work dir` `/u/certuer/autoresearch/contribution-group-vnode/.`, and a
   data-loading traceback in the log originates at that dir's own
   `src/aimanager/generic/data.py:93`, so the job imported this branch's code and
   not the shared checkout's. Artifact sha256
   `714bf40be1b640a2df09395164ef053fce7a60de9c9d4e034b0f2627a6d01013`, loading with
   `group_vnode True`, `group_vnode_hidden None`, `copula_rho 0.0` (nothing stamped
   yet); fetched before any further launcher call and verified as real content
   rather than an LFS stub.
10. **The node uses the group, which M0 provably does not -- the mechanism is
    real.** The `shuffle_feature = agent_group` held-out log-loss delta is
    **+0.03366064935799473** (per fold +0.0444 / +0.0262 / +0.0322 / +0.0354 /
    +0.0300 -- positive in every fold), against M0's
    **+0.0008633987587345349** computed by the identical filter, i.e. **~39x
    larger**. Permuting which group an agent belongs to now costs the model real
    likelihood. This is the architectural claim of §1 confirmed on the trained
    model rather than on a constructed test: M0's group blindness was not a
    modelling choice but a capacity gap, and the virtual node closes it.
11. **It is bought with ~1% of held-out likelihood, which is the gate-2 risk
    named in advance.** Mean held-out log-loss at epoch 574 is
    **2.008562759499938** against M0's **1.9897416823554699**, +0.018821, or
    **0.95% of M0** -- almost exactly the 1% cost PR #173 note 17 found
    propagating to every C row. The signs are mixed rather than uniform (folds 0
    and 4 *improve*, -0.0258 and -0.0028; folds 1, 2, 3 worsen, +0.0215 / +0.0533
    / +0.0480), so this is a variance-vs-capacity trade at a fixed 575-epoch
    budget, not a uniformly worse model. Per amendment A3 nothing was retuned and
    no re-run was made. The honest reading before the simulation: the mechanism is
    installed and active, and the C block is now the thing most likely to fail
    gate 2 -- exactly the anti-correlation §6 warns of, arriving through the
    likelihood rather than through the sampler.
12. **Step 8, pre-sim diagnostic (report-only, gates nothing) -- route, and
    the own-group and punishment-climate channels.** Ran locally on macOS,
    not on Raven and not via `sbatch`. The plan warned that unpickling a
    saved `GraphNetwork` needs the real `torch_geometric.nn.MetaLayer`
    class; that is only true because the switch-joint-exodus test's
    stand-in registers `MetaLayer` at the wrong module path. Both artifacts
    here pickle it at `torch_geometric.nn.models.meta.MetaLayer` (confirmed
    directly: the flat-path stand-in fails `torch.load` with
    `ModuleNotFoundError: torch_geometric.nn.models`); registering that
    fuller nested path with the same CPU-exact `scatter_mean` stand-in
    loads and runs both M0 and the candidate trunk locally, verified before
    anything else was built. `squeue` showed two PENDING jobs, both a
    sibling experiment's own isolated dir (`pna-aggregation`) -- clear under
    note 7's gate -- so `sbatch` was available but unneeded once the local
    route worked. Method: this session's own `preflight.py` (the script
    behind section 1's Hypothesis table, still sitting in the scratchpad)
    supplies the regressors and the episode-cluster bootstrap verbatim, run
    unmodified on `evaluation_suite/convert.py::load_human`'s canonical
    frame -- confirmed exact by first reproducing the Declaration's own
    numbers bit for bit (human own 0.695394 / group-blind peer -0.000526 /
    own-group 0.278345 [0.221769, 0.326813], R² 0.712440, n = 7,137, and the
    punishment-climate 0.164942 [0.105535, 0.243264] against own-punishment
    0.020500 -- both match section 1 to the digits quoted there). The model
    side substitutes E[c(t)] (`predict_independent(sample=False,
    reset_rnn=True)`, teacher-forced on the real sequence via
    `generic/data.py::parse_agent_rounds` / `create_torch_data_new`, reused
    unmodified) for the real `contribution` column as the OLS dependent
    variable, keeping every regressor computed from the real human history
    -- teacher forcing means the model reads real history, so what it is
    scored against has to stay real too. Bootstrap: 200 episode-cluster
    resamples (the 50 human episodes, with replacement), seed 0, exactly
    `preflight.py`'s own setting for the main table. **Item 1, own-group
    weight** (n = 7,137, same population as the Declaration's table): M0 own
    0.762008, group-blind peer 0.122235, **own-group 0.040183** [0.016518,
    0.059822], R² 0.929921; **new trunk** own 0.704746, group-blind peer
    0.082629, **own-group 0.198157** [0.151127, 0.226345], R² 0.926556 --
    against the reproduced human 0.278345. **The trunk closes 66.3% of the
    gap** ((0.198157-0.040183)/(0.278345-0.040183)), the largest single move
    of this coefficient in the record. **Item 2, group punishment-climate
    weight** (n = 7,131): M0 own-punishment 0.040835, **climate 0.041055**
    [0.013464, 0.066244], R² 0.927291; **new trunk** own-punishment
    0.030497, **climate 0.077785** [0.049204, 0.106158], R² 0.925608 --
    against human 0.164942. **The trunk closes 29.6% of the gap**, roughly
    doubling M0's climate weight but leaving most of the deficit.
13. **Step 8, continued -- arrival channel and the teacher-forced pull
    (n = 496 arrival transitions on the exact preflight population, close to
    but not identical to the 513 the plan names -- this population also
    requires `own_lag2`/`own_lag3`/`grp_lag2`/`grp_lag3` valid, one lag
    deeper than the two regressors item 3 actually uses).** **Item 3**, the
    plan's own two-regressor spec (own c(t-1), receiving group's t-2..t-4
    mean): human (reproduced) own **0.465026** [0.337037, 0.564594],
    history **0.405411** [0.243683, 0.520210], R² 0.355001; **M0** own
    **0.715260** [0.679841, 0.749347], history **0.109145** [0.048992,
    0.162129], R² 0.839488 -- matching the plan's cited PR #176 note-4
    numbers (0.707 / 0.111) almost exactly, the strongest available
    validation that this reconstruction matches the prior measurement;
    **new trunk** own **0.614898** [0.561426, 0.668751], history
    **0.329871** [0.261117, 0.380133], R² 0.786069. Own moves 40.1% of the
    way from M0's overshoot toward human ((0.715260-0.614898)/
    (0.715260-0.465026)); history closes 74.5% of its gap
    ((0.329871-0.109145)/(0.405411-0.109145)). The three-term decomposition
    (own / last-round mean / older t-2..t-4 history together) reproduces the
    Declaration's own/last-round/history split almost exactly on the human
    data (0.464717/0.117905/0.279486 against the quoted 0.460/0.118/0.280)
    and shows the same pattern on the models: M0's last-round coefficient is
    slightly *negative* (-0.024637) with essentially all of its (small)
    peer response loaded onto the older-history term, while the trunk's
    last-round coefficient turns positive (0.080577) and its older-history
    term more than doubles M0's (0.243813 vs 0.135458). **Item 4,
    teacher-forced pull** (dc = E[c(t)] - own c(t-1) at the arrival row,
    gap = receiving group's mean at the decision round - own c(t-1),
    pull = Cov(gap, dc)/Var(gap), same 496 events, RCD's own definition):
    human (reproduced) **0.434106** [0.346939, 0.539369] -- matches the
    plan's cited 0.430; **M0 0.186990** [0.151958, 0.225028] -- matches the
    plan's cited 0.186 almost exactly; **new trunk 0.324036** [0.273138,
    0.377459] -- **55.5% of M0's deficit to human closed**
    ((0.324036-0.186990)/(0.434106-0.186990)). **Reading: every one of the
    four comparisons moves substantially toward the human number under an
    exactly-validated methodology (30-75% of each gap closed), led by the
    own-group weight (M0 0.040 -> trunk 0.198, 66% of the gap) and the
    teacher-forced pull (M0 0.187 -> trunk 0.324, 56% of the gap) -- the
    mechanism was learned, not just installed, and by a wider margin than
    any single-mechanism move recorded for this contributor to date.** This
    is a teacher-forced, report-only reading and settles nothing: per §8 the
    verdict is the single closed-loop evaluation against CG, RCD and RCB,
    which still has to convert this conditional response into a
    round-33-vs-round-14 group-level outcome under free-running dynamics
    (copula, other agents' own responses, and 24 rounds of compounding) --
    exactly the gap PR #157's peer-attention and conformity-mixture record
    warns is not guaranteed to survive that conversion.
14. **Step 10, the recalibration (SLURM 30256907, 10m46s, exit 0:0 -- the clean
    path, not `STOP-ESCALATE`).** `rho_hat = 0.0435568043640977`, SE
    0.009443613923707331, 200-resample episode-cluster CI
    [0.024859738353023016, 0.05941458216657012], excluding zero; pairwise LR
    17.04702559141151 on 15090 pairs. Against PR #165's
    **0.06958238086256316** that is **-37.4%**, and #165's point estimate lies
    **above the new CI's upper bound** (as does #173's 0.0752). Round-trip gate
    PASS (max |bias| 0.010098620450984919, tolerance 0.03). Data path identical
    to #165 -- 7457 rows, 1608 cells, 15090 pairs over the 40 single-copy train
    episodes -- and the base loaded at the step-7 sha256, so the movement is the
    trunk, not the pipeline. **The PR #166 lesson is vindicated concretely:
    carrying #165's rho over would have over-dosed the latent by more than half
    again.** Per amendment A1, `rho_hat > 0` takes the stamp branch; step 11
    stamps rho 0.0435568043640977, phi 1.0, switch_every 1.
15. **The trunk absorbed the shared component, and the preflight says so more
    directly than rho does.** The *copula-off* teacher-forced group-spread ratio
    rose **0.7837119164031583 -> 0.8363286876779291** against a human
    0.8472681041593946: the step-7 trunk on its own now reproduces most of the
    one-step group spread that #165 needed a latent to buy, and the copula's
    remaining one-step contribution shrank from +0.0091 to +0.0057. The
    round-thirds rho falls in every third -- 0.0243 / 0.0515 / 0.0645 against
    #165's 0.0345 / 0.0736 / 0.1187 -- with the largest fall (-45.6%) in the
    final third, the lock-in regime the persistent group state is built to carry.
    Consistent with step 8's own-group weight 0.040 -> 0.198. **The caution for
    step 13:** all of this is measured under teacher forcing, where the model
    reads real history. What the copula still has to supply is shared *variance*
    under free-running dynamics compounding over 24 rounds, which is a different
    job; a lower rho is evidence the trunk took over part of that channel, not
    proof the closed-loop spread ratio lands where the preflight sits. #159,
    #153 and #157 all died in exactly that gap.
16. **Orchestrator ruling: phi stays at the pre-declared 1.0, and the tension is
    recorded rather than resolved.** `phi_hat = 0.6182476558394783`, CI
    [0.21163458285772926, 1.2637034657775452]. The CI includes 1, so the
    pre-declared rule (§2 step 10, amendment A2, PR #165's boundary ruling) gives
    `phi_final = 1.0`, and the implementer applied it mechanically without
    hunting for a variant -- correct. It is recorded that this phi_hat, unlike
    #165's 1.1588, does **not** saturate the boundary: the point estimate is
    mean-reverting, and a rule written for a saturating estimate is now being
    applied to one that is not. Two reasons to keep 1.0 anyway. The rule was
    fixed before any number was visible and pre-declaration binds; revising it
    now, having seen 0.618, is precisely the post-hoc discretion amendment A1
    removed from the rho branch. And on the merits the CI is wide enough
    ([0.21, 1.26]) that the data does not distinguish persistence levels, while
    PR #150's arm comparison found the persistent latent wins and the
    fast-reverting one regresses. **A successor should test phi_hat directly on
    this trunk** -- it is the cleanest small-delta follow-up this experiment
    leaves behind, and it is not claimable here.
17. **Step 11, stamping (SLURM 30257600, 17s, exit 0:0).** Stamped artifact sha256
    `379c2557e21922b12408f359d398fa21075ac31b5eeab425a564306233be8f0e`, carrying
    `copula_rho 0.0435568043640977`, `copula_phi 1.0`, `copula_switch_every 1`,
    all three round-tripping on load. **14 tensors compared bit-identically** to
    the step-7 base, the compared modules being `op1`, `op2`, `rnn_n` and
    **`group_vnode_module`** -- the virtual node's own GRU weights are therefore
    verified untouched by stamping, which is what licenses attributing the
    simulation's difference to the mechanism rather than to a stamping side
    effect. The honesty check reports the **7,457** teacher-forced train-split
    rows bit-identical to the base model's probabilities, so the stamp changed
    the sampler and nothing else.
18. **Two process failures at step 11, recorded because they cost four jobs and
    neither was a modelling error.** First: **a local commit does not reach the
    cluster.** Job 30257491 died in 4s on `assert k not in base` -- the isolated
    dir still held the *pre-step-9* stamper, because the dir was last synced at
    step 7 and the step-11 agent had `scp`'d only its own new `.slurm` wrapper.
    Nine commits of source drift, invisible until an assert caught it. **Any step
    that changes repository code after the initial sync must re-sync (or `scp`
    every changed file, not just the new one) before the next remote job** -- and
    the `rsync --delete` rule pushes toward `scp`, which is exactly what makes
    shipping an incomplete set easy. This belongs with PR #171's isolation
    findings. Second, and the orchestrator's own error: having intervened
    directly on jobs the step-11 subagent still owned (it had a live monitor),
    the two of us cancelled and resubmitted underneath each other, killing
    30257219 and 30257556 at 0:00 elapsed. Once an orchestrator touches a
    resource a subagent owns it must take the resource over or leave it alone,
    not interleave; the agent was stopped and the step finished directly. No
    seed, protocol parameter or artifact was affected by either -- the cost was
    wall-clock only.
19. **Scheduling note.** The first stamp submission sat `PENDING` on
    `QOSGrpCpuLimit` -- the *project's* group CPU allocation, saturated by other
    members of the account (214 jobs running on the partition), not by this
    experiment. The wrapper was trimmed from 4 CPUs / 16 GB / 1 h to 1 CPU /
    8 GB / 20 min for a job that loads a 34 KB model, sets three fields and runs
    a 7,457-row forward pass; the pending reason changed to `(Priority)` and it
    scheduled. Resource requests are scheduling, not protocol: no seed, episode
    count, game parameter or model is touched by them, and §8 is not engaged.
