# Group size for the contributor: the marginal per-capita return as a node feature

## Declaration

**Slot:** contribution.

**Parent:** PR #171 (`auto/switch-joint-exodus`, `[SUCCESS]`), the maintainer-designated
frontier, at `6ba366c`. Branch and worktree created from
`origin/auto/switch-joint-exodus`; the PR opens with `--base auto/switch-joint-exodus`.
Read the parent's log (`notes/autoresearch_log/switch-joint-exodus.md`) and PR #169's
(`switch-exodus-count`, on its own branch) before touching anything here — the notes
below correct one of the parent's structural findings and locate the switch slot's
real residual, and the next switch-slot agent needs both.

**Base model:** the parent stack's contributor — the M0 GNN trunk
`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`
(`x_encoding = prev_contribution (numeric, 21), prev_punishment (numeric, 31),
agent_group (onehot, 2)`; no `edge_encoding`; hidden 20, 575 epochs, batch 4,
lr 3e-4, seed 38381) **plus** PR #165's stamped copula (`copula_rho =
0.06958238086256316`, `copula_phi = 1.0`), shipped as
`artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`.
The base model is "trunk + stamped copula", and so is the candidate.

**Evaluation stack (§3 under the parent rule of §9):** the parent's own stack and
config, `configs/simulation/manager_testing/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml`
— this contributor x the joint-exodus GNN switch predictor x the severity-copula
`lin_multinomial` punisher, single pairing `lin_multinomial_copula_self`, seed 42,
100 episodes, 24 rounds, `save_per_round: true`.

**Baseline (the parent's confirmed scores; both §2 gates judged against these).**
Source: `plots/simulation/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch/evaluation/scores.csv`
on this branch.

| quantity | value |
|---|---|
| CE | 1.105386005744275 (band 1-2); numerator 1.6091089668822895, ceiling 1.455698695768135 |
| mean over 21 rows | 1.3040409569053069 |
| gate-2 ceiling (mean x 1.10) | 1.4344450525958376 |
| rows <= 1 | 11/21 |

The parent's log note 20 quotes a ceiling of 1.4182995541296117. That was the ceiling
for *its* run against *its* baseline (1.2893632310269196, PR #165's mean); it is not
ours. Under §9 the parent's confirmed mean becomes the baseline, and the ceiling is
1.4344450525958376.

**Target row: CE alone**, 1.105386005744275, band 1-2 -> requires <= 1: the
resampled numerator 1.6091089668822895 must fall below the ceiling
1.455698695768135, a -9.5% move.

**Not targets, not secondaries.** RCD (2.764919035295771): its degradation from the
independent draw is within-cell, not through the switch mechanism (mix-reweighting
moves the slope 0.2062 -> 0.2002), and its band position is noise — 1.9 sigma on an
episode-bootstrap sd of 0.018/0.029, with the band edge 0.001 from the independent
run's own slope (Notes 2). RCB (2.0242478714062093): 0.024 (1.2%) from the edge,
its deficit is the flat punishment response of the contribution model in both sims
alike (dc by rate bin 0.73/0.51/0.54/0.10 vs human 0.89/1.34/1.68/2.01), and
excluding merged-group or switcher rows recovers none of it — no mechanism claim
from either slot, so declaring it would be shopping.

**Watch items:** CG (expected 4.27 -> ~3, within band — Notes 4), CC (expected
down), SC and SB (expected to improve through the switch trunk's common-good
feature once singletons carry human-like common good, but *not claimed*: the switch
trunk's own conditional at high common good is also off, on n ~ 30), and the
C-block marginals CA/CB/CD/CF (a retrained trunk re-randomises everything the
contributor does; bimodal singletons are human-like, but the retrain wobble on the
R rows is +-0.2-0.36 in score units from slope noise alone).

**Why CE and not a row >= 2 (§6).** §6's ">= 2" heuristic is not satisfied by CE,
and the list of rows >= 2 on this stack is exhausted by measurement, not avoided by
preference:

- **CG 4.267640451429015.** Human spread ratio 0.848, sim 0.738, gap 0.110 against a
  band edge at 0.053. A counterfactual that hands the sim *perfectly* human-like
  contributions in every size-1 and size-8 cell (resampled human cells) reaches
  CG 0.787, gap 0.061, score ~2.3 — no band upgrade; and excluding sizes 1 and 8
  entirely from both sources the gap is still 0.08. The CG deficit lives in sizes
  2-7, where #165's copula is already the mechanism and the rest of the record
  (#151, #157, #158, #159, #163) is exhausted or vetoed.
- **RCD 2.764919035295771.** At the band edge by noise and not switch-owned (above,
  and Notes 2). What RCD measures — sim movers barely adjust on arrival (mean dc
  0.02 partial / 0.92 full-exodus vs human 1.6 / 2.6; PR #116: new-peers beta
  0.014 vs human 0.247) — is a contribution-model deficit this feature makes no
  claim on.
- **RCB 2.0242478714062093.** 0.024 from the edge, no mechanism claim (above).

The target therefore rests on §2's explicit clause — "from 1-2 into <= 1" is a band
upgrade — plus a counterfactual that spans the edge with margin: the row needs
-9.5%; human-like singleton cells deliver -40% (EMD 1.388 -> 0.841 on the parent's
own parquet); singleton polarisation at 100% of the human excess of 0/20 extremes
delivers -19%; at 50% it delivers -10%. This is not an edge wobble — the mechanism's
expected effect is several times the distance to the edge.

### Hypothesis

**The behaviour.** A player's marginal per-capita return from the pool is 1.6 / k
for a group of k: alone you keep your whole multiplied contribution, in a merged
eight you keep a fifth. Humans respond at both extremes. Lone players *polarise*:
while alone they contribute 10.5 on average with 31% at 20 and 23% at 0 (sd 8.0),
against 8.8-10.4 and 17-19% at 20 in groups of 4-5; and it is a within-player
*response*, not selection — on becoming alone a player goes c(t-1) = 7.5 ->
c(t) = 9.7 -> c(t+1) = 10.8 (dc +1.74, n = 53) where a stayer at the same arrival
round moves -0.06. Merged eights *sag*: 8.45 with 7.8% at 20. The simulated
contributor does neither: its singletons sit at 8.0 (12.5% at 20; on becoming alone
9.1 -> 9.0, dc -0.07), its merged eights at 10.2 (18.7% at 20).

**The mechanism absence — structural, not a learning shortfall.** The frontier
contributor is M0: `prev_contribution` numeric, `prev_punishment` numeric,
`agent_group` onehot, and no `edge_encoding`. `GraphNetwork.create_fully_connected`
(`graph.py:751`) and `train.create_fully_connected` (`train.py:74`) both emit every
`i != j` pair **regardless of membership**, so the edge index carries no group
information at all — the `joint_exodus.py` docstring states this for the switch
trunk, and it is verified here to hold identically for the contribution trunk (same
two functions, and M0's edge features are empty). Group identity enters the model
only as a onehot label on each node; there is no feature that encodes anything
about the agent's *own group* — not its size, not its level (M0 has no
`own_grp_prev_mean_contr`). To respond to size the trunk would have to learn the
bilinear equality of two onehots inside the edge MLP and then count it across
seven neighbours, from ~240 alone agent-rounds in 9,600. The contributor has **no
own-group state whatsoever**, and the 10.5-vs-8.0 alone and 8.45-vs-10.2 merged
numbers are the measured consequence.

**Why CE.** CE is the EMD of signed per-(game, round) group-mean differences, and
its human fat tail is singleton cells: 23% of human CE cells contain a singleton
(sim: 23%), with mean |CE| 7.19 there against 4.98 elsewhere (sim: 5.37 against
4.03). The difference is polarisation, not level: a pure level shift of the sim's
singleton contributions (+1.25 … +3.5) makes CE *worse* (+3-4%), while restoring
the human share of 0/20 extremes among singletons at 50% / 100% / human-like cells
gives -10% / -19% / -40%. The claim this experiment makes is therefore
**distributional** — lone players polarise — and the 21-level categorical head is
the right object to represent it.

**Legality — settled by citation.** `notes/baseline_feature_defs.md:6`: "Membership
itself resolves before contributing, so current membership-derived features
(sizes, tenure counters) are legal for both targets." `group_size` ("current group
size") is a documented feature of the linear baseline's current family
(`notes/baseline_feature_defs.md:28`). The contributor already conditions on
current membership (`agent_group`); current size is the same information class.
The feature is not keyed to any metric's bin edge: RPB's size bins (1-3 / 4-5 /
6-8) are the punisher's row, and SC's bins are larger-group sizes of the switch
outcome. Secondary point: PR #169 closed "add group size" for the *switch* slot,
and that closure rested on the sim already reproducing the human *linear* size
slope of the switch decision; for the contribution decision the size response is
measurably absent at both extremes, so the closure does not transfer.

**Planned change.** One node feature, `own_group_size` — the current-round member
count of the agent's group — with numeric encoding `n_levels: 9` (`IntEncoder`
maps an integer v to `linspace(0, 1, 9)[v] = v / 8`, the convention
`round_number` and the joint head's `k / 8` already use). Derived once in pandas for
training (`generic/data.py`) and once in torch for simulation
(`manager/environment.py`), with a parity test closing the two-implementation
hazard, on the precedent of `own_grp_prev_mean_contr`. The training config is a
verbatim copy of `group_switching_contribution_50ep.yml` plus that one feature and
its own `output_dir`. The base model is trunk + stamped copula, so the candidate is
completed by re-running PR #165's calibrate -> stamp recipe on the retrained trunk:
the same `contribution_copula_rho.py`, the same 40-episode single-copy train
split, `phi_final = 1.0` by #165's boundary ruling, and #165's stop-gate carried
over verbatim. Re-calibration is the recipe's derived step, not a second change —
the rho is model-conditional by construction (teacher-forced marginals from the
model being stamped, and the script asserts its input carries rho 0), so carrying
the stale 0.0696 onto a different trunk would be the frankenstein.

**Iteration budget (§5).** The contribution slot's own base run is ~8 min (575
epochs, 5-fold + full; PRs #144/#147 logs), so the ceiling is ~24 min; one extra
scalar feature is ~1.0x. Cluster wall-clock for the whole experiment: ~8 min GPU
training + ~12.5 min CPU calibration (#165's job 29666293) + 14 s stamp +
~2m20s per simulation, ~25 min in total — inside the maintainer's short-iteration
preference, and every step is a same-day step.

**Probability, stated before anything runs.** Gate 1 ~0.45-0.5: the trunk must learn
at least half the human singleton polarisation from ~240 alone agent-rounds (x2 in
the flip-doubled data) given a first-class size input; the `prev_contribution`
anchor then persists whatever polarisation starts. Gate 2 given gate 1 ~0.85: the
expected movements (CE, CC, CG, likely SC/SB) are favourable and the risk is retrain
wobble on the R rows, ~0.03 on the mean against 0.13 of headroom.

## Plan

Validated by the orchestrator against §2 (target), §5 (legality) and §8 (frozen
surface) before any step runs. One feature enters the model; the copula steps
re-run the parent stack's own recipe unchanged. Nothing under
`src/aimanager/evaluation_suite/`, `notes/evaluation_metric_defs.md`,
`notes/eval_scoring_schema.md` or `experiments/` is touched; the simulation
protocol, seeds and episode count are the parent's. Wall-clock per step is stated
against the contribution slot's ~24 min training ceiling.

1. **Train-side feature** — `src/aimanager/generic/data.py`, `parse_agent_rounds`
   (existing) and `get_default_values` / `create_torch_data_new` (existing). In
   `parse_agent_rounds`, after `agent_group` is set and before the
   `own_grp_prev_mean_contr` block, add
   `df["own_group_size"] = df.groupby(["episode_id", "round_number", "group_id"])["player_id"].transform("size")`
   — the count of *members* at the current round, timeouts and no-input rows
   included (membership is not validity; a member who timed out is still in the
   group). Add `"own_group_size": 4` to `get_default_values` (groups are 4-4 before
   the first switch, so round 0 and absent cells are 4 by construction) and
   `"own_group_size": th.int64` to the `data_names` dtype map. Nothing else in the
   file changes; the human data's singletons must come out at exactly the 44
   valid singleton decision rows / ~240 alone agent-rounds measured in the
   diagnosis. *[correctness-critical — implementer: Opus]*

2. **Sim-side feature, refreshed in the right place** — `src/aimanager/manager/environment.py`.
   Add `update_own_group_size(self)`: `self.state["own_group_size"] =
   self.agent_group_mask.sum(dim=1, keepdim=...).gather(1, self.agent_groups)`
   (the per-group member count, gathered to each agent; shape `(batch, n_agents,
   1)`, int64, matching the other agent-indexed state tensors). Call it at the top
   of `update_contribution`, next to `update_own_grp_prev_mean_contr()`, so it runs
   **after** `apply_switch` (which `step` calls first on arrival rounds and which
   already refreshes `agent_groups` / `state["agent_group"]`) and **before** the
   contributor's forward pass — the same ordering that keeps
   `own_grp_prev_mean_contr` keyed to the *new* group. Initialise the key in
   `reset_state` from the initial `agent_groups` (4 everywhere). Note the hazard
   PR #169 recorded (note 3b): `Environment.default_values` is read from the
   *contribution* artifact, so the new key reaches `reset_state`'s `prev_` loop —
   here that is exactly the artifact carrying it, so no `KeyError` path exists,
   but the step must confirm it. *[correctness-critical — implementer: Opus]*

3. **Train/sim parity test** — new `src/aimanager/tests/test_group_size_train_sim_parity.py`,
   modelled on `test_joint_exodus_train_sim_parity.py` (same PyG stand-in pattern,
   runs locally with plain pytest). On a synthetic membership trajectory with a
   mid-episode switch that creates a singleton and a seven, assert that the
   `own_group_size` column `parse_agent_rounds` derives equals, agent for agent and
   round for round, the `state["own_group_size"]` a real `ArtificialHumanEnv`
   holds at contribution time — including the round the switch materialises. Also
   assert the round-0 default is 4 on both sides and that the encoder maps sizes
   1..8 to 1/8..1. *[correctness-critical — implementer: Opus]*

4. **Training config** — new
   `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_size.yml`,
   a verbatim copy of `group_switching_contribution_50ep.yml` (575 epochs, batch 4,
   lr 3e-4, hidden 20, 5-fold, seed 38381, `shuffle_features` unchanged) with
   exactly three edits: append `- {name: own_group_size, n_levels: 9, encoding:
   numeric}` to `model_args.x_encoding`; `output_dir:
   artifacts/artificial_humans/group_switching_contribution_50ep_group_size`; a
   `description` naming this experiment. Labels unchanged so the artifact filename
   pattern `architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` is preserved.
   *[mechanical — implementer: Sonnet]*

5. **Train on Raven** — `scripts/train_cluster.sh ah <step-4 config>` with
   `AI_REMOTE_DIR='~/autoresearch/contribution-group-size'` (check `squeue` for
   PENDING jobs before any sync). Record SLURM job id, elapsed (expected ~8 min,
   ~1.0x the ~8 min base; ceiling ~24 min), the in-job artifact sha256, and the
   per-fold held-out log-loss against the M0 base (1.9892 best-mean, PR #113) so a
   worse marginal fit is visible before any score exists. *[mechanical — implementer: Sonnet]*

6. **Pre-sim diagnostic, report-only** — scratch script (not committed to `src/`),
   run on Raven from the isolated dir (needs PyG). Teacher-forced on the human
   single-copy data: P(c = 20 | own_group_size = 1) and P(c = 0 | own_group_size =
   1) against the human 0.31 / 0.23, and the mean predicted contribution by group
   size 1..8 against the human 10.5 / 9.9 / 8.9 / 10.4 / 10.4 / 9.5 / 9.4 / 8.5
   (rounds >= 4). **Whatever it says, the simulation runs and the verdict comes from
   the single evaluation per §2/§6**; this step can never gate or replace the sim.
   It exists so the results table can say whether a failure was "not learned" or
   "learned but did not carry through the closed loop". *[mechanical — implementer: Sonnet]*

7. **Calibrate the copula on the new trunk — with #165's stop-gate** — copy
   `scripts/artificial_humans/calibrate_copula.slurm` to
   `calibrate_copula_group_size.slurm` with `BASE` pointing at the step-5 artifact
   and `PARAMS` at
   `artifacts/artificial_humans/group_switching_contribution_50ep_group_size_herding_copula/calibration/copula_params.json`;
   `contribution_copula_rho.py` itself is unchanged (`--model`, `--roundtrip`,
   `--preflight`, `--write-params`). `sbatch` from the isolated dir; expected
   ~12.5 min on 8 CPUs (#165 job 29666293). Record rho, its 200-episode-cluster
   bootstrap CI, phi_hat and its CI, the round-trip gate result. **STOP-GATE,
   verbatim from #165: if the calibrated rho's CI includes 0, the experiment
   terminates here as a calibration-only `[FAIL]`** — no artifact is stamped, no
   simulation runs, the results table records the calibration numbers, and the PR
   opens in that state. Otherwise `phi_final = 1.0` per #165's boundary ruling
   (persistence-class decision; the estimated phi stays in the JSON unaltered) and
   the plan continues. *[correctness-critical — implementer: Opus]*

8. **Stamp** — copy `scripts/artificial_humans/stamp_copula.slurm` to
   `stamp_copula_group_size.slurm` invoking `make_contribution_copula_artifact.py`
   with `--params` (step 7), `--base` (step 5) and `--out
   artifacts/artificial_humans/group_switching_contribution_50ep_group_size_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`;
   the stamper's teacher-forced honesty check runs as in #165 (~14 s, job
   29667403 precedent). Verify the three stamped fields round-trip
   (`copula_rho` = step-7 rho, `copula_phi` = 1.0, `copula_switch_every`) and that
   every weight tensor is `torch.equal` to the step-5 artifact — the stamp changes
   sampling only. Fetch and commit the params JSON, the `.copula.json` sidecar and
   the stamped artifact (LFS). *[mechanical — implementer: Sonnet]*

9. **Simulation config** — new
   `configs/simulation/manager_testing/23_2g8a_contr_group_size_self_gnncopar1_contr_gnn_switch.yml`,
   a byte-copy of the parent's
   `23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml` with exactly
   three edits: `contribution_model` -> the step-8 artifact, `output_dir` ->
   `plots/simulation/23_2g8a_contr_group_size_self_gnncopar1_contr_gnn_switch`,
   `figure_name` likewise. The slug sits before `_self_` so
   `evaluation_sweep.py`'s `DIR_PATTERN` still parses contr/switch, as the parent
   did. Switch model, valid model, punisher, pairing list, seed, episodes, rounds
   unchanged. *[mechanical — implementer: Sonnet]*

10. **Baseline control, then the candidate** — two simulations from the isolated
    dir via `scripts/simulate_cluster.sh` (~2m20s each). First re-run the parent's
    own config unchanged and require a bit-identical `per_round.parquet` (sha256
    `0a34f8280bccb98a75fe002eb3669827358117ce56c44f7c10268f312904b7ab`, the parent's
    reproduced candidate, log note 25) — the licence to compare anything, and the
    proof that steps 1-2 did not alter the environment's behaviour for a model
    that does not read the new key. Then the step-9 candidate. Fetch both with
    `scripts/fetch_cluster.sh` from the isolated dir. *[mechanical — implementer: Sonnet]*

11. **Evaluate and rule** — `python -m aimanager evaluate <step-9 config>`, locally.
    One simulation, one evaluation, no second stage (§3). Record the results row
    (CE, rows <= 1, mean, and the watch items CG / CC / SC / SB / CA / CB / CD / CF
    explicitly). `[SUCCESS]` only if CE < 1.0 *and* the 21-row mean <=
    1.4344450525958376; otherwise `[FAIL]`. Open the PR with `--base
    auto/switch-joint-exodus`, body per §9 step 7 (hypothesis, results table,
    collateral grouped +/-), then delete the remote dir
    `~/autoresearch/contribution-group-size` when the PR closes. *[correctness-critical — implementer: Opus]*

## Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## Notes

1. **The overshoot the parent recorded is a wrong conditional, not state drift, and
   it sits in the singleton cell — not in the founding exodus.** Per-group-cell
   full-exodus rate: human 0.1079 (complete pairs), parent 0.1424, independent
   draw 0.0869. Shapley decomposition of parent minus human over cells keyed by
   k: conditional **+0.0387**, state **-0.0043**; keyed by (k, round): +0.0453 /
   -0.0108 — the sim's state distribution actually *reduces* full exoduses (more
   time at k = 0/8). P(m = k | k), human / parent / independent: k = 1 **0.161
   (n = 31) / 0.422 (n = 90) / 0.350**; k = 2 0.147 / 0.245 / 0.162; k = 3 0.200 /
   0.171 / 0.098; k = 4 **0.177 / 0.181 / 0.081**; k >= 5 ~0 everywhere. At k = 4,
   where the parent's hypothesis lives, the head is *calibrated*; k = 1 alone
   carries ~85% of the smaller-group conditional excess (~0.047 of +0.055). The
   "late-game reversal" is the same phenomenon, not one the mechanism cannot
   express: the merged state is not sticky (P(m = 0 | k = 8) 0.263 human vs 0.287
   parent), the late rise is *entry* — pooled P(7 -> 8) **0.222 vs human 0.082**,
   P(6 -> 8) 0.167 vs 0.018, P(enter 8 | not 8) at rounds 15/19 0.146/0.188 vs
   0.085/0.067 — i.e. singletons and pairs leaving late, which is also SB's late
   excess. Root cause: the sim's singletons are fed the *lowest* per-capita common
   good of any group size (10.5; human singletons the *highest*, 17.3) because the
   contributor puts them at 8.0 instead of 10.5, so the switch trunk sees a
   low-common-good lone player facing a higher other group (71% of sim singleton
   cells vs 33% human) and correctly says "leave". This is the diagnosis this
   experiment acts on.

2. **PR #171's structural finding is not supported by the data, independently of
   whether this experiment succeeds.** The parent wrote that "the assortativity
   defence and the SC lever are in direct tension, because the same event supplies
   both" — that RCD degraded because full-exodus events (where conditional
   Bernoulli selects nobody) rose from 0.1766 to 0.3043 of movers. Measured on the
   two parquets: reweighting the parent's switch events to the independent run's
   (leaving-size x full-exodus) mix moves the pull slope **0.2062 -> 0.2002** (its
   own value), and the independent run to the parent's mix **0.2710 -> 0.2803**.
   The mix explains nothing; the change is within-cell, and given m the
   conditional-Bernoulli draw *is* the independent Bernoulli conditioned on its
   sum, so within-cell change can only come from the contribution dynamics in
   differently composed episodes — or noise: episode-bootstrap sd of the slope is
   0.018 (parent) / 0.029 (independent), making the 0.065 difference ~1.9 sigma,
   and the RCD band edge (slope 0.270) sits 0.001 from the independent run's own
   0.271. Furthermore the human full-exodus mover share is **0.260** — the parent's
   0.304 is *closer* to human than the independent 0.177 — and the human pull is
   *higher* on full-exodus events (0.66) than on partial ones (0.36), so more
   full-exodus events should raise RCD, not lower it. RCD measures the
   contributor's (non-)adjustment on arrival, not who the switch model selects. The
   parent's log file is not edited (§8); this note is the correction.

3. **Candidate 2, deferred — not dropped: the small-minority holdout (switch
   slot).** Humans in groups of 1-2 hold out (leave rate 0.25 / 0.32 vs 0.41 /
   0.42 at sizes 3-4, i.e. rising with size within the minority) where the parent's
   sim leaves at 0.42 / 0.43 / 0.39 / 0.40 regardless of size; the effect is
   non-monotone in size, a different claim from the linear size slope PR #169
   tested and closed. Its **preflight**: run the parent's joint head teacher-forced
   on the human (1,7) cells and read P(m_s = 1 | k_s = 1) — if ~0.16-0.25 the head
   is calibrated and only the state is wrong, so the candidate has no content; if
   ~0.4 the head is miscalibrated and the candidate stands. **Why it must follow
   this experiment:** the sim's singletons currently carry the lowest per-capita
   common good of any group size, so a switch-slot fix would learn on human
   singletons (high common good, stay at 0.143 when cg >= 12, n = 28) and be
   applied to fake ones (the sim's leave at 0.375 / 0.448 above / below cg 12,
   n = 32 / 58); the state has to be right first. **Edge flag:** its natural rows
   are SB (1.06526788518747, 0.0026 raw from <= 1) and SC (1.147392663266986,
   0.0035 EMD from <= 1) — either crossing would look like shopping whatever the
   mechanism, so its declaration would need to name the calibration statistics it
   expects to move (P(m = 1 | k = 1), P(7 -> 8), late minority leave rate) alongside
   the row.

4. **What the size feature can and cannot buy, measured on the parent's parquet
   before anything was built.** CE: human-like singleton cells -40% (1.388 ->
   0.841), singleton 0/20 polarisation at 100% / 50% of the human excess -19% /
   -10%; a pure level shift +3-4% *worse* — the claim is distributional. CG: with
   human-like size-1 cells alone 0.738 -> 0.766 (score ~3.1); size-1 and size-8
   together 0.787 (score ~2.3); no band upgrade available, hence a watch item. CC
   EMD 0.409 -> 0.345 under the same counterfactual. Size 8 does not enter CE at
   all (CE drops rounds with an empty group), so the merged-eight sag is a CG/CC
   watch effect only.
