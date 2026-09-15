# Arrival tenure for the contributor: newcomers let go of their own anchor

## 1. Declaration

**Slot:** contribution.

**Parent:** PR #171 (`auto/switch-joint-exodus`, `[SUCCESS]`), the maintainer-designated
frontier, at `6ba366c`. Branch `auto/contribution-arrival-tenure` and worktree
`.claude/worktrees/arrival-tenure` created from `origin/auto/switch-joint-exodus`; the
PR opens with `--base auto/switch-joint-exodus`. Read the parent's log
(`notes/autoresearch_log/switch-joint-exodus.md`) and PR #173's
(`contribution-group-size`, on its own branch — the one prior contribution experiment
stacked on this parent; its notes 1-2 correct the parent's RCD attribution and its
step 7b fixes the copula stamper this recipe reuses) before touching anything here.

**Base model:** the parent stack's contributor — the M0 GNN trunk
`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`
(`x_encoding = prev_contribution (numeric, 21), prev_punishment (numeric, 31),
agent_group (onehot, 2)`; no `edge_encoding`; hidden 20, 575 epochs, batch 4, lr 3e-4,
seed 38381, flip-doubled data) **plus** PR #165's stamped copula (`copula_rho =
0.06958238086256316`, `copula_phi = 1.0`, `copula_switch_every = 1`), shipped as
`artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`.
The base is "trunk + stamped copula"; so is the candidate.

**Evaluation stack (§3 under the parent rule of §9):** the parent's own config
`configs/simulation/manager_testing/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml`
— this contributor x the joint-exodus GNN switch predictor
(`artifacts/artificial_humans/switch_joint_exodus/...`) x the severity-copula
`lin_multinomial` punisher (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`),
single pairing `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
`save_per_round: true`.

**Baseline (the parent's confirmed scores; both §2 gates are judged against these).**
Source: `plots/simulation/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch/evaluation/scores.csv`
on this branch (`per_round.parquet` sha256
`0a34f8280bccb98a75fe002eb3669827358117ce56c44f7c10268f312904b7ab`).

| quantity | value |
|---|---|
| **RCD** (primary target) | **2.764919035295771** (band 2-5); numerator 0.22148846810729983, noise ceiling 0.08010667411228792 |
| RCA (secondary target) | 1.5746350718021775 (band 1-2); numerator 0.4408869410614397, ceiling 0.2799930910702011 |
| mean over 21 rows | 1.3040409569053069 |
| gate-2 ceiling (mean x 1.10) | 1.4344450525958377 |
| rows <= 1 | 11/21 (context only) |

**Target rows.** **RCD**, primary: band 2-5 -> 1-2 or better requires the pull-slope
gap below 0.1602133482245758, i.e. the simulated slope above **0.2698** (human
0.4302, parent 0.2062). **RCA**, secondary: the mechanism makes a direct claim on
RCA's `switched` stratum (the same dc as RCD's), so RCA is declared; a crossing into
<= 1 would need the other three strata to hold and is not expected — it is declared
because the claim exists, not because the crossing is likely. Gate 1 is met by either
row leaving its band; gate 2 requires the 21-row mean <= 1.4344450525958377.

**Watch items (reported whatever the verdict, never claimed):** RCB and RCC (no
mechanism claim; retrain wobble), the marginal C block CA/CB/CD/CF (a retrained trunk
re-randomises everything the contributor does), CG (arrivals converging on the
receiving group should tighten group means — expected mild improvement, not claimed),
and SA/SB/SC through the common-good channel PR #173 note 18 identified — SC is the
parent's success row and the one most exposed to any contribution-slot change.

### Hypothesis

**The behaviour.** Newcomers let go of their own anchor. Regressing a human's
contribution on their own previous contribution and their current group-mates'
previous-round mean gives own-weight **0.719** / peer-weight **0.250** for stayers
(n = 8,037) and **0.460** / **0.280** at the arrival round (n = 523): on arriving in a
new group a human keeps less than half of their own level, while the peer weight
barely moves. It shows up in the raw transitions too — arrivals repeat their previous
contribution 25.8% of the time against 44.8% for stayers, and make a move of five or
more points 35.4% of the time against 14.2%. It is within-player, not selection: the
same switchers are as sticky as never-switchers once settled (repeat 0.511 at tenure
>= 4 after a switch vs 0.516 for players who never switch), and they settle gradually
(repeat 0.258 at arrival -> 0.365 over tenure 1-3 -> 0.511 from tenure 4).

**Why this is RCD.** RCD regresses the switcher's change dc = c(n+1) - c(n) on the gap
between the receiving group's mean and c(n). Writing the arrival conditional as
c(n+1) = own * c(n) + peer * m + k, the RCD slope is
[(1 - own) * var(c) + peer * var(m)] / var(gap) — with the human variances
(sd(c) ~ 6.4, sd(m) ~ 4, sd(gap) 7.6) the human weights give ~0.42 and the sim's
give ~0.24. **The pull is carried by the anchor release (1 - own), not by the peer
weight**: raising peer from 0.11 to 0.28 alone adds ~0.05 to the slope; releasing own
from 0.71 to 0.46 adds ~0.18. Every prior RCD success in the record (#154, #156,
#163, #165, #170) moved RCD by lowering the effective self-anchor somewhere — the
schedsamp family did it everywhere and paid in RCA. Humans do it at arrival only.

**The deficit is in the conditional, not in closed-loop drift.** The base trunk run
teacher-forced on the human data (single copy, PyG stand-ins, CPU) predicts a pull of
**0.186** on the human switch events (human 0.430) — the closed loop's 0.206 (parent)
and 0.271 (independent draw, PR #165's stack) simply reproduce it. On the arrival rows
the model's expected-value weights are own **0.707** / peer **0.111** (human 0.460 /
0.280), its predicted repeat probability 0.329 (human 0.258), its predicted P(|dc| >= 5)
0.201 (human 0.354), and its NLL on that stratum is **2.338** against 1.748 on
stayers — the arrival rows are the worst-fit cells the model has. The parent's
simulated arrivals show the same non-response: own 0.767 / peer 0.166 against 0.797 /
0.147 for its stayers.

**Why the trunk cannot express it.** M0 sees `agent_group` at every round, so an
arrival is in principle recoverable as "my label differs from last round's" through
the GRU. Nothing else marks it, the event sits on 5.5% of training rows, and the
measured arrival response (0.767 vs 0.797) says the GRU did not find it. The linear
contributors, whose feature grids carry `rounds_since_switch` and `switched_last_choice`
(`configs/training/baselines/contribution/*.yml`, component set B5), sit at RCD
0.7-2.0 across the sweep against the GNN's 2.7-2.9.

**Planned change.** One node feature, `rounds_since_arrival`: the number of rounds
since the agent last arrived in a new group, **0 on the arrival round**, then 1, 2, 3,
**capped at 4**, with 4 also for an agent who has not arrived anywhere yet (rounds
0-3 are static by design, so the earliest arrival is round 4). Encoded **onehot,
`n_levels: 5`** — the human profile is a step at arrival, a plateau over tenure 1-3
(own 0.686 / 0.651 / 0.662, repeat 0.36-0.37) and a second step at 4 (own 0.752,
repeat 0.484); a bool would force tenure 1-3 (2,122 single-copy rows, 12 points less
sticky than settled) onto the settled cell, and a numeric ramp would impose a slope
the data does not show — PR #173's own encoding lesson. Derived once in pandas for
training (`generic/data.py`) and once in torch for the simulation
(`manager/environment.py`), with a parity test closing the two-implementation hazard,
on the precedent of `own_grp_prev_mean_contr` and PR #173's `own_group_size`. The
training config is a verbatim copy of `group_switching_contribution_50ep.yml` plus
this feature. The base is trunk + stamped copula, so the candidate is completed by
PR #165's calibrate -> stamp recipe re-run on the retrained trunk (rho is
model-conditional; PR #173 re-ran it for the same reason, and its stamper precondition
fix is ported here). The MLE sets the dose: **no knob, one arm, one evaluation.**

**Legality (§5).** `notes/baseline_feature_defs.md`: "Membership itself resolves
before contributing, so current membership-derived features (sizes, tenure counters)
are legal for both targets." The linear family already defines and trains on
`rounds_since_switch` (`scripts/baselines/handcrafted_grid.py::_rounds_since_switch`,
"tenure; 0 on the arrival round"). The feature is not keyed to any bin edge or stratum
boundary: RCD has no strata, and RCA's round types are labelled at the *decision*
round from the switch label, not from tenure. RCD and RCA do condition on the switch
event — the orchestrator is right to press on this — and the answer is that the
feature encodes the player's own tenure, information they trivially hold, and the
behaviour it captures is visible in raw transitions without any metric (the 0.72 ->
0.46 anchor release, the 0.45 -> 0.26 repeat rate, the within-player settling). The
cap comes from the human profile flattening at 4 and the four-round decision cadence,
not from any metric definition.

**Why not the exposure route the steer named.** The maintainer's steer reads PR #163
as "RCD is bought by free-running exposure"; the measurements above say exposure
bought RCD *by accident of shape*. Scheduled sampling noises the self-anchor on every
row, so it releases the anchor for stayers too — and the sim's stayers are **already
less sticky than humans** (overall repeat 0.354 vs 0.403; settled 0.393 vs 0.484),
which is exactly the RCA/RCB bill #163 paid dose-linearly (RCA 2.08 -> 3.32 -> 5.91).
A budget-legal one-step approximation (candidate A below) would inherit that shape.
The human release is confined to arrivals, and the teacher-forced diagnostic locates
the whole deficit in the arrival conditional, so the targeted feature dominates the
blunt curriculum on both gates. Recorded so the steer's premise is answered with a
number rather than sidestepped.

**Iteration budget (§5).** Recent plain contribution trains on Raven: 07:57 (job
29891768) and 08:52 (job 29898154), so the 3x ceiling is ~24-27 min; one extra
5-wide onehot input is ~1.0x (~8-9 min). Whole experiment: ~9 min GPU training +
~11-12 min CPU calibration + ~30 s stamp + 2 x ~2.5 min simulations, ~30 min of
cluster wall-clock; every step is a same-day step.

**Probability, stated before anything runs.** Gate 1 ~0.6: the deficit is a
mis-fit conditional on the worst-fit stratum, the feature makes it linearly
accessible, and RCD needs only ~30% of the human anchor release to carry into the
closed loop (slope 0.206 -> 0.270; episode-bootstrap sd of the slope 0.02-0.03, so
the required move is ~3 sigma of noise — PR #173 note 2). Gate 2 given gate 1 ~0.8:
the feature should *gain* held-out likelihood (the arrival cells are where the model
is worst), so the C-block tax that killed #154/#156/#158/#173 is not expected; the
risks are retrain wobble (~0.03-0.05 on the mean against 0.13 of headroom) and the
S-row channel.

## 2. Plan

To be validated by the orchestrator against §2 (targets), §5 (legality) and §8
(frozen surface) before any step runs; implementer tags are the orchestrator's. One
feature enters the model; the copula steps re-run the parent stack's own recipe
unchanged. Nothing under `src/aimanager/evaluation_suite/`,
`notes/evaluation_metric_defs.md`, `notes/eval_scoring_schema.md` or `experiments/`
is touched; simulation protocol, seeds and episode count are the parent's. Every
remote call sets `AI_REMOTE_DIR='~/autoresearch/contribution-arrival-tenure'` and
checks `squeue` for PENDING jobs first. **`rsync --delete` hazard (PR #173 note 11):**
the launchers sync `artifacts/` (excluding only `artifacts/manager/`) with `--delete`,
so every artifact a remote step produces must be fetched into this worktree *before*
the next launcher call, or it is deleted; single files (slurm wrappers) go by `scp`.

**Orchestrator validation (2026-09-15).** Validated and released to implementation.
§2: RCD (2.764919035295771) primary and RCA (1.5746350718021775) secondary are
pre-declared; gate-2 ceiling 1.4344450525958377. §5 legality: `rounds_since_arrival`
is a port of an existing, documented feature — `notes/baseline_feature_defs.md` states
that current membership-derived features including tenure counters are legal for both
targets, and line 120 already defines the linear family's `rounds_since_switch`
("0 at the arrival round"). It is a game-observable fact the player holds at decision
time, not a quantity keyed to a metric definition or a stratum boundary. §5 budget:
~1.0x against a ~27 min ceiling. §8: nothing frozen is touched. §3: the parent's
config, one simulation, one evaluation, with a bit-identical control.

**§4 ruling on step 7.** Porting PR #173's copula-stamper precondition fix is shared
code and would normally be its own experiment. Admitted here as a precondition, on
#173's precedent: the diff is assertion logic only (`NEUTRAL_FIELDS`,
`assert_only_copula_fields_changed`), it cannot alter a stamped value, and step 9
cannot run without it — any trunk trained on this branch carries the three copula keys
at neutral defaults, which the parent's stamper refuses. Recorded, not waved through.

**Steer.** The maintainer's steer named the exposure route (PR #163). This experiment
declines it on the measurement in note 4 — the pull deficit is already present
teacher-forced (0.186 vs human 0.430), so exposure cannot be what is missing — and the
maintainer confirmed the arrival-tenure route over the fully-specified exposure
runner-up (candidate A) before any step ran.

1. **Train-side feature** `[Opus]` — `src/aimanager/generic/data.py`, `parse_agent_rounds`
   (existing), after the `does_switch` / `switch_valid` block and before the
   `own_grp_prev_mean_contr` block (the frame is already sorted by episode, player,
   round there and `group_id` is still present): `arrived = (round_number > 0) &
   (group_id != by_player["group_id"].shift(1))`; `last = round_number.where(arrived)`
   forward-filled within `(episode_id, player_id)`; `df["rounds_since_arrival"] =
   (round_number - last).fillna(4).clip(upper=4).astype(int)`. `get_default_values`
   (existing): add `"rounds_since_arrival": 4`. `create_torch_data_new` (existing)
   `data_names`: add `"rounds_since_arrival": th.int64`. Nothing else in the file
   changes. Acceptance, on the human single-copy data (50 games, 9,600 rows): tenure
   0 / 1 / 2 / 3 / 4 counts **572 / 572 / 572 / 572 / 7,312** over all rows, and
   **548 / 548 / 558 / 560 / 6,582** over rows with valid current and previous
   contribution and round > 0; every one of the 572 arrival rows follows a row where
   the same player's `group_id` differed; rounds 0-3 are 4 everywhere. (The
   Declaration's n = 523 arrivals is the same set further restricted to rows with at
   least one valid group-mate, the regression's requirement.)

2. **Sim-side feature** `[Opus]` — `src/aimanager/manager/environment.py`. `reset_state`
   (existing): add `"rounds_since_arrival": th.full(size, 4, dtype=th.int64)` to the
   state dict (the `prev_` comprehension then also creates a harmless
   `prev_rounds_since_arrival`). New method `update_rounds_since_arrival(self)`: if
   `self.round_number[0, 0, 0] == 0`, fill 4; else `arrived = self.state["agent_group"]
   != self.state["prev_agent_group"]` and `self.state["rounds_since_arrival"] =
   th.where(arrived, 0, (self.state["rounds_since_arrival"] + 1).clamp(max=4))`. Call
   it first thing in `update_contribution` (existing), before
   `update_own_grp_prev_mean_contr()`. Why the ordering is right: `step` rolls every
   `prev_*` key (`prev_agent_group` = the pre-switch label) *then* calls
   `apply_switch` *then* `update_contribution`, so the arrival is detected exactly at
   s+1 with the post-arrival `agent_group`, matching `data.py`'s shift. Hazards to
   respect: the round-0 branch is mandatory because `reset_state` fills
   `prev_agent_group` with the default 0, which would mark every group-1 agent as
   "arrived" at round 0; write through `self.state[...]` (the class's `__setattr__`
   redirects attribute writes into the state); the parent's own model does not name
   the key, so the step-11 control must reproduce its parquet bit-identically.

3. **Train/sim parity test** `[Opus]` — new
   `src/aimanager/tests/test_arrival_tenure_train_sim_parity.py`, modelled on
   `origin/auto/contribution-group-size:src/aimanager/tests/test_group_size_train_sim_parity.py`
   (PyG stand-ins, runs locally with plain pytest). Synthetic membership over enough
   rounds to cover: a switch at a decision round (tenure 0 at s+1, then 1, 2, 3), a
   second switch by the same agent (reset to 0), an agent who never switches (4
   throughout), and tenure reaching and holding the cap. Assert the pandas column
   equals the env's `state["rounds_since_arrival"]` at contribution time, agent by
   agent and round by round, including the arrival round; round 0 is 4 on both
   sides; `IntEncoder(onehot, 5)` maps 0..4 to the five unit vectors.

2a. **Step 2 revision, forced by step 3b** `[Sonnet]` — `reset_state`'s `prev_state`
   comprehension only creates `prev_<k>` for keys that appear in the model's
   `default_values`, so `update_rounds_since_arrival`'s unconditional read of
   `state["prev_agent_group"]` raises `KeyError` for any env whose `default_values`
   omits `agent_group`. `generic/data.py`'s `get_default_values` does carry
   `"agent_group": 0`, so every model trained through the pipeline — including the
   step-11 control — is unaffected and the simulation path was never wrong; the 9
   failures are hand-built test fixtures (`test_environment.py`,
   `test_contribution_copula_graph.py`, `test_switch_copula_graph.py`). Fix in
   `reset_state`: after `prev_state` is built, seed the key only when it is absent,
   `prev_state.setdefault("prev_agent_group", th.zeros_like(state["agent_group"]))` —
   value 0 is exactly what the existing path fills from `default_values["agent_group"]`,
   so the fix is a no-op wherever the key already exists and inertness for the step-11
   bit-identical control is preserved. Re-run every suite from step 3b afterwards.

3b. **Tests and lint** `[Sonnet]` — local: the new parity test, `src/aimanager/tests/test_eval_*.py`,
   `tests/` (torch-only suites). Raven: create the isolated dir once with
   `AI_REMOTE_DIR=... scripts/train_cluster.sh --sync-only ah <step-4 config>` and run
   the PyG suites by login-node pytest *inside* it (`test_encoder`, `test_edge_encoder`,
   `test_environment`, `test_linear_manager`, the copula and joint-exodus graph
   suites; eval-suite fixture failures are expected there per #165 note 4 and are not
   this experiment's) — never `remote_test.sh`'s shared-checkout sync. One batched
   `black` + `flake8` (88, `E203,W503`) pass over the touched `src/` files before
   staging.

4. **Training config** `[Sonnet]` — new
   `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_arrival_tenure.yml`,
   a verbatim copy of `group_switching_contribution_50ep.yml` (575 epochs, batch 4,
   lr 3e-4, hidden 20, 5-fold, seed 38381) with exactly four edits: append
   `- {name: rounds_since_arrival, n_levels: 5, encoding: onehot}` to
   `model_args.x_encoding`; append `rounds_since_arrival` to `shuffle_features` (a
   test-time importance readout only — it changes nothing about training);
   `output_dir: artifacts/artificial_humans/group_switching_contribution_50ep_arrival_tenure`;
   a `description` naming this experiment. Labels unchanged so the artifact filename
   `architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` is preserved.

5. **Train on Raven** `[Opus]` — `AI_REMOTE_DIR='~/autoresearch/contribution-arrival-tenure'
   scripts/train_cluster.sh ah <step-4 config>` (squeue PENDING check first). Record:
   SLURM job id, elapsed (expected ~8-9 min against a ~27 min ceiling), the in-job
   provenance line (`aimanager` resolving inside the isolated dir — PR #171 notes 8-9),
   the artifact sha256, and from the run's metrics parquet the per-fold held-out
   log-loss at epoch 574 against M0's **1.989742** (folds 2.048224 / 2.035988 /
   1.963259 / 1.999429 / 1.901808, PR #173 note 5) plus the
   `shuffle_feature = rounds_since_arrival` log-loss delta. Then **immediately fetch**
   `artifacts/artificial_humans/group_switching_contribution_50ep_arrival_tenure/` into
   this worktree with `fetch_cluster.sh` and commit it (LFS) — before any further
   launcher call.

6. **Pre-sim diagnostic, report-only, local** `[Sonnet]` — run the scratch teacher-forced script
   (the one behind the Declaration's numbers; PyG stand-ins, CPU, ~1 min; not
   committed) on the step-5 trunk and report against the base: arrival-row
   expected-value weights (base own 0.707 / peer 0.111; human 0.460 / 0.280),
   predicted repeat at arrival (0.329; human 0.258), predicted P(|dc| >= 5) at arrival
   (0.201; human 0.354), arrival-stratum NLL (2.338), and teacher-forced pull on the
   human switch events (0.186; human 0.430). **Whatever it says, the simulation runs
   and the verdict comes from the single evaluation** (§2, §6); this step exists so
   a failure can be read as "not learned" or "learned but did not carry".

7. **Port the stamper precondition fix** `[Sonnet]` — `git checkout
   origin/auto/contribution-group-size -- scripts/artificial_humans/make_contribution_copula_artifact.py`
   (PR #173 step 7b: `NEUTRAL_FIELDS` and `assert_only_copula_fields_changed`; the
   diff is 32 lines of assertion logic and cannot alter any stamped value). Any trunk
   trained on this branch carries the three copula keys at their neutral defaults,
   which the parent's stamper refuses. The orchestrator records the same §4 ruling
   PR #173 did.

8. **Calibrate the copula on the new trunk, with PR #165's stop-gate** `[Opus]` — copy
   `scripts/artificial_humans/calibrate_copula.slurm` to
   `calibrate_copula_arrival_tenure.slurm` with `BASE` = the step-5 artifact and
   `PARAMS` =
   `artifacts/artificial_humans/group_switching_contribution_50ep_arrival_tenure_herding_copula/calibration/copula_params.json`;
   `contribution_copula_rho.py` unchanged. `scp` the wrapper, `sbatch` from the
   isolated dir (~11-12 min: jobs 29666293 / 29892494). The job exits `1:0` by design
   (`STOP-ESCALATE` on phi >= 1); that is not a crash. Record rho, its
   200-episode-cluster bootstrap CI, phi_hat and CI, the round-trip gate, the
   preflight ratio pair, the round-thirds rho. **STOP-GATE, verbatim from #165: a rho
   CI that includes 0 ends the experiment as a calibration-only `[FAIL]`** — no
   stamp, no simulation. Otherwise `phi_final = 1.0` by #165's boundary ruling (the
   estimate stays in the JSON). Fetch and commit the params JSON and the job log
   (`git add -f`, the #165 precedent).

9. **Stamp** `[Sonnet]` — copy `scripts/artificial_humans/stamp_copula.slurm` to
   `stamp_copula_arrival_tenure.slurm` invoking the step-7 stamper with `--params`
   (step 8), `--base` (step 5) and `--out
   artifacts/artificial_humans/group_switching_contribution_50ep_arrival_tenure_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`.
   Verify: the three fields round-trip (`copula_rho` = step-8 rho, `copula_phi` = 1.0,
   `copula_switch_every` = 1), every weight tensor `torch.equal` to the step-5 trunk,
   the honesty check reports the 7,457 teacher-forced train-split rows bit-identical,
   and the loaded `x_encoding` ends in `rounds_since_arrival` with
   `default_values["rounds_since_arrival"] == 4`. Fetch and commit the artifact (LFS)
   and its `.copula.json` sidecar.

10. **Simulation config** `[Sonnet]` — new
    `configs/simulation/manager_testing/23_2g8a_contr_arrival_tenure_self_gnncopar1_contr_gnn_switch.yml`,
    a byte-copy of the parent's
    `23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml` with exactly
    three edits: `contribution_model` -> the step-9 artifact; `output_dir` ->
    `plots/simulation/23_2g8a_contr_arrival_tenure_self_gnncopar1_contr_gnn_switch`;
    `figure_name` likewise (slug before `_self_` so `evaluation_sweep.py`'s
    `DIR_PATTERN` still parses). Switch model, valid model, punisher, pairing list,
    seed, episodes, rounds untouched.

11. **Baseline control, then the candidate** `[Opus]` — two simulations via
    `scripts/simulate_cluster.sh` with `AI_REMOTE_DIR` (~2.5 min each; all remote
    artifacts already fetched, squeue checked). First the parent's own config
    unchanged: its `per_round.parquet` must reproduce sha256
    `0a34f8280bccb98a75fe002eb3669827358117ce56c44f7c10268f312904b7ab` **bit for
    bit** — the licence to compare anything and the proof that steps 1-2 are inert
    for a model that does not name the feature. Then the step-10 candidate; its
    parquet must *differ*. Fetch both with `fetch_cluster.sh` from the isolated dir.

12. **Evaluate and rule** `[Opus]` — `python -m aimanager evaluate <step-10 config>`, locally.
    One simulation, one evaluation, no second stage (§3). Record the results row: RCD
    and RCA with full precision, rows <= 1, mean; the watch items RCB / RCC / CA / CB /
    CD / CF / CG / SA / SB / SC explicitly; and, from the candidate parquet with the
    scratch diagnostics, the closed-loop arrival own/peer weights and the RCD slope
    by full/partial exodus, so the mechanism is read directly and not only through
    the score. `[SUCCESS]` only if (RCD < 2 or RCA <= 1) **and** the 21-row mean <=
    1.4344450525958377; otherwise `[FAIL]`. Open the PR with `--base
    auto/switch-joint-exodus`, body per §9 step 7 (Hypothesis / Results / Collateral
    grouped +/-), then delete `~/autoresearch/contribution-arrival-tenure` when the PR
    closes.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. **Diagnostic pass on the parent's parquet (Fable, before anything was built).**
   RCD slope over switch events with a valid receiving mean and dc: human **0.4302**
   (n = 513), parent **0.2062** (n = 1,024), the independent draw of PR #165's stack
   **0.2710** (n = 1,156). Mean dc of switchers: human 1.86, parent 0.32. Switchers'
   exact-repeat share: human 0.261, parent 0.338; share moving >= 5 points: human
   0.351, parent 0.153. The under-pull is on both signs of the gap (dc/gap: human
   0.30 below / 0.51 above the receiving mean; parent 0.21 / 0.17), so it is a slope
   deficit everywhere, worst for movers joining a *higher* group.
2. **Full exodus is not the cause, confirming PR #173 note 2.** Human pull is
   *higher* on full-exodus events (0.658, 25.3% of events) than partial ones (0.363);
   the parent under-pulls in both types (0.328 on its 33.3% full-exodus events, 0.166
   partial). A contribution-slot fix therefore has to raise the pull in every switch
   type, and nothing about the exodus mix is the contributor's to fix. The parent's
   partial-exodus events also carry a smaller mean gap (1.38 vs 2.94 human), a
   composition effect that lowers its overall slope slightly and is not addressed
   here.
3. **The anchor-release finding.** Regression c(t) ~ own c(t-1) + current
   group-mates' mean c(t-1): human stayers own 0.719 / peer 0.250 (n = 8,037), human
   arrivals **0.460 / 0.280** (n = 523); parent stayers 0.797 / 0.147, parent arrivals
   **0.767 / 0.166**; independent draw arrivals 0.697 / 0.238. Human tenure profile
   (own / repeat): 0: 0.460 / 0.258; 1: 0.686 / 0.362; 2: 0.651 / 0.353; 3: 0.662 /
   0.370; >= 4: 0.752 / 0.484. Within switchers: settled after a switch 0.720 / 0.511
   against never-switchers 0.797 / 0.516 — the release is a within-player response
   with gradual settling, not volatile people self-selecting into switching. Sim
   overall repeat rate 0.354 against human 0.403 — the sim's *stayers* are already
   under-sticky, which is why an indiscriminate anchor release (exposure) costs RCA.
   Unpunished human arrivals lean on peers more (own 0.373 / peer 0.380, n = 264)
   than punished ones (0.409 / 0.210, n = 241).
4. **Teacher-forced pass of the base trunk on the human single-copy data** (PyG
   stand-ins from `test_joint_exodus_train_sim_parity.py`, CPU, `sample=False`,
   `reset_rnn=True`): stayers own 0.759 / peer 0.117 (E[c] regression); arrivals own
   **0.707 / peer 0.111**; predicted repeat 0.370 stayers (human 0.448) and 0.329
   arrivals (human 0.258); predicted P(|dc| >= 5) 0.148 / 0.201 (human 0.142 /
   0.354); stratum NLL 1.748 stayers vs **2.338 arrivals**; teacher-forced pull on
   the 513 human switch events **0.186** (human 0.430). The closed-loop slopes (0.206,
   0.271) bracket the teacher-forced one, so the deficit is the conditional itself.
   The peer weight is also half the human value for stayers (0.117 vs 0.250, the
   PR #157 finding); the copula partially compensates in the loop (0.147-0.166).
5. **Candidates considered, with verdicts.** (A) *One-step parallel scheduled
   sampling* — one extra no-grad teacher-forced pass per batch, sample c_hat(t-1)
   from the model's own predictive, substitute it into `prev_contribution` with
   probability p (ramp 86 -> 345 as #163, p_max 0.5); ~1.3-1.5x, budget-legal, the
   literal answer to the steer. Rejected: it releases the anchor on every row where
   humans release it only on arrival (note 3), the sim's stayers are already
   under-sticky, and #163's RCA/RCB bill (RCA +3.8 at p50) against a 0.13 mean
   headroom is the gate-2 killer; the teacher-forced diagnostic (note 4) shows the
   deficit is a wrong conditional, which a closed-loop remedy does not target. The
   runner-up if the tenure feature under-delivers. (B) *Punishment-coherent variant
   of A* — substitute only where `prev_punishment == 0` (68.8% of human rows), the
   successor #163 named. Rejected: same shape objection, plus it teaches the model to
   trust its own level *because* it was punished, an artefact with no human
   counterpart. (C) *GRU hidden-state perturbation / zoneout* — no behavioural
   sentence; rejected. (D) *Raise the peer-conditioning weight directly* — an explicit
   `own_grp_prev_mean_contr` feature (#116/#144: peer beta 0.014 -> 0.031, RCD 3.04
   for the p = 0 control), gap parameterisation (a linear re-parameterisation of
   inputs the MLP already has), a change-anchored head (likelihood-invariant). Rejected
   on the arithmetic in the Hypothesis: RCD is carried by the anchor release, and the
   peer weight was pinned by teacher-forced MLE in #157. (E) *Mean-matching auxiliary
   loss* (NLL + lambda * MSE on E[c]) — would push the conditional mean where the
   regression heads got RCD right (#156, #167) while keeping the categorical emission
   that protects RCA. Rejected for this run: lambda has no a-priori value and one
   evaluation forbids shopping it, it risks CF/CD by pulling mass toward the mean, and
   the arrival deficit is a dispersion deficit as much as a level one (P(|dc| >= 5)
   0.20 vs 0.35), which a first-moment term does not address. Recorded as a successor
   family. (F) *Full-exodus structure* — measured, not the cause (note 2). (G)
   *`prev_agent_group` onehot as a node feature* — zero code, config-only, the same
   information in latent form (`prev_agent_group` already exists in both the training
   tensors and the env state). Rejected because `reset_state` and `data.shift` fill it
   with the default 0, so every group-1 agent looks "arrived" at round 0 (400
   single-copy rows against 523 real arrivals) and because it asks the one-layer tanh
   to discover an XOR from 5.5% of rows. (H) *Arrival-round hidden-state reset* —
   hand-built, and it would erase the own-trajectory memory that carries CA;
   rejected. (I) *Up-weighting arrival rows in the loss* — engineered at the stratum;
   rejected as illegal-adjacent. **Pick: (J) the capped tenure counter**, the GNN-safe
   form of the linear family's `rounds_since_switch`.
6. **Encoding ruling.** Onehot over 5 levels rather than a `just_arrived` bool or a
   numeric ramp: the human profile is a step at 0, a plateau over 1-3 and a second
   step at >= 4 (note 3). The bool would mis-fit 2,122 single-copy rows; the ramp
   would impose a monotone slope the plateau contradicts — the same trade PR #173
   paid for with `k / 8`. Never-arrived agents share the settled cell (4): settled
   switchers and never-switchers are equally sticky (0.511 vs 0.516), so a sixth
   "never" level would encode nothing the data shows.
7. **Tooling facts checked at planning.** `prev_agent_group` is rolled in
   `environment.step` *before* `apply_switch`, so the arrival test in step 2 is
   well-posed at s+1. The isolation union template (`SBATCH_EXPORT=ALL` plus the
   in-job `PYTHONPATH` export) is on this branch (`fbec309`) and was exercised by
   PR #171's and PR #173's jobs. The stamper on this branch still carries the stale
   key-absence precondition PR #173 fixed (step 7 ports it). The parent's committed
   parquet hashes to `0a34f828...` locally, the step-11 control target. Recent plain
   contribution trains: 07:57 and 08:52. Remote `~/autoresearch/` currently holds
   other experiments' dirs (`contribution-fresh-start`, `contribution-size-onehot`,
   `switch-exodus-k-onehot`, ...); the slug here collides with none of them, and the
   queue was empty at planning time.

8. **Step 5, the candidate trunk (SLURM 30252677, 8m09s, inside the ~27 min §5
   ceiling; sha256 `dc7227ed4846…`).** Provenance confirmed: the job's `aimanager`
   resolved at `/u/certuer/autoresearch/contribution-arrival-tenure/src/aimanager/`,
   not the shared checkout. Held-out log-loss at epoch 574 is **flat**: mean
   1.997552 against M0's 1.989742, **+0.007811**, with the folds splitting 2 better /
   3 worse and a per-fold spread (-0.0237 to +0.0397) several times the mean shift.
   The hypothesis predicted a *gain*, so this is nominally against it — but the
   honest reading is no measurable aggregate change, which the Declaration's own
   arithmetic anticipates: arrivals are 5.5% of training rows, so a real
   stratum-local improvement need not surface in a full-sample mean. This run
   cannot separate "the feature does nothing" from "the feature fixes 5% of the
   rows invisibly"; step 6's arrival-stratum teacher-forced diagnostic is what
   separates them. Recomputed independently by the orchestrator from the fetched
   parquet, with the recipe validated by reproducing M0's published 1.989742 and
   its five folds exactly.

9. **The model uses the feature, and the trunk demonstrably never had the
   information.** Shuffle-importance at epoch 574: `rounds_since_arrival`
   **+0.017590**, positive in all five folds and about half the weight of
   `prev_punishment` (+0.036326) — so the "silently ignored" failure mode did not
   occur. The neighbouring number is the striking one: shuffling **`agent_group`
   costs essentially nothing** — -0.000156 in the candidate, +0.000863 in M0 — i.e.
   destroying the group label leaves the model's predictions intact. That is direct
   confirmation of the Declaration's claim that the GRU never recovered arrival from
   `agent_group`, and it establishes the precondition the hypothesis needed: the new
   feature carries information the trunk did not have. What it has not yet shown is
   conversion into behaviour.

10. **Step 6, teacher-forced diagnostic on the candidate trunk (report-only, gates
    nothing).** Base reproduces the Declaration exactly, so the comparison is sound.
    Arrivals (n=523): own-weight **0.707 -> 0.637** (human 0.460), peer 0.111 ->
    0.086 (human 0.280), P(repeat) **0.329 -> 0.254** against a human 0.258 — that
    cell is now essentially exact — P(|dc| >= 5) 0.201 -> 0.280 (human 0.354),
    stratum NLL 2.338 -> 2.206. Stayers (n=8,037) barely move: own 0.759 -> 0.757,
    peer 0.117 -> 0.120, NLL 1.7480 -> 1.7463, P(|dc| >= 5) 0.148 -> 0.144.
    **The release is targeted, which is the whole design claim** — this is not
    #163's indiscriminate shape, and it is why the flat aggregate likelihood of note
    8 is consistent with a real stratum-local change. Verified independently by the
    orchestrator by re-running the script on the candidate artifact.

11. **The mechanism is real but weak, and gate 1 is in doubt before the simulation.**
    Teacher-forced pull on the human switch events moved **0.186 -> 0.220** against a
    human 0.430 — roughly 14% of the deficit closed. The base's teacher-forced 0.186
    sat against a closed-loop 0.206, so if that relationship holds the candidate lands
    near 0.24, short of the 0.2698 the RCD band upgrade needs. Recorded before the
    simulation so the prediction is on the record and cannot be written after the
    fact. It changes nothing procedurally: the copula steps and the single evaluation
    run as planned, and the verdict comes from that evaluation alone (§2, §6).
