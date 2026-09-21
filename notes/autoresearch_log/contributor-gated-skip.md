# Gating the contributor's direct punishment path (RCE)

## 1. Declaration

**Slot:** contribution. One change: the immediate-stimulus skip is **gated** instead of concatenated.

**Parent:** PR #192 (`auto/punisher-ceiling-fix`, at `e2306292eb667f60ae2e9fd1b7189975f8b3efa2`), the model the maintainer has accepted as current. Branch `auto/contributor-gated-skip` is created from it and the PR opens with `--base auto/punisher-ceiling-fix`. Isolated remote dir `~/repros/ai-runs/gated-skip` (delete when this PR closes).

**Base model.** The frontier contributor: the M0 GNN trunk plus the per-group virtual node plus the immediate-stimulus skip of PR #181, copula-stamped, `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` (`rho = 0.03949863621805423`, `phi = 1.0`, `switch_every = 1`). `x_encoding = prev_contribution (numeric, 21), prev_punishment (numeric, 31), agent_group (onehot, 2)`; hidden 20; `add_global_model: False`; `group_vnode: True`; `stimulus_skip: True`; 575 epochs, batch 4, lr 3e-4, seed 38381, flip-doubled data. Switch and punisher slots untouched.

**Evaluation stack (§3 under the parent rule of §9):** the parent's own frontier config `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling.yml` -- this contributor x the joint-exodus GNN switch predictor x the ceiling-fixed severity-copula `lin_multinomial`, single pairing `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds, `save_per_round: true`.

**Baseline (the parent's confirmed scores at full precision, `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling/evaluation/scores.csv`, in this worktree since the branch was cut from it).**

| row | score | band | seed sd (PR #195) |
|---|---|---|---|
| **RCE** (declared target) | **0.882320486446721** | <= 1 | 0.106 (**ungateable on one run**) |
| RCC | 1.296894519602628 | 1-2 | 0.163 |
| RCB | 1.659050738274753 | 1-2 | 0.142 |
| RCA | 1.6525551040598578 | 1-2 | 0.141 |
| RCD | 1.251543083466537 | 1-2 | 0.270 |
| CG | 1.758780949194544 | 1-2 | 0.301 |
| SC | 1.4632340845533836 | 1-2 | 0.136 |
| mean over 22 rows | **1.0330897999314548** | | 0.047 |
| gate-2 ceiling (mean x 1.10) | **1.1363987799246003** | | |
| rows <= 1 | 14/22 (context only) | | 3.16 |

The maintainer's declared baseline for this experiment is the **shipped run** above (RCE 0.8823, mean 1.0331), not §2's six-arm mean (RCE 1.0683, mean 1.0923, rows <= 1 10/22). Both are quoted wherever a movement is read, because the shipped run is the favourable tail of its own distribution.

**Target row: RCE**, the punishment-response slope row. Per §2 as amended on `docs/post-rebaseline-program`, an experiment whose declared target is the contributor's reaction to punishment is judged on RCE for gate 1 rather than only protected by it. Gate 1 is a band improvement that also exceeds RCE's seed sd of **0.106**. Note that RCE already sits in band <= 1 on the shipped baseline, so on that reading there is no band above it to reach; against §2's six-arm mean of 1.0683 the row is in band 1-2 and an upgrade to <= 1 is formally available, though PR #195 marks the row ungateable on a single run either way. What this experiment can demonstrate cleanly is the band slopes moving toward the human values by more than their own floors. Watch rows: **RCB** and **RCA** (the other response rows the same channel feeds), **CG** and **RCD** (the two persistence rows the ungated skip sold, PR #181 note 16), the marginal C block, and **SC**.

**Protected-row rule (§2 as amended on `docs/post-rebaseline-program`).** All three RCE clauses take the row's own seed sd as their threshold: the band-drop clause fires only on a drop larger than 0.106, and the sign clause is **retired on the 10-14 and 15-19 bands**, where an unchanged model flips sign on the training draw alone (PR #195: 10-14 runs +0.037 to -0.043, 15-19 runs +0.011 to -0.130 across six arms). The 0-4 and 5-9 bands keep it. No movement smaller than its row's seed deviation counts either for or against.

### Hypothesis

**The gap.** RCE fits, per contribution band, the OLS slope of the next-round contribution change on the punishment received. Real people give **+0.140 / +0.104 / -0.077 / -0.161** across bands 0-4 / 5-9 / 10-14 / 15-19. The frontier contributor gives **+0.095 / +0.020 / -0.058 / -0.160**; the best stack on this row, from the Gaussian lineage, gives **+0.120 / +0.071 / -0.112 / -0.205**. The frontier is not uniformly worse -- it matches the top band almost exactly where the Gaussian overshoots. The gap is the **middle**, and mostly the **5-9 band**, where the frontier responds at a fifth of human strength.

**What is already eliminated.** PR #190 ported the Gaussian lineage's switch head onto this trunk and the row got worse -- and the RCE row is a contributor property the switch head touches only through group composition. PR #191 falsified the emission-head hypothesis. The earlier RCB investigation measured both families teacher-forced and found the Gaussian family's slopes near-human there with its deficit in composition instead, so the difference between the families is a genuine difference in the learned conditional and not an artefact of who gets punished in simulation.

**The mechanism that distinguishes them.** The Gaussian contributor takes punishment received as a **direct feature**. This trunk routes it through the edge layer into the recurrent memory, and PR #181 added a direct path as a **plain concatenation with no gate** (§5's "ties go to the simpler model"). That PR's own successor note (note 17) argues the concatenation is the limitation: `op2` weights the immediate stimulus and the carried state at **one fixed ratio for every round**, whereas the behaviour wants the stimulus to dominate on rounds where something happened to the player and the memory to dominate otherwise. Note 16 is the evidence: the ungated skip bought the individual-fit rows and sold exactly the two persistence rows, CG (<= 1 -> 1-2) and RCD (1-2 -> 2-5).

**Behavioural rationale (one sentence, §5):** how much of my next decision is what just happened to me and how much is what I remember should depend on whether anything happened to me this round -- the row that should move is **RCE**, with RCB and RCA as the neighbours and CG and RCD as the persistence the gate is supposed to give back.

### The change, and exactly what the gate conditions on

One flag, `model_args.stimulus_gate: True`, in `GraphNetwork`. With the flag on, `op2` no longer receives the post-`op1` embedding and the post-RNN embedding side by side. They are mixed by a per-agent, per-round scalar gate:

```
g = sigmoid(w . x_skip + b)          # one scalar per agent per round
x = g * x_skip + (1 - g) * x_rnn     # broadcast over the hidden channels
```

`x_skip` is the post-`op1` node embedding -- this round's own contribution and punishment after message passing over the group -- and `x_rnn` is the same tensor after the per-agent GRU.

**The gate conditions on `x_skip` and on nothing else.** That is the same information the skip already sees, which is the form PR #181 note 17 proposed, and it is the right conditioning set on the behavioural argument: "did something happen to me this round" is a property of the stimulus, not of the memory the stimulus would displace. Conditioning the gate on `x_rnn` as well would let the model open the gate whenever its own carried state is unusual, which is a self-consistency criterion rather than an event detector, and it would make the gate a second recurrent readout rather than one scalar.

Everything else is unchanged: one `Linear(hidden_size, 1)`, no other architectural change, the recurrent path, the edge model, the virtual node and the encodings all untouched. Because the gate mixes rather than appends, `op2` returns to the **un-skipped width** -- the gate replaces the extra slice instead of adding to it. The gate module is constructed **last of all**, after the virtual node, so with the flag off no RNG is drawn and every existing artifact loads and behaves bit-identically.

**The copula is carried, not recalibrated.** §2's frozen-noise-model rule: `rho = 0.03949863621805423`, `phi = 1.0`, `switch_every = 1` would be stamped onto the retrained trunk unchanged. A recalibration riding along with a trunk change makes the two indistinguishable.

### Method: screen before simulating

`scripts/data_analysis/rcb_teacher_forced.py` scores a contribution-model candidate against the response rows from the human trajectories in about a minute, and PR #181 records that its pre-simulation prediction came within **1.2%** of the closed-loop result. The gated trunk is trained and screened teacher-forced **before** any simulation is spent. If the middle bands do not move teacher-forced, the experiment stops there and reports a screen result; a simulation is only spent if the mechanism installs.

### Three caveats, stated rather than discovered

1. **The gap between the two stacks is modest.** The 5-9 band difference between the frontier (+0.020) and the Gaussian (+0.071) is about **1.8 seed deviations** -- real, but not large. PR #195 measured the 5-9 band slope's seed sd at 0.0223 across six retrains of an unchanged model, and the whole-row seed sd at 0.106.
2. **The Gaussian stack is not simply the better model.** Its 22-row mean is **1.19** against this stack's **1.04**, with a badly failing round-type row. It wins this row and loses overall, so "be more like the Gaussian" is a claim about one conditional, not about the model.
3. **The 10-14 band has had the wrong sign in every condition ever tested, including held out** (PR #183, and PR #181's step 0: the trunk gives +0.0425 teacher-forced where humans give -0.0767). No closed-loop change will supply it. If a correct sign appears there it will not be claimed as a result of this change.

### Artifact naming contract

| what | path |
|---|---|
| training config | `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_vnode_gated_skip.yml` |
| bare artifact | `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_gated_skip/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` |
| copula-stamped copy (only if the screen passes) | `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_gated_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` |
| tests | `tests/skip/test_stimulus_gate.py` |
| sim config (only if the screen passes) | `configs/simulation/manager_testing/23_2g8a_contr_gated_skip_self_gnncopar1_contr_gnn_switch_ceiling.yml`; sim dir `plots/simulation/<same>` |
| tables | `plots/data_analysis/evaluation/contributor_gated_skip/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | **The gate in `GraphNetwork`.** `src/aimanager/generic/graph.py`, constructor and `forward`. Add `stimulus_gate` (requires `stimulus_skip`) and a `stimulus_gate_module`; mix instead of concatenating; save/load both keys; build the module last so the flag off is bit-identical. | done |
| 2 | **Tests for the gate.** New `tests/skip/test_stimulus_gate.py`: off is bit-identical, on returns `op2` to the un-skipped width and adds only the gate's two parameters, `op2` receives exactly the mix, gradient reaches the gate, 24 single-round calls reproduce one 24-round call, save/load round-trips, an artifact without the keys loads ungated, and both guards fire. | done |
| 3 | **The training config**, a copy of the frontier's with `stimulus_gate: True` and a new `output_dir`; everything else byte-identical. | done |
| 4 | **Train the candidate trunk** on Raven in the isolated dir. Record held-out log loss against the frontier's and wall time against §5's ~33 min ceiling. | done |
| 5 | **The screen.** `rcb_teacher_forced.py` on the human data for the **current trunk** and the **candidate**, four band slopes each against the human reference. Decide out loud; stop here if the middle bands do not move. | done |
| 6 | Stamp the frozen copula onto the candidate (carry `rho`/`phi`, verify bit-identical weights). | **not reached** -- the screen stopped the experiment |
| 7 | Simulate the frontier stack with the candidate swapped in; fetch; evaluate all 22 rows with `PYTHONPATH=<worktree>/src`. | **not reached** |
| 8 | The copula-off state-spread diagnostic, `Var(E[c \| history])` against the human 27.9 and the current 18.9. | **not reached** -- it is judged alongside a simulation, and none was spent |
| 9 | Judge, log, PR against the parent. | done |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | (baseline) the frontier stack, PR #192 (`_ceiling`) | RCE 0.882320486446721 | 14/22 | 1.0330897999314548 | baseline |
| 2026-09-21 | **the gated skip**, screened teacher-forced only | four band slopes, tables below | -- | -- | **FAIL (screen)** -- no simulation spent |

Full table: `plots/data_analysis/evaluation/contributor_gated_skip/screen_teacher_forced.csv`.

### Step 4: the candidate trunk trained (measured)

SLURM **30398571** on Raven in `~/repros/ai-runs/gated-skip`, one A100, **11 min 40 s**, exit 0:0 -- inside §5's ~33 min ceiling and materially the same as the frontier's own runs, as expected for one extra `Linear(20, 1)`.

Cross-validated log loss (`scripts/data_analysis/best_test_loss.py`, 5 folds, best epoch per fold):

| trunk | best mean | best sd | globally best epoch | final epoch |
|---|---|---|---|---|
| pre-skip vnode (PR #179) | 2.0002 | 0.0636 | 450 | 2.0086 |
| frontier ungated skip (PR #181) | 2.0028 | 0.0646 | 450 | 2.0206 |
| **gated skip (this branch)** | **2.0033** | 0.0598 | 400 | 2.0356 |

The gate is worth **0.0005 of best-epoch log loss** against the frontier -- an order of magnitude inside the spread PR #188 measured over five retrains of the unchanged architecture (2.0201 to 2.0293), i.e. **not distinguishable from a retrain on fit**. That is neither a surprise nor a disappointment: the change is about where information is routed on a minority of rounds, and a 21-class log loss barely registers it. The screen, not the CV, is the measurement that decides.

### Step 5a: the screen on the human trajectories (measured)

`scripts/data_analysis/rcb_teacher_forced.py`, unchanged, one job per model over the 50 single-copy human games: each model sees the human history and never its own draws, under its own stored `default_values`. The script's four alignment checks and its (C) self-check -- which reproduces all four human slopes and all four human bin means to seven decimals on the identical population, weighted discrepancy 1.49e-07 -- **PASS in every run**, so the three models are compared like for like. Population: punished, non-full contributors, next-round contribution valid, **n = 2,660**. SLURM 30398593 (frontier), 30398661 (candidate), 30398680 (pre-skip reference), each 13-18 s, exit 0:0.

Slope of the next-round contribution change on the punishment received, within contribution band:

| band | human | pre-skip vnode (PR #179) | current trunk (ungated skip) | **candidate (gated skip)** | candidate - current |
|---|---|---|---|---|---|
| 0-4 | **+0.1397** | +0.1157 | +0.1482 | **+0.0948** | **-0.0534** |
| 5-9 | **+0.1038** | +0.0791 | +0.1038 | **+0.0703** | **-0.0335** |
| 10-14 | **-0.0767** | +0.0425 | +0.0171 | **+0.0496** | +0.0325 |
| 15-19 | **-0.1615** | -0.1659 | -0.2197 | **-0.1152** | +0.1045 |
| weighted bin discrepancy | 0 | 0.0930 | 0.1094 | **0.0939** | -0.0155 |

Both reference columns reproduce published numbers digit for digit, which pins the provenance of the whole table: the pre-skip column is PR #181's step 0 (+0.1157 / +0.0791 / +0.0425 / -0.1659) and the frontier column is PR #183's shipped-artifact row for this trunk (+0.148 / +0.104 / +0.017 / -0.220, statistic 0.1094 against the 0.1094077799 measured here).

**The middle bands move the wrong way.** 5-9 falls from **exactly human** (+0.1038 against the human +0.1038, ratio 0.99997) to +0.0703, 68% of human. 0-4 falls from +0.1482, an overshoot, to +0.0948, 68% of human. The 10-14 band stays wrong-signed and gets more so. Only 15-19 moves toward human in absolute error, from a 36% overshoot to a 29% undershoot, and that is what makes the aggregate bin discrepancy look flat (0.1094 -> 0.0939); both numbers are far inside the human-vs-human noise ceiling of 0.3479 and the aggregate is not the quantity this experiment declared.

### Step 5b: the screen at the frontier's own realised states (measured)

The human-data screen measures the conditional; the row is lost in the closed loop. The same script's sim mode teacher-forces a trunk over an already realised trajectory, which **is** that trajectory's conditional expectation -- the construction PR #181 note 14 used to predict its closed-loop result to within 1.2%. All three trunks are run over the **frontier's own** `per_round.parquet` (`23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling`, 19,200 rows, RCB population n = 5,771), so the states are held fixed and only the model varies. SLURM 30398618 / 30398662 / 30398681.

| band | human | frontier's observed closed loop | frontier trunk at those states | pre-skip vnode at those states | **candidate at those states** |
|---|---|---|---|---|---|
| 0-4 | **+0.1397** | +0.0868 | +0.0844 | +0.0637 | **+0.0433** |
| 5-9 | **+0.1038** | +0.0379 | +0.0432 | +0.0148 | **+0.0206** |
| 10-14 | **-0.0767** | -0.0427 | -0.0323 | -0.0140 | **-0.0154** |
| 15-19 | **-0.1615** | -0.1298 | -0.1344 | -0.0773 | **-0.0550** |
| weighted bin discrepancy | 0 | 0.5413 | 0.3996 | 0.4826 | **0.5029** |

The frontier trunk's teacher-forced column tracks its own observed closed loop to within 0.011 on every band (0.0024 / 0.0053 / 0.0104 / 0.0046), which validates the instrument on this stack exactly as PR #181 validated it on the previous one. Read against that, the candidate's column is a forecast of what a simulation would produce: **roughly half the response in every band**, and further from human in all four.

Note that the script's sim-mode (C) self-check exits 2 by construction here -- its `SIM_REF_SLOPES` are hard-coded to PR #179's run, not this stack's. The observed column above (+0.0868 / +0.0379 / -0.0427 / -0.1298) reproduces PR #195's record of this very artifact's closed loop digit for digit, so the population and the parquet are right and only the stale reference fires.

### Step 5c: what the trained gate actually learned (measured)

`scripts/data_analysis/gated_skip_gate_stats.py`, a hook on the gate module over the same teacher-forced pass, 9,320 valid human rows. Table: `plots/data_analysis/evaluation/contributor_gated_skip/gate_stats.csv`.

The gate's parameters are `|w| = 2.9258`, `b = -0.8282` (`sigmoid(b) = 0.3040`), but on real data the logit is driven far below the bias:

| quantity | value |
|---|---|
| gate mean | **0.00044** |
| gate sd | 0.00050 |
| gate min / max | 0.000055 / **0.0019** |
| punished last round (n 2,704) | 0.000374 |
| not punished last round (n 6,616) | 0.000467 |
| punished hard, p >= 5 (n 1,153) | 0.000479 |
| corr(gate, prev_punishment) | +0.0746 |

**The gate closed.** It never exceeds 0.002 on any of the 9,320 rows, so `x = g * x_skip + (1 - g) * x_rnn` is the post-RNN embedding to three decimal places on every round of every episode: training switched the skip off. And it is not an event detector even at that scale -- the punished rows sit **below** the unpunished ones (0.000374 against 0.000467, a separation of -0.19 of the gate's own sd), the opposite of the sign the hypothesis predicted.

### The decision, out loud

**No simulation was spent.** The pre-declared rule was: if the middle bands do not move teacher-forced, say so and stop. They did not move toward human -- they moved away, by **-0.0534** (0-4) and **-0.0335** (5-9) on the human states, and the sim-state screen forecasts roughly half the frontier's response in all four bands. There is no reading of the screen under which a simulation could produce a band improvement on RCE, and PR #181 established that this screen predicts the closed loop to 1.2%. Spending an A100 simulation and a 22-row evaluation to confirm a null this clear is not a use of the budget.

The experiment is a **`[FAIL]` at the screen**. Gate 1 is not reached (no simulation, therefore no band movement to claim), gate 2 is not evaluated, and the protected-row clauses are not triggered because no scores exist. The useful content is section 4.

## 4. Notes

1. The declaration, the plan and the caveats above were written before any candidate number existed; section 3 was filled in only from measurements.
2. **The gate collapsed to zero, so the candidate is approximately the pre-skip trunk retrained.** That is what the three-way tables show: on the human states the candidate (+0.0948 / +0.0703 / +0.0496 / -0.1152) sits much nearer the pre-skip vnode trunk (+0.1157 / +0.0791 / +0.0425 / -0.1659) than the frontier it was built from (+0.1482 / +0.1038 / +0.0171 / -0.2197) -- nearer in all four bands, and nearer in all four again at the frontier's own realised states. The experiment did not gate the skip; it trained a model that learned to discard it.
3. **Why the optimiser closed the gate, and the one-line reason the change could not have worked.** A convex mix is not a generalisation of the concatenation -- it is a **restriction**. Under concatenation `op2` holds two independent weight blocks, one reading the post-`op1` embedding and one reading the post-RNN embedding, so it can give the stimulus its own direction in the readout. Under the mix there is a single block applied to `g * x_skip + (1 - g) * x_rnn`, so the stimulus must be read through the same linear map as the memory, and the gate can only rescale it. Those two embeddings are not in a common coordinate system -- one is `tanh(op1)`, the other a GRU hidden state -- so blending them corrupts the memory's representation, and the cheapest way out is `g -> 0`. The "one fixed ratio for every round" that PR #181 note 17 objected to is in fact one fixed **pair of linear maps**, which is strictly more expressive than a shared map with a learned scalar.
4. **So note 17 of `contribution-punishment-response.md` is refuted as stated, but only in its convex-mix reading.** The successor note says "a scalar gate on the skip's contribution". This branch implemented the maintainer's phrasing of it -- mixing the post-stimulus embedding against the post-memory one instead of concatenating at a fixed ratio -- and that reading is now dead. The **multiplicative** reading, keeping the concatenation and scaling only the skip's own slice (`[x_rnn | g_node | g * x_skip]`, with `op2` still holding a separate block for the skip), is untested and is not ruled out by anything here. It is a strictly-more-general model than the current frontier, which the mix was not, so it cannot lose by construction the way this one did.
5. **The declaration's premise was already false in the ledger, and this branch reproduced the disproof instead of noticing it first.** Teacher-forced on the human trajectories the shipped frontier trunk gives **+0.1482 / +0.1038 / +0.0171 / -0.2197** against human +0.1397 / +0.1038 / -0.0767 / -0.1615: its 5-9 band -- the band the maintainer identifies as most of the gap, where the closed loop delivers +0.020, a fifth of human -- is **exactly human in the conditional**, to four decimals, and the 0-4 band overshoots. These are not new numbers. PR #183 published exactly them (+0.148 / +0.104 / +0.017 / -0.220, statistic 0.1094) as its secondary model, and its note 7 already drew the conclusion: "the skip trunk is not better teacher-forced ... its closed-loop gain therefore came from the loop, not from a better conditional." Reading that table before training would have shown that a change to the stimulus route had nothing left to buy. The reproduction is worth something as a provenance check across two independent scripts; it is not a discovery, and the credit belongs to PR #183.
6. **The sim-state screen localises where the row is actually lost, and that part is new.** At the frontier's own realised states the same weights give +0.0844 / +0.0432 -- against +0.1482 / +0.1038 at the human states. The map is unchanged; only which `(x_t, h_{t-1})` it is evaluated at changed, and that alone costs 43% and 58% of the two low bands. PR #183 showed the conditional is learned and generalises, and PR #181 note 8 showed the same teacher-forced-vs-loop structure for RCB; what was not on record is the size of the loss at the loop's own states for this trunk and these bands. It also bears on the hypothesis that motivated this branch: the Gaussian lineage's advantage on this row is very unlikely to be "it takes punishment as a direct feature", because this trunk's direct response is already human-sized and the deficit sits entirely downstream of it.
7. **The 10-14 band did not change sign and was never going to.** It is +0.0425 (pre-skip), +0.0171 (frontier), +0.0496 (candidate) teacher-forced -- wrong-signed in every measurement of this lineage, as PR #183 found held out. Recorded so the next agent does not spend a hypothesis on it without first explaining why the human sign flip is absent from the conditional. Nothing about it is claimed here.
8. **What a simulation would probably have done to the protected row, for the record.** The sim-state forecast has the 15-19 band at about -0.055 against the frontier's -0.1298 and the 0-4 band at about +0.043 against +0.0868 -- both more than a halving, and both **further** from the human value, so RCE's magnitude clause would fire without its qualification (a) to excuse it, and the 15-19 movement of 0.075 is about 1.8 of that band's within-run SE of 0.0417. On the forecast the simulation would have produced a protected-row `[FAIL]` on top of no gate-1 upgrade. That is a second, independent reason not to have spent it.
9. **The copula was not recalibrated and nothing was stamped**, because the experiment stopped before the step that would have needed it. `rho = 0.03949863621805423`, `phi = 1.0` remain the frozen family parameters. The bare candidate trunk is committed so a successor can re-screen it without retraining.
10. **The state-spread diagnostic was not run.** §2 requires it for a contributor-slot candidate alongside the usual gates, and the usual gates require a simulation; with no simulation there is nothing for it to be reported alongside. The frontier's own number is unchanged by this branch: human `Var(E[c | hist])` 27.9 against the copula-off trunk's 18.9.
11. **Cost, because the screening rule is the transferable part.** One 11 min 40 s A100 training and six CPU jobs of 13-18 s each. PR #181 built this screen and used it to confirm a decision already taken; this is the first experiment to use it as a **stopping rule**, and it fired on the first use. Contribution-slot experiments should screen by default, and should screen at the sim's realised states as well as the human ones -- the two disagree here about which model is better, and only the sim-state one predicts the row.
12. **Gate 1 was formally unavailable on this baseline in any case.** RCE sits at 0.8823 in band <= 1 on the shipped frontier run, so there is no better band to reach, and it is one of the ten rows PR #195 marks ungateable on a single run (seed sd 0.106, six arms spanning 0.8823 to 1.2110 across the 1.0 boundary). Against §2's six-arm mean of 1.0683 an upgrade to <= 1 is formally possible but not demonstrable on one run. The honest target was always the band slopes, which is exactly what the screen measures and refutes.

## 5. Successor

**Stop attacking the route from the stimulus to the readout.** Five measurements now say the contributor's punishment conditional is not the binding constraint on RCE's bands: PR #181's step 0 (pre-skip trunk at 83% / 76% of human on the two low bands), PR #183's secondary table (the shipped frontier trunk at 106% / **100%**, held out as well as in sample), this branch's reproduction of it, this branch's note 6 (the same weights at 60% / 42% once the states are the loop's own), and the screen itself (a gate makes it worse). PR #190 eliminated the switch head and PR #191 the emission head. What is left is **which states the rollout visits**.

**And read PR #183's tables before proposing anything in this family.** They contain the frontier trunk's teacher-forced band slopes, held out and in sample; this branch trained a model before consulting them and would not have run had it done so first. The declaration's framing -- "the Gaussian takes punishment as a direct feature, this trunk routes it through memory" -- survives as a description of the two architectures but not as an explanation of the row.

In the order it should be taken:

1. **Say which half of the sim-state loss is composition and which is the carried state.** The tooling is already here: `rcb_teacher_forced.py --sim-parquet` gives the conditional at the realised states, and the frontier's `per_round.parquet` gives the realised `(contribution, punishment)` cells. Run PR #181 note 4's cell-level decomposition for the **RCE** bands -- give the sim's own cell composition the human conditional and vice versa. Nobody has run it for this row, and RCE's band definition makes it a different question from RCB's rate bins. Cost: minutes, no training.
2. **If it is the carried state**, measure it before modelling it: the distribution of the per-agent GRU hidden state over sim rounds against its distribution over human rounds, which needs no training at all. §5 vetoes the schedsamp family on wall clock, so a state-regularisation term during training is the realistic lever, and it should not be attempted before the drift is characterised.
3. **If the multiplicative gate is still wanted**, note 4 says what it would have to be: keep the concatenation and scale only the skip's slice, so `op2` retains a separate weight block for the stimulus and the gate is a strict generalisation of the frontier rather than a restriction of it. It is a different experiment with a different declaration, and on the evidence of notes 5 and 6 it should not be expected to move RCE either -- the conditional it would sharpen is already human-sized.

One measurement this branch could not make and a successor should: the frontier's teacher-forced 5-9 slope landing at 0.99997 of human is arresting enough to want a second draw. PR #188's five seed replicas of this architecture are already trained and sitting on `auto/copula-seed-ensemble`; running the human-mode screen over them would cost six 15-second jobs and would say whether "the conditional is exactly right" is a property of the architecture or of the shipped draw.

