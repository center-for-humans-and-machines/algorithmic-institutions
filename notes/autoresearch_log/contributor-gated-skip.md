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
| 4 | **Train the candidate trunk** on Raven in the isolated dir. Record held-out log loss against the frontier's and wall time against §5's ~33 min ceiling. | pending |
| 5 | **The screen.** `rcb_teacher_forced.py` on the human data for the **current trunk** and the **candidate**, four band slopes each against the human reference. Decide out loud; stop here if the middle bands do not move. | pending |
| 6 | Stamp the frozen copula onto the candidate (carry `rho`/`phi`, verify bit-identical weights). | pending |
| 7 | Simulate the frontier stack with the candidate swapped in; fetch; evaluate all 22 rows with `PYTHONPATH=<worktree>/src`. | pending |
| 8 | The copula-off state-spread diagnostic, `Var(E[c \| history])` against the human 27.9 and the current 18.9. | pending |
| 9 | Judge, log, PR against the parent. | pending |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | (baseline) the frontier stack, PR #192 (`_ceiling`) | RCE 0.882320486446721 | 14/22 | 1.0330897999314548 | baseline |

## 4. Notes

1. The declaration, the plan and the caveats above are written before any candidate number exists; nothing in section 3 is filled in until it is measured.
