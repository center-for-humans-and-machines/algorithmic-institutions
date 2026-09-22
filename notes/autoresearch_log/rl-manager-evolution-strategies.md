# RL manager, exploration comparison: the evolution-strategies arm

One of four arms. The siblings are annealed local epsilon-greedy, bootstrapped DQN and parameter-space noise. Method: evolution strategies in the sense of Salimans, Ho, Chen, Sidor and Sutskever (2017) — mirrored sampling of parameter perturbations, centered-rank fitness shaping, Adam on the fitness-weighted average. There is no action noise anywhere: every policy that is run or scored is a fixed deterministic policy evaluated over complete episodes.

**Status: RESULT.** All five seeds finished cleanly (job ids 30413786 s42, 30413787 s43, 30413789 s44, 30413790 s45, 30413791 s46, in `~/repros/ai-runs/rl-es`). "Result" below is what they produced; "Measured" is the pre-launch evidence, kept unedited; "Predicted" is the pre-registration, also unedited, and "R4" scores it. One of the five predictions was wrong and it is the one worth reading.

---

# Result

**All five seeds converged to a policy with no contingency on contribution at all.** Not a wrong-signed contingency — no contingency. Three seeds ended punishing nothing; two ended levying a **flat tax**, punishing every player exactly 2.0 (s44) or exactly 1.0 (s46) whatever they contributed. The evaluated contribution-punishment slopes are 0.0000, 0.0000, +0.0038, 0.0000, +0.0019, against the human managers' −0.857 and this project's human clone's −0.659.

**The number that decides it:** at generation 0 the 40 population members' slopes had a standard deviation of 0.93 to 1.33 and spanned −1.57 to +3.10, bracketing every shape the DQN seeds ever produced. By generation **220 to 320** — all five seeds inside a 100-generation window — that spread had fallen below a tenth of its starting value, and it never came back. Selection did not choose a direction for the contingency. It **removed the contingency**, in the first 8% of the budget, and spent the other 92% on a level.

So the arm does not recover the human shape, and the reason is not the one I registered in advance.

## R1. The collapse detector, first, as the successor section demands

Two detectors, and **the one I shipped before launch was the wrong one.**

`members_punishing_nothing` counts members whose mean punishment is exactly 0. On that criterion:

| seed | punishment collapse | generations of random walk after it | dead members at the end | s/n over last 100 |
|---|---|---|---|---|
| 42 | generation **3287** | 713 (17.8% of budget) | 40/40 | 1.01 |
| 43 | generation **3987** | 13 | 38/40 | 1.07 |
| 44 | **never** | 0 | 0/40 | 1.03 |
| 45 | generation **3991** | 9 | 38/40 | 1.03 |
| 46 | **never** | 0 | 0/40 | 1.02 |

Read alone this says two seeds stayed healthy to the end. **That reading is wrong**, and the disagreement between the two columns is what gives it away: signal-to-noise is ≈ 1.0 in *all five* seeds, including the two with a fully live population. A live population whose fitness ranking carries nothing is not a working optimiser.

The right detector is the spread of the per-member contribution-punishment **slope** — the shape variance the population still carries. A flat tax is exactly as degenerate as a zero policy for the question this arm exists to answer, and the zero-punishment criterion cannot see it:

| seed | slope sd at gen 0 | slope sd at the end | shrinkage | **shape variance dies at** | s/n falls below 1.3 at |
|---|---|---|---|---|---|
| 42 | 0.928 | 0.0000 | 9×10⁸ | **gen 220** | gen 192 |
| 43 | 1.293 | 0.0158 | 82× | **gen 280** | gen 193 |
| 44 | 1.335 | 0.0038 | 349× | **gen 300** | gen 187 |
| 45 | 1.281 | 0.0000 | 1×10⁹ | **gen 300** | gen 248 |
| 46 | 0.999 | 0.0323 | 31× | **gen 320** | gen 320 |

The member-slope spread over training, the series behind that column:

| generation | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|
| 0 | 0.928 | 1.293 | 1.335 | 1.281 | 0.999 |
| 100 | 0.385 | 0.450 | 0.761 | 0.183 | 0.456 |
| 200 | 0.157 | 0.126 | 1.016 | 0.515 | 0.636 |
| **400** | **0.046** | **0.051** | **0.001** | **0.023** | **0.038** |
| 800 | 0.000 | 0.000 | 0.001 | 0.003 | 0.094 |
| 3980 | 0.000 | 0.016 | 0.004 | 0.000 | 0.032 |

The two events coincide: the population stops disagreeing about shape at the same moment the fitness ranking stops carrying signal. After generation ~320, every seed is a random walk on the level, and **everything downstream of that generation must be read as such** — including the late punishment collapses at 3287/3987/3991, which are random-walk events rather than optimisation.

`collapse.csv`, `collapse_trajectory.csv`, `shape_degeneracy.csv`, `shape_degeneracy_trajectory.csv`.

## R2. Policy shape, the primary outcome

Mean punishment by contribution bin, the evaluation suite's own `RPA_EDGES`, evaluated (mean-parameter) policy, from each run's own final evaluation rollouts.

| bin | human managers | lin_punisher (clone) | es_s42 | es_s43 | es_s44 | es_s45 | es_s46 |
|---|---|---|---|---|---|---|---|
| {0} | 4.755 | 3.721 | 0.000 | 0.000 | 2.000 | 0.000 | 1.000 |
| 1-5 | 2.973 | 2.930 | 0.000 | 0.000 | 2.000 | 0.000 | 1.000 |
| 6-10 | 1.672 | 1.808 | 0.000 | 0.000 | 1.867 | 0.000 | 0.934 |
| 11-15 | 0.978 | 1.300 | 0.000 | 0.000 | 2.000 | 0.000 | 1.000 |
| 16-19 | 0.692 | 1.066 | 0.000 | 0.000 | 2.000 | 0.000 | 1.000 |
| {20} | 0.267 | 0.327 | 0.000 | 0.000 | 2.000 | 0.000 | 1.000 |
| **slope** | **−0.857** | **−0.659** | **0.000** | **0.000** | **+0.004** | **0.000** | **+0.002** |

Agent-rounds behind each ES mean: 4,560 to 35,126 per bin (`policy_shape_n.csv`). These counts are an order of magnitude larger than the two-worlds table's 555–3,287 because they come from 1000-episode training evaluation rollouts rather than a 100-episode simulation; R3's table is the like-for-like one.

For orientation, the DQN seeds on the same bins: rl_s42 **+1.218** (inverted), rl_s43 −1.596, rl_s44 **+0.422** (inverted).

**Per member, at the end**, for the two seeds a zero-punishment detector would have called healthy: seed 44 has 40/40 members with a positive slope and seed 46 has 39/40 — but the slopes are 0.002 to 0.027 and −0.202 to +0.003 respectively. The sign is arithmetically defined and substantively meaningless; these are flat policies with rounding. Against generation 0's 24/40 human-sign and a spread of 4.7 slope units, the population has not picked a side, it has stopped having sides.

## R3. A silent failure in the reference rows, found before it reached a table

Recorded before R4's numbers because it nearly corrupted them, and because the
failure mode generalises.

The cross-evaluation config began as a copy of
`24_rl_new_clones_cross_eval.yml`, reference rows and all: `never`,
`thr9_p10`, `prop10`. The simulation ran, produced a clean
`per_round.parquet`, and raised nothing. The numbers were nonsense:

| row | mean punishment | max | fraction exactly 0 |
|---|---|---|---|
| `never` | **2.569** | 20 | 0.655 |
| `thr9_p10` | 2.576 | 20 | 0.658 |
| `prop10` | 2.431 | 20 | 0.693 |

A manager called `never` punishing a mean of 2.57, and three supposedly
different rules agreeing to two decimal places.

**The cause.** This branch starts at `0ff44a9`, and at that commit
`api_manager.RuleBasedManager` is a *single fixed formula*,
`clamp((20 − contribution − round_number) / k, 0, 30)`, with the constructor
`(self, k=1, n_punishments=31, **_)`. The named rules — `never`, `threshold`,
`proportional` — were added **later**, on `auto/rl-manager-two-worlds`. So
`rule: never` was swallowed by `**_`, all three rows silently ran the same
k=1 shortfall manager, and nothing in the stack objected. A config written
against a sibling branch's vocabulary is accepted verbatim by a tree that does
not have it.

**What caught it** was not the guard I had written. It was the arithmetic
cross-check that three ES seeds ended punishing exactly 0.000 and so *are*
never-punish managers, which meant a separate `never` row had to agree with
them and did not. Structural redundancy caught what an assertion did not,
which is an argument for building tables that contain their own controls.

**The fix**, and it makes the table better rather than merely correct. The
references are now built only from what this tree has, and chosen so the
learned policies have exact analogues:

- `never` = `DummyManager(0)` — the floor, and the analogue of seeds 42, 43, 45;
- `flat1` = `DummyManager(1)` — the exact analogue of **es_s46**'s learned policy;
- `flat2` = `DummyManager(2)` — the exact analogue of **es_s44**'s learned policy;
- `shortfall_k1`, `shortfall_k4` = `RuleBasedManager` at k = 1 and 4, named for
  what the formula computes rather than borrowed from a branch this tree is not
  on. Both punish *less* the more a player contributed, which is the human
  direction, so they are the targeted end of the ladder.

The flat rows are the control that decides R4's central question: if es_s44's
leaver differential equals `flat2`'s and es_s46's equals `flat1`'s, then those
two seeds are doing nothing beyond a flat tax, and the "it never collapsed"
reading of R1 is dead on its own terms.

`scripts/rl_es/leaver_selection.py` now refuses to print a table whose
reference rows fail their contract (`REFERENCE_CONTRACTS`): a row called
`never` must punish 0 everywhere, `flat1` must only ever punish 0 or 1, and so
on. That check would have caught this in one second instead of one
simulation.

## R3b. Who leaves — the targeting direction, read off behaviour

**Measurement queued, not yet returned.** The corrected cross-evaluation
(`configs/simulation/manager_testing/25_rl_es_cross_eval.yml`) is in the Raven
queue. When it lands, run:

```
python scripts/rl_es/leaver_selection.py \
    plots/simulation/25_rl_es_cross_eval/per_round.parquet
```

It will refuse to print if the reference rows fail their contract (R3), so a
table that appears is a table that can be read.

What the table decides, stated in advance so the reading is not chosen after
seeing it:

* the three zero-punishment seeds (42, 43, 45) must land on `never`. They are
  never-punish managers by construction, so any gap is a bug, not a finding;
* **es_s44 must land on `flat2` and es_s46 on `flat1`.** If they do, those two
  seeds are doing nothing beyond a flat tax and R1's "never collapsed" reading
  is dead on its own terms. If they differ, the flat tax is doing something a
  constant cannot, which would be the one genuinely surprising outcome left in
  this arm;
* all five should sit well short of `shortfall_k1`/`shortfall_k4`, the
  targeted end of the ladder, and none should be positive. A positive value
  would mean the manager sheds contributors rather than free-riders, which is
  the inversion the DQN arm showed and which a flat policy cannot produce.

An earlier run of this measurement, on the config whose reference rows were
silently broken (R3), gave the ES rows −1.530 (s42), −1.174 (s43), −1.655
(s44), −0.990 (s45), −1.789 (s46) and lin_punisher −1.556. Those ES and
linear rows were unaffected by the defect — the ES punishment levels in that
simulation matched the training evaluation exactly, 0/0/1.963/0/0.980 — so
they are recorded here as an indication. They are **not** the result: the
manager set changed between the two configs and MultiManager's RNG draw
depends on it, so the corrected run supersedes them. Worth noting even so:
the three zero-punishment seeds averaged −1.23 there, against the −1.20
quoted independently for never-punishing on this world.

## R4. The five pre-registered predictions, scored

`prediction_scorecard.csv`. One wrong, four right, and **the wrong one is the informative one.**

**P1 — WRONG.** I predicted all five seeds converge to punishment 0.000 between generations 100 and 300. Only 3 of 5 reach ~0 at all (final levels 0.00, 0.00, 1.80, 0.00, 0.92), and the three that do reach it at generations 3287, 3987 and 3991 — none inside the window. What actually happens in generations 187–320 is that the *shape variance* and the *fitness signal* die, while the *level* freezes wherever it happens to be. I predicted the right window and the wrong quantity.

**P2 — RIGHT.** The shape is identically flat and the human sign is not recovered: slopes 0.0000, 0.0000, +0.0038, 0.0000, +0.0019. Right for a reason only half anticipated — I expected flat *at zero*, and two seeds are flat at a non-zero level.

**P3 — RIGHT.** Seed spread in final return is smaller than the DQN arm's: ES reward sd **3.13** (range 7.49, n=5) against DQN **9.02** (range 17.61, n=3). With the asterisk I wrote in advance, and it is now a heavier asterisk: five copies of "no contingency" agreeing with each other is not evidence that consistent behaviour reduces seed spread in general.

**P4 — RIGHT, and trivial, as registered.** Late per-bin behaviour shift 0.008 to 0.071 (mean 0.02), `slope_vs_uniform_pull` 0.0005 to 0.0014 — against the control arm's 0.886 / 0.891 and the annealed arm's 0.067. A factor of ~1000 below the control on the discriminator. Under the corrected framing this describes what was sampled, and it is near zero partly *because* the evaluated policy is flat.

**P5 — RIGHT, and must not be read as a win.** ES mean evaluated reward **57.69** (60.0, 59.3, 52.5, 59.6, 57.1) against DQN **50.06** (40.1, 57.8, 52.3). This is the three-rung ladder: indiscriminate punishment < no punishment < targeted punishment. ES climbed from the bottom rung to the middle one and stopped. It did not reach the rung the humans are on.

Final evaluated metrics, mean of the last 10 evaluation rollouts (`final_evaluated_metrics.csv`):

| run | contribution | next_reward | punishment |
|---|---|---|---|
| es_s42 | 7.875 | 59.96 | 0.000 |
| es_s43 | 7.799 | 59.32 | 0.000 |
| es_s44 | 8.707 | 52.47 | 1.800 |
| es_s45 | 7.860 | 59.62 | 0.000 |
| es_s46 | 8.416 | 57.08 | 0.916 |
| dqn_s42 | 7.327 | 40.14 | 1.615 |
| dqn_s43 | 8.348 | 57.75 | 1.127 |
| dqn_s44 | 8.429 | 52.28 | 1.588 |

Worth reading across those rows: the flat tax **does** raise contributions — seed 44 punishes 1.80 flat and gets contribution 8.71 against the zero-punishment seeds' 7.86, a gain of 0.85. It still loses on the pool, because 1.6 × 0.85 = 1.36 of gain does not cover 1.80 of punishment. An untargeted tax buys contribution at a price the common pool cannot pay. That is the local landscape this arm climbed, measured from inside it.

## R5. What the plateau finding does and does not now support

The maintainer's simulation of deliberately inverted and near-ceiling rules has since measured the landscape I inferred from two probes: a level-matched inverted rule loses 32.75 pool points to never-punishing, while a correctly-targeted rule is indistinguishable from it. **Inside the wrong-direction family, the best attainable policy really is to punish nothing at all.**

That supports my reading in one direction and undercuts a stronger reading I should not make:

- **Supported.** Never-punishing is a genuine local rung, not an artefact of my sigma or learning rate. An optimiser that has no contingency available to it *should* climb to zero punishment, and three of five seeds did exactly that. The arm's behaviour is consistent with the landscape rather than with a defect in the method.
- **Not supported.** I cannot claim the method *failed* to find targeting. The measured landscape says a correctly-targeted rule is only *indistinguishable from* never-punishing on the pool, not better than it. If the return difference between correct targeting and no punishment is within noise, then **there is no fitness gradient toward the human shape for a return-maximising method to follow**, and no amount of exploration or population would supply one. On that reading the flat outcome is the right answer to the objective as posed, and the shape's absence is a property of the objective, not of evolution strategies.

Those two possibilities — the noise floor swallowed a real signal, versus there is no signal — are **not separated by these runs**, and R1's signal-to-noise of ≈ 1.0 is exactly what both look like. Separating them needs the return difference between a targeted and an untargeted policy measured against the per-member noise, which is a simulation question rather than a training one.

It also remains true, and is now the third live possibility, that the artificial humans respond to punishment without regard to desert, in which case the contingency is arbitrary for reasons unrelated to exploration or to the objective. This arm does not touch that.

---

# The claim, and the objection to it

## The objection, stated first because it is correct

The brief I was given framed the behaviour-versus-evaluated gap as a defect: the DQN runs' behaviour policy punishes 1.7 to 6.6 times as hard as the policy being evaluated, and that was offered as a candidate explanation for the seed spread.

**That framing is wrong and should not be repeated.** DQN is off-policy. A behaviour policy that differs from the target is what the algorithm is *for*: Q-learning bootstraps toward the max over actions and evaluates the greedy policy whatever collected the data. Broad, decorrelated action coverage is exploration doing its job. The existence of a gap is not evidence of anything, and neither is its size.

## What survives, and it is the claim this arm is strongest on

Off-policy correction buys the correctness of the **action choice given a state**. It does not supply **states the behaviour policy never visits**. And here the state distribution is endogenous to the manager's own behaviour, through two channels that are both in the code and neither of which is a modelling choice I made:

- the contribution model is recurrent (`GraphNetwork.rnn_n`, a GRU whose hidden state is carried across all 24 rounds), so a punishment at round 3 moves every later round's contributions through that hidden state, not only round 4's;
- group composition responds to punishment through the switch predictor, so *who the manager is even paid on* at round 20 depends on what it did at round 3.

The sharpest version is **trajectory coverage**. What a consistently contingent manager produces across 24 rounds is a trajectory that dithering never generates cleanly, so a value function trained on dithered data never sees the returns of a coherent contingent policy. This method has no value function and no one-step backup at all: it scores whole-episode returns of fixed policies, which is exactly the object the argument says is missing.

That is the justification for this arm. M8 measures the trajectory divergence rather than asserting it, and marks precisely what is and is not measured.

## What this arm cannot rule out

If the artificial humans respond to punishment **regardless of whether it was deserved**, then no targeting choice changes the return, the contribution → punishment contingency is arbitrary for reasons that have nothing to do with exploration, and every arm of this comparison is answering a question the environment cannot support. **This arm cannot distinguish that from the trajectory-coverage story.** A separate probe of the desert-sensitivity of the contribution response is running (`contrib_iv`, `contrib_ro` on the cluster); this arm's result must be read against it, not instead of it.

## The primary outcome is shape, not return

Human managers are monotone decreasing in the contributor's own contribution — 4.76 at contribution 0 falling to 0.27 at 20, punish the free rider and leave the full contributor alone. Two of the three finished DQN seeds came out **inverted**: s42 runs 0.08 → 5.00 and s44 runs 0.23 → 2.00, monotone in the wrong direction across all six bins. The question this arm exists to answer is whether scoring coherent whole-episode policies recovers the human shape when dithered one-step learning does not.

Return is the fourth question, after shape, the gap and the seed spread. **If evolution strategies underperforms on return but recovers the shape, that is the result and it leads.**

## Contract (identical in all four arms, not mine to choose)

- Branched from `auto/rl-manager-two-worlds` at `0ff44a9`. `reward_mode: common_pool`, reward untouched.
- Same four model artifacts and the same opponent as `configs/training/rl_manager/rl_new_clones_s42.yml`. Not retyped: `scripts/rl_es/make_configs.py` lifts the keys out of that file and `scripts/tests/test_rl_es_configs.py` asserts equality on all nine shared keys.
- Env unchanged: n_groups 2, n_agents 8, agent_groups [0,0,0,0,1,1,1,1], rl_group_id 0, switch_every 4, n_rounds 24, n_contributions 21, n_punishments 31, batch_size 1000.
- Seeds 42, 43, 44, 45, 46. Evaluation identical in all arms. Own remote dir `~/repros/ai-runs/rl-es`.

---

# Measured

## M1. The episode budget: 4,200,000, and it does not trade against population size

Evidence: `plots/data_analysis/evaluation/rl_manager_es/budget.json`, `scripts/rl_es/pilot.py --mode budget`.

Counted by instrumenting `ArtificialHumanEnv.step` over a **real, shortened training call** and dividing by `n_rounds`. Not by counting `reset`: the env's constructor calls `reset` once without playing an episode, so a reset count overstates the budget by one rollout per run.

Over 6 generations: 144 behaviour `step` calls = 6 rollouts = **6,000 behaviour episodes = 1,000 per generation**, plus 48 evaluation `step` calls = 2 rollouts, matching the 2 expected at that eval period. Projected to the full config: **4,000,000 behaviour + 200,000 evaluation = 4,200,000**, which is exactly the DQN arm's 4000 × 1000 + 200 × 1000.

**The sibling arms were told I would have to trade generations against population size to hit this number. I do not, and this is the measurement that shows it:** `rollouts_per_generation = 1.0` and `independent_of_population_size = true`. The whole population is scored inside one rollout (M2), so a generation costs 1,000 behaviour episodes whether the population is 10 or 100. `n_generations = n_update_steps = 4000` therefore matches the budget exactly, with no rounding and no trade. This is the one place the arm could be accused of an unfair comparison, so it is instrumented rather than argued.

**`update_step` is the generation index.** For both arms it now means, and means only,

```
update_step = (behaviour episodes consumed so far) / 1000
```

It is not a gradient step — this method has none. The training log prints the mapping at startup under `[ES] AXIS NOTE` and the generated configs carry it in a header comment. `q_mean`, `q_min` and `q_max` are **absent** from this arm's parquet rather than renamed: there are no Q values, and putting something else in those columns would corrupt the one place a cross-arm reader will not think to check.

## M2. Batched population evaluation: it works, batch-partitioned rather than weight-vectorised

The question was whether a population can be evaluated inside one batched rollout by assigning members across the batch dimension. **Yes.** What I found reading `manager/manager.py`, `manager/environment.py` and `generic/graph.py`:

- Episodes never interact. `GraphNetwork.create_fully_connected` wires each episode's 8 agents to each other and to nobody else, and every pooling operation is a `scatter` over that index. A contiguous slice of the batch dimension is therefore a self-contained set of episodes that a different policy can drive without leaking.
- A single forward pass cannot carry 40 parameter vectors: the `GraphNetwork` shares weights across nodes. Torch is 1.11.0 on Raven, so there is no `torch.func.functional_call`/`vmap` to vectorise an ensemble with, and hand-writing a weight-batched forward would mean a second copy of op1, the GRU cell, op2 and the bias head that can silently drift from `graph.py`.
- Recurrent state is per-instance (`self.rnn_n_h0`), so P cloned networks each carry their own memory with no juggling.

So: member p owns episodes `[p*E, (p+1)*E)`, the manager head runs P times per round on 1/P of the batch, and the environment — the three artificial-human GNNs and the opponent punisher — runs **once** over the whole batch, exactly as in the DQN arm. That is where the win is, and M4 shows it makes the arm *cheaper* than the one it is compared with.

Two consequences recorded rather than assumed:

1. Each member sees its own episodes, so fitnesses are independent draws. **Common random numbers across mirrored pairs are not available**: the trajectories diverge as soon as two members punish differently, so there is nothing to hold fixed. This is the largest source of fitness noise and M5 measures it.
2. `GraphNetwork.encode` rebuilds the edge index in Python on every call (112,000 index pairs at full batch). It depends only on the shape, so `PopulationRollout` builds it once per shape and reuses it. Guard G2, which demands bit-equality against `run_batch`, is what proves that changed nothing.

## M3. Guards

Evidence: `guards.json`. Asked of the real world built from the real config. All three pass.

**G1, paired start.** 78,532 parameters. The vector built by `es_manager.build_world` and the vector built by the DQN arm's construction sequence are **identical, max absolute difference 0.0**, under seed 42. The five-seed comparison is paired, not merely matched: ES seed 42 and DQN seed 42 start from the same policy.

**G2, the evaluation is the other arms' evaluation.** A single-member population rollout against `rl_manager.run_batch(..., on_policy=True)`, same env, same seed, batch 100, 24 rounds, 11 metrics per round: **264 comparisons, max absolute difference 0.0**. Bit-identical, not close. Pinned as a test at `rel=0, abs=0`.

**G3, no action noise, checked with teeth.** The deterministic policy called twice on one state returns the same action tensor. The DQN arm's epsilon-greedy selection called twice does **not**, moving 86 of 800 cells off greedy in a single call and lifting mean punishment from 7.00 to 8.02. The second half is what makes the first worth anything.

## M4. Cost: this arm is cheaper than the arm it is compared with

Evidence: `cost.csv`. The DQN step is measured in the same session on the same A100 rather than quoted.

| arm | population | s/step | vs DQN | hours for 4000 |
|---|---|---|---|---|
| DQN (rollout + replay write + update) | – | 5.153 | 1.00 | 5.73 |
| ES | 10 | 1.794 | 0.35 | 1.99 |
| ES | 20 | 2.292 | 0.45 | 2.55 |
| **ES** | **40** | **3.313** | **0.64** | **3.68** |
| ES | 100 | 5.851 | 1.14 | 6.50 |

The DQN row reproduces the quoted six-hour estimate (5.73 h), which is the check that the harness measures the right thing. The chosen configuration is **0.64× the DQN arm**, which I did not expect before measuring: the environment runs once per rollout whatever the population size, and this arm does no replay write and no backward pass. Cost grows sub-linearly in the population (10 → 100 is 10× the manager forwards for 3.3× the wall clock), the signature of a rollout whose fixed cost dominates. The 20-hour SLURM limit is now 5.4× the expected run time.

## M5. Signal against noise, at the initialisation

Evidence: `noise.csv`. 40 members, 25 episodes each, two repeats per sigma.

```
signal_to_noise = (sd of fitness ACROSS members) / (standard error of one member's own fitness)
```

| sigma | fitness mean | **signal/noise** | member punishment mean |
|---|---|---|---|
| 0.005 | 778.6 / 738.4 | 1.24 / 1.45 | 5.42 / 5.69 |
| 0.01 | 716.4 / 629.2 | 1.69 / 1.96 | 6.09 / 6.71 |
| 0.02 | 637.7 / 513.6 | 2.38 / 2.48 | 6.99 / 8.02 |
| 0.05 | 348.2 / 406.5 | 3.54 / 4.19 | 10.65 / 9.73 |

Above 1 everywhere, so the arm is not dead on arrival at the initialisation. **But the initialisation is the least demanding point, and reading sigma off this table alone would have been a mistake — see M7.**

An unplanned finding that sets expectations: the Spearman correlation between a member's fitness rank and its mean punishment is **strongly negative at every sigma** (−0.87 and −0.90 at sigma 0.02, −0.95 at 0.05). At the initialisation, members that punish *less* score *higher*, consistently. The untrained policy punishes about 5 to 7 points per member per round indiscriminately and the reward is the common pool, which punishment subtracts from directly. **The first thing this arm does is drive the punishment level down, and that fall must not be read as the method failing — nor as an improvement.**

## M6. Generation-0 policy shape: the population already disagrees about the sign

Evidence: `shape_generation0_*.csv`. Bins are the evaluation suite's own `RPA_EDGES`, imported from `evaluation_suite/metrics.py` rather than restated, so these columns and `.../rl_manager_two_worlds/policy_shape.csv` are the same bins.

Untrained theta is **flat**: a near-constant 5.0 across all six bins (5.15, 5.07, 4.68, 5.00, 5.00, 5.00; slope −0.019) on 8,662 to 19,915 agent-rounds per bin.

| | all cells | valid cells only |
|---|---|---|
| members with the human sign (slope < 0) | **24 / 40** | **27 / 40** |
| members with the inverted sign (slope > 0) | **16 / 40** | **13 / 40** |
| slope range across members | **−1.574 to +3.101** | −1.578 to +3.079 |
| population pooled | −0.188 | −0.201 |

Against the finished runs, same bins, same slope definition (least squares across the six bin means):

| manager | slope | sign |
|---|---|---|
| human managers | −0.857 | human (decreasing) |
| lin_punisher (the clone) | −0.659 | human |
| rl_s42 (DQN) | **+1.218** | inverted |
| rl_s43 (DQN) | −1.596 | human |
| rl_s44 (DQN) | **+0.422** | inverted |
| ES theta, generation 0 | −0.019 | ~flat |

**A single generation of untrained perturbations spans −1.57 to +3.10, which brackets the entire range the three finished DQN seeds landed in (−1.60 to +1.22).** Read carefully: this says the raw variation in policy shape is present in the parameter noise before any selection happens, and it establishes that this population has something to select between. It does **not** say selection cannot remove that variation, and it does **not** say the DQN seeds' shapes came from the same source.

## M7. The never-punish attractor, and the saturation ceiling

**This is the most important pre-launch finding and it largely determines what the runs can deliver.**

Two short training runs through the real entry point, plus two probes of the policy they reach.

**The run drives punishment to exactly zero and stays there.** At the paper's lr 0.01, the 40-generation pilot went from mean punishment 5.68 to **exactly 0.000 by generation 7**, with fitness rising 629 → 1480 and eval reward 35.9 → 59.2. At generations 10, 20 and 30, **0 of 40 members punished anything at all**, `fitness_std` (194–250) sat on top of `within_member_se` (192–211) — signal to noise ≈ 1.0 — and `theta_norm` climbed 18.08 → 26.63, a pure random walk at 0.21 per generation.

**Never-punish is not the optimum.** The rule-based sweep on the sibling branch measures, on this same world: `prop10` (punish in proportion to the shortfall) reaches common good 131.76 against `never` at 98.20, a contrast of **+33.55 [23.70, 43.33]**. Indiscriminate heavy punishment is worse than never (`thr19_p10`, 90.38), but *targeted* punishment is decisively better. So the landscape is **indiscriminate < never < targeted**, and the arm has found the middle rung and stopped.

**Why it stops: the policy reads a discrete argmax over 31 ordinal levels.** Once the Q-gap between action 0 and action 1 exceeds what a perturbation of scale sigma can bridge, every member implements the identical zero policy, every fitness is the same policy scored on different episodes, and the ranking is noise. Measured at two different reached policies:

| | sigma 0.02 | sigma 0.05 | sigma 0.1 | sigma 0.2 |
|---|---|---|---|---|
| **θ reached by random walk** (lr 0.01, 40 gens) | 40/40 dead, s/n 0.86 | 22/40 dead, s/n 2.00, fitness 1279 | 1/40 dead, s/n 4.73, **fitness 530** | 0/40 dead, **fitness 221** |
| **θ reached by selection** (lr 0.001, 200 gens) | 40/40 dead, s/n 0.86 | **36/40 dead, s/n 0.84** | 3/40 dead, s/n 5.42, **fitness 506** | 0/40 dead, **fitness 205** |

("dead" = members whose mean punishment is exactly 0; the undisturbed policy's fitness is ~1420.)

**There is no sigma in [0.02, 0.2] that both keeps the population non-degenerate and preserves the policy.** At 0.05 the population is 90% dead at the selection-reached policy; at 0.1 it is alive but the perturbation has cut fitness by 64%, which means the sigma-smoothed objective is no longer the objective. The gap is hard, and it is a property of pairing a perturbation-based method with a saturating discrete readout, not of this particular sigma.

**What lowering the learning rate bought, and what it did not.** At lr 0.001 the same first move takes ~130 generations instead of 7 (behaviour punishment 8.42 → 6.64 → 5.42 → 4.38 → 2.98 → 1.33 → 0.215 at generations 0/20/40/60/80/100/120), fitness rises 343 → 1446 and eval reward 35.9 → 63.7. So the collapse is 18× slower and the parameter random walk is √10 smaller. **It did not prevent the collapse**, and by generation 140 the shakeout is back on the plateau. This is launched anyway, for the reasons in "Predicted" below.

Because a pilot cannot say when in a 4000-generation run this happens, the plateau detector is now logged **every generation** in `<job>_generations.parquet`: `members_punishing_nothing`, `member_punishment_sd` and `signal_to_noise`.

## M8. Trajectory coverage — the claim of the corrected framing, measured

Evidence: `trajectory_coverage.csv`, `scripts/rl_es/trajectory_coverage.py`.

Both arms log per-round state summaries for a behaviour rollout and an evaluation rollout at the same `update_step`. Per round, the gap between the two on each summary, scaled by the evaluated trajectory's own across-round spread so the summaries can be averaged, over the second half of training:

| run | state trajectory distance | contribution gap | group size gap | common good gap |
|---|---|---|---|---|
| DQN s42 | **0.988** | 0.293 | 0.399 | 0.875 |
| DQN s43 | **1.284** | 0.360 | 0.434 | 0.804 |
| DQN s44 | **1.486** | 0.287 | 0.418 | 0.963 |
| ES shakeout (200-generation pilot) | *0.679* | 0.204 | 0.216 | 0.427 |

**MEASURED:** over the second half of DQN training, the states its replay buffer is filled with sit about **one full across-round standard deviation** from the states its evaluated policy visits — in contribution, group size and common good alike. That is the trajectory-coverage claim, in numbers, for the arm the objection was raised about.

**NOT MEASURED:** the divergence between the two *state distributions*. These are batch means over 1000 episodes and say nothing about spread, the joint distribution, or rarely-reached regions. A small distance here is consistent with very different coverage.

**NOT MEASURED:** that trajectory coverage causes any difference in what is learned. This describes the data each arm trained on, nothing more.

**NOT A RESULT:** the ES row is a 200-generation pilot over 5 logged steps, italicised for that reason. The comparable number comes from the launched runs.

## M9. The per-bin behaviour shift — a description of what was sampled

Evidence: `scripts/rl_es/behaviour_shift.py`. Under the corrected framing this is a **description of what was sampled, not of what was learned**, and near-zero here is a triviality rather than a finding. It is reported because it is the clean end of the scale and makes the other three arms readable against it.

The sibling arms' prediction with no free parameters: epsilon-greedy over 31 uniform levels drags each bin toward the uniform mean of 15 by `eps × (15 − evaluated)`. On the control that holds, regression slope 0.891, mean absolute per-bin shift 0.886; the annealed arm cuts the shift to 0.067.

This arm on the 200-generation shakeout, `slope_vs_uniform_pull` being the discriminator:

| generation | mean abs shift | slope vs uniform pull | evaluated mean punishment |
|---|---|---|---|
| 40 | 0.814 | **+0.079** | 4.94 |
| 60 | 0.669 | **−0.033** | 4.94 |
| 140 | 0.031 | +0.002 | 0.00 |
| 180 | 0.008 | +0.001 | 0.00 |

Against the control's 0.891, the pre-collapse slopes of +0.08 and −0.03 are the honest comparison: the shift exists (the members hold perturbed weights) but has no pull toward 15 and flips sign between generations. The near-zero late rows are near-zero **partly because the policy is at zero punishment**, which makes the statistic trivially small; they should not be quoted as the arm's number. The script also reports a demeaned slope and an `evaluated_bin_spread` column, because with a flat evaluated policy the raw slope degenerates to (level shift)/(15 − level) and tests nothing.

## M10. Tests

| suite | where | result |
|---|---|---|
| `src/aimanager/tests/test_es_manager.py` | Raven | 14 passed |
| `src/aimanager/tests/test_es_rollout.py` | Raven | 4 passed |
| `scripts/tests/test_rl_es_configs.py` | local | 9 passed |

Against the brief's list: *the perturbation* (antithetic structure exact, even-population precondition, reproducibility from a seed, successive generations differ — a generator reset per generation would silently score the same population 4000 times); *the fitness aggregation* (centered ranks an even grid summing to zero, invariant to return scale, estimator points up a known gradient at cosine > 0.9, plus the negative control where raw-return weighting lets one outlier capture the direction at cosine > 0.99); *the deterministic evaluation of the mean parameters* (perturbed members never leak into the evaluation rollout, and the bit-equality with `run_batch`); *reproducibility from a seed* (identical per-episode returns and metric rows through the real environment, different for a different seed); plus the RPA binning against `pd.cut` and the shape accumulator against hand-computed answers.

Beyond the unit tests, the whole path was run end to end through `python -m aimanager train-manager-es` twice (40 and 200 generations) before the seeds went out, because the guards exercise `PopulationRollout` but not the Adam step, the parquet writing or the model save.

---

# Decisions, and why

**Evolution strategies rather than the cross-entropy method.** CEM keeps the top-k and refits; ES uses every member, weighted by rank. With 40 members in 78,532 dimensions, discarding 75% discards most of the little information a generation carries; and CEM's hard elite threshold on a fitness this noisy admits a member that made the cut on a lucky draw of group sizes at full weight, where centered ranks give it its neighbour's weight.

**Population 40, 25 episodes each.** The partition must be exact (1000/P integer) and P even for mirrored sampling. More members means more directions but a noisier fitness each. 40 gives signal-to-noise 2.4 at the initialisation and is priced at 0.64× the DQN step. The alternatives are in `cost.csv` and `noise.csv` rather than asserted.

**Mirrored sampling: yes.** Each pair becomes a two-sided finite difference and theta's own fitness level cancels. With P=40 the mean of 40 independent perturbations is not small, so without mirroring the estimator carries that level as bias.

**Centered-rank shaping: yes, and skipping it would have needed a defence I do not have.** Beyond the usual argument, there is a specific one here: the episode return's largest variance component is group size, so a member scored on an unusually large group posts a return far above the rest for reasons unrelated to its parameters. `test_raw_fitness_weighting_is_captured_by_one_outlier` shows the raw estimator's cosine similarity to that one member's perturbation exceeding 0.99. Ranks bound every member's influence to 1/P.

**Fitness = the undiscounted sum of the RL group's per-round common pool over 24 rounds, averaged over the member's episodes.** The natural ES objective and what the evaluation reports. The DQN arm optimises the discounted version (γ=0.98) but is *read* on the same per-round reward. Stated because it is a real difference between arms that a reader should not have to infer.

**sigma 0.05, not the paper's 0.02.** Chosen on M7, not on M5. At the policies the run actually reaches, 0.02 leaves all 40 members implementing the identical zero policy with signal-to-noise 0.86. 0.05 is the largest sigma that does not itself destroy the policy (fitness 1415 against the undisturbed 1423, versus 506 at sigma 0.1). It is measurably insufficient at the deepest saturation — that ceiling is M7 and is stated as a limitation rather than hidden.

**lr 0.001, not the paper's 0.01.** Arithmetic, not taste. Adam normalises per coordinate, so a step is ‖Δθ‖ ≈ lr·√78532 = 2.8 at lr 0.01 against ‖θ‖ ≈ 18 — a 15% relative change per generation driven by a rank vector that is at best half signal. Over 4000 generations the random-walk component alone inflates ‖θ‖ about tenfold, which shrinks sigma *relative to the weights* and re-creates the degeneracy sigma was raised to avoid. The measured `theta_norm` trajectory (0.21 per generation at lr 0.01) confirms the arithmetic. At lr 0.001 the collapse takes ~130 generations instead of 7.

**L2 0.005**, the paper's, kept — with the caveat that at this gradient scale (‖grad‖ ≈ 650, l2·‖θ‖ ≈ 0.09) it does essentially nothing. The learning rate is what bounds the walk. Recorded so nobody credits the L2 term with the bounding.

**A dedicated RNG stream for the perturbations** (`seed + 10007`), so the noise a generation sees is a function of the seed and generation index alone and stays so if the environment's RNG consumption changes.

**The initial parameter vector is the DQN arm's**, via `build_world` reproducing `rl_manager.train_manager`'s construction order (the env constructor draws round 0's contributions and so consumes RNG before the networks are built) and the manager being built with the DQN arm's full argument set, target network included. Guard G1 checks equality.

**A device workaround, not a fix.** `ArtificalManager.load` ignores the device it is handed: `save` moves the model to CPU and `load` assigns straight through, so loading onto cuda yields CPU weights with a cuda `self.device` and dies on the first forward. Worked around with an explicit `.to(device)` at each of this arm's call sites rather than by changing the manager, because the other three arms share that file.

---

# Predicted, in advance, so it can be scored

Not results. Written before any seed finished.

1. **All five seeds converge to mean punishment 0.000 and stay there**, reaching it between generations 100 and 300. `members_punishing_nothing` goes to 40/40 or close and `signal_to_noise` falls to ≈ 1.
2. **The policy shape is therefore identically flat**, slope 0, and the arm does **not** recover the human sign. Not because the human sign is unreachable, but because the optimiser reaches never-punish first and the argmax saturates behind it.
3. **Seed spread in final return is small** — much smaller than the DQN arm's — because all five land on the same degenerate policy. That is a confirmation with an asterisk: zero spread among five copies of never-punish is not evidence that consistent behaviour reduces seed spread in general.
4. **The behaviour-versus-evaluated per-bin shift ends near zero**, which under the corrected framing is a triviality.
5. **Return beats the DQN seeds** (eval reward ~59–64 against the DQN arm's evaluated policies), because never-punish beats indiscriminate punishment. **This must not be reported as ES winning.** It is ES finding the middle rung of a three-rung ladder faster.

If any seed escapes the plateau and produces a non-flat shape, predictions 1–3 are wrong and that seed is the most interesting object in the whole comparison.

# The honest risk, as it actually turned out

I wrote before launch that evolution strategies ignores within-episode credit assignment and might underperform for reasons unrelated to the hypothesis, and that the specific failure I expected was the punishment **level** reaching its local optimum before **targeting** could be explored, with the discrete argmax saturating behind it.

**Half of that is what happened, and the half I got wrong matters.** The level did freeze early. But the binding constraint is not argmax saturation: two seeds (44 and 46) ended with every member still punishing and still varying, and they are *just as flat*. What actually died, in all five seeds within a 100-generation window, is the population's **shape variance**, simultaneously with the fitness ranking's signal-to-noise falling to 1. The method did not saturate against a readout it could not move. It ran out of anything to select on.

That reframes the result. It is not "ES underperforms because it ignores credit assignment", and it is not "ES is defeated by a discrete readout". It is: **with 25 episodes per member, the return difference between a contingent policy and a flat one is below the sampling noise, so rank-based selection has nothing to rank and the contingency decays.** Whether that is because the signal is small or because there is no signal — R5 — is the open question, and these runs do not settle it.

The pilot caution from the annealed arm held up and is worth repeating: **the inversion is a late-training phenomenon**, and nothing in M6, M7 or M9 was evidence about whether this arm fixes it. The final answer (R2) is that this arm neither fixes nor reproduces the inversion, because it produces no contingency of either sign.

---

# Successor

Ordered by what the finished data now says, not by what I expected before it.

1. **Read R1's two detectors together, and use the shape one.** `shape_degeneracy.csv` is the detector that works; `collapse.csv` alone would have told you two seeds were healthy when their policy is a flat tax. Everything after generation ~320 in any seed is a random walk on the level and must be read as such — including the late punishment collapses at 3287/3987/3991, which are not optimisation.

2. **Settle R5, because it decides what this whole comparison means.** The measured landscape says a correctly-targeted rule is *indistinguishable from* never-punishing on the pool. If that is literally true, there is no return gradient toward the human shape and **no exploration method can find it** — in which case all four arms of this comparison have been measuring the wrong thing, and the finding is about the objective, not about exploration. The test is a simulation, not a training run: take a correctly-targeted rule and a level-matched flat rule, run enough episodes to put a tight interval on the pool difference, and compare that difference to the per-member sampling noise at 25 episodes (within-member se ≈ 190–210 on the episode return, from `collapse_trajectory.csv`). If the true difference is inside that noise, the noise floor is the whole story.

3. **If there IS a signal, the lever is episodes per member, not sigma and not the population.** This is arithmetic: signal-to-noise scales as √(episodes per member). These runs sat at s/n ≈ 1.0 with 25. Reaching s/n ≈ 2 needs 100 episodes per member, which at a fixed 4,000,000-episode budget is 10 members × 100 episodes over 4000 generations, or 40 members × 100 episodes over 1000 generations (four rollouts per generation). Both are one config change; `es_manager.PopulationRollout` already partitions the batch by member and nothing else has to move. **That is the experiment I would run next**, and it is a better use of a GPU day than a fifth exploration rule.

4. **The non-saturating readout is still worth doing, but it is now second, not first.** My pre-launch recommendation was an ordinal head or a monotone scalar intensity so that small parameter changes make small action changes. Seeds 44 and 46 show saturation was not the binding constraint, so this buys less than I claimed — but it would still remove one confound from (3), and it is a manager-architecture change outside this comparison's one-variable contract, so it belongs in its own experiment either way.

5. **The rival explanation this arm still cannot rule out.** If the artificial humans respond to punishment regardless of desert, the contingency is arbitrary for reasons unrelated to both exploration and the objective. Hold a group's total punishment fixed, vary only *which* member receives it, read the contribution response. A probe is running (`contrib_iv`, `contrib_ro`). Its result is a precondition for interpreting (2).

6. **Reproducing this analysis.** `scripts/rl_es/collapse.py`, `shape_degeneracy.py`, `policy_shape.py`, `leaver_selection.py`, `behaviour_shift.py`, `trajectory_coverage.py`, `score_predictions.py`. The cross-evaluation simulation is `configs/simulation/manager_testing/25_rl_es_cross_eval.yml`, which re-runs the clone and the rule rows inside its own file rather than quoting them, because MultiManager's RNG draw depends on the config's manager set and two managers are only stream-comparable within one config.
