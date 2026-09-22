# RL manager, exploration comparison: the evolution-strategies arm

One of four arms. The siblings are annealed local epsilon-greedy, bootstrapped DQN and parameter-space noise. Method: evolution strategies in the sense of Salimans, Ho, Chen, Sidor and Sutskever (2017) — mirrored sampling of parameter perturbations, centered-rank fitness shaping, Adam on the fitness-weighted average. There is no action noise anywhere: every policy that is run or scored is a fixed deterministic policy evaluated over complete episodes.

**Status: LAUNCHED.** Five seeds submitted, none finished. Job ids 30413786 (s42), 30413787 (s43), 30413789 (s44), 30413790 (s45), 30413791 (s46), in `~/repros/ai-runs/rl-es`. Everything under "Measured" is a number this branch produced before launch. Everything under "Predicted" is not, and is written down in advance so it can be scored.

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

# The honest risk, restated

Evolution strategies ignores within-episode credit assignment, so on an equal episode budget it may underperform for reasons unrelated to the hypothesis. On the evidence above the specific failure is sharper than that and is *not* a credit-assignment failure: it is that the method optimises the punishment **level** to its local optimum before it can explore **targeting**, and the discrete argmax then saturates. **If that is what the runs show, that is the result, and it is a result about this method on this readout — not evidence against consistency mattering.** The rival explanation in "What this arm cannot rule out" remains live and this arm does not touch it.

A note on pilots, from the annealed arm and worth repeating: at 300 steps both the control and the annealed pilots are monotone decreasing with the human sign and both punish far harder than any human. **The inversion is a late-training phenomenon.** Nothing in M6, M7 or M9 should be read as evidence about whether this arm fixes it, and the fall in punishment level in M7 is not an improvement.

---

# Successor

1. **Read `members_punishing_nothing` and `signal_to_noise` in `<job>_generations.parquet` first.** They say at which generation each seed hit the plateau, and everything after that generation is a random walk that should not be interpreted.
2. **Shape:** `python scripts/rl_es/policy_shape.py --seeds 42,43,44,45,46 --reference plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape.csv`. Report the per-member table as well as theta's; members disagreeing about the sign at the *end* against M6's reading at the start is the cleanest measure of what selection did.
3. **Make the shape numbers comparable across arms before quoting them together.** The in-training shape comes from each run's own evaluation rollouts; the two-worlds numbers come from the cross-evaluation *simulation* (`configs/simulation/manager_testing/24_rl_new_clones_cross_eval.yml`). Pairing and episode counts differ. Add the five ES models to that config and re-run `measure.py` first.
4. **The gap and the spread:** `scripts/rl_es/behaviour_shift.py` and `scripts/rl_es/trajectory_coverage.py`, the latter against the DQN parquets for the cross-arm number. Both mark measured against not measured; keep those marks.
5. **The experiment that would break the tie this arm cannot break.** Hold a group's total punishment fixed and vary only *which* member receives it, then read the contribution response. If it is flat, the contingency has no gradient to correct it, no exploration method will supply one, and all four arms are answering an unsupported question. A probe is already running (`contrib_iv`, `contrib_ro`).
6. **If the plateau is confirmed, the follow-up that is indicated is not another exploration arm.** It is a readout that does not saturate — the saturation in M7 is a property of argmax over 31 ordinal levels, and a policy parameterised so that small parameter changes make small *action* changes (an ordinal regression head, or a scalar intensity through a monotone map) would let a perturbation method keep a live population at any policy. That is a change to the manager architecture, not to the exploration rule, and it is outside this comparison's one-variable contract — which is exactly why it should be a separate experiment rather than smuggled into this one.
