# RL manager, exploration comparison: the evolution-strategies arm

One of four arms testing the same hypothesis: that this task needs *consistent* behaviour and that action-level dithering destroys it. The siblings are annealed local epsilon-greedy, bootstrapped DQN and parameter-space noise. This arm is the sharpest test, because it removes action noise entirely — every policy that is ever run or scored is a fixed deterministic policy evaluated over complete episodes.

Method: evolution strategies in the sense of Salimans, Ho, Chen, Sidor and Sutskever (2017). Mirrored sampling of parameter perturbations, centered-rank fitness shaping, Adam on the fitness-weighted average.

Status: **LAUNCHED**. Five seeds submitted, none finished. Everything below the "Measured" heading is a measurement taken before launch; everything under "Expected" is not.

## The two questions this arm answers

**Primary — policy shape.** The three finished DQN seeds punish, and two of the three punish the wrong people. Human managers are monotone decreasing in the contributor's own contribution (4.76 at contribution 0 falling to 0.27 at 20: punish the free rider, leave the full contributor alone). Seed 42 runs 0.08 → 5.00 and seed 44 runs 0.23 → 2.00, both monotone in the wrong direction across all six bins. Uniform epsilon-greedy over 31 punishment levels applies punishment *independently of the contribution it is aimed at*, which decorrelates the two in the replay buffer; that is a candidate route to a policy whose shape is arbitrary and seed-determined. This arm has no action noise at all, so every policy it scores maps contribution to punishment coherently and is judged on the whole-episode consequences of that mapping. If the decorrelation story is right, this arm should recover the human sign.

**Mechanism — the behaviour-versus-evaluation gap.** Measured on the finished DQN runs, the behaviour policy punishes 1.7 to 6.6 times as hard as the policy being evaluated. Here that gap can only come from the parameter perturbation, and it is logged per generation (`behaviour_eval_punishment_gap` in `<job>_generations.parquet`, and directly comparable from the metrics parquet as `punishment` under `sampling='es-population'` against `sampling='greedy'`).

Return is the third question, not the first. See "The honest risk" below.

## What this arm cannot distinguish, and must not claim to

A competing explanation for the inverted shape is that the artificial humans respond to punishment **without regard to whether it was deserved**. If that is so there is no gradient toward correct targeting, no exploration method can produce one, and a null result here says nothing about the decorrelation story. This arm cannot tell the two apart. Separating them needs an intervention on the contribution model's response — hold the group's total punishment fixed and move only *who* receives it — which is a different experiment and is written up in the successor section.

## Contract (identical in all four arms, not a choice of mine)

- Branched from `auto/rl-manager-two-worlds` at `0ff44a9`.
- `reward_mode: common_pool`. Reward untouched.
- Same four model artifacts and the same opponent as `configs/training/rl_manager/rl_new_clones_s42.yml`. Not retyped: `scripts/rl_es/make_configs.py` lifts the keys out of that file, and `scripts/tests/test_rl_es_configs.py::test_models_env_and_architecture_are_the_reference_config_verbatim` asserts equality for all nine shared keys.
- Env unchanged: n_groups 2, n_agents 8, agent_groups [0,0,0,0,1,1,1,1], rl_group_id 0, switch_every 4, n_rounds 24, n_contributions 21, n_punishments 31, batch_size 1000.
- Seeds 42, 43, 44, 45, 46.
- Evaluation identical in all arms: a fully deterministic rollout, batch 1000, at the same intervals, in the existing long schema.
- Own remote dir: `~/repros/ai-runs/rl-es`.

---

# Measured

Everything in this section is a number this branch produced, with the command that produced it.

## M1. The episode budget, and how it was measured

The DQN config's behaviour consumption is mechanical rather than estimated. `rl_manager.train_manager` runs exactly one `run_batch(on_policy=False)` per update step, and `ArtificialHumanEnv` steps `batch_size` episodes in parallel per rollout:

```
behaviour episodes = n_update_steps x batch_size = 4000 x 1000 = 4,000,000
evaluation episodes = ceil(n_update_steps / eval_period) x batch_size
                    = 200 x 1000 = 200,000
```

**Matched exactly, by construction.** One ES generation scores the entire population inside a *single* env rollout of 1000 episodes (M2), so one generation consumes exactly the 1000 behaviour episodes one DQN update step consumes. Setting `n_generations = 4000` therefore matches 4,000,000 behaviour episodes with no rounding, and `eval_period = 20` puts the 200 evaluation rollouts at exactly the same episode counts. `scripts/tests/test_rl_es_configs.py::test_the_episode_budget_matches_the_dqn_arm` asserts the three equalities.

**The `update_step` column is the generation index.** A reader comparing arms must read it as an episode counter, not a gradient step: this method has no gradient steps on a replay buffer. For both arms,

```
update_step = (behaviour episodes consumed so far) / 1000
```

and nothing else. The training log prints this mapping at startup under `[ES] AXIS NOTE`, and the generated configs carry it in a header comment.

`q_mean`, `q_min` and `q_max` are **absent** from this arm's parquet rather than renamed. There are no Q values here and putting something else in those columns would corrupt the one place a cross-arm reader will not think to check.

## M2. Batched population evaluation: it works, and here is what the code actually allows

The question was whether a population of perturbations can be evaluated inside one batched rollout by assigning members across the batch dimension. **Yes, with one qualification, and the qualification is the interesting part.**

What I found reading `manager/manager.py`, `manager/environment.py` and `generic/graph.py`:

- The environment already runs `batch_size` independent episodes in parallel, and **episodes never interact**: the fully-connected edge index (`GraphNetwork.create_fully_connected`) wires the 8 agents of each episode to each other and to nobody else, and every pooling operation is `scatter` over that index. So a contiguous slice of the batch dimension is a self-contained set of episodes and can be driven by a different policy without leaking.
- The manager's `GraphNetwork` shares its weights across all nodes, so a *single* forward pass cannot carry 40 different parameter vectors. Torch is 1.11.0 on Raven, so there is no `torch.func.functional_call` / `vmap` to vectorise an ensemble with, and hand-writing a weight-batched forward would mean duplicating op1, the GRU cell, op2 and the bias head out of `graph.py` — a second copy of the model that can silently drift from the first.
- The recurrent state is per-instance (`self.rnn_n_h0`), so P independent `GraphNetwork` clones each carry their own memory with no juggling.

So the implementation is **batch-partitioned, not weight-vectorised**: member p owns episodes `[p*E, (p+1)*E)`, the manager head runs P times per round on 1/P of the batch each, and the expensive part — the contribution, validity and switch artificial humans, and the opponent punisher — runs **once** over the whole batch of 1000, exactly as it does in the DQN arm. That is where the win is. The three artificial humans plus the opponent are the environment; the manager head is a small fraction of a round.

Two consequences worth recording:

1. The env's own stochasticity is shared across the generation in the sense that all members are drawn from the same batched sampling call, but **each member sees its own episodes**, so their fitnesses are independent draws. Common random numbers across mirrored pairs are *not* available: the trajectories diverge as soon as two members punish differently, so there is nothing to hold fixed. This is the single largest source of fitness noise and M5 measures it rather than assuming it away.
2. `GraphNetwork.encode` rebuilds the edge index in Python on every call (112,000 index pairs at the DQN arm's full batch). It depends only on the shape, so `PopulationRollout` builds it once per shape and reuses it. The values are whatever `create_fully_connected` returns, so nothing about the forward pass changes — and guard G2 below, which demands bit-equality against `run_batch`, is what proves it.

## M3. Guards (pre-launch, real world, real artifacts)

`scripts/rl_es/pilot.py --mode guards`, evidence in `plots/data_analysis/evaluation/rl_manager_es/guards.json`.

Run on an A100 from `configs/training/rl_manager/rl_es_pilot.yml`. All three pass.

**G1, paired start.** The policy network has **78,532 parameters**. The vector built by `es_manager.build_world` and the vector built by the DQN arm's construction sequence are **identical**, maximum absolute difference **0.0** over all 78,532 entries, under seed 42. So the five-seed comparison is paired: ES seed 42 and DQN seed 42 start from the same policy.

**G2, the evaluation is the other arms' evaluation.** A single-member population rollout against `rl_manager.run_batch(..., on_policy=True)` on the same env, same seed: batch 100 (the guard's size; the unit test uses 4 and the real runs 1000), 24 rounds, 11 metrics per round, **264 comparisons, maximum absolute difference 0.0**. Not "close": every value is bit-identical. The same claim is pinned as a test at `src/aimanager/tests/test_es_rollout.py::test_single_member_rollout_reproduces_run_batch` with `rel=0, abs=0`.

**G3, no action noise, checked with teeth.** Calling the deterministic policy twice on the same state returns the same action tensor (`deterministic_repeat_identical: true`). Calling the DQN arm's epsilon-greedy selection twice on the same state does **not** (`eps_greedy_repeat_identical: false`), and it moves 86 of the 800 cells off the greedy action in one call, lifting the mean punishment from 7.00 to 8.02 on that single round. That second half is what makes the first half worth anything: the check would pass vacuously if it could not detect the noise it is asserting the absence of.

## M4. Cost

`scripts/rl_es/pilot.py --mode cost`, evidence in `plots/data_analysis/evaluation/rl_manager_es/cost.csv`. The DQN arm's per-step cost is measured in the same session on the same node rather than quoted, because the six-hour figure for the DQN arm does not transfer to a work profile of P small forwards per round.

| arm | population | episodes/step | s/step | vs DQN step | hours for 4000 |
|---|---|---|---|---|---|
| DQN (`run_batch` + replay write + update) | - | 1000 | 5.153 | 1.00 | 5.73 |
| ES | 10 | 1000 | 1.794 | 0.35 | 1.99 |
| ES | 20 | 1000 | 2.292 | 0.45 | 2.55 |
| ES | **40** | 1000 | **3.313** | **0.64** | **3.68** |
| ES | 100 | 1000 | 5.851 | 1.14 | 6.50 |

The DQN row reproduces the six-hour estimate the brief quotes (5.73 h), which is the check that the measurement harness is measuring the right thing.

**The chosen configuration is cheaper than the arm it is being compared with**: 3.68 h against 5.73 h, 0.64x. That is not what I expected before measuring. Two things pay for the 39 extra manager forwards per round: the environment — three artificial-human GNNs and the opponent punisher — runs once over all 1000 episodes regardless of the population size, and this arm does no replay write and no backward pass. The cost grows sub-linearly in the population (10 -> 100 members is 5.7x the manager forwards for 3.3x the wall clock), which is the signature of a rollout whose fixed cost dominates.

The SLURM template keeps the DQN arm's 20-hour wall limit, which is now 5.4x the expected run time.

## M5. Signal against noise — the number that decides whether this arm can learn

`scripts/rl_es/pilot.py --mode noise`, evidence in `plots/data_analysis/evaluation/rl_manager_es/noise.csv`.

A member's fitness is the mean of its own `1000/P` episode returns. The episode return varies a lot on its own here, because the switch predictor moves players between groups and the group the manager is paid on is between 0 and 8 players wide. So the quantity that matters is

```
signal_to_noise = (sd of fitness ACROSS members) / (standard error of one member's own fitness)
```

Below 1, the ranking the update is built from is mostly sampling noise and no budget fixes it.

40 members, 25 episodes each, at generation 0, two independent repeats per sigma.

| sigma | fitness mean | between-member sd | within-member se | **signal/noise** | member punishment sd | member punishment mean |
|---|---|---|---|---|---|---|
| 0.005 | 778.6 / 738.4 | 177.0 / 199.4 | 143.1 / 137.2 | **1.24 / 1.45** | 0.83 / 1.18 | 5.42 / 5.69 |
| 0.01 | 716.4 / 629.2 | 228.4 / 247.9 | 134.8 / 126.6 | **1.69 / 1.96** | 1.66 / 2.05 | 6.09 / 6.71 |
| **0.02** | 637.7 / 513.6 | 315.1 / 278.7 | 132.2 / 112.4 | **2.38 / 2.48** | 2.56 / 2.57 | 6.99 / 8.02 |
| 0.05 | 348.2 / 406.5 | 375.1 / 431.0 | 106.1 / 103.0 | **3.54 / 4.19** | 4.94 / 4.62 | 10.65 / 9.73 |

**sigma = 0.02 is chosen on this table, not on the paper's authority** (it happens to be the paper's default too). Above 1 at every sigma tested, so the arm is not dead on arrival; but at 0.005 the ratio is 1.24–1.45, which is too close to the floor to build 4000 updates on. Going the other way, 0.05 buys a better ratio by *damaging the policies*: the population's mean fitness falls from 638 to 348, a 45% drop against sigma = 0.02, and the mean punishment nearly doubles. That is the smoothed objective drifting away from the objective. 0.02 sits at ratio ~2.4 with a 14% fitness cost against 0.01 and no member collapsing to a degenerate constant policy (`identical_member_policies` false at every sigma, member punishment sd 2.56).

**An unplanned finding worth recording, because it sets expectations.** The Spearman correlation between a member's fitness rank and its mean punishment is **strongly negative at every sigma**, reaching **-0.87 and -0.90 at sigma = 0.02** and -0.95 at 0.05. At the initialisation, the members that punish *less* score *higher*, consistently. That is not surprising — the untrained policy punishes about 5 to 7 points per member per round and the reward is the common pool, which punishment subtracts from directly — but it does mean the first thing this arm will do is drive the punishment level down, and the interesting question is what shape it settles into once the level is no longer the dominant term. A reader should not mistake an early collapse in mean punishment for the method failing.

## M6. Generation-0 policy shape — do members already disagree about the sign?

`scripts/rl_es/pilot.py --mode shape`, evidence in `plots/data_analysis/evaluation/rl_manager_es/shape_generation0_*.csv`.

This answers the coordinator's question directly: whether untrained population members already disagree about the *sign* of the contribution → punishment contingency says how much of the seed spread is decided before any selection happens. Bins are the evaluation suite's own `RPA_EDGES`, imported from `evaluation_suite/metrics.py` rather than restated, so these columns and `plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape.csv` are the same bins.

The reference columns, for reading the table against (from the two-worlds branch, `policy_shape.csv` / `policy_shape_n.csv`):

| bin | human managers | lin_punisher (clone) | rl_s42 | rl_s43 | rl_s44 |
|---|---|---|---|---|---|
| {0} | 4.755 (n=809) | 3.721 (n=1034) | 0.077 (n=1702) | 10.741 (n=555) | 0.231 (n=1197) |
| 1-5 | 2.973 (n=1955) | 2.930 (n=2011) | 0.159 (n=2954) | 0.721 (n=3287) | 0.279 (n=2561) |
| 6-10 | 1.672 (n=2614) | 1.808 (n=2808) | 1.245 (n=2051) | 0.000 (n=3166) | 1.256 (n=2539) |
| 11-15 | 0.978 (n=1794) | 1.300 (n=1409) | 4.745 (n=1064) | 0.000 (n=1633) | 2.000 (n=1715) |
| 16-19 | 0.692 (n=510) | 1.066 (n=393) | 5.000 (n=352) | 0.000 (n=429) | 2.000 (n=659) |
| {20} | 0.267 (n=1232) | 0.327 (n=1529) | 5.000 (n=1165) | 0.000 (n=1482) | 2.000 (n=1733) |

Measured at generation 0 with sigma = 0.02, 40 members, 1000 episodes.

**theta itself is essentially flat.** The untrained evaluated policy punishes a near-constant 5.0 across all six bins (5.15, 5.07, 4.68, 5.00, 5.00, 5.00; slope **-0.019**), on 8,662 to 19,915 agent-rounds per bin. The `valid` subset (timed-out cells dropped, which is how the human frame treats them) differs only in the 6-10 bin: 5.01 rather than 4.68, slope -0.029. So the untrained network has no contingency to speak of, in either direction.

**The population already disagrees about the sign, before any selection has happened.**

| | all cells | valid cells only |
|---|---|---|
| members with the human sign (slope < 0) | **24 / 40** | **27 / 40** |
| members with the inverted sign (slope > 0) | **16 / 40** | **13 / 40** |
| members exactly flat | 0 | 0 |
| slope range across members | **-1.574 to +3.101** | -1.578 to +3.079 |
| population pooled | -0.188 | -0.201 |
| theta | -0.019 | -0.029 |

Put beside the finished runs, using the same bins and the same slope definition (least squares across the six bin means):

| manager | slope | sign |
|---|---|---|
| human managers | -0.857 | human (decreasing) |
| lin_punisher (the clone) | -0.659 | human |
| rl_s42 (DQN) | **+1.218** | inverted |
| rl_s43 (DQN) | -1.596 | human |
| rl_s44 (DQN) | **+0.422** | inverted |
| ES theta, generation 0 | -0.019 | ~flat |
| ES population, generation 0 | -0.188 | weakly human |

**A single generation of untrained perturbations spans -1.57 to +3.10, which brackets the entire range the three finished DQN seeds landed in (-1.60 to +1.22).** Read carefully, this says the raw variation in policy shape is present in the parameter noise itself and does not have to be manufactured by exploration; it does *not* say selection cannot remove it, and it does not say the DQN seeds' shapes came from the same source. What it does establish is that a 40-member population at sigma = 0.02 is sampling shapes on both sides of the human sign every generation, so this arm has something to select between — which is the precondition for the primary question to be answerable at all. The same read at the end of training (`member_shape_summary.csv`, successor step 1) is the measurement of what the selection actually did.

## M7. Tests

All on Raven (`scripts/remote_test.sh --test-only` against `AI_REMOTE_DIR=~/repros/ai-runs/rl-es`), plus the config contract tests locally.

| suite | where | result |
|---|---|---|
| `src/aimanager/tests/test_es_manager.py` | Raven | **14 passed** (154 s) |
| `src/aimanager/tests/test_es_rollout.py` | Raven | **4 passed** (441 s) |
| `scripts/tests/test_rl_es_configs.py` | local | **9 passed** |

What they cover, against the brief's list:

- *the perturbation* — antithetic structure exact, even-population precondition, reproducibility from a seed, and that successive generations differ (a generator reset per generation would silently score the same population 4000 times);
- *the fitness aggregation* — centered ranks are an even grid in [-0.5, 0.5] summing to zero, invariant to the scale of the returns, and the estimator points up a known fitness gradient (cosine > 0.9 on a linear fitness). Plus the negative control: with raw-return weighting, one outlier member captures the update direction almost completely (cosine > 0.99 to that member's own perturbation), which is the measured form of the argument for rank shaping;
- *the deterministic evaluation of the mean parameters* — `test_the_evaluation_scores_theta_not_a_population_member`: with members holding grossly perturbed weights, the evaluation rollout still runs theta and theta is unchanged afterwards; and `test_single_member_rollout_reproduces_run_batch`, the bit-equality with the DQN arm's evaluation;
- *reproducibility from a seed* — `test_a_generation_reproduces_from_its_seed`: same seed gives identical per-episode returns and identical metric rows through the real environment, a different seed does not;
- plus the policy-shape binning against `pd.cut` with the evaluation suite's own `RPA_EDGES`, and the accumulator's member/group/validity handling against hand-computed answers.

The run itself was smoke-tested end to end through the real entry point (`python -m aimanager train-manager-es` on the 40-generation pilot config) before the five seeds went out, because the guards above exercise `PopulationRollout` but not the Adam step, the parquet writing or the model save.

---

# Decisions, and why

Each of these was mine to make and each is defended here rather than inherited.

**Evolution strategies rather than the cross-entropy method.** CEM keeps the top-k members and refits a distribution to them; ES uses every member, weighted by rank. Two reasons ES fits better here. First, with a population of 40 in a 78,532-dimensional parameter space, throwing away 75% of the members throws away most of the little information a generation carries. Second, CEM's elite selection is a *hard* threshold on a fitness that M5 shows is noisy; a member that made the elite set on a lucky draw of group sizes enters the refit with full weight, where centered ranks give it the weight of its neighbour.

**Population size 40, 25 episodes each.** The constraint is that the batch partition must be exact (1000 / P an integer) and P must be even for mirrored sampling. That leaves a one-dimensional trade: more members means more independent directions per generation but a noisier fitness for each. 40 is the point where the population is large enough for the mirrored-pair estimator to be more than a handful of finite differences and each member still gets 25 episodes. The alternative shapes — P=20 with 50 episodes, P=100 with 10 — are priced in `cost.csv` and their signal-to-noise is in `noise.csv`, so this choice is a reading of those two tables rather than a prior.

**Mirrored (antithetic) sampling: yes.** It turns each pair into a two-sided finite difference and cancels theta's own fitness level from the estimator. With P=40 the mean of 40 independent perturbations is not small, so without mirroring the estimator carries that level as a bias term. `test_mirrored_perturbations_are_exactly_antithetic` pins the structure.

**Centered-rank fitness shaping: yes, and skipping it would have needed a defence I do not have.** Rank shaping is usually load-bearing for ES's stability and its absence is a common reason ES underperforms. Here there is a specific reason on top of the general one: the episode return's largest variance component is group size, so a member scored on an unusually large group can post a return far above the rest for reasons that have nothing to do with its parameters. Under raw-return weighting that member sets the update direction almost single-handedly — `test_raw_fitness_weighting_is_captured_by_one_outlier` demonstrates exactly that, with the raw estimator's cosine similarity to the outlier's own perturbation above 0.99. Centered ranks bound every member's influence to 1/P.

**Fitness = the undiscounted sum of the RL group's per-round common pool over the 24 rounds, averaged over the member's episodes.** The undiscounted episode return is the natural ES objective and it is what the evaluation reports; the DQN arm optimises the discounted version (γ=0.98) but is *read* on the same per-round reward. Stated here because it is a real difference between the arms that a reader should not have to infer.

**Noise scale sigma.** Chosen from M5 rather than from the paper alone; the paper's 0.02 is the starting point and the sweep says whether it produces a fitness spread that clears the sampling noise without collapsing members into degenerate constant policies.

**Adam at lr 0.01 with L2 0.005.** The paper's own defaults, unchanged. The ES estimator's scale drifts as the fitness spread changes, so a plain SGD step would change effective size with it; the L2 term keeps ‖theta‖ from growing and quietly shrinking the effective noise scale.

**A dedicated RNG stream for the perturbations** (`seed + 10007`), so the noise a generation sees is a function of the seed and the generation index alone, and stays so if the environment's RNG consumption ever changes.

**The initial parameter vector is the DQN arm's.** `build_world` reproduces `rl_manager.train_manager`'s construction *order*, because the environment's constructor draws the first round's contributions and so consumes the RNG before the manager's networks are built; and the manager is built with the DQN arm's full argument set, target network included, so the policy network's initialisation draws are identical. Guard G1 checks the two vectors are equal. This makes the five-seed comparison paired rather than merely matched.

---

# The honest risk

Evolution strategies ignores within-episode credit assignment. On an equal episode budget it may simply underperform DQN for reasons that have nothing to do with the hypothesis under test. **If that happens it is the result, and it will be reported as the result, not dressed up as evidence against consistency mattering.** The informative comparisons are the policy shape and the behaviour-versus-evaluation gap and the seed spread, in that order; the final return is the fourth.

There is also a specific way this arm can fail that is worth naming in advance: if M5's signal-to-noise is below 1 at every sigma, the arm learns nothing and the run is a measurement of that fact rather than a test of the hypothesis. That is a real possible outcome and the pilot is where it would show.

---

# Expected (not measured)

Nothing here is a result.

- If the decorrelation story is right, the ES seeds recover the human sign — negative slope of mean punishment across the six contribution bins — and agree with each other more than the DQN seeds do.
- The behaviour-versus-evaluation punishment gap should be far smaller than the DQN arm's 1.7–6.6x, by construction, and should shrink as theta moves away from the initialisation (where the argmax is most sensitive to perturbation).
- Seed spread in the final return should be smaller than the DQN arm's, for the same reason.

---

# Successor

1. **Read the shape first.** `python scripts/rl_es/policy_shape.py --seeds 42,43,44,45,46 --reference plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape.csv`. Writes `policy_shape.csv`, `policy_shape_n.csv`, `policy_shape_slope.csv`, `member_shape.csv`, `member_shape_summary.csv`, `shape_trajectory.csv` under `plots/data_analysis/evaluation/rl_manager_es/`. Report the per-member table as well as theta's: whether members disagree about the sign at the END of training, compared with M6's reading at the start, is the cleanest available measure of how much selection actually did.
2. **Then make the shape numbers comparable across arms.** The in-training shape comes from this run's own evaluation rollouts; the two-worlds numbers come from the cross-evaluation *simulation* (`configs/simulation/manager_testing/24_rl_new_clones_cross_eval.yml`). The pairing and the episode count differ. Add the five ES models to that cross-eval config and re-run `measure.py` before putting an ES number and a DQN number in the same sentence.
3. **The gap and the spread.** `<job>_generations.parquet` carries `behaviour_eval_punishment_gap`, `fitness_std`, `within_member_se` and `grad_norm` per generation. The cross-arm comparison the whole exercise exists for is: gap, seed spread, then return.
4. **The experiment this arm cannot do.** Whether the artificial humans respond to punishment *as a function of desert* is separable from exploration and should be tested directly: hold a group's total punishment fixed and vary only which member receives it, then read the contribution response. If the response is flat, the inverted shape has no gradient to correct it and no exploration method will, in which case all four arms of this comparison are answering a question the environment cannot support.
5. If M5's signal-to-noise came out marginal, the lever that does not touch the budget is the population/episodes split: the same 4,000,000 behaviour episodes can be spent as 4000 generations of 40x25, or 2000 of 40x50 (two env rollouts per generation), or 4000 of 20x50. `noise.csv` is the table to decide from.
