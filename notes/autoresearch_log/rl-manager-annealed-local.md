# rl-manager-annealed-local

**Result, in one line: the arm closed the behaviour-versus-evaluated gap by a factor of four on every seed and did not recover the human policy shape. 1 seed of five targets free-riders, the same as the control and the same seed. See Result, and the Correction within it.**

## Declaration

**Not a slot experiment.** This branch changes no artificial-human model and is not judged by the §2 gates of `notes/autoresearch.md`. It changes the RL manager's *behaviour policy* and nothing else.

**Arm.** One of four arms of an exploration comparison. Mine is **annealed, local epsilon-greedy**. The siblings are bootstrapped DQN, parameter-space noise and evolution strategies. I have not touched their code, configs or remote directories.

**The hypothesis all four arms test.** The task needs *consistent* behaviour, and action-level dithering destroys it.

**Base.** `auto/rl-manager-two-worlds` at `0ff44a9` — common-pool reward, free-punishment lever closed, both guards passing.

**Shared contract, unchanged here.** Reward `common_pool`. The same four artifacts and the same opponent as `configs/training/rl_manager/rl_new_clones_s42.yml`, byte-identical (diffed, note 1). Env: n_groups 2, n_agents 8, agent_groups `[0,0,0,0,1,1,1,1]`, rl_group_id 0, switch_every 4, n_rounds 24, n_contributions 21, n_punishments 31, batch_size 1000. Seeds 42, 43, 44, 45, 46. Evaluation is the existing fully deterministic `greedy=True` rollout at batch 1000 every 20 update steps, written to `artifacts/manager/<job>/metrics/<job>.parquet` in the existing long schema with the existing metric names; no metric was added, renamed or dropped.

**Remote directory.** `~/repros/ai-runs/rl-anneal-local`. I have not written to `~/repros/ai-runs/rl-two-worlds` or `~/repros/ai-runs/rl-percapita`.

## What the arm changes

Two switchable mechanisms, three keys under `manager_args`, both mechanisms defaulting to off in code (`src/aimanager/manager/exploration.py`), so the unmodified behaviour is recoverable from config alone — and is byte-for-byte the old code path, not an approximation of it (note 2).

1. **`eps_final` / `eps_anneal_steps` — anneal epsilon.** Linear decay from 0.1 to 0.01, reaching the floor at step 3000 of 4000 and holding it for the last quarter of training. *Justification, one sentence:* linear-decay-to-a-floor is the DQN standard and is the only common schedule that reaches its floor at a stated step rather than asymptotically, which is what turns "the behaviour policy has converged onto the evaluated one" into a claim a test can check.

2. **`explore_sigma` — explore locally.** The exploratory action is drawn from a discretised Gaussian centred on the current greedy action, sigma = 2 on the 0-to-30 ordinal scale, instead of uniformly over all 31 levels. sigma = 2 probes one to two punishment points around the current policy — the scale of the learned signal — rather than the whole 30-point range.

**Boundary handling, stated explicitly.** Truncation and renormalisation, not clipping. The sampled action is the discretised Gaussian *conditioned* on landing in [0, 30]:

    P(a | a0) = exp(-(a - a0)^2 / 2 sigma^2) / sum over b in [0,30] of exp(-(b - a0)^2 / 2 sigma^2)

Clipping would push the entire out-of-range tail onto the endpoint: at a0 = 0 with sigma = 2 that is **0.600** of the mass on the single action 0. Truncation puts **0.333** there, and that 0.333 is not a pile-up — it is the interior value 0.199 rescaled by the mass the lost half of the kernel used to carry, so every pair of in-range actions keeps exactly the Gaussian ratio it would have had and only the normaliser changes. Both numbers are asserted in `test_boundary_is_truncated_not_clipped`. The alternative I did not take is reflection at the boundary, which also avoids the spike but does not correspond to any conditional distribution and so cannot be stated in one line.

This matters more than bookkeeping here: the endpoint 0 is exactly where a near-zero learned policy sits, so a clipping rule would have manufactured a spike at the action the policy already takes, in the region the primary outcome is measured.

## Primary outcome: policy shape

Revised twice before the runs finished. First after launch-time input from the coordinator, which displaced the level-based reading I started with. Then again after the maintainer's off-policy objection, which changed no number but changed what they may be claimed to show. **The answer is in Result: the shape did not come back.**

Human managers are monotone **decreasing** in the contributor's own contribution: 4.76 at contribution 0 falling to 0.27 at contribution 20. Two of the three finished control seeds came out **inverted** — s42 runs 0.08 up to 5.00, s44 runs 0.23 up to 2.00, monotone in the wrong direction across all six bins; the third has the human sign but fires on 7.4% of rounds, and no learned seed is closer to the human policy than never punishing at all.

Mean punishment per contribution bin is reported for every seed on the evaluation suite's own bins — `{0}`, `1-5`, `6-10`, `11-15`, `16-19`, `{20}` — by importing `RPA_EDGES` and `RPA_LABELS` from `aimanager.evaluation_suite.metrics` rather than re-declaring them, with the row count per bin beside every mean, and with the human and clone columns alongside. `scripts/rl_anneal_local/guard.py shape` does this for a rollout, and Measured 7 shows that reading the clone out of such a rollout reproduces the human curve bin for bin, so the path is sound. The finished runs should still be read through the same cross-evaluation simulation the control used, so that the columns sit in the same table rather than merely on the same bins.

### The off-policy objection, and what survives it

**The objection, which is correct and which this log states before a reader has to raise it.** DQN is off-policy. A behaviour policy that differs from the target is what the algorithm is *for*, not evidence against it: Q-learning bootstraps toward the max over actions, so it evaluates the greedy policy whatever collected the data. **The existence of a behaviour-versus-evaluated gap is therefore not a defect.** My earlier framing treated it as one, and the framing was wrong.

The same objection disposes of the inference I originally drew from Measured 7. Broad, decorrelated action coverage within a contribution band is exploration doing its job — it is the standard justification for exploring at all. "The buffer's action distribution is flatter in contribution than the evaluated policy is" does not license "the learned contingency will therefore be arbitrary". That step does not follow from an action-distribution measurement, and I no longer make it.

**What survives is a different and narrower claim.** Off-policy correction buys correctness of the action choice *given a state*. It does not supply states the behaviour policy never visits. Two features of this environment make the state distribution endogenous to the manager's own behaviour in a way that stationary-MDP intuition misses:

- the contributors are **recurrent**, so extra punishment early in an episode moves their hidden state for the remainder of it;
- group composition is **endogenous**, because members switch groups in response to punishment.

A manager punishing 2.85 therefore produces a different behavioural regime, and a differently composed group, than one punishing 1.5. The buffer holds transitions from a world the evaluated policy does not inhabit — and no amount of off-policy correction manufactures the states it never saw.

The sharpest version is **trajectory** coverage rather than state coverage: what a consistently contingent manager produces over 24 rounds is a trajectory that dithering does not generate cleanly, so the value function never sees the returns of a coherent contingent policy.

**How much of that this branch measured: the aggregate part, weakly; the trajectory part, not at all.** The pilot measured *action* distributions. Measured 8 is a cheap follow-up on the same artifacts that measures a handful of *state summaries*, and it does find a shift — but a few marginal means are not a state distribution, and nothing here tests trajectory coverage. Every claim in this subsection beyond Measured 8's four numbers is argument, not measurement, and it is filed under Inferred accordingly.

### The rival explanation, which is now the stronger one

If the artificial humans respond to punishment **regardless of whether it was deserved**, then no targeting choice changes the return, the learned contingency is arbitrary for reasons that have nothing to do with exploration, and no exploration method fixes the shape. On the maintainer's objection this is the leading alternative, not a footnote: it explains the inversion without needing any claim about buffers at all.

This arm cannot distinguish it from the state-coverage story, and does not try to. A probe of the contribution model's response to deserved versus undeserved punishment is running elsewhere; **its result should be read before this arm's.** If it comes back flat, all four exploration arms are measuring the wrong thing, and my seven runs answer a question that was not the live one.

## Measured

Everything in this section was run; nothing here is extrapolated except where the arithmetic is stated.

**Read the gap numbers (5, 6) and the action-shape numbers (7) as descriptions of what the behaviour policy sampled, not as a defect being diagnosed.** DQN is off-policy and a gap is expected; the reason to measure it is that it is the input to the state-distribution question of Measured 8, not that a gap is itself wrong.

### 1. The configs differ in three keys and nothing else

`diff rl_new_clones_s42.yml rl_anneal_local_s42.yml` outside the header comments is exactly `job_id`, `output_dir` and the three added `manager_args` keys. The control seeds 45 and 46 come from the *existing* generator, `scripts/rl_two_worlds/make_configs.py`, with its `SEEDS` tuple extended; regenerating leaves `rl_new_clones_s42/43/44.yml` byte-identical (git reports them unmodified after the run), which is what makes s45 and s46 the same experiment as the three finished runs rather than a lookalike.

### 2. The off path is the old path, bit for bit

`test_off_path_reproduces_the_original_draws_bit_for_bit` reseeds the RNG and compares `Exploration(eps=0.1)` against the inlined `th.randint` / `th.rand` / `th.where` it replaced. Identical tensors. This is not cosmetic: the five-seed control is only one experiment if seeds 45 and 46 consume randomness the way 42 to 44 did.

### 3. Unit tests

23 cases across 17 functions in `src/aimanager/tests/test_exploration.py`, all passing on Raven. The two the brief asked for by name:

- `test_schedule_reaches_floor_exactly_at_anneal_steps_and_holds` — epsilon is strictly above the floor at step 2999, at the floor at 3000, and still there at 3001, 4000 and 1e9.
- `test_boundary_is_truncated_not_clipped` and `test_boundary_preserves_the_gaussian_ratios` — the sampler's distribution at a0 = 0 and a0 = 30 is the renormalised truncated Gaussian to 1e-9, the endpoint carries 0.333 rather than clipping's 0.600, and every in-range pair keeps its Gaussian ratio.

Two real defects were found by writing these tests rather than by reading the code:

- The kernel was built in float32, in which the far tail of a 31-level row (about `exp(-112)` at sigma = 2) underflows to exactly zero, silently falsifying the ratio claim at the ends of the scale. Fixed by building the 31x31 table in float64, which costs nothing.
- My own first threshold asserted the truncated endpoint mass was below 0.25. It is 0.333. The assertion was wrong, not the code; the number is now pinned against both the analytic value and the clipped alternative.

### 4. Episode budget

**The arm matches the control at 4,200,000 environment episodes**: 4,000,000 behaviour episodes plus 200,000 evaluation episodes, equal to 100,800,000 episode-rounds or 806,400,000 agent-rounds. Evidence: `plots/data_analysis/rl_anneal_local/budget.json`.

*How it was measured.* `guard.py budget` monkeypatches `ArtificialHumanEnv.reset` and `.step`, runs the real `train_manager` on the real config shortened to 6 update steps with `eval_period` 2 and `batch_size` 4, and counts. Result: 216 env rounds, which over `n_rounds` 24 is exactly 9 rollouts, against the 9 the arithmetic predicts (6 behaviour + 3 evaluation). `rollouts_match: true`.

`step` is the counter, not `reset`, and that distinction was a defect the first probe caught: resets came back 10 against 9 expected, because `ArtificialHumanEnv.__init__` resets once without playing an episode. Counting resets would have overstated the budget by one rollout per run. The rollout count is affine in `n_update_steps`, so 4000 steps at `eval_period` 20 and `batch_size` 1000 is 4000 + 200 rollouts times 1000 episodes.

Matching the budget is trivial in this arm and that is worth saying plainly: the behaviour policy changes *what* action is drawn, never how many rollouts are run, so `n_update_steps`, `eval_period` and `batch_size` are byte-identical to the control and the two consume the same episodes by construction rather than by tuning. An arm that changes the number of rollouts per update — evolution strategies, for one — cannot match it that way and has to trade update steps for population size.

The probe had to be moved off the login node to get this number: at load 35 with a hundred users, six update steps took over half an hour and one step alone 380 seconds. `scripts/rl_anneal_local/run_budget.sh` submits it as a batch job, where it takes under a minute. The count is device- and batch-independent, so nothing about the measurement changed.

### 5. The gap on the finished control run, reproduced

Before measuring my own, I reproduced the brief's number from the control's own parquet, read-only out of `~/repros/ai-runs/rl-two-worlds` (nothing was written there). `rl_new_clones_s42`, 4000 steps:

| update_step | behaviour | evaluated | gap | ratio |
|---|---|---|---|---|
| 0 | 4.917 | 4.215 | 0.702 | 1.17 |
| 3980 | 2.788 | 1.620 | 1.168 | **1.72** |

1.72 is the bottom of the 1.7-to-6.6 range the brief quotes, so `guard.py gap` is reading the quantity the comparison is about and not a lookalike. Note the shape of it: the gap *grows* over training, from 0.70 to 1.17 punishment points, because the evaluated policy comes down while the uniform 1.5-point injection does not.

That growth is what the anneal removes. It is worth being precise about why that might matter, because the obvious reading is wrong: a growing gap is not a growing error. Q-learning would still evaluate the greedy policy correctly from this data. What a growing gap does mean is that the world the buffer is drawn from drifts further from the world the evaluated policy would produce, exactly as training is converging — and states are the one thing off-policy correction cannot supply.

### 6. The behaviour-versus-evaluated gap, on a short pilot

Two 300-step pilots, `rl_anneal_local_pilot` and `rl_anneal_control_pilot`, differing only in the three exploration keys. Both are seed 42, so at update step 0 they hold the *same* untrained policy and the evaluated rollout is literally the same number in both — 4.2149, and the same 4.2149 the finished `rl_new_clones_s42` reports at its step 0. The step-0 row is therefore a like-for-like contrast of two behaviour policies over one identical greedy policy, which is as clean as this measurement gets. Full table: `plots/data_analysis/rl_anneal_local/gap.md`.

| run | update_step | behaviour | evaluated | gap | ratio |
|---|---|---|---|---|---|
| control pilot | 0 | 4.917 | 4.215 | **+0.702** | 1.17 |
| control pilot | 280 | 7.208 | 6.577 | **+0.631** | 1.10 |
| arm pilot | 0 | 4.183 | 4.215 | **-0.032** | 0.99 |
| arm pilot | 280 | 5.523 | 5.615 | **-0.092** | 0.98 |

The gap closes by a factor of roughly 7 at both ends, and what is left is negative and of order 0.1 punishment points on a level of 4 to 6 — the sign says it is rollout-to-rollout noise, not injected punishment, since an injection can only push the behaviour policy up.

This says the arm does the thing it was built to do. It does **not** say the control was broken: a 1.17x behaviour-to-evaluated ratio is an ordinary epsilon-greedy DQN doing what epsilon-greedy DQN does. The claim this measurement supports is narrow — that the two arms differ in how far the collected data's *action* distribution sits from the evaluated policy's — and the interesting question is what that does to the *states* collected, which is Measured 8.

**Which of the two mechanisms did it.** At update step 0 epsilon is still 0.1 in both pilots, so the entire closure in the step-0 row is the *local proposal*; the anneal has not moved yet. By step 280 the pilot's schedule has been at its 0.01 floor for 55 steps, and the gap there (-0.09) is indistinguishable from the gap at step 0 (-0.03). On this evidence **local sampling does the work and the anneal is insurance** — which is worth knowing, because it means a sibling arm that fixes only the level and not the locality would not reproduce this.

**What the pilot does not show.** 300 steps is 7.5% of a run. Over the full 4000 steps the control's gap *grows*, 0.70 to 1.17 (Measured 5); over 300 steps it shrinks slightly, 0.70 to 0.63. The pilot is too short to reproduce that growth, so it cannot be read as showing the arm beats the control's late-training gap specifically. What it does show is that the arm's gap is near zero from the first step and stays there, which is the property the arm was built for.

### 7. Policy shape on the pilots, and what the behaviour policy sampled

`guard.py shape`, both pilot managers, batch-1000 deterministic rollout, the evaluation suite's own RPA bins. Evidence: `plots/data_analysis/rl_anneal_local/shape_rl_anneal_{local,control}_pilot.csv`.

**First, a validity check on the measurement path.** My clone column, binned out of my own rollout, lands on 5.24 / 3.24 / 1.88 / 1.14 / 0.89 / 0.37 (control pilot) and 4.79 / 3.13 / 1.76 / 1.10 / 0.85 / 0.36 (arm pilot) against the human managers' 4.76 / 2.97 / 1.67 / 0.98 / 0.69 / 0.27. The clone is this project's stand-in for a human manager, and reading it out of a training-env rollout reproduces the human curve bin for bin. So the path is measuring the thing the established analysis measures, not a lookalike.

**The evaluated policies at 300 steps, against the human and the clone:**

| bin | human | clone | control pilot | arm pilot |
|---|---|---|---|---|
| {0} | 4.76 | 5.24 | 23.82 | 15.23 |
| 1-5 | 2.97 | 3.24 | 13.14 | 9.25 |
| 6-10 | 1.67 | 1.88 | 6.90 | 5.77 |
| 11-15 | 0.98 | 1.14 | 5.27 | 5.08 |
| 16-19 | 0.69 | 0.89 | 5.04 | 5.01 |
| {20} | 0.27 | 0.37 | 5.00 | 5.00 |

Row counts per bin are in the CSVs; the evaluated columns run 4,753 to 17,696 rows. **Both** pilots are monotone decreasing — the human *sign* — and both punish far harder than any human. At 300 steps neither has inverted, so **the pilot says nothing about whether this arm fixes the inversion**; the inversion is a late-training phenomenon and 300 steps cannot reach it. I am not going to read the arm's lower level as an improvement either: it is one seed at 7.5% of a run.

**What the pilot does show is an action distribution, and it is exactly the one the arithmetic predicts.** Compare each pilot's *behaviour* shape to its own *evaluated* shape. If uniform exploration spreads punishment independently of the contribution it is aimed at, the behaviour policy should be dragged toward the uniform mean of 15 — down where the policy punishes above 15, up where it punishes below — by about `eps * (15 - evaluated)`. That is a quantitative prediction with no free parameters, and the control obeys it:

| bin | control evaluated | control behaviour | observed shift | predicted shift |
|---|---|---|---|---|
| {0} | 23.82 | 23.03 | **-0.79** | -0.88 |
| 1-5 | 13.14 | 13.87 | +0.73 | +0.19 |
| 6-10 | 6.90 | 7.76 | +0.86 | +0.81 |
| 11-15 | 5.27 | 6.20 | +0.93 | +0.97 |
| 16-19 | 5.04 | 6.07 | +1.02 | +1.00 |
| {20} | 5.00 | 5.99 | +0.99 | +1.00 |

The sign flips exactly where the prediction says it should, at the one bin where the policy punishes above 15. Regressing observed shift on predicted shift gives a slope of **0.891** for the control -- the behaviour policy is, to within a tenth, the evaluated policy plus the uniform pull. The shape's range across bins shrinks by **9.4%** in the buffer relative to what is evaluated.

The arm's shifts are -0.19, -0.16, -0.04, 0.00, 0.00, 0.00. Mean absolute shift **0.067 against the control's 0.886, thirteen times smaller**; regression slope **0.199 against 0.891**; range flattening **1.9% against 9.4%**.

So the replay buffer the control trains on carries a measurably different *action* distribution from the one the evaluated policy would produce, in the direction uniform exploration predicts and at the magnitude it predicts, and this arm shrinks that by an order of magnitude.

**What this is a claim about, and what it is not.** It is a claim about *what was sampled*, not about *what was learned*. I originally wrote this section up as the decorrelation mechanism "caught in the act", with the implication that a flattened buffer shape is a route to an arbitrary learned contingency. That inference does not follow and I withdraw it: DQN is off-policy, it bootstraps toward the max over actions, and broad action coverage within a contribution band is exploration working as intended rather than a fault. An action-distribution measurement cannot on its own say anything about the shape of the policy that comes out the other end.

The reason to keep the numbers is that they are the input to the question that does survive — whether the *states* the buffer holds differ from the ones the evaluated policy would visit, which off-policy correction cannot fix. Measured 8 takes a first, cheap cut at that. Whether any of it changes where training ends up is what the seven runs are for.

### 8. Does the behaviour rollout visit a different world? A first, cheap cut

Added after the maintainer's off-policy objection. It cost no GPU: the pilots' metrics parquets already log both rollouts at the same update steps, and they carry state alongside the action. `punishment` is what the manager chose; `contribution`, `rl_end_group_size`, `common_good` and `next_reward` are what the world did back.

Behaviour rollout minus evaluated rollout, same update step, percent of the evaluated value:

| | control @0 | control @280 | arm @0 | arm @280 |
|---|---|---|---|---|
| punishment (the action) | +16.7% | +9.6% | −0.8% | −1.6% |
| contribution | **−5.0%** | **−1.1%** | −3.0% | −2.0% |
| rl_end_group_size | **−12.2%** | **−5.0%** | −3.7% | −2.1% |
| common_good | −14.6% | −9.5% | −4.2% | −2.4% |
| next_reward | −22.6% | −12.3% | −6.9% | −5.8% |

**The state distribution is endogenous, and measurably so.** The control's behaviour rollout does not merely punish differently — it ends up with groups 12.2% smaller at step 0 and contributors giving 5.0% less. Those are not actions the manager selected; they are the recurrent contributors and the switching mechanism responding. That is the concrete form of "the buffer holds transitions from a world the evaluated policy does not inhabit", and it is the part of the argument that off-policy correction genuinely cannot repair.

**Three things this does not show, stated plainly.**

- **It is four marginal means, not a state distribution.** A shift in the average group size is consistent with many different distributions over states, and nothing here touches the joint.
- **It does not test trajectory coverage at all**, which is the sharpest version of the claim. I ran the obvious cheap probe — the per-round contribution shift at the last logged step, to see whether it accumulates over an episode as the recurrence story predicts. The control moves from +0.072 (rounds 0–3) to −0.357 (rounds 20–23) and the arm from −0.115 to −0.167, which is the predicted direction for the control. But the round-to-round swings are the same size as the effect (+0.372 at round 16, −0.295 at round 20), it is one seed at one update step, and I am not prepared to call it evidence. It is under-powered and I am recording it as such rather than quietly dropping it.
- **The cross-arm comparison is confounded.** The two arms' evaluated policies sit at different operating points (6.58 versus 5.62 punishment at step 280), so "the arm's state shift is smaller" mixes *how far the behaviour policy is from its own target* with *where that target sits*. The within-arm comparison is sound; the between-arm ratio is not clean.

And one honest complication rather than a tidy story: on the contribution channel at step 280 the arm's shift (−2.0%) is **larger** than the control's (−1.1%). The arm reduces the state shift on three of four summaries and at both ends on group size, but it does not eliminate it and does not dominate on every channel. Its own behaviour rollout still visits a measurably different world than its evaluated policy would.

### 9. Full test suite

`python -m pytest src/` on Raven: 186 passed, 11 failed, 4 errors on the first run. Six of the failures were mine and are fixed — `_RecordingManager` in `test_rl_manager_timeout_view.py` is a stand-in whose `get_action` did not accept the new `update_step` keyword, so every rollout test in that file raised `TypeError`. Adding the keyword to the stub fixed all six. The other five failures and all four errors are `FileNotFoundError` on `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet`, which the isolated remote directory does not carry because `train_cluster.sh` excludes `plots/` from the sync; they are a property of the sandbox, not of this branch.

Re-run of the four files that touch this change — `test_exploration.py`, `test_rl_manager_timeout_view.py`, `test_manager_reward.py`, `test_free_punishment.py` — **49 passed**.

## Result

**The arm does not recover the human policy shape.** One of its five seeds targets free-riders — the same seed that already did so in the control. Two still target contributors outright and two target nothing cleanly. The mechanism it was built to change did change, cleanly and on every seed, and the shape did not follow.

The number that decides it is the contrast between the two end bins, mean punishment at contribution 0 minus mean punishment at contribution 20. Human managers run **+4.49**; the clone **+3.0**; an inverted manager is negative.

| seed | arm | control | paired diff |
|---|---|---|---|
| 42 | **−4.40** | **−4.97** | +0.57 |
| 43 | +11.49 | +12.38 | −0.90 |
| 44 | +10.84 | **−1.91** | +12.74 |
| 45 | **−3.56** | **−5.99** | +2.43 |
| 46 | **−4.85** | **−9.17** | +4.32 |

On the endpoint contrast alone this reads as arm 2 of 5 against control 1 of 5. **That was too generous and is corrected below**: on the three-statistic check seed 44 fails on rank, and the honest count is **arm 1 of 5, control 1 of 5** — seed 43 in both. The arm does not increase the number of seeds that target free-riders.

The directional signal is real but small: the statistic moves toward the human sign on four of the five seeds, median +2.43. Against a distance to the human curve of 9 to 14 points on the inverted seeds, that is a few percent of the way. On seed 42 the arm delivered +0.57 of the +9.46 that would have been needed.

Nor is the shape right where the sign is right. Only one run of ten — arm seed 43 — is monotone decreasing across all six bins (rank correlation with the human curve +0.94), and it punishes **11.49** at contribution 0 against the human **4.76**. Arm seed 44 has the right end-to-end contrast but is not monotone (12.84, 1.86, 1.01, 2.00, 2.00, 2.00; rank correlation −0.06): it punishes free-riders hardest and then everyone else flat.

### Correction: the endpoint contrast was not enough, and seed 44 does not survive

A sibling arm withdrew one of its own headline seeds after finding that a difference of endpoint bin means confuses **force** with **aim**, and asked me to run the same check on mine. It was right to, and **one of my two correctly-signed seeds does not survive it.**

Three statistics together, because each alone fails in a known way. A rank correlation scores a profile that is flat to within noise as perfectly targeted, since six nearly equal numbers still have an ordering. An endpoint contrast scores a single spike at bin {0} with everything else flat as perfectly targeted, since it never looks at the middle. A magnitude with no noise gate scores a rounding error as a policy. So: **rank** (Spearman of the six bin means against the bin index; human −1.000), **relative range** (max minus min over the mean), and **gate** (range over its standard error, resampled over episodes, threshold 1.65).

My implementation reproduces the sibling's reference exactly on the human managers — rank −1.000, range 4.488, relative 2.375 — so the two arms are scored the same way.

| profile | rank | tau-b | mono | range | rel. range | gate | verdict |
|---|---|---|---|---|---|---|---|
| human | -1.000 | -1.000 | dec | 4.49 | 2.38 | pending | targets free-riders |
| clone | -1.000 | -1.000 | dec | 3.00 | 1.85 | pending | targets free-riders |
| arm 42 | +0.928 | +0.828 | — | 4.97 | 1.69 | pending | targets contributors |
| arm 43 | -0.941 | -0.894 | dec | 11.49 | 4.38 | pending | targets free-riders |
| arm 44 | +0.058 | +0.138 | — | 11.83 | 3.27 | pending | no targeting |
| arm 45 | +0.928 | +0.828 | — | 5.90 | 1.51 | pending | targets contributors |
| arm 46 | +0.600 | +0.467 | — | 7.96 | 2.68 | pending | no targeting |
| control 42 | +1.000 | +1.000 | inc | 4.97 | 1.90 | pending | targets contributors |
| control 43 | -0.845 | -0.775 | dec | 12.38 | 5.70 | pending | targets free-riders |
| control 44 | +0.941 | +0.894 | inc | 1.91 | 1.52 | pending | targets contributors |
| control 45 | +0.943 | +0.867 | — | 5.99 | 2.06 | pending | targets contributors |
| control 46 | +0.943 | +0.867 | — | 9.18 | 3.46 | pending | targets contributors |

Thresholds: |rank| >= 0.8, relative range >= 1.0, gate >= 1.65. Kendall tau-b and the monotonicity flag are carried because Spearman is dragged toward zero by ties and two profiles saturate at exactly 0 in several bins: control seed 43 is monotone decreasing across all six bins and still scores only −0.845 on three tied bins, which is why the rank threshold is 0.8 rather than the 0.9 I first tried. **No verdict in the table turns on that choice.** The gate column is still pending on a congested cluster; rank and relative range are functions of the six committed bin means alone, and a noise gate can only ever turn a verdict into "no targeting", never rescue one, so nothing below depends on it.

**What changes.** My headline said "correctly signed: arm 2 of 5, control 1 of 5". On this check it is **arm 1 of 5 and control 1 of 5** — seed 43 in both. Seed 44 was the entirety of the arm's apparent advantage and it fails on rank at **+0.058**: its profile is 12.84 at contribution 0 and then 1.86, 1.01, 2.00, 2.00, 2.00, a spike with no gradation above it. That is precisely the shape that passes a contrast test and fails a rank test. **I withdraw the claim.**

Seed 46 also fails, at rank +0.600, and I had it wrong in the other direction: its profile is U-shaped (3.11, 0.06, 0.00, 0.52, 6.16, 7.96), punishing both extremes. I counted it as cleanly inverted. It is not cleanly anything.

**What survives is a smaller claim than the one I made.** The arm does not increase the number of seeds that target free-riders: 1 in five, the same as the control, and the same seed. What it does is **weaken the inversion without replacing it** — 2 arm seeds target contributors against 4 control seeds, and the 2 that moved (44 and 46) went from graded inversion to no clean targeting rather than to correct targeting. That is a real effect on the wrong-direction seeds and it stops short of the thing the arm was built to produce.

**One honest tension, which I am not going to resolve by picking a favourite.** The leaver/stayer diagnostic disagrees about seed 44: it reads −2.05, correctly signed and close to the human's −1.92, where the profile check says "no targeting". Both are right about different things. Seed 44 punishes the true free-riders at contribution 0 very hard, which does drive them out and is real aim at that one bin; it makes no distinction at all among contributions 1 to 20, which is what the rank statistic refuses to call a policy shape. The human contingency is graded across all six bins and seed 44's is not. The 9-of-10 agreement I reported between the two diagnostics becomes 8 of 10 under the stricter criterion, and the disagreements are informative rather than noise.

### Policy shape, every seed

Mean punishment per contribution bin, evaluated policy, batch-1000 deterministic rollout, on the evaluation suite's own RPA bins. Full table with row counts: `plots/data_analysis/rl_anneal_local/policy_shape.csv`.

| bin | human | clone | arm 42 | arm 43 | arm 44 | arm 45 | arm 46 | ctl 42 | ctl 43 | ctl 44 | ctl 45 | ctl 46 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| {0} | 4.76 | 3.91 | 0.60 | 11.49 | 12.84 | 2.44 | 3.11 | 0.03 | 12.38 | 0.09 | 0.01 | 0.11 |
| 1-5 | 2.97 | 2.64 | 0.03 | 3.98 | 1.86 | 0.10 | 0.06 | 0.05 | 0.66 | 0.13 | 0.01 | 0.10 |
| 6-10 | 1.67 | 1.70 | 2.07 | 0.27 | 1.01 | 2.92 | 0.00 | 0.99 | 0.00 | 1.31 | 1.15 | 1.56 |
| 11-15 | 0.98 | 1.05 | 4.93 | 0.00 | 2.00 | 5.97 | 0.52 | 4.61 | 0.00 | 2.00 | 4.33 | 2.00 |
| 16-19 | 0.69 | 0.83 | 5.00 | 0.00 | 2.00 | 6.00 | 6.16 | 5.00 | 0.00 | 2.00 | 5.97 | 2.88 |
| {20} | 0.27 | 0.35 | 5.00 | 0.00 | 2.00 | 6.00 | 7.96 | 5.00 | 0.00 | 2.00 | 6.00 | 9.28 |

Row counts run 3,172 to 36,392 per cell for the learned managers and 510 to 2,614 for the humans, so none of this is thin. Two notes on the reference columns. The human column is one fixed curve, read through `convert.load_human`. The clone column is **not** a constant: it shares the world with the manager under test, so its {0} bin moves between 3.16 and 5.24 across the ten rollouts as group composition changes; the column above is its mean and the per-rollout values are in each `shape_*.csv`.

### The leaver/stayer diagnostic agrees, seed for seed

The sibling arm's diagnostic, run per seed: at the rounds where membership actually changes, what did the leavers contribute against what the stayers contributed. Negative means the free-riders leave, which is what a correctly targeted manager produces. Measured on the same rollout as the shape table.

Reference on this rollout: **human managers −1.92**, **clone −3.11**.

| seed | arm | control |
|---|---|---|
| 42 | **+0.41** | **+0.40** |
| 43 | −2.43 | −2.12 |
| 44 | −2.05 | −0.74 |
| 45 | **+0.41** | **+0.70** |
| 46 | **+0.45** | **+0.73** |

Six of the ten runs are **positive**: they do not merely fail to discipline free-riders, they select for them, driving out the contributors they punish. That is three arm seeds and three control seeds — the same seeds the shape statistic calls inverted.

On the endpoint contrast the two diagnostics agree on 9 of 10 runs, with a correlation of −0.976. Under the stricter three-statistic check in the Correction that becomes 8 of 10, the extra disagreement being arm seed 44. They are independent measurements — one reads punishment against contribution, the other reads who left — and where they part, they are each right about something different; see the Correction.

This diagnostic is worth keeping. It is one number per run, needs no counterfactual, and it would have flagged the inversion from the managers' own output.

### The mechanism did work, at full length, on every seed

The behaviour-versus-evaluated gap at the final evaluation, which was the number the pilot could not supply:

| | arm | control |
|---|---|---|
| mean gap | **0.273** | **1.192** |
| range | 0.09 – 0.63 | 1.08 – 1.25 |
| mean ratio | 1.13× | 1.93× |
| range | 1.06 – 1.23× | 1.60 – 2.32× |

Paired, the arm's gap is smaller on all five seeds, by 0.61 to 1.15 points. The control's ratio sits at the low end of the 1.7-to-6.6 range the brief quoted. So the intervention did exactly what it was designed to do — and the shape did not follow. That conjunction is the result.

### Outcome and spread

Final-window means of the evaluated policy, last ten evaluation points (`final_window.csv`):

| | punishment | group size | contribution | common good |
|---|---|---|---|---|
| arm mean | 2.195 | 3.506 | 8.056 | 10.626 |
| arm spread (max−min) | 1.563 | 1.266 | 1.626 | 3.867 |
| control mean | 1.619 | 3.605 | 7.773 | 10.750 |
| control spread (max−min) | 0.914 | 0.980 | 1.201 | 2.436 |

Paired, arm minus control: common good **−0.12** on average, with only 2 of 5 seeds favouring the arm. **The arm buys no outcome improvement.** It punishes more (+0.58, 4 of 5 seeds) and contributes slightly more (+0.28), and none of it reaches the common good.

**The seed spread got wider, not narrower**, on every metric — punishment 1.56 against 0.91, common good 3.87 against 2.44. The original motivation for this comparison was a seed spread that extra training did not close. Closing the behaviour-versus-evaluated gap did not close it either; it widened it, because the arm added a second cluster of outcomes (the two correctly-targeted seeds) rather than pulling all five together.

### One thing the data says about the rival explanation

Across the ten runs, those whose targeting is correctly signed average **12.11** common good per member against **9.74** for the rest — a 2.4-point difference on a base of about 10. Targeting is not free in this world; correct targeting pays.

That is a cross-run correlation over ten runs with confounds, not a controlled comparison — the correctly-targeted runs also punish somewhat less. The controlled version is the sibling's level-matched inverted-rule simulation, which puts the cost at 32.75 pool points. Both point the same way, and together they **weaken the strongest form of the rival explanation**: it is not that the clones are indifferent to whether punishment is deserved and no return distinguishes the policies. A return difference exists and is large.

What that leaves is a learning failure rather than an environment-indifference failure: the gradient toward correct targeting is there, and four of ten runs found it, and consistency of the behaviour policy is not what separates the ones that did from the ones that did not. **Read the deservedness probe anyway** — it measures the response directly, where this is inference from outcomes.

## Launched

Seven runs, `~/repros/ai-runs/rl-anneal-local`, 22 September 2026. Each is 4000 update steps, ~6.2 hours on one A100 at the 5.5 s/step the pilots measured. The uuid is the run-directory name under `.log/training/manager/rl_manager/<job>/`; the SLURM id is what `squeue` shows. Both are recorded because the account is shared with the sibling arms, so `squeue -u levinb` alone does not identify whose job is whose.

| job_id | seed | arm | uuid | SLURM |
|---|---|---|---|---|
| rl_anneal_local_s42 | 42 | annealed, local | 80dd8416 | 30413209 |
| rl_anneal_local_s43 | 43 | annealed, local | 68c2430a | 30413210 |
| rl_anneal_local_s44 | 44 | annealed, local | aa2ee5f9 | 30413211 |
| rl_anneal_local_s45 | 45 | annealed, local | b242feb4 | 30413212 |
| rl_anneal_local_s46 | 46 | annealed, local | 51a2e447 | 30413223 |
| rl_new_clones_s45 | 45 | control (unmodified) | 85f229b0 | 30413225 |
| rl_new_clones_s46 | 46 | control (unmodified) | f06346f0 | 30413226 |

The two control runs complete the five-seed control; 42, 43 and 44 are already finished in the two-worlds arm's directory and were not re-run. They were launched from *this* branch, whose off path is bit-for-bit the original (Measured 2), and they write into my directory, not the finished arm's.

Guard jobs, same directory: budget 30413186, shape 30413233 (30413194 was the first attempt and died on a device mismatch, note 4). Guard pilots: 30412504 (arm, uuid ce266636) and 30412507 (control, uuid 1509271a).

## Inferred

Clearly separated from the above: none of this was run. Two items that sat here before the runs finished have since been measured and moved to Result — whether the arm recovers the shape (it does not) and what the full-length gap is (0.27 against the control's 1.19). What is left is still argument.

- **The state-coverage argument, which is the one that survives the off-policy objection.** Off-policy correction buys the action choice given a state; it does not supply states the behaviour policy never visited. With recurrent contributors and endogenous group membership, a manager punishing harder produces a different behavioural regime and a differently composed group, so the buffer holds transitions from a world the evaluated policy does not inhabit. Measured 8 shows four state summaries shifting, which is consistent with this and is the only part of it that was measured. **That the shift matters for what gets learned is inference, not measurement.**
- **Trajectory coverage, which is the sharpest version and is wholly unmeasured.** What a consistently contingent manager produces over 24 rounds is a trajectory dithering does not generate cleanly, so the value function never sees the returns of a coherent contingent policy. My pilot measured action distributions; the one cheap trajectory probe I ran (Measured 8) is under-powered and I do not count it. **Nothing on this branch tests this claim.**
- **What I withdrew.** My first write-up argued that a flattened buffer *action* shape is a route to an arbitrary learned contingency. It does not follow — DQN bootstraps toward the max over actions, and broad action coverage within a contribution band is exploration working, not failing. The numbers in Measured 7 stand; that inference from them does not.

- **The perturbation this arm injects, from the sampler alone.** With the greedy action at 0, the control injects a mean of **1.5** punishment points per member per round (eps 0.1 times the uniform mean 15). This arm injects **0.130** at step 0 and **0.013** at step 3000 — 11.5x and 115x smaller. These are exact properties of the two distributions, asserted in `test_annealed_local_injects_far_less_punishment_than_uniform`; they are listed as inferred rather than measured because the realised gap in training also depends on how the greedy policy itself moves, which only the pilot and the runs can show.
- **Why local exploration might fix shape and not merely level.** Stated above under the primary outcome. It is a mechanism, not a measurement, and the pilot's shape table is the first evidence either way.
- **The cost this arm pays, stated up front.** A local proposal buys consistency by giving up coverage. With sigma = 2 truncated at 0, a policy sitting at punishment 0 probes roughly 0 to 6 and effectively never sees 20 or 30. The human means per bin run 0.27 to 4.76, so the region that matters is inside that reach, and the greedy action can drift upward over training with the proposal following it. But if the optimum were far from where the policy initialises, this arm would fail to find it where uniform exploration would have stumbled onto it. That is the trade the arm *is*, not a flaw in it, and it is precisely why the comparison has four arms rather than one.
- **This is the null result I said I would report.** Before the runs I wrote that if the gap closed and the shape stayed inverted, the exploration explanation for the inversion would be dead. The gap closed fourfold on every seed and three of five seeds stayed inverted, so I am reporting it: **action-level dithering is not what makes this manager punish the wrong people.** The one caveat I will allow is that the end-bin contrast did move toward the human sign on four of five seeds, so the effect is not exactly zero — it is a few percent of the distance that would have to be covered, and it changed the sign on one seed.

## Successor

The runs are finished and analysed; this is what I would do next, in order.

1. **Do not run a second exploration arm on the strength of this one.** The gap closed fourfold on every seed and the shape did not follow. Whatever makes three seeds in five punish the wrong people, it is not the consistency of the behaviour policy at the action level. The sibling arms are worth reading for whether any of them moved the shape, but the prior on "more exploration engineering" should now be low.
2. **Read the deservedness probe.** My Result section gives indirect evidence against its strongest form — correctly-targeted runs earn 12.11 common good per member against 9.74 for the rest, so a return difference exists — but that is a cross-run correlation with confounds and the probe measures the response directly.
3. **Keep the leaver/stayer diagnostic and run it on everything, alongside the profile check rather than instead of it.** One number per run, no counterfactual, agreeing with the endpoint contrast on 9 of 10 runs at r = −0.976 and with the stricter profile check on 8 of 10; it is wired into `guard.py shape --targeting-out` and `scripts/rl_anneal_local/profile_check.py`. The two disagree exactly where a manager aims at one bin and nowhere else, which is worth knowing rather than averaging away. It would have flagged the inversion from the managers' own output without anyone having to bin punishment against contribution.
4. **The interesting question is seed 43, not the arm.** Seed 43 is the only run of ten that targets free-riders on the three-statistic check, and it does so in *both* arms. That points at initialisation rather than exploration, and it is where the next experiment is. Do not chase seed 44: on the endpoint contrast it looked like the arm's one success and on the rank test it is a spike with no gradation.
5. **The state- and trajectory-coverage claims remain unmeasured** beyond Measured 8's four marginal means. The upgrades are still worth doing if anyone wants to make the coverage argument properly: compare per-round distributions rather than means (free, on the parquets committed here); dump joint occupancy from `guard.py shape` (about a GPU-minute per manager); score whole episodes for whether a coherent contingent policy's returns are represented at all (not cheap, and the one that matters).
6. **sigma and the floor were chosen, not tuned**, and given the null result there is no reason to sweep them.

## Notes

1. **The contract was diffed, not assumed.** See Measured 1.
2. **`sampling` tags were left exactly as they were** — `eps-greedy` for the behaviour rollout, `greedy` for the evaluation rollout. Renaming them would have made my parquets need their own reader, and the whole point of the four arms is that one reader works on all of them. The behaviour rollout is *run* every update step but *logged* every `eval_period`, which is pre-existing behaviour and is unchanged; both rollouts therefore appear at the same update steps, which is what makes the per-step gap well defined.
3. **No metric was added.** Recording epsilon per step was tempting and was dropped: it is a deterministic function of `update_step` and the config, and adding a row would have made the schema differ from the siblings' for no information.
4. **`ArtificalManager.load` does not move the model to the device it is handed.** `save` puts the policy model on the CPU first and `load` assigns the unpickled object straight through, so a load onto cuda returns CPU weights with a cuda `self.device` and the first forward pass dies on a device mismatch. The only other caller, `api_manager.RLManager`, loads on the CPU and never meets it. My guard works around it with one `.to(device)` rather than changing the manager: that is a fix for its own branch, and an exploration arm is not the place to smuggle it in.
5. **`src/aimanager/rl_manager.py` is not black-clean on the base commit** and I left it that way. Running black on the file reformats four hunks I did not touch; including them would have put unrelated churn in an arm's PR. My own hunk is black-clean and flake8 passes at 88 across everything. If the pre-commit hook reformats the file on the maintainer's next commit, that is the pre-existing drift surfacing, not this branch.
6. **The interpretation in this log was corrected after the runs were launched, and the correction was the maintainer's.** My original framing treated the behaviour-versus-evaluated gap as a defect and read Measured 7 as showing that a flattened buffer action-shape produces an arbitrary learned contingency. Both were wrong, for the same reason: DQN is off-policy, a gap is what the algorithm is for, and broad action coverage within a contribution band is exploration working. **No number changed, no run was restarted, and the arm is exactly what it was.** What changed is the claim: the gap is now presented as the input to a state-distribution question (Measured 8), and the action-shape result as a statement about what was sampled rather than about what will be learned. I am recording the error rather than quietly editing over it, because a reader who knows DQN would have spotted the original framing immediately and should be able to see that it was caught.