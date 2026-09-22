# rl-manager-annealed-local

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

Revised twice. First after launch-time input from the coordinator, which displaced the level-based reading I started with. Then again after the maintainer's objection below, which does not change the arm, the runs or any number, but does change what they may be claimed to show. The second revision was written after the runs were launched; nothing was re-run.

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

Clearly separated from the above: none of this was run.

- **The state-coverage argument, which is the one that survives the off-policy objection.** Off-policy correction buys the action choice given a state; it does not supply states the behaviour policy never visited. With recurrent contributors and endogenous group membership, a manager punishing harder produces a different behavioural regime and a differently composed group, so the buffer holds transitions from a world the evaluated policy does not inhabit. Measured 8 shows four state summaries shifting, which is consistent with this and is the only part of it that was measured. **That the shift matters for what gets learned is inference, not measurement.**
- **Trajectory coverage, which is the sharpest version and is wholly unmeasured.** What a consistently contingent manager produces over 24 rounds is a trajectory dithering does not generate cleanly, so the value function never sees the returns of a coherent contingent policy. My pilot measured action distributions; the one cheap trajectory probe I ran (Measured 8) is under-powered and I do not count it. **Nothing on this branch tests this claim.**
- **What I withdrew.** My first write-up argued that a flattened buffer *action* shape is a route to an arbitrary learned contingency. It does not follow — DQN bootstraps toward the max over actions, and broad action coverage within a contribution band is exploration working, not failing. The numbers in Measured 7 stand; that inference from them does not.

- **The perturbation this arm injects, from the sampler alone.** With the greedy action at 0, the control injects a mean of **1.5** punishment points per member per round (eps 0.1 times the uniform mean 15). This arm injects **0.130** at step 0 and **0.013** at step 3000 — 11.5x and 115x smaller. These are exact properties of the two distributions, asserted in `test_annealed_local_injects_far_less_punishment_than_uniform`; they are listed as inferred rather than measured because the realised gap in training also depends on how the greedy policy itself moves, which only the pilot and the runs can show.
- **Why local exploration might fix shape and not merely level.** Stated above under the primary outcome. It is a mechanism, not a measurement, and the pilot's shape table is the first evidence either way.
- **The cost this arm pays, stated up front.** A local proposal buys consistency by giving up coverage. With sigma = 2 truncated at 0, a policy sitting at punishment 0 probes roughly 0 to 6 and effectively never sees 20 or 30. The human means per bin run 0.27 to 4.76, so the region that matters is inside that reach, and the greedy action can drift upward over training with the proposal following it. But if the optimum were far from where the policy initialises, this arm would fail to find it where uniform exploration would have stumbled onto it. That is the trade the arm *is*, not a flaw in it, and it is precisely why the comparison has four arms rather than one.
- **What a null result would mean.** The gap is already closed (Measured 6), so if the shape still comes back inverted across five seeds, the exploration explanation for the inversion is dead and the clone-response explanation is the live one. That is a useful result and I will report it as one. Given the maintainer's objection, it is also now the outcome I would bet on.

## Successor

For whoever picks this up:

1. **The shape tables for the five seeds are not in this branch.** The runs were launched, not awaited. Read them with `scripts/rl_anneal_local/guard.py shape` against each saved manager, and through the same cross-evaluation simulation the control used, so the columns sit beside `plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape.csv`. The pilot tables (Measured 7) are not a preview of the answer: at 300 steps neither arm has inverted, because the inversion happens late.
2. **The paired comparison is seed-for-seed.** Five arms times five seeds; compare `rl_anneal_local_s{N}` to `rl_new_clones_s{N}` at the same N, never arm mean to arm mean.
3. **Read the deservedness probe first — it may moot all four arms.** If the artificial humans respond to punishment regardless of whether it was deserved, no targeting choice changes the return, the learned contingency is arbitrary for reasons unrelated to exploration, and no exploration method fixes the shape. On the maintainer's off-policy objection this is the *leading* explanation for the inversion, not a footnote: it needs no claim about buffers at all. A probe of it is running elsewhere. If it comes back flat, every arm of this comparison is measuring the wrong thing and my seven runs answer a dead question. Nothing this branch produced can distinguish the two explanations.
4. **Measuring the state-coverage claim properly, which Measured 8 only gestures at.** Measured 8 compares four marginal state summaries between the behaviour and evaluated rollouts and finds them shifted. Three upgrades, in increasing cost:
    - *Free, on the existing pilot parquets.* Compare the **distributions** rather than the means — per-round histograms of contribution and group size, behaviour against evaluated, with a distance rather than a difference of averages. The parquets hold every round of 1000 episodes at 15 update steps.
    - *Cheap, one GPU-minute per manager.* `guard.py shape` already collects per-cell data; extend it to dump the joint (contribution, group size, round) occupancy of each rollout and report the share of the evaluated policy's occupancy mass that the behaviour rollout covers. That is the state-coverage claim stated as a number.
    - *The one that actually matters, and it is not cheap.* Trajectory coverage. Score whole 24-round episodes, not cells: sample episodes from both rollouts, and ask whether the return distribution of a consistently contingent policy is represented in the buffer at all. This is the claim the argument rests on and the one nothing on this branch touches.
5. **The between-arm state-shift comparison in Measured 8 is confounded** and should not be quoted as-is. The two arms' evaluated policies sit at different punishment levels, so the comparison mixes how far each behaviour policy is from its own target with where that target sits. Compare within an arm, or match the operating point first.
6. **sigma and the floor were chosen, not tuned.** sigma = 2 and eps_final = 0.01 were picked from the scale of the signal, with no sweep. If the arm half-works, they are the obvious next knobs — but one variable at a time, and not before item 3 is settled.

## Notes

1. **The contract was diffed, not assumed.** See Measured 1.
2. **`sampling` tags were left exactly as they were** — `eps-greedy` for the behaviour rollout, `greedy` for the evaluation rollout. Renaming them would have made my parquets need their own reader, and the whole point of the four arms is that one reader works on all of them. The behaviour rollout is *run* every update step but *logged* every `eval_period`, which is pre-existing behaviour and is unchanged; both rollouts therefore appear at the same update steps, which is what makes the per-step gap well defined.
3. **No metric was added.** Recording epsilon per step was tempting and was dropped: it is a deterministic function of `update_step` and the config, and adding a row would have made the schema differ from the siblings' for no information.
4. **`ArtificalManager.load` does not move the model to the device it is handed.** `save` puts the policy model on the CPU first and `load` assigns the unpickled object straight through, so a load onto cuda returns CPU weights with a cuda `self.device` and the first forward pass dies on a device mismatch. The only other caller, `api_manager.RLManager`, loads on the CPU and never meets it. My guard works around it with one `.to(device)` rather than changing the manager: that is a fix for its own branch, and an exploration arm is not the place to smuggle it in.
5. **`src/aimanager/rl_manager.py` is not black-clean on the base commit** and I left it that way. Running black on the file reformats four hunks I did not touch; including them would have put unrelated churn in an arm's PR. My own hunk is black-clean and flake8 passes at 88 across everything. If the pre-commit hook reformats the file on the maintainer's next commit, that is the pre-existing drift surfacing, not this branch.
6. **The interpretation in this log was corrected after the runs were launched, and the correction was the maintainer's.** My original framing treated the behaviour-versus-evaluated gap as a defect and read Measured 7 as showing that a flattened buffer action-shape produces an arbitrary learned contingency. Both were wrong, for the same reason: DQN is off-policy, a gap is what the algorithm is for, and broad action coverage within a contribution band is exploration working. **No number changed, no run was restarted, and the arm is exactly what it was.** What changed is the claim: the gap is now presented as the input to a state-distribution question (Measured 8), and the action-shape result as a statement about what was sampled rather than about what will be learned. I am recording the error rather than quietly editing over it, because a reader who knows DQN would have spotted the original framing immediately and should be able to see that it was caught.