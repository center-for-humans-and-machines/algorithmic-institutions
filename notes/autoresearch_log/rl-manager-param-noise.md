# rl-manager-param-noise

## Declaration

**Not a slot experiment.** This branch changes no artificial-human model and is not judged by the §2 gates of `notes/autoresearch.md`.

**One arm of a four-way exploration comparison.** The other three — annealed local epsilon-greedy, bootstrapped DQN, evolution strategies — are being built by sibling agents against the same contract. This arm is **parameter-space noise**. Nothing here touches their branches.

**The shared contract, not restated but pinned.** Base `auto/rl-manager-two-worlds` at `0ff44a9`. Reward `common_pool`. The four artifacts and the opponent byte-identical to `configs/training/rl_manager/rl_new_clones_s42.yml`. Env unchanged. Seeds 42–46. Equal environment episodes, not equal update steps. Evaluation identical in every arm: a fully deterministic rollout, every exploration mechanism off, batch 1000, at the same intervals, in the existing long schema.

## What this arm tests

The manager trains with epsilon-greedy fixed at 0.1 over 31 ordinal punishment levels. Uniform sampling there injects ~1.5 punishment points per member per round, the size of the entire learned signal, and on the finished runs the behaviour policy punished 1.7 to 6.6 times as hard as the policy being evaluated.

Worse for the outcome that actually matters: a uniform draw over 31 levels is **independent of the contribution it is aimed at**. It writes punishment-contribution pairs into the replay buffer whose contingency is noise. Two of the three finished seeds came out with that contingency *inverted* — punishing the full contributor harder than the free-rider — which is a plausible consequence.

Weight-space noise perturbs the *function* rather than the output. A perturbed policy still maps contribution to punishment in some coherent way, and a whole episode is collected under one such mapping. **Shape is the primary outcome; the behaviour-versus-evaluated gap is a description of what was sampled.**

### The off-policy objection, stated and answered

The objection is correct and has to be met head on, not left for a reader to raise.

**DQN is off-policy. A behaviour policy that differs from the one being evaluated is what the algorithm is for.** Q-learning bootstraps toward the max over actions and evaluates the greedy policy whatever collected the data. So the *existence* of a behaviour-versus-evaluated gap is not a defect, and broad, decorrelated action coverage is exploration doing its job. The framing this arm started from — "each run's own policy is poorly evaluated, and that is a candidate explanation for the seed spread" — overstated the case. **The behaviour-versus-evaluated punishment ratio is reported here as a description of what was sampled, never as a fault.**

**What survives is a narrower claim, about states rather than actions.** Off-policy correction buys correctness of the action choice *given a state*. It does not supply states the behaviour policy never visits. Here the state distribution is endogenous to the manager's own behaviour in two ways that are structural, not incidental:

- the contributors are recurrent, so a punishment early in an episode shifts their hidden state for the remaining rounds;
- group composition changes because members switch in response to punishment, so who the manager is even managing at round 12 depends on what it did at rounds 0 to 11.

The sharpest form of the claim is therefore **trajectory coverage**: a manager that is consistently contingent across all 24 rounds produces trajectories that per-action dithering does not generate cleanly, because dithering's 24 independent perturbations rarely line up into a coherent 24-round policy. Whether those trajectories matter for what is learned is **not measured** — by this arm or, so far, by anything else. It is stated as the hypothesis, not as a finding.

**And the cost, which belongs here in the same breath.** Weight noise held to the same mean action displacement explores *less of the action space* than uniform dithering does. Uniform dithering reaches every one of the 31 levels from every state with probability eps/31; weight noise reaches only the levels some perturbed network prefers, which is a far smaller and structured set. Under the off-policy view that is a straightforward loss. This arm trades action coverage for trajectory coherence. **Reduced action coverage is not presented here as an improvement.** If the arm loses, that trade is the first place to look.

**The rival explanation this arm cannot rule out, and which is now the stronger one.** If the artificial humans respond to punishment regardless of whether it was deserved, then no targeting choice changes the return, the contingency is unidentified by the reward, and it is arbitrary for reasons that have nothing to do with exploration. No exploration method fixes that. A probe of it is running separately. This arm cannot distinguish the two, and a negative result here is evidence for that explanation only in the weak sense of failing to be evidence against it.

## The variant, and why

**Plappert et al. (2018), adaptive additive weight noise — not Fortunato et al. (2018), NoisyNet.**

Three reasons, in order of weight:

1. **The contract requires the target network to be unperturbed and a zero scale to reproduce the existing agent exactly.** NoisyNet cannot give either cheaply. Its noise parameters are *part of the network*, learned through the TD gradient, so the target network is itself a noisy net and would have to be forced to its mean by hand; and the architecture changes, so "zero scale" is not a switch but a different model. Plappert's perturbation lives on a private copy of the acting network and touches nothing else, so both guarantees are structural rather than argued.

2. **NoisyNet does not fit this network.** Its unit is a `NoisyLinear` replacing `nn.Linear`. Here the policy is a `MetaLayer` graph net with a `GRU` trunk and a bias MLP; the GRU's weights are not `nn.Linear` modules, so a faithful NoisyNet would either skip the recurrent trunk — the part that carries the within-episode state this arm is about — or need a hand-written noisy GRU.

3. **The adaptive scale is the part that earns its keep**, exactly as the brief says. A fixed weight-noise scale means something different at every point in training; the adaptation holds a target divergence in action space instead.

**One documented deviation from the paper.** Plappert uses layer normalisation so that one global sigma means the same thing in every layer. Adding layer norm here would change the network *at zero noise* and break the reproduction guarantee, so instead each parameter tensor's noise is scaled by that tensor's own RMS (`relative: true`). That addresses the same failure — a sigma that is huge in one layer and negligible in another — without touching the architecture. Switchable.

## The ordinal divergence measure

The judgment call the brief left open, and the coordinator sharpened: an adaptation target defined over an *unweighted* action distribution cannot tell a policy that shifted every punishment by one level from one that inverted its contingency on contribution — and for this experiment those are not remotely the same perturbation.

Decided **measured, not assumed**. All three candidates are computed every round and all three are logged; one of them drives the adaptation.

- `l2` — Plappert's own DQN distance, the RMS difference between the two softmax policies, unweighted over the action set.
- `w1` — 1-Wasserstein between the softmax policies on the integer punishment line. Ordinal, in punishment points.
- `mad` — mean |a − ã| between the two greedy actions, in punishment points.

**Chosen: `mad`.** Two reasons, both measured on the real stack rather than argued (`plots/data_analysis/evaluation/rl_manager_param_noise/probe.json`, section B and the sigma sweep):

1. It is in the units the whole comparison is stated in. Epsilon-greedy's injection is quoted as punishment points per member per round; setting the target to the measured value of exactly that quantity is what makes this arm a one-variable change — the exploration becomes *consistent*, not *smaller*.
2. The unweighted measure is blind to the distinction the coordinator named. Section B of the probe perturbs the real Q tensor two ways — every punishment moved one level up (contingency preserved) and the action axis reversed (contingency inverted) — and reports all three measures on each.

`w1` shares `mad`'s ordinal sensitivity but is a distance between *distributions*, and these policies are near-deterministic, so it mostly reports the argmax gap through a softmax. `mad` reports it directly.

**How the corrected framing sharpened this.** Under the off-policy correction above, the question the target is supposed to answer is *not* "is the perturbed policy's action distribution a given distance from the unperturbed one" — off-policy learning is indifferent to that. It is "does the perturbation produce a coherent alternative *contingency* that gets played out for 24 rounds and valued". A target that assigns a uniform one-level shift and an inverted contingency the same number is therefore not merely imprecise for this experiment; it is measuring a different quantity from the one the arm is about. That is what settled the choice, and the measurement below is what settled the size of the effect.

**What `mad` still cannot do, stated plainly.** It is sensitive to the *size* of a perturbation but not to its *kind*: a large coherent shift and an inversion can both be large. No scalar adaptation target would separate them, so `mad` is a magnitude control, not a shape control. That is why the shape itself is recorded as a first-class metric rather than inferred from the divergence.

## What the run records

`rl_manager.run_batch` now records, per rollout-round, on the evaluation suite's own RPA bins (`RPA_EDGES` / `RPA_LABELS` imported, not re-invented, so the arms are comparable):

- `rpa_mean_{bin}` / `rpa_n_{bin}` — mean punishment and row count per contribution bin, RL manager's own group;
- `rpa_opp_mean_{bin}` / `rpa_opp_n_{bin}` — the same for the artificial punisher on its own group, so the clone column is measured on the same rollouts at no extra cost;
- `param_noise_scale`, `param_noise_divergence`, and the divergence under all three measures, on behaviour rollouts only.

Both are new metric names. No contract metric was renamed into another slot, and none was dropped: `punishment`, `opp_punishment`, `rl_avg_group_size`, `opp_avg_group_size`, `rl_end_group_size`, `opp_end_group_size`, `contribution`, `common_good`, `contributor_payoff`, `group_payoff`, `group_payoff_sum`, `opp_sum_payoff`, `next_reward` are all present and unchanged.

`sampling` distinguishes the two rollout kinds: `greedy` for the deterministic evaluation (unchanged, identical in every arm) and `param-noise` for the behaviour rollouts. With the mechanism off the tag is `eps-greedy`, as before.

## Measured

All of this is `plots/data_analysis/evaluation/rl_manager_param_noise/probe.json`, produced by `scripts/rl_param_noise/probe.py` on the real stack — the same four artifacts, the same env, the same opponent — at batch 256 on one A100, job 30413190, 57 seconds. The policy is at **initialisation**; every number below is about the *geometry of the two exploration mechanisms*, not about which policies are good.

### The episode budget matched

Counted, not asserted. `scripts/rl_param_noise/budget.py` monkeypatches `ArtificialHumanEnv.step` over a real, shortened `train_manager` call and divides by `n_rounds` (`budget.json`). On a 6-step run with eval period 2 it measured **216 step calls / 24 rounds = 9 rollouts**, exactly the 6 behaviour + 3 evaluation the loop should run. Counting `reset` instead gives 10, because the env constructor resets once without playing an episode — an off-by-one rollout per run, which is why steps and not resets are the unit.

Extrapolation to the full config is exact, because the loop runs one behaviour rollout per update step and one evaluation rollout every `eval_period`.

| | reference (`rl_new_clones_s42`) | this arm (`rl_pnoise_s4x`) |
|---|---|---|
| behaviour rollouts | 4,000 | 4,000 |
| evaluation rollouts | 200 | 200 |
| episodes per rollout | 1,000 | 1,000 |
| **behaviour episodes** | **4,000,000** | **4,000,000** |
| **evaluation episodes** | **200,000** | **200,000** |
| **total environment episodes** | **4,200,000** | **4,200,000** |

24 rounds each, so 100,800,000 agent-rounds per run. The arm changes no term in that product: it keeps 4000 update steps, eval period 20 and batch 1000, and each behaviour rollout still runs exactly once per update step. Equal episodes and equal update steps happen to coincide here, which they will not in every arm.

**Compute is not matched, and is not meant to be.** The arm forwards the unperturbed policy as well as the perturbed one every behaviour round, to measure the divergence against, and computes three divergence measures. Measured on the matched guard pair: 5.3 s/update step for epsilon-greedy against 9.1 for parameter noise, so ~5.9 h against ~10.1 h for a full run. Both are inside the 20 h SLURM limit. The contract's budget is episodes.

### The ordinal divergence question

Two perturbations of the real Q tensor, over 24 rounds × 256 episodes of real states:

| perturbation | `l2` (Plappert) | `w1` | `mad` | RPA contrast {0}−{20} |
|---|---|---|---|---|
| unperturbed | 0 | 0 | 0 | +3.792 |
| every punishment +1 level | 0.004450 | 0.135 | 1.000 | +3.792 |
| action axis reversed | 0.004686 | 0.412 | 5.775 | −3.792 |

**Plappert's own unweighted measure gives the two the same number to within 5.3%.** One preserves the contingency on contribution exactly; the other inverts its sign — the failure mode two of the three finished seeds landed in. `mad` separates them by a factor of 5.8, `w1` by 3.1. Decided: `mad`, ordinal, in punishment levels.

### What each mechanism does to the policy's shape

Mean punishment per RPA bin, weighted by the 13,374 agent-rounds in the pool. `disp` is the mean |a − a_greedy| in punishment levels; the two mechanisms are compared at matched `disp`.

| policy | disp | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} | contrast |
|---|---|---|---|---|---|---|---|---|
| greedy (unperturbed) | 0 | 15.44 | 13.68 | 12.97 | 11.96 | 12.06 | 11.64 | **+3.79** |
| epsilon-greedy, eps 0.1 | **0.832** | 15.44 | 13.84 | 13.33 | 12.25 | 12.27 | 11.93 | **+3.51** |
| weight noise, scale 0.02 | 0.505 | 15.71 | 14.21 | 13.93 | 13.41 | 13.36 | 13.01 | +2.70 |
| weight noise, scale 0.05 | **0.975** | 15.33 | 14.20 | 14.55 | 14.71 | 15.03 | 14.67 | **+0.66** |
| weight noise, scale 0.1 | 1.379 | 13.95 | 13.45 | 14.44 | 15.16 | 16.22 | 16.19 | **−2.25** |
| weight noise, scale 0.2 | 2.025 | 10.49 | 10.20 | 12.41 | 14.31 | 16.60 | 16.71 | −6.22 |

Row counts per bin, identical for every row of the table: 1906 / 2259 / 2977 / 1917 / 906 / 3409.

**At matched displacement the two mechanisms do different things.** Epsilon-greedy at 0.832 levels moves the contingency by 7% (+3.79 → +3.51) and nudges every bin toward the mean by about the same 0.3 points. Weight noise at 0.975 levels moves it by 83% (+3.79 → +0.66), and at 1.379 it changes sign. That is the arm's hypothesis made visible: a uniform draw over 31 levels is independent of the contribution it is aimed at, so averaged over rows it can only add a constant to every bin — it explores *level*. Weight noise generates coherent alternative *contingencies*, one per episode, which is what a search over policy shapes needs.

### The epsilon-greedy reference, and why the target tracks it

Measured at initialisation: `mean_abs_action_displacement` 0.832 punishment levels. That is **not** the 1.5 the brief quotes, and the difference is not a discrepancy — it is the whole reason the target had to be made to track. The displacement epsilon-greedy injects depends on where the greedy action sits (15.0 levels at action 0, 7.7 at action 15). At initialisation the argmax sits mid-range, giving 0.83; on the finished runs, whose policies punish near 0, the same formula gives about 1.5.

So the target is the sentinel `eps_matched`: every episode, the exact expected displacement epsilon-greedy *would* produce on that episode's own states, computed in closed form from the unperturbed greedy actions. Not sampled, not frozen.

**Starting scale 0.05**, the smallest on the sweep whose displacement already reaches the target (0.975 against 0.832).

### The guard pilots

Two 200-step training runs, seed 42, identical but for the exploration mechanism: `rl_epsgreedy_guard` (jobs 30413251) and `rl_pnoise_guard` (30413252, then 30414124 after the adaptation fix below). Evidence in `guard_gap.md`, `guard_shape.csv`, `guard_drag.csv`, `guard_noise.csv`.

**Read them as pilots.** 200 steps is 5% of a run. At this point *both* evaluated policies have collapsed to the constant action 5 in every contribution bin — contrast exactly 0.000 — and both punish far harder than any human manager. The inversion the finished runs showed is a late-training phenomenon and neither pilot has it. **Nothing below is evidence that this arm fixes the inversion, and the lower or higher punishment level in a pilot is not an improvement either way.** What the pilots establish is that the mechanism engages and what it does to the buffer.

#### The behaviour-versus-evaluated ratio

| run | behaviour | evaluated | ratio |
|---|---|---|---|
| `rl_epsgreedy_guard` (eps-greedy) | 4.931 | 4.191 | 1.176 |
| `rl_pnoise_guard` (param noise) | 4.974 | 4.194 | 1.186 |

Matched, to 1%, which is what the epsilon-matched target is for. A ratio away from 1 is a description of what the buffer holds, not a defect.

#### What the buffer's shape looks like

Mean punishment per contribution bin, evaluation-suite RPA bins, over 1.1–1.5 million agent-rounds per bin.

| | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} | contrast |
|---|---|---|---|---|---|---|---|
| evaluated policy (both runs) | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 | **0.000** |
| eps-greedy buffer | 6.014 | 6.004 | 6.011 | 5.997 | 6.027 | 6.001 | **0.014** |
| param-noise buffer | 6.555 | 6.089 | 5.752 | 5.532 | 5.546 | 5.716 | **0.839** |
| artificial punisher (clone) | 3.722 | 2.658 | 1.720 | 1.094 | 0.844 | 0.322 | 3.400 |
| human managers | 4.755 | 2.973 | 1.672 | 0.978 | 0.692 | 0.267 | 4.488 |

Row counts, param-noise buffer: 167,001 / 247,804 / 331,710 / 242,879 / 90,534 / 313,865. Clone: 442,067 / 876,416 / 1,209,967 / 890,903 / 279,313 / 947,541. Human: 809 / 1,955 / 2,614 / 1,794 / 510 / 1,232.

**Epsilon-greedy adds a constant; weight noise adds a contingency.** Every eps-greedy bin rises by 1.00 ± 0.02, which is exactly `eps × (15 − 5)`, the uniform drag. The param-noise buffer's contrast is 0.839 from an evaluated contrast of exactly zero.

That is **what was sampled, not what was learned**, and the human sign of that 0.839 is an accident of this particular seed's perturbations, not a result.

#### Targeting, on a statistic force cannot move

`contrast` is a difference of bin means and therefore scales with how hard a manager punishes. Two managers with *identical* contingency, one punishing half as hard, differ by a factor of two on it. The arm is judged on targeting, so targeting needs a statistic that is invariant to force: `rho`, the count-weighted rank correlation between contribution bin and punishment served, which depends only on the *order* of the bin means and so cannot be moved by any monotone rescaling. `scripts/rl_param_noise/targeting.py`; the trap is a test in `scripts/tests/test_param_noise_targeting.py`.

| | `rho` | contrast | contrast / mean | mean punishment | profile SNR |
|---|---|---|---|---|---|
| evaluated policy (both runs) | — (flat) | 0.000 | 0.000 | 5.000 | — |
| eps-greedy buffer | −0.540 | 0.014 | **0.002** | 6.006 | 2.90 |
| param-noise buffer | **−0.827** | 0.839 | **0.143** | 5.848 | 3.02 |
| artificial punisher (clone) | **−1.000** | 3.400 | 2.087 | 1.630 | — |
| human managers | **−1.000** | 4.488 | 2.430 | 1.847 | — |

Negative is the human sign. NaN for the evaluated policies is the honest answer to a perfectly flat profile: there is nothing to rank, which is a different statement from a measured absence of relationship.

**The trap, demonstrated on this table's own reference columns.** On `contrast` the clone (3.400) and the human managers (4.488) differ by 1.088 punishment points, which reads as a targeting difference. On `rho` they are **identical at −1.000**: both are strictly monotone decreasing over all six bins, so their *aim* is the same and only their force differs. The clone punishes less hard overall — mean 1.630 against 1.847. Rescaling the clone's profile to the human mean puts its contrast at 3.853, so **42% of that apparent 1.088-point targeting gap is intensity, not aim**. Had I compared the arms on `contrast` alone I would have read force as targeting on exactly this axis.

**And the opposite error, which `rho` alone would have caused.** The eps-greedy buffer's `rho` is −0.540, which sounds like a real contingency. Its `contrast_over_mean` is 0.002 — the relationship is genuine in rank and utterly negligible in magnitude, which is what a uniform drag plus sampling noise looks like when you rank six nearly equal numbers. `rho` carries no magnitude and must never be read alone; `profile_snr` of 2.90 is barely above the point where it would be ranking noise outright. Both columns, always.

Read together, the param-noise buffer is the only non-reference row with a contingency that is both strong in rank (−0.827) and non-negligible in size (0.143) — against an evaluated policy that has none at all.

#### The uniform-drag discriminator

| run | drag slope | mean abs per-bin shift | buffer contrast | mean shift spread across episodes |
|---|---|---|---|---|
| eps-greedy | 1.0089 | 1.0089 | 0.014 | **0.0515** |
| param noise | 0.8651 | 0.8651 | 0.839 | **1.5848** |

**The drag slope is uninformative on this pilot and must not be read as if it were.** The prediction `eps × (15 − evaluated)` varies across bins only when the evaluated policy varies across bins, and here it is constant at 5, so the predictor is the constant 1.0 and the regression degenerates to "mean shift ÷ 1". The 1.0089 for eps-greedy does confirm the no-free-parameter prediction on its level; the 0.8651 for param noise says only that its mean shift happened to be 0.865, not that its shift is uniform drag. On a run whose evaluated policy has shape, the slope separates the two; on this one it cannot.

**The informative column is the last one.** `mean_shift_spread` is the standard deviation, *across episodes*, of each bin's behaviour mean: 0.0515 for epsilon-greedy against 1.5848 for weight noise, a factor of 31. Every epsilon-greedy rollout is flattened in the same way, because a batch of 1000 episodes averages its own dithering away. Every weight-noise rollout carries its own contingency. That is the designed property, measured.

#### The adaptation, and a defect the pilot found

The first `rl_pnoise_guard` **failed to explore at all**, and this is the most useful thing the pilot produced. `mad` counts argmax flips, so while the perturbation is too small to flip any argmax it reads *exactly* zero, not small — a dead zone in which Plappert's fixed 1% step is climbing a signal that carries no information about how far it has to go. Measured: the scale walked from 0.05 to 0.331 over 200 episodes, `mad` stayed at 0.000 throughout (while `l2` read 0.00003, so the function *was* being perturbed), and the behaviour buffer was bit-identical in shape to the evaluated policy — ratio 1.004, per-bin shift 0.0000, shift spread 0.0000. Under the old cap of 1.0 the search would have saturated and the logs would have looked like a working mechanism.

Two corrections, both from that measurement:

- inside the dead zone the geometric search runs at `adapt_coef ** 6` instead of `adapt_coef`; outside it, exactly Plappert's step. `dead_zone_steps: 0` restores the paper.
- `max_scale` from 1.0 to 100. A cap that binds is indistinguishable in the logs from a mechanism that works.

The re-run then behaves as designed. The scale climbs 0.05 → 1.71 through the dead zone over 60 episodes, escapes at episode 70 (`mad` 1.72 against a target of 1.097), and regulates around 2.8–3.0 for the rest of the run. The target sits at 1.0968 throughout, which is `0.1 × E|u − 5|` for the collapsed constant policy — the epsilon-matched target doing its job.

The per-episode divergence is very noisy (0.002 to 6.47 at an essentially constant scale) because on a near-degenerate policy an argmax flip is a threshold event: a perturbation either flips many cells or none. That is a property of the ordinal measure on this policy, not a controller fault, and it is why the scale is the thing to watch rather than any single episode's divergence.

**A scale near 3 is not a local perturbation.** With per-tensor relative scaling it means noise about three times each tensor's own RMS, so the acting network is mostly noise. If that persists once the policy has real structure, the honest reading is that weight noise cannot match epsilon-greedy's displacement on this task *while staying local*, and the logged `param_noise_scale` is where to see it.

### The invariants

Policy network, target network and opponent all bit-identical before and after a full behaviour episode with the noise active and adapting (`invariants` in probe.json, all `true`; the probe asserts it rather than only reporting it). Structurally, `ParameterNoise.refresh` writes only into a private deep copy, and `test_the_acting_copy_shares_no_storage_with_anything_else` pins the aliasing.

## Inferred

Kept separate on purpose. None of the following is measured here.

1. **That the shape result is an exploration failure at all.** If the artificial humans respond to punishment without regard to whether it was deserved, there is no gradient toward correct targeting, the replay buffer's contingency is irrelevant, and no exploration method will help. This arm cannot distinguish that from the exploration story. If the shape does not move, that is the next experiment, and it is a different one.
2. **That coherent contingency-exploration helps rather than hurts.** The sweep shows weight noise reaching inverted contingencies as readily as human-signed ones. Collecting an episode under an inverted policy is a *better* experiment than collecting one under a policy with no contingency at all, but only if the reward can tell them apart.
3. **That a shared perturbation across the 1000 parallel episodes is enough.** One rollout is one replay-memory episode, so "once per episode" is once per rollout, and all 1000 parallel episodes share the draw. That gives 4000 distinct policy samples over a run and 1000 environment samples of each, rather than 4,000,000 independently dithered trajectories. Whether that trade is right is not measured.
4. **That the untrained-network geometry carries over.** It demonstrably does not, in one respect already measured: at initialisation a scale of 0.05 gave 0.975 levels of displacement, and 200 update steps later the same scale gave 0.000 and the adaptation had to climb to ~2.9 to hold 1.1. The sigma sweep is a statement about the *geometry of the two mechanisms*, which is what it is used for; the numbers on its axis do not transfer, and the adaptation exists precisely so that they need not.

5. **That state-distribution or trajectory coverage differs between the arms.** Not measured, by anything, anywhere. The shape and spread numbers above are about *actions given states*. Whether weight noise visits different states, or different 24-round trajectories, is the hypothesis and not a finding.

## Status: launched, five seeds

| seed | config | SLURM job | output |
|---|---|---|---|
| 42 | `rl_pnoise_s42.yml` | 30414853 | `artifacts/manager/rl_pnoise_s42/metrics/rl_pnoise_s42.parquet` |
| 43 | `rl_pnoise_s43.yml` | 30414854 | `artifacts/manager/rl_pnoise_s43/…` |
| 44 | `rl_pnoise_s44.yml` | 30414855 | `artifacts/manager/rl_pnoise_s44/…` |
| 45 | `rl_pnoise_s45.yml` | 30414856 | `artifacts/manager/rl_pnoise_s45/…` |
| 46 | `rl_pnoise_s46.yml` | 30414857 | `artifacts/manager/rl_pnoise_s46/…` |

Remote dir `~/repros/ai-runs/rl-param-noise`, isolated from every sibling arm. ~10 h each at the measured 9.1 s/update step. Earlier jobs on this branch: probe 30413190, guard pair 30413251 / 30413252, re-guard after the dead-zone fix.

**No results yet, and none are claimed.** The PR is tagged `[LAUNCHED]`.

## Successor

1. **Read the shape first, from the parquet, no simulation needed.** `rpa_mean_{bin}` / `rpa_n_{bin}` at `sampling == "greedy"` is the evaluated policy's contingency at every evaluation point of every seed; `rpa_opp_mean_{bin}` is the artificial punisher on the same rollouts. Human reference: 4.755 / 2.973 / 1.672 / 0.978 / 0.692 / 0.267.

   **Judge targeting on `rho` from `targeting.py`, never on `contrast`.** Both references sit at `rho` −1.000 while their contrasts differ by 1.088 points, 42% of which is force rather than aim; the arms will differ in how hard they punish, so a contrast comparison across arms would read intensity as targeting. Read `contrast_over_mean` beside it for magnitude and `profile_snr` before either — the eps-greedy pilot buffer scores `rho` −0.540 on a relationship of size 0.002. The question is `rho`'s **sign and spread across the five seeds**, not its mean.
2. **Then read `param_noise_scale` over update steps.** If it sits at or near `max_scale` for a long stretch, weight noise could not match epsilon-greedy's displacement and the arm under-explored; that is a finding about the method, not a bug, and it changes how the shape result should be read.
3. **Then the paired comparison.** Same five seeds in all four arms, so pair by seed rather than comparing means of five.
4. **Do not stop at the shape.** If no arm moves it, the rival explanation — artificial humans that respond to punishment regardless of desert — is the live one, and the probe of that is the experiment to read next. This arm cannot distinguish the two and does not claim to.
5. **Two hazards this arm checked and is clear of, recorded so the check is not repeated.**
   - **Named rules are silently ignored at this commit.** `RuleBasedManager.__init__(self, k=1, n_punishments=31, **_)` swallows `rule: never` into `**_` and hands back the default formula wearing the label. A sibling arm published a `never` row punishing 2.57 with a maximum of 20; nothing raised. **This arm's reference columns are the artifact-loaded clone (`load_opponent`, a `.joblib`) and the human CSV, neither of which goes through that dispatcher**, verified by grep over `rl_manager.py`, `linear_opponent.py`, the configs and the analysis scripts — no reference to `RuleBasedManager` anywhere in the path. `guard_report.assert_rule_labels_are_real` asserts realised behaviour against the label anyway, so this stays true if a rule column is ever added.
   - **The leaver diagnostic is not used here and should not be classified on.** Measured across ten managers, only one inverted manager goes positive and the other three sit between −0.02 and −1.21; it tracks shape at r = −0.95 and orders managers correctly, with a measured noise floor of 0.577 at 100 episodes. A ranking, not a classifier, and this arm reports `rho` instead.

6. **Unfinished business in this arm.** (a) One perturbation is shared across the 1000 parallel episodes of a rollout; a chunked rollout would give K perturbations per update step at the same episode cost and is the obvious next variant. (b) The per-episode divergence is extremely noisy on a near-degenerate policy; a controller on a percentile rather than the mean would regulate better. (c) Neither state coverage nor trajectory coverage is measured anywhere, and that is the claim the corrected framing leaves standing. (d) `rho` is computed from per-bin means, so it is the rank correlation on the bin-aggregated profile, not the agent-round joint distribution; recording the (contribution, punishment) joint histogram per rollout would give the exact statistic. Not worth discarding 33 GPU-hours of in-flight runs for, since bin means and counts are sufficient for everything reported here.
