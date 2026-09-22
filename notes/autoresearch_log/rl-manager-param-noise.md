# rl-manager-param-noise

## Declaration

**Not a slot experiment.** This branch changes no artificial-human model and is not judged by the §2 gates of `notes/autoresearch.md`.

**One arm of a four-way exploration comparison.** The other three — annealed local epsilon-greedy, bootstrapped DQN, evolution strategies — are being built by sibling agents against the same contract. This arm is **parameter-space noise**. Nothing here touches their branches.

**The shared contract, not restated but pinned.** Base `auto/rl-manager-two-worlds` at `0ff44a9`. Reward `common_pool`. The four artifacts and the opponent byte-identical to `configs/training/rl_manager/rl_new_clones_s42.yml`. Env unchanged. Seeds 42–46. Equal environment episodes, not equal update steps. Evaluation identical in every arm: a fully deterministic rollout, every exploration mechanism off, batch 1000, at the same intervals, in the existing long schema.

## What this arm tests

The manager trains with epsilon-greedy fixed at 0.1 over 31 ordinal punishment levels. Uniform sampling there injects ~1.5 punishment points per member per round, the size of the entire learned signal, and on the finished runs the behaviour policy punished 1.7 to 6.6 times as hard as the policy being evaluated.

Worse for the outcome that actually matters: a uniform draw over 31 levels is **independent of the contribution it is aimed at**. It writes punishment-contribution pairs into the replay buffer whose contingency is noise. Two of the three finished seeds came out with that contingency *inverted* — punishing the full contributor harder than the free-rider — which is a plausible consequence.

Weight-space noise perturbs the *function* rather than the output. A perturbed policy still maps contribution to punishment in some coherent way, and a whole episode is collected under one such mapping. **Shape is the primary outcome; the behaviour-versus-evaluated gap is the mechanism.**

**A competing explanation this arm cannot rule out.** If the artificial humans respond to punishment without regard to whether it was deserved, there is no gradient toward correct targeting and no exploration method will produce one. This arm cannot distinguish that from an exploration failure. If the shape does not move, that is the first thing to test, and it is a different experiment.

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

**What `mad` still cannot do, stated plainly.** It is sensitive to the *size* of a perturbation but not to its *kind*: a large coherent shift and an inversion can both be large. No scalar adaptation target would separate them. That is why the shape itself is recorded as a first-class metric rather than inferred from the divergence — see below.

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

Counted from the config, not asserted: `n_update_steps` behaviour rollouts plus one evaluation rollout every `eval_period`, each of `batch_size` parallel episodes.

| | reference (`rl_new_clones_s42`) | this arm (`rl_pnoise_s4x`) |
|---|---|---|
| behaviour rollouts | 4,000 | 4,000 |
| evaluation rollouts | 200 | 200 |
| episodes per rollout | 1,000 | 1,000 |
| **behaviour episodes** | **4,000,000** | **4,000,000** |
| **evaluation episodes** | **200,000** | **200,000** |
| **total environment episodes** | **4,200,000** | **4,200,000** |

24 rounds each, so 100,800,000 agent-rounds per run. The arm changes no term in that product: it keeps 4000 update steps, eval period 20 and batch 1000, and each behaviour rollout still runs exactly once per update step. Equal episodes and equal update steps happen to coincide here, which they will not in every arm.

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

### The invariants

Policy network, target network and opponent all bit-identical before and after a full behaviour episode with the noise active and adapting (`invariants` in probe.json, all `true`; the probe asserts it rather than only reporting it). Structurally, `ParameterNoise.refresh` writes only into a private deep copy, and `test_the_acting_copy_shares_no_storage_with_anything_else` pins the aliasing.

## Inferred

Kept separate on purpose. None of the following is measured here.

1. **That the shape result is an exploration failure at all.** If the artificial humans respond to punishment without regard to whether it was deserved, there is no gradient toward correct targeting, the replay buffer's contingency is irrelevant, and no exploration method will help. This arm cannot distinguish that from the exploration story. If the shape does not move, that is the next experiment, and it is a different one.
2. **That coherent contingency-exploration helps rather than hurts.** The sweep shows weight noise reaching inverted contingencies as readily as human-signed ones. Collecting an episode under an inverted policy is a *better* experiment than collecting one under a policy with no contingency at all, but only if the reward can tell them apart.
3. **That a shared perturbation across the 1000 parallel episodes is enough.** One rollout is one replay-memory episode, so "once per episode" is once per rollout, and all 1000 parallel episodes share the draw. That gives 4000 distinct policy samples over a run and 1000 environment samples of each, rather than 4,000,000 independently dithered trajectories. Whether that trade is right is not measured.
4. **That the untrained-network geometry carries over to the trained regime.** The shape table is measured at initialisation.

## Status

## Successor
