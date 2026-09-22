# rl-manager-bootstrapped-dqn

## Declaration

**Not a slot experiment.** This branch changes no artificial-human model and is not judged by the §2 gates of `notes/autoresearch.md`.

**One arm of a four-way exploration comparison.** The other three arms — annealed local epsilon-greedy, parameter-space noise, evolution strategies — are being built by sibling agents against the identical contract. Nothing here touches their work.

**Base branch.** `auto/rl-manager-two-worlds` at `0ff44a9` ("Both guards pass; launch the three seeds"). The world, the reward and the opponent are that commit's, unchanged.

**The one variable.** The exploration mechanism. `reward_mode: common_pool`, the four artifacts, the opponent, and every `env_args` key are byte-identical to `configs/training/rl_manager/rl_new_clones_s42.yml` at `0ff44a9`. The only config keys that differ are `seed`, `job_id`, `output_dir` and the four `manager_args` keys that define the mechanism.

## The question

The manager trains with epsilon-greedy fixed at 0.1 over 31 ordinal punishment levels, 0 to 30. A uniform resample there has expected value 15, so at eps=0.1 the behaviour policy injects an expected **1.5 punishment points per member-round** that the evaluated policy never chose — the size of the entire learned signal (human mean punishment is 1.79). Measured on the finished runs, the behaviour policy punishes 1.7 to 6.6 times as hard as the policy being evaluated.

The hypothesis under test across all four arms is that this task needs **consistent** behaviour and that action-level dithering destroys it. This arm is the most direct test, because bootstrapped DQN (Osband, Blundell, Pritzel and Van Roy, 2016) replaces dithering with a coherent alternative policy held fixed for a whole episode.

**The primary outcome is policy shape, not the gap.** Human managers are monotone decreasing in the contributor's own contribution: 4.76 at contribution 0 falling to 0.27 at 20 — punish the free-rider, leave the full contributor alone. Two of the three finished epsilon-greedy seeds came out **inverted** (seed 42: 0.08 rising to 5.00; seed 44: 0.23 rising to 2.00), monotone in the wrong direction across all six bins on 1,000–3,000 rows per bin; the third has the human sign but fires on only 7.4% of rounds. Against the evaluation suite's own distance measure no learned seed is closer to the human policy than never punishing at all.

Uniform exploration over 31 levels applies punishment **independently of the contribution it is aimed at**, which decorrelates punishment from contribution in the replay buffer — a plausible route to a policy whose shape is arbitrary and seed-determined. A head held fixed across an episode produces a coherent contingency between contribution and punishment rather than a dithered one. So: **shape is the outcome, the behaviour-versus-evaluated gap is the mechanism.**

**A competing explanation this arm cannot rule out.** The artificial humans may respond to punishment without regard to whether it was deserved. If so there is no gradient toward correct targeting and no exploration method will fix the shape. This arm cannot distinguish "exploration destroyed the contingency" from "there is no contingency to learn", and nothing below should be read as distinguishing them. Separating the two needs a probe of the contribution model's response to punishment conditioned on the punished player's own contribution — see **Successor**.

## The mechanism, as implemented

K value heads on a shared torso. The head **is** the `op2` readout: everything below it (the encoders, the `op1` message-passing layer, the per-agent and global GRUs) is shared; the final linear map is private to a head. In `GraphNetwork` this is one integer — `out_features = y_levels * n_heads` — and the reshape `(E, G, A, T, K, n_actions)` in the manager.

- **Behaviour.** One head drawn uniformly per parallel episode, held for all 24 rounds, followed greedily. **No epsilon at all**: `eps` remains in the config at its historical 0.1 and is unused under `exploration: bootstrap`.
- **Training.** A Bernoulli(p) bootstrap mask per (episode, head), drawn once when the episode is generated and stored in the replay buffer with that episode's transitions, so it is a fixed property of the data rather than resampled on every draw. Head k's Huber loss sees only the episodes its own mask kept, normalised by exactly those terms. Each head bootstraps off **its own** target head: head k's TD target is `max_a Q_target_k`, never the ensemble's.
- **Evaluation.** The consensus, defined precisely below, with every exploration mechanism off.

### The consensus rule

**Mean-of-Q**: `a* = argmax_a (1/K) Σ_k Q_k(s, a)`.

Not a vote over per-head argmaxes. On an ordinal action space of 31 levels a plurality vote fragments across neighbouring levels and its winner turns on tie-breaking; the head-averaged Q is the ensemble's own value function and moves smoothly with the heads. `test_consensus_is_mean_of_q_not_a_vote` pins the difference with a case where three heads split 2–1 and the two rules disagree. The two rules are not assumed equivalent: `consensus_vote_agree` and `consensus_vote_gap` are logged every round, so how often they differ on the visited states is measured.

### K = 1 with no masking reproduces the existing agent

Four assertions, all on Raven (`src/aimanager/tests/test_bootstrapped_manager.py`):

- `test_k1_builds_bit_identical_parameters` — a seeded `GraphNetwork` built with `n_heads` absent and one built with `n_heads=1` agree parameter for parameter. The head machinery draws no RNG and changes no shape.
- `test_k1_reproduces_the_existing_agent` — the chosen action and the returned loss are `th.equal` to the pre-bootstrap code, which is copied verbatim into the test file as `_legacy_action` / `_legacy_loss`.
- `test_k1_with_all_ones_mask_matches_the_unmasked_loss` — turning the masking path on with nothing masked out changes no number.
- `test_a_masked_out_head_receives_no_gradient` — with only head 1's mask set, heads 0, 2 and 3's readout rows move by exactly 0.0.

## Measured

Everything in this section is a number a run printed or a CSV in `plots/data_analysis/evaluation/rl_manager_bootstrapped_dqn/`. Nothing here is an interpretation.

### 1. The episode budget, counted

`rl_manager.EPISODE_BUDGET` increments once per `run_batch` call, after the rollout, and prints a `[budget]` line at the end of training. Each 200-step pilot printed:

```
[budget] {"behaviour_episodes": 200000, "episode_rounds": 5280000,
          "episodes": 220000, "eval_episodes": 20000, "rollouts": 220}
```

200 behaviour rollouts + 20 evaluation rollouts = 220, times `batch_size` 1000 = 220,000 episodes, times `n_rounds` 24 = 5,280,000 episode-rounds. Both identities hold exactly, and the total cross-checks against env rounds over `n_rounds`, which is how the annealed-local arm counts it.

The counter deliberately does **not** count `env.reset()`. `ArtificialHumanEnv.__init__` resets once without playing an episode, so a reset-counted budget overstates every run by one rollout. `_StubEnv` in the test resets in its constructor and counts resets specifically so that mistake would fail the test.

**The budget I matched: 4,200,000 environment episodes** — 4,000,000 behaviour (4000 update steps x 1000 parallel episodes) and 200,000 evaluation (200 rollouts at `eval_period` 20 x 1000), 100,800,000 episode-rounds. My arm consumes exactly that at exactly the reference's `n_update_steps`, because the mechanism changes which action is selected, not how many episodes are played.

### 2. Cost of K

Seconds per update step, 200-step pilots, from each job's final tqdm line:

| K | 1 | 4 | 10 | 20 |
|---|---|---|---|---|
| s/update step | 5.56 | 5.72 | 6.89 | 5.84 |

Not monotone: the K=10 job drew a slower node. **Cost does not constrain K up to 20.** The head is `31*K` outputs off a 100-wide readout, negligible beside the artificial humans' forward passes. The 4000-step runs price at ~6.2-7.7 h each on an A100, inside the template's 20 h limit.

### 3. Ensemble survival, and the K decision

`head_survival.csv`. Seed 42, 200 update steps, one variable per row.

| run | K | assignment | disagree@0 | disagree@end | spread@0 | spread@end | behaviour p@end | evaluated p@end |
|---|---|---|---|---|---|---|---|---|
| k1 (control, eps-greedy) | 1 | — | — | — | — | — | 4.92 | 4.18 |
| k4 | 4 | per_episode | 1.000 | **0.000** | 19.2 | **0.0** | **0.000** | **0.000** |
| k10 | 10 | per_episode | 1.000 | 1.000 | 25.1 | 2.0 | 0.179 | 0.000 |
| k20 | 20 | per_episode | 1.000 | 1.000 | 25.7 | 3.0 | 2.908 | 2.628 |
| k10_per_rollout | 10 | per_rollout | 1.000 | **0.000** | 25.4 | **0.0** | 7.695 | 7.666 |

At initialisation every ensemble is maximally diverse — disagreement 1.000, mean action spread 19-26 levels out of 31 — which is what `test_heads_are_separate_parameters_and_disagree_at_init` asserts and what makes exploration possible with no epsilon at all. What happens next is the finding: **at K=4 the ensemble is dead within 10 update steps.** Disagreement drops from 1.000 to exactly 0.000 between step 0 and step 10 and never recovers; every head picks punishment 0 on every cell for the remaining 190 steps. With no dithering there is then no exploration of any kind.

**K = 20, chosen on this.** Survival is the binding constraint and it is monotone in K over the range tested: dead at 4, alive with spread 2.0 at 10, alive with spread 3.0 at 20. K=20 is also the only bootstrap configuration whose **evaluated** policy still punishes at all at step 190 (2.63, against 0.00 at both K=4 and K=10) — a consensus pinned at zero cannot express a policy shape, and shape is this comparison's outcome measure. Cost, per §2, does not argue against it.

### 4. How the heads map onto the batch — the correlation check

The question was whether assigning heads across the 1000 parallel episodes quietly correlates them through the shared replay. **Measured: it is the other assignment that correlates them.** At the same K=10 and otherwise identical config, `per_rollout` collapses to disagreement 0.000 while `per_episode` holds at 1.000 for all 200 steps.

Two further facts, both asserted in tests rather than argued: the bootstrap mask is drawn independently of the head draw (`test_mask_is_bernoulli_and_independent_of_the_head_draw` checks every per-head correlation between "head k generated this episode" and "head k's mask kept this episode" is under 0.05 on 20,000 draws), and a masked-out head's readout rows move by exactly 0.0 (`test_a_masked_out_head_receives_no_gradient`).

### 5. The behaviour-versus-evaluated gap

`gap_summary.csv`, last evaluation point of each pilot.

| run | behaviour | evaluated | gap | ratio |
|---|---|---|---|---|
| k1 (control) | 4.919 | 4.179 | 0.740 | 1.18 |
| k4 | 0.000 | 0.000 | 0.000 | — |
| k10 | 0.179 | 0.000 | 0.179 | — |
| k20 | 2.908 | 2.628 | 0.280 | 1.11 |
| k10_per_rollout | 7.695 | 7.666 | 0.030 | 1.00 |

At the chosen configuration the ratio is **1.11 against the control's 1.18**, and the absolute gap 0.28 against 0.74.

### 6. The shape-distortion diagnostic is not identified on this pilot

`shape_distortion_summary.csv` and the per-job `shape_distortion.csv`. Reporting the negative result first: **at 200 steps every configuration, including the epsilon-greedy control, is a constant policy with no dependence on contribution.** Evaluated slopes (top RPA bin minus bottom) run 0.00 to 0.16 against the human −4.49. With the evaluated bin means near-identical, the predicted per-bin shift `eps * (15 − evaluated)` has almost no variance across bins, so the fitted regression slopes it produces (8.89 for the control, −7.43 at K=20, −0.80 for per_rollout) are noise and **must not be read as the annealed arm's 0.891 can be**. That arm's pilot ran 300 steps and had bin spread to work with; mine does not.

What the diagnostic does give at this length:

- The **magnitude** prediction holds on the control. Epsilon-greedy predicts a mean per-bin shift of `0.1 * (15 − 4.18) = 1.08`; the measured mean absolute per-bin shift is **1.028**.
- Mean absolute per-bin shift: control 1.028, K=10 0.326, **K=20 0.345**, per_rollout 0.006. But with every shape flat, that shift is a level offset, so at this run length the diagnostic degenerates into §5 and adds nothing to it.

Per-bin table, count-weighted over the last five evaluation points, human and clone columns from `auto/rl-manager-two-worlds` (`policy_shape_all.csv`):

| bin | human | clone | k1 | k4 | k10 | k20 | k10_per_rollout |
|---|---|---|---|---|---|---|---|
| {0} | 4.755 | 3.721 | 4.827 | 0.0 | 0.0 | 2.893 | 9.727 |
| 1-5 | 2.973 | 2.930 | 4.860 | 0.0 | 0.0 | 2.930 | 9.663 |
| 6-10 | 1.672 | 1.808 | 4.906 | 0.0 | 0.0 | 2.944 | 9.800 |
| 11-15 | 0.978 | 1.300 | 4.943 | 0.0 | 0.0 | 2.962 | 9.866 |
| 16-19 | 0.692 | 1.066 | 4.926 | 0.0 | 0.0 | 2.953 | 9.849 |
| {20} | 0.267 | 0.327 | 4.938 | 0.0 | 0.0 | 2.964 | 9.891 |

Row counts are in `policy_shape_all.csv`; every bin carries 22,000-164,000 agent-rounds, so flatness is not a sample-size artefact.

Every head's own RPA slope at K=10 step 190 is exactly 0.000 — the surviving heads are **constant** policies differing only in level, so `head_slope_sign_spread` is 0 and the question "do the heads disagree about the sign" has no answer yet: there are no signs to disagree about.

### 7. Launched

Five seeds, 42-46, K=20, `per_episode`, `bootstrap_p` 0.5, `reward_mode: common_pool`, 4000 update steps. SLURM 30413435, 30413438, 30413439, 30413441, 30413443. `AI_REMOTE_DIR=~/repros/ai-runs/rl-bootstrapped-dqn`.

## Inferred

Kept separate on purpose. None of this is measured.

1. **Why the ensembles die, probably.** The common-pool reward is `1.6*sum(c) − sum(p)`, so punishing carries an immediate, unambiguous cost and the delayed benefit only arrives through the contributors' response. With the head being a single linear readout on a torso that is otherwise entirely shared, and with each head's Bernoulli(0.5) subsample still containing ~500 episodes and ~48,000 agent-rounds, the per-head data are statistically near-identical and all heads receive the same unambiguous "punishment costs" signal. Larger K appears to help only because more heads means more initial draws far enough from the mode to survive a while. This is a hypothesis consistent with the table in §3; I did not test it.

2. **The pilot is a poor predictor of the full run, and for a concrete reason.** `target_update_freq` is 1000 and the pilots ran 200 steps, so **the target network never synced once**. Every TD target in every pilot was bootstrapped off the randomly initialised target net. The mechanism by which punishment could ever look worthwhile — long-horizon value propagating back from higher contributions — cannot operate before the first sync. I would not predict the 4000-step behaviour from these 200 steps in either direction.

3. **Taking the coordinator's caution seriously, and extending it.** The warning was not to read a pilot's shape as evidence the arm fixes the inversion, because the inversion is a late-training phenomenon. The same caution cuts the other way here and I want it on the record: **do not read this pilot's collapse as evidence the arm fails either.** At 200 steps my control has no shape at all, so the pilot cannot discriminate any hypothesis about shape. What it does establish is narrower and does not depend on run length: an ensemble that has lost its disagreement cannot explore, and at K=4 that happens by step 10.

4. **What a lower behaviour punishment is not.** K=20's behaviour punishment of 2.91 against the control's 4.92 is not an improvement. Both are far above the human 1.79, the shapes are flat in both, and per the coordinator a lower level in a pilot is not a result.

5. **The competing explanation this arm still cannot touch.** If the artificial humans respond to punishment without regard to whether it was deserved, there is no gradient toward correct targeting and no exploration method fixes the shape. Nothing above distinguishes that from "exploration destroyed the contingency". I am not claiming it does.

## Successor

1. **Read the five runs on shape, not on level.** `python scripts/rl_bootstrapped/guard.py artifacts/manager/rl_bootstrap_s4*/metrics/*.parquet`. The primary outcome is the per-bin table against the human and clone columns; the sign of `rpa_slope` is the headline and `head_slope_sign_spread` is the number this arm exists to produce. For a shape comparable cell-for-cell with `auto/rl-manager-two-worlds`, run a cross-evaluation simulation of the saved managers and feed it to `scripts/rl_two_worlds/measure.py`, which is the path the finished epsilon-greedy seeds went through.

2. **A device trap on that path.** `ArtificalManager.load` ignores the device it is handed: `save` puts the model on CPU and `load` assigns straight through, so loading onto cuda gives CPU weights with a cuda `self.device` and the first forward pass dies. The only existing caller loads on CPU and never hits it. Add one `.to(device)` at your own call site rather than changing the manager — that file is being edited by three other exploration arms concurrently and is not an exploration arm's business to fix.

3. **If the ensembles have died again at 4000 steps**, the question to ask is whether the head has enough private capacity to hold a different policy, not whether K was large enough. The head here is one linear map on an otherwise fully shared torso; the 2016 paper's Atari heads are small networks on a shared *convolutional* trunk, a far smaller shared fraction. A per-head hidden layer is the minimal in-frame change and is a separate experiment, one variable, not a patch to this one.

4. **`bootstrap_p` is untested.** It is switchable and was held at 0.5 throughout. At 1000 parallel episodes, p=0.5 leaves each head ~48,000 agent-rounds, which is very likely too many for the bootstrap to create any real diversity. A p-sweep at fixed K=20 is cheap (200-step pilots price at ~20 min each) and is the obvious next measurement if §Inferred-1 is right.

5. **Re-run the shape-distortion diagnostic when there is shape to distort.** It is implemented in `guard.py` and produces `shape_distortion.csv` per job; it just had nothing to bite on at 200 steps. On the finished runs, compare `distortion_slope` against the annealed arm's 0.891 control and 0.067 treatment.
