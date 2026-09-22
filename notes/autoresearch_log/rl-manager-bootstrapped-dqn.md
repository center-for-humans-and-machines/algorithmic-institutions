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

*(filled in below as the pilots land)*

## Inferred

*(kept separate on purpose)*

## Successor

*(filled in below)*
