# rl-manager-reward-targeting

## Declaration

**Not a slot experiment.** No artificial-human model changes here. Nothing is trained: the six managers already exist and this branch is a simulation and an analysis.

**The question.** Learned RL managers in this project came out inverted — they punish full contributors hardest and spare free-riders, the reverse of the human policy (`notes/autoresearch_log/rl-manager-two-worlds.md`, §2). Four exploration arms were launched on the theory that this is an exploration failure. The first to report is a clean negative: its mechanism worked exactly as designed, the behaviour-versus-evaluated gap fell on all five seeds from 1.19 to 0.27, and the policy shape did not follow.

This branch tests the alternative, and it is arithmetic rather than speculation. Correct targeting raises contributions and costs members, and on the **undivided group pool** those two cancel. Every learned manager in this project was paid on that undivided pool. If that is right, correct targeting earns a manager nothing, the inversion is cheap by construction, and no exploration method can fix a gradient that is not there.

**The two arms.** Three seeds each, trained identically except for the reward.

| arm | managers | `reward_mode` | what it pays |
|---|---|---|---|
| pool | `rl_new_clones_s{42,43,44}` | `common_pool` | the group's undivided pool, `1.6*sum(c) - sum(p)` |
| per-capita | `rl_new_clones_percapita_s{42,43,44}` | `common_pool_per_capita` | that same pool over the valid headcount |

**Base branch.** `origin/auto/rl-manager-two-worlds`, which carries the cross-evaluation harness and the rule-based comparison points. `origin/auto/rl-manager-percapita-reward` is merged in for the per-capita arm's reward mode and training configs; it branched from the shared `0ff44a9` before the rule-sweep merge, so the merge is clean and additive.

**What the answer means.** If the per-capita managers are better targeted than their paired pool twins, the reward is implicated and the exploration programme has been aimed at the wrong thing. If they are not, the reward is exonerated and the inversion survives a change of objective, which makes it a property of the learning or the environment rather than the incentive.

## Measured

All tables below are committed under `plots/data_analysis/evaluation/rl_manager_reward_targeting/`: `shape_level_leaver.csv` and its `_s142` twin, `policy_shape.csv`, `policy_shape_n.csv`, `paired_differences.csv`, `incentive_contrasts.csv`, `noise_floor.csv`, `power.csv`, `guards.csv`, and `two_worlds_format/` — the same run read through the existing `scripts/rl_two_worlds/measure.py`, unmodified, so its rows drop straight into that script's tables. Two figures: `policy_shape_arms.jpg` (the six curves, paired by seed, against the human and clone) and `paired_differences.jpg` (the three paired differences against the measured noise floor).

### 1. The comparison really is controlled

Checked at parse level, not by eye. For each of the three seeds the two training configs are loaded with `yaml.safe_load` and compared key by key. Exactly three keys differ:

| key | pool arm | per-capita arm |
|---|---|---|
| `env_args.reward_mode` | `common_pool` | `common_pool_per_capita` |
| `job_id` | `rl_new_clones_s{S}` | `rl_new_clones_percapita_s{S}` |
| `output_dir` | `artifacts/manager/rl_new_clones_s{S}` | `artifacts/manager/rl_new_clones_percapita_s{S}` |

The last two are labels. `seed` pairs exactly (42↔42, 43↔43, 44↔44). Every world artifact — contribution model, valid model, switch model, opponent — is the same path in both arms, as are `lr`, `gamma`, `eps`, `target_update_freq`, the encodings, `n_update_steps` and every `env_args` key but the reward.

The **training code** is shared too, which the config diff alone would not show. Both arms descend from `0ff44a9`. Against that base the per-capita branch touches only `manager/environment.py` and its test; the environment change adds a new branch that returns `common_good_per_group` and leaves the `common_pool` path untouched. The two-worlds branch's own `src/` changes (`api_manager.py`, `simulate.py`) are simulation-side and post-date the training. So the reward is the only thing that differed while these six policies were being learned.

The six checkpoints are six distinct files (`md5sum`, all different), copied from the two arms' own run directories.

**And the two arms got the same amount of training**, which neither the config diff nor the code diff would have caught. All six SLURM jobs finished `COMPLETED` with exit code `0:0` in 5 h 59 m – 6 h 29 m, and all six training-metric parquets hold 163,200 rows reaching `update_step` 3980. Not "about the same": the same. A checkpoint saved early from a run that died would have been the quietest possible way to fake a reward effect, and it did not happen.

| job | arm | seed | elapsed | state | max `update_step` |
|---|---|---|---|---|---|
| 30401560 | pool | 42 | 06:04:57 | COMPLETED | 3980 |
| 30401561 | pool | 43 | 05:59:06 | COMPLETED | 3980 |
| 30401562 | pool | 44 | 06:02:44 | COMPLETED | 3980 |
| 30403651 | per-capita | 42 | 06:16:38 | COMPLETED | 3980 |
| 30403686 | per-capita | 43 | 06:28:36 | COMPLETED | 3980 |
| 30403808 | per-capita | 44 | 06:25:00 | COMPLETED | 3980 |

### 2. The harness

`configs/simulation/manager_testing/25_reward_targeting_cross_eval.yml`. Ten managers in one file: the six learned ones, the clone `lin_punisher`, and PR #207's rule controls `never`, `thr9_p10`, `prop10`. Each sits in group 0 against `lin_punisher` in group 1 — the competing setting. 100 episodes, 24 rounds, `switch_every: 4`, the training world slot for slot.

All ten share one config because `MultiManager` evaluates every manager in the config on every round, so a config's RNG stream depends on its manager **set** and two managers are only stream-comparable inside one file (PR #207 §3.0; `rule-based-manager-sweep.md` note 4). Adding the three per-capita managers therefore changed the set, and the pool-arm rows here are **not** expected to reproduce `24_rl_new_clones_cross_eval`'s digits. The claim rests on pool versus per-capita *within this file*.

`reseed_per_run: true` restarts every run from the config seed, so each manager meets the same episode-0 draw and the same validity pattern.

**The noise floor is measured, not assumed.** The whole config was run a second time as `25_reward_targeting_cross_eval_s142.yml`, identical in every parsed key but `seed` (42 → 142) and the two output labels. How far a row moves between the two runs is this harness's own Monte-Carlo noise, and a paired arm difference has to clear it.

### 3. Guards on the measurement itself

- **Validity.** `contribution` carries an imputed 9 for a player who gave no input, and `env.punish` forces that player's punishment to 0. `convert.load_sim` does *not* mask those rows, though `load_human` masks the human equivalents via `player_no_input`. Every mean in this log is therefore taken over `contribution_valid` rows only, from the raw parquet.
- **The pool identity.** The parquet's `common_good` is the env's state field, which is the pool **already divided** by the valid headcount, not the group total — the opposite sense to the `common_good` column of the human CSV. The group total is rebuilt as `common_good * n_valid` and checked against `1.6*sum(c) - sum(p)` per group-round.
- **The leaver flag comes from realised membership change.** `convert._derive_switching` compares a player's group at the decision round with its group at the next round. The switch predictor is never re-run, so its recurrent state is never disturbed.

### 4. Policy shape

Cross-evaluation `25_reward_targeting_cross_eval`, job **30419671**, 33 m 42 s, `COMPLETED 0:0`; replicate `..._s142`, job **30419818**, 33 m 31 s, `COMPLETED 0:0`. 100 episodes each. Every figure below is group 0's, over `contribution_valid` rows.

Mean punishment by the contribution it responds to, the suite's RPA bins, pool arm against its paired per-capita twin:

| bin | human | clone | `rl_s42` | `rl_pc_s42` | `rl_s43` | `rl_pc_s43` | `rl_s44` | `rl_pc_s44` |
|---|---|---|---|---|---|---|---|---|
| {0} | 4.755 | 3.913 | **0.077** | **0.349** | 11.997 | 0.888 | **0.193** | **0.751** |
| 1-5 | 2.973 | 2.580 | 0.159 | 0.328 | 0.979 | 0.476 | 0.279 | 1.133 |
| 6-10 | 1.672 | 1.683 | 1.357 | 1.201 | 0.000 | 0.364 | 1.384 | 1.252 |
| 11-15 | 0.978 | 1.019 | 4.745 | 2.581 | 0.000 | 0.213 | 2.000 | 1.850 |
| 16-19 | 0.692 | 0.905 | 5.000 | 3.406 | 0.000 | 0.072 | 2.000 | 2.239 |
| {20} | 0.267 | 0.254 | **5.000** | **4.261** | 0.000 | 0.000 | **2.000** | **2.230** |

Row counts (`policy_shape_n.csv`) run 345–3,172 per cell; the {0} and {20} figures that carry the claim rest on 951–1,702 agent-rounds each, so none of this is a support artefact.

**The per-capita managers are still inverted.** In seeds 42 and 44 the per-capita curve rises with contribution exactly as its pool twin does — punishing the full contributor hardest and sparing the free-rider. It rises a little less steeply in seed 42, and very slightly more in seed 44. Seed 43 is the one that changes a lot, and §6 shows that change is a level change, not a targeting change.

The three targeting measures, paired:

| measure | `rl_s42` | `rl_pc_s42` | `rl_s43` | `rl_pc_s43` | `rl_s44` | `rl_pc_s44` | human | clone |
|---|---|---|---|---|---|---|---|---|
| `shape_delta` = p{0} − p{20} | −4.923 | −3.912 | 11.997 | 0.888 | −1.807 | −1.480 | 4.488 | 3.659 |
| `shape_delta_norm` | −2.844 | −2.797 | 11.871 | 2.470 | −1.775 | −1.014 | 2.430 | 2.135 |
| **`targeting_rho`** | **+0.814** | **+0.564** | −0.457 | −0.459 | **+0.716** | **+0.499** | −0.319 | −0.289 |

`targeting_rho` is the Spearman correlation between contribution and punishment: negative is the human direction, positive is the inversion, and no rescaling of punishment can move it. The three paired differences on it:

| seed | pool | per-capita | difference | same, replicate |
|---|---|---|---|---|
| 42 | +0.814 | +0.564 | **−0.250** | −0.283 |
| 43 | −0.457 | −0.459 | **−0.002** | +0.032 |
| 44 | +0.716 | +0.499 | **−0.217** | −0.239 |
| mean | | | **−0.157** | −0.163 |

Two seeds move toward the human direction by about a quarter of a rho unit; the third, which was already correctly aimed, does not move. The mean reproduces almost exactly on the independent sim seed (−0.157 against −0.163), and the measured sim-seed noise floor for this statistic is **0.020** — so the two moves are real and roughly ten times the noise.

**They are also far too small.** `rl_pc_s42` at +0.564 and `rl_pc_s44` at +0.499 are still on the wrong side of zero, and the human sits at −0.319. Un-inverting seed 42 needs about −1.13; the reward change bought −0.25, some 22% of it.

The same verdict on the evaluation suite's own metric, straight out of `scripts/rl_two_worlds/measure.py` (`rpa_distance_vs_human`, lower is closer to the human policy):

| seed | pool | per-capita | difference | same, replicate |
|---|---|---|---|---|
| 42 | 3.120 | 2.347 | **−0.773** | −0.900 |
| 43 | 1.898 | 1.553 | **−0.345** | −0.476 |
| 44 | 2.317 | 1.926 | **−0.391** | −0.431 |
| mean | 2.445 | 1.942 | **−0.503** | −0.602 |

Six paired differences over two sim seeds, all six negative: the per-capita arm is consistently closer to the human policy. And yet `never` sits at **1.847** and the clone at 0.273 — so after the improvement, two of the three per-capita managers are *still further from the human policy than a manager that never punishes at all*. The reward change moves the arm about a quarter of the way to `never` and nowhere near the clone.

### 5. The leaver diagnostic

What leavers contributed minus what stayers contributed, at the decision rounds, from realised membership change.

**First, the diagnostic checks out as a classifier.** Across the ten managers in this config it tracks `shape_delta` at Pearson **r = −0.951** (sim seed 42) and **−0.887** (seed 142), against the −0.98 the earlier arms reported, and `targeting_rho` at **+0.902 / +0.927** — opposite sign only because the two conventions run opposite ways. Ordered by `targeting_rho`, the ordering is monotone almost without exception:

| manager | `targeting_rho` | `leaver_gap` |
|---|---|---|
| `prop10` | −1.000 | −5.444 |
| `thr9_p10` | −0.786 | −3.846 |
| `rl_pc_s43` | −0.459 | −1.278 |
| `rl_s43` | −0.457 | −2.402 |
| *human managers* | *−0.319* | *−2.027* |
| `lin_punisher` | −0.289 | −1.988 |
| `rl_pc_s44` | +0.499 | −1.213 |
| `rl_pc_s42` | +0.564 | −0.407 |
| `rl_s44` | +0.716 | −0.020 |
| `rl_s42` | +0.814 | +0.291 |

One correction to the framing this branch was handed: the inverted managers do **not** reliably "go positive". Only `rl_s42` crosses zero (+0.291); the other three inverted policies sit between −0.02 and −1.21. What the diagnostic reproduces is the *ordering*, not a sign test — the zero crossing is not at `targeting_rho` = 0.

**Second, it is useless for this particular contrast.** The three paired differences are −0.697, +1.124, −1.192 (mean −0.255, p = 0.75), and the replicate gives a quite different pattern, −0.549, −0.043, +0.454 (mean −0.046, p = 0.89). Its measured sim-seed noise floor is **0.577**, which is the same size as the differences themselves. The leaver gap orders managers well and resolves a quarter-rho shift not at all.

### 6. Level, not only shape

The arms do **not** differ by "the per-capita one punishes less", and the question has to be asked per statistic:

| statistic | `rl_s42`→`rl_pc_s42` | `rl_s43`→`rl_pc_s43` | `rl_s44`→`rl_pc_s44` | mean d | p | replicate mean d |
|---|---|---|---|---|---|---|
| `mean_punishment` | 1.731 → 1.398 | 1.011 → 0.359 | 1.018 → 1.460 | −0.181 | 0.63 | −0.306 |
| `punish_rate` | 0.390 → 0.387 | 0.082 → 0.231 | 0.489 → 0.722 | +0.126 | 0.21 | +0.106 |

Mean punishment falls in two seeds and rises in the third, and the paired difference is nowhere near consistent. The **rate** moves the other way and does so reproducibly: the per-capita arm punishes *more often* (+0.126, replicate +0.106) and, in two of three seeds, less hard. "Punished less overall" is not what happened.

**And here is why that matters more than it looks.** The largest paired difference anywhere in this study is seed 43's `shape_delta` of **−11.1** — and it is entirely a level artefact:

| | `rl_s43` | `rl_pc_s43` |
|---|---|---|
| `shape_delta` | **+11.997** | **+0.888** |
| `mean_punishment` | 1.011 | 0.359 |
| `punish_rate` | 0.082 | 0.231 |
| **`targeting_rho`** | **−0.457** | **−0.459** |

Seed 43's per-capita twin punishes a third as hard, spread over nearly three times as many rounds, aimed at *exactly the same people* — the rank correlation is unchanged in the third decimal. Read through `shape_delta` alone, that seed looks like the biggest targeting change in the experiment and it dominates the arm mean (which is why `shape_delta`'s paired sd is 6.8 and its `mde80` 21.1, both useless). Read through `targeting_rho`, it is a null. This is exactly the shape-versus-level confusion the brief warned about, and it is present in the data rather than hypothetical.

### 7. Both outcome measures, and the premise re-measured

**The premise's arithmetic is confirmed.** `thr9_p10` — the correctly-targeted rule — against `never`, re-measured in this run rather than quoted, with the brief's figures beside them:

| quantity | measured here | 95% CI | brief's figure | brief's CI |
|---|---|---|---|---|
| mean contribution | **+3.19** | [1.98, 4.39] | +3.01 | [2.33, 3.68] |
| group size | **−0.51** | [−0.92, −0.10] | −0.65 | [−0.91, −0.39] |
| pool per member | **+1.74** | [−0.56, 4.04] | +1.21 | [−0.03, 2.43] |
| **group total pool** | **+5.02** | **[−7.87, 17.90]** | +0.10 | [−7.07, 7.45] |

All four reproduce in sign and every one is compatible with the quoted interval. Correct targeting clearly raises contributions and clearly costs members, and on the undivided pool the two cancel into an interval that comfortably covers zero. **The incentive defect is real.** A manager paid the undivided pool genuinely is not paid for aiming correctly.

The two arms' own outcomes, against `never`:

| manager | contribution | group size | pool per member | group total pool |
|---|---|---|---|---|
| `rl_s42` | −0.36 | −0.47 | −2.60 | −19.6 |
| `rl_pc_s42` | −0.45 | −0.73 | −2.71 | −20.6 |
| `rl_s43` | +0.56 | −0.03 | −0.54 | +0.9 |
| `rl_pc_s43` | +0.27 | −0.25 | +0.20 | −2.4 |
| `rl_s44` | −0.46 | −0.44 | −1.82 | −15.9 |
| `rl_pc_s44` | +0.50 | −0.38 | −0.72 | −7.6 |
| `thr9_p10` | **+3.19** | −0.51 | **+1.74** | **+5.0** |

Paired arm differences: group total pool +1.36 (p = 0.74; replicate +3.21, p = 0.049), pool per member +0.58 (p = 0.25; replicate +0.48, p = 0.22). **The managers paid per capita did not raise the per-capita share** — `rl_pc_s42` sits at −2.71 against its twin's −2.60, and five of the six learned managers are still below a manager that never punishes. The one-line rule beats all six on every column.

### 8. The noise floor, and whether a null here is tight

Running the whole config twice at sim seeds 42 and 142 gives the movement a row shows for no reason but the draw. Averaged over the six learned managers:

| statistic | noise floor | mean paired difference | is the difference above the floor? |
|---|---|---|---|
| `targeting_rho` | **0.020** | −0.157 | yes, ~8× |
| `shape_delta` | 0.134 | −3.257 | yes, but see §6 |
| `mean_punishment` | 0.149 | −0.181 | marginal |
| `punish_rate` | 0.045 | +0.126 | yes |
| `leaver_gap` | **0.577** | −0.255 | **no** |
| `pool_per_member` | 0.984 | +0.581 | no |
| `group_total_pool` | 4.695 | +1.363 | no |

And the power read, from `power.csv`:

| statistic | mean d | sd d | paired-t 95% CI | `mde80` | within-arm seed spread |
|---|---|---|---|---|---|
| `targeting_rho` | −0.157 | 0.135 | [−0.492, +0.179] | **0.418** | pool 1.271, pc 1.022 |
| `shape_delta` | −3.257 | 6.809 | [−20.17, +13.66] | 21.08 | pool 16.92, pc 4.80 |
| `leaver_gap` | −0.255 | 1.220 | [−3.29, +2.78] | 3.78 | pool 2.69, pc 0.87 |
| `mean_punishment` | −0.181 | 0.562 | [−1.58, +1.22] | 1.74 | pool 0.72, pc 1.10 |

**On the question that matters, the null is tight.** For the reward to be the explanation, `targeting_rho` would have to move about **−1.1** — far enough to carry +0.81 and +0.72 across zero to the human's −0.32. The 95% interval on the paired difference is [−0.49, +0.18] and the design's `mde80` is 0.42, so an effect of that size was well inside reach and was not there. The hypothesis is excluded, not merely unsupported.

**On the small effect that *is* there, the evidence is suggestive and underpowered.** The −0.157 shift on `targeting_rho` and the −0.503 shift on the suite's RPA distance have p = 0.18 and p = 0.066, with the replicate at 0.24 and 0.056. That is a hard limit, not bad luck: with three paired seeds even a perfectly consistent effect cannot reach p = 0.05 on a sign test, which caps at 0.125. What carries the small effect is not its p-value but its reproducibility — 6 of 6 RPA-distance differences negative across two independent sim seeds, with arm means agreeing to within 0.1.

## Verdict

**The reward is a real defect and not the explanation. The inversion survives the change of objective.**

Both halves are needed, and the brief's binary does not quite fit what the data says.

**The arithmetic was right.** §7 reproduces it: correct targeting buys +3.19 contribution and costs −0.51 members, and on the undivided pool those cancel into [−7.87, +17.90]. Every learned manager in this project was paid on that pool, so it is true that correct targeting earned them nothing. The diagnosis of the incentive was sound.

**The predicted consequence does not follow.** Paying the per-capita share instead — the same world, the same seeds, the same 3,980 update steps, one key changed — leaves the managers inverted. `rl_pc_s42` at `targeting_rho` +0.564 and `rl_pc_s44` at +0.499 are on the same side of zero as the twins they replaced, and two of the three are still further from the human policy than never punishing at all. Removing the headcount term from the objective bought about a fifth of the distance to the human sign.

**The three paired differences, individually** (`targeting_rho`; per-capita minus pool):

| seed 42 | seed 43 | seed 44 |
|---|---|---|
| **−0.250** | **−0.002** | **−0.217** |

Consistent in sign where it moves at all, reproduced on an independent sim seed at −0.283 / +0.032 / −0.239, and roughly eight times the measured noise floor of 0.020. There is a small real effect of the reward on targeting. It is about 22% of the shift that would be needed, and it is not enough.

**Is this a tight null or an absence of evidence? Tight, for the claim under test.** The effect that would have mattered — a −1.1 shift on `targeting_rho` — sits far outside the paired 95% interval of [−0.49, +0.18] and well above the design's `mde80` of 0.42. Three paired seeds could have caught it. They did not, so the reward is excluded as the explanation rather than merely unconfirmed. The *smaller* claim, that the reward has some modest influence on targeting, is underpowered at p = 0.18 (replicate 0.24) and rests on reproducibility instead: six of six RPA-distance differences negative across two sim seeds.

**What this means for the programme.** The exploration arms were not aimed at the wrong thing on the evidence here — but neither were they aimed at the right one, because this branch has now removed the incentive explanation from contention. Two clean negatives now stand side by side: exploration was fixed and the shape did not follow; the reward was changed and the shape did not follow. That pushes the cause toward the learning dynamics or the environment — the credit assignment through a 24-round episode with a moving membership, the value function's treatment of a delayed benefit, or the world model itself.

**The level question, asked separately as instructed.** The arms do not differ by how much they punish in any consistent way: mean punishment −0.181 (p = 0.63), rate +0.126 the *other* way (p = 0.21). And the single biggest shape difference in the study, seed 43's −11.1 on `shape_delta`, is a pure level artefact whose `targeting_rho` is unchanged in the third decimal. A shape difference that is really a level difference is not the finding, and in this data it very nearly was.

## Inferred, not measured

Flagged separately because none of it is established by the measurements above.

1. **The scale confound is not settled, and it bears directly on the small effect.** `rl-manager-percapita-reward.md` inferred item 3 flagged it: the per-capita reward is the pool over roughly 4, so it has about a quarter of the pool's scale while `lr` and `gamma` are held fixed — as they must be for "only the reward differs" to hold. A weaker learning signal leaves a policy nearer its initialisation, and if the initialisation is less inverted than the converged policy, that alone would produce a small drift toward "less inverted" with no incentive content whatsoever. The observed −0.157 is exactly the size such an artefact could be. This arm cannot separate the two; the `pool / 4` control can, and it is the first successor item. **The tight null is unaffected** — a scale artefact would not hide a −1.1 effect — but the small effect might be scale rather than incentive.
2. **"Inverted" is read off `targeting_rho`'s sign**, which is a summary of a monotone relationship. Two managers with the same rho can punish quite differently; the bin table in §4 is the primary evidence and the rho is the compression of it.
3. **Three seeds per arm, and the within-arm spread is larger than the between-arm difference** on every statistic in §8's power table. The pairing is what makes three seeds informative at all, and it is doing all the work. Nothing here should be read as a statement about what a *typical* manager under either reward does.
4. **The claim about the exploration programme is an inference across branches**, resting on this branch's result together with the first exploration arm's reported negative, which was not re-verified here.
5. **The `leaver_gap` correlation is measured over ten managers, four of which are rules** whose behaviour is deterministic by construction. `prop10` at (−1.000, −5.444) and `thr9_p10` at (−0.786, −3.846) sit at one end and pull the correlation up. Over the six learned managers alone the relationship is weaker.

## Successor

1. **Run the `pool / 4` arm and settle the scale confound** (inferred item 1). A constant divisor has the pool's *shape* and the per-capita's *scale*. If it behaves like `common_pool`, the −0.157 found here is incentive; if it behaves like the per-capita arm, it is scale. That is one `reward_mode`, one `make_configs.py` entry and three seeds — and it is the only way the small effect in this branch becomes interpretable. It is cheap and it should be done before any further exploration arm.
2. **Stop testing the incentive and start testing credit assignment.** Two independent negatives now bracket it: exploration fixed, shape unchanged; reward changed, shape unchanged. The remaining candidates are the 24-round episode with moving membership (punishment costs now and pays back after a switch that may remove the beneficiary), the value function's horizon at `gamma` 0.98, and the world model's own response to punishment. A diagnostic worth more than another arm: measure the *realised* discounted return attributable to punishing a low contributor versus a high one, directly from training rollouts. If that quantity is itself inverted, the environment is teaching the inversion and no reward reshaping will help.
3. **Use `targeting_rho` as the headline targeting statistic, not `shape_delta`.** §6 shows `shape_delta` reporting the largest effect in the study for a seed whose targeting did not change at all. `shape_delta`'s paired sd here is 6.8 against `targeting_rho`'s 0.135, and its `mde80` is 21.1 against 0.42 — it is roughly fifty times the weaker instrument. Any future arm comparing policy shapes should carry both, and lead with the rank measure.
4. **Twelve seeds, if a per-arm claim is ever wanted.** `rl-manager-two-worlds.md` already priced this for the common-good comparison. For `targeting_rho` the paired sd of 0.135 means three seeds resolve 0.42; detecting the 0.157 actually observed at 80% power would need about **9 paired seeds**, roughly 108 A100-hours across both arms. Worth it only if item 1 first shows the effect is incentive rather than scale.
5. **Mask the imputed contribution in `convert.load_sim`.** `load_human` NaNs a timed-out player's contribution via `player_no_input`; `load_sim` does not, so the imputed 9 and its forced-zero punishment land in the `6-10` bin of every sim policy-shape table in the project. It moves the `{0}`−`{20}` statistic barely and the `6-10` bin visibly (for `rl_s42`, 1.357 filtered against 1.245 unfiltered in the published table). Left alone here deliberately — it changes numbers on several branches at once and is not this branch's call to make.
6. `notes/autoresearch_log/rl-manager-two-worlds.md` is the parent log; `rl-manager-percapita-reward.md` documents the arm whose checkpoints this branch consumed; `rule-based-manager-sweep.md` holds the rule comparison points.

## Notes

1. **`ArtificalManager.load` ignores the device it is handed.** `save` puts the model on CPU and `load` assigns straight through, so loading onto `cuda` yields CPU weights carrying a cuda device and the first forward pass dies. It did not bite here: `api_manager.RLManager` already loads with `device=th.device("cpu")` and the whole manager path in `simulate.py` is CPU — `create_data` builds CPU tensors and the returned punishments are moved to the env's device afterwards. The file was left alone, as other branches are touching it.

2. **The isolated remote dir is `~/repros/ai-runs/rl-reward-targeting`**, created fresh for this branch. The four exploration arms' dirs (`rl-anneal-local`, `rl-bootstrapped-dqn`, `rl-es`, `rl-param-noise`) and the two source arms' dirs were not touched; the checkpoints were copied out of them, never moved. `simulate_cluster.sh` excludes `artifacts/manager/` and `plots/` from its `--delete` sync, so a later `--sync-only` cannot remove the staged checkpoints or the results.

3. **Why `measure_arms.py` exists next to `scripts/rl_two_worlds/measure.py`.** `measure.py` was reused for the rows it already defines and its output format is unchanged. The new script adds three things it does not do: the validity filter of guard 3, the separation of shape from level, and the group-total-pool measure. It reuses the suite's own bins (`RPA_EDGES`) and the suite's own switch labelling rather than redefining either.
