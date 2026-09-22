# The per-capita reward arm: a headcount control on `rl-manager-two-worlds`

## Declaration

**Not a slot experiment.** No artificial-human model, feature or training config changes. This branch adds one reward mode to the RL environment and launches three RL-manager runs under it. It is not judged by the §2 gates of `notes/autoresearch.md`.

**Base.** `auto/rl-manager-two-worlds` at `0ff44a9` ("Both guards pass; launch the three seeds") — the exact tree the running seeds launched from, deliberately not rebased onto anything newer. Rebasing would have made the arms differ in more than the reward, which is the one thing this experiment exists to hold fixed.

**Scope.** A control arm, and nothing else. The running arm (jobs **30401560 / 61 / 62**, `~/repros/ai-runs/rl-two-worlds`) pays the manager its own group's *undivided* common pool. This arm pays the *per-capita share* of that same pool. `reward_mode`, `job_id` and `output_dir` are the only config keys that differ; the isolated remote dir is separate (`~/repros/ai-runs/rl-percapita`) and nothing about the running arm was touched.

**The question the control answers.** The undivided pool pays for headcount: a bigger group is a bigger pool. All three running seeds are learning to punish less, gain members and grow the pool while contribution per member stays flat or falls, so it cannot be told whether the manager learned something about cooperation or simply learned to collect people. The per-capita share is, to first order, headcount-neutral. If the punish-less behaviour survives the change it is about cooperation. If it disappears it was about collecting people.

## The reward, exactly

```
reward_mode: common_pool_per_capita

reward(group, round) = (1.6 * sum(c) - sum(p)) / n_valid
```

where the sums run over the group's members with the timed-out cells zeroed, and

**the divisor `n_valid` is `count_valid_per_group`** — the number of players in the group who **gave an input that round**, *not* the group's membership (`count_members_per_group`). It comes from the env's single `share_pool_per_group`, which is the same divisor the `common_good` *state field* is built from; no second divisor was written.

**The two can differ, and here is what happens when they do.** They differ whenever a member times out: that member counts towards the membership but not towards `n_valid`. A timed-out player contributed 0, was punished 0 and took no share, so they move neither the numerator nor the divisor — the reward simply does not see them. This is right for three reasons, in descending order of force:

1. **It is the game's own division.** `reports/basics.md`: the pool "is splitted equally between the contributors". `payoff = 20 - c - p + pool/n_valid` reproduces the human `payoff` column to a maximum residual of 7.1e-15 over 19,166 agent-rounds, including all 526 timed-out rows with `n_valid > 0` (measured in `notes/autoresearch_log/manager-common-pool-reward.md`, re-derived here at 2.842e-14 for the pool identity).
2. **Dividing by the membership would invent a penalty the game never charged.** It would pay the manager less for holding a member who did not play than the game actually paid that group's players.
3. **It keeps the identity with the `common_good` state field**, which is what lets the launch guard cross-check the reward through a code path (`update_common_good`) that the reward path never touches. A second divisor would have destroyed that check.

A group where nobody gave an input has a zero pool and no share to hand out; `share_pool_per_group` returns 0 there, as it does for an empty group.

**The naming trap, named.** `common_good` already carries two meanings in this project: the env's `common_good` **state field** is the per-capita share, while the `common_good` **column** of `experiments/2group_8agent_50ep.csv` is the undivided pool. The new mode therefore carries no `common_good` in its name — it says per-capita on its face, so the word is not handed a third sense. The comment at `REWARD_MODES` in `src/aimanager/manager/environment.py` states this at the definition.

## Measured

Everything in this section is a number that was computed. Nothing here is a prediction.

### 1. How much of each reward is headcount (human data, local)

`experiments/2group_8agent_50ep.csv`, group-rounds keyed `(episode_id, round_number, group_id)`: 4,512 non-empty, of which 4,478 have `n_valid > 0`. `1.6*sum(c) - sum(p)` reproduces the CSV's `common_good` column to **2.842e-14**, which re-derives the sibling log's identity and fixes the numerator.

| reward | corr with `n_valid` | R² on `n_valid` | R² on mean contribution per valid member |
|---|---|---|---|
| `common_pool` | **+0.5750** | **0.3307** | 0.4209 |
| `common_pool_per_capita` | **−0.0299** | **0.0009** | **0.8771** |

| reward | OLS slope per extra valid member | in sd of the reward | reward sd |
|---|---|---|---|
| `common_pool` | **+13.279** | +0.290 | 45.762 |
| `common_pool_per_capita` | **−0.145** | −0.015 | 9.623 |

| `n_valid` | group-rounds | mean `common_pool` | mean per-capita | mean contribution |
|---|---|---|---|---|
| 1 | 486 | 15.29 | 15.29 | 10.53 |
| 2 | 564 | 28.21 | 14.11 | 9.88 |
| 3 | 566 | 37.70 | 12.57 | 8.90 |
| 4 | 1,132 | 48.56 | 12.14 | 9.35 |
| 5 | 508 | 77.14 | 15.43 | 10.39 |
| 6 | 528 | 82.46 | 13.74 | 9.52 |
| 7 | 460 | 95.29 | 13.61 | 9.37 |
| 8 | 234 | 97.92 | 12.24 | 8.45 |

So on the data the humans generated, an extra playing member is worth **+13.3 pool points** under the running arm's reward and **−0.1** under this one — a factor of 92 — and the variance headcount explains falls from **33.1% to 0.09%** while the variance explained by per-member cooperation rises from **42.1% to 87.7%**. That is what "headcount-neutral to first order" means here, stated as a measurement rather than as an argument. The probe is `scratchpad`-only and not committed; it is 40 lines of pandas over the committed CSV and the group-by key is the only choice it makes.

### 2. The three launch guards (Raven, `scripts/rl_two_worlds/launch_guards.py`)

Real config, real models, real opponent, batch 64 × 24 rounds × 2 groups, seed 42, with the manager forced to punish the **maximum** on every cell so the reward sits far from 0 and nothing is satisfied trivially. Evidence committed at `plots/data_analysis/evaluation/rl_manager_percapita/guards.json`. **`ALL_PASS: true`**, all eleven verdict flags true.

**Guard 1 — the reward is exactly the pool over that divisor.**

| check | max abs residual |
|---|---|
| vs the guard file's *own* division of the env's state tensors | **0.0** (exact) |
| vs the `common_good` state field — produced by `update_common_good`, never by the reward path | 1.9e-06 (float32) |
| vs the same quantity without zeroing the timed-out cells | 14.4 — so the zeroing is load-bearing, an accounting-side second read on guard 2 |

**Guard 2 — the free-punishment lever is closed.** Of **36,864** agent cells served, **1,659** were timed out (4.50%) and every one was served `punishment` **0.0**; all **627** previously-timed-out cells were served `prev_punishment` **0.0**. Identical to the running arm's evidence, cell for cell.

**Guard 3 — divided, not relabelled.**

| check | value |
|---|---|
| (a) same trajectory as the `common_pool` rollout | max residual on the pool **0.0** |
| (a) gap between the two rewards | **max 224.0**, mean **46.4** points per group-round |
| (a) ratio `common_pool / per_capita`, free rollout | takes **all of 1…8**, max deviation from an integer **4.8e-07**, over 2,725 group-rounds |
| (b) `reward * n_valid` returns the pool | max residual **7.6e-06** (float32) |
| (c) equal-headcount rollout: members / `n_valid` | 4 / 4 on every group-round, by construction |
| (c) ratio there | the **single** value **4.0**, max deviation **0.0**, over 3,067 group-rounds |
| (c) `reward * 4` vs the `common_pool` reward there | max residual **0.0** (exact) |

**One deviation from the brief, stated.** The brief asked that the two rewards "agree exactly when every group holds the same number of members". `pool` and `pool / n` can only be numerically equal at `n = 1`, so that leg is implemented in the only form it can hold: on the controlled equal-headcount rollout the ratio is a **single** value and it is **exactly the integer headcount**, with `reward * 4` returning the `common_pool` reward to residual 0.0. That is still the test the brief wanted — a relabelling would give a ratio of 1, and a divisor that was not really the per-group headcount would not give 1…8 in the free rollout and a single 4 here.

**The guard rewrite did not break the running arm's certification.** Two checks, both passed.

First, three quantities *inside this run* reproduce the running arm's committed evidence bit for bit: the `common_pool` mean reward `25.729751586914062`, the whole `sum` contrast (`24.383913040161133`, max 240.0, mean 42.83356857299805) and all four guard-2 counts. Same world, same trajectory, only the reward moved.

Second, the rewritten guard was re-run against `configs/training/rl_manager/rl_new_clones_s42.yml` itself — the running arm's own config, untouched. `ALL_PASS: true`, and its `guard1`, `guard2` and `guard1c_sum_contrast` blocks are **equal object-for-object** to the committed `guards_after_fix.json` from before the rewrite (the only addition is a new `expected` annotation string). Evidence at `plots/data_analysis/evaluation/rl_manager_percapita/guards_pool_arm_recheck.json`.

### 3. The configs differ in three keys

Generated from `scripts/rl_two_worlds/make_configs.py`, never hand-edited. Regenerating also re-emits the running arm's three configs, and `git status` shows them **unmodified** — which is the mechanical proof that the generator change did not disturb them.

Per seed, over the parsed YAML: **34 leaf keys, 31 identical, 3 different.**

```
env_args.reward_mode : 'common_pool'                      -> 'common_pool_per_capita'
job_id               : 'rl_new_clones_s42'                -> 'rl_new_clones_percapita_s42'
output_dir           : 'artifacts/manager/rl_new_clones_s42'
                                                          -> 'artifacts/manager/rl_new_clones_percapita_s42'
```

and within the new arm, s42 vs s43 and s42 vs s44 differ in `seed`, `job_id` and `output_dir` only. `lr`, `gamma`, `eps`, `target_update_freq`, the encodings, `n_update_steps: 4000`, `eval_period: 20`, `batch_size: 1000`, `n_rounds: 24`, `switch_every: 4`, all four model artifacts and the opponent are byte-identical across both arms and all six runs.

### 4. Tests

`src/aimanager/tests/test_manager_reward.py`: **18 pass on Raven** (`scripts/remote_test.sh --test-only`, isolated dir), 18 pass locally too — the file is plain torch and imports no PyG. Seven are new and pin the per-capita mode: the plain round, the divisor parting company with the membership when somebody times out, the empty group, the everyone-timed-out group, agreement with the `common_good` state field, the ratio to `common_pool` being exactly the headcount and never 1, and the reshuffle round where group 0 gains a fifth member and the divisor grows with the pool.

## Inferred, not measured

Flagged separately because none of it has been observed yet.

1. **"Headcount-neutral" is a static measurement on human data, not a closed-loop guarantee.** Section 1 measures the *reward function's* sensitivity to headcount on rounds humans played. Under training the manager also moves the switch model, and a member who contributes *above* the group average still raises the per-capita share. A headcount channel can therefore survive; it is 92× weaker at the function level, not absent.
2. **The reading of the outcome is a design, not a finding.** "Survives ⇒ cooperation, disappears ⇒ headcount" is the intended interpretation of a result that does not exist yet.
3. **The arm changes the reward's scale as well as its shape, and this is a real confound.** The per-capita reward is the pool over ≈4, so its sd on the human data is 9.62 against the pool's 45.76, and its mean under the guard probe is 1.33 against 25.73. `lr` and `gamma` are held fixed — they must be, for "only the reward differs" to be true — so smaller rewards mean smaller TD errors and effectively slower value learning. **If this arm shows less of everything, scale is a candidate explanation alongside incentive, and this arm alone cannot separate them.** The successor section says how to.

## Status: all three guards pass, three seeds running

Submitted from `AI_REMOTE_DIR=~/repros/ai-runs/rl-percapita` — a **separate** isolated dir from the running arm's `~/repros/ai-runs/rl-two-worlds`, which was not touched. 20 h wall limit, unchanged from the shared `scripts/manager/run_training.sh` template.

## Results

| seed | config | SLURM job | job name | state at hand-back |
|---|---|---|---|---|
| 42 | `rl_new_clones_percapita_s42.yml` | **30403651** | `e25fd1a5` | `RUNNING`, step 40/4000, 5.54 s/step, ETA ~6 h 06 m |
| 43 | `rl_new_clones_percapita_s43.yml` | **30403686** | `7afb019a` | `RUNNING`, step 23/4000, ~6.5 s/step |
| 44 | `rl_new_clones_percapita_s44.yml` | **30403808** | `752929fb` | `RUNNING`, step 7/4000, ~7.7 s/step |

All three are past environment construction and into the step loop with no errors, and each printed the reward mode it is actually training on:

```
Creating environment with {..., 'reward_mode': 'common_pool_per_capita'}
[two-manager] rl_group_id=0, env.n_groups=2, env.n_agents=8,
              reward_mode=common_pool_per_capita, switch_predictor=on
```

The per-step rate matches the cost pilot's 5.2–5.65 s/step, so the reward change costs nothing in wall clock, as expected. The running arm (30401560/61/62) was at 2 h 24 m of its own 20 h limit when these were submitted, and all six now share the association's 8 concurrent slots.

The behavioural table this arm exists to fill, to be read **against the running arm's, per seed, never pooled**:

| seed | punish rate | mean p | mean p given p>0 | mean p at `contribution_valid=False` | members held | contribution per member | common good | verdict |
|---|---|---|---|---|---|---|---|---|
| — | not finished | | | | | | | |

## Successor

1. **Read the two arms together, per seed.** The comparison is `rl_new_clones_s{42,43,44}` against `rl_new_clones_percapita_s{42,43,44}` on the same axes, with the seed spread reported and never averaged away. Three points estimate a spread poorly; the pilot priced a run at ~6 A100-hours, so five seeds per arm would still fit one wave if the spread turns out to matter.
2. **Settle the scale confound before interpreting any difference** (inferred item 3). The clean control is a third arm with a *constant* divisor — `pool / 4` — which has the pool's shape and the per-capita's scale. If `pool/4` behaves like `common_pool`, the difference this arm finds is incentive; if it behaves like the per-capita arm, it is scale. That is one more `reward_mode` and one more `make_configs.py` entry.
3. **Report `members held` and `contribution per member` as first-class rows in both arms**, not as diagnostics. They are the quantities the whole question turns on and the running arm's table does not yet have them.
4. `notes/autoresearch_log/rl-manager-two-worlds.md` remains the parent log; `notes/autoresearch_log/rule-based-manager-sweep.md` holds the rule-based comparison points.

## Notes

1. **Why the base is pinned to `0ff44a9` and not rebased.** The running seeds launched from that tree. Any newer commit would put a second difference between the arms, and the arm's entire value is that there is exactly one. The cost is that this branch does not carry whatever landed on `main` afterwards; that is the right trade and it is why the PR should be read as a control, not as current `main`.

2. **`share_pool_per_group` was reused rather than re-derived.** The per-capita branch in `compute_reward_per_group` returns the very tensor the `common_good` state field is built from, three lines above where the `sum` / `avg` modes gather it. There is one divisor in the file and the guard cross-checks against the other consumer of it.

3. **The guard script now dispatches on `reward_mode`.** Guard 1 asks "is the reward what the config says it is", which is a strictly larger question than "is the reward the common pool", and guard 3 only runs for the per-capita mode. The old evidence's key names are preserved for the pool mode, so the running arm's JSON and a fresh one are directly comparable — which is what made the bit-for-bit reproduction in Measured §2 checkable at all.

4. **Guard 3c's rollout is a probe, not a second configuration.** It drops the switch model (so membership cannot move) and forces every player valid, purely to manufacture an equal headcount. Nothing is trained under it and neither knob is reachable from a config.

5. **The isolated remote dir is `~/repros/ai-runs/rl-percapita`**, created fresh for this arm. `train_cluster.sh` adds `--exclude='artifacts/manager/'` for isolated dirs, so a later `--sync-only` will not delete the trained checkpoints. Delete the dir when this PR closes.

6. **The guard ran twice on the per-capita config and the two JSONs are byte-identical.** The first run is what cleared the launch; its measured numbers were then written into `GUARD_EVIDENCE` in the generator and the configs regenerated, which changes only comment lines. The second run was made against that final, committed config, so the committed evidence provably comes from the committed file, and `diff` reports no difference between the two outputs. The three seeds were submitted between the two runs, on the strength of the first — the config's parsed keys were identical throughout, so the re-run confirmed the evidence rather than discovering it.
