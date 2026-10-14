# What the RL manager is rewarded for: the common pool (a reward-specification fix)

## 1. Declaration

**Slot:** shared RL-manager code -- a specification fix plus a correctness fix under §5's "a bug fix in shared code is legal but is its own experiment". **No model is trained and no manager run is started.** No artificial-human weights, features or training configs change.

**Parent:** `auto/sim-timeout-imputation` at `3fe1f44` (PR #196, `[FAIL]` on the gates with a confirmed correctness gain). This branch is created from it and its PR opens with `--base auto/sim-timeout-imputation`. Isolated remote dir `~/repros/ai-runs/manager-reward` (delete when this PR closes).

**Source.** `notes/reviews/rl-manager-review.md` on branch `review/rl-manager`, a read-only review of exactly this path at exactly this commit. This branch implements its S1 (the reward is not the manager's incentive), its D2 (the reward throws away a timed-out player's payoff) and the dead-code half of its D6. It deliberately does **not** touch D1, D3, D4 or D5 -- those are other branches' changes and folding them in here would confound the reward comparison.

**The specification.** `reports/basics.md`: "The contributors get a payout proportionally to their private accounts. The manager is receiving a payout proportionally to the common pool." The common pool is 1.6 times the group's contributions minus the punishments the manager dealt. The environment offered only `sum` and `avg`, which reward the sum or the mean of the *contributors'* payoffs -- a different objective, and the one every RL manager run so far has optimised.

### The change

1. **`reward_mode: common_pool`**, selectable alongside `sum` and `avg`. Neither of the existing modes is removed: earlier results were produced under them and stay comparable.
2. **The reward is computed where the manager's action resolves** -- in `punish()`, by `compute_reward_per_group`, a pure function of that round's contributions, punishments and validity flags.
3. **Review finding D2 fixed in the same place**, because it changes the reward: a player who timed out is paid, as the real game paid them.
4. **No shaping.** The reward for acting at round s is round s's pool. Punishment costs in the round it is given and pays back later through raised contributions.

### Artifact naming contract

| what | path |
|---|---|
| env | `src/aimanager/manager/environment.py` (`REWARD_MODES`, `compute_common_pool_per_group`, `share_pool_per_group`, `compute_reward_per_group`) |
| config | `configs/training/rl_manager/04_2g8a_common_pool.yml` |
| tests | `src/aimanager/tests/test_manager_reward.py` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Verify the review's arithmetic myself against `experiments/2group_8agent_50ep.csv` before relying on any of it. | **done** -- section 3, step 1 |
| 2 | Add the `common_pool` mode; move the reward computation into `punish()`; fix D2; check whether the common-good computation has the same defect. | **done** -- section 3, step 2 |
| 3 | Unit tests: the new mode on hand-built cases (timeout, empty group, reshuffle) and a test pinning the reward to the acting round. | **done** -- 11 tests, section 3, step 4 |
| 4 | Prove `sum` and `avg` unchanged except where D2 corrects them, and quantify the correction. | **done** -- section 3, step 3 |
| 5 | Establish whether any trace of the historical round offset remains in code or configs. | **done** -- section 3, step 5 |
| 6 | Remove the dead code the review names, in its own commit, after verifying it is unreachable on every branch. | **done** -- section 3, step 6 |

## 3. Results

Nothing was trained, simulated or evaluated, so there is no scores table and no gate verdict. What follows is measured on the human data and on the environment itself.

### Step 1: the arithmetic, verified on the human data (measured)

`experiments/2group_8agent_50ep.csv`, 19,200 agent-rounds = 100 flip-doubled episodes x 24 rounds x 8 players, grouped by `(episode_id, round_number, group_id)` into 4,512 non-empty group-rounds (288 of the nominal 4,800 are empty because all eight players merged). The probes were scratch scripts and are not committed; every number below is reproducible from the CSV with pandas alone, and the group-by key is the only choice they make.

| identity | max residual | n |
|---|---|---|
| `common_good == 1.6 * sum(c) - sum(p)` | 2.842e-14 | 4,512 group-rounds |
| `payoff == 20 - c - p + common_good / n_valid` | 7.105e-15 | 19,166 agent-rounds with n_valid > 0 |
| same, on timed-out rows only | 7.105e-15 | 526 |
| `sum_valid_payoff == 20 * n_valid + 0.6 * sum(c) - 2 * sum(p)` | 5.684e-14 | 4,512 |

So the review's closed form for `reward_mode: sum` is exact, and the `common_good` column is the pool, not a per-capita share. The env's `common_good` *state field* is the share (pool / n_valid); `parse_agent_rounds` divides the same way, so training and env agree -- that is a naming collision, not a discrepancy.

What the two objectives price (measured over the 4,512 group-rounds):

| | 1 punishment point costs, in contribution points | variance share, contribution | variance share, punishment | variance share, headcount |
|---|---|---|---|---|
| `sum` (env reward) | 2 / 0.6 = **3.333** | 23.24% | 13.74% | **63.01%** |
| `common_pool` | 1 / 1.6 = **0.625** | 95.15% | 4.85% | -- |

Correlation between the two across group-rounds: **0.8142**. Of the headcount term's variance, `Var(20 * n)` is 100.55% of `Var(20 * n_valid)` -- i.e. essentially all of it is group size, not timeouts. All four of the review's S1 numbers reproduce exactly.

### Step 2: the change, and where it lives (measured)

`punish()` now ends by settling the round:

```python
self.reward = self.compute_reward_per_group(
    self.contribution, self.punishment, self.contribution_valid
)
```

and `step()` no longer calls anything that touches the reward. `compute_reward_per_group` recomputes the pool, the per-capita share and the payoffs from its three arguments rather than reading `self.common_good` / `self.group_payoff*`, so the reward is independent of any field another method may have written. Before this, `punish()` computed round s's payoff, `step()` advanced the round number and drew round s+1's contributions, and *then* `update_reward()` read the leftover fields; the result was round s's outcome, but only because nothing in between rewrote them.

**Does the common-good computation have the D2 defect? No.** `compute_common_good_per_group` already zeroed the invalid cells' contribution and punishment before summing, which is exactly what makes the pool identity hold on the human data (all 560 timed-out rows carry `contribution == 0` and `punishment == 0`), and it already divided by the valid headcount, which is what the game did. It is refactored here into `compute_common_pool_per_group` + `share_pool_per_group` and is numerically unchanged.

**What D2 was.** `compute_payoff_per_group` zeroed the *payoff* of an invalid player and dropped them from the group total. The fix zeroes their *inputs* instead and keeps them in the total. The human data settles it: 526 timed-out rows with n_valid > 0 satisfy `payoff = 20 - 0 - 0 + pool/n_valid` to 7.1e-15, their mean payoff is 33.94, the minimum is -10, none is 0; and the 34 group-rounds where every member timed out pay exactly 20.0 per player with `common_good` 0, where the env paid 0. Once every member has a payoff, `avg` divides by the group's membership rather than by the valid headcount -- dividing a full sum by `n_valid` would be a mean of nothing.

### Step 3: what the payoff fix corrects, and by how much (measured)

The corrected `sum` reproduces the human **total** group payoff (all players) to 5.684e-14 over all 4,512 group-rounds; the pre-fix `sum` reproduces the **valid-only** total to the same precision. Both are therefore exactly characterised, and the difference between them is exactly the timed-out players' own payoffs.

516 of 4,512 group-rounds (11.44%) contain at least one timeout.

| mode | scope | pre-fix | corrected | delta |
|---|---|---|---|---|
| `sum` | all 4,512 group-rounds | 91.4675 | 95.6796 | +4.2121 (+4.61%) |
| `sum` | the 516 with a timeout | 83.0023 | 119.8336 | +36.8313 (+44.37%) |
| `avg` | all 4,512 group-rounds | 21.9231 | 22.4220 | +0.4990 (+2.28%) |
| `avg` | the 516 with a timeout | 20.5073 | 24.8705 | +4.3632 (+21.28%) |

The 30.7% figure in the review is the same fact read the other way round: the discarded payoffs are 30.74% of the *true* total on those group-rounds.

The `avg` correction has two parts, and they pull in opposite directions: paying the timed-out player takes the all-group-round mean from 21.9231 to 23.2984, and then dividing by the membership instead of the valid headcount brings it back to 22.4220.

On the 3,996 group-rounds where nobody timed out, both modes are unchanged -- bit-identical, not approximately.

**Direct old-vs-new comparison of the environment.** The parent's `environment.py` at `3fe1f44` was loaded side by side with the new one and driven by identical fixed draws over 12-round episodes, under both `avg` and `sum`, at timeout rates 0%, 2.9% (the human rate), 25% and 100%, with a reshuffling switch model and with one group empty. Result:

* the four fields the simulation records -- `punishment`, `common_good`, `contribution`, `agent_group` -- are **identical in every scenario**, including 100% timeouts and the empty group;
* at a 0% timeout rate, `reward`, `group_payoff`, `group_payoff_sum` and `contributor_payoff` are identical too;
* everywhere else they move by exactly the timed-out players' payoffs.

`per_round.parquet` is built by `simulate.mem_to_df` from those four recorded fields alone (`payoff` is derived downstream in pandas from `20 - c - p + common_good`), so **no simulation output and no evaluation score can move**. That is the byte-level argument; it did not need a cluster rerun, because the quantities that changed are not written to disk by the simulation at all. `contributor_payoff` / `group_payoff*` appear only in the RL training diagnostics (`rl_manager.rec_keys`), never in any model's `x_encoding` and never in `linear_ah`'s feature pool, which rebuilds `payoff` features from contribution / punishment / common_good.

### Step 4: unit tests

`src/aimanager/tests/test_manager_reward.py`, 11 tests, in the style of `test_sim_timeout_serving.py` (plain torch, no PyG, so they also run locally):

| test | what it pins |
|---|---|
| `test_common_pool_reward_equals_the_group_pool` | both groups, plain round: reward == 1.6*sum(c) - sum(p), and != the payoff sum |
| `test_common_pool_reward_with_a_timed_out_player` | neither the imputed contribution nor the punishment aimed at the timed-out player enters the pool; the raw state still carries both |
| `test_common_pool_reward_of_an_empty_group_is_zero` | empty group -> 0, no division by zero |
| `test_common_pool_reward_follows_a_reshuffle` | round 4's pools use the post-arrival membership, not the pre-reshuffle one |
| `test_reward_is_the_acting_rounds_outcome` | the reward out of round s is round s's pool, and is *not* round s+1's -- the test that fails if anyone shifts the reward by a round |
| `test_reward_is_settled_by_punish_not_by_step` | `punish()` sets it; `step()` only hands it on unchanged |
| `test_sum_mode_is_unchanged_when_everyone_plays` | `sum` == 20*n_valid + 0.6*sum(c) - 2*sum(p) |
| `test_sum_mode_pays_the_timed_out_player` | D2: the reward is the pre-fix value plus exactly `20 + pool/n_valid` |
| `test_avg_mode_divides_by_the_group_membership` | the divisor, stated against the alternative |
| `test_all_players_timed_out_pays_the_endowment` | the 34 human all-timeout group-rounds' 20.0, and 0 for the pool |
| `test_unknown_reward_mode_is_rejected` | the validation still rejects a typo |

**Full suite on Raven, `AI_REMOTE_DIR=~/repros/ai-runs/manager-reward`, `remote_test.sh --test-only`: 155 passed, 0 failed.** The first cluster run showed 6 failures and 4 errors, all `FileNotFoundError` on `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet` and `artifacts/baselines/punishment_multinomial_best_with_contr.joblib` -- fixtures the sync script excludes by design (`--exclude='plots/'`, `--exclude='artifacts/'`). They were copied over by hand and the suite went green; that was a missing fixture in a fresh isolated dir, not a regression. Locally, 131 pass and one pre-existing `torch_scatter` import failure remains (`test_punisher_current_contribution.py::test_simulation_round_carries_the_env_validity_flag`), unchanged from the parent.

### Step 5: the historical round offset (measured)

The project memory is real and the history is documented; **nothing live remains in the reward path**.

* **Live code: clean.** No `reward_formula`, `reward_shift`, `impact_on_group_payoff` or `group_payoff_round` branch exists in `src/aimanager/manager/environment.py` or anywhere else under `src/` that the reward reaches.
* **The history.** The offset entered as the `group_payoff` reward formula, which combined `self.contribution` (already advanced to round t+1 by `step()`'s `update_contribution()`) with `self.prev_punishment` (round t's action) -- visible in `d4cd680`, "Update rewards in environment to be computed per group". It was issue **#40**, investigated in `doc/literature/reward-timing-rl.md` and planned in `doc/plans/reward-timing-alignment.md`, and fixed on branch `41-reward-timing` by `1279456` ("Fix group_payoff to use same-round data instead of next-round contributions", Refs #41), merged as **PR #56**. Issue **#42** then removed the lookahead formula entirely and dropped the `reward_formula` parameter, merged as **PR #57**. A later pass produced today's `reward_mode` switch.
* **Traces that remain, all inert, none kept by this branch and none removed by it:**
  1. `src/aimanager/manager/archive/environment.py:160-163` still contains the literal offset, `self.contributions * 1.6 - self.prev_punishments`. The `archive/` package is imported nowhere and is already broken independently (`archive/evaluation.py` imports `aimanager.manager.archive.memory`, which does not exist). **Left in place deliberately**: it is an explicitly named archive directory, the review did not list it, and deleting a whole archived package is exactly the kind of ambiguity this branch was told to leave alone. Flagged, not touched.
  2. `run/manager/*.yml` -- 7 of 23 legacy DJX files still carry a `reward_formula:` key (`group_payoff`, `group_payoff_round`, `impact_on_group_payoff`, `payoff`, `true_common_good`). `rl_manager.py:250` pops and warns on the key before the env is built, so none of them can produce an offset reward; `doc/plans/archive/remove-group-payoff-formula.md` explicitly decided to leave `run/` untouched as historical record. **Left in place**, per that decision.
  3. `reports/manager.md:28-33` and `reports/draft_publishable.md:262-275` still assert the offset as current behaviour: "A reward is calculated, that is composed of the punishment and next rounds contribution. `R_i = C_{i+1} * 1.6 - P_i`". This is the project memory, written down and never corrected after PR #56. It is documentation drift with no functional effect. **Left in place** -- both files are under a separate, unfinished review (`doc/plans/archive/repo-cleanup.md`, issue #52) and rewriting reports is not this PR's business -- but it is the single most likely source of the belief that the offset is still live, and a maintainer should correct or delete them.

### Step 6: the dead code (measured)

Review finding D6, removed in its own commit so the reward diff stays readable. The audit method: `git grep` for each symbol across **all 162 refs `git branch -a` lists** -- every `auto/*` branch including the three active siblings (`rl-manager-timeout-view`, `rl-manager-two-worlds`, `rule-based-manager-sweep`), `main`, `review/rl-manager`, `results/september-measurements`, every `worktree-agent-*` and every `origin/*` -- run both unrestricted and restricted to `*.py`, with `git show <ref>:<path>` on every suspicious hit.

| removed | what it was | what the audit found |
|---|---|---|
| `ArtificalManager.get_punishment` | calls `policy_model.encode_pure`, which `GraphNetwork` does not define -> `AttributeError` | on every ref with the current layout the only `.py` match is the definition itself; real call sites exist only in `notebooks/archive/manager_evaluation/*.ipynb` on ~16 old branches, most already merged into `main`, where `notebooks/` no longer exists |
| `ArtificalManager.encode_pure` | same broken call | `def encode_pure` exists on exactly one ref, `origin/aggregate_reward` (2022-08-14, pre-`src/` layout, not an ancestor of `main`) -- which is where these two methods last worked, so this is a refactor regression whose caller was never cleaned up |
| `src/aimanager/manager/artificial_human_group.py` | `_get_better_avg_movement_map` builds a dict and returns `None` and reads `group.average_contributor_payoff`, not a state key; `do_group_selection` is `pass` | zero hits of any kind on any ref, in any file type, outside its own definitions and three prose mentions (`CLAUDE.md`'s tree, `doc/plans/archive/group-switching-predictor.md` calling it a stub to be replaced, and the review). Group switching went a different way, inside `ArtificialHumanEnv` |
| `environment.create_fully_connected` | unused; `update_groups` builds `batch_edge_index` inline | the one genuine import is `src/aimanager/generic/archive/graph_encode.py`, which does not exist on `main` or anything descending from it. Not to be confused with `GraphNetwork.create_fully_connected` or `train.py`'s own function, both alive and unrelated |

`CLAUDE.md`'s project tree lost the line for the deleted module; nothing else referenced it.

**Left alone, and why.** Three things are unreferenced or inert but ambiguous, so the "if anything is ambiguous, leave it and say so" rule applies:

1. `src/aimanager/manager/archive/` -- inert (see step 5) but an explicitly named archive, and not on the review's list.
2. `environment.compute_average_per_group` -- also unreferenced anywhere in the repo, but a general-purpose helper rather than a broken one, and not on the review's list.
3. `run/manager/*.yml`'s `reward_formula` keys -- inert, and kept deliberately by a `[DONE]` plan.

## 4. Notes

1. **The reward is a specification call, so it was made explicitly and both old modes were kept.** `sum` is not a bug: PR #69 chose it so the manager would be rewarded for retaining and attracting members, and the config comment says so. What it is not is the human manager's incentive, and the arithmetic in step 1 says the gap is large -- a factor of 5.3 in the price of punishment and 63% of the reward's variance spent on a quantity the manager barely controls. Every run made under `sum` stays reproducible.

2. **No shaping, on purpose.** The straightforward version was asked for and is what is implemented: reward(s) = pool(s). The manager therefore pays for punishment in the round it punishes and collects the benefit only if contributions rise later, discounted by gamma. That is the real game's incentive and it is also the version whose failure mode (a manager that never punishes) is legible rather than hidden inside a shaping term.

3. **The `avg` divisor changed, and that is a change of meaning, not only of value.** Pre-fix `avg` was the mean payoff *among the players who gave input*; it is now the mean payoff *among the group's members*. There is no third option that is a mean of anything once timed-out players are paid, and the second reading is the one that matches `sum / n_members`. Anyone comparing to a pre-fix `avg` run on timeout-heavy episodes should use the +2.28% / +21.28% numbers in step 3 rather than assuming continuity.

4. **The same defect exists one layer downstream, in the simulation's payoff column, and was deliberately not touched.** `simulate.mem_to_df` computes `payoff = 20 - contribution - punishment + common_good` from the *recorded* state, where a timed-out player's contribution is the imputed 9 and their punishment is whatever the manager assigned. So `per_round.parquet` carries a wrong payoff for ~2.2% of agent-rounds in the same way the env's reward did. Fixing it would change every evaluation score that reads `payoff`, which is a re-baselining exercise and not a reward change. **Escalated, not fixed.** It is adjacent to the successor note the parent already left about `evaluation_suite/convert.py`.

5. **The reward's round is now pinned by a test that must fail if it moves.** `test_reward_is_the_acting_rounds_outcome` asserts both that the reward equals round s's pool and that it differs from round s+1's, with a different contribution vector in every round so the two can never coincide by accident. That is the guard the "correct by construction" requirement actually needs; an assertion that `step()` must follow `punish()` was written and then removed, because `test_joint_exodus_train_sim_parity.py` legitimately steps the env without punishing to exercise the switch machinery, and breaking that to guard a case no production path can reach would have been a bad trade.

6. **D1 is not fixed here and it interacts with this change.** The review's D1 (a punishment aimed at a timed-out player costs the manager nothing, yet the artificial humans are shown it) is being fixed on `auto/rl-manager-timeout-view`. Under `common_pool` that free punishment is still free -- the pool zeroes it, exactly as `sum` did -- so this branch neither creates nor closes the incentive. The two fixes are independent and compose.

## 5. Verdict

Not a gated experiment: no training, no simulation, no evaluation, no band. What can be claimed is what was measured:

* the manager can now be rewarded for what the real manager was paid on, and the three modes are explicit and selectable;
* the reward is computed where the action resolves, and the round is pinned by a test;
* a timed-out player is paid, as the game paid them, which corrects `sum` by +4.61% overall and +44.37% on group-rounds containing a timeout, and `avg` by +2.28% and +21.28%;
* nothing the simulation records changes, so no existing evaluation score moves.

## 6. For a successor

1. **The first `common_pool` run should not also carry D1, D3 and D5.** `04_2g8a_common_pool.yml` is config D with the reward swapped and nothing else, precisely so the comparison is clean. It therefore inherits the pre-fix lagged opponent punisher (D3) and the raw-state RL observations (D5). Run it against `03_2g8a_sum_d_lr1e3_freq500.yml` first, then move the corrected opponent and the served state in as separate steps -- otherwise the reward change and the opponent change are inseparable, which is the mistake this branch was written to avoid.

2. **Watch the punishment distribution, not just the return.** Under `common_pool` a punishment point costs 0.625 contribution points instead of 3.333, so the expected direction is *more* punishment than `sum` produced. The review's S2 is the constraint: only 4.49% of the contribution model's training rows follow a punishment above 10 and 1.50% above 20, so a policy that settles above p ~ 10 is extrapolating, not succeeding. Log the realised punishment histogram against the human one (`P(p = 0) = 0.694`, mean 1.79, `P(p > 15) = 0.026`) from the first run.

3. **The reward scale changed, and the tuning may not carry over.** `sum` returns are on the order of 2,000 over an episode and are dominated by a near-constant `20 * n_valid` offset (review S5); `common_pool` has mean 55.21 and sd 45.84 per group-round, with 95% of its variance in the part the manager influences. Both `lr` and `target_update_freq` were tuned against the old scale. If the first `common_pool` run is unstable, that is the first thing to look at, not the reward.

4. **`sum` is still the right control, and it is now a *different* `sum` on timeout-heavy episodes.** Any comparison against a `sum` run made before this branch is a comparison against a reward that was 4.61% low on average and 44.37% low whenever somebody timed out. Re-run the `sum` control on this branch rather than quoting an older number.

5. **The simulation's `payoff` column still has the D2 defect** (note 4). Whoever re-baselines the evaluation should fix it in the same pass as the `contribution_valid` column the parent's successor note asks for -- they are the same fix, in the same file, and the parquet has to be rewritten either way.

6. **`reports/manager.md` and `reports/draft_publishable.md` still document the round offset as current behaviour** (step 5). They are the reason the memory persists. A one-line correction in each would retire it; this branch left them alone on purpose.
