# RL manager review — before the first training run against the corrected simulated players

Read-only review of `src/aimanager/rl_manager.py`, `src/aimanager/manager/{manager,environment,artificial_human_group,api_manager,memory}.py` and `configs/training/rl_manager/`, on `origin/auto/sim-timeout-imputation` at `3fe1f449f2a97ad77899affcaf8eeeec0722398d`. Nothing was fixed; this file is the only change on branch `review/rl-manager`.

The question carried through: **does each piece of the manager's training loop see what a real manager would have seen, and is it rewarded for what we actually want?** Everything below that can be settled against `experiments/2group_8agent_50ep.csv` (19,200 agent-rounds, 100 flip-doubled episodes, 24 rounds each) or by running the env locally was settled that way rather than argued.

Counts: **6 confirmed defects, 6 suspicions / specification calls, 16 things checked and found correct.** Two confirmed defects (D1, D2) change what a trained manager learns; one (D3) changes what it learns to compete against.

---

## 1. Confirmed defects

### D1 — A punishment aimed at a player who gave no input costs the manager nothing, yet the artificial humans are shown it

**What the code does.** `ArtificialHumanEnv.punish` stores the manager's action verbatim (`self.punishment = punishment`, environment.py:393) for all eight agents, including the ones the validity model has just marked as timed out. The accounting then discards it: `compute_common_good_per_group` zeroes punishment at `~contribution_valid` (environment.py:201) and `compute_payoff_per_group` zeroes the whole invalid contributor's payoff (environment.py:234). But `step()` copies the raw value into `prev_punishment`, and `served_state()` corrects only `contribution` and `prev_contribution` — it never touches `punishment` or `prev_punishment`.

**What it should do, and why.** In the human data a player who gave no input was never punished: all 560 `player_no_input == 1` rows carry `punishment == 0.0` exactly, and the group identity `common_good = 1.6·Σc − Σp` holds to a maximum residual of 2.8e-14 across all 4,512 group-rounds with that zero in place, so nothing was deducted for them either. A manager in the real game could not spend punishment on a timed-out player, and no artificial human was ever trained on a row where one had been.

**How I verified it.** A local probe (`ArtificialHumanEnv` with mock contribution/validity models, no PyG needed) ran one round with agent 0 timed out, once with the manager playing 0 on that agent and once with 30:

| manager's action on the timed-out agent | group common good | group_payoff_sum | `prev_punishment` next round, raw | served |
|---|---|---|---|---|
| 0 | 16.000 | 78.000 | 0 | 0 |
| 30 | 16.000 | 78.000 | 30 | 30 |

Identical reward, different input to every downstream model.

**Why it matters.** `contribution_valid` is in the manager's own `x_encoding` (both configs), so the network can condition on exactly the cells where the action is free. There is no downward pressure on the argmax there, and the action is not inert: the contribution AH's only channel from the manager is `prev_punishment`, and the switch AH reads the current round's `punishment`. This is free deterrence — a behavioural lever the real game did not offer — available on the 2.2–2.9% of agent-rounds that time out. It is shared with the simulation path: the artificial punisher assigns a punishment to timed-out players there too, and the recorded `per_round.parquet` carries it.

**Shape of the fix.** Zero the punishment at `~contribution_valid` where the action is realised (in `punish`, so the recorded value matches the game) — the same rule the common-good and payoff accounting already applies, applied one step earlier. Confidence: high.

### D2 — The reward throws away a timed-out player's payoff; the real game paid it

**What the code does.** `compute_payoff_per_group` sets `contributor_payoff` to 0 wherever `contribution_valid` is False and drops that agent from both `sum_payoff_per_group` and the `valid_per_group` divisor. `update_reward` hands `group_payoff_sum` straight to the agent as the reward (`reward_mode: sum` in both configs).

**What it should do, and why.** The timed-out player was paid. Across all 560 timed-out human rows, `payoff = 20 − contribution − punishment + common_good/n_valid` holds **exactly** (max residual 0.0000 wherever `n_valid > 0`), their mean payoff is 33.94, the minimum is −10 and not one of them is zero. The env scores them 0.

**How I verified it.** Same probe, plus the human data. A group of four contributing 10 each: the env's `group_payoff_sum` is 104.00 with everyone valid and 78.00 when one of the four times out — a drop of 26.00, while the real game would have paid that player `20 − 0 − 0 + 16 = 36.00`. On the human data, over the 516 group-rounds that contain a timeout, the true total group payoff averages 119.83 against the env's valid-only 83.00: a **30.7% shortfall**, exactly the timed-out players' own payoffs.

**Why it matters.** Under `reward_mode: sum` each simulated timeout removes roughly 34 points from a per-round reward whose human-data mean is 91.5. And the manager cannot do anything about it: the validity model in both configs (`raven_script_22`) reads only `prev_contribution_valid` — measured, not assumed, by the probe in `notes/autoresearch_log/sim-timeout-imputation.md` step 1 — so timeouts are exogenous to the policy. It is a large, unlearnable shock that grows in relative size as the manager's group shrinks, which is precisely the region `reward_mode: sum` is meant to teach the manager to avoid.

**Shape of the fix.** Pay the invalid contributor `20 − 0 − 0 + common_good` and include them in the group total; leave the common-good divisor alone (see V4 — that one is right). Confidence: high.

### D3 — The opponent is the pre-fix, lagged punisher

**What the code does.** `configs/training/rl_manager/03_2g8a_sum_d_lr1e3_freq500.yml` sets `opponent_manager: artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`. That artifact's training config (`configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled.yml`) has `x_encoding = prev_contribution, prev_punishment, is_first` — it is the exact artifact the current-contribution branch was created to replace.

**What it should do, and why.** Two corrected GNN punishers are already on this branch and are drop-in path swaps: `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr/` (adds the current round's `contribution`) and `.../rnn_edge_50ep_doubled_timeout/` (adds `contribution_max` and `contribution_valid` on top). The lagged one punishes round t on round t−1's contribution — the defect the whole re-baseline was about.

**How I verified it.** Config path against the two training configs and `ls artifacts/artificial_humans/punishment/`, plus the mechanism table in `notes/autoresearch_log/punisher-current-contribution.md`: the lagged GNN's OLS of expected punishment on (c_t, c_{t−1}) is +0.019 / −0.131 against the human −0.242 / +0.067, and it punishes full contributors at P(p>0 | c=20) = 0.158 against the human 0.038 (which I reconfirmed on the raw data as 0.0381, n = 2,464).

**Why it matters.** The opponent is the only thing the RL manager competes with, and `reward_mode: sum` makes "keep and attract members" the dominant reward channel (see S1). A manager that learns to out-compete a punisher which reacts a round late, and which punishes full contributors four times too often, has learned about a straw man. Confidence: high.

**Adjacent, and worth a decision before the run.** `rl_manager.py` loads the opponent through `AH_MODELS[config["artificial_humans_model"]]`, and `AH_MODELS` is `{"graph": GraphNetwork}` — `.pt` only. The reference punisher in `notes/autoresearch.md` §3 is `artifacts/baselines/punishment_multinomial_current_contr.joblib`, a linear bundle, and the frontier stack's is `punishment_multinomial_timeout_severity_copula.joblib`. Neither can be used as the RL opponent at all; the simulation path has a `LinearAHAdapter` for exactly this and the RL path does not.

### D4 — `03_2g8a_sum.yml` cannot run, and its switch model carries the pre-#123 anchoring

Two of its four artifacts are absent from the tree: `group_switching_contribution_50ep/model/...__epochs_1000.pt` and `punishment_autoregressive_50ep/...` (the config's own comment says the latter was removed). It will fail at load.

More interesting than the missing files: its `switch_model` is `switch_pred_opt_50ep`, whose config (`switch_predictor/opt_50ep.yml`) uses `prev_common_good` / `prev_punishment` and `mask_name: switch_mask` — the arrival-round anchoring from before #123. `environment.step()` now runs the re-anchored convention, calling the predictor at the end of round s on round-s post-punish features. Feeding a pre-#123 model there hands it round s−1 values where it expects round s: a one-round lag of exactly the shape this review was commissioned to hunt. So the config must not simply be repointed at the missing artifacts. `03_2g8a_sum_d_lr1e3_freq500.yml` uses `switch_pred_opt_50ep_doubled_reanchored` and is correct on this axis. Confidence: high on the anchoring difference (config diff plus `git log -S`), medium on the size of the effect since the config cannot run anyway.

### D5 — The RL path never calls `served_state()` (the sibling's defect) — and two more call sites need the same treatment

The defect itself is being fixed on `auto/rl-manager-timeout-view` and `notes/autoresearch_log/sim-timeout-imputation.md` §step 2 already names it ("`rl_manager`'s observations. Unchanged; `served_state()` is opt-in and the RL call sites were not touched"). I record only what is adjacent to it, because fixing `get_action` alone will not close it:

- `rl_manager.py:80` — the **opponent punisher** is handed the raw `state` too, so it reads the imputed 9 at timed-out cells. It is the one model in the loop whose corrected artifact (`rnn_edge_50ep_doubled_timeout`) has a `contribution_valid` feature built specifically for this population.
- `rl_manager.py:66` — `statecopy` clones the **raw** state into the replay buffer, and `replay_keys` includes `contribution`. So even after `get_action` is switched to the served view, `manager.update` would keep computing the TD error on `contribution = 9` at timed-out cells. The online view and the replayed view would then disagree, which is worse than either.
- `prev_punishment` is not part of the served view at all — that is D1.

Confidence: high (code read plus the local probe, part D: `served_state()` returns `contribution = 0` and `prev_contribution = 0` at a timed-out cell while `prev_punishment` still reads the manager's 7).

### D6 — Dead and broken code inside the reviewed surface

- `ArtificalManager.get_punishment` (manager.py:92) and `ArtificalManager.encode_pure` (manager.py:65) both call `self.policy_model.encode_pure(...)`, which `GraphNetwork` does not define — `grep -rn encode_pure --include=*.py` returns only these two lines. Either would raise `AttributeError`. Nothing calls them (`api_manager.RLManager` goes to `policy_model.predict` directly), so this is latent, not live.
- `src/aimanager/manager/artificial_human_group.py` is entirely dead: `_get_better_avg_movement_map` builds a dict and returns `None`, and reads `group.average_contributor_payoff`, a key the env's `__getattr__` would raise `KeyError` on; `do_group_selection` is `pass`; `build_group` asserts `batch_size == 1`. The only reference anywhere is a line in `doc/plans/archive/group-switching-predictor.md` describing it as a stub to be replaced.
- `environment.create_fully_connected` (module level, environment.py:6) is unused; the real one is `update_groups`'s inline construction.

Confidence: high. Impact on a trained manager: none — listed because it is in scope and it makes the module harder to read than it is.

---

## 2. Suspicions and specification calls

### S1 — The reward is neither the common good nor the human manager's incentive

`update_reward` returns `group_payoff_sum`, the sum of the valid contributors' payoffs. That quantity has a closed form which I verified against every human group-round (max residual 8.5e-14, n = 4,512):

> `group_payoff_sum = 20·n_valid + 0.6·Σc − 2·Σp`

The common pool — which `reports/basics.md` says the human manager was actually paid on ("The manager is receiving a payout proportionally to the common pool") and which `CLAUDE.md` states as the project's objective — is `1.6·Σc − Σp`. The two correlate at 0.81 across human group-rounds but they are not the same objective:

| | marginal rate of substitution (contribution units bought per unit of punishment spent) | variance share, contribution term | variance share, punishment term | variance share, headcount term |
|---|---|---|---|---|
| env reward (`group_payoff_sum`) | 2 / 0.6 = **3.33** | 23.2% | 13.7% | **63.0%** |
| common pool | 1 / 1.6 = **0.62** | 95.1% | 4.9% | — |

So the manager is being trained to be roughly five times more reluctant to punish than the stated objective implies, and under `reward_mode: sum` the single largest signal is headcount — 63% of the reward's variance across human group-rounds comes from `20·n_valid`, of which 98.7% is group size rather than timeouts. That is the documented intent of PR #69 and the `reward_formula` deprecation warning, and the config comment says so; it is a specification call, not a coding error. But it contradicts `CLAUDE.md`, `reports/basics.md` §Optimal Manager and the human manager's own incentive, and it should be a conscious choice before a 15,000-step run, not a discovery afterwards. Confidence: high on the arithmetic, this is a question not a bug.

### S2 — The action space is right; the evidence behind its upper half is thin

The range is exactly the human interface: `n_punishments: 31` (0–30), matching `reports/basics.md`, and all 31 levels occur in the data. Nothing to fix there. The support is the issue:

| statistic (18,386 rows with a valid manager input) | value |
|---|---|
| P(p = 0) | 0.6942 |
| P(p ≤ 5) / P(p ≤ 6) | 0.9004 / 0.9148 |
| P(p > 15) | 0.0258 (n = 474) |
| levels with fewer than 30 rows in the flip-doubled data | 18, 19, 22, 23, 24, 26, 27, 28, 29 (i.e. fewer than 15 independent observations each) |
| mean punishment | 1.79 |
| group-round total punishment: median / p90 / p99 / max | 2 / 20 / 65 / 114 |
| P(p > 0 \| c = 20) | 0.0381 (n = 2,464) |

The channel that matters is the contributor AH's `prev_punishment` (numeric, encoded as p/30, its *only* input from the manager): only **4.49%** of its training rows follow a punishment above 10 and **1.50%** follow one above 20. A greedy policy that settles above 10 is optimising against under a twentieth of the model's evidence.

Exploration itself is close to human scale and is not the main worry: `eps = 0.1` uniform over 31 levels puts 4.8% of agent-rounds above p = 15, against the human 2.6%. `eps` does not decay, which is fine for a fixed-eps DQN. The recommendation is diagnostic rather than structural — log the realised punishment distribution against the human one during training and treat a policy that lives above p ≈ 10 as extrapolation, not as a result. Confidence: medium-high.

### S3 — Training and deployment disagree about what the manager sees of the other group

In training, `ArtificalManager.expand_obs_for_groups` replicates the full eight-agent state per group and adds an `in_group` bool — the manager sees the other group's **true** contributions and validity flags, flagged as out-of-group. That is the documented design (`notes/groub_competition.md`: "We add to the input information if a group member is in the group, or the other group").

At deployment through `MultiManager`, `api_manager.create_data` does something different: every other-group cell is overwritten with the model's own `default_values["contribution"]` (9) and `contribution_valid` is set to `False` for every out-of-group agent regardless of whether they actually gave input. The checkpoint is therefore evaluated on inputs it was never trained on, through a GNN whose edges span both groups. The same file also already serves `MISSING_CONTRIBUTION = 0` at own-group timeouts — the mirror image of D5, so once the sibling's fix lands the two sides will agree on that one point and still disagree on this one.

I could not find a description of what the human manager's screen actually showed of the other group, so I cannot say which side is right — only that they differ. Confidence: high on the code, medium on the direction.

### S4 — The training diagnostics reproduce the known evaluation asymmetry

`rl_manager.run_batch` computes `metrics` from `state[k].mean()` over the raw state: mean `contribution` therefore includes the fabricated 9 at timed-out cells, and mean `contributor_payoff` includes the hard zeros of D2. The two-manager branch recomputes per-group means dividing by `rl_count` (group size), not by the number of valid players. These feed the parquet and the wandb `eval/*` keys, so the curves a human reads during the run will overstate contributions and understate payoffs by roughly the timeout rate. Diagnostics only — no effect on the gradient. Confidence: high, impact low.

### S5 — Reward scale and the constant headcount term

Returns under `reward_mode: sum` are on the order of 2,000 (≈ 90 per round × 24 at γ = 0.98) and are never normalised. `smooth_l1_loss` uses the default β = 1, so it is pure L1 everywhere that matters, and gradients are further clamped to ±1 per parameter. Most of what the network must represent is the near-constant `20·n_valid` offset (S1) rather than the part the policy controls. This is a plausible reason for a flat or slow reward curve and is worth watching (`q_mean` against `next_reward`) rather than pre-emptively changing. Confidence: medium, suspicion only.

### S6 — What an imperfect opponent does to what the manager learns

Even with D3 fixed, the opponent is a *sampled* per-round model of the average human manager, not a manager. Three consequences worth stating before the results are interpreted:

- Its severity profile is flat where the human's is steep. Per the mechanism table, the human's `E[p | p>0]` runs 7.99 → 3.87 across the 0–4 to 15–19 contribution bands; the corrected GNN manages 7.26 → 4.63 and the lagged one is essentially flat at ~5.5. The RL manager competing against it faces an opponent that under-rewards high contributors' compliance.
- It never times out. `punish()` sets `punishment_valid` to all-ones unconditionally, while human managers gave no input on 814 of 19,200 rows (4.2%). Both managers in the env are always present.
- Its punishments reach the RL manager's own group. The contributor AH's graph is fully connected across all eight agents (correctly — see V10), so the opponent's actions influence the RL group's contributions through message passing, and any competitive effect the manager learns is entangled with that.

None of this is a code defect; it is the standing limit on what a number coming out of this training run means. Confidence: high on the mechanism, unquantified in effect.

---

## 3. Checked and found correct

A short list of where not to look again.

1. **Observation timing — the central question, and it is right.** At the moment `manager.get_action` is called, the state holds round t's `contribution` and `contribution_valid` (set by `update_contribution` inside `reset`/`step`), round t−1's `prev_punishment`, round t's `round_number`, and round t's membership via `agent_group` → `in_group`. That is the human interface: the accounting identity `common_good_t = 1.6·Σc_t − Σp_t` holds on all 4,512 human group-rounds to a maximum residual of 2.8e-14, which fixes `punishment` at (episode, t, player) as the punishment applied to round t's contribution. The manager conditions on the current round, unlike the punisher before its fix. The manager does **not** see `prev_contribution`, `common_good` or payoffs — a feature choice, not a timing error.
2. **Reward window.** `env.step()` returns the payoff computed during `punish()` at round t; `update_reward` only re-selects an already-computed field and is not disturbed by the `update_contribution` and `apply_switch` calls that precede it. Verified by probe: punishing 0/1/2 in rounds 0/1/2 returned 104.0/96.0/88.0, each equal to `group_payoff_sum` at the matching `punish()`. Replay stores `(s_t, a_t, r_t)` at `episode_step = t`.
3. **Terminal bootstrap and the TD target.** `next_v[:, :, -1] = 0`; `max_next_q_value` is taken from index `1:` and weighted by `in_group` at t+1, so a member who arrives at t+1 contributes to the new group's continuation value. The per-group Q is a sum over in-group agents and the reward is a group sum, which is consistent.
4. **The common-good divisor.** `compute_common_good_per_group` divides by `sum_contribution_valid`, and that is what the game did: `payoff = 20 − c − p + common_good/n_valid` reproduces the human `payoff` column exactly on every row with `n_valid > 0` (0.18% of rows are group-rounds where everyone timed out, where the quantity is undefined), whereas dividing by group size matches only 87.4% overall and 2.1% on rounds containing a timeout. `parse_agent_rounds` uses the same divisor, so training and env agree.
5. **Punishment is deducted twice on purpose.** `20·n_valid + 0.6·Σc − 2·Σp` reproduces the human data to 8.5e-14, so the fact that punishment costs the punished player *and* the pool is the game's rule (basics.md says so explicitly), not a double-count bug.
6. **Switch cadence and anchoring.** `switch_every: 4` with `next_round % switch_every == 0 and next_round < n_rounds` fires membership changes entering rounds 4, 8, 12, 16 and 20. The human data has membership changes after rounds 3, 7, 11, 15 and 19 and none after 23 — an exact match. The predictor is run every round to keep its RNN warm and its output used only at arrival rounds, and it reads round-s post-`punish()` features, matching the re-anchored `does_switch` labels.
7. **Episode length.** `n_rounds: 24` matches the data: every one of the 100 episodes runs `round_number` 0…23, with 8 players in every episode-round.
8. **Action range.** 0–30 is exactly the interface in `reports/basics.md`, and every one of the 31 levels is observed in the human data. (The support inside that range is S2.)
9. **Group masking in the rollout.** `rl_mask` is rebuilt from the live `env.agent_groups` every round, so an agent arriving at s+1 is punished by its new group's manager — the requirement `notes/groub_competition.md` records. Storing `action` rather than `final_punishment` in the replay is safe because `manager.update` slices the TD error to `rl_group_id`, and in the legacy single-manager path the two are identical anyway.
10. **The graph.** `get_action` and `update` both call `encode` with `edge_index=None`, which rebuilds a fully-connected graph over the *group-expanded* batch (`E × n_groups` graphs of 8 nodes). That is the correct size — passing `env.batch_edge_index`, which is built for `batch_size` graphs, would have been wrong — and it matches AH training, where `train.py::create_fully_connected(n_player=8, …)` also spans both sub-groups. The `batch`, `edge_index` and `agent_group_mask` kwargs `rl_manager` passes into `manager.update` are absorbed into `**obs` and never used; harmless, but misleading to read.
11. **`api_manager.RLManager` zeroing `round_number`.** The comment claims it cannot affect the output and the comment is right: the bias head is `Seq(Lin(b_size, hidden), Tanh(), Lin(hidden, 1))`, a single scalar broadcast across all 31 action logits, so it shifts every Q-value equally and cannot move the argmax.
12. **The replay ring buffer.** `Memory`'s `current_row` / `episode_queue` bookkeeping is consistent: after k episodes rows 0…k−1 hold them and `len()` is k, so `get_random`'s `randint(0, len(self))` never reads an unwritten row, and the wrap at `n_episodes` overwrites in order.
13. **Empty groups.** Verified in the probe: with all eight agents in group 0, group 1's `group_payoff` and `group_payoff_sum` are both 0 and no division by zero occurs. Empty groups are not hypothetical — 288 of 4,800 human group-rounds (6.0%) are empty because all eight players merged. The Q-sum over an empty group is 0, consistent with a 0 reward and a 0 continuation.
14. **No aliasing from the `prev_` shift.** `step()` rebinds `state["prev_x"]` to the same tensor object as `state["x"]`, but every subsequent writer (`update_contribution`, `punish`, `update_common_good`, `is_first`, `punishment_valid`) assigns a freshly allocated tensor, so the previous round's copy is never mutated underneath.
15. **Round-0 defaults.** `reset_state` fills `prev_punishment`, `prev_contribution` etc. from `default_values`, reproducing what `create_torch_data`'s `shift()` puts at index 0 in training; `is_first` is True and `round_number` is 0.
16. **Input validation.** `reward_mode` is checked against `("avg", "sum")` with a clear error; `reward_formula` is explicitly deprecated and warned about; `punish()` asserts the action is in range and int64; `n_levels: 24` for the `round_number` one-hot covers 0…23 exactly in config D (config `03_2g8a_sum.yml` uses 32, which is harmless slack).

---

## 4. What I could not check, and why

- **Anything that needs PyG.** `torch_scatter` is Linux-only, so I could not instantiate a real `GraphNetwork`, load any `.pt` artifact, or run `rl_manager.run_batch` end to end. Every model-level claim here (the opponent's feature set, the manager's encoding, the contributor's and switch model's inputs) is read off the training configs and artifact naming, not off the loaded weights. The decisive test for D1 and D5 is a probe in the style of `scripts/data_analysis/sim_timeout_serving_probe.py`, wrapping each model's `predict` around `rl_manager.run_batch` and reporting the value served at every timed-out cell, on Raven. I did not run the repo's PyG test suite either.
- **What the human manager's screen showed of the other group.** This decides the direction of S3. `notes/groub_competition.md` records the RL-side design intent (show both, flag which is yours), but I found nothing describing the human interface, and `api_manager.create_data` assumes the opposite.
- **Whether the RL manager will in fact exploit D1.** The mechanism is confirmed and the manager observes `contribution_valid`, so the incentive is there and is unopposed; whether 15,000 update steps with ε = 0.1 find it on 2.2% of agent-rounds is an empirical question that only a run answers. Watching the mean punishment conditioned on `contribution_valid == False` would settle it in one plot.
- **The size of the effect of D3.** I compared the two punishers' documented mechanism statistics; I did not run the RL loop against both.
