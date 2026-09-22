# What the language model is told, and whether the description is any good

## 1. Declaration

**Slot:** the prompt, the trace format and the answer parser for the LLM manager of `doc/plans/llm-manager.md`, plus a replay harness that judges a prompt without a rollout. Nothing is trained on this branch: no artificial human, no manager, no copula, no weights, no training config. The environment is untouched. The frozen evaluation directory is untouched.

**Parent:** `dev` at `fddf37a`, which carries the four timeout and free-punishment fixes, the rule family and the paired competition harness.

**Scope boundary.** A sibling owns `LLMManager`, the client and the serving; another owns the rollout evaluation. This branch owns only what the model reads and how its answer is read back, and it builds against an injected client so it runs without a GPU.

### Why replay, and what it is for

A prompt judged only by how the manager scores in a rollout conflates two things: whether the description communicates the game, and whether the model's judgement is any good. A rollout also needs a GPU, 24 sequential batched calls per episode, and a contribution model that reacts to the manager, so every number in it depends on the whole stack.

Replay removes all of that. `experiments/2group_8agent_50ep.csv` holds 50 real games; a **decision point** is one (game, manager, round). The trace is built from what actually happened up to that round, the model is asked for that round's punishments, and the answer is compared against what the real manager did **on the same state**. No simulation, no contribution model, no feedback loop. The states are identical across prompt variants, so a difference between two variants is a difference between the prompts.

### The instruction that shaped this branch

Do not tune the prompt until the model reproduces the human policy. The interesting result is what a model does when told the rules and the objective, not what it does when steered to an answer we already have. Every variant below was written once, evaluated once, and is reported, including the ones that did badly. No variant contains a hint about whom to punish; `test_llm_prompt.py::test_no_version_hints_at_whom_to_punish` is the guard that keeps it that way.

## 2. What the model is told

**The objective is settled and is not an axis.** The maintainer fixed it: the group's **undivided** common pool, summed over the game -- what the real managers were paid on (`reports/basics.md`). The pool is the welfare measure; it counts what contributors produce and charges punishment at full price, so a manager that raises contributions by punishing enormously does not score on it. That is the measure working. The objective discriminates: a correctly-targeted threshold rule is indistinguishable from never punishing on the pool (about +0.5, interval spanning zero), while the capped fitted rule of PR #219 beats that threshold rule by +5.0 pool points. The pool rewards good management; it does not reward punishment for its own sake. An earlier `@per_member` switch was built and then removed when the maintainer settled the question.

Five variants, chosen so that **every comparison against `v3_explicit_pool` is a single-factor change**.

| version | arithmetic of the pool | trace shows pool | answer | fingerprint |
|---|---|---|---|---|
| `v1_bare_pool` | left to the rules | yes | direct | `5d40be831fdd` |
| `v2_stated_pool` | one sentence | yes | direct | `59c1fc1dbd95` |
| `v3_explicit_pool` | two sentences | yes | direct | `b211d81b411a` |
| `v4_explicit_nopool` | two sentences | **no** | direct | `9e1fb13d7a59` |
| `v5_explicit_reason` | two sentences | yes | **reason first** | `0edc2e953d18` |

The fingerprint is the sha256 of exactly the blocks a version selects, including `TraceRenderer.spec`, so a change to a template moves every fingerprint that uses it. `test_llm_prompt.py` pins all five. From the first commit on, a wording change fails that test and has to be given a new name; a version is never edited in place.

Common to every variant: 24 rounds; four members to start and a second group never seen; 20 points each per round, any whole number of them into the pool; the pool multiplied by 1.6 and split equally regardless of what each put in; punishment 0 to 30 out of the player's account **and** out of the pool; reshuffling after every fourth round; a member who gave no input put nothing in and cannot be punished. The objective is stated explicitly in all five. No variant says the contributors are models.

**The framing the text avoids.** No word that reads as discouraging punishment, and no suggestion that punishing works. Both are conclusions the comparison exists to watch the model reach or fail to reach, and either in the text would make a good result uninterpretable. `v3`'s whole trade-off is two sentences of accounting and nothing else:

> Every point put in enters the pool multiplied by 1.6, and every point you punish leaves it at full price, in the round you set it. A punishment therefore pays for itself only through what it changes about the contributions of the rounds that follow.

`test_llm_prompt.py::test_no_version_discourages_or_encourages_punishment` is the guard.

### The trace

Typed events through one renderer (`src/aimanager/llm/trace.py`), never string concatenation. The history and the round being decided go through the same templates, and the round being decided is marked `punishment not set yet` rather than being recognisable by a missing clause -- an asymmetry that is only an absence is the cheap way to leak. Values are sanitised of newlines and double quotes before substitution.

```
Round 7
  (Player 5 joined your group; Player 1 left your group)
  Player 2 put in 1, you punished 0
  Player 3 gave no input (put in nothing, cannot be punished)
  Pool: 41.2 (1.6 x 32 put in, minus 10 punished)
```

**The imputed contribution never reaches the model.** The human CSV stores 0 on a timed-out player and the simulation's `per_round.parquet` stores the dataset median 9; neither is a contribution the game used. `PlayerRound.from_masked` is the single place a validity flag becomes `contribution is None`, and `None` renders as "gave no input". The same mask is applied at source in the battery, because `convert.load_sim` does not mask timeouts the way `load_human` does.

**The pool line is the game's own accounting identity**, `1.6 * contributed - punished`, with punishment counted only on players who gave input. `test_llm_trace.py::test_pool_matches_the_human_accounting_identity` checks it against `common_good` on real group-rounds.

**A known defect in this version of the format, stated rather than fixed.** When every player leaves for the other group, a manager's group is empty and that round produces no block, so the trace jumps from `Round 5` to `Round 7` with nothing to say why. **Measured:** 144 of the 2400 game-rounds have an empty group, and **218 of the 2152 decision points (10.1%)** carry at least one such gap before them; 2 of the 24 sampled decision points do. The round numbers are printed, so the gap is visible to the model but unexplained. The fix is an `empty` event kind rendering "your group was empty this round"; it is not applied here because it moves every fingerprint, and it belongs in a `v6` rather than in an edit to a version that has produced results.

### The answer, and the parser

One line: `PUNISHMENT: Player 1 = 3, Player 2 = 0, ...`, covering exactly the players listed, each a whole number 0 to 30.

The parser is a **guard, not the mechanism**. On a served endpoint the answer should be constrained at the token level (vLLM guided decoding), which makes the failure rate zero by construction. The guard and its counter stay anyway, because a constrained decode can be misconfigured or silently dropped and nothing else downstream would notice.

**What the guided decode has to constrain.** Not "four integers in 0..30": the group is four only at the start. **Measured:** 1597 of the 2152 decision points have a group that is not four, with sizes running 1 to 8. The schema has to be built per round from the labels the prompt listed, which is exactly what `prompt.labels` carries and what the parser checks against.

**The fallback is not neutral, which is why it is loud.** A failure returns zero punishment for that episode-round, and zero punishment is exactly the policy every collapsed learned manager in this project converged on. A silent fallback would push a result toward the outcome the comparison exists to distinguish, in proportion to how often it fires. So every failure logs a warning, carries `fallback="zero"`, and is counted; `strict=True` raises instead. A non-zero rate is a bug to fix, not a property of the model to report and move past.

Out-of-range values are **not** clamped: clamping would turn a model that does not know the action space into one that scores well at the boundary. A punishment set for a no-input player parses fine, is recorded as `wasted`, and is zeroed only by `enforce` -- so a wasted decision shows up in the log instead of vanishing.

## 3. Results

### Step 0: the battery reproduces the published human reference

**Measured.** `scripts/data_analysis/llm_prompt_replay.py` computes the policy shape on the evaluation suite's own `RPA_EDGES` / `RPA_LABELS`, with `contribution_valid` masked at source, and the targeting triple from `scripts/rl_param_noise/targeting.py`. Run through `evaluation_suite.convert.load_human` and `ResponseMetrics().rpa`, it returns

```
4.755 / 2.973 / 1.672 / 0.978 / 0.692 / 0.267   over {0}, 1-5, 6-10, 11-15, 16-19, {20}
rho -1.000   contrast 4.488   contrast/mean 2.430   mean punishment 1.847
```

which is the published human profile to **0.0004** on the largest bin. The binning is the evaluation suite's, not a look-alike.

**Measured.** The harness finds **2152 decision points** in the 50 deduped games: 2256 group-rounds minus the **104** where the human manager timed out (no human decision to compare against). Those 104 rounds stay in the trace, marked. 280 of 9600 agent-rounds carry `player_no_input`.

### Step 0b: what the battery separates, before any model is asked

**Measured**, `plots/data_analysis/llm_prompt_replay/reference/`. Four reference policies driven through the real prompt, the real stub client and the real parser, on all 2152 decision points.

| source | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} | rho | contrast | contrast/mean | mean pun. | distinct bins | profile SNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| human managers (whole CSV, eval suite) | 4.755 | 2.973 | 1.672 | 0.978 | 0.692 | 0.267 | -1.000 | 4.488 | 2.430 | 1.847 | 6 | -- |
| human managers (same decision points) | 4.755 | 2.973 | 1.672 | 0.978 | 0.692 | 0.267 | -1.000 | 4.488 | 2.430 | 1.847 | 6 | 13.788 |
| `human_table` stub | 5.000 | 2.231 | 1.000 | 0.455 | 0.000 | 0.000 | -0.998 | 5.000 | 3.765 | 1.328 | 5 | 259.278 |
| `never` stub | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **NaN** | 0.000 | NaN | 0.000 | 1 | **0.000** |
| `flat3` stub | 3.000 | 3.000 | 3.000 | 3.000 | 3.000 | 3.000 | **NaN** | 0.000 | 0.000 | 3.000 | 1 | **0.000** |
| `inverted` stub | 0.000 | 0.000 | 0.000 | 5.000 | 5.000 | 5.000 | **+0.868** | -5.000 | -2.521 | 1.983 | 2 | inf |

Three things this establishes and one it does not.

1. The whole path is lossless: a human-shaped policy handed to the stub comes back through the prompt and the parser at `rho` -0.998 with the human sign.
2. The inversion is caught with the opposite sign, at a *higher* mean punishment (1.983 against the human 1.847). Reading the magnitude alone would have called it the busier manager.
3. **The two flat policies are refused, not scored.** `rho` is NaN and `profile_snr` is 0 on both, and `distinct_bins` is 1. This is the whole reason the triple is reported together: a flat policy has no targeting direction, and a rank correlation asked for one on six identical numbers would have invented one.
4. It does **not** establish anything about a language model. These are stubs; they exist to show what the instrument can separate.

**Measured**, the agreement side, which is context for reading any collapsed model: **68.5%** of human agent-rounds carry punishment 0, so a manager that never punishes matches the human decision exactly 68.5% of the time and scores a mean absolute difference of 1.847. Exact-match rate is therefore close to useless as a quality measure here, and is reported only so that fact is visible.

### Step 1: the replay comparison

**How the completions were obtained, and why that is a caveat rather than a footnote.** No vLLM endpoint was available on this branch -- serving is a sibling's scope -- so the model calls were made through nested agents, one fresh context per decision point, with a small Claude model (`haiku`) standing in for a small served model. This is a PROXY. Five things follow and all five are limitations of the collection, not of the prompt:

0. **The model-size comparison was not obtained.** The coordinator asked for the chosen prompt on both Qwen3-8B and Qwen3-32B, because a prompt that works at 32B and fails at 8B is a different finding from one that fails at both. The whole concurrency budget went into the five-variant sweep at one size, so that comparison is outstanding and is the first thing to run once an endpoint exists. Nothing below separates "the prompt does not communicate the game" from "this model cannot do the task".

1. The answering context carries a coding-agent system prompt in front of the manager prompt. A served `vllm serve Qwen/Qwen3-8B` does not.
2. The model is not the model the plan names. A number here does not transfer to Qwen3-8B; what transfers is the harness, the battery and the direction of the variant differences.
3. The wrapper the proxy needs ("reply with ONLY your answer") pulls against `v5_explicit_reason`, which asks for three sentences of reasoning before the answer line. The `v5` row is therefore the weakest of the five and should be re-run first on a real endpoint, where no wrapper is needed at all.
4. **Parallel collection can mis-assign a reply to the wrong prompt**, which downstream looks exactly like the model answering for the wrong players -- that is, like a parse failure, the number this branch reports. A real client pairs request and response directly and has no such failure mode. So every cached completion is checked before scoring on DEMONSTRABLE evidence only: a row is dropped when the parser rejects it **and** its own text names rounds none of which is this decision's round. **Measured: 5 of 117 completions (4.3%) were dropped this way** -- 1 in `v3`, 4 in `v4`, none in `v1`, `v2` or `v5` -- and their text is unambiguous, one opening "Decision for Round 8" under a key for round 18 and reasoning about rounds 6 to 8. A further 11 answers name no round at all and answer for a roster the decision point does not have; 9 of those 11 are in `v4`, against 0 to 2 elsewhere. Those are NOT dropped, because there is no evidence either way and dropping on suspicion would quietly select which failures count. The consequence is stated rather than cleaned away: `v4`'s row is contaminated and is reported as such.

**The sample.** A seed-42 sample of 24 of the 2152 decision points, the SAME 24 for every variant, dumped once and answered once per variant. Of those, **17 were answered by every variant after cleaning** and are what the table below scores: 62 agent-rounds, 58 of them with a valid contribution, 4 no-input. The sample spans rounds 0 to 23 and group sizes 1 to 8.

**Measured**, `plots/data_analysis/llm_prompt_replay/replay/`.

| source | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} | rho | contrast | contrast/mean | mean pun. | distinct bins | profile SNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| human managers (whole CSV) | 4.755 | 2.973 | 1.672 | 0.978 | 0.692 | 0.267 | -1.000 | 4.488 | 2.430 | 1.847 | 6 | -- |
| human managers (these 17 points) | 1.750 | 0.538 | 1.167 | 0.000 | 2.333 | 0.000 | -0.510 | 1.750 | 2.071 | 0.845 | 5 | 2.487 |
| `v1_bare_pool` | 1.250 | 0.769 | 0.111 | 0.000 | 0.000 | 0.000 | **-0.984** | 1.250 | 3.295 | 0.379 | 4 | 4.330 |
| `v2_stated_pool` | 0.000 | 1.077 | 0.111 | 0.000 | 0.000 | 0.000 | -0.460 | 0.000 | 0.000 | 0.276 | 3 | 7.423 |
| `v3_explicit_pool` | **2.500** | 0.231 | 0.611 | 0.556 | 0.000 | 0.000 | -0.543 | **2.500** | 3.718 | **0.672** | 5 | 4.249 |
| `v4_explicit_nopool` (contaminated) | 0.000 | 0.000 | 0.167 | 0.111 | 0.000 | 0.000 | +0.225 | 0.000 | 0.000 | 0.069 | 3 | 4.927 |
| `v5_explicit_reason` | 1.500 | 0.077 | 0.444 | 0.111 | 0.000 | 0.000 | -0.543 | 1.500 | 3.955 | 0.379 | 5 | 4.791 |

`shape.csv` also carries a "(parsed answers only)" row per variant, which is what separates what the model chose from what the zero fallback chose for it.

**What this shows.**

1. **Every clean variant has the human's sign.** `rho` is negative on `v1`, `v2`, `v3` and `v5`: told the rules and the objective and nothing else, the model punishes low contributors more than high ones. Nothing in any prompt says to.
2. **All of them punish far below the human level.** Mean punishment 0.28 to 0.67 against 0.845 for the real managers on the same 17 points and 1.847 across the whole data. The model's *aim* is human-shaped; its *force* is a third to a fifth.
3. **Relative to its own force, the aim is if anything sharper than the humans'.** `contrast_over_mean` runs 3.3 to 4.0 on `v1`, `v3`, `v5` against the human 2.071. This is the pair of statements the targeting triple exists to keep apart, and reading either one alone would have got it wrong: on `contrast` the model looks much worse than the humans, on `contrast_over_mean` slightly better, and both are true.
4. **The cost ladder moves force, not sign.** `v1` (bare) 0.379, `v3` (two sentences of accounting) 0.672. Stating the arithmetic did not suppress punishment -- the plausible worry -- it raised it, and gave the largest `{0}` bin of any variant. `v2` (one sentence) is the odd one out at 0.276 with `contrast` 0: it punishes the 1-5 bin and not the 0 bin, so its rank is negative while its contrast is nil. On 17 points that is as likely to be noise as structure.
5. **`v5` (reason first) matches `v3` on rank and halves its force.** Its collection is the one the proxy wrapper fights (caveat 3), so this row is the least trustworthy of the clean four.
6. **`v4` is not interpretable.** Its collection is the one with demonstrable mis-assignment, and its parse failure rate is 52.9% against 0 to 17.6% elsewhere. Since a failure falls back to zero, a high failure rate *manufactures* the never-punish policy `v4` appears to have. The attractive reading -- "removing the pool line stops the model tracking which round's roster it owes an answer for" -- is a real hypothesis and is exactly what the data cannot distinguish from a sloppy collection. **The pool axis is unresolved and is the second thing to re-run.**

**The n, stated plainly.** 17 decision points and 58 valid agent-rounds. Several bins are exactly 0.000 because nothing was punished in them, `distinct_bins` runs 3 to 5, and the human row's own `profile_snr` on these points is 2.487 -- barely above the gate. The differences among `v1`, `v3` and `v5` are inside what this sample can resolve. What is outside it: the sign, which is negative on four of five variants; the level, which is far below human on all of them; and `v4`'s failure rate.

### Step 2: the parse failure rate, and what constrained decoding has to constrain

**Measured, per answer, on the same 17 decision points.** This is the headline number the run is not a result without.

| variant | failure rate | failures | `no_marker` | `label_mismatch` | `not_integer` | answers in the requested form |
|---|---|---|---|---|---|---|
| `v1_bare_pool` | **17.6%** | 3 | 1 | 1 | 1 | 14 |
| `v2_stated_pool` | **0.0%** | 0 | 0 | 0 | 0 | 17 |
| `v3_explicit_pool` | **11.8%** | 2 | 2 | 0 | 0 | 15 |
| `v4_explicit_nopool` | **52.9%** | 9 | 1 | 8 | 0 | 8 |
| `v5_explicit_reason` | **0.0%** | 0 | 0 | 0 | 0 | 17 |

Every parsed answer used the labelled form; the positional fallback never fired. No variant ever set a punishment for a player who gave no input, so the wasted-punishment rate is 0 across the board -- the prompt's statement that those players cannot be punished, plus the trace marking them, was enough.

**These rates are unacceptable and are a bug, not a finding.** On `v1` and `v3` the numbers in the answer were usually well formed and the model simply wrote `Punishment decision:` or `Round 17 Punishment Decision:` instead of the literal `PUNISHMENT:`. A guided decode removes that entire family. Until it is in place, one round in six to one in nine on `v1` and `v3` silently becomes never-punish, which is the policy under test.

**Measured, on a superseded generation of the text, and reported because it is about the PARSER rather than about the prompt.** 64 real completions collected on an earlier wording, parsed with the final parser: failure rates of 16.7% and 22.5%, in two families.

- `no_marker` (8 of 13): the model wrote `Punishment decision:`, `Decision:`, `Final punishment decision:` or `DECISION:` instead of the literal `PUNISHMENT:` it was asked for, usually with the numbers themselves perfectly well formed after it.
- `label_mismatch` (5 of 13): **the model answered for the wrong set of players** -- naming a `Player 8` who was not in the group that round, or omitting five of the seven who were.

The second family is the one that matters for the design. A positional parser, or a lenient labelled one that took what it could match, would have silently assigned those numbers to the wrong players and produced a policy shape that looked fine. The strict label check turns a silent corruption into a counted failure.

It also says exactly what a guided decode must constrain: not only "four integers in 0..30" but **the label set for that round**, which changes round to round as the group reshuffles. A schema fixed at four players would itself be wrong on the 1597 of 2152 decision points where the group is not four.

### Step 3: which prompt, and on what grounds

**Chosen: `v3_explicit_pool`, fingerprint `b211d81b411a`.**

The grounds, in the order they weighed:

1. **It exercises the action space most.** Mean punishment 0.672 against 0.276 to 0.379 for the other clean variants. That matters more than it looks: zero is both a policy and the parse fallback, so the variant that punishes least is the one whose result is hardest to tell apart from a failure. `v3` is the furthest from that confound.
2. **Its `{0}` bin is the largest of any variant** at 2.500 (2.857 on parsed answers only), which is the bin the human managers are most distinctive in (4.755 overall, 1.750 on these 17 points). None of the variants reaches the human level; `v3` gets closest.
3. **Its shape does not move when the failed answers are dropped**: `rho` -0.543 to -0.584, mean 0.672 to 0.750. So what it shows is what the model chose, not what the fallback chose for it. That check is why the "(parsed answers only)" rows exist.
4. **Low parse failure** at 11.8%, and its failures are `no_marker` only -- the family a guided decode removes entirely.
5. **It states the trade-off most completely**, which is what the task asked the prompt to make legible, and the replay says that statement costs nothing: it raised punishment rather than suppressing it.

Against it: `v2` and `v5` both parsed at 0%. That did not decide it, because the failure rate is a serving problem with a known fix (constrained decoding) while the policy shape is the thing the prompt is actually for. `v5_explicit_reason` is the runner-up -- same `rho`, highest `contrast_over_mean`, 0% failures -- and is the first variant to re-test once no collection wrapper is fighting its "reason first" instruction.

**What was NOT done, deliberately.** No variant was revised after seeing a result. Every one was written once, run once, and is reported here including `v2`, which came out oddly, and `v4`, which came out uninterpretable. There was an obvious temptation after seeing the level gap -- add a sentence encouraging the model to use the range -- and it was not taken, because a prompt tuned until the model matches the human policy measures the tuning.

**The finding worth carrying forward, stated as the caution asked.** Told only the rules and the objective, a small model *aims* like a human manager and *punishes* at a third to a fifth of the human level. If the interesting question is whether an LLM manager arrives at human-like punishment unprompted, the answer on this evidence is: it arrives at the human's target and not at the human's intensity. Closing that gap by telling it to punish harder would be the same error as tuning toward the human policy, and the gap itself is the result.

## 4. Measured versus inferred

### Measured

Everything in this section came out of a run and can be recomputed from the committed tables and the code on this branch.

- The human reference profile through this script's binning: `4.755 / 2.973 / 1.672 / 0.978 / 0.692 / 0.267`, `rho` -1.000, contrast 4.488, contrast/mean 2.430, mean punishment 1.847 -- the published numbers to 0.0004.
- 2152 decision points; 104 manager timeouts excluded; 280 of 9600 agent-rounds with `player_no_input`; 1597 of 2152 decision points with a group that is not four; 218 of 2152 with a gap in the trace from an empty group; 68.5% of human agent-rounds at punishment 0.
- The reference-policy battery: `human_table` -0.998, `inverted` +0.868 at a higher mean punishment than the human managers, `never` and `flat3` refused by the gate.
- The prompt-variant replay on 17 paired decision points and 58 valid agent-rounds: `rho` -0.984 / -0.460 / -0.543 / +0.225 / -0.543 and mean punishment 0.379 / 0.276 / 0.672 / 0.069 / 0.379 for `v1` to `v5`, against a human -0.510 and 0.845 on the same points.
- The parse failure rate per answer: 17.6% / 0.0% / 11.8% / 52.9% / 0.0%; every parsed answer in the labelled form; wasted punishment 0 everywhere.
- The collection audit: 5 of 117 completions demonstrably mis-assigned and dropped, 11 more with a wrong roster and no round named, 9 of those in `v4`.

### Inferred

Everything here is judgement on top of those numbers and is labelled as such.

- **That a number from this proxy transfers to Qwen3-8B.** It does not. The collection used a different model behind a coding-agent system prompt. What is meant to transfer is the harness, the battery, the direction of the variant differences, and the failure modes the parser caught.
- **That the replay ordering predicts a rollout ordering.** Replay holds the state fixed; a rollout lets the manager move it. A manager that looks quiet on replay may look different once contributions react to it. Replay bounds what the prompt communicates, not what the manager achieves.
- **Where the noise gate's threshold sits.** `guard_report.py` reads "below about 2 the six bin means are within their own sampling noise" off a different replication axis (evaluation points, not episodes). The number is carried over as a rule of thumb, not re-derived for this axis.
- **The n.** 24 decision points is a small paired sample, chosen against a wall-clock budget, not a statistical one. With a served endpoint the whole 2152 is affordable and should be run before any of this is treated as settled.
