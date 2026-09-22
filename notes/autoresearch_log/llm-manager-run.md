# A language model managing the public goods game, against the managers we already understand

## 1. Declaration

**Slot:** run the experiment the three sibling branches built the parts for. Combine #220 (the prompt), #222 (the serving and the manager class) and #221 (the battery and the paired harness), then measure the language model in the seat every baseline has been measured in. Nothing is trained here: no artificial human, no manager, no copula, no weights. The prompt is not touched at all.

**Parents.** `main` at `617683c`, which is an ancestor of `dev` at `fddf37a`, which is the base all three siblings share.

| branch | PR | base | what it owns |
|---|---|---|---|
| `llm-manager-prompt` | #220 | `dev` | the five prompt versions, the typed trace renderer, the parser |
| `feat/llm-manager-serving` | #222 | #220 | `LLMManager`, the batched client, vLLM serving, the per-call constrained decode |
| `auto/llm-manager-eval` | #221 | `dev` | the battery, the paired harness, the stub, the power table |

**The objective is the group's undivided common pool**, settled by the maintainer and not an axis here. Total contribution sits beside it as the diagnostic that says by which route an arm reached its pool. The two are never averaged.

## 2. The merge, and what was asserted afterwards

`main ⊂ dev ⊂ #220 ⊂ #222`, so merging `feat/llm-manager-serving` is a fast-forward that carries the prompt with it. `auto/llm-manager-eval` then merges on top. The two sides touch exactly one file in common, `src/aimanager/manager/api_manager.py`, and in different places — #222 registers the `llm` manager type, #221 adds the `sigmoid` rule — so git merged it without a conflict and both changes are present. Tests were run after each merge, not at the end.

Three things were asserted after the merge, because each is a claim a sibling made that the merge could have broken.

**The prompt fingerprint is unchanged.** All five versions hash to what #220 pinned, and the chosen one is `v3_explicit_pool` = `b211d81b411a`. Nothing in `aimanager.llm` was edited; a wording change would have moved the hash and failed `test_llm_prompt.py`.

```
v1_bare_pool 5d40be831fdd   v2_stated_pool 59c1fc1dbd95   v3_explicit_pool b211d81b411a
v4_explicit_nopool 9e1fb13d7a59   v5_explicit_reason 0edc2e953d18
```

**The battery still reproduces its validation.** #221's run B (seeds 45/46/47, 2,048 episodes per seed, eight arms) was re-run on the merged tree and compared against its committed `battery.csv` column by column: **75 float columns across 8 arms, largest absolute difference 0.0** — bit for bit, not "within noise". Only `wall_clock_s` differs, which measures the machine rather than the managers. Tables in `plots/data_analysis/llm_manager_battery/merge_revalidation_s45_47/`.

**The decode is still built per call from the roster present.** `_constraint_for(prompt)` builds the regex from `prompt.labels`, which is that decision point's roster, at sizes 1 through 8. A constraint for `{2, 3, 5}` admits `PUNISHMENT: Player 2 = 10, Player 3 = 0, Player 5 = 30` and refuses the same three numbers filed against Players 1, 2 and 3. No separator carries an unbounded whitespace quantifier, which is the trap #222 recorded.

## 3. Two integration seams, and one bug the smoke run found

**Telemetry.** The harness reads `telemetry()` off any manager through `stub.collect_telemetry` and records **NaN** for a manager that keeps none, precisely so that "this rule has no tokens" can never be read as "the language model spent none". So the one manager that does spend tokens has to answer to those six key names, and `LLMManager.telemetry()` / `reset_telemetry()` is that bridge. `n_calls` means different things on the two managers and the code says so rather than papering over it: the stub answers a whole batch in one call, this manager issues one completion per episode-round.

**The arm.** `run_battery.py --llm <config>` passes the config's `manager:` block to `LLMManager` unchanged, so 8B to 32B is a different file on the command line and never an edit to code. The endpoints come from `$HOSTED_VLLM_API_BASE`, which the serving job exports once its servers answer `/health`; they are deliberately not in the config, because the config describes the model and the job describes where it is running.

**The seat used is `predict`, not `get_punishments`.** #222 recorded that `get_punishments` has unit tests but has never run inside a real job, while `predict` is exercised end to end, and `harness.build_manager` passes any object exposing `predict` straight through. Every number below came through `predict`.

### The bug: an empty group is not a decision point

Found by the 8-episode smoke run that runs inside the job before the real one, which is why it is there.

When all eight players have gone to the other group this seat holds nobody. The roster is then empty, `_constraint_for` has no labels to build a constraint over and returns `None`, and the model is asked — **unconstrained** — to punish nobody. Measured on Qwen3-8B: it answers with numbers anyway, the parser rejects them as `wrong_count` ("4 numbers for 0 players"), and the fallback to zero is counted as a parse failure.

**No result moved.** The punishment row is filled by a loop over `members`, so an empty group contributes zeros whether or not the call goes out. What moved was the one telemetry number that is supposed to be zero by construction — and a run cannot report "zero parse failures" while an empty group manufactures them. The question is now simply not asked; the round still enters the trace exactly as before, and the count of rounds this seat sat out is reported rather than dropped, because a run where it is large is a run where this manager was rarely consulted.

This is the same defect #220 documented on the replay side ("144 of 2400 game-rounds have an empty group") arriving in the rollout path, where it has a different consequence.

## 4. How the budget was spent

**Many small rollouts, not few large ones.** #222 measured that the limit is the KV cache rather than concurrency: while the whole population's traces fit in GPU memory a round re-prefills only the block that changed, and past that every round re-prefills its entire trace. The crossover is sharp between 200 and 400 episodes at 8B on four A100s, and below it a decision is **2.3× cheaper**. `harness.run_arm` already splits a budget into one batched rollout per seed and pools them into a single frame of independent episodes, so the cheap regime is bought simply by setting the rollout width to 200 and buying the total with seeds. Every arm faces the same seed schedule.

The width is a property of *this* prompt length, *this* round count and *this* GPU count, not a constant. At 32B there is no cheap regime at all — 64 GB of weights across two instances leaves too little KV cache — so there the width buys nothing and only the total matters; it is kept at 200 anyway so the two arms have identical designs.


**What was run.** Two jobs, each with its own vLLM servers on its own compute node, submitted with `scripts/llm_manager/battery_raven.sh`. Every arm in a job ran in the same invocation, against the same rival, on the same stack, from the same seed schedule.

| run | model | servers | width x seeds | episodes per arm | LLM wall clock |
|---|---|---|---|---|---|
| `qwen3-8b` | Qwen3-8B | 4 data-parallel, 1 card each | 200 x 30 (seeds 42-71) | **6,000** | 28.8 min |
| `qwen3-32b` | Qwen3-32B | 2 at tensor-parallel 2 | 200 x 5 (seeds 42-46) | **1,000** | see below |

Weights were reused from the 77 GB cache at `/ptmp/levinb/llm-manager-2026-09-22`; nothing was re-downloaded. The isolated run directory is `~/repros/ai-runs/llm-manager-experiment` on Raven, account `levinb`; no `rsync --delete` was used against it. As in #221, `episodes.parquet` (one row per episode) stays on the cluster and every committed table is derived from it; it regenerates from the command in each run's `run_args.json`.

**What 6,000 episodes buys, measured on this run's own per-episode spread** (`mdd.csv`, clone-world `sd_episode` 44.7): the minimum detectable difference on the pool is **2.23** points unpaired at 80% power, and **1.34** on total contribution. So this budget resolves an effect the size of the capped rule's 5-point margin (which needs ~1,256 episodes) and does not resolve the 0.51-point margin of punishing correctly over never punishing, which needs ~120,629. That last one no budget in this project reaches, and it is the reason the contribution column is reported beside the pool rather than underneath it.

## 5. Result: Qwen3-8B pays for punishment and buys nothing

**Measured**, `plots/data_analysis/llm_manager_battery/qwen3-8b/`, 6,000 episodes per arm, pool first.

| arm | **pool** | contribution | pool / member | spend | members |
|---|---|---|---|---|---|
| *`thr9_vs_thr9` (control)* | *65.43* | *47.61* | *15.96* | *2.62* | *4.10* |
| `capped_sigmoid` | **65.38** | 43.47 | 14.73 | 0.94 | 4.44 |
| `thr9_p10` = `stub_thr9_p10` | **60.87** | 45.01 | 15.66 | 2.87 | 3.89 |
| `never` = `stub_zero` | **60.50** | 37.81 | 13.17 | 0 | 4.59 |
| clone | **57.26** | 40.35 | 14.17 | 1.81 | 4.04 |
| **`llm_qwen3_8b`** | **56.45** | **37.50** | **13.00** | **0.82** | **4.34** |
| *`never_vs_never` (control)* | *51.47* | *32.17* | *12.87* | *0* | *4.00* |

**The language model against each baseline**, unpaired bootstrap over episodes, pool first and contribution beside it. Neither is averaged with the other.

| reference | **pool** delta [95%] | contribution delta [95%] |
|---|---|---|
| never-punish | **−4.05 [−5.59, −2.51]** | **−0.31 [−1.24, +0.63]** |
| `thr9_p10` | **−4.42 [−6.06, −2.77]** | −7.51 [−8.49, −6.52] |
| `capped_sigmoid` | **−8.93 [−10.48, −7.30]** | −5.97 [−6.90, −5.00] |
| clone (its own rival) | −0.81 [−2.35, +0.75] | −2.85 [−3.77, −1.93] |

**This is not the ambiguous case the design was braced for.** The caution going in was that a model landing level with never-punishing on the pool would not have been shown to fail, because the best hand-written threshold rule is also level with never-punishing there (+0.51, and 120,629 episodes to resolve). That escape does not apply: the model is **4.05 points below** never-punishing on the pool and the interval excludes zero at almost twice the run's own 2.23-point detection floor.

**And the contribution column says by which route.** It is **level with never-punishing** — −0.31, interval spanning zero — while spending 0.82 per member-round. So the model punishes, and the punishment buys no collaboration at all.

The pool is the game's own accounting identity, `1.6 x contribution − spend`, so every arm's margin over never-punishing splits exactly into what its punishment cost and what the contribution it bought was worth:

| arm | realised spend | contribution bought | that contribution is worth | **net on the pool** |
|---|---|---|---|---|
| `capped_sigmoid` | 4.18 | +5.66 | +9.06 | **+4.88** |
| `thr9_p10` | 11.15 | +7.20 | +11.51 | **+0.37** |
| clone | 7.31 | +2.54 | +4.06 | **−3.24** |
| **`llm_qwen3_8b`** | **3.55** | **−0.31** | **−0.50** | **−4.05** |

Read down the last column: the incumbent rule buys almost exactly what it pays, which is the +0.5 break-even this project already knew about; the capped rule buys twice what it pays; and **the language model is the only arm that pays and buys nothing.** Its spend is a third of the incumbent's and the least of any punishing arm — this is not a model punishing wildly — and it still loses the whole of it.

### Does it target? Yes, weakly, and it is the restraint that is missing

**Measured**, `targeting.csv` and `policy_shape.csv`, the suite's own bins with `contribution_valid` masked at source (a timed-out cell carries an imputed 9 the manager never saw, so it is dropped rather than binned).

| arm | `rho` | floor | ceiling | `rho_rel` | magnitude | noise gate | tie frac (p) | punish rate | mean p when positive |
|---|---|---|---|---|---|---|---|---|---|
| `thr9_p10` | −0.794 | −0.794 | +0.792 | **−1.000** | 2.466 | inf | 0.586 | 0.292 | 10.00 |
| `capped_sigmoid` | −0.571 | −0.825 | +0.821 | **−0.692** | 2.599 | 211.0 | 0.490 | 0.316 | 3.03 |
| clone | −0.336 | −0.818 | +0.813 | **−0.410** | 2.200 | 74.9 | 0.489 | 0.308 | 5.98 |
| **`llm_qwen3_8b`** | **−0.162** | −0.460 | +0.439 | **−0.353** | 2.366 | 31.7 | **0.853** | **0.079** | **10.60** |
| `never` | nan | nan | nan | nan | nan | nan | 1.000 | 0 | nan |

**The tie structure is why the raw rank must not be read alone.** The model leaves **85.3%** of its punishment column tied at zero against the clone's 48.9%, which attenuates any rank correlation computed on it. Its raw `rho` is −0.162, less than half the clone's −0.336; as a share of the Frechet bound in its own direction it is −0.353 against the clone's −0.410, and the gap nearly closes. This is exactly the effect #221 warned a sparing manager would sit in. Its noise gate is 31.7, far above the ~2 below which a profile is its own sampling noise, so the aim is real and not an artefact.

Policy shape, mean punishment per contribution bin:

| arm | {0} | 1-5 | 6-10 | 11-15 | 16-19 | {20} |
|---|---|---|---|---|---|---|
| clone | 4.69 | 3.04 | 1.82 | 1.06 | 0.79 | 0.37 |
| `capped_sigmoid` | 1.79 | 2.30 | 1.23 | 0.00 | 0.00 | 0.00 |
| **`llm_qwen3_8b`** | **2.59** | 0.98 | 0.58 | 0.41 | **0.61** | **0.40** |
| `never` | 0 | 0 | 0 | 0 | 0 | 0 |

**Told the rules and the objective and nothing else, the model does punish low contributors more than high ones** — the sign is the human one and nothing in the prompt says to aim that way. But it punishes **rarely and enormously**: it fires on 7.9% of decisions against the clone's 30.8%, and when it fires it issues **10.60** on average, the heaviest of any arm including the incumbent's flat 10. And it never stops: it still puts 0.40 on a full 20-point contributor, where both hand-written rules put exactly zero. The 16-19 bin (0.61) sits above the 11-15 bin (0.41), so the profile is not even monotone.

**The comparison with `capped_sigmoid` is the one that locates the deficit.** That rule spends 4.18 and the model spends 3.55 — almost the same budget — and the rule buys +5.66 contribution where the model buys none. The difference is not how much is spent but on whom. Decision-weighted, the share of each manager's total punishment that lands on contributors already giving 11 or more: **`llm_qwen3_8b` 17.9%, clone 17.4%, `capped_sigmoid` 0.0%**. The model wastes the same fraction the human-fitted clone does, and the clone also loses on the pool; what the winning rule does is spend nothing at all up there.

**Leaver diagnostic, as an ordering only** (`c_gap`, leavers' mean contribution minus stayers'). `thr9_p10` −3.53, `thr9_vs_thr9` −3.05, clone −2.28, `never_vs_never` −1.74, **`llm_qwen3_8b` −1.62**, `capped_sigmoid` −1.57, `never` −1.20. Only one of the eight neighbour gaps clears the 0.577 noise floor, so this is a rank and no sign test is made or can be.

## 6. Result: the model-size arm. 32B targets better, spends more, and lands in the same place

**Measured**, `plots/data_analysis/llm_manager_battery/qwen3-32b/`, 1,000 episodes per arm (5 seeds x 200). This is the comparison #220 could not obtain at all, and at a fifth of the 8B budget it is a reduced read: the pool MDD on this run's own spread is **5.63**, against 2.23 at 8B.

| quantity | reference | Qwen3-8B (6,000 ep) | Qwen3-32B (1,000 ep) |
|---|---|---|---|
| **pool** | never-punish | **−4.05 [−5.59, −2.51]** | **−4.58 [−8.46, −0.73]** |
| **pool** | clone | −0.81 [−2.35, +0.75] | −1.47 [−5.42, +2.55] |
| **pool** | `capped_sigmoid` | −8.93 [−10.48, −7.30] | −9.56 [−13.63, −5.73] |
| contribution | never-punish | **−0.31 [−1.24, +0.63]** | **+2.86 [+0.45, +5.27]** |
| contribution | clone | −2.85 [−3.77, −1.93] | +0.25 [−2.15, +2.70] |

**The two sizes reach the same pool by different routes, and the contribution column is what shows it.** 8B is level with never-punishing on contribution; 32B is **above** it, +2.86 with an interval excluding zero. So the larger model does raise collaboration — and then overpays for it:

| arm | realised spend | contribution bought | worth | **net on the pool** |
|---|---|---|---|---|
| `capped_sigmoid` | 4.21 | +5.74 | +9.19 | **+4.98** |
| `thr9_p10` | 11.02 | +6.83 | +10.93 | **−0.08** |
| clone | 7.28 | +2.60 | +4.16 | **−3.11** |
| **`llm_qwen3_32b`** | **9.15** | **+2.86** | **+4.57** | **−4.58** |

**32B is, within this run's resolution, the behavioural clone.** Pool −1.47 [−5.42, +2.55] and contribution +0.25 [−2.15, +2.70] against it: both intervals span zero. It spends 9.15 where the clone spends 7.28 and buys the same collaboration. The clone is a model fitted to what the human managers actually did, and it loses 3.11 pool points to never-punishing — so "managing like a human" is not the same as managing well on this objective, and the larger model has arrived at the former.

**Size buys targeting, not restraint.**

| | `rho` | `rho_rel` | punish rate | mean p when positive | {0} bin | {20} bin |
|---|---|---|---|---|---|---|
| Qwen3-8B | −0.162 | −0.353 | 0.079 | 10.60 | 2.59 | 0.40 |
| **Qwen3-32B** | **−0.343** | **−0.608** | 0.120 | **18.22** | **10.72** | 0.24 |
| clone | −0.339 | −0.414 | 0.308 | 5.95 | 4.73 | 0.35 |
| `capped_sigmoid` | −0.575 | −0.698 | 0.316 | 3.05 | 1.79 | 0.00 |

32B's `rho_rel` of −0.608 is most of the way to `capped_sigmoid`'s −0.698 and well past the clone's −0.414, and its profile falls monotonically from {0} to {20} but for one 0.09 blip. **It has learned whom to punish.** What it has not learned is how much: it puts **10.72** on the {0} bin against the human managers' 4.755 and the capped rule's 1.79, and averages 18.22 whenever it fires. The prompt's two sentences of accounting — that a punished point leaves the pool at full price — did not produce restraint at either size.

## 7. Measured versus inferred

### Measured

Everything here came out of a run on this branch and can be recomputed from the committed tables.

- The merge assertions: five prompt fingerprints unchanged with `v3_explicit_pool` = `b211d81b411a`; #221's run B reproduced across 75 float columns x 8 arms at a largest absolute difference of **0.0**; the decode constraint built per call at roster sizes 1 to 8, admitting the present roster and refusing an absent one.
- **Zero parse failures, zero call errors, zero truncations, 100% labelled form, in 134,272 calls at 8B and 22,432 at 32B.** Every answer came back in the form the prompt asked for. The counters close exactly: 134,272 answers + 9,728 empty-group rounds = 144,000 episode-rounds at 8B, and 22,432 + 1,568 = 24,000 at 32B.
- The 8B battery at 6,000 episodes per arm: pool 56.45 and contribution 37.50 for the model, against never 60.50 / 37.81, clone 57.26 / 40.35, `thr9_p10` 60.87 / 45.01, `capped_sigmoid` 65.38 / 43.47.
- The contrasts and their intervals, both columns, all four references, at both sizes, as tabled above.
- Targeting at both sizes, the tie structure behind each rank, and the six-bin policy shape.
- **The cheap regime held across 30 sequential rollouts.** 8B ran at 0.01199 s per episode-round (1,726 s over 144,000), against #222's 0.0111 measured on a single 200-episode rollout and 0.0258 at its asymptote — **2.15x cheaper than the asymptote**, against the 2.3x that branch predicted. 32B ran at 0.1325 s per episode-round against its measured 0.1174-0.1202, confirming there is no cheap regime there.
- A **seat bias in the `thr9` world**: `thr9_vs_thr9` gives the focal seat +3.38 [+1.36, +5.38] on the pool at 6,000 episodes here, where #221 measured −0.59 [−2.59, +1.33] at 6,144. The `clone` and `never` controls show none (+1.01 [−0.72, +2.75] and +0.68 [−0.90, +2.18]). This does not touch any contrast above, because every arm reported sits in the focal seat against the same clone rival, but it is a disagreement with a published table and is flagged rather than absorbed.

### Inferred

- **That either number is what "a language model" does.** It is what two Qwen3 checkpoints do at temperature 0 under one prompt version, `v3_explicit_pool`. A different prompt is a different experiment, which is why the fingerprint is on every table.
- **That the deficit is restraint rather than comprehension.** The evidence is a coincidence of spend (8B 3.55 against `capped_sigmoid`'s 4.18) with an opposite contribution outcome, plus 32B's good `rho_rel` at a ruinous level. That is consistent with "knows whom, not how much", and it is not the only story consistent with it — a manager that punishes rarely and hugely may also be failing to make its policy legible to the contributors, which is a different defect with the same signature.
- **That 32B "is" the clone.** Two intervals spanning zero at 1,000 episodes is weak evidence of sameness; the contribution interval alone is ±2.4 wide. What is measured is that this budget cannot tell them apart.
- **The power arithmetic.** The `sd` is measured on each run's own episodes; the minimum detectable difference is the standard normal approximation at 80% power, as #221 built it.
- **That the empty-group rounds are innocuous for the result.** The punishment row is zero either way, which is arithmetic, but 6.8% of 8B's episode-rounds and 6.5% of 32B's were rounds this seat did not act in, and nothing here checks whether an arm that empties more often is thereby advantaged.

## 8. What this does not show

1. **One prompt version, never varied here.** #220 chose `v3_explicit_pool` on replay evidence from a proxy model and left the pool axis (`v4`) unresolved on a contaminated collection. Nothing in this run re-opens that, and nothing here was tuned: the comparison set was fixed before the run and no prompt was touched.
2. **Temperature 0, one sample per decision.** The manager is reproducible and has no variance of its own; a sampled manager might behave differently and would need its own seed treatment.
3. **32B at 1,000 episodes is a reduced read.** Its pool intervals are 4 points wide either side. The +2.86 contribution result is the one that would most repay a longer run.
4. **The membership channel is not decomposed.** 8B holds 4.34 members and never-punishing holds 4.59; whether the pool gap runs through headcount or through per-member behaviour is #209's open item 4 and is untouched here.
5. **`get_punishments` still has not run inside a real job.** Every number came through `predict`. The `api_manager` seat remains tested only by unit tests.

## 9. Successor

1. **Run 32B at 6,000 episodes.** It is the one arm whose interesting result — contribution above never-punishing — sits at a fifth of the budget the 8B arm got. At 0.1325 s per episode-round that is a 5.3-hour job on four A100s, which is affordable and is the single highest-value follow-up.
2. **The restraint hypothesis is directly testable without touching the prompt.** `capped_sigmoid` is exactly the incumbent rule with its ceiling brought down. Capping the model's action at 10 or at 3 — a change to the decode constraint's `max_punishment`, not to the text — would say whether the deficit is the level or the aim. If a capped Qwen3-32B crosses into positive pool, the finding is "size buys targeting and the cap supplies the restraint"; if it does not, the aim is worse than `rho_rel` suggests.
3. **Re-measure the `thr9_vs_thr9` seat control.** Two runs at ~6,000 episodes disagree on it with non-overlapping intervals (+3.38 here, −0.59 in #221). One of them is wrong, or the episodes inside a rollout are correlated enough that both intervals are too narrow — which would matter for every contrast this project bootstraps over episodes.
4. **The `v6` prompt #220 deferred is now also a rollout concern.** An empty group produces a gap in the trace that the round numbers make visible but nothing explains, and this run measured that it happens on 6.8% of 8B's episode-rounds — a larger share than the 10.1% of replay decision points that merely had one *before* them.
5. **Carry `rho_rel` rather than the bare rank.** At 85.3% ties the 8B arm's raw `rho` understates its aim by a factor of two, exactly as #221 predicted a sparing manager would.
