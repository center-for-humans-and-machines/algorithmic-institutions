# LLM manager: serving, client, and what it costs

The serving and client half of the language-model manager, and the throughput measurement that was meant to decide whether the project is feasible at all. It is, comfortably: a 3,000-episode rollout of Qwen3-8B takes 31 minutes on four A100s, and 6,144 episodes — the largest count the existing baselines use — is about an hour. Throughput is no longer the question. What the numbers now decide is how to spend the budget, and there the answer is less obvious than expected.

Everything under **Measured** was observed on Raven on 2026-09-22 with vLLM 0.19.1 and is reproducible from the committed artifacts. Everything under **Inferred** is arithmetic on those measurements and is marked as such. Nothing here was run at 32B beyond 1,000 episodes.

## What was built

`LLMManager` fills two existing seats. `predict(state) -> (punishment, None)` is the opponent seat of `rl_manager.run_batch`, which makes a 24-round rollout 24 batched calls rather than 24×B serial ones; `get_punishments(data)` is the `api_manager` seat, registered as manager type `llm`. The prompt, the trace format and the parser are not owned here: they are `aimanager.llm` (PR #220), injected exactly as that package's docstring specifies. This module's job is turning `served_state()` into a list of `trace.RoundRecord` and getting the batch to the server.

Serving is `scripts/llm_manager/serve_vllm.slurm.sh`: vLLM on the job's own compute node, so a run has no external dependency and no endpoint that can be reconfigured under it. The MPCDF hosted service was not used; it fixes its model at launch and is shared, and neither is worth the loss of reproducibility when launching on the node costs 201 seconds.

## Measured: the trap in the plan is real

Qwen3 turns on a `<think>` mode by default, and the plan said to verify rather than trust it. Verified, on Qwen3-8B with a 64-token budget:

| | finish_reason | completion tokens | content |
|---|---|---|---|
| default (no flag) | `length` | 64 | `<think>\nOkay, let's see. The user mentioned...` — no answer |
| `enable_thinking: false` | `stop` | 10 | `12,3,20,7` |

Two details the sibling note does not carry. The `<think>` text arrives in `content`, not in `reasoning_content`, unless the server is started with a reasoning parser — a client watching `reasoning_content` would not see it and would report an empty answer with no explanation. And the failure is total rather than partial: the budget is gone before any answer starts, so raising `max_tokens` trades cost for a chance rather than fixing it. Any model id containing `qwen3` now gets the flag off without being asked.

## Measured: 8B throughput, four servers on four A100s

Qwen3-8B, prompt `v3_explicit_pool`, constrained decode, temperature 0, 24 rounds, four independent vLLM servers with episodes sharded across them. Prompt grows 636 → 2,672 tokens across a rollout; completions are 34.5 tokens; **zero parse failures and zero truncations in 133,504 calls** (129,600 in the rollouts, 3,904 in the saturation sweep).

| episodes | wall clock | s per episode-round | completions/s | per card | round 0 | round 23 |
|---|---|---|---|---|---|---|
| 200 | 53.5 s | 0.0111 | 89.8 | 22.4 | 1.6 s | 2.7 s |
| 400 | 3.3 min | 0.0207 | 48.4 | 12.1 | 2.7 s | 18.9 s |
| 800 | 8.1 min | 0.0252 | 39.6 | 9.9 | 5.0 s | 37.7 s |
| 1,000 | 10.3 min | 0.0256 | 39.0 | 9.8 | 5.8 s | 47.1 s |
| 3,000 | 31.0 min | 0.0258 | 38.8 | 9.7 | 16.8 s | 140.5 s |

Peak in a single batch, at a realistic 12-round trace: **267 completions/s at width 1,024**, 67 per card, falling to 231 at 2,048.

## Measured: the limit is the KV cache, not concurrency

The interesting column is the last two. At 200 episodes the per-round wall clock barely moves — 1.6 s to 2.7 s — while the prompt nearly quadruples. At 3,000 it goes from 16.8 s to 140.5 s, tracking prompt length almost exactly.

That is vLLM's prefix cache holding and then failing. Prefix caching is on by default (`CacheConfig.enable_prefix_caching` is `True` in 0.19.1), and the trace grows by appending, so while the whole population's traces fit in GPU memory a round re-prefills only the one block that changed. Once they do not, every round re-prefills its entire trace and cost becomes linear in trace length.

The crossover is sharp and sits between 200 and 400 episodes on this hardware. Per episode-round: 0.0111 s at 200, 0.0207 at 400, 0.0252 at 800, and then flat — 0.0256 at 1,000 and 0.0258 at 3,000, an asymptote reached by 800.

**This is the number that should drive the budget, and it is not the saturation width.** At 200 episodes a decision costs 53.5 s / 4,800 = 0.0111 s, against 0.0258 s at the asymptote: the cheap regime is **2.3× cheaper per decision**. Above ~800 episodes the cost per episode is constant, so episodes are simply linear and there is no efficiency argument for any particular count — only a statistical one. The awkward region is 200–800, where each extra episode costs more than the last.

Two consequences worth stating plainly. A sweep of small runs is much cheaper than one large run for the same number of decisions: **five 200-episode rollouts give the same 24,000 decisions as one 1,000-episode rollout in 4.5 minutes against 10.3** — and they parallelise across jobs on top of that. Whether five 200-episode rollouts are statistically equivalent to one 1,000-episode rollout is a question for whoever owns the evaluation, not for this note; the seeds differ and nothing here checks that the pooled estimate has the same properties. And the 200-episode figure is a property of *this* trace length and *this* GPU count, not a constant — it will move with prompt version, rounds, and cards.

## Measured: 32B, two servers at tensor-parallel two

Qwen3-32B on the same four cards, two instances at TP=2. Derived from the per-call log rather than the benchmark's own JSON, because the 3,000-episode rollout was cancelled before the run could write it; the 1,000-episode row is 23,280 of 24,000 calls for the same reason, which the per-call normalisation handles.

| episodes | wall clock | s per episode-round | completions/s | per card |
|---|---|---|---|---|
| 200 | 9.6 min | 0.1202 | 8.3 | 2.08 |
| 1,000 | 45.5 min | 0.1174 | 8.5 | 2.13 |

**32B is 4.6× slower per episode-round than 8B, and it shows no cache benefit at all.** The 200-episode row is no cheaper per episode than the 1,000-episode row — 0.1202 against 0.1174 — because 64 GB of weights across two instances leaves too little KV cache to hold even a small population's traces. So the cheap regime that makes 8B attractive does not exist at 32B on four cards.

Parse failures and truncations: zero in both rows.

## Inferred: what the baselines' episode counts cost

Arithmetic on the asymptotic rate, not measured.

| episodes | 8B (0.0258 s) | 32B (0.1174 s) |
|---|---|---|
| 300 | 3.1 min | 14 min |
| 1,000 | 10.3 min (measured) | 45.5 min (measured) |
| 3,000 | 31.0 min (measured) | 2.3 h |
| 6,144 | 63 min | 4.8 h |

All of 8B is affordable. At 32B, 6,144 episodes is a five-hour job on four cards, which is a real cost but not a prohibitive one; 3,000 is comfortable. The honest recommendation is to iterate at 8B, confirm at 32B on a smaller episode count, and treat a result that appears only at 32B as worth a dedicated long run rather than as something to get for free.

## Measured: constrained decode, and the whitespace trap in it

The action is now constrained at the token level rather than parsed out of free text. The constraint is a regex built per call over the labels of the roster present at that decision point:

```
PUNISHMENT: Player 2 = 4, Player 3 = 0, Player 7 = 30
```

vLLM 0.19.1 takes this as `structured_outputs: {"regex": ...}` (older builds used a top-level `guided_regex`; both are supported). Across 133,504 calls at 8B and 28,080 at 32B, the parse failure rate was **zero**, which is what "by construction" is supposed to mean. The parser and its counter stay as a guard, and the counters are pinned by test to agree with `parse.summarise`.

**The plan's arity was wrong and the correction matters.** A group is four players in only 26% of human group-rounds; size runs 1 to 8. The constraint is therefore built from the roster, not from a fixed count, and binds each value to a named label, so a model cannot answer for a roster the decision point does not have. Verified against the real server at sizes 1, 2, 3, 5 and 8, and verified directly: asked to answer for Players 1–4 when the roster was Players 2, 3 and 5, Qwen3-8B returned `PUNISHMENT: Player 2 = 10, Player 3 = 10, Player 5 = 10`. A positional reader would have filed those three numbers against Players 1, 2 and 3.

**A trap inside the fix, found by the integration test and worth recording.** The separators were first written `\s*`, which is the natural thing and is wrong: whitespace is then always a legal next token, so a model with nothing to say can emit it forever and stay inside the grammar. Measured: asked to punish with no context, Qwen3-8B produced `PUNISHMENT:` followed by 128 tab tokens and stopped only at `finish_reason: length`, having answered nothing. Seven of twelve integration assertions failed this way — while the full-rollout assertions in the same run passed, because the real prompt gives the model something to say. That is what makes it the kind of bug that waits. The separators are now the literal single spaces the prompt's template shows, so the shortest legal continuation is an answer, and a unit test rejects any unbounded whitespace quantifier. vLLM exposes `disable_any_whitespace` for its JSON backends for the same reason.

## What the fallback costs, and why it is loud

An unparsable answer falls back to zero punishment, and zero punishment is the policy every collapsed learned manager in this project converged on. A silent fallback would push a result toward the outcome the comparison exists to distinguish, in proportion to how often it fires. So every failure logs, carries `fallback="zero"`, and is counted, and the rate is reported next to every result. Under a constraint it must be zero; a non-zero rate is a bug in the constraint or the server, not a property of the model.

## Decisions a reader might want to re-take

**Data parallel, not tensor parallel, for 8B.** Four independent servers rather than one four-way instance. The workload is throughput-bound, not memory-bound, and TP across a small model mostly buys communication overhead. Not measured against the TP=4 alternative — the choice was made on the coordinator's instruction and the numbers are consistent with it, but no head-to-head exists.

**Episodes are sharded by `episode % n_endpoints`, fixed and recorded.** Two instances of the same weights differ in sampling detail, so an episode that met a different server between rounds would turn that into episode-level noise indistinguishable from variance in the manager. The endpoint is written into every log line.

**The client speaks the OpenAI protocol from the standard library by default.** LiteLLM's `hosted_vllm/` provider is supported and wire-equivalent (`backend="litellm"`), but it is not a dependency of this project and installing it into the shared cluster venv would have disturbed other work running under the same account. The `hosted_vllm/` model prefix is accepted and stripped, so one config string serves both.

**Temperature is 0 and recorded**, which makes the manager reproducible — the one thing none of the sampling baselines are.

## Open, and what the next person should check

- **The 200-episode cheap regime was not exploited.** Nothing yet runs a sweep of small rollouts instead of one large one, and the arithmetic says it should.
- **No 32B run past 1,000 episodes.** The 3,000-episode row is arithmetic.
- **`get_punishments` (the `api_manager` seat) has unit tests but has never run inside a real `simulate` job.** The `predict` path is the one that has been exercised end to end.
- **The 32B numbers come from one cancelled job.** They are consistent across two rollout widths but were not repeated.
- **Prompt length drives everything**, so a prompt version longer than `v3_explicit_pool` moves the crossover down and every wall-clock figure up. The measurement should be re-run when the prompt version changes.

## Reproducing

```bash
# On Raven, from a checkout with PYTHONPATH set to its own src:
scripts/llm_manager/bench_raven.sh                      # 8B, 4 servers, 4 cards
MODEL=Qwen/Qwen3-32B TP=2 scripts/llm_manager/bench_raven.sh

# The integration test, against a real server:
sbatch --gres=gpu:a100:1 --cpus-per-task=18 --mem=120000 \
  scripts/llm_manager/serve_vllm.slurm.sh \
  ~/algorithmic-institutions/.venv/bin/python -m pytest \
  src/aimanager/tests/test_llm_integration.py -v
```

Raven refuses a job taking fewer GPUs than its share of node memory, so a one-card run must scale cores and memory down with it — roughly 18 cores and 120 GB per card.

Artifacts: `plots/data_analysis/llm_manager_throughput/`. The 8B file `bench-qwen3-8b-v3.json` carries per-round timings; `bench-qwen3-8b-crossover.json` is an earlier run on the placeholder prompt, kept because it locates the same crossover independently; `bench-qwen3-32b-derived.json` is reconstructed from the call log.
