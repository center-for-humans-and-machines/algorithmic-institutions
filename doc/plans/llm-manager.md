# [DRAFT] An LLM as the manager

A manager whose punishment decisions come from a language model reading a
natural-language description of the game and an accumulating trace of play.

Base: `dev` (`fddf37a`), which carries the four timeout and free-punishment
fixes, the rule family and the paired competition harness.

## Why this is cheap to try

`src/aimanager/manager/api_manager.py` already defines the seat a manager
plugs into: a class with `get_punishments(data)`, dispatched by the simulation.
`src/aimanager/manager/paired_rollout.py` (PR #219) takes any object exposing
`predict(state) -> (punishment, None)`. An LLM manager is one more class
behind that interface; nothing in the environment changes.

The rollout harness is CPU-bound rather than GPU-bound, so the language model
is the only thing needing a GPU.

## Throughput, which sets every other choice

The environment normally runs 1000 episodes in parallel. A language model
cannot. But a round is one prompt per episode, and vLLM batches those, so a
rollout is **24 sequential batched calls**, not 24,000 serial ones.

At 50 episodes that is 24 batches of 50 completions. Feasible. At 1000 it is
not. **Episode count is the budget knob**, and it must be reported next to
every number, because the noise floor in this project scales with it and the
existing baselines were measured at 300 to 6,144 episodes.

## Serving

vLLM exposing an OpenAI-compatible endpoint, reached through LiteLLM's
`hosted_vllm/` provider. Two options, both used in the sibling project
`../understanding-billy` (`doc/vllm-backend.md`):

- `vllm serve Qwen/Qwen3-8B --port 8000` on the same compute node as the job,
  which keeps the run self-contained and has no external dependency;
- the MPCDF LLM Inference Service at <https://llm.mpcdf.mpg.de>, whose model is
  fixed at launch and whose URL needs a `/v1` suffix.

Prefer the first for reproducibility. **Qwen3 enables a `<think>` mode by
default**; the sibling project disables it with
`chat_template_kwargs={"enable_thinking": false}` and documents truncation
failures when it is left on.

Start at Qwen3-8B for iteration speed. The step to 32B is a config change.

## What the model is told

**The task.** The rules of the game in plain language: 24 rounds, four
contributors in its group, 20 points each per round, contributions pooled and
multiplied by 1.6 then split equally, punishment 0 to 30 deducted from both the
player's account and the pool, and reshuffling every fourth round.

**The objective, explicitly.** The manager is paid on its group's common pool.
This is what the real managers were paid on and what every learned manager here
was trained on.

**The trace.** An accumulating record, one block per round: each player's
contribution, what the manager punished them, whether they gave any input, and
the resulting pool. It grows across the episode and is the only memory.

**Its own group only**, four players, matching what the human managers saw.

**Not told** that the contributors are models rather than people. The framing
is the game, not the simulation.

## What it must return

Four integers in 0 to 30, one per player in its group, parsed strictly.

A player who gave no input cannot be punished. The environment enforces this
since the free-punishment fix, but the prompt states it and the trace marks
those players, so a wasted decision is visible in the log rather than silently
zeroed.

Every prompt and completion is logged. A parse failure is recorded and falls
back to zero punishment for that episode-round, and the failure rate is
reported. A run whose failure rate is material is not a result.

## Setting and baselines

The paired competing setting: the language model holds one group against the
behavioural clone in the other, members free to move. Self-play rankings do not
survive competition here, so self-play would not be informative.

Against: the clone, `thr9_p10`, never-punish, and the capped sigmoid rule from
PR #219.

## What gets measured

The standard battery, so the result drops into the existing tables:

- policy shape by contribution bin, on the evaluation suite's own bins, with
  `contribution_valid` masked at source;
- targeting as **three statistics together** — rank correlation, a magnitude
  column, and a noise gate — never the rank alone, and with the tie structure
  exposed, because a quiet manager ties more and ties attenuate the rank;
- the leaver diagnostic, as an ordering only, never as a sign test;
- total contribution, group pool, pool per member, mean punishment, members;
- prompt and completion token counts, wall clock, and the parse failure rate.

## The interface contract, so three agents can work in parallel

```python
class LLMManager:
    def __init__(self, *, model, api_base, api_key, prompt_version,
                 objective, n_punishments=31, **_): ...

    # Batched: one prompt per episode, one vLLM call per round.
    # state: the env's served_state() dict, tensors (B, A, T)
    # returns: (punishment int64 (B, A, 1), None)
    def predict(self, state, **_): ...

    # The api_manager seat, for the simulation dispatcher.
    def get_punishments(self, data): ...
```

Agent A owns serving and the client. Agent B owns the prompt, the trace format
and the parser, against a stub client. Agent C owns the evaluation, against a
stub manager that returns fixed punishments. All three build against this
signature.

## Risks

Throughput is the one that bites. If a rollout cannot reach a few hundred
episodes in reasonable wall clock, every comparison against the existing
baselines is underpowered and the honest report is about the budget rather than
about the model.

Determinism is the second. Fix the sampling temperature and record it. A
temperature of zero makes a run reproducible and removes a source of variance
this project already struggles with; sampling makes the manager stochastic in a
way none of the baselines are.
