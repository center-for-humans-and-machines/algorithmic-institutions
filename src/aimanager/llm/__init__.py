"""The language-model manager's prompt, trace format and answer parser.

Three pieces, all source-agnostic, so the serving layer can inject them as
callables and the replay harness can drive them off the human CSV:

- `trace`: the accumulating record the model reads, built from plain
  `RoundRecord`s rather than from env tensors or a dataframe;
- `prompt`: immutable, named prompt versions; a result names the version and
  the version pins the text;
- `parse`: a strict reader of the model's answer that records a failure
  instead of guessing, kept as a guard behind constrained decoding.

## The boundary `LLMManager` plugs into

`LLMManager` (a sibling's) owns the client, the batching and the
`get_punishments(data)` seat in `manager/api_manager.py`. It injects two
callables from here and nothing else:

    from aimanager.llm.prompt import build_prompt
    from aimanager.llm.parse import enforce, mark_wasted, parse_punishments

    prompt = build_prompt(records, version=self.prompt_version)
    completion = self.client(prompt.messages)          # the sibling's
    result = mark_wasted(parse_punishments(completion, prompt.labels), target)
    charged = enforce(result, target)                  # what the env is given

`records` is a list of `trace.RoundRecord`, oldest first, whose last element
is the round being decided (every player's `punishment is None`). Building
those from the env's `served_state()` is the manager's job, but two rules are
not negotiable and `PlayerRound.from_masked` exists so they cannot be missed:

1. a player with `contribution_valid == False` must be passed
   `contribution_valid=False`, never their stored contribution -- the state
   carries the dataset median 9 there and the game used neither that nor the
   CSV's 0;
2. `punishment` on a past round must be the punishment the env CHARGED (zero
   on a no-input player), not the raw action, matching the free-punishment
   fix.

What to log per call, so a result can name what produced it:
`prompt.version`, `prompt.fingerprint`, `result.ok`, `result.reason`,
`result.form`, `result.wasted`, and the raw completion.
`parse.summarise(results)` is the single place the failure rate is computed.
"""
