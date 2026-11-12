"""Preflight: one structured request per schema against the configured endpoint.

Run by a cluster job before the real stage. Bypasses the cache, reports per
schema whether the reply parsed, the latency and whether reasoning text leaked
(a `reasoning_content` field or a leading <think> block). Exit code 1 if any
schema fails.
"""

import time

from . import prompting as P
from .client import Abort, Call
from .common import log
from .schemas import (
    FreeCodes,
    NotesRecord,
    Proposal,
    assignment_model,
    judge_model,
)

SAMPLE_SETTING = "A sample setting: one group of 4 players, 24 rounds."
SAMPLE_RULE = """params:
  k:
    definition: punishment per point a player gives below the top contributor
    type: float
sweep_config:
  k: [0, 3]
code: |
  gap = c[..., -1].max(dim=1, keepdim=True).values - c[..., -1]
  punishment = k * gap
"""
SAMPLE_NOTES = """# sample

## Explorations

1. **Do low contributors respond to punishment?** Looked at the data. Players
   punished for a low contribution raised it by 2 points the next round.

## Key findings

1. Punished players raise their contribution (exploration 1).

## Hypothesis

Punish each player in proportion to the gap to the top contributor.

### Justification

Finding 1 supports punishing the gap; I chose not to model timeouts.
"""
CODES = [
    {"name": "gap_punishment", "label": "punish the gap", "definition": "x"},
    {"name": "late_stop", "label": "stop late", "definition": "y"},
]


def _calls():
    names = [c["name"] for c in CODES]
    shared = P.render_codebook(CODES)
    items = [("1", "punish below the top"), ("2", "punish the gap to the best")]
    version = {"stem": "s", "lineage": "s", "version": 1, "parent": None}
    notes_ctx = f"{SAMPLE_SETTING}\n\n{P.lineage_position(version)}"
    meta = {"doc": SAMPLE_RULE, "stem": "s", "codes": names}
    return [
        Call(
            "preflight",
            "open_rules",
            P.messages(P.load("open_rules"), SAMPLE_RULE, context=SAMPLE_SETTING),
            FreeCodes,
            "open_rules",
            meta,
        ),
        Call(
            "preflight",
            "open_notes",
            P.messages(P.load("open_notes"), SAMPLE_NOTES, context=notes_ctx),
            NotesRecord,
            "open_notes",
            {"doc": SAMPLE_NOTES, "stem": "s"},
        ),
        Call(
            "preflight",
            "propose",
            P.messages(P.load("what_propose"), P.render_items(items)),
            Proposal,
            "propose",
            {"kind": "what", "items": items, "round": 0},
        ),
        Call(
            "preflight",
            "assign",
            P.messages(P.load("assign"), items[0][1], shared),
            assignment_model(names),
            "assign",
            {"codes": names, "text": items[0][1]},
        ),
        Call(
            "preflight",
            "judge",
            P.messages(P.load("judge"), SAMPLE_RULE, shared, context=SAMPLE_SETTING),
            judge_model(names),
            "judge",
            meta,
        ),
    ]


def preflight(client):
    """Print one line per schema; return the number of failed schemas."""
    n_failed = 0
    log(f"preflight: model {client.model}, extra body {client.extra}")
    for call in _calls():
        t0 = time.time()
        try:
            res = client._one(call)
            error = res.error
            info = res.info
        except Abort as e:
            error, info = f"aborted: {e}", {}
        latency = info.get("latency_s", time.time() - t0)
        leaked = bool(info.get("reasoning_chars") or info.get("think_stripped"))
        ok = error is None
        n_failed += not ok
        print(
            f"{call.kind:<11} {'ok' if ok else 'FAILED':<7} {latency:7.2f}s  "
            f"reasoning leaked: {'yes' if leaked else 'no'}"
            + ("" if ok else f"  {error[:300]}"),
            flush=True,
        )
    print(f"preflight: {n_failed} of {len(_calls())} schemas failed", flush=True)
    return n_failed
