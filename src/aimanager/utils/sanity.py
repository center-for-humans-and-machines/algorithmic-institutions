"""Sanity-check instrumentation for snapshot tests.

Toggled by the ``AIM_SANITY_LOG`` env var. When enabled, ``emit()`` writes
one ``[SANITY] {json}`` line per call. When disabled (the default),
``emit()`` returns immediately with no overhead beyond a single boolean
check.

If ``AIM_SANITY_LOG_FILE=<path>`` is also set, lines are appended to that
file (parent dir created if needed); otherwise they go to stdout.

Both env vars are read once at import time. The intended flow is:

1. A pipeline run on the cluster sets both env vars and writes a JSONL
   sanity log alongside its other artifacts.
2. The log is fetched back to local.
3. Local pytest reads the fetched log and diffs against a checked-in
   golden snapshot.
"""

import json
import os
import sys

_ENABLED = os.getenv("AIM_SANITY_LOG", "0") == "1"
_LOG_FILE = os.getenv("AIM_SANITY_LOG_FILE")


def _to_jsonable(value):
    """Coerce common ML container/scalar types into JSON-safe forms."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    return value


def emit(name: str, value, **ctx) -> None:
    """Write one ``[SANITY] {...}`` JSON line if AIM_SANITY_LOG=1, else no-op.

    Goes to ``AIM_SANITY_LOG_FILE`` if set, otherwise stdout.
    """
    if not _ENABLED:
        return
    payload = {"name": name, "value": _to_jsonable(value)}
    if ctx:
        payload["ctx"] = _to_jsonable(ctx)
    line = f"[SANITY] {json.dumps(payload, sort_keys=True)}"
    if _LOG_FILE:
        parent = os.path.dirname(_LOG_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(_LOG_FILE, "a") as f:
            f.write(line + "\n")
    else:
        print(line, flush=True)
