"""Print an `opencode run --format json` event stream briefly: tool calls with
their input and output, text, and each step's tokens.

    python infra/dais/agent_events.py <events.jsonl>
"""

import json
import sys


def main(path):
    for line in open(path, errors="replace"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        if part.get("type") == "tool":
            state = part.get("state") or {}
            print("     tool |", part.get("tool"), json.dumps(state.get("input"))[:200])
            out = str(state.get("output"))[:600].replace("\n", "\n     out  | ")
            print("     out  |", out)
        elif part.get("type") == "text":
            print("     text |", part.get("text", "")[:800])
        elif part.get("type") == "step-finish":
            print("     tokens |", json.dumps(part.get("tokens")))


if __name__ == "__main__":
    main(sys.argv[1])
