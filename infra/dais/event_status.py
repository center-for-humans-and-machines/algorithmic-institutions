"""Did a tool call of an `opencode run --format json` event stream complete?

    python infra/dais/event_status.py <events.jsonl> <tool> <text in its input>

Exit 0 when a call of <tool> whose input contains <text> completed, 1 when
none did (denied, failed or never made); prints the calls it found.
"""

import json
import sys


def main(path, tool, text):
    ok = False
    for line in open(path, errors="replace"):
        try:
            part = json.loads(line).get("part") or {}
        except (json.JSONDecodeError, AttributeError):
            continue
        if part.get("type") != "tool" or part.get("tool") != tool:
            continue
        state = part.get("state") or {}
        if text not in json.dumps(state.get("input")):
            continue
        print(tool, state.get("status"), str(state.get("output") or state.get("error"))[:200])
        ok = ok or state.get("status") == "completed"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
