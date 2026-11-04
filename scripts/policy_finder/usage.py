"""The Claude plan's usage limits, for the policy-finder agents.

The agents run on a plan's OAuth login (the token the container spawner gives
them), not on an API budget, so a usage or rate limit stops them like any
other work on that account. Two parts:

- `limited(record)`: whether a failed agent run stopped on a limit. The
  pipeline then does not count it as one of the version's attempts: it waits
  for the reset (`wait_after_limit`: the time the message names, else
  PF_USAGE_POLL_SECONDS, default 30 min) and starts the agent again.
- `gate(log)`: before an agent starts, wait while the account's 5-hour session
  or weekly window is at or above its ceiling (PF_USAGE_SESSION_MAX, default
  80; PF_USAGE_WEEKLY_MAX, default 90; percent). It reads the endpoint behind
  Claude Code's /usage with the token in PF_USAGE_CREDENTIALS (a
  .credentials.json or a file holding only the token). Unset, or unreadable,
  the gate is open: this container's own login can be another account than
  the agents'.
"""

import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.request

URL = "https://api.anthropic.com/api/oauth/usage"
#: The agents' login: a Claude Code .credentials.json or a file holding only
#: the token. Unset, the gate is off: this container's own login can belong to
#: another account than the one the spawner gives the agents.
CREDENTIALS = os.environ.get("PF_USAGE_CREDENTIALS", "")
SESSION_MAX = float(os.environ.get("PF_USAGE_SESSION_MAX", "80"))
WEEKLY_MAX = float(os.environ.get("PF_USAGE_WEEKLY_MAX", "90"))
#: Seconds between checks while waiting, and the wait after a limit whose
#: message names no reset time
POLL = int(os.environ.get("PF_USAGE_POLL_SECONDS", "1800"))
#: A reset time in a limit message: "...limit reached|1793802604"
RESET_EPOCH = re.compile(r"limit reached\|(\d{10})")
#: A run that hit a limit, by its result, report, stderr or transcripts
LIMIT = re.compile(
    r"usage limit|hit your (usage )?limit|limit reached|rate[ _]limit|"
    r"(status|error|code)\W{0,3}429|429\W{0,3}too many|out of (extra )?usage",
    re.I,
)


def read():
    """{"session": (percent, resets_at), "weekly": (...)}, or None."""
    if not CREDENTIALS:
        return None
    try:
        with open(CREDENTIALS) as f:
            text = f.read().strip()
        token = (json.loads(text)["claudeAiOauth"]["accessToken"]
                 if text.startswith("{") else text)
        req = urllib.request.Request(URL, headers={
            "Authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20",
        })
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
    except (OSError, KeyError, ValueError, urllib.error.URLError):
        return None
    out = {}
    for key, window in (("session", "five_hour"), ("weekly", "seven_day")):
        w = data.get(window) or {}
        if w.get("utilization") is not None:
            out[key] = (float(w["utilization"]), w.get("resets_at"))
    return out or None


def _seconds_until(stamp):
    try:
        when = dt.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return (when - dt.datetime.now(dt.timezone.utc)).total_seconds()


def over(usage):
    """The windows at or above their ceiling: [(name, percent, resets_at)]."""
    ceilings = {"session": SESSION_MAX, "weekly": WEEKLY_MAX}
    return [(k, p, r) for k, (p, r) in (usage or {}).items() if p >= ceilings[k]]


def describe(usage):
    if not usage:
        return "usage unknown"
    return ", ".join(f"{k} {p:.0f}% (resets {r})" for k, (p, r) in usage.items())


def gate(log=print, read_usage=None, sleep=time.sleep):
    """Return once no window is above its ceiling (unknown usage passes)."""
    read_usage = read_usage or read
    while True:
        usage = read_usage()
        full = over(usage)
        if not full:
            return usage
        resets = [s for s in (_seconds_until(r) for _, _, r in full) if s is not None]
        wait = max(60, min(POLL, max(resets) + 60)) if resets else POLL
        log(f"usage over the ceiling ({describe(usage)}): waiting {wait / 60:.0f} min")
        sleep(wait)


def reset_wait(texts, now=None):
    """Seconds until the reset a limit message names (plus a minute), or POLL."""
    now = time.time() if now is None else now
    epochs = [int(m) for t in texts for m in RESET_EPOCH.findall(t or "")]
    future = [e - now + 60 for e in epochs if e > now]
    return min(max(future), 6 * 3600) if future else POLL


def wait_after_limit(texts=(), log=print, read_usage=None, sleep=time.sleep):
    """After a run hit a limit: wait for its reset, then for the gate."""
    wait = reset_wait(texts)
    log(f"the agent hit a usage limit: waiting {wait / 60:.0f} min")
    sleep(wait)
    return gate(log, read_usage, sleep)


def _transcript_errors(home):
    """API error messages in a run home's transcripts."""
    errors = []
    if not home:
        return errors
    root = os.path.join(home, ".claude", "projects")
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            try:
                with open(os.path.join(dirpath, name), errors="replace") as f:
                    for line in f:
                        if '"isApiErrorMessage":true' in line.replace(" ", ""):
                            errors.append(line[:2000])
            except OSError:
                pass
    return errors


def limited(record, home=None):
    """Whether a failed agent run (agent_runner.finish's record) stopped on a
    usage or rate limit. Ask only about runs that failed: a working agent's
    report may mention anything."""
    result = record.get("result") or {}
    texts = [record.get("report") or "", record.get("stderr_tail") or "",
             str(record.get("error") or "")]
    if result.get("is_error") or record.get("status") != "succeeded":
        texts.append(json.dumps(result))
    texts += _transcript_errors(home)
    return [t for t in texts if LIMIT.search(t)]
