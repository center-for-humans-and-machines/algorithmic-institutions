"""Run a policy-finder agent with OpenCode, inside its container (harness opencode).

agent_runner.py passes this file's source to the container as `python3 -c`, so
it runs with the container's environment: the endpoint's key in `--key-env`
(OPENROUTER_API_KEY: the spawner's token for project algoinst_oc) and the
egress proxy in HTTPS_PROXY. It needs nothing of Docker or the spawner: any
backend that starts it in the export with OpenCode installed can use it.

1. with `--usage-url` (OpenRouter's key endpoint), reads the key's usage and
   refuses to start once it is at `--max-usage` USD (the development cap;
   exit 3); without it (a local server), usage is not read;
2. runs `opencode run --agent policy-finder --format json` on the prompt in
   the repo (the export, whose .opencode/ holds the model and the agent),
   its JSON events to <out>/events.jsonl, its stderr to <out>/stderr.log;
3. exports the session (`opencode export`, the transcript with every
   message's tokens and cost) to <out>/session.json and copies OpenCode's log;
4. reads the key's usage again: the difference is what OpenRouter billed
   while the agent ran (other runs on the same key at the time included);
5. replaces the key wherever it was written (the out folder, the log, the
   version's write folders) and removes OpenCode's database, which holds the
   same transcript unredacted;
6. decides whether the run succeeded (`failures`): not when OpenCode exited
   non-zero, the agent did not load (OpenCode's fallback to its default
   agent), the session ended in an error, the final message has no text, or
   the session used another model than the export's config names; a failed
   run exits non-zero (4 when OpenCode itself exited 0);
7. prints one JSON line, the summary agent_runner.py reads: the report (the
   final assistant message's text), tokens, OpenCode's cost (its own price
   catalogue, not OpenRouter's bill), the models, the usage difference and
   the failures.

Usage (in the container):
    python3 -c "<source>" <prompt> --title T --out DIR --repo DIR
        [--write-dir REL ...] [--key-env NAME] [--usage-url URL [--max-usage USD]]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

KEY_ENV = "OPENROUTER_API_KEY"
#: OpenRouter's endpoint for a key's usage (agent_runner.py passes it)
KEY_URL = "https://openrouter.ai/api/v1/key"
REDACTED = "[redacted]"
#: OpenCode's data folder in the run's home (its database and logs)
DATA = os.path.expanduser("~/.local/share/opencode")


def key_usage(url, key_env=KEY_ENV):
    """The key's usage in USD so far, or None when it cannot be read."""
    key = os.environ.get(key_env, "")
    if not key or not url:
        return None
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return float(json.loads(r.read().decode())["data"]["usage"])
    except Exception:  # noqa: BLE001 - any failure: unknown
        return None


def session_id(events_path):
    """The session id of the first event that names one."""
    try:
        with open(events_path, errors="replace") as f:
            for line in f:
                try:
                    sid = json.loads(line).get("sessionID")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if sid:
                    return sid
    except OSError:
        pass
    return None


def summarize(session):
    """Report, tokens, cost and models of an exported session.

    The report is the final assistant message's text: an earlier message's
    text ("Let me look at ...") is not a report. `final_error` is the final
    assistant message's error, if it ended in one.
    """
    tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0,
              "cache_write": 0}
    cost, models, steps, tools, errors = 0.0, set(), 0, 0, []
    final = None
    for message in session.get("messages", []):
        info = message.get("info") or {}
        if info.get("role") != "assistant":
            continue
        final = message
        t = info.get("tokens") or {}
        cache = t.get("cache") or {}
        tokens["input"] += t.get("input") or 0
        tokens["output"] += t.get("output") or 0
        tokens["reasoning"] += t.get("reasoning") or 0
        tokens["cache_read"] += cache.get("read") or 0
        tokens["cache_write"] += cache.get("write") or 0
        cost += info.get("cost") or 0
        if info.get("modelID"):
            models.add(f"{info.get('providerID')}/{info['modelID']}")
        if info.get("error"):
            errors.append(json.dumps(info["error"])[:2000])
        parts = message.get("parts") or []
        steps += sum(1 for p in parts if p.get("type") == "step-finish")
        tools += sum(1 for p in parts if p.get("type") == "tool")
    report, final_error = "", None
    if final is not None:
        parts = final.get("parts") or []
        report = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        final_error = (final.get("info") or {}).get("error")
    return {"result": report, "tokens": tokens, "cost_catalog_usd": round(cost, 6),
            "models": sorted(models), "steps": steps, "tool_calls": tools,
            "errors": errors, "final_error": final_error,
            "agent": (session.get("info") or {}).get("agent")}


#: OpenCode's warning when the agent named with --agent does not load: it
#: then runs its default agent, without our definition
AGENT_FALLBACK = "Falling back to default agent"


def configured_model(repo):
    """The model the export's OpenCode config names (`<provider>/<model>`)."""
    try:
        with open(os.path.join(repo, ".opencode", "opencode.json")) as f:
            return json.load(f).get("model")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def last_event_error(events_path):
    """The error of an `error` event after the last finished step, if any."""
    error = None
    try:
        with open(events_path, errors="replace") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "error":
                    error = json.dumps(event.get("error"))[:2000]
                elif event.get("type") == "step_finish":
                    error = None
    except OSError:
        pass
    return error


def failures(summary, code, stderr, expected_model, event_error, agent="policy-finder"):
    """Why a run is not a success: [] when it is one."""
    out = []
    if code != 0:
        out.append(f"OpenCode exited with {code}")
    if AGENT_FALLBACK in stderr:
        out.append(f"the agent {agent} did not load: OpenCode fell back to its default")
    if summary.get("agent") and summary["agent"] != agent:
        out.append(f"the session ran agent {summary['agent']}, not {agent}")
    if summary.get("final_error"):
        out.append("the session ended in an error: "
                   + json.dumps(summary["final_error"])[:500])
    if event_error:
        out.append(f"OpenCode reported an error at the end: {event_error[:500]}")
    if not (summary.get("result") or "").strip():
        out.append("the final message has no text (no report)")
    if not expected_model:
        out.append("the export's OpenCode config names no model")
    elif summary.get("models") != [expected_model]:
        out.append(f"the session used {summary.get('models')}, not {expected_model}")
    return out


def redact_file(path, secret):
    """Replace `secret` in a text file; True when it was there."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return False
    needle = secret.encode()
    if not needle or needle not in data:
        return False
    with open(path, "wb") as f:
        f.write(data.replace(needle, REDACTED.encode()))
    return True


def redact_tree(root, secret):
    """Redact every file under `root`; return the paths that held it."""
    hits = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path) and redact_file(path, secret):
                hits.append(path)
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--write-dir", action="append", default=[])
    ap.add_argument("--key-env", default=KEY_ENV)
    ap.add_argument("--usage-url")
    ap.add_argument("--max-usage", type=float)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    summary = {"harness": "opencode"}
    before = key_usage(args.usage_url, args.key_env)
    summary["key_usage_before"] = before
    if args.usage_url and args.max_usage is not None and (
            before is None or before >= args.max_usage):
        summary.update(is_error=True, budget_exhausted=True, exit_code=3,
                       result="", error=f"key usage {before} USD is unknown or at "
                       f"the cap of {args.max_usage} USD: not started")
        print(json.dumps(summary), flush=True)
        return 3
    events = os.path.join(args.out, "events.jsonl")
    started = time.time()
    with open(events, "w") as out, open(os.path.join(args.out, "stderr.log"), "w") as err:
        code = subprocess.run(
            ["opencode", "run", "--agent", "policy-finder", "--format", "json",
             "--title", args.title, args.prompt],
            cwd=args.repo, stdout=out, stderr=err,
        ).returncode
    summary["seconds"] = round(time.time() - started)
    sid = session_id(events)
    summary["session_id"] = sid
    session_path = os.path.join(args.out, "session.json")
    if sid:
        with open(session_path, "w") as f:
            subprocess.run(["opencode", "export", sid], cwd=args.repo, stdout=f,
                           stderr=subprocess.DEVNULL)
    try:
        with open(session_path) as f:
            summary.update(summarize(json.load(f)))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        summary.update(result="", errors=["no session export"])
    for log in glob.glob(os.path.join(DATA, "log", "*.log")):
        shutil.copy(log, os.path.join(args.out, "opencode-" + os.path.basename(log)))
    after = None
    if args.usage_url:
        time.sleep(10)  # OpenRouter books a request's cost after it ends
        after = key_usage(args.usage_url, args.key_env)
    summary["key_usage_after"] = after
    if before is not None and after is not None:
        summary["key_usage_delta_usd"] = round(after - before, 6)
    secret = os.environ.get(args.key_env, "")
    hits = redact_tree(args.out, secret)
    for rel in args.write_dir:
        hits += redact_tree(os.path.join(args.repo, rel), secret)
    summary["key_redacted_in"] = hits
    for path in glob.glob(os.path.join(DATA, "opencode.db*")):
        os.remove(path)
    try:
        with open(os.path.join(args.out, "stderr.log"), errors="replace") as f:
            stderr = f.read()
    except OSError:
        stderr = ""
    why = failures(summary, code, stderr, configured_model(args.repo),
                   last_event_error(events))
    summary["opencode_exit_code"] = code
    summary["failures"] = why
    summary["is_error"] = bool(why)
    # a failed session fails the run, whatever OpenCode's own exit code
    code = code or (4 if why else 0)
    summary["exit_code"] = code
    print(json.dumps(summary), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
