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
6. prints one JSON line, the summary agent_runner.py reads: the report (the
   last assistant message's text), tokens, OpenCode's cost (its own price
   catalogue, not OpenRouter's bill), the models, and the usage difference.

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
    """Report, tokens, cost and models of an exported session."""
    tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0,
              "cache_write": 0}
    cost, models, steps, tools, report, errors = 0.0, set(), 0, 0, "", []
    for message in session.get("messages", []):
        info = message.get("info") or {}
        if info.get("role") != "assistant":
            continue
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
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if text.strip():
            report = text
    return {"result": report, "tokens": tokens, "cost_catalog_usd": round(cost, 6),
            "models": sorted(models), "steps": steps, "tool_calls": tools,
            "errors": errors}


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
    except (OSError, json.JSONDecodeError):
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
    summary["exit_code"] = code
    summary["is_error"] = code != 0 or not summary.get("result")
    print(json.dumps(summary), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
