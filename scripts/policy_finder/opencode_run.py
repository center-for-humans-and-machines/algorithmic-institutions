"""Run a policy-finder agent with OpenCode, inside its container (harness opencode).

agent_runner.py passes this file's source to the container as `python3 -c`, so
it runs with the container's environment: the endpoint's key in `--key-env`
(OPENROUTER_API_KEY: the spawner's token for project algoinst_oc) and the
egress proxy in HTTPS_PROXY. It needs nothing of Docker or the spawner: any
backend that starts it in the export with OpenCode installed can use it.

1. writes a salted hash of the key to <out>/../.key_hash.json, so that the
   pipeline can find the key in what it copies without being given it
   (agent_runner.scrub);
2. with `--usage-url` (OpenRouter's key endpoint), reads the key's usage
   and credit limit (three tries), warns when the key has no limit or they
   cannot be read, and refuses to start (exit 3) once the usage is at
   `--max-usage` USD, when the key is at its credit limit (nothing remaining)
   or, with `--require-key-limit`, when the key has no numeric limit;
   without it (a local server), nothing is read and nothing but the model
   server is called;
3. runs `opencode run --agent policy-finder --format json` on the prompt in
   the repo (the export, whose .opencode/ holds the model and the agent),
   its JSON events to <out>/events.jsonl, its stderr to <out>/stderr.log,
   killed with its process group after `--timeout` seconds (below the
   container's, so that this runner always gets to the redaction);
4. exports the session (`opencode export`, the transcript with every
   message's tokens and cost, `--export-timeout`) to <out>/session.json;
5. reads the key's usage again: the difference is what OpenRouter billed
   while the agent ran, every other run on the same key at the time
   included;
6. decides whether the run succeeded (`failures`): not when OpenCode exited
   non-zero or timed out, the agent did not load (OpenCode's fallback to its
   default agent), the session ended in an error, the final turn has no
   assistant text, or the session used another model than the export's
   config names;
   a failed run exits non-zero (4 when OpenCode itself exited 0);
7. in any case (`finally`): with `--kill-strays` (passed by the spawner's
   start path only, where every other process in the container is the
   agent's), kills what OpenCode's tools left running; copies OpenCode's
   logs to the out folder, replaces the key wherever this run wrote it (the
   out folder, the version's write folders; a read-only file is made
   writable first, and a file that still cannot be rewritten fails the run,
   so that its notes are not collected) and deletes OpenCode's logs and
   database in the home, which hold the same transcript unredacted;
8. prints one JSON line, the summary agent_runner.py reads, with the key
   redacted: the report (the final turn's last assistant text), tokens,
   OpenCode's cost (its own price catalogue, not OpenRouter's bill), the
   models, the usage difference and the failures.

What this protects against: accidental copies of the literal key. The
redaction is an exact byte match (a base64 copy, or one split by a space, is
not found), and nothing stops an agent from printing its environment into
the model's context; with the key and a shell, an agent can also call any
model through openrouter.ai, past the provider pin and the cap above. The
control is a dedicated key with a credit limit set at OpenRouter, which
unattended use requires.

Usage (in the container):
    python3 -c "<source>" <prompt> --title T --out DIR --repo DIR
        [--write-dir REL ...] [--key-env NAME] [--usage-url URL [--max-usage USD]]
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.request

KEY_ENV = "OPENROUTER_API_KEY"
#: OpenRouter's endpoint for a key's usage (agent_runner.py passes it)
KEY_URL = "https://openrouter.ai/api/v1/key"
REDACTED = "[redacted]"
#: The salted hash of the key, in the run's home beside the out folder
KEY_HASH = ".key_hash.json"
#: OpenCode's data folder in the run's home (its database and logs)
DATA = os.path.expanduser("~/.local/share/opencode")


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


def _texts(message, synthetic=None):
    """A message's text parts joined; `synthetic` True or False keeps only
    those parts (OpenCode marks the text it inserts itself)."""
    return "\n".join(
        p.get("text", "") for p in message.get("parts") or []
        if p.get("type") == "text"
        and (synthetic is None or bool(p.get("synthetic")) == synthetic))


def _compaction_summary(info):
    """OpenCode's summary of a compacted session (compaction.ts: an assistant
    message of the compaction agent, `summary: true`)."""
    return bool(info.get("summary")) or info.get("mode") == "compaction" or (
        info.get("agent") == "compaction")


def _new_prompt(message, prompts):
    """Whether a user message starts a turn: not one OpenCode inserts when it
    compacts a session (compaction.ts, 1.19.6): the marker with a
    `compaction` part, the "Continue if you have next steps" message (its
    text part `synthetic`), or the replay of an earlier prompt after an
    overflow (a copy of its parts, unmarked)."""
    parts = message.get("parts") or []
    if any(p.get("type") == "compaction" for p in parts):
        return False
    text = _texts(message, synthetic=False)
    if not text.strip():
        return False
    return text not in prompts


def _ending(turn):
    """How the final turn ended: `text` (its last message has text),
    `empty_after_tool_call`, `tool_call`, `error`, `empty` or `none`."""
    if not turn:
        return "none"
    last = turn[-1]
    if (last.get("info") or {}).get("error"):
        return "error"
    if _texts(last).strip():
        return "text"
    if any(p.get("type") == "tool" for p in last.get("parts") or []):
        return "tool_call"
    if len(turn) > 1 and any(p.get("type") == "tool" for p in turn[-2].get("parts") or []):
        return "empty_after_tool_call"
    return "empty"


def summarize(session):
    """Report, tokens, cost and models of an exported session.

    The report is the last non-empty assistant text of the final turn (the
    assistant messages after the last prompt): a model may end on an empty
    message after a tool call, and its report is the text before it. The
    messages OpenCode inserts when it compacts a long session (the marker,
    the summary, the "Continue" prompt or a replay of the prompt) neither
    start a turn nor are a report. `final_error` is the final assistant
    message's error, if it ended in one (the summary's too: a compaction that
    failed); an interim text ("Let me look at ...") followed by an error is no
    report. `ended_on` and `report_chars` record how the turn ended, so that
    an interim text taken for a report can be found afterwards.
    """
    tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0,
              "cache_write": 0}
    cost, models, steps, tools, errors = 0.0, set(), 0, 0, []
    final, turn, prompts, compactions = None, [], set(), 0
    for message in session.get("messages", []):
        info = message.get("info") or {}
        if info.get("role") == "user":
            if _new_prompt(message, prompts):
                prompts.add(_texts(message, synthetic=False))
                turn = []
            continue
        if info.get("role") != "assistant":
            continue
        final = message
        if _compaction_summary(info):
            compactions += 1
        else:
            turn.append(message)
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
    for message in turn:
        text = _texts(message)
        if text.strip():
            report = text
    if final is not None:
        final_error = (final.get("info") or {}).get("error")
    return {"result": report, "tokens": tokens, "cost_catalog_usd": round(cost, 6),
            "models": sorted(models), "steps": steps, "tool_calls": tools,
            "errors": errors, "final_error": final_error,
            "ended_on": _ending(turn), "report_chars": len(report.strip()),
            "compactions": compactions,
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
        out.append("no assistant text in the final turn (no report)")
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
    with open(path, "wb") as f:  # an OSError here: the caller records it
        f.write(data.replace(needle, REDACTED.encode()))
    return True


def redact_text(text, secret):
    return text.replace(secret, REDACTED) if secret else text


def redact_tree(root, secret, errors=None):
    """Redact every file under `root`; return the paths that held it. A file
    that cannot be rewritten is listed in `errors` and the walk goes on."""
    hits = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            try:
                if redact_file(path, secret):
                    hits.append(path)
            except OSError as e:
                if errors is not None:
                    errors.append(f"{path}: {e.strerror or e}")
    return hits


def key_info(url, key_env=KEY_ENV, tries=3, wait=5):
    """The key's `usage`, `limit` and `limit_remaining` (USD), or None after
    `tries` failed reads (a network blip is not an unknown usage)."""
    key = os.environ.get(key_env, "")
    if not key or not url:
        return None
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())["data"]
            return {"usage": float(data["usage"]), "limit": data.get("limit"),
                    "limit_remaining": data.get("limit_remaining")}
        except Exception:  # noqa: BLE001 - any failure: try again, then unknown
            if attempt + 1 < tries:
                time.sleep(wait)
    return None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def own_pid_namespace(proc="/proc", ppid=None):
    """The guard of kill_strays: this runner is the direct child of PID 1 and
    PID 1 is docker's init (docker-init or tini, the spawner's `--init`).
    The guard alone does not decide that every other process is the agent's:
    kill_strays also needs --kill-strays, which only the spawner passes."""
    try:
        with open(os.path.join(proc, "1", "cmdline"), "rb") as f:
            init = f.read().split(b"\0")[0].decode(errors="replace")
    except OSError:
        return False
    ppid = os.getppid() if ppid is None else ppid
    return ppid == 1 and os.path.basename(init) in ("docker-init", "tini")


def kill_strays(enabled, proc="/proc", kill=os.kill, wait=2.0, ppid=None):
    """Kill what OpenCode's tools left running (a child in its own session
    survives the kill of the process group), when `enabled` (--kill-strays:
    the spawner's container, where every other process is the agent's) and
    the guard matches. Waits up to `wait` seconds for them to go.

    Returns {"enabled", "guard_matched", "killed"}: `killed` 0 with
    `enabled` False means it never looked."""
    out = {"enabled": bool(enabled), "guard_matched": False, "killed": 0}
    if not enabled:
        return out
    out["guard_matched"] = own_pid_namespace(proc, ppid)
    if not out["guard_matched"]:
        return out
    me, victims = os.getpid(), []
    for entry in os.listdir(proc):
        if not entry.isdigit() or int(entry) in (1, me):
            continue
        try:
            kill(int(entry), signal.SIGKILL)
            victims.append(entry)
        except OSError:
            pass
    out["killed"] = len(victims)
    deadline = time.time() + wait
    while victims and time.time() < deadline:
        victims = [v for v in victims if os.path.exists(os.path.join(proc, v))]
        if victims:
            time.sleep(0.1)
    return out


def key_hash(secret, salt=None):
    """A salted hash of the key: lets the pipeline find the key in what it
    copies (agent_runner.scrub) without being given the key."""
    salt = salt or os.urandom(16).hex()
    digest = hashlib.sha256((salt + secret).encode()).hexdigest()
    return {"salt": salt, "sha256": digest, "length": len(secret)}


def run_bounded(argv, timeout, **kwargs):
    """Run `argv` in its own process group; kill the group at `timeout`.

    Returns the exit code, or None when it was killed.
    """
    proc = subprocess.Popen(argv, start_new_session=True, **kwargs)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return None


def scrub(args, secret, summary):
    """Redact the key everywhere this run wrote it, copy and drop OpenCode's
    own logs and database. Runs whatever happened before (`finally`)."""
    summary["stray_processes"] = kill_strays(args.kill_strays)
    for log in glob.glob(os.path.join(DATA, "log", "*.log")):
        try:
            shutil.copy(log, os.path.join(args.out, "opencode-" + os.path.basename(log)))
        except OSError:
            pass
    errors = []
    hits = redact_tree(args.out, secret, errors)
    for rel in args.write_dir:
        hits += redact_tree(os.path.join(args.repo, rel), secret, errors)
    remaining = []
    for error in errors:
        path = error.rsplit(": ", 1)[0]
        try:
            # a file the agent made read-only: make it writable, then again
            os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)
            if redact_file(path, secret):
                hits.append(path)
        except OSError as e:
            remaining.append(f"{path}: {e.strerror or e}")
    summary["key_redacted_in"] = hits
    if remaining:
        summary["redaction_errors"] = remaining
    for path in glob.glob(os.path.join(DATA, "opencode.db*")) + glob.glob(
            os.path.join(DATA, "log", "*")):
        try:
            os.remove(path)
        except OSError:
            pass


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
    ap.add_argument("--require-key-limit", action="store_true",
                    help="refuse to start when the key has no credit limit")
    ap.add_argument("--timeout", type=float, default=6600,
                    help="seconds for `opencode run`, below the container's")
    ap.add_argument("--export-timeout", type=float, default=300)
    ap.add_argument("--kill-strays", action="store_true",
                    help="kill every other process after OpenCode (the spawner's "
                    "container only, where all of them are the agent's)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    secret = os.environ.get(args.key_env, "")
    if secret:
        # beside the transcripts, not among them: the pipeline's scrub reads it
        with open(os.path.join(os.path.dirname(os.path.abspath(args.out)),
                               KEY_HASH), "w") as f:
            json.dump(key_hash(secret), f)
    summary = {"harness": "opencode"}
    info = key_info(args.usage_url, args.key_env)
    before = info and info["usage"]
    summary["key_usage_before"] = before
    if args.usage_url:
        summary["key_limit"] = info and info["limit"]
        summary["key_limit_remaining"] = info and info["limit_remaining"]
        if info is None:
            summary["warnings"] = ["the key's usage and credit limit could not be "
                                   "read from OpenRouter"]
        elif info["limit"] is None:
            summary["warnings"] = ["the key has no credit limit at OpenRouter: a "
                                   "dedicated key with a limit is required for "
                                   "unattended use"]
    stop = None
    remaining = info and info["limit_remaining"]
    if args.usage_url and args.max_usage is not None and (
            before is None or before >= args.max_usage):
        stop = (f"key usage {before} USD is unknown or at the cap of "
                f"{args.max_usage} USD: not started")
    elif args.usage_url and _number(remaining) and remaining <= 0:
        stop = (f"the key is at its credit limit at OpenRouter ({remaining} USD "
                "remaining): not started")
    elif args.usage_url and args.require_key_limit and (
            not info or not _number(info["limit"])):
        stop = ("the key has no numeric credit limit at OpenRouter (or it is "
                "unknown): not started")
    if stop:
        summary.update(is_error=True, budget_exhausted=True, exit_code=3, result="",
                       error=stop)
        print(redact_text(json.dumps(summary), secret), flush=True)
        return 3
    events = os.path.join(args.out, "events.jsonl")
    code = 1
    try:
        started = time.time()
        with open(events, "w") as out, open(os.path.join(args.out, "stderr.log"),
                                            "w") as err:
            code = run_bounded(
                ["opencode", "run", "--agent", "policy-finder", "--format", "json",
                 "--title", args.title, args.prompt],
                args.timeout, cwd=args.repo, stdout=out, stderr=err)
        summary["seconds"] = round(time.time() - started)
        if code is None:
            summary["timed_out"] = True
            code = 124
        sid = session_id(events)
        summary["session_id"] = sid
        session_path = os.path.join(args.out, "session.json")
        if sid:
            with open(session_path, "w") as f:
                run_bounded(["opencode", "export", sid], args.export_timeout,
                            cwd=args.repo, stdout=f, stderr=subprocess.DEVNULL)
        try:
            with open(session_path) as f:
                summary.update(summarize(json.load(f)))
        except (OSError, ValueError, AttributeError, TypeError):
            summary.update(result="", errors=["no session export"])
        if args.usage_url:
            time.sleep(10)  # OpenRouter books a request's cost after it ends
            after = key_info(args.usage_url, args.key_env)
            summary["key_usage_after"] = after and after["usage"]
            if before is not None and after is not None:
                # every other run on the key in the meantime included
                summary["key_usage_delta_usd"] = round(after["usage"] - before, 6)
        try:
            with open(os.path.join(args.out, "stderr.log"), errors="replace") as f:
                stderr = f.read()
        except OSError:
            stderr = ""
        why = failures(summary, code, stderr, configured_model(args.repo),
                       last_event_error(events))
    except Exception as e:  # noqa: BLE001 - the runner's own failure fails the run
        why = [f"the runner failed: {type(e).__name__}: {e}"]
        code = code or 1
    finally:
        scrub(args, secret, summary)
    if summary.get("redaction_errors"):
        # the key may remain in a write folder: its notes are not collected
        why = [*why, "the key could not be redacted in: "
               + "; ".join(summary["redaction_errors"])[:500]]
    summary["opencode_exit_code"] = code
    summary["failures"] = why
    summary["is_error"] = bool(why)
    # a failed session fails the run, whatever OpenCode's own exit code
    code = code or (4 if why else 0)
    summary["exit_code"] = code
    print(redact_text(json.dumps(summary), secret), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
