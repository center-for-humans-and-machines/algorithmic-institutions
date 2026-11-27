"""Run a policy-finder agent in its own container, through the container spawner.

The container is the only sandbox (no Claude Code permissions, sandbox or
hooks inside; the spawner starts `claude` with --dangerously-skip-permissions).
It mounts the instance's export (export_instance.py) read-only at the repo
path, with the version's write folders read-write on top, and has no network
but api.anthropic.com. It cannot reach the command server, the spawner or the
proxy container.

`run_agent` starts `claude -p <prompt> --agent policy-finder-host` there,
waits for it, and returns the run: its status, Claude's JSON result (the
report in `result`, cost and token usage) and the models the session used,
read from its transcripts in the run's home (kept by the spawner, mounted at
SPAWNER_STATE). At most PF_AGENT_SLOTS agents run at once across all
pipelines (lock files under `<exports>/../locks`), below the spawner's own cap.

That is harness `claude`, the default. An instance whose settings name
harness `opencode` runs OpenCode instead (`opencode run --agent policy-finder`,
no relay), through a second spawner (PF_SPAWNER_OC, project algoinst_oc)
whose token is the OpenRouter key and whose only egress is openrouter.ai.
The command is opencode_run.py, passed as `python3 -c`: it runs OpenCode,
exports the session and prints a summary line, which takes the place of
Claude's JSON result. Its transcripts are the files it writes to `opencode/`
in the run's home (the spawner's state for that project, PF_SPAWNER_STATE_OC).
The key: unattended use requires a dedicated OpenRouter key with a credit
limit set at OpenRouter. The runner redacts accidental copies of the
literal key, but an agent can read its environment, and with the key reach
any model through openrouter.ai. PF_OPENROUTER_REQUIRE_LIMIT=1 refuses a key
without a credit limit (else the runner warns); the cap below is a check
before each start, not a limit on a running agent (runner_options).
PF_OPENROUTER_MAX_USAGE (USD, the key's total usage) stops an opencode agent
before it starts once the key is at it.
"""

import contextlib
import fcntl
import glob
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

URL = os.environ.get("PF_SPAWNER", "http://container_spawner:8877")
#: The spawner's state dir as mounted here (run homes, saved logs)
SPAWNER_STATE = os.environ.get("PF_SPAWNER_STATE", "/var/spawner")
PROFILE = "cpu"
REPO = "/opt/algorithmic-institutions"
#: The run's home in the container (the profile's `home`)
HOME = "/home/claude"
SLOTS = int(os.environ.get("PF_AGENT_SLOTS", "4"))
CPUS = float(os.environ.get("PF_AGENT_CPUS", "2"))
MEMORY_MB = int(os.environ.get("PF_AGENT_MEMORY_MB", "4096"))
TIMEOUT = int(os.environ.get("PF_AGENT_TIMEOUT", "7200"))
POLL = 30
TERMINAL = ("succeeded", "failed", "cancelled", "expired", "error")
HARNESSES = ("claude", "opencode")
#: The opencode harness's spawner and its state as mounted here
URL_OC = os.environ.get("PF_SPAWNER_OC", "http://container_spawner_oc:8877")
SPAWNER_STATE_OC = os.environ.get("PF_SPAWNER_STATE_OC", f"{SPAWNER_STATE}/oc")
#: The in-container runner of harness opencode, and where it writes in the home
OPENCODE_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opencode_run.py")
OPENCODE_OUT = "opencode"
MAX_USAGE = os.environ.get("PF_OPENROUTER_MAX_USAGE")
#: Refuse to start an opencode agent whose OpenRouter key has no credit limit
REQUIRE_KEY_LIMIT = os.environ.get("PF_OPENROUTER_REQUIRE_LIMIT", "") not in ("", "0")
#: The runner's salted hash of the key, in the run's home (opencode_run.py)
KEY_HASH = ".key_hash.json"


class AgentError(Exception):
    pass


class BudgetError(AgentError):
    """An opencode run that did not start: the key is at PF_OPENROUTER_MAX_USAGE."""


def _url(harness):
    if harness not in HARNESSES:
        raise AgentError(f"unknown harness {harness!r}; harnesses: {HARNESSES}")
    return URL_OC if harness == "opencode" else URL


def _state(harness):
    return SPAWNER_STATE_OC if harness == "opencode" else SPAWNER_STATE


def run_home(run_id, harness="claude"):
    """The run's home as mounted here."""
    return os.path.join(_state(harness), "runs", run_id, "home")


def _request(method, path, body=None, timeout=120, harness="claude"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        _url(harness) + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise AgentError(f"{method} {path}: HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise AgentError(f"{method} {path}: {e.reason}") from None


@contextlib.contextmanager
def slot(lock_dir, slots=None):
    """Hold one of `slots` agent slots (flock; freed when the process dies)."""
    slots = SLOTS if slots is None else slots
    os.makedirs(lock_dir, exist_ok=True)
    while True:
        for i in range(slots):
            handle = open(os.path.join(lock_dir, f"agent-{i}.lock"), "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield i
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(20)


def mounts(stem, write_dirs):
    """The export read-only at the repo path, its write folders read-write.

    Paths are relative to the spawner's `pf_exports` root, which must be the
    exports folder the pipeline builds into (PF_RUNS/exports).
    """
    out = [{"root": "pf_exports", "path": stem, "target": REPO, "read_only": True}]
    for d in write_dirs:
        out.append({
            "root": "pf_exports",
            "path": f"{stem}/{d}",
            "target": f"{REPO}/{d}",
            "read_only": False,
        })
    return out


def command(prompt, harness="claude", title="policy-finder", write_dirs=()):
    if harness == "opencode":
        with open(OPENCODE_RUN) as f:
            source = f.read()
        out = ["python3", "-c", source, prompt, "--title", title,
               "--out", f"{HOME}/{OPENCODE_OUT}", "--repo", REPO,
               "--key-env", "OPENROUTER_API_KEY",
               "--usage-url", "https://openrouter.ai/api/v1/key"]
        for d in write_dirs:
            out += ["--write-dir", d]
        if MAX_USAGE:
            out += ["--max-usage", MAX_USAGE]
        return out
    return [
        "claude", "-p", prompt,
        "--agent", "policy-finder-host",
        "--strict-mcp-config",
        "--output-format", "json",
    ]


def runner_options():
    """opencode_run.py's own limits, appended to `command` by a backend:
    OpenCode stops before the container's timeout (so the runner redacts),
    and the key-limit requirement."""
    out = ["--timeout", str(max(TIMEOUT - 600, 60))]
    if REQUIRE_KEY_LIMIT:
        out.append("--require-key-limit")
    return out


def _key_matcher(home):
    """A function finding the key in a text, from the runner's salted hash in
    the run's home, or None without one."""
    try:
        with open(os.path.join(home, KEY_HASH)) as f:
            info = json.load(f)
        salt, digest, length = info["salt"], info["sha256"], int(info["length"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if length < 16:
        return None
    token = re.compile(r"[A-Za-z0-9_.:+/=-]{%d,}" % length)

    def find(text):
        found = set()
        for match in token.finditer(text):
            run = match.group(0)
            # every window, also of a long run (a base64 blob): about a
            # microsecond per character
            for i in range(len(run) - length + 1):
                piece = run[i:i + length]
                if hashlib.sha256((salt + piece).encode()).hexdigest() == digest:
                    found.add(piece)
        return found
    return find


def scrub(run_id, harness, paths=(), texts=(), log=None):
    """Redact the key (found by the runner's salted hash) in `paths`, in the
    run's own opencode/ folder, and in `texts` (returned redacted); delete
    OpenCode's database and logs in the run's home. For a run whose runner
    did not get to its own redaction (killed with its container). Returns
    (redacted texts, paths that held the key)."""
    home = run_home(run_id, harness)
    find = _key_matcher(home)
    hits = []
    if harness == "opencode":
        data = os.path.join(home, ".local", "share", "opencode")
        for path in glob.glob(os.path.join(data, "opencode.db*")) + glob.glob(
                os.path.join(data, "log", "*")):
            try:
                os.remove(path)
            except OSError:
                pass
    if find is None:
        if harness == "opencode" and log:
            log(f"no key hash in the run home of {run_id}: its transcripts were not "
                "scanned for the key (the runner did not start?)")
        return list(texts), hits
    own = transcripts(run_id, harness)[1] if harness == "opencode" else []
    for path in [*paths, *own]:
        try:
            with open(path, "rb") as f:
                text = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        keys = find(text)
        if keys:
            for key in keys:
                text = text.replace(key, "[redacted]")
            with open(path, "w") as f:
                f.write(text)
            hits.append(str(path))
    out = []
    for text in texts:
        for key in find(text):
            text = text.replace(key, "[redacted]")
        out.append(text)
    return out, hits


def transcripts(run_id, harness="claude"):
    """The run's transcript files: (folder, [paths under it])."""
    home = run_home(run_id, harness)
    if harness == "opencode":
        source = os.path.join(home, OPENCODE_OUT)
        pattern = os.path.join(source, "*")
    else:
        source = os.path.join(home, ".claude", "projects")
        pattern = os.path.join(source, "**", "*.jsonl")
    return source, sorted(p for p in glob.glob(pattern, recursive=True)
                          if os.path.isfile(p))


def models(run_id):
    """The models the run's Claude sessions called, from their transcripts."""
    found = set()
    home = run_home(run_id)
    for path in glob.glob(os.path.join(home, ".claude", "projects", "**", "*.jsonl"),
                          recursive=True):
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    if '"model"' not in line:
                        continue
                    try:
                        msg = json.loads(line).get("message") or {}
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and msg.get("model"):
                        found.add(msg["model"])
        except OSError:
            continue
    return sorted(found - {"<synthetic>"})


def parse_result(stdout):
    """Claude's --output-format json result: the last JSON object in stdout."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def start(stem, write_dirs, prompt, harness="claude"):
    """Start the agent on the export `<stem>`; return the spawner's run id.

    Waits while the spawner is at its cap (HTTP 409).
    """
    body = {
        "profile": PROFILE,
        # --kill-strays here only: in the spawner's container every other
        # process is the agent's (not in command() or runner_options(), which
        # other backends reuse)
        "command": command(prompt, harness, stem, write_dirs) + (
            [*runner_options(), "--kill-strays"] if harness == "opencode" else []),
        "workdir": REPO,
        "mounts": mounts(stem, write_dirs),
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": TIMEOUT,
    }
    while True:
        try:
            return _request("POST", "/runs", body, harness=harness)["id"]
        except AgentError as e:
            if "HTTP 409" not in str(e):
                raise
            time.sleep(60)


def finish(run_id, log=print, harness="claude"):
    """Wait for the run; return its record (status, report, cost, models).

    Raises BudgetError when an opencode run did not start because the key is
    at PF_OPENROUTER_MAX_USAGE (or has no credit limit, with
    PF_OPENROUTER_REQUIRE_LIMIT): not the agent's failure. The pipeline then
    stops the version (status `stopped`, resumable once there is budget) and
    the campaign stops.
    """
    started = time.time()
    run = _request("GET", f"/runs/{run_id}", harness=harness)
    while run["status"] not in TERMINAL:
        time.sleep(POLL)
        try:
            run = _request("GET", f"/runs/{run_id}", harness=harness)
        except AgentError as e:
            log(f"poll of run {run_id} failed: {e}")
    logs = _request("GET", f"/runs/{run_id}/logs?tail=5000", harness=harness)
    result = parse_result(logs.get("stdout", ""))
    if (result or {}).get("budget_exhausted"):
        raise BudgetError(f"agent run {run_id}: {result.get('error')}")
    status, error = run["status"], run.get("error")
    if harness == "opencode":
        # the runner's verdict (opencode_run.failures), or no summary at all
        why = (result or {}).get("failures") or (
            [] if result else ["no summary from the runner"])
        if status == "succeeded" and (result is None or result.get("is_error")):
            status = "failed"
        if why and status != "succeeded":
            error = "; ".join([*([error] if error else []), *why])
    return {
        "run_id": run_id,
        "status": status,
        "exit_code": run.get("exit_code"),
        "error": error,
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "waited_seconds": round(time.time() - started),
        "home_host_path": run.get("home_host_path"),
        "result": {k: v for k, v in (result or {}).items() if k != "result"},
        "report": (result or {}).get("result") or "",
        "models": ((result or {}).get("models") or [] if harness == "opencode"
                   else models(run_id)),
        "harness": harness,
        "stderr_tail": logs.get("stderr", "")[-3000:],
    }
