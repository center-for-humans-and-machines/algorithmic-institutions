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
"""

import contextlib
import fcntl
import glob
import json
import os
import time
import urllib.error
import urllib.request

URL = os.environ.get("PF_SPAWNER", "http://container_spawner:8877")
#: The spawner's state dir as mounted here (run homes, saved logs)
SPAWNER_STATE = os.environ.get("PF_SPAWNER_STATE", "/var/spawner")
PROFILE = "cpu"
REPO = "/opt/algorithmic-institutions"
SLOTS = int(os.environ.get("PF_AGENT_SLOTS", "4"))
CPUS = float(os.environ.get("PF_AGENT_CPUS", "2"))
MEMORY_MB = int(os.environ.get("PF_AGENT_MEMORY_MB", "4096"))
TIMEOUT = int(os.environ.get("PF_AGENT_TIMEOUT", "7200"))
POLL = 30
TERMINAL = ("succeeded", "failed", "cancelled", "expired", "error")


class AgentError(Exception):
    pass


def _request(method, path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        URL + path, data=data, method=method,
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


def command(prompt):
    return [
        "claude", "-p", prompt,
        "--agent", "policy-finder-host",
        "--strict-mcp-config",
        "--output-format", "json",
    ]


def models(run_id):
    """The models the run's Claude sessions called, from their transcripts."""
    found = set()
    home = os.path.join(SPAWNER_STATE, "runs", run_id, "home")
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


def start(stem, write_dirs, prompt):
    """Start the agent on the export `<stem>`; return the spawner's run id.

    Waits while the spawner is at its cap (HTTP 409).
    """
    body = {
        "profile": PROFILE,
        "command": command(prompt),
        "workdir": REPO,
        "mounts": mounts(stem, write_dirs),
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": TIMEOUT,
    }
    while True:
        try:
            return _request("POST", "/runs", body)["id"]
        except AgentError as e:
            if "HTTP 409" not in str(e):
                raise
            time.sleep(60)


def finish(run_id, log=print):
    """Wait for the run; return its record (status, report, cost, models)."""
    started = time.time()
    run = _request("GET", f"/runs/{run_id}")
    while run["status"] not in TERMINAL:
        time.sleep(POLL)
        try:
            run = _request("GET", f"/runs/{run_id}")
        except AgentError as e:
            log(f"poll of run {run_id} failed: {e}")
    logs = _request("GET", f"/runs/{run_id}/logs?tail=5000")
    result = parse_result(logs.get("stdout", ""))
    return {
        "run_id": run_id,
        "status": run["status"],
        "exit_code": run.get("exit_code"),
        "error": run.get("error"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "waited_seconds": round(time.time() - started),
        "home_host_path": run.get("home_host_path"),
        "result": {k: v for k, v in (result or {}).items() if k != "result"},
        "report": (result or {}).get("result") or "",
        "models": models(run_id),
        "stderr_tail": logs.get("stderr", "")[-3000:],
    }
