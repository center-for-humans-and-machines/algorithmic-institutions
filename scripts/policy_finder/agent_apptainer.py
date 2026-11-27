"""Run a policy-finder agent as an Apptainer process on this node (DAIS).

The second backend behind agent_runner.py's `start`, `finish` and `run_home`,
selected by PF_AGENT_BACKEND=apptainer (agent_runner.py installs these
functions in place of its own). Built for one DAIS job that also serves the
model (infra/dais/campaign.slurm.sh), where no Docker-compatible runtime
exists and the container spawner cannot run (docs/reports/dais_feasibility.md).

The agent gets what the spawner gives it, in the agent image converted to a
SIF (PF_AGENT_SIF):

  - the instance's export read-only at the repo path, the version's write
    folders writable on top (the same binds as agent_runner.mounts);
  - a home of its own (<run>/home at /home/claude) and a /tmp of its own;
  - the same command: opencode_run.py, unchanged (agent_runner.command);
  - CPUs: one group of PF_AGENT_CPUS CPUs out of PF_AGENT_POOL_CPUS, taken
    with a lock file and set with `taskset`; with PF_AGENT_STEP=1 the
    process runs in a Slurm job step (`srun --overlap -c N --mem=M`) that
    caps its memory at PF_AGENT_MEMORY_MB as well (Apptainer's own --cpus
    and --memory are refused on DAIS);
  - a wall-clock limit (PF_AGENT_TIMEOUT): the process group is killed and
    the run's status is `expired`, as the spawner's.

How many agents run at once is agent_runner.slot's PF_AGENT_SLOTS, as with
the spawner; the CPU groups must be at least that many.

The model: harness opencode against the job's vLLM server. The export's
OpenCode config (export_instance.opencode_config with base_url) names
provider `local`, its base URL `{env:PF_LLM_BASE_URL}` and its key
`{env:PF_LLM_API_KEY}`. Both reach the container as environment variables
only: the URL from PF_LLM_BASE_URL, the key read here from the file
PF_LLM_API_KEY_FILE (mode 600, written by the job) and passed through
APPTAINERENV_, never on a command line, in the export or in a log;
opencode_run.py redacts it from the transcripts and write folders.

Network. DAIS gives users no network namespace, so the container shares the
node's network. OpenCode's web tools are off in its config (the image's
global opencode.json), and nothing asks for confirmation. The agent's shell
commands run through PF_AGENT_SHELL, a wrapper bound at SHELL_TARGET and set
as $SHELL, the shell OpenCode's bash tool uses: infra/dais/agent_shell.sh
runs them under infra/dais/noinet.py, a seccomp filter that refuses every
socket but AF_UNIX, so they cannot reach the internet, while OpenCode itself
keeps its TCP connection to the model. Without PF_AGENT_SHELL the commands
run in plain bash. The filter holds only while nothing the unfiltered side
runs or loads is writable by a command: the wrapper, noinet.py, the export
(with its .opencode config) and the image are read-only; the home's config
folders (HOME_READONLY) are empty read-only binds; no host /run, /tmp or home
is bound, and --containall hides the host's user services (D-Bus, systemd,
tmux sockets). It is a barrier against ordinary network use by the agent's
commands, not a sandbox against a hostile process. Either way `audit`
scans the finished run's transcript for
network use (web tools, curl, wget, pip install, git clone, URLs, network
libraries in written scripts) and the run's record holds what it found
(`network_audit`, in the version's state).

A run is a folder <PF_AGENT_ROOT>/<id> (default $PF_RUNS/agent_runs):
run.json (settings), home/, tmp/, stdout.log, stderr.log, status.json
(status, exit code, times, CPUs). Its worker is a detached process, so a
pipeline that dies leaves the run going; a run whose worker died without a
status (the Slurm job ended) is `lost`, which the pipeline does not count as
one of the agent's attempts.

Settings (environment):
  PF_AGENT_SIF          the agent image (algoinst_claude:opencode as a SIF)
  PF_AGENT_ROOT         run folders (default $PF_RUNS/agent_runs)
  PF_AGENT_POOL_CPUS    the agents' CPUs, "8-23" or "8,9,104,105"; default:
                        the CPUs this process may run on
  PF_AGENT_CPUS         CPUs per agent (agent_runner's, default 2)
  PF_AGENT_LOCKS        the CPU groups' lock files (default <root>/locks)
  PF_AGENT_STEP         1: each agent in its own `srun --overlap` step
  PF_AGENT_MEMORY_MB    the step's memory cap (agent_runner's, default 4096)
  PF_AGENT_TIMEOUT      seconds (agent_runner's, default 7200)
  PF_AGENT_SHELL        host path of the shell wrapper (infra/dais/agent_shell.sh)
  PF_AGENT_NOINET       host path of noinet.py, bound beside the wrapper
  PF_LLM_BASE_URL       the model server's OpenAI-compatible URL (.../v1)
  PF_LLM_API_KEY_FILE   a file holding its key
"""

import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

#: Where the shell wrapper and noinet.py appear in the container
SHELL_DIR = "/opt/pf-agent"
SHELL_TARGET = f"{SHELL_DIR}/shell"
POLL = 15
#: The image's global OpenCode config (Dockerfile.opencode) and where it is
GLOBAL_CONFIG = (Path(__file__).resolve().parents[2]
                 / "infra/claude_proxy/container/CC_env/opencode/opencode.json")
GLOBAL_CONFIG_TARGET = "/opt/opencode/config/opencode/opencode.json"
#: Folders of the run's home bound empty and read-only (see binds)
HOME_READONLY = (".opencode", ".config", ".claude", ".agents")
#: Shell commands and tools that use the network (audit)
NET_COMMANDS = re.compile(
    r"(?<![\w.-])(curl|wget|aria2c|nc|ncat|netcat|ssh|scp|sftp|rsync|telnet|ftp"
    r"|huggingface-cli|hf)\s"
    r"|\b(pip3?|uv\s+pip|conda|mamba|npm|npx|yarn|pnpm|bun|apt(-get)?)\s+(install|add|i)\b"
    r"|\bgit\s+(clone|fetch|pull|push|ls-remote|submodule\s+update)\b"
    r"|\bpython3?\s+-m\s+(pip|http\.server)\b"
)
NET_URL = re.compile(r"\b(?:https?|ftp|wss?)://[^\s'\"<>)]+")
NET_LIBRARIES = re.compile(
    r"\b(import\s+(requests|httpx|socket|urllib3|aiohttp)\b"
    r"|from\s+(requests|httpx|socket|urllib|urllib3|aiohttp|http\.client)\s+import\b"
    r"|urllib\.request|http\.client|socket\.socket\()"
)
WEB_TOOLS = ("webfetch", "websearch", "codesearch")


def _ar():
    """agent_runner (imported late: it imports this module)."""
    import agent_runner

    return agent_runner


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def root():
    path = _env("PF_AGENT_ROOT")
    if not path:
        pf_runs = _env("PF_RUNS")
        if not pf_runs:
            raise _ar().AgentError("set PF_AGENT_ROOT or PF_RUNS")
        path = os.path.join(pf_runs, "agent_runs")
    return Path(path)


def exports():
    pf_runs = _env("PF_RUNS")
    if not pf_runs:
        raise _ar().AgentError("set PF_RUNS (its exports/ holds the agents' copies)")
    return Path(pf_runs) / "exports"


def run_home(run_id, harness="opencode"):
    return str(run_dir(run_id) / "home")


def run_dir(run_id):
    if not re.fullmatch(r"apt-\d+", str(run_id)):
        raise _ar().AgentError(f"not an apptainer agent run: {run_id!r}")
    return root() / str(run_id)


def parse_cpus(text):
    """"0,96,1-3" -> [0, 96, 1, 2, 3], in the given order."""
    cpus = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpu_groups(cpus=None, size=None):
    """The agents' CPUs in groups of `size` (a last, smaller group dropped)."""
    if cpus is None:
        given = _env("PF_AGENT_POOL_CPUS")
        cpus = (parse_cpus(given) if given
                else sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                else list(range(os.cpu_count() or 1)))
    size = max(1, int(_ar().CPUS if size is None else size))
    groups = [cpus[i:i + size] for i in range(0, len(cpus) - size + 1, size)]
    if not groups:
        raise _ar().AgentError(f"no group of {size} CPUs in {cpus}")
    return groups


def endpoint(stem):
    """`local` when the export's OpenCode config uses the job's server."""
    path = exports() / stem / ".opencode" / "opencode.json"
    try:
        conf = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise _ar().AgentError(f"no OpenCode config in the export: {e}") from None
    return "local" if "local" in (conf.get("provider") or {}) else "openrouter"


def binds(stem, write_dirs, run):
    """Apptainer --bind values: the spawner's mounts, the home and /tmp."""
    ar = _ar()
    base = exports()
    out = []
    for m in ar.mounts(stem, write_dirs):
        mode = "ro" if m["read_only"] else "rw"
        out.append(f"{base / m['path']}:{m['target']}:{mode}")
    out.append(f"{run / 'tmp'}:/tmp:rw")
    # the repo's global OpenCode config (an allowlist of tools) over the one
    # the SIF was converted with, so a SIF of an older image gets the same
    # tool set as the spawner's image
    out.append(f"{GLOBAL_CONFIG}:{GLOBAL_CONFIG_TARGET}:ro")
    # folders in the home that OpenCode would read config, plugins or skills
    # from: empty and read-only, so a command cannot add to what it loads
    for d in HOME_READONLY:
        out.append(f"{run / 'home' / d}:{ar.HOME}/{d}:ro")
    shell = _env("PF_AGENT_SHELL")
    if shell:
        noinet = _env("PF_AGENT_NOINET")
        if not noinet:
            raise ar.AgentError("PF_AGENT_SHELL needs PF_AGENT_NOINET")
        out.append(f"{shell}:{SHELL_TARGET}:ro")
        out.append(f"{noinet}:{SHELL_DIR}/noinet.py:ro")
    return out


def argv(stem, write_dirs, prompt, run, cpus, harness="opencode"):
    """The worker's command line: [srun step] taskset apptainer exec ..."""
    ar = _ar()
    if harness != "opencode":
        raise ar.AgentError("the apptainer backend runs harness opencode only")
    sif = _env("PF_AGENT_SIF")
    if not sif or not os.path.exists(sif):
        raise ar.AgentError(f"PF_AGENT_SIF is not an image: {sif!r}")
    point = endpoint(stem)
    n = str(len(cpus))
    cmd = ["apptainer", "exec", "--containall", "--cleanenv",
           "--no-mount", "cwd,tmp,bind-paths",
           "--home", f"{run / 'home'}:{ar.HOME}"]
    for b in binds(stem, write_dirs, run):
        cmd += ["--bind", b]
    env = {"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    if _env("PF_AGENT_SHELL"):
        env["SHELL"] = SHELL_TARGET
    if point == "local":
        url = _env("PF_LLM_BASE_URL")
        if not url:
            raise ar.AgentError("the export uses the local server: set PF_LLM_BASE_URL")
        env["PF_LLM_BASE_URL"] = url
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += ["--pwd", ar.REPO, sif]
    cmd += ar.command(prompt, "opencode", stem, write_dirs, endpoint=point)
    # OpenCode's own time limit, below ours, so the runner always redacts and
    # reports; --require-key-limit is OpenRouter's and stays off the local path
    options = ar.runner_options()
    if point == "local":
        options = [o for o in options if o != "--require-key-limit"]
    cmd += options
    pin = ["taskset", "-c", ",".join(map(str, cpus))]
    if _env("PF_AGENT_STEP") == "1":
        step = ["srun", "--overlap", "--ntasks=1", f"--cpus-per-task={n}",
                f"--mem={ar.MEMORY_MB}M", "--cpu-bind=none", "--kill-on-bad-exit=1",
                f"--job-name=agent-{run.name}"]
        return step + pin + cmd
    return pin + cmd


def secret_env(stem):
    """The key for the container, as APPTAINERENV_ (not on a command line)."""
    if endpoint(stem) != "local":
        return {}
    path = _env("PF_LLM_API_KEY_FILE")
    if not path:
        raise _ar().AgentError("the export uses the local server: set PF_LLM_API_KEY_FILE")
    return {"APPTAINERENV_PF_LLM_API_KEY": Path(path).read_text().strip()}


def _next_id():
    base = root()
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "next_id.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = base / "next_id"
        n = int(counter.read_text()) if counter.exists() else 1
        counter.write_text(f"{n + 1}\n")
    return f"apt-{n}"


def start(stem, write_dirs, prompt, harness="opencode"):
    """Start the agent on the export `<stem>`; return the run's id at once."""
    ar = _ar()
    if harness != "opencode":
        raise ar.AgentError("the apptainer backend runs harness opencode only")
    endpoint(stem)  # fails here, not in the worker, without a config
    run_id = _next_id()
    run = run_dir(run_id)
    for d in ("home", "tmp") + tuple(f"home/{h}" for h in HOME_READONLY):
        (run / d).mkdir(parents=True)
    # OpenCode writes this file into every config folder it loads, ~/.opencode
    # included, and fails when it cannot (export_instance.OPENCODE_GITIGNORE)
    import export_instance

    (run / "home/.opencode/.gitignore").write_text(export_instance.OPENCODE_GITIGNORE)
    settings = {
        "id": run_id, "stem": stem, "write_dirs": list(write_dirs), "prompt": prompt,
        "harness": harness, "groups": cpu_groups(), "timeout": ar.TIMEOUT,
        "locks": _env("PF_AGENT_LOCKS", str(root() / "locks")),
        "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    with open(run / "worker.log", "w") as log:
        worker_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(run)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=str(run),
        )
    tmp = run / "worker.tmp"  # atomic: finish() may read it at any time
    tmp.write_text(json.dumps({"host": socket.gethostname(), "pid": worker_proc.pid}) + "\n")
    os.replace(tmp, run / "worker.json")
    return run_id


def _take_group(groups, folder, poll=2.0):
    """Block until one CPU group is free; (group, its open lock file)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        for i, group in enumerate(groups):
            handle = open(folder / f"agent-cpus-{i}.lock", "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return group, handle
            except BlockingIOError:
                handle.close()
        time.sleep(poll)


def _write_status(run, status):
    tmp = run / "status.tmp"
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, run / "status.json")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def worker(folder):
    """The run: take CPUs, run the container to its end or its time limit,
    record the status. Exit code: the agent's."""
    run = Path(folder)
    s = json.loads((run / "run.json").read_text())
    cpus, lock = _take_group(s["groups"], s["locks"])
    started = _now()
    _write_status(run, {"status": "running", "cpus": cpus, "started_at": started})
    status, code, error = "failed", None, None
    try:
        cmd = argv(s["stem"], s["write_dirs"], s["prompt"], run, cpus, s["harness"])
        env = {**os.environ, **secret_env(s["stem"])}
        with open(run / "stdout.log", "w") as out, open(run / "stderr.log", "w") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                    env=env, start_new_session=True)
            try:
                code = proc.wait(timeout=s["timeout"])
                status = "succeeded" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                status, code = "expired", proc.returncode
                error = f"killed after its limit of {s['timeout']} s"
    except Exception as e:  # noqa: BLE001 - any failure is the run's error
        status, error = "error", f"{type(e).__name__}: {e}"
    _write_status(run, {"status": status, "exit_code": code, "error": error,
                        "cpus": cpus, "started_at": started, "finished_at": _now()})
    lock.close()
    return 0 if status == "succeeded" else 1


def _alive(info):
    """Is the worker of this worker.json alive (on this host)?"""
    if info.get("host") != socket.gethostname():
        return False  # a worker of an earlier job, on another node: gone
    pid = info["pid"]
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False
    return True


def status(run_id):
    path = run_dir(run_id) / "status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def finish(run_id, log=print, harness="opencode"):
    """Wait for the run; return its record, as agent_runner.finish does."""
    ar = _ar()
    run = run_dir(run_id)
    started = time.time()
    try:
        info = json.loads((run / "worker.json").read_text())
    except (OSError, json.JSONDecodeError):
        info = {}  # start() died before writing it: the run is lost
    while True:
        state = status(run_id)
        if state.get("status") not in (None, "running"):
            break
        if not _alive(info):
            state = status(run_id)
            if state.get("status") in (None, "running"):
                state = {**state, "status": "lost",
                         "error": "the worker ended without a status (job ended?)"}
            break
        time.sleep(POLL)
    stdout = (run / "stdout.log").read_text(errors="replace") if (
        run / "stdout.log").exists() else ""
    stderr = (run / "stderr.log").read_text(errors="replace") if (
        run / "stderr.log").exists() else ""
    parsed = ar.parse_result(stdout)
    result = parsed or {}
    verdict, error = state["status"], state.get("error")
    if verdict == "succeeded" and (parsed is None or result.get("is_error")):
        # the runner's verdict (opencode_run.failures), as agent_runner.finish
        verdict = "failed"
        error = "; ".join(result.get("failures") or ["no summary from the runner"])
    elif verdict != "succeeded" and not error and result.get("failures"):
        error = "; ".join(result["failures"])
    source, paths = ar.transcripts(run_id, harness)
    record = {
        "run_id": run_id,
        "backend": "apptainer",
        "status": verdict,
        "exit_code": state.get("exit_code"),
        "error": error,
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "waited_seconds": round(time.time() - started),
        "home_host_path": str(run / "home"),
        "cpus": state.get("cpus"),
        "result": {k: v for k, v in result.items() if k != "result"},
        "report": result.get("result") or "",
        "models": result.get("models") or [],
        "harness": harness,
        "network_audit": audit(paths),
        "stderr_tail": stderr[-3000:],
    }
    if verdict in ("failed", "expired", "error") and endpoint_of_run(run) == "local":
        why = endpoint_trouble(result)
        if why:
            # the job's model server, not the model: not one of the attempts
            record["endpoint_error"] = why
            raise ar.EndpointError(f"agent run {run_id}: {why}", record)
    return record


#: A session's errors that are the endpoint's, not the model's: no connection,
#: a time-out, or a server error (HTTP 5xx)
ENDPOINT_ERRORS = re.compile(
    r"ECONNREFUSED|ECONNRESET|ETIMEDOUT|EPIPE|socket hang up|fetch failed"
    r"|Unable to connect|Cannot connect|Connection (refused|reset|error)"
    r"|timed? ?out|Service Unavailable|Internal Server Error|Bad Gateway"
    r"|Gateway Timeout|status(Code)?\W{0,6}5\d\d\b", re.I)


def server_health(tries=3, wait=10):
    """(healthy, detail) of the job's model server: /health answers 200 and
    /models lists a model with the job's key; `tries` tries `wait` s apart."""
    base = _env("PF_LLM_BASE_URL", "")
    key_file = _env("PF_LLM_API_KEY_FILE")
    if not base:
        return False, "PF_LLM_BASE_URL is not set"
    key = Path(key_file).read_text().strip() if key_file else ""
    detail = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(base.rstrip("/").rsplit("/v1", 1)[0] + "/health",
                                        timeout=20) as r:
                ok = r.status == 200
            req = urllib.request.Request(base.rstrip("/") + "/models",
                                         headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=20) as r:
                ok = ok and bool(json.loads(r.read().decode()).get("data"))
            if ok:
                return True, "healthy"
            detail = "answered without a model"
        except Exception as e:  # noqa: BLE001 - any failure is the server's
            detail = f"{type(e).__name__}: {e}"
        if i + 1 < tries:
            time.sleep(wait)
    return False, detail


def endpoint_trouble(result):
    """Why a failed run is the endpoint's fault, or "" (the model's)."""
    # the provider's own errors only: the runner's failures name its own
    # time limit ("timeout"), which is the model's run taking too long
    errors = json.dumps([result.get("errors"), result.get("final_error")])
    found = ENDPOINT_ERRORS.search(errors)
    healthy, detail = server_health()
    if not healthy:
        return f"the model server is down after the run ({detail})"
    if found:
        return f"the session hit an endpoint error ({found.group(0)}): {errors[:500]}"
    return ""


def endpoint_of_run(run):
    """The endpoint of a run, from its settings (its export may be gone)."""
    try:
        return endpoint(json.loads((run / "run.json").read_text())["stem"])
    except Exception:  # noqa: BLE001 - no export: not ours to judge
        return "unknown"


# -- the network audit ---------------------------------------------------------


def _tool_parts(paths):
    """Every tool call of the run: from session.json, else events.jsonl."""
    by_name = {os.path.basename(p): p for p in paths}
    if "session.json" in by_name:
        try:
            with open(by_name["session.json"], errors="replace") as f:
                session = json.load(f)
            return [p for m in session.get("messages", []) for p in m.get("parts") or []
                    if p.get("type") == "tool"]
        except (OSError, json.JSONDecodeError):
            pass
    parts = []
    if "events.jsonl" in by_name:
        with open(by_name["events.jsonl"], errors="replace") as f:
            for line in f:
                try:
                    part = json.loads(line).get("part") or {}
                except (json.JSONDecodeError, AttributeError):
                    continue
                if part.get("type") == "tool":
                    parts.append(part)
    return parts


def audit(paths):
    """Network use in a finished run's transcript: {tool_calls, findings}.

    A finding is a web tool call, a shell command that uses the network
    (NET_COMMANDS) or names a URL, or a file written with a network library
    (NET_LIBRARIES). Each says whether the call failed. URLs in written
    files (a citation in the notes) are not network use and are not listed.
    """
    findings, parts = [], _tool_parts(paths)
    for part in parts:
        tool = part.get("tool") or ""
        state = part.get("state") or {}
        args = state.get("input") or {}
        failed = state.get("status") == "error"
        if tool in WEB_TOOLS:
            findings.append({"tool": tool, "kind": "web tool", "failed": failed,
                             "text": json.dumps(args)[:300]})
        elif tool == "bash":
            command = str(args.get("command") or "")
            hits = [m.group(0).strip() for m in NET_COMMANDS.finditer(command)]
            hits += NET_URL.findall(command)
            # python run inline (-c, heredocs, `cat > a.py <<E`) and bash's /dev/tcp
            hits += [m.group(0) for m in NET_LIBRARIES.finditer(command)]
            hits += re.findall(r"/dev/(?:tcp|udp)/\S+", command)
            if hits:
                findings.append({"tool": tool, "kind": "shell", "failed": failed,
                                 "matches": sorted(set(hits))[:10],
                                 "text": command[:300]})
        elif tool in ("write", "edit"):
            text = str(args.get("content") or args.get("newString") or "")
            hits = sorted({m.group(0) for m in NET_LIBRARIES.finditer(text)})
            if hits:
                findings.append({"tool": tool, "kind": "network library",
                                 "failed": failed, "matches": hits[:10],
                                 "file": str(args.get("filePath") or "")})
    return {"tool_calls": len(parts), "findings": findings}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        sys.exit(worker(sys.argv[2]))
    sys.exit("usage: agent_apptainer.py worker <run folder>")
