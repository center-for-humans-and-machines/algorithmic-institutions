"""Raven through the Claude proxy's command server, for the policy finder.

The proxy has no Raven SSH keys; every cluster call is an HTTP request to the
command server (the `run_on_cluster` skill in the proxy). A sim runs on an
immutable snapshot of one commit: `snapshot(sha)` publishes the instance
branch's commit to `/u/ykeller/cc-snapshots/<sha>/`, and a job or login-node
command with `checkout=<sha>` runs on it. Its `plots/` and `.log/` are
`/u/ykeller/cc-snapshot-outputs/<sha>/`, so instances never share a tree and
nothing is staged in the live checkout. The AH models come from the live
checkout's `artifacts/`, read-only.

  - `snapshot(ref)`: publish a commit, return its sha
  - `run(command, sha)`: a login-node command in the image; exit code and log
  - `validate_rule(...)`: validate-rule in the image (torch 1.11), which
    rejects code torch 2.5 accepts
  - `submit(config, sha)`: one sim job; on HTTP 409 (the job cap) it waits
    and retries
  - `wait(jobs)`: until none of the jobs is in the queue
  - `fetch(sha, remote_dir, local_dir)`: a snapshot output dir, through
    /opt/cluster_transfer and /sync_from_cluster
"""

import contextlib
import fcntl
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request

URL = os.environ.get("PF_COMMAND_SERVER", "http://cc_command_server:8765")
CLUSTER = "raven"
#: The local mirror of Raven's /opt/cluster_transfer (/sync_from_cluster)
TRANSFER_LOCAL = os.environ.get(
    "PF_TRANSFER_LOCAL", "/var/cluster_transfer/cluster_transfer"
)
TRANSFER_REMOTE = "/opt/cluster_transfer"
#: The command server's run logs, as mounted here
LOG_DIR = os.environ.get("PF_CC_OUTPUT", "/var/cc_output")
#: Inside the image, a snapshot is mounted at the repo path
REPO = "/opt/algorithmic-institutions"
SIM_JOB = {"time": "01:00:00", "n_cpu": 2, "mem": "16G", "n_gpu": 1, "n_nodes": 1}
#: Seconds between queue polls, and between retries of a 409
POLL_SECONDS = int(os.environ.get("PF_POLL_SECONDS", "120"))
RETRY_SECONDS = int(os.environ.get("PF_RETRY_SECONDS", "120"))
#: A fetch above this many bytes is refused (the transfer guideline is ~1 GB)
FETCH_LIMIT = 1_000_000_000
#: One /sync_from_cluster at a time across pipelines: it mirrors the whole
#: transfer area, so a fetch running beside another can copy the other's
#: half-written files
FETCH_LOCK = os.environ.get(
    "PF_FETCH_LOCK", os.path.join(os.environ.get("PF_RUNS", "/tmp"), "locks", "fetch.lock")
)
#: rsync's temporary files (".<name>.<random>") are never part of a fetch
TEMP_FILES = shutil.ignore_patterns(".*")


@contextlib.contextmanager
def _fetch_lock():
    os.makedirs(os.path.dirname(FETCH_LOCK), exist_ok=True)
    with open(FETCH_LOCK, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


class ClusterError(Exception):
    pass


class NoOutput(ClusterError):
    """The snapshot has no such output: its job wrote nothing (not an outage)."""


class Busy(ClusterError):
    """HTTP 409: the command server's job cap is reached."""


def _request(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        if e.code == 409:
            raise Busy(detail) from None
        raise ClusterError(f"{method} {path}: HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise ClusterError(f"{method} {path}: {e.reason}") from None


def _log(run_id):
    path = os.path.join(LOG_DIR, f"{run_id}.log")
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def wait_run(run_id, poll=3, timeout=7200):
    """Wait for a command-server run; return (exit_code, log)."""
    deadline = time.time() + timeout
    time.sleep(1)
    while time.time() < deadline:
        state = _request("GET", f"/runs/{run_id}")
        if state.get("status") != "running":
            return state.get("exit_code"), _log(run_id)
        time.sleep(poll)
    raise ClusterError(f"run {run_id} still running after {timeout} s")


def _start(path, body, retries=None):
    """POST a run-starting endpoint, retrying on 409; return the response."""
    attempt = 0
    while True:
        try:
            return _request("POST", path, body)
        except Busy:
            attempt += 1
            if retries is not None and attempt > retries:
                raise
            time.sleep(RETRY_SECONDS * (0.75 + random.random() / 2))


def snapshot(ref):
    """Publish `ref` (a sha or branch of the mounted repo); return its sha."""
    response = _start("/sync_to_cluster", {"cluster": CLUSTER, "ref": ref})
    code, log = wait_run(response["run_id"], timeout=3600)
    sha = response.get("sha")
    if code != 0 or not sha:
        raise ClusterError(f"snapshot of {ref} failed ({code}): {log[-2000:]}")
    return sha


def run(command, sha=None, timeout=3600):
    """A login-node command in the image (on snapshot `sha`); (code, log)."""
    if "\n" in command or len(command) > 4096:
        raise ClusterError("a login-node command is one line of at most 4096 bytes")
    body = {"cluster": CLUSTER, "command": command}
    if sha:
        body["checkout"] = sha
    response = _start("/run_in_container", body)
    return wait_run(response["run_id"], timeout=timeout)


def validate_rule(sha, rule, min_params, max_params, sobol_points, setting="exp2"):
    """validate-rule on the snapshot, in the image; (ok, log)."""
    upper = "none" if max_params is None else max_params
    code, log = run(
        f"cd {REPO} && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
        f" python -m aimanager validate-rule {rule}"
        f" --min-params {min_params} --max-params {upper}"
        f" --sobol-points {sobol_points} --setting {setting}",
        sha,
    )
    return code == 0, log


def submit(config, sha, run_group="policy-finder"):
    """Submit `python -m aimanager simulate <config>` on snapshot `sha`.

    Waits and retries while the command server's job cap is reached.
    Returns the slurm job id.
    """
    body = {
        "cluster": CLUSTER,
        "launcher": "python",
        "program_call": f"-m aimanager simulate {config}",
        "run_group": run_group,
        "project": "algorithmic-institutions",
        "checkout": sha,
        **SIM_JOB,
    }
    response = _start("/submit_slurm_job", body)
    code, log = wait_run(response["run_id"], timeout=600)
    match = re.search(r"Submitted batch job (\d+)", log)
    if code != 0 or not match:
        raise ClusterError(f"submit of {config} failed ({code}): {log[-2000:]}")
    return match.group(1)


def queue():
    """The job ids in the queue (squeue --me)."""
    response = _request("GET", f"/jobs?cluster={CLUSTER}")
    code, log = wait_run(response["run_id"], timeout=600)
    # no header means squeue did not run: that must not read as an empty queue
    if code != 0 or "JOBID" not in log:
        raise ClusterError(f"squeue failed ({code}): {log[-1000:]}")
    return set(re.findall(r"^\s*(\d+)\s", log, flags=re.M))


def wait(jobs, poll=None):
    """Until none of `jobs` is in the queue. A failed squeue is retried."""
    jobs = set(map(str, jobs))
    poll = POLL_SECONDS if poll is None else poll
    failures = 0
    while True:
        try:
            left = jobs & queue()
            failures = 0
        except ClusterError:
            failures += 1
            if failures > 10:
                raise
            left = jobs
        if not left:
            return
        time.sleep(poll * (0.75 + random.random() / 2))


NO_OUTPUT = "__PF_NO_OUTPUT__"
VANISHED = 24  # rsync: some source files vanished


def fetch(sha, remote_dir, local_dir, key):
    """Copy snapshot `sha`'s output `remote_dir` (repo-relative, under plots/)
    into `local_dir`, through /opt/cluster_transfer/pf/<key>.

    Returns the bytes fetched. The transfer copies are removed on both ends.
    """
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", remote_dir + key) or ".." in (
        remote_dir + key
    ):
        raise ClusterError(f"unsafe path {remote_dir!r} or key {key!r}")
    remote = f"{TRANSFER_REMOTE}/pf/{key}"
    code, log = run(
        f"rm -rf {remote} && mkdir -p {remote}"
        f" && {{ test -d {REPO}/{remote_dir} || {{ echo {NO_OUTPUT[:-2]}\"\"__; exit 3; }}; }}"
        f" && cp -r {REPO}/{remote_dir}/. {remote}/ && du -sb {remote}",
        sha,
    )
    if code != 0:
        # only the marker (printed, not echoed with the command: the shell
        # drops the "") means the output is missing; anything else (Raven
        # unreachable, the login node failing) says nothing about the job
        if NO_OUTPUT in log:
            raise NoOutput(f"no {remote_dir} on snapshot {sha[:12]}")
        raise ClusterError(f"could not look for {remote_dir} on snapshot {sha[:12]}"
                           f" ({code}): {log[-1500:]}")
    size = re.findall(rf"^(\d+)\s+{re.escape(remote)}\s*$", log, flags=re.M)
    if size and int(size[-1]) > FETCH_LIMIT:
        run(f"rm -rf {remote}")
        raise ClusterError(f"{remote_dir} is {int(size[-1]) / 1e9:.1f} GB, above the limit")
    local_copy = os.path.join(TRANSFER_LOCAL, "pf", key)
    with _fetch_lock():
        response = _start("/sync_from_cluster", {"cluster": CLUSTER})
        code, log = wait_run(response["run_id"], timeout=7200)
        # 24: files vanished mid-sync. The sync mirrors the whole transfer
        # tree, where other pipelines stage and remove their own keys; ours
        # was complete before the sync and the size check below covers it.
        if code not in (0, VANISHED):
            raise ClusterError(f"sync_from_cluster failed ({code}): {log[-1500:]}")
        if not os.path.isdir(local_copy):
            raise ClusterError(f"{local_copy} missing after sync_from_cluster")
        os.makedirs(local_dir, exist_ok=True)
        shutil.copytree(local_copy, local_dir, dirs_exist_ok=True, ignore=TEMP_FILES)
    fetched = sum(
        os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(local_dir)
        for f in fs if not f.startswith(".")
    )
    # du -sb also counts directories (~4 kB each): a margin, not equality
    if size and fetched < int(size[-1]) * 0.98 - 65536:
        raise ClusterError(
            f"{remote_dir}: fetched {fetched} of {size[-1]} bytes (a sync was cut short)")
    shutil.rmtree(local_copy, ignore_errors=True)
    run(f"rm -rf {remote}")
    return fetched
