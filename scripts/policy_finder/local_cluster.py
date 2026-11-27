"""The policy finder's cluster calls on the machine the pipeline runs on.

The second backend behind cluster_proxy.py's functions, selected by
PF_CLUSTER_BACKEND=local (cluster_proxy.py installs these functions in place
of its own). Built for a DAIS CPU job (infra/dais/pipeline.slurm.sh), where
the unchanged sim image runs fastest as many single-threaded processes, one
per hyperthread (docs/reports/dais_feasibility.md):

  - `snapshot(ref)`: the commit's files (`git archive`) in
    $PF_LOCAL_ROOT/snapshots/<sha>/tree; its sims write their outputs there
  - `run(command, sha)`: the command in the sim image (Apptainer) on that
    tree, at /opt/algorithmic-institutions as on Raven; (exit code, log)
  - `validate_rule(...)`: validate-rule through `run`
  - `submit(config, sha)`: one sim into the pool: a detached worker process
    that takes a free CPU slot, pins itself to it and runs
    `python -m aimanager simulate <config>` single-threaded in the image
  - `queue()`, `wait(jobs)`: the pool's jobs whose worker is still alive
  - `fetch(sha, remote_dir, local_dir)`: a copy of the snapshot's output dir
  - `sweep_parts(...)`: how many sims a sweep is split into (below)
  - `cleanup(sha)`: a snapshot removed, unless a live job runs on it (the
    pipeline calls it for a scored version's snapshots)

The pool needs no daemon: each job is its own worker, and the slots are lock
files (one per CPU of PF_POOL_CPUS, node-local), so pipelines running side by
side (a campaign) share one pool. A job's folder ($PF_LOCAL_ROOT/jobs/<id>)
holds its settings (job.json), its sim's log and its result (status.json:
exit code, CPU, wall seconds, peak memory). A worker killed with its Slurm job
leaves no status: it is then out of the queue, its output has no sweep.json,
and the pipeline reruns it, as on Raven.

Slicing: the pipeline asks this backend how many parts a sweep has
(`sweep_parts`), and splits it with generate_sim_config.py's own `--n-parts`;
merge_sweeps.py joins them as for Raven's two parts. Parts of about
PF_LOCAL_PART_EPISODES episodes each (default 4000, a slice of the
feasibility report) keep every CPU of the pool busy. A sim seeds its batch i
with `seed + i` and part k starts after the batches of the parts before it, so
a split plays the same games as one whole sim exactly when every part but the
last fills whole batches (docs/reports/dais_gpu_sim_equivalence.md). The
smallest such split is chosen; when none exists (exp2_multi: 1024 episodes a
point in batches of 1000), Raven's own split is used, so the result does not
depend on the pool's size either way. The pipeline records which in its state.

Settings (environment):
  PF_LOCAL_ROOT          snapshots, jobs (default $PF_RUNS/local_cluster)
  PF_SIM_SIF             the sim image (algoinst.sif)
  PF_ARTIFACTS           the model files, bound over the snapshot's artifacts/
                         (snapshots hold git-lfs pointers; default: the
                         checkout's artifacts/)
  PF_LOCAL_CONTAINER     apptainer (default) or none: the host's python on
                         the snapshot (tests, development)
  PF_POOL_CPUS           the pool's CPUs, one slot each, "0,96,1,97" or
                         "0-3"; default: the CPUs this process may run on
  PF_POOL_LOCKS          the slots' lock files (default: node-local temp dir)
  PF_SIM_THREADS         threads per sim (default 1; results are exactly
                         reproducible only at a fixed thread count)
  PF_LOCAL_PART_EPISODES episodes per sweep part (default 4000)
  PF_LOCAL_JOB_SECONDS   a sim is killed after this long (default 3600)
  PF_LOCAL_POLL_SECONDS  between queue polls in wait() (default 5)
"""

import fcntl
import json
import math
import os
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
#: Inside the image, a snapshot is mounted at the repo path (as on Raven)
REPO = "/opt/algorithmic-institutions"
#: The batch size of each setting's sweep sims and whether its parts play
#: common games (generate_sim_config.py SETTINGS; a test keeps them equal)
SWEEP_BATCH = {"exp2": 1000, "exp2_multi": 1000, "exp1": 1000, "exp2_diff": None}
WORKER_EXIT = {"killed": -9, "timeout": 124}


def _cp():
    """cluster_proxy, for its exception classes (imported late: it imports
    this module when the backend is selected)."""
    import cluster_proxy

    return cluster_proxy


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def root():
    path = _env("PF_LOCAL_ROOT")
    if not path:
        pf_runs = _env("PF_RUNS")
        if not pf_runs:
            raise _cp().ClusterError("set PF_LOCAL_ROOT or PF_RUNS")
        path = os.path.join(pf_runs, "local_cluster")
    return Path(path)


def threads():
    return int(_env("PF_SIM_THREADS", "1"))


# -- snapshots ---------------------------------------------------------------


def _git(*args, **kw):
    return subprocess.run(["git", "-C", str(MAIN), *args], capture_output=True,
                          text=True, **kw)


def tree(sha):
    return root() / "snapshots" / sha / "tree"


def snapshot(ref):
    """The files of commit `ref` in a directory of their own; its sha.

    The checkout's object store holds every instance branch (worktrees share
    it). git-lfs files stay pointers: the models come from PF_ARTIFACTS.
    """
    result = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    if result.returncode != 0:
        raise _cp().ClusterError(f"no commit {ref}: {result.stderr.strip()}")
    sha = result.stdout.strip()
    target = tree(sha)
    if (target.parent / "complete").exists():
        return sha
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix="tree.", dir=target.parent))
    archive = subprocess.Popen(
        ["git", "-C", str(MAIN), "-c", "filter.lfs.smudge=", "-c", "filter.lfs.process=",
         "-c", "filter.lfs.required=false", "archive", "--format=tar", sha],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    untar = subprocess.run(["tar", "-x", "-C", str(partial)], stdin=archive.stdout,
                           capture_output=True)
    archive.stdout.close()
    err = archive.stderr.read().decode(errors="replace")
    if archive.wait() != 0 or untar.returncode != 0:
        shutil.rmtree(partial, ignore_errors=True)
        raise _cp().ClusterError(f"snapshot of {sha[:12]} failed: {err}"
                                 f"{untar.stderr.decode(errors='replace')}")
    shutil.rmtree(target, ignore_errors=True)
    os.replace(partial, target)
    (target.parent / "complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return sha


# -- running in the image ----------------------------------------------------


def argv(sha, command, tmp, n_threads):
    """(argv, cwd, env) of `bash -c command` on snapshot `sha` (None: the
    checkout, read-only), in the sim image as common.sh's in_sim_image runs
    it, or on the host with PF_LOCAL_CONTAINER=none."""
    repo = tree(sha) if sha else MAIN
    thread_env = {"OMP_NUM_THREADS": str(n_threads), "MKL_NUM_THREADS": str(n_threads)}
    mode = _env("PF_LOCAL_CONTAINER", "apptainer")
    if mode == "none":
        path = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"
        env = {**os.environ, **thread_env, "PYTHONPATH": str(repo / "src"), "PATH": path,
               "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
        return ["bash", "-c", command], str(repo), env
    if mode != "apptainer":
        raise _cp().ClusterError(f"PF_LOCAL_CONTAINER is apptainer or none, not {mode}")
    sif = _env("PF_SIM_SIF")
    if not sif or not os.path.exists(sif):
        raise _cp().ClusterError(f"PF_SIM_SIF is not an image: {sif!r}")
    artifacts = _env("PF_ARTIFACTS", str(MAIN / "artifacts"))
    cmd = ["apptainer", "exec", "--containall", "--cleanenv", "--writable-tmpfs",
           "--no-mount", "cwd,tmp,bind-paths", "--bind", f"{tmp}:/tmp",
           "--bind", f"{repo}:{REPO}:{'rw' if sha else 'ro'}",
           "--bind", f"{artifacts}:{REPO}/artifacts:ro",
           "--env", "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
           "--env", f"PYTHONPATH={REPO}/src", "--env", "PYTHONDONTWRITEBYTECODE=1",
           "--env", "CUDA_VISIBLE_DEVICES="]
    for key, value in thread_env.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += ["--pwd", REPO, sif, "bash", "-c", command]
    return cmd, None, None


def run(command, sha=None, timeout=3600):
    """`command` in the sim image on snapshot `sha`; (exit code, log)."""
    with tempfile.TemporaryDirectory(prefix="pf-run.") as tmp:
        cmd, cwd, env = argv(sha, command, tmp, threads())
        try:
            result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                                    text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise _cp().ClusterError(f"{command[:200]} still running after {timeout} s"
                                     f"\n{e.stdout or ''}") from None
    return result.returncode, result.stdout + result.stderr


def validate_rule(sha, rule, min_params, max_params, sobol_points, setting="exp2"):
    """validate-rule on the snapshot, in the image; (ok, log)."""
    upper = "none" if max_params is None else max_params
    code, log = run(
        f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m aimanager validate-rule {rule}"
        f" --min-params {min_params} --max-params {upper}"
        f" --sobol-points {sobol_points} --setting {setting}",
        sha,
    )
    return code == 0, log


# -- the pool ----------------------------------------------------------------


def parse_cpus(text):
    """"0,96,1-3" -> [0, 96, 1, 2, 3], in the given order."""
    cpus = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def pool_cpus():
    """The pool's CPUs, one slot each: PF_POOL_CPUS, else those this process
    may run on."""
    given = _env("PF_POOL_CPUS")
    if given:
        return parse_cpus(given)
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def lock_dir():
    return Path(_env("PF_POOL_LOCKS",
                     os.path.join(tempfile.gettempdir(), f"pf-pool-{os.getuid()}")))


def job_dir(job):
    if not re.fullmatch(r"local-\d+", str(job)):
        raise _cp().ClusterError(f"not a local job id: {job!r}")
    return root() / "jobs" / str(job)


def _next_id():
    jobs = root() / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    with open(jobs / "next_id.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = jobs / "next_id"
        n = int(counter.read_text()) if counter.exists() else 1
        counter.write_text(f"{n + 1}\n")
    return f"local-{n}"


def submit(config, sha, run_group="policy-finder"):
    """`python -m aimanager simulate <config>` on snapshot `sha`, into the
    pool. Returns the job id at once; the sim waits for a free slot."""
    if not (tree(sha).parent / "complete").exists():
        raise _cp().ClusterError(f"no snapshot {sha[:12]}: snapshot() it first")
    if not (tree(sha) / config).is_file():
        raise _cp().ClusterError(f"no {config} in snapshot {sha[:12]}")
    job = _next_id()
    folder = job_dir(job)
    folder.mkdir()
    settings = {
        "job": job, "config": config, "sha": sha, "run_group": run_group,
        "root": str(root()),
        "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threads": threads(), "cpus": pool_cpus(),
        "container": _env("PF_LOCAL_CONTAINER", "apptainer"),
        "image": _env("PF_SIM_SIF"), "artifacts": _env("PF_ARTIFACTS"),
        "timeout": int(_env("PF_LOCAL_JOB_SECONDS", "3600")),
    }
    (folder / "job.json").write_text(json.dumps(settings, indent=2) + "\n")
    with open(folder / "worker.log", "w") as log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(folder)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=str(folder),
        )
    (folder / "worker.json").write_text(json.dumps(
        {"host": socket.gethostname(), "pid": worker.pid}) + "\n")
    return job


def _take_slot(cpus, poll=1.0):
    """Block until one of `cpus` is free; (cpu, its open lock file)."""
    folder = lock_dir()
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        for cpu in cpus:
            handle = open(folder / f"cpu{cpu}.lock", "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return cpu, handle
            except BlockingIOError:
                handle.close()
        time.sleep(poll)


def _write_status(folder, status):
    tmp = folder / "status.tmp"
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, folder / "status.json")


def worker(folder):
    """A pool job: wait for a slot, run the sim pinned to it, record the
    result. Its exit code is the sim's."""
    folder = Path(folder)
    settings = json.loads((folder / "job.json").read_text())
    os.environ["PF_LOCAL_ROOT"] = settings["root"]
    os.environ["PF_LOCAL_CONTAINER"] = settings["container"]
    if settings["image"]:
        os.environ["PF_SIM_SIF"] = settings["image"]
    if settings["artifacts"]:
        os.environ["PF_ARTIFACTS"] = settings["artifacts"]
    cpu, lock = _take_slot(settings["cpus"])
    started = time.time()
    _write_status(folder, {"state": "running", "cpu": cpu,
                           "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    (folder / "tmp").mkdir(exist_ok=True)
    cmd, cwd, env = argv(settings["sha"], f"python -m aimanager simulate {settings['config']}",
                         folder / "tmp", settings["threads"])

    def pin():
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {cpu})

    with open(folder / "sim.log", "w") as log:
        sim = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, preexec_fn=pin,
                               start_new_session=True)
        try:
            code = sim.wait(timeout=settings["timeout"])
        except subprocess.TimeoutExpired:
            os.killpg(sim.pid, signal.SIGKILL)
            sim.wait()
            code = WORKER_EXIT["timeout"]
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    _write_status(folder, {
        "state": "done", "exit_code": code, "cpu": cpu,
        "wall_seconds": round(time.time() - started, 1),
        "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 1),
        # kB on Linux
        "maxrss_kb": usage.ru_maxrss,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    shutil.rmtree(folder / "tmp", ignore_errors=True)
    lock.close()
    return code


def _alive(info):
    """Is the worker of this worker.json alive (on this host)?"""
    if info.get("host") != socket.gethostname():
        return False  # a worker of an earlier job, on another node: gone
    pid = info["pid"]
    try:
        # reaps the worker if it is our child, so a dead one is no zombie
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False  # the pid was reused by another user's process
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False
    return True


def status(job):
    """A job's status.json, {} while it has none."""
    path = job_dir(job) / "status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _live():
    """{job id: its folder} of the pool's jobs not yet done whose worker is
    alive."""
    jobs = root() / "jobs"
    live = {}
    if not jobs.is_dir():
        return live
    for folder in jobs.iterdir():
        if not folder.name.startswith("local-") or not (folder / "worker.json").exists():
            continue
        if status(folder.name).get("state") == "done":
            continue
        if _alive(json.loads((folder / "worker.json").read_text())):
            live[folder.name] = folder
    return live


def queue():
    """The pool's jobs not yet done whose worker is alive."""
    return set(_live())


def cleanup(sha):
    """Remove snapshot `sha` unless a live job runs on it; True if removed.

    The folder is renamed first (atomically), so a later snapshot() of the
    commit makes a fresh one rather than seeing a half-removed tree.
    """
    folder = root() / "snapshots" / sha
    if not folder.is_dir():
        return False
    for job in _live().values():
        try:
            if json.loads((job / "job.json").read_text())["sha"] == sha:
                return False
        except (OSError, ValueError, KeyError):
            return False  # a job being written: keep the snapshot
    doomed = folder.with_name(f"{sha}.removing.{os.getpid()}")
    os.replace(folder, doomed)
    shutil.rmtree(doomed, ignore_errors=True)
    return True


def wait(jobs, poll=None):
    """Until none of `jobs` is in the queue; a failed sim's log tail is printed
    (the pipeline then finds no sweep.json and reruns it once)."""
    jobs = set(map(str, jobs))
    poll = float(_env("PF_LOCAL_POLL_SECONDS", "5")) if poll is None else poll
    while jobs & queue():
        time.sleep(poll)
    for job in sorted(jobs):
        state = status(job)
        if state.get("exit_code") != 0:
            log = job_dir(job) / "sim.log"
            tail = log.read_text(errors="replace")[-1500:] if log.exists() else ""
            print(f"local job {job}: {state or 'no status (killed)'}\n{tail}",
                  file=sys.stderr, flush=True)


def fetch(sha, remote_dir, local_dir, key=None):
    """Copy snapshot `sha`'s output `remote_dir` (repo-relative) into
    `local_dir`; the bytes copied."""
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", remote_dir) or ".." in remote_dir:
        raise _cp().ClusterError(f"unsafe path {remote_dir!r}")
    source = tree(sha) / remote_dir
    if not source.is_dir():
        raise _cp().NoOutput(f"no {remote_dir} on snapshot {sha[:12]}")
    os.makedirs(local_dir, exist_ok=True)
    shutil.copytree(source, local_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".*"))
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(local_dir)
               for f in fs if not f.startswith("."))


# -- slicing -----------------------------------------------------------------


def edges(points, n_parts):
    """generate_sim_config.build_sim_configs's part edges."""
    return [round(k * points / n_parts) for k in range(n_parts + 1)]


def aligned(points, per_point, batch, n_parts):
    """Does every part but the last fill whole batches (so the split plays the
    games of one whole sim)? `batch` None: common games, any split does."""
    if batch is None or n_parts == 1:
        return True
    e = edges(points, n_parts)
    return all((e[k + 1] - e[k]) * per_point % batch == 0 for k in range(n_parts - 1))


def sweep_parts(points, per_point, setting, max_job_episodes):
    """(parts, aligned): the smallest split into parts of at most
    PF_LOCAL_PART_EPISODES episodes that plays the games of one whole sim;
    without one, Raven's split (by its job size)."""
    batch = SWEEP_BATCH[setting]
    target = int(_env("PF_LOCAL_PART_EPISODES", "4000"))
    total = points * per_point
    for n in range(max(1, math.ceil(total / target)), points + 1):
        if aligned(points, per_point, batch, n):
            return n, True
    n = max(1, math.ceil(total / max_job_episodes))
    return n, aligned(points, per_point, batch, n)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        sys.exit(worker(sys.argv[2]))
    sys.exit("usage: local_cluster.py worker <job folder>")
