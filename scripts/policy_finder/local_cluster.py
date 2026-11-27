"""The policy finder's cluster calls on the machine the pipeline runs on.

The second backend behind cluster_proxy.py's functions, selected by
PF_CLUSTER_BACKEND=local (cluster_proxy.py installs these functions in place
of its own). Built for a DAIS CPU job (infra/dais/pipeline.slurm.sh), where
the unchanged sim image runs fastest as many single-threaded processes, one
per hyperthread (docs/reports/dais_feasibility.md):

  - `snapshot(ref)`: the commit's files (`git archive`) in
    $PF_LOCAL_ROOT/snapshots/<sha>/tree, made under a lock; its sims write
    their outputs there
  - `run(command, sha)`: the command in the sim image (Apptainer) on that
    tree, at /opt/algorithmic-institutions as on Raven; (exit code, log)
  - `validate_rule(...)`: validate-rule through `run`
  - `submit(config, sha)`: one sim into the pool: a detached worker process
    that takes a free CPU slot, pins itself to it, clears the config's output
    folder (so no earlier sim's result can be read as this one's) and runs
    `python -m aimanager simulate <config>` single-threaded in the image
  - `queue()`, `wait(jobs)`: the pool's jobs whose worker is still alive
  - `lost(job)`: the job's worker died before its sim finished (killed with
    its Slurm job), as opposed to a sim that ran and failed
  - `fetch(sha, remote_dir, local_dir)`: a copy of the snapshot's output dir
  - `sweep_parts(...)`: how many sims a sweep is split into (below)
  - `cleanup(sha)`: a snapshot removed, unless a live job runs on it or it is
    being made (the pipeline calls it for a scored version's snapshots)

The pool needs no daemon: each job is its own worker, and the slots are lock
files on node-local storage (one per CPU of PF_POOL_CPUS), so pipelines
running side by side in one job (a campaign) share one pool. A job's folder
($PF_LOCAL_ROOT/jobs/<id>) holds its settings (job.json), its worker
(worker.json: host, pid, process start time), its sim's log and its result
(status.json: exit code, CPU, start and end, peak memory); jobs/active/<host>/
lists that host's jobs not yet done, so a queue poll reads only those. Done jobs' folders
are pruned after PF_LOCAL_KEEP_DAYS. A worker killed with its Slurm job
leaves no final status: the job is `lost`, and the pipeline reruns it without
counting an attempt.

Slicing: the pipeline asks this backend how many parts a sweep of the real
design (distinct Sobol points) has (`sweep_parts`) and splits it with
generate_sim_config.py's own `--n-parts`; merge_sweeps.py joins them as for
Raven's two parts. Parts of about PF_LOCAL_PART_EPISODES episodes each
(default 4000, a slice of the feasibility report) keep every CPU of the pool
busy. A sim seeds its batch i with `seed + i` and part k starts after the
batches of the parts before it, so a split plays the same games as one whole
sim exactly when every part but the last fills whole batches
(docs/reports/dais_gpu_sim_equivalence.md). The smallest such split is
chosen; when none exists (exp2_multi: 512 episodes a point in batches of
1000), Raven's own split is used (for exp2_multi 2 parts, so 2 single-threaded
sims, slow but correct), and the pipeline records `parts_aligned: false`.
Either way the result does not depend on the pool's size.

Settings (environment):
  PF_LOCAL_ROOT          snapshots, jobs (default $PF_RUNS/local_cluster)
  PF_SIM_SIF             the sim image (algoinst.sif)
  PF_ARTIFACTS           the model files, bound over the snapshot's artifacts/
                         (they are not in git, docs/large_files.md; default:
                         the checkout's artifacts/)
  PF_LOCAL_CONTAINER     apptainer (default) or none: the host's python on
                         the snapshot (tests, development)
  PF_POOL_CPUS           the pool's CPUs, one slot each, "0,96,1,97" or
                         "0-3"; default: the CPUs this process may run on
  PF_POOL_LOCKS          the slots' lock files (default: /dev/shm, per Slurm
                         job; all pipelines of one job must share it)
  PF_SIM_THREADS         threads per sim (default 1; results are exactly
                         reproducible only at a fixed thread count)
  PF_LOCAL_PART_EPISODES episodes per sweep part (default 4000)
  PF_LOCAL_JOB_SECONDS   a sim is killed after this long (default 3600)
  PF_LOCAL_POLL_SECONDS  between queue polls in wait() (default 5)
  PF_LOCAL_KEEP_DAYS     done jobs' folders are kept this long (default 7)

A version started on Raven cannot resume here: its snapshots are Raven's, and
`submit` refuses a snapshot it does not have.
"""

import contextlib
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
WORKER_EXIT = {"timeout": 124, "setup": 125}


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


@contextlib.contextmanager
def _snapshot_lock(sha, wait=True):
    """The snapshot's lock: held while it is made or removed. Yields False
    (not taken) when `wait` is False and another process holds it."""
    folder = root() / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{sha}.lock", "w") as handle:
        try:
            # POSIX locks: GPFS honours them across nodes (flock it may not)
            fcntl.lockf(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except OSError:
            yield False
            return
        yield True


def snapshot(ref):
    """The files of commit `ref` in a directory of their own; its sha.

    The checkout's object store holds every instance branch (worktrees share
    it). The model files are not in git: they come from PF_ARTIFACTS.
    """
    result = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    if result.returncode != 0:
        raise _cp().ClusterError(f"no commit {ref}: {result.stderr.strip()}")
    sha = result.stdout.strip()
    target = tree(sha)
    with _snapshot_lock(sha):
        if (target.parent / "complete").exists():
            return sha
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(prefix="tree.", dir=target.parent))
        archive = subprocess.Popen(
            ["git", "-C", str(MAIN), "-c", "filter.lfs.smudge=", "-c",
             "filter.lfs.process=", "-c", "filter.lfs.required=false",
             "archive", "--format=tar", sha],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        untar = subprocess.run(["tar", "-x", "-C", str(partial)], stdin=archive.stdout,
                               capture_output=True)
        archive.stdout.close()
        err = archive.stderr.read().decode(errors="replace")
        if archive.wait() != 0 or untar.returncode != 0:
            shutil.rmtree(partial, ignore_errors=True)
            raise _cp().ClusterError(f"snapshot of {sha[:12]} failed: {err}"
                                     f"{untar.stderr.decode(errors='replace')}")
        (partial / "artifacts").mkdir(exist_ok=True)  # the models' bind point
        shutil.rmtree(target, ignore_errors=True)  # a half-made one (lock held)
        os.replace(partial, target)
        (target.parent / "complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return sha


def cleanup(sha):
    """Remove snapshot `sha` unless a live job runs on it or another process
    holds its lock; True if removed.

    The folder is renamed first (atomically), so a later snapshot() of the
    commit makes a fresh one rather than seeing a half-removed tree.
    """
    folder = root() / "snapshots" / sha
    with _snapshot_lock(sha, wait=False) as taken:
        if not taken or not folder.is_dir():
            return False
        # this host's live jobs, and every job another node's index lists
        others = [jobs_dir() / e.name for e in (jobs_dir() / "active").glob("*/*")
                  if e.parent.name != socket.gethostname()]
        for job in [*_live().values(), *others]:
            try:
                if json.loads((job / "job.json").read_text())["sha"] == sha:
                    return False
            except (OSError, ValueError, KeyError):
                return False  # a job being written: keep the snapshot
        doomed = folder.with_name(f"{sha}.removing.{os.getpid()}")
        os.replace(folder, doomed)
    shutil.rmtree(doomed, ignore_errors=True)
    return True


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
    """The slots' locks: node-local, per Slurm job (/dev/shm where there is
    one), so that waiting workers never poll the shared file system."""
    given = _env("PF_POOL_LOCKS")
    if given:
        return Path(given)
    base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    return Path(base) / f"pf-pool-{os.getuid()}-{_env('SLURM_JOB_ID', 'local')}"


def jobs_dir():
    return root() / "jobs"


def job_dir(job):
    if not re.fullmatch(r"local-\d+", str(job)):
        raise _cp().ClusterError(f"not a local job id: {job!r}")
    return jobs_dir() / str(job)


def active_dir():
    """This host's index of jobs not yet done: a queue poll reads only it, and
    never judges another node's workers (pipelines of several Slurm jobs may
    share one PF_LOCAL_ROOT)."""
    return jobs_dir() / "active" / socket.gethostname()


def _new_job():
    """A fresh job id and its folder. The counter is read under a POSIX lock
    and the folder made with mkdir, which fails if it exists: ids stay unique
    across processes and nodes."""
    jobs = jobs_dir()
    jobs.mkdir(parents=True, exist_ok=True)
    with open(jobs / "next_id.lock", "w") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        counter = jobs / "next_id"
        try:
            n = int(counter.read_text())
        except (OSError, ValueError):  # none yet, or cut short: past every id
            n = 1 + max((int(f.name[6:]) for f in jobs.glob("local-*")
                         if f.name[6:].isdigit()), default=0)
        while True:
            try:
                (jobs / f"local-{n}").mkdir()
                break
            except FileExistsError:
                n += 1
        tmp = jobs / f".next_id.{os.getpid()}.tmp"
        tmp.write_text(f"{n + 1}\n")
        os.replace(tmp, counter)
    return n


def _start_time(pid):
    """A process's start time (clock ticks since boot, Linux), None elsewhere:
    with the pid, it names one process even after the pid is reused."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def submit(config, sha, run_group="policy-finder"):
    """`python -m aimanager simulate <config>` on snapshot `sha`, into the
    pool. Returns the job id at once; the sim waits for a free slot."""
    if not (tree(sha).parent / "complete").exists():
        raise _cp().ClusterError(f"no snapshot {sha[:12]}: snapshot() it first")
    if not (tree(sha) / config).is_file():
        raise _cp().ClusterError(f"no {config} in snapshot {sha[:12]}")
    n = _new_job()
    if n % 50 == 0:
        prune()
    job = f"local-{n}"
    folder = job_dir(job)
    settings = {
        "job": job, "config": config, "sha": sha, "run_group": run_group,
        "root": str(root()),
        "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threads": threads(), "cpus": pool_cpus(), "locks": str(lock_dir()),
        "container": _env("PF_LOCAL_CONTAINER", "apptainer"),
        "image": _env("PF_SIM_SIF"), "artifacts": _env("PF_ARTIFACTS"),
        "timeout": int(_env("PF_LOCAL_JOB_SECONDS", "3600")),
    }
    _write_json(folder / "job.json", settings)
    active_dir().mkdir(parents=True, exist_ok=True)
    (active_dir() / job).touch()
    with open(folder / "worker.log", "w") as log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(folder)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=str(folder),
        )
    _write_json(folder / "worker.json", {"host": socket.gethostname(), "pid": worker.pid,
                                         "start": _start_time(worker.pid)})
    return job


def _take_slot(cpus, locks, poll=1.0):
    """Block until one of `cpus` is free; (cpu, its open lock file)."""
    locks.mkdir(parents=True, exist_ok=True)
    while True:
        for cpu in cpus:
            handle = open(locks / f"cpu{cpu}.lock", "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return cpu, handle
            except BlockingIOError:
                handle.close()
        time.sleep(poll)


def _write_status(folder, status):
    _write_json(folder / "status.json", status)


def _clear_output(repo, config):
    """Remove the config's output folder in the snapshot before its sim runs:
    a reused snapshot may hold an earlier sim's result there."""
    import yaml

    with open(repo / config) as f:
        out = (yaml.safe_load(f) or {}).get("output_dir")
    if not out:
        return
    target = (repo / out).resolve()
    if repo.resolve() not in target.parents:
        raise _cp().ClusterError(f"{config}: output_dir {out} is outside the snapshot")
    shutil.rmtree(target, ignore_errors=True)


def _write_json(path, data):
    """Write JSON through a rename, so a reader sees the old file or the new
    one, never half of it."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path):
    """A JSON file's content; {} if it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def worker(folder):
    """A pool job: wait for a slot, run the sim pinned to it, record the
    result. Its exit code is the sim's; a failure of the worker itself (its
    slot, pinning, starting the sim, its files) is recorded as a done job
    with exit code WORKER_EXIT["setup"] and the error, so that it reads as a
    failed job, never as one lost with its Slurm job."""
    folder = Path(folder)
    started, cpu = time.time(), None
    try:
        settings = json.loads((folder / "job.json").read_text())
        os.environ["PF_LOCAL_ROOT"] = settings["root"]
        os.environ["PF_LOCAL_CONTAINER"] = settings["container"]
        if settings["image"]:
            os.environ["PF_SIM_SIF"] = settings["image"]
        if settings["artifacts"]:
            os.environ["PF_ARTIFACTS"] = settings["artifacts"]
        cpu, lock = _take_slot(settings["cpus"], Path(settings["locks"]))
        started = time.time()
        _write_status(folder, {"state": "running", "cpu": cpu, "started_at": started})
        _clear_output(tree(settings["sha"]), settings["config"])
        (folder / "tmp").mkdir(exist_ok=True)
        cmd, cwd, env = argv(settings["sha"],
                             f"python -m aimanager simulate {settings['config']}",
                             folder / "tmp", settings["threads"])

        def pin():
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {cpu})

        with open(folder / "sim.log", "w") as log:
            sim = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   preexec_fn=pin, start_new_session=True)
            try:
                code = sim.wait(timeout=settings["timeout"])
            except subprocess.TimeoutExpired:
                os.killpg(sim.pid, signal.SIGKILL)
                sim.wait()
                code = WORKER_EXIT["timeout"]
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        _write_status(folder, {
            "state": "done", "exit_code": code, "cpu": cpu,
            "started_at": started, "finished_at": time.time(),
            "wall_seconds": round(time.time() - started, 1),
            "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 1),
            # kB on Linux
            "maxrss_kb": usage.ru_maxrss,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        shutil.rmtree(folder / "tmp", ignore_errors=True)
        lock.close()
    except Exception as e:  # noqa: BLE001 - a failed job, not a lost one
        code = WORKER_EXIT["setup"]
        print(f"worker failed: {e!r}", file=sys.stderr, flush=True)
        try:
            _write_status(folder, {"state": "done", "exit_code": code, "cpu": cpu,
                                   "error": repr(e), "started_at": started,
                                   "finished_at": time.time()})
        except OSError:
            pass  # status.json not writable: the job reads as lost
    try:
        (active_dir() / folder.name).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - the index is pruned by queue polls too
        pass
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
    except (ProcessLookupError, PermissionError):
        return False  # gone, or the pid now belongs to another user
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            return False
        if fields[0] == "Z":
            return False
        if info.get("start") and fields[19] != info["start"]:
            return False  # the pid was reused by a later process
    return True


def status(job):
    """A job's status.json, {} while it has none."""
    return _read_json(job_dir(job) / "status.json")


def _live():
    """{job id: its folder} of this host's jobs not yet done whose worker is
    alive. Reads only this host's index; an entry of a done or dead job is
    dropped."""
    active = active_dir()
    live = {}
    if not active.is_dir():
        return live
    for entry in active.iterdir():
        folder = jobs_dir() / entry.name
        info = _read_json(folder / "worker.json")
        if not info:
            # being submitted (worker.json not yet written), or its submit died
            try:
                young = time.time() - entry.stat().st_mtime < 60
            except OSError:
                continue  # removed meanwhile
            if young and folder.is_dir():
                live[entry.name] = folder
            else:
                entry.unlink(missing_ok=True)
            continue
        if status(entry.name).get("state") != "done" and _alive(info):
            live[entry.name] = folder
        else:
            entry.unlink(missing_ok=True)
    return live


def queue():
    """The pool's jobs not yet done whose worker is alive."""
    return set(_live())


def lost(job):
    """True if the job's worker died before its sim finished (killed with its
    Slurm job): not a sim that ran and failed. A worker on another node is
    treated as lost: this host cannot wait for it."""
    info = _read_json(job_dir(job) / "worker.json")
    if not info or status(job).get("state") == "done":
        return False
    return not _alive(info)


def prune(days=None):
    """Remove done jobs' folders older than PF_LOCAL_KEEP_DAYS."""
    days = float(_env("PF_LOCAL_KEEP_DAYS", "7")) if days is None else days
    cutoff = time.time() - days * 86400
    for folder in jobs_dir().glob("local-*"):
        state = status(folder.name)
        if state.get("state") == "done" and state.get("finished_at", cutoff) < cutoff:
            shutil.rmtree(folder, ignore_errors=True)


def wait(jobs, poll=None):
    """Until none of `jobs` is in the queue; a failed sim's log tail is printed
    (the pipeline then reruns it)."""
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
    """(parts, aligned) for a design of `points` distinct points: the smallest
    split into parts of at most PF_LOCAL_PART_EPISODES episodes that plays the
    games of one whole sim; without one, Raven's split (by its job size).
    Never more parts than points."""
    batch = SWEEP_BATCH[setting]
    target = int(_env("PF_LOCAL_PART_EPISODES", "4000"))
    total = points * per_point
    for n in range(max(1, math.ceil(total / target)), points + 1):
        if aligned(points, per_point, batch, n):
            return n, True
    n = min(points, max(1, math.ceil(total / max_job_episodes)))
    return n, aligned(points, per_point, batch, n)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        sys.exit(worker(sys.argv[2]))
    sys.exit("usage: local_cluster.py worker <job folder>")
