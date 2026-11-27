"""pipeline.py's image check: only validate-rule's own verdict fails a version."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "policy_finder"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("pipeline", HERE / "pipeline.py")
pl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pl)


@pytest.fixture
def version(tmp_path, monkeypatch):
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    monkeypatch.delenv("PF_WORKTREE_ROOT", raising=False)
    conf = tmp_path / "worktrees/pf-x/.claude/policy_finder.json"
    conf.parent.mkdir(parents=True)
    conf.write_text(json.dumps({"rule_name": "pf-x", "version": 1, "min_params": 1,
                                "max_params": 5, "sobol_points": 256}))
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)
    v = pl.Version("pf-x")
    v.data["sweep_sha"] = "a" * 40
    return v


def answers(monkeypatch, *logs):
    queue = list(logs)
    calls = []

    def fake(sha, rule, lo, hi, n, setting="exp2"):
        calls.append(rule)
        log = queue.pop(0)
        return log.startswith("Valid"), log

    monkeypatch.setattr(pl.cluster_proxy, "validate_rule", fake)
    return calls


def test_invalid_is_a_verdict(version, monkeypatch):
    answers(monkeypatch, "Traceback ...\nInvalid: the rule reads th.os\n")
    with pytest.raises(pl.Failed, match="th.os"):
        version.step_image_check()


def test_the_login_node_is_retried(version, monkeypatch):
    calls = answers(monkeypatch, "libgomp: Thread creation failed\n",
                    'FATAL: container creation failed: "fuseapps"\n', "Valid: 256 points\n")
    version.step_image_check()
    assert len(calls) == 3


def test_a_login_node_that_never_works_is_an_error(version, monkeypatch):
    answers(monkeypatch, *["  File numpy\nKeyboardInterrupt\n"] * pl.IMAGE_CHECK_TRIES)
    with pytest.raises(pl.cluster_proxy.ClusterError):
        version.step_image_check()


def test_a_second_pipeline_waits_and_sees_the_state(version, monkeypatch):
    import fcntl
    import threading
    holder = open(version.wt_root / "pf-x.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("state", version.run()))
    t.start()
    t.join(0.3)
    assert t.is_alive()  # waiting on the lock
    version.state_path.write_text(json.dumps({"status": "scored", "done": [], "data": {}}))
    fcntl.flock(holder, fcntl.LOCK_UN)
    t.join(5)
    assert result["state"]["status"] == "scored"


def test_a_fetch_during_an_outage_costs_no_job(version, monkeypatch):
    version.data["sweep_jobs"] = {"c.yml": ["1"]}
    submitted = []
    monkeypatch.setattr(pl.cluster_proxy, "submit", lambda *a, **k: submitted.append(a) or "2")

    def down(*a, **k):
        raise pl.cluster_proxy.ClusterError("could not look: Connection closed")

    monkeypatch.setattr(pl.cluster_proxy, "fetch", down)
    with pytest.raises(pl.cluster_proxy.ClusterError):
        version._fetch_all("sweep_jobs", "a" * 40, [("c.yml", "plots/x", version.wt_root / "x")])
    assert submitted == []

    def missing(*a, **k):
        raise pl.cluster_proxy.NoOutput("no plots/x on snapshot")

    monkeypatch.setattr(pl.cluster_proxy, "fetch", missing)
    monkeypatch.setattr(pl.cluster_proxy, "wait", lambda jobs: None)
    with pytest.raises(pl.Failed):
        version._fetch_all("sweep_jobs", "a" * 40, [("c.yml", "plots/x", version.wt_root / "x")])
    assert len(submitted) == pl.JOB_ATTEMPTS - 1


def test_transcripts_are_kept(version, tmp_path, monkeypatch):
    """Every agent run's session transcripts are copied next to the state file."""
    state = tmp_path / "spawner"
    projects = state / "runs/run-1/home/.claude/projects/-opt-repo"
    (projects / "sub").mkdir(parents=True)
    (projects / "a.jsonl").write_text("{}\n")
    (projects / "sub/b.jsonl").write_text("{}\n")
    (projects / "notes.txt").write_text("not a transcript\n")
    monkeypatch.setattr(pl.agent_runner, "SPAWNER_STATE", str(state))
    assert version.keep_transcripts("run-1") == 2
    kept = tmp_path / "worktrees/pf-x.transcripts/run-1/-opt-repo"
    assert (kept / "a.jsonl").exists() and (kept / "sub/b.jsonl").exists()
    assert not (kept / "notes.txt").exists()
    assert version.keep_transcripts("run-missing") == 0  # logged, not an error


def test_single_idea_version_1_is_the_seed(tmp_path, monkeypatch):
    """Version 1 of a single-idea instance: the seed rule and notes, no agent."""
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    monkeypatch.delenv("PF_WORKTREE_ROOT", raising=False)
    wt = tmp_path / "worktrees/pf-s"
    (wt / ".claude").mkdir(parents=True)
    (wt / ".claude/policy_finder.json").write_text(json.dumps(
        {"rule_name": "pf-s", "version": 1, "mode": "single_idea"}))
    v = pl.Version("pf-s")
    assert v.seeded
    monkeypatch.setattr(pl.agent_runner, "start", lambda *a: pytest.fail("agent started"))
    v.step_export()
    v.step_agent()
    v.step_collect()
    rule = (wt / "configs/managers/rule_based/pf-s.yml").read_text()
    assert "punishment = th.zeros_like(c[..., -1])" in rule
    # no parameters: the design is one point, so the sweep is one part
    monkeypatch.setattr(pl, "MAX_JOB_EPISODES", 1000)
    v.state = {"data": {"episodes": {"sweep": 500}}}
    v.conf["sobol_points"] = 256
    assert v.design_points() == 1 and v.parts() == ["pf-s_sweep"]
    (wt / "configs/managers/rule_based/pf-s.yml").write_text(
        "params: {a: {definition: d, type: float}}\nsweep_config: {a: [0, 1]}\n"
        "code: |\n  punishment = a\n")
    v.state = {"data": {"episodes": {"sweep": 500}}}  # the split is kept in the state
    assert v.design_points() == 256 and len(v.parts()) == 128
    notes = (wt / "notes/policy_finder/pf-s.md").read_text()
    assert notes.startswith("# pf-s\n") and "### Justification" in notes
    (wt / ".claude/policy_finder.json").write_text(json.dumps(
        {"rule_name": "pf-s-v2", "version": 2, "mode": "single_idea"}))
    assert not pl.Version("pf-s").seeded


def test_snapshots_go_only_when_scored(version, monkeypatch):
    removed = []
    monkeypatch.setattr(pl.cluster_proxy, "cleanup", lambda sha: removed.append(sha) or True)
    version.data["final_sha"] = "b" * 40
    version.state["done"] = list(pl.STEPS)
    assert version._run()["status"] == "scored"
    assert removed == ["a" * 40, "b" * 40]
    removed.clear()
    version.state.update(status="running", done=list(pl.STEPS[:-1]))
    monkeypatch.setattr(version, "step_final_fetch", lambda: (_ for _ in ()).throw(pl.Failed("x")))
    assert version._run()["status"] == "failed" and removed == []


class FakePool:
    """A backend whose jobs die with their runner in the first `kills` rounds
    (the pipeline is killed in wait, as with its Slurm job)."""

    def __init__(self, kills, truncate=()):
        self.kills, self.truncate = kills, set(truncate)
        self.n, self.config, self.dead, self.rounds = 0, {}, set(), []

    def submit(self, config, sha, run_group=None):
        self.n += 1
        job = f"local-{self.n}"
        self.config[job] = config
        return job

    def wait(self, jobs):
        self.rounds.append(sorted(jobs))
        if self.kills:
            self.kills -= 1
            self.dead.update(jobs)
            raise RuntimeError("killed with the Slurm job")

    def lost(self, job):
        return job in self.dead

    def fetch(self, sha, remote, local, key=None):
        job = max((j for j, c in self.config.items() if remote.endswith(
            Path(c).stem)), key=lambda j: int(j.split("-")[1]))
        if job in self.dead:
            raise pl.cluster_proxy.NoOutput("none")
        Path(local).mkdir(parents=True, exist_ok=True)
        text = json.dumps({"points": [{"name": "x"}]})
        if remote in self.truncate:
            self.truncate.discard(remote)
            text = text[:10]
        (Path(local) / "sweep.json").write_text(text)
        return 1


def test_killed_twice_mid_sweep_resumes(version, monkeypatch):
    pool = FakePool(kills=2, truncate={f"{pl.OUT_DIR}/pf-x_sweep_p3of4"})
    for name in ("submit", "wait", "lost", "fetch"):
        monkeypatch.setattr(pl.cluster_proxy, name, getattr(pool, name))
    version.data.update(episodes={"sweep": 500, "final": 2000}, n_parts=4)
    for _ in range(2):  # two kills: each resume resubmits all four together
        with pytest.raises(RuntimeError):
            version.step_sweep_jobs()
    version.step_sweep_jobs()
    version.step_sweep_fetch()  # the truncated part is rerun once, as an attempt
    assert [len(r) for r in pool.rounds] == [4, 4, 4, 1]
    jobs = version.data["sweep_jobs"]
    assert sorted(len(j) for j in jobs.values()) == [1, 1, 1, 2]
    assert all(len(j) == 2 for j in version.data["sweep_jobs_lost"].values())


def test_the_split_is_of_the_real_design_and_kept(version, monkeypatch):
    sys.path.insert(0, str(HERE))
    import local_cluster

    monkeypatch.setattr(pl.cluster_proxy, "sweep_parts", local_cluster.sweep_parts)
    version.data.update(episodes={"sweep": 500, "final": 2000}, design_points=11)
    assert len(version.parts()) == 2 and version.data["parts_aligned"] is True
    monkeypatch.setenv("PF_LOCAL_PART_EPISODES", "1000")  # a resume keeps the split
    assert len(version.parts()) == 2


def test_jobs_dying_while_the_pipeline_runs_fail_the_version(version, monkeypatch):
    """Workers that die without a status every time are not rerun forever."""
    pool = FakePool(kills=0)
    pool.wait = lambda jobs: (pool.rounds.append(sorted(jobs)), pool.dead.update(jobs))
    for name in ("submit", "wait", "lost", "fetch"):
        monkeypatch.setattr(pl.cluster_proxy, name, getattr(pool, name))
    monkeypatch.setattr(pl, "LOST_PAUSE", 0)
    version.data.update(episodes={"sweep": 500, "final": 2000}, n_parts=2)
    with pytest.raises(pl.Failed, match="keep dying"):
        version.step_sweep_jobs()
    assert len(pool.rounds) == pl.LOST_ROUNDS + 1


def test_the_version_lock_holds_across_processes(version, tmp_path):
    """A second pipeline waits for the POSIX lock another process holds."""
    import subprocess
    import time

    lock = version.wt_root / f"{version.stem}.lock"
    holder = subprocess.Popen([sys.executable, "-c", (
        "import fcntl, sys, time; f = open(sys.argv[1], 'w');"
        " fcntl.lockf(f, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(1.5)"),
        str(lock)], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "locked"
    version.state["status"] = "scored"
    version.save()
    started = time.time()
    assert version.run()["status"] == "scored"
    assert time.time() - started > 0.5
    assert "another pipeline runs this version" in version.log_path.read_text()
    holder.wait()


def test_a_failed_sweep_restarts_as_the_readme_says(version, monkeypatch):
    """README: status error, the jobs step out of done, its lists deleted."""
    pool = FakePool(kills=0)
    for name in ("submit", "wait", "lost", "fetch"):
        monkeypatch.setattr(pl.cluster_proxy, name, getattr(pool, name))
    version.data.update(episodes={"sweep": 500, "final": 2000}, n_parts=2,
                        sweep_jobs={"c": ["local-1", "local-2"]})
    version.state.update(status="error", done=["export", "agent", "collect", "check",
                                               "sweep_config", "sweep_snap", "image_check"])
    version.data.pop("sweep_jobs")
    version.step_sweep_jobs()
    version.step_sweep_fetch()
    assert all(len(j) == 1 for j in version.data["sweep_jobs"].values())
