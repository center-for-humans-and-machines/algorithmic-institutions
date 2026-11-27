"""Tests for scripts/policy_finder/local_cluster.py: the switch, the slicing,
snapshots and the pool, on the host (PF_LOCAL_CONTAINER=none) with a fake
`python -m aimanager simulate`."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "policy_finder"
sys.path.insert(0, str(HERE))
import local_cluster as lc  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "reevaluate", HERE.parents[1] / "infra/dais/reevaluate.py")
rv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rv)

FAKE_SIM = """import json, os, sys, time
config = sys.argv[2]
out = open(config).read().split()[1]
if "fail" in config or os.path.exists("FAIL_ALL"):
    sys.exit(3)
if "slow" in config:
    time.sleep(30)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "sweep.json"), "w") as f:
    json.dump({"threads": os.environ["OMP_NUM_THREADS"]}, f)
"""


def load_proxy(monkeypatch, backend):
    monkeypatch.setenv("PF_CLUSTER_BACKEND", backend)
    spec = importlib.util.spec_from_file_location("cluster_proxy_t", HERE / "cluster_proxy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_switch(monkeypatch):
    assert load_proxy(monkeypatch, "raven").submit.__module__ == "cluster_proxy_t"
    local = load_proxy(monkeypatch, "local")
    for name in ("snapshot", "run", "validate_rule", "submit", "queue", "wait", "fetch",
                 "sweep_parts", "cleanup"):
        assert getattr(local, name) is getattr(lc, name)
    with pytest.raises(ImportError):
        load_proxy(monkeypatch, "slurm")


@pytest.mark.parametrize("points, per_point, setting, expected", [
    (256, 500, "exp2", (32, True)),     # a real sweep: 32 slices of 4000
    (16, 20, "exp2", (1, True)),        # the smoke size
    (1, 500, "exp2", (1, True)),        # the seed: one point
    (256, 500, "exp2_diff", (32, True)),
    (256, 512, "exp2_multi", (2, False)),  # no aligned split: Raven's
    (100, 500, "exp1", (25, True)),     # 13 to 24 parts give odd ones
    (11, 500, "exp2", (2, True)),       # a design of few distinct points
    (3, 100, "exp2", (1, True)),        # never more parts than points
])
def test_sweep_parts(monkeypatch, points, per_point, setting, expected):
    monkeypatch.delenv("PF_LOCAL_PART_EPISODES", raising=False)
    assert lc.sweep_parts(points, per_point, setting, 125000) == expected


def test_batches_match_generate_sim_config():
    gsc = pytest.importorskip("generate_sim_config")
    import yaml

    for setting, spec in gsc.SETTINGS.items():
        template = yaml.safe_load((gsc.ROOT / spec["template"]).read_text())
        batch = spec["overrides"].get("episode_batch_size",
                                      template.get("episode_batch_size", 1))
        assert lc.SWEEP_BATCH[setting] == (None if spec.get("common_games") else batch)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo with a fake simulator, and the pool's settings."""
    main = tmp_path / "main"
    (main / "src/aimanager").mkdir(parents=True)
    (main / "src/aimanager/__init__.py").write_text("")
    (main / "src/aimanager/__main__.py").write_text(FAKE_SIM)
    for name in ("ok1", "ok2", "fail", "slow"):
        (main / f"configs/{name}.yml").parent.mkdir(exist_ok=True)
        (main / f"configs/{name}.yml").write_text(f"output_dir: plots/{name}\n")
    git = ["git", "-C", str(main)]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"],
                   check=True)
    monkeypatch.setattr(lc, "MAIN", main)
    monkeypatch.setenv("PF_LOCAL_ROOT", str(tmp_path / "pool"))
    monkeypatch.setenv("PF_POOL_LOCKS", str(tmp_path / "locks"))
    monkeypatch.setenv("PF_LOCAL_CONTAINER", "none")
    monkeypatch.setenv("PF_POOL_CPUS", "0")
    monkeypatch.delenv("PF_SIM_THREADS", raising=False)
    return main


def test_snapshot_pool_and_fetch(repo, tmp_path):
    sha = lc.snapshot("HEAD")
    assert lc.snapshot(sha) == sha  # reused
    assert (lc.tree(sha) / "configs/ok1.yml").exists()
    jobs = [lc.submit(f"configs/{n}.yml", sha) for n in ("ok1", "ok2", "fail")]
    assert len(set(jobs)) == 3
    lc.wait(jobs, poll=0.2)
    assert not lc.queue()
    codes = [lc.status(j)["exit_code"] for j in jobs]
    assert codes == [0, 0, 3]
    # one slot: the three sims ran one after another
    spans = sorted((lc.status(j)["started_at"], lc.status(j)["finished_at"]) for j in jobs)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert not list(lc.active_dir().iterdir())  # the index is emptied
    assert lc.fetch(sha, "plots/ok1", tmp_path / "out") > 0
    # one thread per sim, and the settings kept with the job
    assert json.loads((tmp_path / "out/sweep.json").read_text()) == {"threads": "1"}
    assert json.loads((lc.job_dir(jobs[0]) / "job.json").read_text())["threads"] == 1
    with pytest.raises(lc._cp().NoOutput):
        lc.fetch(sha, "plots/fail", tmp_path / "none")


def test_a_dead_worker_leaves_the_queue(repo):
    folder = lc.root() / "jobs" / "local-99"
    folder.mkdir(parents=True)
    done = subprocess.Popen(["true"])
    done.wait()
    (folder / "worker.json").write_text(
        json.dumps({"host": socket.gethostname(), "pid": done.pid}))
    assert "local-99" not in lc.queue()
    (folder / "worker.json").write_text(json.dumps({"host": "elsewhere", "pid": 1}))
    assert "local-99" not in lc.queue()


def test_cleanup_keeps_a_snapshot_in_use(repo):
    sha = lc.snapshot("HEAD")
    folder = lc.root() / "jobs" / "local-7"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"sha": sha}))
    (folder / "worker.json").write_text(
        json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    lc.active_dir().mkdir(parents=True)
    (lc.active_dir() / "local-7").touch()
    assert not lc.cleanup(sha) and lc.tree(sha).exists()  # a live job runs on it
    (folder / "status.json").write_text(json.dumps({"state": "done"}))
    assert lc.cleanup(sha) and not (lc.root() / "snapshots" / sha).exists()
    assert lc.snapshot(sha) == sha and lc.tree(sha).exists()  # made afresh


def test_reevaluate_keeps_only_the_gp_peak():
    final = {"seed": 1000, "n_episodes": 2000, "episode_batch_size": 1000,
             "managers": {"ah": {}, "r-v2_gp": {"params": {"a": 1}}, "r-v2_s003": {}},
             "pairings": [{"name": "x", "group_0": "r-v2_gp", "group_1": "ah"},
                          {"name": "y", "group_0": "r-v2_s003", "group_1": "ah"}]}
    config = rv.gp_config(final, "r-v2_gp", 100000, "plots/x")
    assert list(config["managers"]) == ["ah", "r-v2_gp"] and len(config["pairings"]) == 1
    assert config["seed"] == 100000 and rv.batches(config) == 2
    assert rv.combine([(70.0, 1.2), (72.0, 1.2)]) == (71.0, pytest.approx(1.2 / 2 ** 0.5))


def test_reevaluate_curve_selects_on_recorded_scores():
    rows = [
        {"generation": 1, "stem": "a", "recorded": 70.0, "recorded_se": 1,
         "reevaluated": 68.0, "reevaluated_se": 0.3},
        {"generation": 1, "stem": "b", "recorded": 69.0, "recorded_se": 1,
         "reevaluated": 69.5, "reevaluated_se": 0.3},
        {"generation": 2, "stem": "b-v2", "recorded": None, "recorded_se": None,
         "reevaluated": None, "reevaluated_se": None},  # no final
        {"generation": 3, "stem": "a-v3", "recorded": 71.0, "recorded_se": 1,
         "reevaluated": 70.0, "reevaluated_se": 0.4},
    ]
    # common seeds: slice j of each version shares a large game effect
    slices = {"a": [(60.0, 1), (70.0, 1), (74.0, 1)],
              "a-v3": [(62.5, 1), (71.5, 1), (76.0, 1)]}
    points = rv.curve(rows, slices)
    assert [p["best_so_far"] for p in points] == ["a", "a", "a-v3"]
    gain = rv.improvement(points)
    assert gain["improvement"] == pytest.approx(2.0) and gain["bar_70"] == pytest.approx(1.4)
    # paired: the se of the differences (2.5, 1.5, 2.0), not of the scores
    assert gain["improvement_se"] == pytest.approx(0.5 / 3 ** 0.5)


@pytest.mark.parametrize("points", [11, 151, 230, 255])
def test_aligned_parts_play_the_whole_sims_batches(points):
    """A split marked aligned starts every part at the whole sim's batch (the
    seeds build_sim_configs gives the parts); one not marked aligned does not."""
    gsc = pytest.importorskip("generate_sim_config")
    rule = gsc.ROOT / "configs/managers/rule_based/levin_sigmoid.yml"
    design = [{"x": float(i)} for i in range(points)]
    n, ok = lc.sweep_parts(points, 500, "exp2", 125000)
    assert 1 <= n <= points
    configs = gsc.build_sim_configs(rule, design, 500, 42, n, "exp2")
    starts = lc.edges(points, n)
    batch = configs[0][1]["episode_batch_size"]
    matches = [cfg["seed"] - 42 == starts[k] * 500 / batch
               for k, (_, cfg) in enumerate(configs)]
    assert all(matches) == ok


def test_lost_worker_and_timeout(repo, monkeypatch):
    sha = lc.snapshot("HEAD")
    folder = lc.root() / "jobs" / "local-98"
    folder.mkdir(parents=True)
    done = subprocess.Popen(["true"])
    done.wait()
    (folder / "worker.json").write_text(
        json.dumps({"host": socket.gethostname(), "pid": done.pid}))
    (folder / "status.json").write_text(json.dumps({"state": "running"}))
    assert lc.lost("local-98") and "local-98" not in lc.queue()
    monkeypatch.setenv("PF_LOCAL_JOB_SECONDS", "1")
    job = lc.submit("configs/slow.yml", sha)
    lc.wait([job], poll=0.2)
    assert lc.status(job)["exit_code"] == 124 and not lc.lost(job)


def test_a_rerun_never_returns_an_old_result(repo, tmp_path):
    sha = lc.snapshot("HEAD")
    first = lc.submit("configs/ok1.yml", sha)
    lc.wait([first], poll=0.2)
    assert (lc.tree(sha) / "plots/ok1/sweep.json").exists()
    (lc.tree(sha) / "FAIL_ALL").touch()  # the same config, now failing
    again = lc.submit("configs/ok1.yml", sha)
    lc.wait([again], poll=0.2)
    assert lc.status(again)["exit_code"] == 3
    with pytest.raises(lc._cp().NoOutput):
        lc.fetch(sha, "plots/ok1", tmp_path / "stale")


def test_another_nodes_jobs_are_left_alone(repo):
    """A queue poll never drops another node's index entries."""
    other = lc.jobs_dir() / "active" / "other-node"
    other.mkdir(parents=True)
    (other / "local-5").touch()
    folder = lc.jobs_dir() / "local-5"
    folder.mkdir()
    (folder / "worker.json").write_text(json.dumps({"host": "other-node", "pid": 1}))
    assert lc.queue() == set() and (other / "local-5").exists()
    assert lc.lost("local-5")  # this host cannot wait for it: rerun here


def test_a_worker_that_cannot_start_is_failed_not_lost(repo, tmp_path, monkeypatch):
    sha = lc.snapshot("HEAD")
    (tmp_path / "not-a-dir").write_text("")
    monkeypatch.setenv("PF_POOL_LOCKS", str(tmp_path / "not-a-dir"))
    job = lc.submit("configs/ok1.yml", sha)
    lc.wait([job], poll=0.2)
    assert lc.status(job)["exit_code"] == lc.WORKER_EXIT["setup"] and not lc.lost(job)


def test_half_written_files_are_tolerated(repo):
    lc.active_dir().mkdir(parents=True)
    folder = lc.jobs_dir() / "local-3"
    folder.mkdir()
    (folder / "worker.json").write_text("")  # a poll between creation and write
    (lc.active_dir() / "local-3").touch()
    assert lc.queue() == {"local-3"} and not lc.lost("local-3")
    (lc.jobs_dir() / "next_id").write_text("")  # cut short by a kill
    sha = lc.snapshot("HEAD")
    assert lc.submit("configs/ok1.yml", sha) == "local-4"


def test_prune_removes_old_done_and_lost_jobs_only(repo):
    import time

    old = time.time() - 30 * 86400
    jobs = lc.jobs_dir()
    for name, state in (("local-1", {"state": "done", "finished_at": old}),
                        ("local-2", {"state": "running"}),  # lost: worker gone
                        ("local-3", {"state": "done", "finished_at": time.time()})):
        (jobs / name).mkdir(parents=True)
        (jobs / name / "status.json").write_text(json.dumps(state))
        (jobs / name / "worker.json").write_text(
            json.dumps({"host": socket.gethostname(), "pid": 999999999}))
        if name != "local-3":
            os.utime(jobs / name, (old, old))
    lc.prune(days=7)
    assert sorted(p.name for p in jobs.glob("local-*")) == ["local-3"]
    (jobs / "local-4").mkdir()
    os.utime(jobs / "local-4", (old, old))
    lc.prune(days=7)  # within the hour: not run again
    assert (jobs / "local-4").exists()


def test_cleanup_drops_entries_of_gone_slurm_jobs(repo, monkeypatch):
    sha = lc.snapshot("HEAD")
    other = lc.jobs_dir() / "active" / "node-a"
    other.mkdir(parents=True)
    (other / "local-8").touch()
    folder = lc.jobs_dir() / "local-8"
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({"sha": sha}))
    (folder / "worker.json").write_text(
        json.dumps({"host": "node-a", "pid": 1, "slurm_job": "42"}))
    monkeypatch.setattr(lc, "_slurm_job_gone", lambda job: False)
    assert not lc.cleanup(sha)  # node-a's job may still run on it
    monkeypatch.setattr(lc, "_slurm_job_gone", lambda job: job == "42")
    assert lc.cleanup(sha) and not (other / "local-8").exists()


def test_reevaluate_robust_improvements():
    rows = [{"generation": 1, "stem": s, "recorded": r} for s, r in
            (("a", 70.5), ("b", 70.3))] + [{"generation": 2, "stem": "c", "recorded": 73.0}]
    games = [0.0, 4.0, -2.0, 1.0, 3.0, -1.0]  # shared by all rules, slice by slice
    slices = {"a": [(69.8 + g, 1) for g in games], "b": [(70.1 + g, 1) for g in games],
              "c": [(73.4 + g + 0.1 * (-1) ** j, 1) for j, g in enumerate(games)]}
    robust = rv.robust(rows, slices)
    assert robust["runner_up"]["first"] == "b"
    assert robust["runner_up"]["improvement"] == pytest.approx(3.3)
    assert robust["in_sample"]["first"] == "b"  # chosen by re-evaluated score
    assert robust["cross_fit"]["improvement"] == pytest.approx(3.3)
    lo, hi = robust["runner_up"]["ci95"]
    assert lo < 3.3 < hi


def test_reevaluate_plays_other_code_and_needs_its_branches(tmp_path, monkeypatch):
    main = tmp_path / "m"
    (main / "src/aimanager").mkdir(parents=True)
    git = ["git", "-C", str(main), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    for text in ("old", "new"):
        (main / "src/aimanager/sim.py").write_text(text)
        (main / "rule.yml").write_text(f"rule {text}")
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["commit", "-qm", text], check=True)
    monkeypatch.setattr(rv, "MAIN", main)
    old = rv.git("rev-parse", "HEAD~1")
    mixed = rv.with_code(old, "HEAD")
    assert rv.git("show", f"{mixed}:src/aimanager/sim.py") == "new"
    assert rv.git("show", f"{mixed}:rule.yml") == "rule old"
    with pytest.raises(SystemExit, match="no branch"):
        rv.version_files("origin/policy-finder/missing", "missing")
