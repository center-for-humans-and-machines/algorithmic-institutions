"""Tests for scripts/policy_finder/local_cluster.py: the switch, the slicing,
snapshots and the pool, on the host (PF_LOCAL_CONTAINER=none) with a fake
`python -m aimanager simulate`."""

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "policy_finder"
sys.path.insert(0, str(HERE))
import local_cluster as lc  # noqa: E402

FAKE_SIM = """import json, os, sys
config = sys.argv[2]
out = open(config).read().split()[1]
if "fail" in config:
    sys.exit(3)
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
                 "sweep_parts"):
        assert getattr(local, name) is getattr(lc, name)
    with pytest.raises(ImportError):
        load_proxy(monkeypatch, "slurm")


@pytest.mark.parametrize("points, per_point, setting, expected", [
    (256, 500, "exp2", (32, True)),     # a real sweep: 32 slices of 4000
    (16, 20, "exp2", (1, True)),        # the smoke size
    (1, 500, "exp2", (1, True)),        # the seed: one point
    (256, 500, "exp2_diff", (32, True)),
    (256, 1024, "exp2_multi", (3, False)),  # no aligned split: Raven's
    (100, 500, "exp1", (25, True)),     # 13 to 24 parts give odd ones
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
    for name in ("ok1", "ok2", "fail"):
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
