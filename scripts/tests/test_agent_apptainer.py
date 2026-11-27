"""Tests for scripts/policy_finder/agent_apptainer.py (no Apptainer needed)."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "policy_finder"
sys.path.insert(0, str(HERE))
import agent_apptainer as aa  # noqa: E402
import agent_runner as ar  # noqa: E402
import export_instance as ex  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    export = runs / "exports" / "pf-x-v2"
    (export / ".opencode").mkdir(parents=True)
    conf = ex.opencode_config({"model": "Qwen/Qwen3-Coder-Next",
                               "base_url": "{env:PF_LLM_BASE_URL}"})
    (export / ".opencode/opencode.json").write_text(json.dumps(conf))
    sif = tmp_path / "agent.sif"
    sif.write_text("")
    key = tmp_path / "key"
    key.write_text("s3cret\n")
    for k, v in {"PF_RUNS": str(runs), "PF_AGENT_SIF": str(sif),
                 "PF_LLM_BASE_URL": "http://127.0.0.1:4242/v1",
                 "PF_LLM_API_KEY_FILE": str(key), "PF_AGENT_POOL_CPUS": "8-11",
                 "PF_AGENT_SHELL": "/x/agent_shell.sh", "PF_AGENT_NOINET": "/x/noinet.py"
                 }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("PF_AGENT_STEP", raising=False)
    monkeypatch.delenv("PF_AGENT_ROOT", raising=False)
    monkeypatch.setattr(aa, "server_health", lambda: (True, "healthy"))
    return tmp_path


def test_argv_binds_command_and_key(env):
    run = aa.root() / "apt-1"
    cmd = aa.argv("pf-x-v2", ["notes/policy_finder"], "Design.", run, [8, 9])
    assert cmd[:3] == ["taskset", "-c", "8,9"]
    binds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
    export = env / "runs/exports/pf-x-v2"
    assert f"{export}:{ar.REPO}:ro" in binds
    assert f"{export}/notes/policy_finder:{ar.REPO}/notes/policy_finder:rw" in binds
    assert "/x/agent_shell.sh:/opt/pf-agent/shell:ro" in binds
    assert f"{run}/home/.opencode:{ar.HOME}/.opencode:ro" in binds
    assert "--containall" in cmd and f"{run}/home:{ar.HOME}" in cmd
    assert "SHELL=/opt/pf-agent/shell" in cmd
    assert "PF_LLM_BASE_URL=http://127.0.0.1:4242/v1" in cmd
    inner = cmd[cmd.index(str(env / "agent.sif")) + 1:]
    assert inner[:2] == ["python3", "-c"]
    assert inner[inner.index("--key-env") + 1] == "PF_LLM_API_KEY"
    assert "--usage-url" not in inner and "--require-key-limit" not in inner
    # the stray-process kill is the spawner's (Docker's PID 1) only
    assert "--kill-strays" not in cmd
    assert inner[inner.index("--timeout") + 1] == str(ar.TIMEOUT - 600)
    assert f"{aa.GLOBAL_CONFIG}:{aa.GLOBAL_CONFIG_TARGET}:ro" in binds
    # the key never on a command line, only in the worker's environment
    assert not any("s3cret" in a for a in cmd)
    assert aa.secret_env("pf-x-v2") == {"APPTAINERENV_PF_LLM_API_KEY": "s3cret"}


def test_argv_in_a_job_step(env, monkeypatch):
    monkeypatch.setenv("PF_AGENT_STEP", "1")
    cmd = aa.argv("pf-x-v2", [], "p", aa.root() / "apt-1", [8, 9])
    assert cmd[:2] == ["srun", "--overlap"]
    assert f"--mem={ar.MEMORY_MB}M" in cmd and "--cpus-per-task=2" in cmd
    assert cmd[cmd.index("taskset"):][:3] == ["taskset", "-c", "8,9"]


def test_cpu_groups_and_slot_limit(env, tmp_path):
    assert aa.cpu_groups(size=2) == [[8, 9], [10, 11]]
    groups = aa.cpu_groups([0, 1, 2], size=2)
    assert groups == [[0, 1]]  # a short last group is dropped
    first, h1 = aa._take_group([[0, 1], [2, 3]], tmp_path / "locks")
    second, h2 = aa._take_group([[0, 1], [2, 3]], tmp_path / "locks")
    assert (first, second) == ([0, 1], [2, 3])
    h1.close()
    third, h3 = aa._take_group([[0, 1], [2, 3]], tmp_path / "locks")
    assert third == [0, 1]
    h2.close(), h3.close()


def fake_apptainer(tmp_path, body):
    """An `apptainer` on PATH that runs `body` (bash) instead of a container."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for name, text in (("apptainer", body), ("taskset", 'shift 2; exec "$@"')):
        f = bindir / name
        f.write_text("#!/bin/bash\n" + text + "\n")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return bindir


def test_start_finish_and_timeout(env, monkeypatch):
    out = json.dumps({"harness": "opencode", "result": "my report",
                      "tokens": {"input": 5}, "models": ["local/Qwen/Qwen3-Coder-Next"]})
    bindir = fake_apptainer(env, f"echo noise; echo '{out}'; "
                            'mkdir -p "$PF_FAKE_HOME"/opencode; echo x > "$PF_FAKE_HOME"/opencode/'
                            "events.jsonl; [ -n \"$APPTAINERENV_PF_LLM_API_KEY\" ] || exit 9")
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setattr(aa, "POLL", 0.2)
    monkeypatch.setenv("PF_FAKE_HOME", str(aa.root() / "apt-1" / "home"))
    run_id = aa.start("pf-x-v2", ["notes/policy_finder"], "Design.")
    assert run_id == "apt-1"
    record = aa.finish(run_id, log=lambda m: None)
    assert record["status"] == "succeeded", record
    assert record["report"] == "my report" and record["result"]["tokens"] == {"input": 5}
    assert record["cpus"] in ([8, 9], [10, 11])
    assert record["network_audit"] == {"tool_calls": 0, "findings": []}

    # the runner's verdict: a summary with is_error fails a run that exited 0
    bad = json.dumps({"result": "r", "is_error": True, "failures": ["the session used x"]})
    fake_apptainer(env, f"echo '{bad}'")
    record = aa.finish(aa.start("pf-x-v2", [], "p"), log=lambda m: None)
    assert record["status"] == "failed" and "the session used x" in record["error"]

    monkeypatch.setattr(ar, "TIMEOUT", 1)
    fake_apptainer(env, "sleep 30")
    record = aa.finish(aa.start("pf-x-v2", [], "p"), log=lambda m: None)
    assert record["status"] == "expired" and "limit" in record["error"]


def test_a_failed_run_on_a_failing_server_is_not_an_attempt(env, monkeypatch):
    bindir = fake_apptainer(env, "echo '{\"result\": \"\", \"is_error\": true}'; exit 4")
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setattr(aa, "POLL", 0.2)
    # the server is down: the job's fault
    monkeypatch.setattr(aa, "server_health", lambda: (False, "ConnectionRefusedError"))
    with pytest.raises(ar.EndpointError, match="server is down") as e:
        aa.finish(aa.start("pf-x-v2", [], "p"), log=lambda m: None)
    assert e.value.record["status"] == "failed"
    # the server is fine but the session saw a server error: still the job's
    monkeypatch.setattr(aa, "server_health", lambda: (True, "healthy"))
    bad = json.dumps({"result": "", "is_error": True,
                      "errors": ['{"name": "APIError", "statusCode": 503}']})
    fake_apptainer(env, f"echo '{bad}'; exit 4")
    with pytest.raises(ar.EndpointError, match="endpoint error"):
        aa.finish(aa.start("pf-x-v2", [], "p"), log=lambda m: None)
    # an error only in the runner's failures (OpenCode's final error event)
    quoted = json.dumps({"result": "", "is_error": True, "failures": [
        "OpenCode reported an error at the end: {\"message\": \"fetch failed\"}"]})
    fake_apptainer(env, f"echo '{quoted}'; exit 4")
    with pytest.raises(ar.EndpointError, match="endpoint error"):
        aa.finish(aa.start("pf-x-v2", [], "p"), log=lambda m: None)
    # the model's own failure (its time limit, no report): one of its attempts
    own = json.dumps({"result": "", "is_error": True,
                      "failures": ["OpenCode stopped at its timeout of 60 s"]})
    fake_apptainer(env, f"echo '{own}'; exit 4")
    record = aa.finish(aa.start("pf-x-v2", [], "p"), log=lambda m: None)
    assert record["status"] == "failed" and "timeout" in record["error"]


def test_a_dead_worker_is_lost(env):
    run = aa.root() / "apt-7"
    (run / "home").mkdir(parents=True)
    (run / "worker.json").write_text(json.dumps({"host": "elsewhere", "pid": 1}))
    (run / "status.json").write_text(json.dumps({"status": "running"}))
    record = aa.finish("apt-7", log=lambda m: None)
    assert record["status"] == "lost"


def test_audit_finds_network_use(tmp_path):
    def tool(name, args, status="completed"):
        return {"type": "tool", "tool": name, "state": {"status": status, "input": args}}

    session = {"messages": [{"info": {"role": "assistant"}, "parts": [
        tool("bash", {"command": "ls notes; python -m aimanager validate-rule r.yml"}),
        tool("bash", {"command": "curl -s https://example.org/x | head"}, "error"),
        tool("bash", {"command": "pip install seaborn && git clone https://github.com/a/b"}),
        tool("bash", {"command": "python3 - <<EOF\nimport socket\nEOF\n"
                      "echo > /dev/tcp/1.1.1.1/80"}),
        tool("webfetch", {"url": "https://arxiv.org"}),
        tool("write", {"filePath": "scripts/a.py", "content": "import requests\n"}),
        tool("write", {"filePath": "notes/n.md", "content": "see https://doi.org/10.1"}),
    ]}]}
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session))
    found = aa.audit([str(path)])
    assert found["tool_calls"] == 7
    kinds = [(f["tool"], f["kind"]) for f in found["findings"]]
    assert kinds == [("bash", "shell"), ("bash", "shell"), ("bash", "shell"),
                     ("webfetch", "web tool"), ("write", "network library")]
    assert {"import socket", "/dev/tcp/1.1.1.1/80"} <= set(found["findings"][2]["matches"])
    assert found["findings"][0]["failed"] is True
    assert "https://example.org/x" in found["findings"][0]["matches"]
    assert {"pip install", "git clone"} <= set(found["findings"][1]["matches"])


def test_local_model_config_has_sampling_and_env_url():
    conf = ex.opencode_config({"model": "Qwen/Qwen3-Coder-Next",
                               "base_url": "{env:PF_LLM_BASE_URL}"})
    local = conf["provider"]["local"]
    assert local["options"]["baseURL"] == "{env:PF_LLM_BASE_URL}"
    model = local["models"]["Qwen/Qwen3-Coder-Next"]
    assert model["options"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 40}
    assert model["limit"]["context"] == 131072
