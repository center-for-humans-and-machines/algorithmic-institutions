"""Tests for scripts/policy_finder/cluster_proxy.py against a stub command server."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/cluster_proxy.py"
SPEC = importlib.util.spec_from_file_location("cluster_proxy", SCRIPT)
cp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cp)

SHA = "a" * 40


class Stub:
    """The command server's endpoints the client uses, with scripted answers."""

    def __init__(self, log_dir, transfer):
        self.log_dir, self.transfer = log_dir, transfer
        self.calls, self.busy, self.queue_answers = [], 0, []
        self.n = 0
        self.exit_codes = {}
        self.fetch_answer = None  # (log, exit code) for the remote copy

    def start(self, kind, log, code=0, extra=None):
        self.n += 1
        run_id = f"r{self.n}_{kind}"
        (self.log_dir / f"{run_id}.log").write_text(log)
        self.exit_codes[run_id] = code
        return {"run_id": run_id, "output_file": "x", **(extra or {})}

    def handle(self, method, path, body):
        self.calls.append((method, path, body))
        if path.startswith("/runs/"):
            run_id = path.split("/")[2]
            return 200, {"status": "exited", "exit_code": self.exit_codes[run_id]}
        if path == "/sync_to_cluster":
            return 200, self.start("sync", f"publishing\n{SHA}\n", extra={"sha": SHA})
        if path == "/submit_slurm_job":
            if self.busy:
                self.busy -= 1
                return 409, {"detail": "at most 16 jobs"}
            return 200, self.start("submit", "Submitted batch job 123\n")
        if path.startswith("/jobs"):
            answer = self.queue_answers.pop(0) if self.queue_answers else ""
            return 200, self.start("jobs", "JOBID PARTITION NAME\n" + answer)
        if path == "/run_in_container":
            command = body["command"]
            if command.startswith("rm -rf /opt/cluster_transfer/pf/") and "cp -r" in command:
                if self.fetch_answer:
                    return 200, self.start("cmd", *self.fetch_answer)
                key = command.split()[2].split("/opt/cluster_transfer/pf/")[1]
                out = self.transfer / "pf" / key
                out.mkdir(parents=True, exist_ok=True)
                (out / "sweep.json").write_text("{}")
                return 200, self.start("cmd", f"12 /opt/cluster_transfer/pf/{key}\n")
            code = 1 if "bad.yml" in command else 0
            return 200, self.start("cmd", "checked\n", code=code)
        if path == "/sync_from_cluster":
            return 200, self.start("pull", "ok\n")
        return 404, {"detail": path}


@pytest.fixture
def stub(tmp_path, monkeypatch):
    logs, transfer = tmp_path / "logs", tmp_path / "transfer"
    logs.mkdir()
    transfer.mkdir()
    state = Stub(logs, transfer)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n)) if n else None
            code, answer = state.handle(method, self.path, body)
            data = json.dumps(answer).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply("GET")

        def do_POST(self):
            self._reply("POST")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(cp, "URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(cp, "LOG_DIR", str(logs))
    monkeypatch.setattr(cp, "TRANSFER_LOCAL", str(transfer))
    monkeypatch.setattr(cp, "RETRY_SECONDS", 0)
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)
    yield state
    server.shutdown()


def test_snapshot(stub):
    assert cp.snapshot("policy-finder/x") == SHA
    assert stub.calls[0][2] == {"cluster": "raven", "ref": "policy-finder/x"}


def test_submit_retries_on_409(stub):
    stub.busy = 2
    assert cp.submit("configs/simulation/policy_finder/a.yml", SHA) == "123"
    submits = [c for c in stub.calls if c[1] == "/submit_slurm_job"]
    assert len(submits) == 3
    body = submits[-1][2]
    assert body["checkout"] == SHA
    assert body["program_call"] == "-m aimanager simulate configs/simulation/policy_finder/a.yml"
    assert (body["n_gpu"], body["time"], body["mem"]) == (1, "01:00:00", "16G")


def test_wait_until_the_jobs_leave(stub):
    stub.queue_answers = ["  123 gpu x y R\n  999 gpu\n", "  999 gpu\n"]
    cp.wait(["123"], poll=0)
    assert len([c for c in stub.calls if c[1].startswith("/jobs")]) == 2


def test_queue_without_header_is_an_error(stub, monkeypatch):
    monkeypatch.setattr(stub, "start", lambda kind, log, code=0, extra=None: Stub.start(
        stub, kind, "ssh: connection refused\n", code, extra))
    with pytest.raises(cp.ClusterError, match="squeue failed"):
        cp.queue()


def test_validate_rule(stub):
    ok, _ = cp.validate_rule(SHA, "configs/managers/rule_based/good.yml", 1, None, 256)
    assert ok
    body = [c[2] for c in stub.calls if c[1] == "/run_in_container"][-1]
    assert "--max-params none" in body["command"] and body["checkout"] == SHA
    ok, _ = cp.validate_rule(SHA, "configs/managers/rule_based/bad.yml", 1, 4, 256)
    assert not ok


def test_fetch(stub, tmp_path):
    local = tmp_path / "out"
    size = cp.fetch(SHA, "plots/simulation/policy_finder/r_sweep", local, key="r/r_sweep")
    assert (local / "sweep.json").exists() and size == 2
    # the transfer copy is gone locally, and removed remotely
    assert not (stub.transfer / "pf/r/r_sweep").exists()
    commands = [c[2]["command"] for c in stub.calls if c[1] == "/run_in_container"]
    assert commands[-1] == "rm -rf /opt/cluster_transfer/pf/r/r_sweep"


def test_fetch_refuses_unsafe_paths(stub, tmp_path):
    with pytest.raises(cp.ClusterError, match="unsafe"):
        cp.fetch(SHA, "plots/../x", tmp_path, key="k")


def test_fetch_skips_rsync_temp_files(stub, tmp_path):
    def with_temp(method, path, body):
        code, answer = Stub.handle(stub, method, path, body)
        if path == "/sync_from_cluster":
            (stub.transfer / "pf/r/r_sweep/.per_round.parquet.2Xadbj").write_text("partial")
        return code, answer

    stub.handle = with_temp
    local = tmp_path / "out"
    cp.fetch(SHA, "plots/simulation/policy_finder/r_sweep", local, key="r/r_sweep")
    assert sorted(p.name for p in local.iterdir()) == ["sweep.json"]


def test_fetch_tells_missing_output_from_an_outage(stub, tmp_path):
    stub.fetch_answer = ("__PF_NO_OUTPUT__\n", 3)
    with pytest.raises(cp.NoOutput):
        cp.fetch(SHA, "plots/simulation/policy_finder/r_sweep", tmp_path, key="r/r_sweep")
    # the command itself (were it echoed) does not carry the marker
    command = [c[2]["command"] for c in stub.calls if c[1] == "/run_in_container"][-1]
    assert "__PF_NO_OUTPUT__" not in command
    stub.fetch_answer = ("Connection closed by UNKNOWN port 65535\n", 255)
    with pytest.raises(cp.ClusterError) as e:
        cp.fetch(SHA, "plots/simulation/policy_finder/r_sweep", tmp_path, key="r/r_sweep")
    assert not isinstance(e.value, cp.NoOutput)


def test_fetch_accepts_files_vanishing_elsewhere(stub, tmp_path):
    def vanished(method, path, body):
        if path == "/sync_from_cluster":
            return 200, stub.start("pull", "rsync warning: some files vanished (code 24)\n", 24)
        return Stub.handle(stub, method, path, body)

    stub.handle = vanished
    local = tmp_path / "out"
    assert cp.fetch(SHA, "plots/simulation/policy_finder/r_sweep", local, key="r/r_sweep") == 2
