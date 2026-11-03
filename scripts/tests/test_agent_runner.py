"""Tests for scripts/policy_finder/agent_runner.py against a stub spawner."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/agent_runner.py"
SPEC = importlib.util.spec_from_file_location("agent_runner", SCRIPT)
ar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ar)

RESULT = {"type": "result", "result": "my report", "total_cost_usd": 1.5,
          "usage": {"output_tokens": 10}}


@pytest.fixture
def spawner(tmp_path, monkeypatch):
    state = {"posts": [], "busy": 1, "polls": 0}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, answer):
            data = json.dumps(answer).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["posts"].append(body)
            if state["busy"]:
                state["busy"] -= 1
                return self._send(409, {"detail": "at most 8 runs"})
            self._send(200, {"id": "abc", "status": "running"})

        def do_GET(self):
            if self.path.startswith("/runs/abc/logs"):
                return self._send(200, {"stdout": "noise\n" + json.dumps(RESULT) + "\n",
                                        "stderr": ""})
            state["polls"] += 1
            status = "running" if state["polls"] < 3 else "succeeded"
            self._send(200, {"id": "abc", "status": status, "exit_code": 0,
                             "home_host_path": "/h/runs/abc/home"})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(ar, "URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)
    # a transcript in the run's home names the model
    home = tmp_path / "runs/abc/home/.claude/projects/-opt/s.jsonl"
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"message": {"model": "claude-opus-5-5"}}) + "\n"
                    + json.dumps({"message": {"model": "claude-sonnet-5-5"}}) + "\n")
    monkeypatch.setattr(ar, "SPAWNER_STATE", str(tmp_path))
    yield state
    server.shutdown()


def test_start_and_finish(spawner):
    run_id = ar.start("pf-x", ["notes/policy_finder"], "Design your rule.")
    assert run_id == "abc" and len(spawner["posts"]) == 2  # one 409, then started
    body = spawner["posts"][-1]
    assert body["command"][:3] == ["claude", "-p", "Design your rule."]
    assert "--agent" in body["command"] and "policy-finder-host" in body["command"]
    assert body["mounts"] == [
        {"root": "pf_exports", "path": "pf-x", "target": ar.REPO, "read_only": True},
        {"root": "pf_exports", "path": "pf-x/notes/policy_finder",
         "target": f"{ar.REPO}/notes/policy_finder", "read_only": False},
    ]
    record = ar.finish(run_id, log=lambda m: None)
    assert record["status"] == "succeeded"
    assert record["report"] == "my report"
    assert record["result"]["total_cost_usd"] == 1.5 and "result" not in record["result"]
    assert record["models"] == ["claude-opus-5-5", "claude-sonnet-5-5"]


def test_parse_result_takes_the_last_json_line():
    assert ar.parse_result('x\n{"a": 1}\n{"result": "r"}\n') == {"result": "r"}
    assert ar.parse_result("no json") is None


def test_slots_are_exclusive(tmp_path):
    with ar.slot(tmp_path, slots=2) as a:
        with ar.slot(tmp_path, slots=2) as b:
            assert {a, b} == {0, 1}
