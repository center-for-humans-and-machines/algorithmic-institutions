"""Tests for joining a sweep's parts (#227)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/merge_sweeps.py"
SPEC = importlib.util.spec_from_file_location("merge_sweeps", SCRIPT)
ms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ms)


def _part(root, name, points):
    d = root / name
    d.mkdir()
    result = {
        "rule": "r.yml",
        "anchor": "ah",
        "score": "def 4",
        "n_episodes": 500,
        "episode_batch_size": 1000,
        "seed": 42,
        "points": [{"name": n, "params": {"a": p}, "pool": p} for n, p in points],
    }
    (d / "sweep.json").write_text(json.dumps(result))
    return str(d)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["merge", *argv])
    ms.main()


def test_merge_writes_the_whole_sweep(tmp_path, monkeypatch):
    p1 = _part(tmp_path, "toy_sweep_p1of2", [("toy_s000", 1.0)])
    p2 = _part(tmp_path, "toy_sweep_p2of2", [("toy_s001", 2.0)])
    _run(monkeypatch, p1, p2)
    merged = json.loads((tmp_path / "toy_sweep/sweep.json").read_text())
    assert merged["best"] == {"a": 2.0} and len(merged["points"]) == 2


def test_merge_needs_every_part(tmp_path, monkeypatch):
    p1 = _part(tmp_path, "toy_sweep_p1of3", [("toy_s000", 1.0)])
    p3 = _part(tmp_path, "toy_sweep_p3of3", [("toy_s002", 2.0)])
    with pytest.raises(SystemExit, match=r"needs parts 1..3, got \[1, 3\]"):
        _run(monkeypatch, p1, p3)
    other = _part(tmp_path, "toy_sweep", [("toy_s001", 2.0)])
    with pytest.raises(SystemExit, match="pass --out"):
        _run(monkeypatch, p1, other)
