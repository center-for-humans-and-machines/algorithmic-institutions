"""preflight.py --campaigns: a campaign's branches of another run, and the
model files of the setting's reference managers."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "policy_finder"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("preflight_t", HERE / "preflight.py")
pf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pf)


def test_branches_of_another_run_and_reference_models(tmp_path, monkeypatch, capsys):
    camp = tmp_path / "c.yml"
    camp.write_text("prefix: pfsi1-exp1-{env:PF_MODEL_TAG}\nsetting: exp1\n"
                    "mode: single_idea\nn_fresh: 2\ngenerations: 1\ntop_k: 2\n")
    monkeypatch.setenv("PF_MODEL_TAG", "qwen3-coder-next")
    runs, artifacts = tmp_path / "runs", tmp_path / "artifacts"
    existing = {"policy-finder/pfsi1-exp1-qwen3-coder-next-1"}
    monkeypatch.setattr(pf, "git", lambda *a: (0, "") if a[-1] in existing else (1, ""))
    monkeypatch.setattr(pf, "failures", 0)
    pf.check_campaigns([camp], runs, artifacts)
    out = capsys.readouterr().out
    assert pf.failures == 2  # the foreign branch, and exp1's human-like clone
    assert "policy-finder/pfsi1-exp1-qwen3-coder-next-1" in out
    assert "23_punishment_autoregressive_v4" in out
    # this PF_RUNS holds the lineage, and the clone's file is there
    (runs / "worktrees/pfsi1-exp1-qwen3-coder-next-1").mkdir(parents=True)
    clone = artifacts / ("behavioral_cloning/23_punishment_autoregressive_v4/model/"
                         "architecture_node+edge.pt")
    clone.parent.mkdir(parents=True)
    clone.write_text("x")
    monkeypatch.setattr(pf, "failures", 0)
    pf.check_campaigns([camp], runs, artifacts)
    assert pf.failures == 0
