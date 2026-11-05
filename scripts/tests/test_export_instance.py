"""Tests for scripts/policy_finder/export_instance.py: what an agent sees."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/export_instance.py"
SPEC = importlib.util.spec_from_file_location("export_instance", SCRIPT)
ex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ex)

NAME = "pf-20990101-1"
OTHER = "pf-20990101-12"


def _write(root, rel, text="x\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def layout(tmp_path):
    """A worktree at version 2 with foreign files, a main checkout with data."""
    wt, main, exports = tmp_path / "wt", tmp_path / "main", tmp_path / "exports"
    exports.mkdir()
    _write(wt, ".git", "gitdir: /somewhere\n")
    _write(wt, ".claude/policy_finder.json", json.dumps(
        {"name": NAME, "version": 2, "rule_name": f"{NAME}-v2"}))
    _write(wt, ".claude/agents/policy-finder.md", "agent\n")
    _write(wt, ".claude/agents/policy-finder-host.md", "host\n")
    _write(wt, ".claude/skills/policy-finder/SKILL.md", "skill\n")
    _write(wt, ".claude/settings.json", "{}\n")
    _write(wt, "src/aimanager/x.py", "y = 1\n")
    _write(wt, "configs/managers/rule_based/.gitkeep", "")
    _write(wt, f"configs/managers/rule_based/{NAME}.yml", "v1 rule\n")
    _write(wt, "configs/managers/rule_based/pf-20261008-3-v6.yml", "cem\n")
    _write(wt, f"configs/managers/rule_based/{OTHER}.yml", "other\n")
    _write(wt, "configs/managers/rule_based/params/pf-20261008-3-v6_best.json", "{}\n")
    _write(wt, f"configs/simulation/policy_finder/{NAME}_sweep_p1of2.yml", "sweep\n")
    _write(wt, "configs/simulation/manager_testing/26_LEVIN_ah_vs_pf_top_batched.yml", "x\n")
    _write(wt, "configs/simulation/manager_testing/25_LEVIN_run1.yml", "ref\n")
    _write(wt, f"notes/policy_finder/{NAME}.md", "v1 notes\n")
    _write(wt, f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json", "{}\n")
    _write(wt, f"plots/simulation/policy_finder/{OTHER}_sweep/sweep.json", "{}\n")
    _write(wt, f"scripts/policy_finder/{NAME}/explore.py", "a = 1\n")
    _write(wt, "scripts/policy_finder/pipeline.py", "code\n")
    _write(wt, "scripts/policy_finder/observation_app/dashboard.py", "dash\n")
    _write(main, "experiments/2group_8agent_50ep.csv", "human,data\n")
    _write(main, "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet",
           "PAR1")
    return wt, main, exports


def _files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_build_keeps_only_the_lineage(layout):
    wt, main, exports = layout
    export = Path(ex.build(wt, main, exports))
    assert export == exports / f"{NAME}-v2"
    assert _files(export) == sorted([
        ".claude/agents/policy-finder-host.md",
        ".claude/agents/policy-finder.md",
        ".claude/policy_finder.json",
        "configs/managers/rule_based/.gitkeep",
        f"configs/managers/rule_based/{NAME}.yml",
        f"configs/simulation/policy_finder/{NAME}_sweep_p1of2.yml",
        "configs/simulation/manager_testing/25_LEVIN_run1.yml",
        "experiments/2group_8agent_50ep.csv",
        f"notes/policy_finder/{NAME}.md",
        f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json",
        "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet",
        f"scripts/policy_finder/{NAME}/explore.py",
        "src/aimanager/x.py",
    ])
    # the write folders exist for the read-write mounts, the manifest outside
    assert (export / "configs/managers/rule_based/params").is_dir()
    manifest = json.loads((exports / f"{NAME}-v2{ex.MANIFEST}").read_text())
    assert f"configs/managers/rule_based/{NAME}.yml" in manifest["files"]


def test_lineage_names():
    assert ex._lineage(NAME, NAME)
    assert ex._lineage(f"{NAME}-v3.yml", NAME)
    assert ex._lineage(f"{NAME}_sweep_p1of2.yml", NAME)
    assert not ex._lineage(OTHER, NAME)
    assert not ex._lineage(f"{OTHER}-v2.yml", NAME)
    assert not ex._lineage("pf-20261008-3-v6.yml", NAME)


def test_build_fails_on_a_leak(layout):
    wt, main, exports = layout
    _write(wt, "src/aimanager/notes.py", f"# compare with {OTHER}\n")
    with pytest.raises(ex.ExportError, match=OTHER):
        ex.build(wt, main, exports)


def test_build_needs_the_data(layout):
    wt, main, exports = layout
    (main / "experiments/2group_8agent_50ep.csv").unlink()
    with pytest.raises(ex.ExportError, match="2group_8agent_50ep"):
        ex.build(wt, main, exports)


def test_collect(layout):
    wt, main, exports = layout
    export = Path(ex.build(wt, main, exports))
    _write(export, f"configs/managers/rule_based/{NAME}-v2.yml", "v2 rule\n")
    _write(export, f"notes/policy_finder/{NAME}-v2.md", "v2 notes\n")
    _write(export, f"scripts/policy_finder/{NAME}/new.py", "b = 2\n")
    assert ex.collect(wt, exports) == []
    assert (wt / f"configs/managers/rule_based/{NAME}-v2.yml").read_text() == "v2 rule\n"
    assert (wt / f"scripts/policy_finder/{NAME}/new.py").exists()
    assert (wt / f"scripts/policy_finder/{NAME}/explore.py").exists()


def test_collect_reports_strays(layout):
    wt, main, exports = layout
    export = Path(ex.build(wt, main, exports))
    _write(export, f"configs/managers/rule_based/{NAME}.yml", "v1 changed\n")
    _write(export, "configs/managers/rule_based/extra.yml", "new\n")
    (export / f"notes/policy_finder/{NAME}.md").unlink()
    assert ex.collect(wt, exports) == sorted([
        f"configs/managers/rule_based/{NAME}.yml",
        "configs/managers/rule_based/extra.yml",
        f"notes/policy_finder/{NAME}.md",
    ])
    # an earlier version's file is never copied back
    assert (wt / f"configs/managers/rule_based/{NAME}.yml").read_text() == "v1 rule\n"


def test_an_exp1_export_holds_its_own_game(layout):
    wt, main, exports = layout
    _write(wt, ".claude/policy_finder.json", json.dumps(
        {"name": NAME, "version": 2, "rule_name": f"{NAME}-v2", "setting": "exp1"}))
    _write(wt, ".claude/agents/policy-finder-exp1.md",
           "---\nname: policy-finder-exp1\nmodel: opus\n---\n\nThe exp1 game.\n")
    _write(wt, "docs/reports/basics.md", "exp2 game\n")
    _write(wt, "docs/reports/basics_exp1.md", "exp1 game\n")
    _write(wt, "configs/simulation/manager_testing/28_EXP1_baselines.yml", "rl\n")
    _write(main, "experiments/pilot_random1_player_round_slim.csv", "pilot\n")
    _write(main, "plots/simulation/29_EXP1_reference/per_round.parquet", "PAR1")
    export = Path(ex.build(wt, main, exports))
    files = _files(export)
    # the exp1 definition, under the name policy-finder-host starts
    agent = (export / ".claude/agents/policy-finder.md").read_text()
    assert agent.startswith("---\nname: policy-finder\n") and "The exp1 game." in agent
    assert ".claude/agents/policy-finder-exp1.md" not in files
    assert "experiments/pilot_random1_player_round_slim.csv" in files
    assert "plots/simulation/29_EXP1_reference/per_round.parquet" in files
    assert "docs/reports/basics_exp1.md" in files
    for gone in ("docs/reports/basics.md", "experiments/2group_8agent_50ep.csv",
                 "configs/simulation/manager_testing/28_EXP1_baselines.yml"):
        assert gone not in files


def test_an_exp2_export_has_no_exp1_game(layout):
    wt, main, exports = layout
    _write(wt, "docs/reports/basics_exp1.md", "exp1 game\n")
    _write(wt, "configs/simulation/manager_testing/27_LEVIN_pf_proxy_vs_cem_batched.yml",
           f"rule: {OTHER}\n")
    files = _files(Path(ex.build(wt, main, exports)))
    assert "docs/reports/basics_exp1.md" not in files
    assert (Path(exports) / f"{NAME}-v2/.claude/agents/policy-finder.md").read_text() == "agent\n"
