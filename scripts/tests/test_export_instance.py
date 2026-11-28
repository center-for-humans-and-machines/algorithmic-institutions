"""Tests for scripts/policy_finder/export_instance.py: what an agent sees."""

import importlib.util
import json
import re
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


def test_sibling_pattern_follows_the_prefix():
    pattern = ex.sibling_pattern("pfq-exp2-20261127-3")
    found = {m.group(0) for m in pattern.finditer(
        "pfq-exp2-20261127-5 and pfq-exp2-20261127-3-v2, pf-20261104-12,"
        " xpfq-exp2-20261127-7, pfq-exp1-20261127-2")}
    assert found == {"pfq-exp2-20261127-5", "pfq-exp2-20261127-3", "pf-20261104-12"}


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


def _multi(layout):
    wt, main, exports = layout
    _write(wt, ".claude/policy_finder.json", json.dumps(
        {"name": NAME, "version": 2, "rule_name": f"{NAME}-v2", "setting": "exp2_multi"}))
    _write(wt, ".claude/agents/policy-finder-exp2-multi.md",
           "---\nname: policy-finder-exp2-multi\nmodel: opus\n---\n\nAny manager.\n")
    _write(wt, "docs/reports/basics.md", "exp2 game, against ah\n")
    _write(wt, "docs/reports/basics_exp2_multi.md", "the game, another manager\n")
    _write(wt, "configs/simulation/opponents/exp2_multi.yml", "opponent_1: {name: x}\n")
    _write(wt, "configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml",
           "template\n")
    _write(wt, "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/x.jpg", "jpg")
    _write(wt, f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json",
           '{"opponents": ["opponent_1"]}\n')
    return wt, main, exports


def test_an_exp2_multi_export_tells_nothing_of_the_opponents(layout):
    wt, main, exports = _multi(layout)
    export = Path(ex.build(wt, main, exports))
    files = _files(export)
    agent = (export / ".claude/agents/policy-finder.md").read_text()
    assert agent.startswith("---\nname: policy-finder\n") and "Any manager." in agent
    assert "docs/reports/basics_exp2_multi.md" in files
    assert "experiments/2group_8agent_50ep.csv" in files
    # its own results, under opaque keys, stay
    assert f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json" in files
    for gone in ("docs/reports/basics.md",
                 "configs/simulation/opponents/exp2_multi.yml",
                 f"configs/simulation/policy_finder/{NAME}_sweep_p1of2.yml",
                 "configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml"):
        assert gone not in files
    assert not any(f.startswith("plots/simulation/25_LEVIN") for f in files)


@pytest.mark.parametrize("text", ["Levin's rule", "x_s001_vs_ah", "punishment_multinomial_x"])
def test_an_exp2_multi_export_fails_on_an_opponent_term(layout, text):
    wt, main, exports = _multi(layout)
    _write(wt, "src/aimanager/hint.py", f"# {text}\n")
    with pytest.raises(ex.ExportError, match="its opponents"):
        ex.build(wt, main, exports)


def test_other_exports_hold_no_opponents_file(layout):
    wt, main, exports = layout
    _write(wt, "configs/simulation/opponents/exp2_multi.yml", "rule: pf-20261104-12-v7\n")
    _write(wt, "docs/reports/basics_exp2_multi.md", "multi game\n")
    files = _files(Path(ex.build(wt, main, exports)))
    assert "configs/simulation/opponents/exp2_multi.yml" not in files
    assert "docs/reports/basics_exp2_multi.md" not in files


def test_an_exp2_diff_export_has_its_own_game_doc_and_the_ah_reference(layout):
    wt, main, exports = layout
    _write(wt, ".claude/policy_finder.json", json.dumps(
        {"name": NAME, "version": 2, "rule_name": f"{NAME}-v2", "setting": "exp2_diff"}))
    _write(wt, ".claude/agents/policy-finder-exp2-diff.md",
           "---\nname: policy-finder-exp2-diff\nmodel: opus\n---\n\nLead over ah.\n")
    _write(wt, "docs/reports/basics.md", "exp2 game\n")
    _write(wt, "docs/reports/basics_exp2_multi.md", "multi game\n")
    _write(wt, "docs/reports/basics_exp2_diff.md", "the lead over ah\n")
    _write(wt, "configs/simulation/opponents/exp2_diff.yml", "ah: {name: ah}\n")
    _write(wt, "plots/simulation/29_EXP1_reference/x.jpg", "jpg")
    export = Path(ex.build(wt, main, exports))
    files = _files(export)
    agent = (export / ".claude/agents/policy-finder.md").read_text()
    assert agent.startswith("---\nname: policy-finder\n") and "Lead over ah." in agent
    assert "docs/reports/basics_exp2_diff.md" in files
    for gone in ("docs/reports/basics.md", "docs/reports/basics_exp2_multi.md",
                 "configs/simulation/opponents/exp2_diff.yml",
                 "plots/simulation/29_EXP1_reference/x.jpg"):
        assert gone not in files
    assert any(f.startswith("plots/simulation/25_LEVIN") for f in files)


def test_other_exports_hold_no_diff_game_doc(layout):
    wt, main, exports = layout
    _write(wt, "docs/reports/basics_exp2_diff.md", "the lead over ah\n")
    assert "docs/reports/basics_exp2_diff.md" not in _files(Path(ex.build(wt, main, exports)))


def test_a_single_idea_export_has_its_agent_and_test_data(layout):
    """Mode single_idea: its own agent definition, and scrambled copies of the
    data files under test_data/, the earlier versions' sweeps included."""
    import pandas as pd

    wt, main, exports = layout
    _write(wt, ".claude/policy_finder.json", json.dumps(
        {"name": NAME, "version": 2, "rule_name": f"{NAME}-v2", "mode": "single_idea",
         "base_version": NAME}))
    _write(wt, ".claude/agents/policy-finder-single-idea.md",
           "---\nname: policy-finder-single-idea\nmodel: claude-opus-5-5\n---\none idea\n")
    _write(wt, f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json",
           json.dumps({"points": [{"pool": 1.5}, {"pool": 2.5}, {"pool": 3.5}]}))
    table = pd.DataFrame({"episode": [0] * 4 + [1] * 4, "round_number": [0, 0, 1, 1] * 2,
                          "contribution": range(8)})
    table.to_csv(main / "experiments/2group_8agent_50ep.csv", index=False)
    table.to_parquet(
        main / "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet")
    export = Path(ex.build(wt, main, exports))
    agent = (export / ".claude/agents/policy-finder.md").read_text()
    assert "name: policy-finder\n" in agent and "one idea" in agent
    assert not (export / ".claude/agents/policy-finder-single-idea.md").exists()
    copies = [f for f in _files(export) if f.startswith("test_data/")]
    assert copies == [
        "test_data/experiments/2group_8agent_50ep.csv",
        "test_data/plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet",
        f"test_data/plots/simulation/policy_finder/{NAME}_sweep/sweep.json",
    ]


def test_other_exports_hold_no_test_data(layout):
    wt, main, exports = layout
    export = Path(ex.build(wt, main, exports))
    assert not (export / "test_data").exists()


AGENT = Path(__file__).resolve().parents[2] / ".claude/agents/policy-finder-single-idea.md"


@pytest.mark.parametrize("name", [
    "policy-finder-single-idea", "policy-finder", "policy-finder-exp1",
    "policy-finder-exp2-multi", "policy-finder-exp2-diff"])
def test_opencode_agent_keeps_the_text(name):
    text = (AGENT.parent / f"{name}.md").read_text()
    port = ex.opencode_agent(text)
    head, _, body = port.partition("\n---\n")
    assert head.startswith("---\ndescription: ") and "\nmode: primary" in head
    assert "policy-finder-host" not in head and "model:" not in head
    # only the tool's name changes in the body
    original = text.partition("\n---\n")[2]
    assert body != original
    assert body == re.sub(r"the(\s+)Edit(\s+)tool", r"the\1edit\2tool", original)


def test_an_opencode_export(layout):
    wt, main, exports = layout
    _write(wt, ".claude/policy_finder.json", json.dumps({
        "name": NAME, "version": 2, "rule_name": f"{NAME}-v2",
        "harness": "opencode", "model": "qwen/qwen3-coder-next",
        "providers": ["parasail/bf16"]}))
    _write(wt, ".claude/agents/policy-finder.md", AGENT.read_text())
    _write(wt, "src/aimanager/AGENTS.md", "repo rules\n")
    export = Path(ex.build(wt, main, exports))
    files = _files(export)
    assert ".opencode/agents/policy-finder.md" in files
    assert "src/aimanager/AGENTS.md" not in files
    assert (export / ".opencode/.gitignore").read_text() == ex.OPENCODE_GITIGNORE
    config = json.loads((export / ".opencode/opencode.json").read_text())
    assert config["model"] == "openrouter/qwen/qwen3-coder-next"
    router = config["provider"]["openrouter"]
    assert router["options"] == {"apiKey": "{env:OPENROUTER_API_KEY}"}
    assert router["models"]["qwen/qwen3-coder-next"]["options"]["provider"] == {
        "order": ["parasail/bf16"], "allow_fallbacks": False}


def test_opencode_config_for_a_local_server():
    config = ex.opencode_config({"model": "qwen3-coder-next",
                                 "base_url": "http://node:8000/v1"})
    assert config["model"] == "local/qwen3-coder-next"
    local = config["provider"]["local"]
    assert local["options"] == {"baseURL": "http://node:8000/v1",
                                "apiKey": "{env:PF_LLM_API_KEY}"}


def test_sweep_files_are_slim_in_the_export(layout):
    wt, main, exports = layout
    points = [{"name": f"{NAME}_s{i:03d}", "params": {"k": 1 / (i + 3), "m": i},
               "pool": 50 + i / 7, "pool_se": 2.123456, "members": 3.98765,
               "n_episodes": 500} for i in range(256)]
    sweep = {"best": {"k": 0.123456789}, "best_name": f"{NAME}_s255", "points": points}
    _write(wt, f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json",
           json.dumps(sweep, indent=2))
    export = Path(ex.build(wt, main, exports))
    text = (export / f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json").read_text()
    slim = json.loads(text)
    # the same keys and points, rounded, best first, one point per line
    assert set(slim) == set(sweep) and len(slim["points"]) == 256
    assert slim["points"][0]["name"] == f"{NAME}_s255"
    assert slim["points"][0]["pool_se"] == 2.123 and slim["best"] == {"k": 0.1235}
    assert text.count("\n") == 259 and len(text) < 50_000
    # the worktree keeps full precision for the pipeline
    full = json.loads((wt / f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json")
                      .read_text())
    assert full["points"][0]["pool_se"] == 2.123456


def test_slim_files_keep_the_scores_resolution(tmp_path, capsys):
    # exp1-like pools of 630 to 660: 4 significant digits keep them apart
    points = [{"name": f"pf-x-v7_s{i:03d}",
               "params": {f"a_long_parameter_name_{j}": 1 / (i + j + 3) for j in range(5)},
               "pool": 630 + 30 * i / 256, "pool_se": 2.123456, "members": 3.98765,
               "n_episodes": 500} for i in range(256)]
    sweep = tmp_path / "plots/simulation/policy_finder/pf-x-v7_sweep/sweep.json"
    sweep.parent.mkdir(parents=True)
    sweep.write_text(json.dumps({"points": points}, indent=2))
    ex.slim_sweep_files(tmp_path)
    slim = json.loads(sweep.read_text())["points"]
    assert len({p["pool"] for p in slim}) == len({round(630 + 30 * i / 256, 1)
                                                  for i in range(256)})
    assert slim[0]["pool_se"] == 2.123
    # over the read limit even so: said loudly, still 4 digits, best first
    assert len(sweep.read_text().encode()) > ex.READ_LIMIT_BYTES
    assert "read limit" in capsys.readouterr().err
    assert slim[0]["name"] == "pf-x-v7_s255"


def test_slim_files_have_no_line_the_read_tool_cuts():
    # a landscapes.json: profiles of 101 points per parameter
    profiles = {f"param_{j}": {"x": [k / 100 for k in range(101)],
                               "mean": [60 + k / 7 for k in range(101)],
                               "sd": [1.2345 + k / 1000 for k in range(101)]}
                for j in range(5)}
    data = {"rule": "r", "profiles": profiles,
            "points": [{"name": f"s{i:03d}", "mean": 60 + i / 3, "sd": 1.1} for i in range(256)]}
    text = ex.slim_json(data, "points", "mean")
    assert json.loads(text)["profiles"]["param_3"]["mean"][100] == round(60 + 100 / 7, 2)
    assert max(len(line) for line in text.splitlines()) <= ex.LINE_CHARS
