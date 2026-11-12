"""End-to-end annotation pipeline on the fake backend, from a fixture git repo."""

import csv
import json
import subprocess

import pytest
import yaml

from aimanager.annotation.cli import main
from aimanager.annotation.common import read_jsonl, write_yaml

RULE = """params:
  slope:
    definition: punishment points per point a member falls below the group mean
  cap:
    definition: largest punishment dealt to any one member in a round
  t_stop:
    definition: last round in which the manager punishes
{extra_params}code: |
  gap = th.relu(mates_mean - cur - tol)
  raw = th.minimum(slope * gap, cap * one)
{extra_code}  punishment = raw * (t <= t_stop).float()
"""
NOTES = """# {stem}

## Explorations

1. Punishing below the group mean raises contributions next round.
2. A reward for high contributors was considered; human managers never used it.
3. Large punishments make members leave, so a cap keeps the group.

## Key findings

1. Players herd to the group: 0.30 weight on the others' mean.
{extra}
## Hypothesis

Punish shortfall below the mean with slope 1.5; stop punishing at the last round.

## Justification

The tolerance keeps low noise unpunished; switch rounds are risky.
"""


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(root):
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    for campaign in ("pf-20261104", "pf-20261105"):
        for n in (1, 2):
            name = f"{campaign}-{n}"
            _git(repo, "checkout", "-q", "--orphan", name)
            _git(repo, "rm", "-rqf", "--ignore-unmatch", ".")
            for k in (1, 2, 3):
                stem = name if k == 1 else f"{name}-v{k}"
                newcomer = k >= 2 and n == 2
                rule = RULE.format(
                    extra_params=(
                        "  pre_switch:\n    definition: factor before a"
                        " switch decision\n"
                        if k == 3
                        else ""
                    ),
                    extra_code=(
                        "  raw = raw * (1 + newcomer_boost * arrived)\n"
                        if newcomer
                        else ""
                    ),
                )
                extra = (
                    f"2. Newcomers in their arrival round respond, as in {name}.\n"
                    if newcomer
                    else ""
                )
                files = {
                    f"configs/managers/rule_based/{stem}.yml": rule,
                    f"notes/policy_finder/{stem}.md": NOTES.format(
                        stem=stem, extra=extra
                    ),
                    # another lineage's copy, which the filter must drop
                    f"configs/managers/rule_based/{campaign}-9.yml": rule,
                }
                if not (campaign == "pf-20261105" and k == 3):
                    sweep = {
                        "best_name": "b",
                        "points": [{"name": "b", "pool": 60.0 + n + k, "pool_se": 1}],
                    }
                    path = f"plots/simulation/policy_finder/{stem}_sweep/final"
                    files[f"{path}/sweep.json"] = json.dumps(sweep)
                for rel, text in files.items():
                    p = repo / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text)
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", name)
            _git(repo, "update-ref", f"refs/remotes/origin/policy-finder/{name}", name)
    return repo


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("annot")
    repo = make_repo(root)
    cfg = root / "cfg.yml"
    write_yaml(
        cfg,
        {
            "corpus": {"repo": str(repo)},
            "sampling": {"expert_check_n": 4},
            "llm": {"concurrency": 4},
        },
    )
    base = ["--run", "t", "--runs-root", str(root / "runs"), "--config", str(cfg)]
    fake = base + ["--backend", "fake"]
    main(["corpus"] + base)
    main(["stage", "A"] + fake)
    main(["stage", "B"] + fake)
    main(["expert-check", "export"] + base)
    run = root / "runs" / "t"
    llm = {
        (r["version"], r["what_code"]): r["present"]
        for r in read_jsonl(run / "codes" / "rule_codes.jsonl")
    }
    for i, p in enumerate(sorted((run / "expert_check" / "sheets").glob("*.yml"))):
        sheet = yaml.safe_load(p.read_text())
        sheet["rater"] = "sim"
        for j, c in enumerate(sheet["codes"]):
            flip = (i + j) % 3 == 0
            c["present"] = llm[(sheet["version"], c["code"])] != flip
        write_yaml(run / "expert_check" / "filled" / p.name, sheet)
    main(["expert-check", "ingest"] + base)
    main(["join"] + base)
    # a reviewer writes the next revision; stage B must pick it up
    main(["review", "begin", "--kind", "what"] + base)
    v2 = yaml.safe_load((run / "codebook" / "what.v2.yml").read_text())
    v2["codes"] = v2["codes"][:2]
    write_yaml(run / "codebook" / "what.v2.yml", v2)
    main(["apply-rules"] + fake)
    # apply-notes still used what.v1.yml: join must refuse
    with pytest.raises(SystemExit, match="apply-notes used what.v1.yml"):
        main(["join"] + base)
    return run


def test_corpus_filter_scores_and_split(run_dir):
    versions = read_jsonl(run_dir / "corpus" / "versions.jsonl")
    assert len(versions) == 12
    assert {v["stem"] for v in versions if v["score_status"] != "ok"} == {
        "pf-20261105-1-v3",
        "pf-20261105-2-v3",
    }
    v = next(v for v in versions if v["stem"] == "pf-20261104-2-v3")
    assert v["parent"] == "pf-20261104-2-v2" and v["version"] == 3
    split = json.loads((run_dir / "corpus" / "split.json").read_text())
    assert len(split["free_coding"]) == 4
    assert len({s.rsplit("-v", 1)[0] for s in split["free_coding"]}) == 4
    assert len(split["expert_check"]) == 4
    assert not set(split["free_coding"]) & set(split["expert_check"])


def test_stage_a_outputs(run_dir):
    assert read_jsonl(run_dir / "open" / "rule_free_codes.jsonl")
    assert len(list((run_dir / "records").glob("*.json"))) == 12
    draft = yaml.safe_load((run_dir / "codebook" / "what.draft.yml").read_text())
    v1 = yaml.safe_load((run_dir / "codebook" / "what.v1.yml").read_text())
    assert len(v1["codes"]) >= len(draft["codes"]) >= 2
    never = yaml.safe_load((run_dir / "codebook" / "never_implemented.yml").read_text())
    assert "reward_cooperators" in [c["name"] for c in never["codes"]]
    members = yaml.safe_load(
        (run_dir / "codebook" / "what.draft.members.yml").read_text()
    )
    assert set(members) >= {c["name"] for c in draft["codes"]} | {"none", "failed"}
    assert yaml.safe_load((run_dir / "codebook" / "why.v1.yml").read_text())["codes"]


def test_stage_b_join_and_agreement(run_dir):
    with open(run_dir / "analysis" / "version_codes.csv") as f:
        rows = list(csv.DictReader(f))
    v1 = yaml.safe_load((run_dir / "codebook" / "what.v1.yml").read_text())
    never = yaml.safe_load((run_dir / "codebook" / "never_implemented.yml").read_text())
    assert len(rows) == 12 * (len(v1["codes"]) + len(never["codes"]))
    assert {"introduced", "kept"} <= {r["event"] for r in rows}
    reward = [r for r in rows if r["what_code"] == "reward_cooperators"]
    assert {r["code_source"] for r in reward} == {"never_implemented"}
    assert {r["in_rule"] for r in reward} == {"0"}
    assert "considered" in {r["notes_status"] for r in reward}
    assert {"no_mechanism", "adopted"} <= {r["notes_status"] for r in rows}
    first = {r["notes_relation"] for r in rows if r["generation"] == "1"}
    assert first <= {"not_stated", "no_mechanism"}
    assert any(r["why_codes_for"] for r in rows)
    assert any(r["reason_evidence"] for r in rows)
    with open(run_dir / "analysis" / "agreement.csv") as f:
        agree = list(csv.DictReader(f))
    overall = [r for r in agree if r["code"] == "__overall__"]
    assert len(overall) == 1 and 0 < float(overall[0]["agreement"]) < 1
    q = json.loads((run_dir / "analysis" / "quality.json").read_text())
    assert q["quote_check"]["rule_free_codes"]["n"] > 0
    assert q["calls"]["total"]["calls"] > 0
    # apply-rules reran with the reviewer's what.v2.yml, keeping rater rows
    codes = read_jsonl(run_dir / "codes" / "rule_codes.jsonl")
    llm = [r for r in codes if r["coder"].startswith("llm:")]
    assert {r["codebook_revision"] for r in llm} == {2}
    assert len(llm) == 12 * 2
    assert any(r["coder"] == "rater:sim" for r in codes)
