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
            # small targets, so the second consolidation pass runs
            "codebook": {"what": {"target_codes": 2}},
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
    # one merge in each fake consolidation pass; candidates may add codes
    assert len(v1["codes"]) >= len(draft["codes"]) - 2 >= 2
    never = yaml.safe_load(
        (run_dir / "codebook" / "never_implemented.v1.yml").read_text()
    )
    assert "reward_cooperators" in [c["name"] for c in never["codes"]]
    members = yaml.safe_load(
        (run_dir / "codebook" / "what.draft.members.yml").read_text()
    )
    assert set(members) >= {c["name"] for c in draft["codes"]} | {"none", "failed"}
    why = yaml.safe_load((run_dir / "codebook" / "why.v1.yml").read_text())
    assert why["codes"] and all(c["family"] for c in why["codes"])
    # consolidation: the fake merges proportional_punishment into
    # punish_below_group_mean; every code has a family, members follow merges
    merges = yaml.safe_load((run_dir / "codebook" / "what.merges.yml").read_text())
    assert any(
        m["applied"] and m["keep"] == "punish_below_group_mean"
        for m in merges["merges"]
    )
    names = {c["name"] for c in v1["codes"]}
    assert "proportional_punishment" not in names and v1["families"]
    assert all(c["family"] for c in v1["codes"])
    v1_members = yaml.safe_load(
        (run_dir / "codebook" / "what.v1.members.yml").read_text()
    )
    assert "proportional_punishment" not in v1_members
    assert (run_dir / "codebook" / "non_discriminating.yml").exists()
    # the fake leaves one code out of the families: it is placed on the re-ask
    assert merges["placement"]["placed_on_reask"]
    assert {1, 2} <= {m["pass"] for m in merges["merges"]}
    for kind in ("what", "why", "basis"):
        un = yaml.safe_load(
            (run_dir / "codebook" / f"{kind}.v1.unassigned.yml").read_text()
        )
        assert un["n"] == len(un["items"]) and all(i["text"] for i in un["items"])
    m = json.loads((run_dir / "manifests" / "consolidate-what.json").read_text())
    assert m["generation_sampling"]["temperature"] > 0
    assert "unplaced" not in {c["family"] for c in v1["codes"]}
    assert all(m["reason"] for m in merges["merges"] if m["applied"])
    basis = yaml.safe_load((run_dir / "codebook" / "basis.v1.yml").read_text())
    assert basis["codes"] and all(c["family"] for c in basis["codes"])
    assert (run_dir / "codebook" / "basis.merges.yml").exists()
    # candidates were judged together with the draft codes
    summary = json.loads((run_dir / "coverage" / "summary.json").read_text())
    assert set(summary["prevalence"]) >= {c["name"] for c in draft["codes"]}
    # clustering used the positive idea, never the negated description
    recs = [json.loads(p.read_text()) for p in (run_dir / "records").glob("*.json")]
    rejected = [m for r in recs for m in r["mechanisms"] if m["status"] == "rejected"]
    assert rejected and all(m["description"].startswith("does not") for m in rejected)
    assert all(not m["idea"].startswith("does not") for m in rejected)


def test_stage_b_join_and_agreement(run_dir):
    with open(run_dir / "analysis" / "version_codes.csv") as f:
        rows = list(csv.DictReader(f))
    v1 = yaml.safe_load((run_dir / "codebook" / "what.v1.yml").read_text())
    never = yaml.safe_load(
        (run_dir / "codebook" / "never_implemented.v1.yml").read_text()
    )
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
    assert any(r["basis_codes"] for r in rows) and any(
        r["basis_families"] for r in rows
    )
    assert {r["event_unreliable_code"] for r in rows if r["event"] == "absent"} == {""}
    with open(run_dir / "analysis" / "theory_mentions.csv") as f:
        assert len(list(csv.DictReader(f))) == 12
    # judge: repeats disagree in the fake, so tie-breaks and re-asks ran (the
    # coverage judge, over all codes; apply-rules last ran with 2 codes)
    with open(run_dir / "analysis" / "judge_reliability.csv") as f:
        rel = {r["code"]: r for r in csv.DictReader(f)}
    n_never = len(
        yaml.safe_load((run_dir / "codebook" / "never_implemented.v1.yml").read_text())[
            "codes"
        ]
    )
    # what.v2 has 2 codes; the never-implemented list is judged in the same calls
    assert int(rel["__overall__"]["n"]) == 12 * (2 + n_never)
    with open(run_dir / "coverage" / "judge_reliability.csv") as f:
        rel = {r["code"]: r for r in csv.DictReader(f)}
    assert 0 < float(rel["__overall__"]["agreement"]) < 1
    reps = read_jsonl(run_dir / "coverage" / "judge_repeats.jsonl")
    assert {r["repeat"] for r in reps} >= {0, 1, "tiebreak"}
    assert any(r["reasked"] for r in reps)
    with open(run_dir / "analysis" / "agreement.csv") as f:
        agree = list(csv.DictReader(f))
    overall = [r for r in agree if r["code"] == "__overall__"]
    assert len(overall) == 1 and 0 < float(overall[0]["agreement"]) < 1
    q = json.loads((run_dir / "analysis" / "quality.json").read_text())
    assert q["quote_check"]["rule_free_codes"]["n"] > 0
    assert q["calls"]["total"]["calls"] > 0
    assert q["judge_reliability"]["overall"]["n"]
    assert 0 < len(q["judge_reliability"]["least_reliable"]) <= 5
    assert q["codes_per_rule"]["rules"] == 12 and "codes_above_max_prevalence" in q
    assert q["judge_flags"]["tiebreaks"] > 0
    assert "codes" in q["judge_reliability"]["unreliable_codes"]
    assert {"0", "1"} <= set(next(iter(q["judge_present_share_by_repeat"].values())))
    assert set(q["family_sizes"]) == {"what", "why", "basis"}
    per_idea = q["what_codes_per_idea"]
    assert set(per_idea) == {"0", "1", "2", "3"}
    assert abs(sum(d["share"] for d in per_idea.values()) - 1) < 1e-3
    recs = [json.loads(p.read_text()) for p in (run_dir / "records_coded").glob("*")]
    mechs = [m for r in recs for m in r["mechanisms"]]
    assert all(
        m["what_codes"][:1] == ([m["what_code"]] if m["what_code"] else [])
        for m in mechs
        if m["what_code_status"] != "failed"
    )
    from aimanager.annotation.fake import assign as fake_assign
    from aimanager.annotation.join import _codes_of

    multi = fake_assign(
        {
            "codes": ["punish_below_group_mean", "punishment_cap"],
            "text": "punish below the mean, up to a cap",
            "multi": True,
        }
    )
    assert multi == {"codes": ["punish_below_group_mean", "punishment_cap"]}
    assert _codes_of({"what_codes": ["a", "b"], "what_code": "a"}) == ["a", "b"]
    assert _codes_of({"what_code": "a"}) == ["a"] and _codes_of({}) == []
    un = q["unassigned_by_codebook"]
    assert "v1" in un["why"] and "v1_stage_b" in un["why"]
    assert (run_dir / "unassigned" / "basis.v1.stage_b.yml").exists()
    # apply-rules reran with the reviewer's what.v2.yml, keeping rater rows
    codes = read_jsonl(run_dir / "codes" / "rule_codes.jsonl")
    llm = [r for r in codes if r["coder"].startswith("llm:")]
    cb_rows = [r for r in llm if r["code_source"] == "codebook"]
    assert {r["codebook_revision"] for r in cb_rows} == {2}
    assert len(cb_rows) == 12 * 2
    # the never-implemented list was judged in the same calls
    assert {r["code_source"] for r in llm} == {"codebook", "never_implemented"}
    assert any(r["coder"] == "rater:sim" for r in codes)


def test_idea_state_and_evidence_kinds():
    from aimanager.annotation.analyse import idea_state, ref_kind

    assert idea_state(True, "introduced", "none", False, False) == "initial"
    assert idea_state(True, "introduced", "none", False) == "introduced"
    assert idea_state(True, "kept", "adopted", True) == "kept"
    assert idea_state(True, "kept", "rejected", True) == "done_but_refused"
    assert idea_state(False, "dropped", "adopted", False) == "dropped"
    assert idea_state(False, "absent", "adopted", True) == "said_not_done"
    assert idea_state(False, "absent", "rejected", True) == "tested_and_refused"
    assert idea_state(False, "absent", "rejected", False) == "refused_untested"
    assert idea_state(False, "absent", "considered", False) == "considered"
    assert idea_state(None, "", "adopted", False) == "not_judged"
    assert idea_state(False, "absent", "none", False) == "absent"
    assert ref_kind("v3 finding 2") == "earlier_version"
    assert ref_kind("exploration 3") == "own_exploration"
    assert ref_kind("pf-20261104-7-v6") == "earlier_version"
    assert ref_kind("none") == "none" and ref_kind(None) == "not_stated"


def test_analyse_tables_figures_and_report(run_dir, tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from aimanager.annotation.analyse import Context, fig_lineages, idea_events
    from aimanager.annotation.common import Run, load_config

    root, base = _copy_run(run_dir, tmp_path)
    cfg_path = run_dir.parent.parent / "cfg.yml"
    run_dir = root / "t"
    # the fixture left join stale (apply-rules reran with what.v2): analyse refuses
    with pytest.raises(SystemExit, match="join"):
        main(["analyse"] + base)
    main(["stage", "B", "--backend", "fake"] + base)
    main(["join"] + base)
    main(["analyse"] + base)
    a = run_dir / "analysis"
    summary = json.loads((a / "summary.json").read_text())
    assert set(summary["tables"]) == {
        "idea_events",
        "idea_flow_by_generation",
        "idea_first_seen",
        "idea_value",
        "why_by_what",
        "basis_by_generation",
        "theory",
        "quality",
    }
    for t in summary["tables"].values():
        assert (a / t["file"]).exists()
    with open(a / "idea_events.csv") as f:
        ev = list(csv.DictReader(f))
    assert len(ev) == len(read_jsonl(run_dir / "corpus" / "versions.jsonl")) * len(
        {r["what_code"] for r in ev}
    )
    assert {"initial", "kept"} <= {r["idea_state"] for r in ev}
    assert {"said_not_done", "done_not_said", "code_unreliable"} <= set(ev[0])
    with open(a / "idea_value.csv") as f:
        val = list(csv.DictReader(f))
    assert "introduced_mean_unfiltered" in val[0]
    figs = {f["file"] for f in summary["figures"]}
    assert {"figures/funnel.png", "figures/reliability.png"} <= figs
    assert any(f.startswith("figures/prevalence_") for f in figs)
    for f in figs:
        assert (a / f).stat().st_size > 1000
    sd = summary["tables"]["idea_events"]["headline"]["say_do"]
    assert set(sd) == {
        "introductions_without_adopted_mention",
        "kept_without_adopted_mention",
        "said_not_done",
    }
    with open(a / "idea_value.csv") as f:
        assert "introduced_solo_mean" in next(csv.DictReader(f))
    q = json.loads((a / "quality.json").read_text())
    assert "say_do_gap" not in q and q["never_implemented_judged"] is True
    report = (a / "REPORT.md").read_text()
    assert "idea_events.csv" in report and "literature.yml" in report
    # the fixture lineages have 3 versions; the lineage figure renders at 3
    ctx = Context(Run(root, "t"), load_config(cfg_path))
    out = fig_lineages(idea_events(ctx)[0], a / "figures", min_versions=3)
    assert out and (a / out[0]["file"]).exists()
    # a newer codebook revision makes the join (and so analyse) stale
    main(["review", "begin", "--kind", "why"] + base)
    with pytest.raises(SystemExit, match="latest why codebook"):
        main(["analyse"] + base)


def test_literature_file_shapes():
    from aimanager.annotation.analyse import _literature

    doc = {
        "publications": {
            "fg2000": {
                "authors": ["Ernst Fehr", "Simon Gächter"],
                "year": 2000,
                "title": "Cooperation and Punishment",
                "venue": "AER",
                "doi": "10.1257/x",
                "verified": True,
            }
        },
        "codes": {
            "what": {
                "group_ref": {
                    "links": [
                        {
                            "publication": "fg2000",
                            "relation": "states",
                            "connection": "c",
                            "strength": "direct",
                        }
                    ]
                },
                "rounding": {"links": [], "note": "detail"},
            }
        },
    }
    lit = _literature(doc)
    assert lit["rounding"] == []
    (link,) = lit["group_ref"]
    assert link["reference"].startswith("Fehr and Gächter (2000)")
    assert link["verified"] and link["strength"] == "direct"
    assert _literature({"x": [{"reference": "r"}]}) == {"x": [{"reference": "r"}]}


def _digest(run):
    import hashlib

    files = [run / "codes" / "rule_codes.jsonl", *sorted(run.glob("records_coded/*"))]
    files += sorted(run.glob("unassigned/*")) + sorted(run.glob("analysis/*.csv"))
    return {
        p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def _copy_run(run_dir, tmp_path):
    import shutil

    root = tmp_path / "runs"
    shutil.copytree(run_dir, root / "t")
    cfg = run_dir.parent.parent / "cfg.yml"
    return root, ["--run", "t", "--runs-root", str(root), "--config", str(cfg)]


def test_resume_skips_and_cached_rerun_is_byte_identical(run_dir, tmp_path):
    root, base = _copy_run(run_dir, tmp_path)
    run = root / "t"
    fake = base + ["--backend", "fake"]
    main(["stage", "B"] + fake)  # bring stage B up to date with what.v2
    main(["join"] + base)
    before = _digest(run)
    calls = run / "logs" / "calls.jsonl"
    n0 = len(read_jsonl(calls))
    # no server: up-to-date steps skip without creating a client
    offline = base + ["--base-url", "http://127.0.0.1:9/v1", "--model", "fake"]
    main(["stage", "B"] + offline)
    assert len(read_jsonl(calls)) == n0
    # forced: every call is answered from the cache, outputs are identical
    main(["stage", "B", "--force"] + offline)
    new = read_jsonl(calls)[n0:]
    assert new and all(r["cache_hit"] for r in new)
    main(["join"] + base)
    assert _digest(run) == before
    # a changed input makes the step run again
    m = json.loads((run / "manifests" / "apply-notes.json").read_text())
    assert m["fingerprint"]


def test_interrupted_step_resumes_to_the_same_result(run_dir, tmp_path, monkeypatch):
    from aimanager.annotation import fake as fake_mod

    root_a, base_a = _copy_run(run_dir, tmp_path / "a")
    root_b, base_b = _copy_run(run_dir, tmp_path / "b")
    args_a = base_a + ["--backend", "fake", "--force", "--concurrency", "1"]
    real = fake_mod.respond
    seen = {"n": 0}

    def crash_midway(call):
        seen["n"] += 1
        if seen["n"] == 15:
            raise KeyboardInterrupt("simulated kill")
        return real(call)

    for p in (root_a / "t" / "cache", root_b / "t" / "cache"):
        import shutil

        shutil.rmtree(p, ignore_errors=True)
    monkeypatch.setattr(fake_mod, "respond", crash_midway)
    with pytest.raises(KeyboardInterrupt):
        main(["apply-notes"] + args_a)
    monkeypatch.setattr(fake_mod, "respond", real)
    main(["apply-notes"] + args_a)
    main(["apply-notes"] + base_b + ["--backend", "fake", "--force"])
    a = {k: v for k, v in _digest(root_a / "t").items() if k.startswith("records")}
    b = {k: v for k, v in _digest(root_b / "t").items() if k.startswith("records")}
    assert a and a == b


def test_never_implemented_judged_and_found():
    from aimanager.annotation.join import _never_found

    rows = [
        {
            "coder": "llm:m",
            "version": "v1",
            "what_code": "reward",
            "present": True,
            "quote": "bonus = 2",
        },
        {
            "coder": "llm:m",
            "version": "v2",
            "what_code": "reward",
            "present": False,
            "quote": "",
        },
        {
            "coder": "rater:a",
            "version": "v1",
            "what_code": "reward",
            "present": True,
            "quote": "x",
        },
    ]
    assert _never_found(rows, "llm:m", ["reward", "expel"]) == {
        "reward": [{"version": "v1", "quote": "bonus = 2"}]
    }


def test_reliability_classes():
    from aimanager.annotation.join import reliability_class

    cfg = {
        "judge": {"min_kappa": 0.8, "min_positive_agreement": 0.85, "min_positives": 5}
    }

    def row(both_yes, a_only, b_only, kappa):
        return {
            "both_yes": both_yes,
            "a_only": a_only,
            "b_only": b_only,
            "kappa": kappa,
        }

    assert reliability_class(row(50, 1, 1, 0.95), cfg) == "reliable"
    assert reliability_class(row(3, 0, 1, 0.8), cfg) == "rare"
    assert reliability_class(row(20, 6, 6, 0.75), cfg) == "unreliable"
    assert reliability_class(row(10, 3, 3, 0.9), cfg) == "unreliable"  # pos 0.77


def test_codebook_validation(tmp_path):
    from aimanager.annotation.common import validate_codebook

    path = tmp_path / "what.v2.yml"
    ok = {
        "families": [{"name": "f"}],
        "codes": [
            {"name": "a", "label": "A", "definition": "d", "family": "f"},
            {"name": "b", "label": "B", "definition": "d", "family": "f"},
        ],
    }
    assert len(validate_codebook(path, ok)) == 2
    bad = [
        ({"codes": [{"name": "a", "label": "A", "definition": "d"}] * 2}, "duplicate"),
        ({"codes": [{"name": "none", "label": "N", "definition": "d"}]}, "reserved"),
        ({"codes": [{"name": "a", "definition": "d"}]}, "label"),
        (
            {
                "codes": [
                    {"name": "a", "label": "A", "definition": "d", "family": "f"},
                    {"name": "b", "label": "B", "definition": "d"},
                ]
            },
            "family",
        ),
        (
            {
                "families": [{"name": "f"}],
                "codes": [
                    {"name": "a", "label": "A", "definition": "d", "family": "g"}
                ],
            },
            "not in",
        ),
    ]
    for doc, msg in bad:
        with pytest.raises(SystemExit, match=msg):
            validate_codebook(path, doc)


def test_ingest_validation_and_rater_pairs(run_dir, tmp_path):
    import shutil

    root, base = _copy_run(run_dir, tmp_path)
    run = root / "t"
    main(["stage", "B", "--backend", "fake"] + base)
    main(["expert-check", "export"] + base)
    filled = run / "expert_check" / "filled"
    shutil.rmtree(filled)
    filled.mkdir()
    sheets = sorted((run / "expert_check" / "sheets").glob("*.yml"))
    for i, p in enumerate(sheets):
        sheet = yaml.safe_load(p.read_text())
        assert sheet["sheet_format"] == 2 and sheet["game_setting"]
        assert "fits" in sheet["codes"][0] and sheet["families"]
        for rater in ("  Rater A ", "rater b"):
            sheet["rater"] = rater
            for j, c in enumerate(sheet["codes"]):
                c["present"] = (i + j + len(rater)) % 3 == 0
            write_yaml(filled / f"{p.stem}.{rater.strip()[-1]}.yml", sheet)
    main(["expert-check", "ingest"] + base)
    rows = read_jsonl(run / "codes" / "rule_codes.jsonl")
    # names are normalised (trimmed, lower case); the fixture's earlier rater
    # stays, since a sheet replaces only its own rater's rows
    assert {r["coder"] for r in rows if r["coder"].startswith("rater:")} >= {
        "rater:rater a",
        "rater:rater b",
    }
    with open(run / "analysis" / "agreement.csv") as f:
        agree = list(csv.DictReader(f))
    assert {r["pair_type"] for r in agree} == {"rater-judge", "rater-rater"}
    # a sheet of another codebook state is rejected, naming the file
    bad = filled / f"{sheets[0].stem}.A.yml"
    sheet = yaml.safe_load(bad.read_text())
    sheet["codebook_sha256"] = "0" * 64
    write_yaml(bad, sheet)
    with pytest.raises(SystemExit, match=bad.name):
        main(["expert-check", "ingest"] + base)
    main(["expert-check", "ingest", "--allow-old-revision"] + base)
    rows = read_jsonl(run / "codes" / "rule_codes.jsonl")
    assert any(r.get("current_revision") is False for r in rows)
    # unknown code names are an error
    sheet["codebook_sha256"] = yaml.safe_load(
        (filled / f"{sheets[1].stem}.A.yml").read_text()
    )["codebook_sha256"]
    sheet["codes"][0]["code"] = "misspelt_code"
    write_yaml(bad, sheet)
    with pytest.raises(SystemExit, match="misspelt_code"):
        main(["expert-check", "ingest"] + base)


def test_held_out_notes_are_left_out_of_codebooks(run_dir):
    from aimanager.annotation.common import Run, load_config, read_json
    from aimanager.annotation.steps import codebook_records

    run = Run(run_dir.parent, "t")
    held = set(read_json(run_dir / "corpus" / "split.json")["expert_check"])
    cfg = load_config()
    kept = codebook_records(run, cfg)
    assert held and not held & set(kept) and len(kept) == 12 - len(held)
    cfg["held_out"] = {"exclude_notes_from_codebook": False}
    assert len(codebook_records(run, cfg)) == 12
