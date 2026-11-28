"""change_note.py on the notes the open-weight agents wrote (2026-11-27:
fixtures/single_idea_notes/, 3 DeepSeek-V4.1-Flash and 3 Qwen3-Coder-Next
trials of version 2, the pipeline's version 3 with DeepSeek and version 4
with Qwen of pfsmoke-20261127-1), in both directions."""

import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/change_note.py"
SPEC = importlib.util.spec_from_file_location("change_note", SCRIPT)
cn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cn)

NOTES = Path(__file__).resolve().parent / "fixtures/single_idea_notes"
BASE = "pfsmoke-20261127-1"


def section(name):
    text = (NOTES / f"{name}.md").read_text()
    return text.split("\n## Change\n", 1)[1]


@pytest.mark.parametrize("name", [
    "si-ds-1",   # "Justification: paper and analysis"
    "si-ds-2",   # the exact layout
    "si-ds-3",   # the exact layout
    "si-qw-1",   # the justification a sentence, the analysis given
    "si-qw-2",   # the same
    "v3-ds",     # "paper and analysis"
    "v4-qw",     # the whole list indented by two spaces
])
def test_the_agents_notes_pass(name):
    assert cn.check(section(name), BASE) == []


def test_a_note_without_paper_or_analysis_fails():
    # si-qw-3: "Justification: Analysis of human data shows ...", but Paper,
    # Claim and Analysis all none, and the base version in backticks (fine)
    reasons = cn.check(section("si-qw-3"), BASE)
    assert len(reasons) == 1 and reasons[0].startswith("no justification")


@pytest.mark.parametrize("change, reason", [
    (lambda t: re.sub(r"(?m)^- Operation:.*\n", "", t), "no change stated"),
    (lambda t: t.replace("- Operation: add", "- Operation: add, then modify"),
     "several changes"),
    (lambda t: t.replace("- Operation: add", "- Operation: add\n- Operation: remove"),
     "several changes"),
    (lambda t: re.sub(r"(?m)^- Idea:.*$", "- Idea: none", t), "no change stated"),
    (lambda t: re.sub(r"(?m)^- (Paper|Analysis):.*$", r"- \1: none", t),
     "no justification"),
    (lambda t: t.replace(f"- Base version: {BASE}", f"- Base version: {BASE}-v3"),
     "must name"),
])
def test_real_notes_made_wrong_fail(change, reason):
    text = change(section("si-ds-2"))
    assert any(r.startswith(reason) or reason in r for r in cn.check(text, BASE))


def test_a_reference_as_title_and_author_is_enough():
    text = re.sub(r"(?m)^- Analysis:.*$", "- Analysis: none", section("si-ds-2"))
    text = re.sub(r"(?m)^- Paper:.*$", "- Paper: Fehr & Gaechter, Altruistic punishment"
                  " in humans", text)
    text = re.sub(r"(?m)^- Claim:.*$", "", text)
    assert cn.check(text, BASE) == []


def test_bold_labels_and_a_remark_pass():
    # the layout of the debug run's report (DeepSeek, 2026-11-27)
    text = ("- **Base version:** `pfsmoke-20261127-1` (the zero-punisher seed)\n"
            "- **Operation:** add\n- **Idea:** punish the shortfall below the mean.\n"
            "- **Justification:** paper + analysis\n"
            "  - Paper: Fehr, E., & Gächter, S. (2002). *Altruistic punishment in humans.*\n"
            "  - Analysis: `scripts/x_analysis.py` printed **1.337195**\n")
    assert cn.check(text, BASE) == []


def _with(**fields):
    """si-ds-2's Change section with some lines replaced (a value None drops
    the line)."""
    text = section("si-ds-2")
    for label, value in fields.items():
        label = label.replace("_", " ").capitalize()
        line = "" if value is None else f"- {label}: {value}"
        text = re.sub(rf"(?m)^- {label}:.*$", line, text)
    return text


@pytest.mark.parametrize("empty", [
    "none", "no paper", "No paper comes to mind", "not applicable", "N/A (no paper used)",
    "N.A.", "(none)", "?", "—", "-", "unknown", "null",
    "<title, lead or senior author (year and journal if you know them)>, or none",
])
def test_empty_references_and_analyses(empty):
    text = _with(paper=empty, analysis="not run")
    assert any(r.startswith("no justification") for r in cn.check(text, BASE))


@pytest.mark.parametrize("fields, reason", [
    ({"idea": "N/A (no change)"}, "no change stated"),
    ({"operation": "nothing was added"}, "no change stated"),
])
def test_no_change(fields, reason):
    assert any(r.startswith(reason) for r in cn.check(_with(**fields), BASE))


@pytest.mark.parametrize("fields", [
    {"operation": "remove the cap added in v3"},
    {"base_version": f"{BASE} (the seed; {BASE}-v3 lowered the result)"},
    {"base_version": f"configs/managers/rule_based/{BASE}.yml"},
    {"paper": "Nowak, M. A. and Sigmund, K., Evolution of indirect reciprocity"},
])
def test_no_false_failures(fields):
    assert cn.check(_with(**fields), BASE) == []


def test_a_value_on_the_next_line():
    text = _with(paper="", analysis="none")
    text = text.replace("- Paper: ", "- Paper:\n  Fehr and Gaechter, Altruistic punishment"
                        " in humans")
    assert cn.check(text, BASE) == []


def test_another_base_version_first_still_fails():
    text = _with(base_version=f"{BASE}-v3 (built on {BASE})")
    assert cn.check(text, BASE)[0].startswith("'- Base version:' must name")


def test_check_notes_needs_the_headings_in_order():
    notes = (NOTES / "si-ds-2.md").read_text()
    assert cn.check_notes(notes, BASE) == []
    swapped = notes.replace("## Key findings", "## Findings")
    assert "## Key findings" in cn.check_notes(swapped, BASE)[0]
