"""The `## Change` section of a single-idea version's notes (check_instance.sh).

The section states the version's one change and its justification
(.claude/agents/policy-finder-single-idea.md). The check holds to the idea,
not the layout (Levin, 2026-11-28): the reference indicates a strand of
research, and a title with its lead or senior author is enough. It accepts
the layouts the open-weight agents wrote (scripts/tests/fixtures/
single_idea_notes/, the 8 runs of 2026-11-27):

- the list indented (`  - Base version: ...`) or its labels in bold
  (`- **Base version:** ...`);
- values in backticks or bold (`` `pfsmoke-20261127-1` ``), followed by a
  remark (`pfsmoke-20261127-1 (the zero-punisher seed)`);
- `- Justification:` as `paper and analysis` or `paper + analysis` for
  `both`, or a sentence ("Analysis of human data showing ...");
- `- Claim:` left out or `none` (the reference alone is enough).

It rejects a section without a change (no operation, or no idea), with
several changes (two operations, in one line or in two), with a base
version other than the instance's, or without a justification: neither a
reference (`- Paper:`) nor an analysis (`- Analysis:`) is given.

Usage (reads the section's lines from stdin; exit 1 with the reasons):
    python3 scripts/policy_finder/change_note.py --base <base_version>
"""

import argparse
import re
import sys

#: A `- Label: value` line, indented or not, its label or value in bold
FIELD = re.compile(
    r"^\s*[-*+]\s+(?:\*\*|__)?(?P<label>[A-Za-z][A-Za-z ]*?)(?:\*\*|__)?\s*:"
    r"\s*(?:\*\*|__)?\s*(?P<value>.*?)\s*$"
)
#: The operations, with their usual inflections
OPERATIONS = {
    "add": re.compile(r"\badd(s|ed|ing|ition)?\b", re.I),
    "remove": re.compile(r"\bremov(e|es|ed|ing|al)\b", re.I),
    "modify": re.compile(r"\bmodif(y|ies|ied|ying|ication)\b", re.I),
}
#: An empty value: nothing, `none`, `n/a`, `-`, or the template's `<...>`
EMPTY = re.compile(r"^(none\b.*|n/?a|-+|<.*>)?$", re.I)


def clean(value):
    """A value without the markup around it: backticks, bold, quotes."""
    value = re.sub(r"(\*\*|__|`)", "", value)
    return value.strip().strip("\"'").strip()


def fields(text):
    """{label: [values]} of the section's `- Label: value` lines, labels in
    lower case."""
    out = {}
    for line in text.splitlines():
        m = FIELD.match(line)
        if m:
            label = " ".join(m.group("label").lower().split())
            out.setdefault(label, []).append(clean(m.group("value")))
    return out


def filled(value):
    return value is not None and not EMPTY.match(value.strip())


def base_ok(value, base):
    """The value names the base version and no other version of its lineage."""
    name = re.sub(r"-v\d+$", "", base)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]|[A-Za-z0-9]", value)
    lineage = {t for t in tokens if t == name or t.startswith(name + "-v")}
    return base in tokens and lineage <= {base}


def check(text, base):
    """The reasons the section fails, [] when it states one justified change."""
    f = fields(text)
    reasons = []
    bases = f.get("base version", [])
    if len(bases) != 1 or not base_ok(bases[0], base):
        reasons.append(f"'- Base version:' must name {base}"
                       + (f" (it says {bases[0]!r})" if len(bases) == 1 else ""))
    ops = f.get("operation", [])
    named = {op for v in ops for op, pattern in OPERATIONS.items() if pattern.search(v)}
    if len(ops) > 1 or len(named) > 1:
        reasons.append("several changes: one '- Operation:' (add, remove or modify)"
                       f" per version, not {ops}")
    elif not named:
        reasons.append("no change stated: '- Operation:' must be add, remove or modify")
    ideas = [v for v in f.get("idea", []) if filled(v)]
    if len(ideas) > 1:
        reasons.append("several changes: one '- Idea:' per version")
    elif not ideas:
        reasons.append("no change stated: '- Idea:' is empty")
    paper = [v for v in f.get("paper", []) if filled(v)]
    analysis = [v for v in f.get("analysis", []) if filled(v)]
    if not paper and not analysis:
        reasons.append("no justification: neither '- Paper:' (a reference: title"
                       " and an author are enough) nor '- Analysis:' is given")
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="the instance's base_version")
    args = ap.parse_args()
    reasons = check(sys.stdin.read(), args.base)
    for reason in reasons:
        print(reason)
    sys.exit(1 if reasons else 0)


if __name__ == "__main__":
    main()
