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
- a value on the line after its label, a base version given as the path to
  its rule or followed by a remark that names another version, an operation
  followed by its object (`remove the cap added in v3`: one removal).

Empty, for a reference or an analysis: `none`, `no paper`, `not run`,
`n/a`, `N.A.`, `(none)`, `?`, a dash, `unknown`, `null`, the template line
left as it was (`<title, ...>, or none`). No change: an idea such as
`N/A (no change)`, an operation such as `nothing was added`.

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
#: The labels of the section (a line with another label continues the value
#: before it)
LABELS = ("base version", "operation", "idea", "justification", "paper", "claim",
          "analysis", "expected")
#: The operations, with their usual inflections
OPERATIONS = {
    "add": re.compile(r"^add(s|ed|ing|ition)?$", re.I),
    "remove": re.compile(r"^remov(e|es|ed|ing|al)$", re.I),
    "modify": re.compile(r"^modif(y|ies|ied|ying|ication)$", re.I),
}
#: Words that may join operations at the start of the value ("add and modify")
JOINERS = {"and", "then", "plus", "also", "or", "&", "+", "/", ","}
#: A reference or analysis that is not there: none, no paper, not run, n/a,
#: (none), ?, a dash, unknown, null, the template line left as it was
EMPTY = re.compile(
    r"^[\s(\[]*(none|no|not|n/?a|n\.a|unknown|null|nil|tbd|\?+|[-\u2013\u2014]+)"
    r"(\b|[\s.,;:)\]]|$).*$|^\s*<.*$|^\s*$",
    re.I,
)
#: An idea that says there is none
NO_IDEA = re.compile(r"^[\s(\[]*(none|n/?a|n\.a|nothing|no change|-+|\?+)\b|^\s*<|^\s*$",
                     re.I)
#: The notes' required headings, in order (check_instance.sh), the Change
#: section last
HEADINGS = ("## Explorations", "## Key findings", "## Hypothesis", "### Justification",
            "## Change")


def clean(value):
    """A value without the markup around it: backticks, bold, quotes."""
    value = re.sub(r"(\*\*|__|`)", "", value)
    return value.strip().strip("\"'").strip()


def fields(text):
    """{label: [values]} of the section's `- Label: value` lines, labels in
    lower case. A label whose value is on the line after it gets that line."""
    out, last = {}, None
    for line in text.splitlines():
        m = FIELD.match(line)
        label = " ".join(m.group("label").lower().split()) if m else None
        if m and label in LABELS:
            out.setdefault(label, []).append(clean(m.group("value")))
            last = label
        elif last is not None and not out[last][-1] and line.strip():
            # the value on the next line: `- Paper:` then `  Fehr, ...`
            out[last][-1] = clean(re.sub(r"^\s*[-*+]\s+", "", line))
        elif not line.strip() or line.lstrip().startswith("#"):
            last = None
    return out


def filled(value):
    return value is not None and not EMPTY.match(value)


def operations(value):
    """The operations the value starts with ("add", "remove the cap added in
    v3": remove; "add and modify": both; "nothing was added": none)."""
    found = []
    for word in re.findall(r"[A-Za-z]+|[&+/,]", value):
        op = next((name for name, pattern in OPERATIONS.items()
                   if pattern.match(word)), None)
        if op:
            found.append(op)
        elif word.lower() not in JOINERS:
            break
    return found


def base_ok(value, base):
    """The value names the base version first (a path to its rule, or a
    remark naming another version after it, is fine)."""
    name = re.sub(r"-v\d+$", "", base)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
    lineage = [t for t in tokens if t == name or re.fullmatch(re.escape(name) + r"-v\d+", t)]
    return bool(lineage) and lineage[0] == base


def check(text, base):
    """The reasons the section fails, [] when it states one justified change."""
    f = fields(text)
    reasons = []
    bases = f.get("base version", [])
    if len(bases) != 1 or not base_ok(bases[0], base):
        reasons.append(f"'- Base version:' must name {base}"
                       + (f" (it says {bases[0]!r})" if len(bases) == 1 else ""))
    ops = f.get("operation", [])
    named = {op for v in ops for op in operations(v)}
    if len(ops) > 1 or len(named) > 1:
        reasons.append("several changes: one '- Operation:' (add, remove or modify)"
                       f" per version, not {ops}")
    elif not named:
        reasons.append("no change stated: '- Operation:' must be add, remove or modify")
    ideas = [v for v in f.get("idea", []) if not NO_IDEA.match(v)]
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


def check_notes(text, base):
    """A single-idea version's whole notes, as check_instance.sh reads them:
    the headings in order, then the Change section (check)."""
    lines = text.splitlines()
    last = -1
    for heading in HEADINGS:
        at = next((i for i, line in enumerate(lines) if line == heading), None)
        if at is None or at <= last:
            return [f"the notes need the section '{heading}', in order"]
        last = at
    return check("\n".join(lines[last + 1:]), base)


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
