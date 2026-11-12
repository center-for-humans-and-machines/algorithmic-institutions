"""Prompt files and message layout.

Layout: a system message, then one user message whose first part (instructions,
then the codebook if any) is byte-identical for every item of a pass and whose
last part is per item: the context (game setting of the campaign, position in
the lineage), then the document. This maximises server-side prefix-cache reuse.
"""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
DOC_MARK = "=== DOCUMENT ==="
CONTEXT_MARK = "=== CONTEXT (not to be coded) ==="


def load(name):
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def messages(instructions, document, shared="", context=""):
    head = instructions + (f"\n\n{shared}" if shared else "")
    ctx = f"{CONTEXT_MARK}\n{context.strip()}\n\n" if context else ""
    return [
        {"role": "system", "content": load("system")},
        {"role": "user", "content": f"{head}\n\n{ctx}{DOC_MARK}\n{document}"},
    ]


def lineage_position(version):
    """Per-document line: first version of a lineage or a later one."""
    if not version.get("parent"):
        return (
            f"Position: {version['stem']} is the first version of lineage "
            f"{version['lineage']}; there is no previous version."
        )
    return (
        f"Position: {version['stem']} is version {version['version']} of lineage "
        f"{version['lineage']}; the previous version is {version['parent']}."
    )


def _render_code(c):
    lines = [f"- `{c['name']}` ({c.get('label', '')}): {c['definition']}"]
    ex = c.get("examples") or {}
    if ex.get("fits"):
        lines.append("  fits: " + "; ".join(ex["fits"]))
    if ex.get("does_not_fit"):
        lines.append("  does not fit: " + "; ".join(ex["does_not_fit"]))
    return lines


def render_codebook(codes, title="Codebook", families=None):
    """Codes in the given order. Codes with a `family` are listed under their
    family (families in order of first appearance), which helps keep
    neighbouring codes apart; `families` maps family name -> definition."""
    lines = [f"## {title}", ""]
    if codes and all(c.get("family") for c in codes):
        defs = families or {}
        order = list(dict.fromkeys(c["family"] for c in codes))
        for fam in order:
            head = f"### Family `{fam}`"
            lines.append(head + (f": {defs[fam]}" if defs.get(fam) else ""))
            for c in codes:
                if c["family"] == fam:
                    lines.extend(_render_code(c))
            lines.append("")
    else:
        for c in codes:
            lines.extend(_render_code(c))
    if not codes:
        lines.append("(empty)")
    return "\n".join(lines).rstrip()


def render_items(items):
    """Numbered list of (id, text) items; numbers are 1-based positions.

    Continuation lines of a multi-line item (a context line) are indented.
    """
    return "\n".join(
        f"{i}. " + text.replace("\n", "\n   ") for i, (_, text) in enumerate(items, 1)
    )
