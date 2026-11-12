"""Prompt files and message layout.

Layout: a system message, then one user message whose first part (instructions,
then the codebook if any) is byte-identical for every item of a pass and whose
last part is the document. This maximises server-side prefix-cache reuse.
"""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
DOC_MARK = "=== DOCUMENT ==="


def load(name):
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def messages(instructions, document, shared=""):
    head = instructions + (f"\n\n{shared}" if shared else "")
    return [
        {"role": "system", "content": load("system")},
        {"role": "user", "content": f"{head}\n\n{DOC_MARK}\n{document}"},
    ]


def render_codebook(codes, title="Codebook"):
    lines = [f"## {title}", ""]
    for c in codes:
        lines.append(f"- `{c['name']}` ({c.get('label', '')}): {c['definition']}")
        ex = c.get("examples") or {}
        if ex.get("fits"):
            lines.append("  fits: " + "; ".join(ex["fits"]))
        if ex.get("does_not_fit"):
            lines.append("  does not fit: " + "; ".join(ex["does_not_fit"]))
    if not codes:
        lines.append("(empty)")
    return "\n".join(lines)


def render_items(items):
    """Numbered list of (id, text) items; numbers are 1-based positions."""
    return "\n".join(f"{i}. {text}" for i, (_, text) in enumerate(items, 1))
