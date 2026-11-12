"""Prompt files and message layout.

Layout: a system message, then one user message whose first part (instructions,
then the codebook if any) is byte-identical for every item of a pass and whose
last part is per item: the context (game setting of the campaign, position in
the lineage), then the document. This maximises server-side prefix-cache reuse.
"""

import hashlib
import json
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
    """Name, label, definition and examples (each example its own list item,
    since examples may contain semicolons). Only these fields reach a prompt:
    theory candidates, review notes and sources never influence the coding."""
    lines = [f"- `{c['name']}` ({c.get('label', '')}): {c['definition']}"]
    ex = c.get("examples") or {}
    for key, title in (("fits", "fits"), ("does_not_fit", "does not fit")):
        if ex.get(key):
            lines.append(f"  {title}:")
            lines.extend(f"    - {e}" for e in ex[key])
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


# Prompts each LLM step reads and a logic version per step (bump it when a
# step's coding logic changes without a prompt change). Together with the
# config that shapes the results they form the step's fingerprint, which the
# manifest records; a step whose fingerprint changed is rerun.
_CONSOLIDATE = ["system", "consolidate", "consolidate_merge", "place", "assign"]
STEP_PROMPTS = {
    "open-rules": ["system", "open_rules"],
    "open-notes": ["system", "open_notes"],
    "what-codebook": ["system", "what_propose", "refine", "assign"],
    "coverage": ["system", "what_propose", "refine", "assign", "judge"],
    "consolidate-what": _CONSOLIDATE,
    "why-codebook": ["system", "why_propose", "refine", "assign"],
    "consolidate-why": _CONSOLIDATE,
    "basis-codebook": ["system", "basis_propose", "refine", "assign"],
    "consolidate-basis": _CONSOLIDATE,
    "apply-rules": ["system", "judge"],
    "apply-notes": ["system", "assign", "assign_multi"],
}
STEP_VERSION = {step: 1 for step in STEP_PROMPTS}
# 2: notes ideas get up to three what codes, with description and quote as
# context; compound reasons are coded by their main claim
STEP_VERSION["apply-notes"] = 2
_CFG_KEYS = ("clustering", "codebook", "coverage", "judge", "sampling", "settings")
_LLM_KEYS = (
    "temperature",
    "temperature_generate",
    "seed_generate",
    "max_tokens",
    "extra_body",
)


def fingerprint(step, cfg):
    """sha256 over the step's prompts, logic version and result-shaping config."""
    h = hashlib.sha256(f"{step}:{STEP_VERSION.get(step, 0)}".encode())
    for name in STEP_PROMPTS.get(step, []):
        path = PROMPT_DIR / f"{name}.md"
        if path.exists():
            h.update(name.encode() + path.read_bytes())
    sub = {k: cfg.get(k) for k in _CFG_KEYS}
    sub["llm"] = {k: (cfg.get("llm") or {}).get(k) for k in _LLM_KEYS}
    h.update(json.dumps(sub, sort_keys=True, default=str).encode())
    return h.hexdigest()
