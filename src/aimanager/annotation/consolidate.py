"""Consolidation of a codebook: merges and families, derived from the codes.

One call receives every code (for what: the draft, the accepted candidates and
the never-implemented candidates; for why: the proposed codes) and returns
(a) merges of codes that are one code, with the surviving name and a rewritten
definition, and (b) a family for every surviving code. No families are given.
The code applies the reply defensively: unknown or already merged names are
skipped, a rewritten definition that is a negation keeps the old one, and a code
that no family lists goes to the family `unplaced`. Every decision is in the
audit file codebook/<kind>.merges.yml.
"""

from . import prompting as P
from .client import Call
from .cluster import is_negated
from .common import StepFailed, log, slugify
from .schemas import consolidation_model

UNPLACED = "unplaced"

KIND_TEXT = {
    "what": (
        "The codes are manager mechanisms: things a manager does to players. "
        "Families group mechanisms by the kind of lever they use (for example, in "
        "an unrelated domain, library rules group into 'fees', 'borrowing limits' "
        "and 'timing of reminders')."
    ),
    "why": (
        "The codes are reasons the search agent gives for or against mechanisms. "
        "Families group reasons by the kind of justification they are, not by the "
        "mechanism they are about and not by their stance (in an unrelated domain, "
        "a librarian's reasons might group into 'effects on borrowers', 'costs to "
        "the library' and 'what the records show')."
    ),
}


def consolidation_call(step, kind, codes):
    instr = P.load("consolidate").replace("$kind_text", KIND_TEXT[kind])
    names = [c["name"] for c in codes]
    return Call(
        step,
        f"{kind}-consolidate",
        P.messages(instr, P.render_codebook(codes, "Codes")),
        consolidation_model(names),
        "consolidate",
        meta={"kind": kind, "codes": names},
    )


def apply_consolidation(codes, merges, families):
    """(surviving codes with `family`, {family: definition}, audit, unplaced).

    `codes` are codebook entries (an `implemented` flag is carried through: a
    survivor is implemented if any merged code was); `merges` and `families`
    are plain dicts from the reply.
    """
    by = {c["name"]: dict(c) for c in codes}
    gone, audit = {}, []
    for m in merges:
        group = [n for n in dict.fromkeys(m["codes"]) if n in by and n not in gone]
        if len(group) < 2:
            audit.append({"merge": m["codes"], "applied": False, "why": "fewer than 2"})
            continue
        keep = m["keep"] if m["keep"] in group else group[0]
        surv = by[keep]
        note = None
        if is_negated(keep, m["label"], m["definition"]):
            note = "rewritten definition is a negation; old one kept"
        else:
            surv["label"], surv["definition"] = m["label"], m["definition"]
        absorbed = [n for n in group if n != keep]
        surv["merged_from"] = sorted(set(surv.get("merged_from", [])) | set(absorbed))
        if "implemented" in surv:
            surv["implemented"] = any(by[n].get("implemented") for n in group)
        for n in absorbed:
            gone[n] = keep
            surv["merged_from"] = sorted(
                set(surv["merged_from"]) | set(by[n].get("merged_from", []))
            )
        audit.append(
            {
                "merge": group,
                "keep": keep,
                "applied": True,
                "definition": m["definition"],
            }
            | ({"note": note} if note else {})
        )
    survivors = [by[c["name"]] for c in codes if c["name"] not in gone]
    names = {c["name"] for c in survivors}
    defs, placed = {}, {}
    for f in families:
        fam = slugify(f["name"])
        defs.setdefault(fam, f["definition"])
        for n in f["codes"]:
            target = gone.get(n, n)
            if target in names and target not in placed:
                placed[target] = fam
    unplaced = [c["name"] for c in survivors if c["name"] not in placed]
    if unplaced:
        defs[UNPLACED] = "codes the consolidation did not place in a family"
    for c in survivors:
        c["family"] = placed.get(c["name"], UNPLACED)
    used = {c["family"] for c in survivors}
    defs = {k: v for k, v in defs.items() if k in used}
    return survivors, defs, audit, unplaced


def consolidate_codes(client, cfg, step, kind, codes):
    """Run the consolidation call and apply it; StepFailed if the call fails."""
    call = consolidation_call(step, kind, codes)
    (res,) = client.map([call])
    if not res.ok:
        raise StepFailed(f"{step}: the consolidation call failed: {res.error[:300]}")
    value = res.value.model_dump()
    survivors, defs, audit, unplaced = apply_consolidation(
        codes, value["merges"], value["families"]
    )
    n_merged = sum(len(a["merge"]) - 1 for a in audit if a["applied"])
    log(
        f"  {step}: {len(codes)} codes -> {len(survivors)} after {n_merged} merges, "
        f"{len(defs)} families, {len(unplaced)} unplaced"
    )
    return survivors, defs, audit, unplaced


def families_doc(defs, codes):
    """The families a set of codes uses, in order, as [{name, definition}]."""
    used = list(dict.fromkeys(c["family"] for c in codes))
    return [{"name": f, "definition": defs.get(f, "")} for f in used]
