"""Consolidation of a codebook: merges and families, derived from the codes.

One call receives every code (for what: the draft, the accepted candidates and
the never-implemented candidates; for why: the proposed codes) and returns
(a) merges of codes that are one code, with the surviving name and a rewritten
definition, and (b) a family for every surviving code. No families are given.
The code applies the reply defensively: unknown or already merged names are
skipped, a rewritten definition that is a negation keeps the old one, and a code
that no family lists is re-asked once against the existing families and only
then gets a family of its own. Every decision is in the audit file
codebook/<kind>.merges.yml, each merge with the model's stated reason.
"""

from . import prompting as P
from .client import Call
from .cluster import is_negated
from .common import StepFailed, log, slugify
from .schemas import consolidation_model, placement_model

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
    "basis": (
        "The codes say what a reason of the search agent rests on. Families are "
        "the few kinds of support a reason can point to; the family is the point "
        "of this codebook. In an unrelated domain, a librarian's reasons might "
        "rest on 'the library's own loan statistics', 'a trial run at one branch', "
        "'what happened last year' or 'an untested expectation'; these illustrate "
        "the level, they are not the families to use. A code that describes the "
        "content of a finding (what was found) rather than the kind of support is "
        "merged into the code for the kind of support it rests on."
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
                "reason": m.get("reason", ""),
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


def placement_call(step, kind, codes, defs):
    """Re-ask: a family from the existing list for each unplaced code."""
    fams = [f for f in defs if f != UNPLACED]
    doc = "## Families\n\n" + "\n".join(f"- `{f}`: {defs[f]}" for f in fams)
    doc += "\n\n" + P.render_codebook([dict(c, family=None) for c in codes], "Codes")
    names = [c["name"] for c in codes]
    return Call(
        step,
        f"{kind}-place",
        P.messages(P.load("place"), doc),
        placement_model(names, fams),
        "place",
        meta={"codes": names, "families": fams},
    )


def apply_placements(survivors, defs, placements):
    """Place unplaced codes as the re-ask says; the rest get a family of their
    own (named after the code). Returns (defs, codes placed by re-ask, own)."""
    by = {c["name"]: c for c in survivors}
    replaced, own = [], []
    for p in placements:
        c = by.get(p["code"])
        if c is not None and c["family"] == UNPLACED and p["family"] in defs:
            c["family"] = p["family"]
            replaced.append(c["name"])
    for c in survivors:
        if c["family"] == UNPLACED:
            c["family"] = c["name"]
            defs[c["name"]] = c.get("definition", "")
            own.append(c["name"])
    defs = {k: v for k, v in defs.items() if k != UNPLACED}
    return defs, replaced, own


def consolidate_codes(client, cfg, step, kind, codes):
    """Run the consolidation call and apply it; codes left unplaced are re-asked
    once against the existing families, then get a family of their own.
    StepFailed if a call fails. Returns (survivors, defs, audit, placement)."""
    call = consolidation_call(step, kind, codes)
    (res,) = client.map([call])
    if not res.ok:
        raise StepFailed(f"{step}: the consolidation call failed: {res.error[:300]}")
    value = res.value.model_dump()
    survivors, defs, audit, unplaced = apply_consolidation(
        codes, value["merges"], value["families"]
    )
    placement = {"unplaced_first": unplaced, "placed_on_reask": [], "own_family": []}
    if unplaced:
        left = [c for c in survivors if c["name"] in set(unplaced)]
        placements = []
        if any(f != UNPLACED for f in defs):
            (pres,) = client.map([placement_call(step, kind, left, defs)])
            if not pres.ok:
                raise StepFailed(
                    f"{step}: the placement call failed: {pres.error[:300]}"
                )
            placements = pres.value.model_dump()["placements"]
        defs, replaced, own = apply_placements(survivors, defs, placements)
        placement.update(placed_on_reask=replaced, own_family=own)
    n_merged = sum(len(a["merge"]) - 1 for a in audit if a["applied"])
    log(
        f"  {step}: {len(codes)} codes -> {len(survivors)} after {n_merged} merges, "
        f"{len(defs)} families; unplaced {len(unplaced)}, placed on re-ask "
        f"{len(placement['placed_on_reask'])}, "
        f"own family {len(placement['own_family'])}"
    )
    return survivors, defs, audit, placement


def families_doc(defs, codes):
    """The families a set of codes uses, in order, as [{name, definition}]."""
    used = list(dict.fromkeys(c["family"] for c in codes))
    return [{"name": f, "definition": defs.get(f, "")} for f in used]
