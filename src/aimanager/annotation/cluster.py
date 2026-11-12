"""Codebook induction for a few hundred short items.

1. one proposal call over all numbered items (seeded sample beyond a cap),
2. an assignment pass: every item to exactly one code or to none,
3. at most N refinement rounds that add codes for the unassigned items,
   each followed by a fresh assignment pass,
then codes with too few members are dropped.
"""

import random

from . import prompting as P
from .client import Call
from .common import log, slugify
from .schemas import Proposal, assignment_model

UNITS = {
    "what": "a mechanism, i.e. something a manager does to players",
    "why": "a reason given for or against a manager mechanism",
}


def to_codebook_entries(proposed, taken, source):
    """Proposed codes -> codebook entries with unique slug names."""
    out = []
    for c in proposed:
        name = base = slugify(c.name)
        k = 2
        while name in taken:
            name, k = f"{base}_{k}", k + 1
        taken.add(name)
        out.append(
            {
                "name": name,
                "label": c.label,
                "definition": c.definition,
                "examples": {"fits": c.fits, "does_not_fit": c.does_not_fit},
                "theory_link": None,
                "source": source,
            }
        )
    return out


def assign(client, step, codes, items, kind, label=None):
    """{item_id: code name | None}, plus the ids whose call failed."""
    names = [c["name"] for c in codes]
    model_cls = assignment_model(names)
    instr = P.load("assign").replace("$unit", UNITS[kind])
    shared = P.render_codebook(codes)
    calls = [
        Call(
            step,
            iid,
            P.messages(instr, text, shared),
            model_cls,
            "assign",
            meta={"codes": names, "text": text},
        )
        for iid, text in items
    ]
    out, failed = {}, []
    for res in client.map(calls, label=label or f"{step} assign"):
        iid = res.call.item_id
        if res.ok:
            out[iid] = None if res.value.code == "none" else res.value.code
        else:
            out[iid] = None
            failed.append(iid)
    return out, failed


def _propose(client, step, kind, items, codes, rnd):
    instr = P.load(f"{kind}_propose")
    shared = ""
    if rnd:
        instr = instr + "\n\n" + P.load("refine")
        shared = P.render_codebook(codes, "Current codebook")
    call = Call(
        step,
        f"{kind}-propose-r{rnd}",
        P.messages(instr, P.render_items(items), shared),
        Proposal,
        "propose",
        meta={
            "kind": kind,
            "items": items,
            "round": rnd,
            "existing": [c["name"] for c in codes],
        },
    )
    (res,) = client.map([call])
    if not res.ok:
        raise SystemExit(f"{step}: codebook proposal failed: {res.error}")
    return res.value.codes


def cluster(client, step, kind, items, ccfg, source, taken=None):
    """Returns (codes, assignment {id: code|None}, failed ids, history)."""
    taken = set(taken or ())
    items = list(items)
    sample = items
    cap = ccfg.get("max_items_in_proposal", 400)
    if len(items) > cap:
        sample = random.Random(len(items)).sample(items, cap)
    codes = to_codebook_entries(
        _propose(client, step, kind, sample, [], 0), taken, source
    )
    history = [{"round": 0, "new_codes": [c["name"] for c in codes]}]
    rounds = ccfg.get("refine_rounds", 2)
    for rnd in range(rounds + 1):
        assignment, failed = assign(client, step, codes, items, kind)
        unassigned = [(i, t) for i, t in items if assignment[i] is None]
        unassigned = [x for x in unassigned if x[0] not in failed]
        history[-1]["unassigned_after"] = len(unassigned)
        log(f"  {step}: round {rnd}, {len(codes)} codes, {len(unassigned)} unassigned")
        if rnd == rounds or len(unassigned) < ccfg.get("min_unassigned", 3):
            break
        new = to_codebook_entries(
            _propose(client, step, kind, unassigned, codes, rnd + 1), taken, source
        )
        history.append({"round": rnd + 1, "new_codes": [c["name"] for c in new]})
        if not new:
            break
        codes = codes + new
    counts = {c["name"]: 0 for c in codes}
    for code in assignment.values():
        if code:
            counts[code] += 1
    keep = [c for c in codes if counts[c["name"]] >= ccfg.get("min_members", 1)]
    kept = {c["name"] for c in keep}
    dropped = sorted(set(counts) - kept)
    if dropped:
        log(f"  {step}: dropped codes without enough members: {dropped}")
        assignment = {i: (c if c in kept else None) for i, c in assignment.items()}
    history.append({"dropped": dropped})
    return keep, assignment, failed, history


def members(codes, assignment, failed):
    """Side file for auditing: item ids per code, plus none and failed."""
    out = {c["name"]: [] for c in codes}
    out["none"] = []
    for iid, code in assignment.items():
        if iid in failed:
            continue
        out[code or "none"].append(iid)
    out["failed"] = list(failed)
    return out
