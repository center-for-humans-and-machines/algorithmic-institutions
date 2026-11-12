"""Codebook induction for a few hundred short items.

1. one proposal call over all numbered items (seeded sample beyond a cap),
2. an assignment pass: every item to exactly one code or to none,
3. at most N refinement rounds that add codes for the unassigned items (same
   cap), each followed by a fresh assignment pass,
then codes with too few members are dropped. Items whose assignment call
failed are never evidence: they are neither unassigned nor members, and a code
is dropped only if it would stay below the minimum even if every failed item
belonged to it.
"""

import random
import re

from . import prompting as P
from .client import Call
from .common import StepFailed, check_failures, log, slugify
from .schemas import RESERVED, Proposal, assignment_model

UNITS = {
    "what": "a mechanism, i.e. something a manager does to players",
    "why": "a reason given for or against a manager mechanism",
    "basis": "what a stated reason rests on, in the search agent's words",
}

# A code is something that is done or claimed; a code defined as an absence
# ("does not punish ...") is true of almost every rule and is rejected.
_NEG_NAME = re.compile(
    r"^(do_not|dont|does_not|doesn_t|avoid|avoids|ignore|ignores|never|no|not|"
    r"refrain|refrains|without)(_|$)"
)
_NEG_TEXT = re.compile(
    r"^(do not|does not|don't|doesn't|avoids?\b|avoiding\b|ignores?\b|ignoring\b|"
    r"never\b|no\b|not\b|refrains?\b|without\b)"
)
_SUBJECT = re.compile(r"^(the )?(manager|rule|manager rule|agent)( rule)? ", re.I)


def is_negated(name, label="", definition=""):
    """Whether a code's name, label or definition starts with a negation."""
    if _NEG_NAME.match(name or ""):
        return True
    for text in (label, definition):
        t = _SUBJECT.sub("", (text or "").strip().lower())
        if _NEG_TEXT.match(t):
            return True
    return False


def drop_negated(codes, step, what):
    """(kept, rejected names) of proposed codes; logs what was rejected."""
    kept, rejected = [], []
    for c in codes:
        if is_negated(c.name, c.label, c.definition):
            rejected.append(c.name)
        else:
            kept.append(c)
    if rejected:
        log(f"  {step}: rejected {len(rejected)} negated {what} codes: {rejected}")
    return kept, rejected


def to_codebook_entries(proposed, taken, source):
    """Proposed codes -> codebook entries with unique, non-reserved slug names."""
    taken.update(RESERVED)
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


def assign(client, cfg, step, codes, items, kind, label=None):
    """({item_id: code name | None}, [(item_id, error)] of failed calls).

    Failed items are not in the mapping. Raises StepFailed beyond the budget.
    """
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
            failed.append((iid, res.error))
    check_failures(cfg, step, "assign", failed, len(items))
    return out, failed


def est_tokens(text):
    """Conservative token estimate for budgeting (about 3 characters per token)."""
    return len(text) // 3 + 1


def fit_items(items, budget_tokens, cap, seed):
    """At most `cap` items whose rendered list fits `budget_tokens`.

    A seeded sample, kept in input order, when the items exceed either limit.
    """
    if len(items) <= cap and est_tokens(P.render_items(items)) <= budget_tokens:
        return items
    order = random.Random(seed).sample(range(len(items)), len(items))
    keep, used = set(), 0
    for i in order:
        cost = est_tokens(items[i][1]) + 3  # number, newline, indentation
        if len(keep) >= cap or used + cost > budget_tokens:
            continue
        keep.add(i)
        used += cost
    return [it for i, it in enumerate(items) if i in keep]


def _propose(client, cfg, step, kind, items, codes, rnd):
    cap = cfg["clustering"].get("max_items_in_proposal", 400)
    instr = P.load(f"{kind}_propose")
    shared = ""
    if rnd:
        instr = instr + "\n\n" + P.load("refine")
        shared = P.render_codebook(codes, "Current codebook")
    # prompt + the largest max_tokens of a proposal must fit the context length
    fixed = est_tokens(P.load("system") + instr + shared) + 256
    budget = client.max_model_len - cfg["llm"]["max_tokens"]["propose"]["retry"] - fixed
    if budget <= 0:
        raise StepFailed(
            f"{step}: no room for items in a proposal: max_model_len "
            f"{client.max_model_len} minus max_tokens.propose.retry minus the "
            "instructions and codebook is not positive"
        )
    n_all = len(items)
    items = fit_items(items, budget, cap, n_all * 1000 + rnd)
    if len(items) < n_all:
        log(
            f"  {step}: {kind} proposal round {rnd} sees {len(items)} of {n_all} "
            f"items ({n_all - len(items)} sampled out; cap {cap}, about {budget} "
            "tokens for items)"
        )
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
        check_failures(cfg, step, "propose", [(call.item_id, res.error)], 1)
        return [], []
    return drop_negated(res.value.codes, step, kind)


def cluster(client, cfg, step, kind, items, source, taken=None):
    """Returns (codes, assignment {id: code|None}, failed [(id, error)], history)."""
    ccfg = cfg["clustering"]
    taken = set(taken or ())
    items = list(items)
    proposed, rejected = _propose(client, cfg, step, kind, items, [], 0)
    codes = to_codebook_entries(proposed, taken, source)
    history = [
        {
            "round": 0,
            "new_codes": [c["name"] for c in codes],
            "rejected_negated": rejected,
        }
    ]
    rounds = ccfg.get("refine_rounds", 2)
    for rnd in range(rounds + 1):
        assignment, failed = assign(client, cfg, step, codes, items, kind)
        unassigned = [(i, t) for i, t in items if i in assignment and not assignment[i]]
        history[-1]["unassigned_after"] = len(unassigned)
        history[-1]["assign_failed"] = len(failed)
        log(f"  {step}: round {rnd}, {len(codes)} codes, {len(unassigned)} unassigned")
        if rnd == rounds or len(unassigned) < ccfg.get("min_unassigned", 3):
            break
        proposed, rejected = _propose(
            client, cfg, step, kind, unassigned, codes, rnd + 1
        )
        new = to_codebook_entries(proposed, taken, source)
        history.append(
            {
                "round": rnd + 1,
                "new_codes": [c["name"] for c in new],
                "rejected_negated": rejected,
            }
        )
        if not new:
            break
        codes = codes + new
    counts = {c["name"]: 0 for c in codes}
    for code in assignment.values():
        if code:
            counts[code] += 1
    floor = ccfg.get("min_members", 1)
    keep = [c for c in codes if counts[c["name"]] + len(failed) >= floor]
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
        out[code or "none"].append(iid)
    out["failed"] = [iid for iid, _ in failed]
    return out
