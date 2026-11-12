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
from .schemas import (
    PROPOSAL_CAP,
    RESERVED,
    Proposal,
    assignment_model,
    multi_assignment_model,
)

UNITS = {
    "what": "a mechanism, i.e. something a manager does to players",
    "why": "a reason given for or against a manager mechanism",
    "basis": "what a stated reason rests on, in the search agent's words",
}

# A mechanism code that states that something is NOT conditioned on or NOT used
# ("ignore the opponent", "does not depend on history", "exclude group size as
# a factor") is true of almost every rule and is rejected. Time-gating levers
# ("no punishment in round X", "stops punishing after round T", "suspends
# punishment before a switch") are positive levers about when punishment
# applies and pass.
_FACTOR = (
    r"(based_on|depend|depends|dependent|dependence|condition|conditioned|"
    r"conditioning|conditional|use|uses|using|react|reacts|reacting|respond|"
    r"adjust|adjusts|adjusting|account|consider|considering|track|tracks|tracking)"
)
_NEG_NAME = re.compile(
    r"(^|_)(ignore|ignores|ignoring|disregard|disregards|independent_of|"
    r"unconditional|unconditionally|regardless_of)(_|$)"
    # a negation followed within two words by what is not used or followed
    r"|(^|_)(do|does|dont|doesn_t|do_not|does_not|should_not|must_not|never|"
    r"avoid|avoids|avoiding|not)_(\w+_){0,2}?" + _FACTOR + r"(_|$)"
    # `no_` only directly before the dependence itself: no_response_to_x, but
    # not no_punishment_until_condition_met (a time-gating lever)
    r"|(^|_)no_(dependence|reaction|response|adjustment|conditioning)(_|$)"
    r"|(^|_)as_a_factor(_|$)"
)
_NEG_TEXT = re.compile(
    r"^(ignores?\b|ignoring\b|disregards?\b"
    r"|(does|do|is|are) not (depend|use|condition|react|respond|adjust|take|"
    r"account|consider|track|vary)\w*"
    r"|(is |are )?not (conditioned|based|dependent) on"
    r"|(is |are )?independent of"
    r"|no (dependence|reaction|adjustment) (on|to)"
    r"|exclud\w* \S+( \S+)? as a factor)"
)
_SUBJECT = re.compile(r"^(the )?(manager|rule|manager rule|agent)( rule)? ", re.I)


def is_negated(name, label="", definition=""):
    """Whether a mechanism code says that something is not conditioned on or not
    used, by its name or by how its label or definition starts."""
    if _NEG_NAME.search(name or ""):
        return True
    for text in (label, definition):
        t = _SUBJECT.sub("", (text or "").strip().lower())
        if _NEG_TEXT.match(t):
            return True
    return False


_EXCLUSION = re.compile(r"^(exclude|excludes|excluding|forbid|forbids|omit|omits)(_|$)")


def is_negated_or_exclusion(code):
    """Merge guard for mechanisms: a negated code, or one that excludes a
    feature (`exclude_group_size_from_punishment`), never merges with another."""
    name = code.get("name", "")
    return bool(
        is_negated(name, code.get("label", ""), code.get("definition", ""))
        or _EXCLUSION.match(name)
    )


def drop_negated(codes, step, what):
    """(kept, rejected) of proposed codes; rejected codes are kept with their
    label and definition for the history file, and logged."""
    kept, rejected = [], []
    for c in codes:
        if is_negated(c.name, c.label, c.definition):
            rejected.append(
                {"name": c.name, "label": c.label, "definition": c.definition}
            )
        else:
            kept.append(c)
    if rejected:
        names = [r["name"] for r in rejected]
        log(f"  {step}: rejected {len(rejected)} negated {what} codes: {names}")
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


def assign(client, cfg, step, codes, items, kind, label=None, families=None):
    """({item_id: code name | None}, [(item_id, error)] of failed calls).

    Failed items are not in the mapping. Raises StepFailed beyond the budget.
    `families` ({family: definition}) puts the definitions under the headings.
    """
    names = [c["name"] for c in codes]
    model_cls = assignment_model(names)
    instr = P.load("assign").replace("$unit", UNITS[kind])
    if kind == "why":
        instr += (
            " A reason that makes several claims gets the code of its main claim, "
            "the one the reason is given for."
        )
    shared = P.render_codebook(codes, families=families)
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


def gen_sampling(cfg):
    """(temperature, seed) for generative calls: proposals and consolidation.

    Long list generation at temperature 0 looped on DeepSeek (prod1: one name 29
    times), and its model card recommends a non-zero temperature.
    """
    llm = cfg["llm"]
    return llm.get("temperature_generate", 0.6), llm.get("seed_generate", 20261112)


def target_codes(cfg, kind):
    """The configured target size of a codebook (a setting, not a limit)."""
    return (cfg.get("codebook", {}).get(kind) or {}).get("target_codes")


def dedupe_proposal(codes):
    """(codes with each name once, keeping the first; names seen more than once
    with their counts; highest count of one name)."""
    seen, out, counts = set(), [], {}
    for c in codes:
        key = slugify(c.name)
        counts[key] = counts.get(key, 0) + 1
        if key not in seen:
            seen.add(key)
            out.append(c)
    dups = {k: n for k, n in counts.items() if n > 1}
    return out, dups, max(counts.values(), default=0)


def is_degenerate(codes, cap, max_repeats):
    """A reply that repeats one name more than `max_repeats` times, or fills the
    schema cap with repeated names, came from a generation loop."""
    _, dups, top = dedupe_proposal(codes)
    return top > max_repeats or (len(codes) >= cap and bool(dups))


def _fill(text, **values):
    for k, v in values.items():
        text = text.replace(f"${k}", str(v))
    return text


def assign_multi(client, cfg, step, codes, items, kind, families=None, max_codes=3):
    """({item_id: [code names, best first]}, failed) for items that may name
    several codes (notes ideas). [] means no code fits. Raises StepFailed beyond
    the assignment failure budget."""
    names = [c["name"] for c in codes]
    model_cls = multi_assignment_model(names, max_codes)
    instr = P.load("assign_multi").replace("$unit", UNITS[kind])
    shared = P.render_codebook(codes, families=families)
    calls = [
        Call(
            step,
            iid,
            P.messages(instr, text, shared),
            model_cls,
            "assign",
            meta={"codes": names, "text": text, "multi": True},
        )
        for iid, text in items
    ]
    out, failed = {}, []
    for res in client.map(calls, label=f"{step} assign (multi)"):
        iid = res.call.item_id
        if res.ok:
            out[iid] = list(dict.fromkeys(res.value.codes))
        else:
            failed.append((iid, res.error))
    check_failures(cfg, step, "assign", failed, len(items))
    return out, failed


def _propose(client, cfg, step, kind, items, codes, rnd, target=None, info=None):
    """Proposed codes (deduplicated; negated what codes dropped) and the names
    rejected as negated. `info` collects degenerate replies and duplicates."""
    info = info if info is not None else {}
    cap = cfg["clustering"].get("max_items_in_proposal", 400)
    target = target or target_codes(cfg, kind) or 20
    instr = _fill(P.load(f"{kind}_propose"), target=target)
    shared = ""
    if rnd:
        min_members = max(cfg["clustering"].get("min_members", 1), 2)
        instr = (
            instr
            + "\n\n"
            + _fill(
                P.load("refine"),
                n_existing=len(codes),
                target=target,
                min_members=min_members,
            )
        )
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
    temperature, seed = gen_sampling(cfg)
    max_repeats = cfg["clustering"].get("max_name_repeats", 3)

    def call(attempt):
        return Call(
            step,
            f"{kind}-propose-r{rnd}" + (f"-retry{attempt}" if attempt else ""),
            P.messages(instr, P.render_items(items), shared),
            Proposal,
            "propose",
            meta={
                "kind": kind,
                "items": items,
                "round": rnd,
                "existing": [c["name"] for c in codes],
                "attempt": attempt,
            },
            temperature=temperature,
            seed=seed + attempt,
        )

    (res,) = client.map([call(0)])
    if res.ok and is_degenerate(res.value.codes, PROPOSAL_CAP, max_repeats):
        _, dups, _ = dedupe_proposal(res.value.codes)
        log(f"  {step}: degenerate {kind} proposal (repeated {dups}); re-asking")
        info.setdefault("degenerate", []).append({"round": rnd, "repeated": dups})
        (res,) = client.map([call(1)])
        if res.ok and is_degenerate(res.value.codes, PROPOSAL_CAP, max_repeats):
            _, dups, _ = dedupe_proposal(res.value.codes)
            log(f"  {step}: re-ask still degenerate ({dups}); keeping it deduplicated")
            info["degenerate"][-1]["reask_also_degenerate"] = dups
    if not res.ok:
        check_failures(cfg, step, "propose", [(res.call.item_id, res.error)], 1)
        return [], []
    proposed, dups, _ = dedupe_proposal(res.value.codes)
    if dups:
        log(f"  {step}: dropped repeated names in one reply: {dups}")
        info.setdefault("duplicates", []).append({"round": rnd, "repeated": dups})
    # only mechanisms: reasons and kinds of support are often legitimately
    # negative ("late punishment has no future benefit", "no data available")
    if kind != "what":
        return proposed, []
    return drop_negated(proposed, step, kind)


def cluster(client, cfg, step, kind, items, source, taken=None, target=None):
    """Returns (codes, assignment {id: code|None}, failed [(id, error)], history).

    `target` is the size the proposal aims for (default codebook.<kind>.target_codes).
    """
    ccfg = cfg["clustering"]
    taken = set(taken or ())
    items = list(items)
    info = {}
    proposed, rejected = _propose(client, cfg, step, kind, items, [], 0, target, info)
    codes = to_codebook_entries(proposed, taken, source)
    history = [
        {
            "round": 0,
            "new_codes": [c["name"] for c in codes],
            "rejected_negated": rejected,
            **info,
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
        info = {}
        proposed, rejected = _propose(
            client, cfg, step, kind, unassigned, codes, rnd + 1, target, info
        )
        new = to_codebook_entries(proposed, taken, source)
        history.append(
            {
                "round": rnd + 1,
                "new_codes": [c["name"] for c in new],
                "rejected_negated": rejected,
                **info,
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
