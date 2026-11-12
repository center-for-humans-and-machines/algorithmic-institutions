"""Deterministic fake LLM: schema-valid canned replies derived from the input.

Uses keyword vocabularies so that free coding, clustering, coverage, judging,
apply and agreement all produce non-trivial output on small corpora.
"""

import hashlib
import json
import re

# (code name, label, regex on the text)
WHAT = [
    ("punish_below_group_mean", "punish shortfall below the group mean", r"\bmean"),
    ("punishment_cap", "cap the punishment per player", r"\bcap\b|largest"),
    ("tolerance_band", "tolerate small shortfalls", r"\btol\b|toleran|slack"),
    (
        "stop_after_round",
        "stop punishing after a round",
        r"t_stop|t_end|last round|stop punish",
    ),
    ("spare_switch_rounds", "spare the rounds before a switch", r"switch"),
    ("proportional_punishment", "punish in proportion", r"slope|\brate\b|proportion"),
    ("absolute_threshold", "punish below a fixed threshold", r"threshold"),
    ("target_newcomers", "target newcomers", r"newcomer|arriv"),
    ("group_size_gate", "depend on group size", r"group size|n_big"),
    ("reward_cooperators", "reward high contributors", r"reward|bonus"),
]
WHY = [
    ("raises_contributions", "punishment raises contributions", r"rais|lift"),
    ("members_leave", "punishment drives members away", r"leave|lose|members"),
    ("punishment_costs_score", "punishment costs score", r"\bcost|loss"),
    ("conformity", "players conform to the group", r"conform|herd"),
    ("human_managers", "human managers did it", r"\bhuman"),
]
BASIS = [
    ("own_data_analysis", "an analysis of the data", r"analysis"),
    ("earlier_result", "a result of an earlier sweep", r"result|sweep"),
]
THEORY = r"(equilibrium|conformity|deterren\w*|Ostrom|Fehr|reciproc\w*)"
VARIANTS = ["", " in each round", " for valid members", " as the rule states"]
RELATIONS = ["new", "kept", "modified", "kept", "removed", "not_stated"]
EVIDENCE = ["analysis of the data", "result of the previous sweep", None]
REFS = ["exploration 1", "none", None]


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


def _first_line(regex, text):
    for line in (text or "").splitlines():
        if re.search(regex, line, re.I) and line.strip():
            return line.strip()
    return None


def _match(vocab, text):
    return [v for v in vocab if re.search(v[2], text or "", re.I)]


def _quote(line, salt):
    # one quote in eight is "paraphrased" so the quote check has work to do
    return line if _h(salt, line) % 8 else f"roughly: {line}"


def _vocab(kind):
    return {"why": WHY, "basis": BASIS}.get(kind, WHAT)


def open_rules(meta):
    doc = meta["doc"]
    mechs = []
    for name, label, rx in WHAT:
        line = _first_line(rx, doc)
        if line:
            var = VARIANTS[_h(meta["stem"], name) % len(VARIANTS)]
            mechs.append(
                {"description": label + var, "quote": _quote(line, meta["stem"])}
            )
    return {"mechanisms": mechs}


def open_notes(meta):
    doc, stem = meta["doc"], meta["stem"]
    mechs = []
    for name, label, rx in WHAT:
        line = _first_line(rx, doc)
        if not line:
            continue
        h = _h(stem, name)
        status = ["adopted"] * 3 + ["considered", "rejected"]
        src = re.search(r"pf-\d{8}-\d+(?:-v\d+)?", line)
        reasons = [
            {
                "description": wl,
                "quote": _quote(line, stem + wn),
                "stance": "against" if _h(stem, wn) % 4 == 0 else "for",
                "evidence": EVIDENCE[_h(stem, wn, "e") % len(EVIDENCE)],
                "evidence_ref": REFS[_h(stem, wn, "r") % len(REFS)],
            }
            for wn, wl, wrx in WHY
            if re.search(wrx, line, re.I)
        ]
        relation = RELATIONS[h % len(RELATIONS)] if meta.get("parent") else None
        st = status[h % len(status)]
        # a rejected idea is described negatively, as small models do; the idea
        # field carries the positive form
        desc = ("does not " if st == "rejected" else "") + label
        mechs.append(
            {
                "description": desc + VARIANTS[h % len(VARIANTS)],
                "quote": _quote(line, stem),
                "idea": label,
                "status": st,
                "tested": bool(h % 3),
                "relation_to_previous": relation or "not_stated",
                "source_version": src.group(0) if src else None,
                "reasons": reasons,
            }
        )
    theory = []
    for line in doc.splitlines():
        m = re.search(THEORY, line, re.I)
        if m and len(theory) < 2:
            theory.append(
                {"name": m.group(1).lower(), "kind": "concept", "quote": line.strip()}
            )
    findings = [
        {"description": "a finding about players", "quote": line.strip()}
        for line in doc.splitlines()
        if re.search(r"\d", line) and not _match(WHAT, line)
    ][:2]
    return {
        "mechanisms": mechs,
        "theory_mentions": theory,
        "unlinked_findings": findings,
    }


def propose(meta):
    vocab = _vocab(meta["kind"])
    texts = [t.splitlines()[0] for _, t in meta["items"]]
    counts = {v[0]: sum(bool(re.search(v[2], t, re.I)) for t in texts) for v in vocab}
    existing = set(meta.get("existing", []))
    if meta.get("round", 0) == 0:
        chosen = [v for v in vocab if counts[v[0]] >= 2]
        chosen = chosen[:-1] if len(chosen) > 1 else chosen  # leave work to refine
    else:
        chosen = [v for v in vocab if counts[v[0]] >= 1 and v[0] not in existing]
    codes = [
        {
            "name": name,
            "label": label,
            "definition": f"The manager rule or reason fits: {label}.",
            "fits": [label],
            "does_not_fit": ["something else"],
        }
        for name, label, _ in chosen
    ]
    return {"codes": codes}


def assign(meta):
    text = meta["text"].splitlines()[0]  # never the context line
    by_name = {v[0]: v for v in WHAT + WHY + BASIS}
    for name in meta["codes"]:
        v = by_name.get(name)
        if v and re.search(v[2], text, re.I):
            return {"code": name}
    return {"code": "none"}


def judge(meta):
    """Regex judge. Repeat r1 flips some answers (so repeats disagree and the
    tie-break runs); a flip to present has no evidence (so the re-ask runs)."""
    doc = meta["doc"]
    by_name = {v[0]: v for v in WHAT}
    out = {}
    for name in meta["codes"]:
        v = by_name.get(name)
        line = _first_line(v[2], doc) if v else None
        present, evidence = bool(line), line or ""
        flip = meta.get("repeat") == "r1" and not meta.get("reask")
        if flip and _h(meta.get("stem"), name) % 5 == 0:
            present, evidence = not present, ""
        out[name] = {
            "evidence": evidence,
            "present": present,
            "switchable_off": present and _h(name, evidence) % 3 == 0,
        }
    return out


def consolidate(meta):
    """Merge proportional_punishment into punish_below_group_mean; two families."""
    names = meta["codes"]
    merges = []
    pair = ["punish_below_group_mean", "proportional_punishment"]
    if all(n in names for n in pair):
        merges.append(
            {
                "codes": pair,
                "keep": pair[0],
                "label": "punish the shortfall",
                "definition": "Punishment grows with the shortfall below a reference.",
            }
        )
    gone = {"proportional_punishment"} if merges else set()
    fams = {}
    for n in names:
        if n not in gone:
            fams.setdefault(f"family_{_h(n) % 2}", []).append(n)
    families = [
        {"name": f, "definition": f"codes of {f}", "codes": cs}
        for f, cs in sorted(fams.items())
    ]
    return {"merges": merges, "families": families}


def respond(call):
    fn = {
        "open_rules": open_rules,
        "open_notes": open_notes,
        "propose": propose,
        "assign": assign,
        "judge": judge,
        "consolidate": consolidate,
    }[call.kind]
    return json.dumps(fn(call.meta))
