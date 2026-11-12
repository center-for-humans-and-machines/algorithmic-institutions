"""Joined tables: analysis/version_codes.csv, quality.json and agreement.csv.

join refuses to run unless apply-rules and apply-notes used the latest codebook
revisions. Never-implemented codes get rows too (in_rule false), so ideas the
notes discuss but no rule implements stay in the analysis.

Several notes mechanisms of one version can share a code. Their status is
reduced by precedence adopted > rejected > considered, their relation to the
previous version by new > modified > removed > kept > not_stated; the full
sorted lists are in the *_all columns. `no_record` means the version has no
notes record, `no_mechanism` that the record has no mechanism with the code.
"""

import csv
import io
import statistics
from collections import Counter, defaultdict

from .common import (
    read_yaml,
    atomic_write_text,
    file_hash,
    latest_codebook,
    log,
    read_jsonl,
    require,
    write_json,
)
from .corpus import load_versions
from .expert import write_agreement
from .steps import load_records, notes_what_codes

STATUS_ORDER = ["adopted", "rejected", "considered"]
RELATION_ORDER = ["new", "modified", "removed", "kept", "not_stated"]

COLUMNS = [
    "campaign",
    "lineage",
    "version",
    "generation",
    "parent",
    "score",
    "score_se",
    "score_status",
    "score_z",
    "what_code",
    "code_source",
    "in_rule",
    "switchable_off",
    "in_parent_rule",
    "event",
    "event_unreliable_code",
    "notes_status",
    "notes_status_all",
    "notes_relation",
    "notes_relation_all",
    "notes_tested",
    "notes_source_versions",
    "why_codes",
    "why_codes_for",
    "why_codes_against",
    "reason_evidence",
    "reason_evidence_refs",
    "basis_codes",
    "basis_families",
    "theory_mentions",
]


def _check_revisions(run, rule_rows):
    """Fail unless stage B ran with the latest what, why and basis codebooks."""
    latest = {}
    for kind in ("what", "why", "basis"):
        rev, path, _ = latest_codebook(run, kind)
        latest[kind] = (rev, file_hash(path), path.name)
    wpath = run.dir / "codebook" / latest["what"][2]
    ar = require(run, "apply-rules")
    an = require(run, "apply-notes")
    used = [("apply-rules", "what", ar["codebook"])]
    used += [("apply-notes", k, an["codebooks"][k]) for k in ("what", "why", "basis")]
    never = an["codebooks"].get("never_implemented") or {}
    if "revision" in never:  # runs before revisions of the list carry only a name
        rev, path, _ = latest_codebook(run, "never_implemented")
        latest["never_implemented"] = (rev, file_hash(path), path.name)
        used.append(("apply-notes", "never_implemented", never))
    for step, kind, cb in used:
        rev, digest, name = latest[kind]
        if (cb["revision"], cb["sha256"]) != (rev, digest):
            raise SystemExit(
                f"{step} used {cb['file']} but the latest {kind} codebook is {name}; "
                f"rerun stage B before join"
            )
    mine = [r for r in rule_rows if r["coder"] == ar["coder"]]
    wrev = latest["what"][0]
    if len(mine) != ar["rows"] or any(r["codebook_revision"] != wrev for r in mine):
        raise SystemExit(
            "codes/rule_codes.jsonl does not hold the rows apply-rules wrote; "
            "rerun apply-rules"
        )
    return ar["coder"], wpath


def _reduce(values, order):
    present = sorted(set(values), key=order.index)
    return (present[0] if present else None), ";".join(present)


def _z_scores(versions):
    by_campaign = defaultdict(list)
    for v in versions:
        if v["score_status"] == "ok":
            by_campaign[v["campaign"]].append(v["score"])
    z = {}
    for v in versions:
        xs = by_campaign[v["campaign"]]
        if v["score_status"] != "ok" or len(xs) < 2:
            continue
        sd = statistics.stdev(xs)
        z[v["stem"]] = (v["score"] - statistics.mean(xs)) / sd if sd else 0.0
    return z


def _event(in_rule, in_parent, has_parent):
    if in_rule is None or (has_parent and in_parent is None):
        return ""
    if in_rule:
        return "kept" if in_parent else "introduced"
    return "dropped" if in_parent else "absent"


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    return round(v, 4) if isinstance(v, float) else v


def join(run, cfg, limit=None):
    versions = load_versions(run)
    rule_rows = read_jsonl(run.dir / "codes" / "rule_codes.jsonl")
    llm_coder, path = _check_revisions(run, rule_rows)
    _, _, codes, never = notes_what_codes(run)
    coder = cfg.get("join", {}).get("coder", "auto")
    coder = llm_coder if coder == "auto" else coder
    names = [c["name"] for c in codes]
    never_names = [c["name"] for c in never]
    rows_of = {
        (r["version"], r["what_code"]): r for r in rule_rows if r["coder"] == coder
    }
    in_rule = {k: r["present"] for k, r in rows_of.items()}
    unreliable = set(_unreliable_codes(run, cfg))
    _, bpath, bcodes = latest_codebook(run, "basis")
    basis_family = {c["name"]: c.get("family") for c in bcodes}
    records = load_records(run, coded=True)
    z = _z_scores(versions)
    out = []
    for v in versions:
        stem, parent = v["stem"], v["parent"]
        rec = records.get(stem)
        by_code = defaultdict(list)
        if rec:
            for m in rec["mechanisms"]:
                for code in _codes_of(m):
                    by_code[code].append(m)
        theory = sorted(
            {t["name"] for t in (rec or {}).get("theory_mentions", []) if _named(t)}
        )
        for code in names + never_names:
            never_impl = code in never_names
            mechs = by_code.get(code, [])
            if never_impl:
                ir, ip, sw = False, (False if parent else None), None
            else:
                ir = in_rule.get((stem, code))
                ip = in_rule.get((parent, code)) if parent else None
                sw = (rows_of.get((stem, code)) or {}).get("switchable_off")
            reasons = [r for m in mechs for r in m["reasons"]]
            status, status_all = _reduce([m["status"] for m in mechs], STATUS_ORDER)
            relation, relation_all = _reduce(
                [m.get("relation_to_previous", "not_stated") for m in mechs],
                RELATION_ORDER,
            )
            empty = "no_record" if rec is None else "no_mechanism"
            event = _event(ir, ip if parent else False, bool(parent))
            srcs = {m["source_version"] for m in mechs if m.get("source_version")}
            row = {
                "campaign": v["campaign"],
                "lineage": v["lineage"],
                "version": stem,
                "generation": v["version"],
                "parent": parent,
                "score": v["score"],
                "score_se": v["score_se"],
                "score_status": v["score_status"],
                "score_z": z.get(stem),
                "what_code": code,
                "code_source": "never_implemented" if never_impl else "codebook",
                "in_rule": ir,
                "switchable_off": sw,
                "in_parent_rule": ip,
                "event": event,
                "event_unreliable_code": (
                    (code in unreliable)
                    if event in ("introduced", "kept", "dropped")
                    else None
                ),
                "notes_status": status or empty,
                "notes_status_all": status_all,
                "notes_relation": relation or empty,
                "notes_relation_all": relation_all,
                "notes_tested": any(m["tested"] for m in mechs) if mechs else None,
                "notes_source_versions": ";".join(sorted(srcs)),
                "why_codes": _codes(reasons),
                "why_codes_for": _codes(reasons, "for"),
                "why_codes_against": _codes(reasons, "against"),
                "reason_evidence": " | ".join(
                    sorted({r["evidence"] for r in reasons if r.get("evidence")})
                ),
                "reason_evidence_refs": " | ".join(
                    sorted(
                        {r["evidence_ref"] for r in reasons if r.get("evidence_ref")}
                    )
                ),
                "basis_codes": ";".join(
                    sorted({r["basis_code"] for r in reasons if r.get("basis_code")})
                ),
                "basis_families": ";".join(
                    sorted(
                        {
                            basis_family.get(r["basis_code"]) or ""
                            for r in reasons
                            if r.get("basis_code")
                        }
                    )
                ),
                "theory_mentions": ";".join(theory),
            }
            out.append(row)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for row in out:
        w.writerow({k: _cell(row[k]) for k in COLUMNS})
    atomic_write_text(run.path("analysis", "version_codes.csv"), buf.getvalue())
    _write_theory(run, versions, records)
    agreement = write_agreement(run, rule_rows)
    quality = _quality(run, versions, records, rule_rows, coder, in_rule, names)
    quality.update(_rule_code_stats(cfg, rule_rows, coder, names))
    quality["judge_reliability"] = _reliability_summary(run) or {}
    quality["judge_reliability"]["unreliable_codes"] = {
        "min_agreement": cfg.get("judge", {}).get("min_agreement", 0.8),
        "codes": sorted(unreliable),
    }
    quality["judge_present_share_by_repeat"] = _present_by_repeat(run)
    quality["family_sizes"] = _family_sizes(run)
    quality["unassigned_by_codebook"] = _unassigned_shares(run)
    quality["agreement_overall"] = [
        r for r in agreement if r["code"] == "__overall__"
    ] or None
    quality["codebooks"] = {"what": path.name, "coder": coder}
    quality["never_implemented_codes"] = len(never_names)
    write_json(run.path("analysis", "quality.json"), quality)
    log(
        f"join: {len(out)} rows ({len(versions)} versions x {len(names)} codes "
        f"+ {len(never_names)} never implemented)"
    )


def _write_theory(run, versions, records):
    """analysis/theory_mentions.csv: per version the named mentions (name found
    in the notes), their kinds, and mentions whose name the notes lack."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "campaign",
            "lineage",
            "version",
            "generation",
            "mentions",
            "kinds",
            "mentions_not_in_notes",
        ]
    )
    for v in versions:
        ms = (records.get(v["stem"]) or {}).get("theory_mentions", [])
        ok = [t for t in ms if _named(t)]
        w.writerow(
            [
                v["campaign"],
                v["lineage"],
                v["stem"],
                v["version"],
                ";".join(t["name"] for t in ok),
                ";".join(t["kind"] for t in ok),
                ";".join(t["name"] for t in ms if not _named(t)),
            ]
        )
    atomic_write_text(run.path("analysis", "theory_mentions.csv"), buf.getvalue())


def _codes_of(m):
    """All what codes of a notes mechanism (best first); older records have one."""
    if m.get("what_codes") is not None:
        return list(m["what_codes"])
    return [m["what_code"]] if m.get("what_code") else []


def _rule_code_stats(cfg, rule_rows, coder, names):
    """Prevalence of every code over the judged rules, codes above
    coverage.max_prevalence flagged, codes per rule, judge flags."""
    mine = [r for r in rule_rows if r["coder"] == coder]
    rules = sorted({r["version"] for r in mine})
    n = max(len(rules), 1)
    prev = {c: 0 for c in names}
    per_rule = Counter()
    for r in mine:
        if r["present"]:
            prev[r["what_code"]] = prev.get(r["what_code"], 0) + 1
            per_rule[r["version"]] += 1
    share = {c: round(k / n, 4) for c, k in prev.items()}
    max_prev = cfg.get("coverage", {}).get("max_prevalence", 0.9)
    counts = sorted(per_rule.get(s, 0) for s in rules)
    q = statistics.quantiles(counts, n=4) if len(counts) > 1 else counts * 3
    return {
        "code_prevalence": dict(sorted(share.items(), key=lambda kv: -kv[1])),
        "codes_above_max_prevalence": {
            "max_prevalence": max_prev,
            "codes": [c for c, v in share.items() if v > max_prev],
        },
        "codes_never_present": [c for c, v in share.items() if not v],
        "codes_per_rule": {
            "rules": len(rules),
            "min": counts[0] if counts else None,
            "q1": q[0] if q else None,
            "median": statistics.median(counts) if counts else None,
            "q3": q[2] if q else None,
            "max": counts[-1] if counts else None,
        },
        "judge_flags": {
            "tiebreaks": sum(bool(r.get("tiebreak")) for r in mine),
            "unsupported": sum(bool(r.get("unsupported")) for r in mine),
        },
    }


def _reliability_rows(run):
    path = run.dir / "analysis" / "judge_reliability.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _unreliable_codes(run, cfg):
    """Codes whose judge test-retest agreement is below judge.min_agreement."""
    floor = cfg.get("judge", {}).get("min_agreement", 0.8)
    return [
        r["code"]
        for r in _reliability_rows(run)
        if r["code"] != "__overall__"
        and r["agreement"] not in ("", None)
        and float(r["agreement"]) < floor
    ]


def _present_by_repeat(run):
    """Per code the share of `present` answers in each judge repeat (and the
    tie-break), so one-sided splits between repeats are visible."""
    rows = read_jsonl(run.dir / "codes" / "judge_repeats.jsonl")
    tally = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        t = tally[r["what_code"]][str(r["repeat"])]
        t[0] += bool(r["present"])
        t[1] += 1
    return {
        code: {rep: round(y / n, 4) for rep, (y, n) in sorted(reps.items()) if n}
        for code, reps in sorted(tally.items())
    }


def _unassigned_shares(run):
    """Share of items no code covers, per codebook: v1 from stage A (members
    files) and the revision stage B used (apply-notes manifest)."""
    out = {}
    for kind in ("what", "why", "basis"):
        entry = {}
        mpath = run.dir / "codebook" / f"{kind}.v1.members.yml"
        if mpath.exists():
            mem = read_yaml(mpath) or {}
            total = sum(len(v) for v in mem.values())
            n = len(mem.get("none", []))
            entry["v1"] = {
                "n": n,
                "of": total,
                "share": round(n / total, 4) if total else None,
            }
        out[kind] = entry
    stage_b = require(run, "apply-notes").get("unassigned") or {}
    for kind, d in stage_b.items():
        out.setdefault(kind, {})[f"v{d['revision']}_stage_b"] = {
            k: d[k] for k in ("n", "of", "share")
        }
    return out


def _family_sizes(run):
    """Codes per family in the latest what, why and basis codebooks."""
    out = {}
    for kind in ("what", "why", "basis"):
        _, _, codes = latest_codebook(run, kind)
        out[kind] = dict(Counter(c.get("family") or "none" for c in codes))
    return out


def _reliability_summary(run):
    """Overall test-retest agreement of the rule judge and the five least
    reliable codes, from analysis/judge_reliability.csv (apply-rules)."""
    path = run.dir / "analysis" / "judge_reliability.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))

    def num(x):
        return float(x) if x not in ("", None) else None

    overall = next((r for r in rows if r["code"] == "__overall__"), None)
    codes = [r for r in rows if r["code"] != "__overall__"]
    codes.sort(key=lambda r: (num(r["agreement"]) or 0, num(r["kappa"]) or 0))
    keep = ("code", "n", "agreement", "kappa", "a_only", "b_only")
    return {
        "overall": overall and {k: overall[k] for k in keep},
        "least_reliable": [{k: r[k] for k in keep} for r in codes[:5]],
    }


def _named(mention):
    """A theory mention counts only if the notes contain its name (older records
    without the flag count)."""
    return mention.get("name_ok", True)


def _codes(reasons, stance=None):
    return ";".join(
        sorted(
            {
                r["why_code"]
                for r in reasons
                if r.get("why_code") and (stance is None or r["stance"] == stance)
            }
        )
    )


def _rate(flags):
    flags = [f for f in flags if f is not None]
    return {
        "n": len(flags),
        "yes": sum(flags),
        "rate": round(sum(flags) / len(flags), 4) if flags else None,
    }


def _quality(run, versions, records, rule_rows, coder, in_rule, names):
    mechs = [m for r in records.values() for m in r["mechanisms"]]
    reasons = [r for m in mechs for r in m["reasons"]]
    n_codes = Counter(
        len(_codes_of(m)) for m in mechs if m.get("what_code_status") != "failed"
    )
    tot = sum(n_codes.values()) or 1
    statuses = {
        "mechanisms": dict(Counter(m.get("what_code_status") for m in mechs)),
        "reasons": dict(Counter(r.get("why_code_status") for r in reasons)),
    }
    cov = {"all": _rate([m.get("what_code") is not None for m in mechs])}
    for status in ("adopted", "considered", "rejected"):
        cov[status] = _rate(
            [m.get("what_code") is not None for m in mechs if m["status"] == status]
        )
    # say-do gap
    adopted = defaultdict(set)
    for stem, rec in records.items():
        for m in rec["mechanisms"]:
            if m["status"] == "adopted":
                adopted[stem].update(_codes_of(m))
    said_not_done = [
        not in_rule[(s, c)]
        for s, cs in adopted.items()
        for c in cs
        if (s, c) in in_rule
    ]
    done_not_said = [
        c not in adopted.get(s, set())
        for (s, c), present in in_rule.items()
        if present and s in records
    ]
    rec_list = list(records.values())
    free = read_jsonl(run.dir / "open" / "rule_free_codes.jsonl")
    quotes = {
        "rule_free_codes": _rate([r["quote_ok"] for r in free]),
        "notes_mechanisms": _rate([m["quote_ok"] for m in mechs]),
        "notes_reasons": _rate([r["quote_ok"] for r in reasons]),
        "theory_mentions": _rate(
            [t["quote_ok"] for r in rec_list for t in r["theory_mentions"]]
        ),
        "unlinked_findings": _rate(
            [t["quote_ok"] for r in rec_list for t in r["unlinked_findings"]]
        ),
        "rule_codes_judge": _rate(
            [r["quote_ok"] for r in rule_rows if r["coder"] == coder]
        ),
        "rule_codes_raters": _rate(
            [r["quote_ok"] for r in rule_rows if r["coder"].startswith("rater:")]
        ),
    }
    explicit = [any(map(_named, r["theory_mentions"])) for r in rec_list]
    return {
        "versions": len(versions),
        "records": len(records),
        "what_codes": len(names),
        "coverage_what_on_notes_mechanisms": cov,
        "assignment_status": statuses,
        "what_codes_per_idea": {
            str(k): {"n": n_codes.get(k, 0), "share": round(n_codes.get(k, 0) / tot, 4)}
            for k in range(4)
        },
        "say_do_gap": {
            "notes_adopted_rule_absent": _rate(said_not_done),
            "rule_present_notes_not_adopted": _rate(done_not_said),
        },
        "records_with_theory_mentions": _rate(explicit),
        "theory_mentions_named_in_notes": _rate(
            [_named(t) for r in rec_list for t in r["theory_mentions"]]
        ),
        "quote_check": quotes,
        "calls": _call_stats(run),
    }


def _call_stats(run):
    rows = read_jsonl(run.dir / "logs" / "calls.jsonl")
    steps = defaultdict(lambda: Counter())
    last = {}
    for r in rows:
        s = steps[r["step"]]
        s["log_rows"] += 1
        if r["cache_hit"]:
            s["cache_hits"] += 1
        else:
            s["calls"] += 1
            s["prompt_tokens"] += r["prompt_tokens"] or 0
            s["completion_tokens"] += r["completion_tokens"] or 0
            s["latency_s"] += r["latency_s"] or 0
        if r["status"] == "length" and not r["cache_hit"]:
            # a replay from the cache is not a new length finish
            s["length_finishes"] += 1
        last[(r["step"], r["kind"], r["item"])] = r["status"]
    for (step, _, _), status in last.items():
        steps[step]["items"] += 1
        if status != "ok":
            steps[step]["failed_items"] += 1
            steps[step][f"failed_{status}"] += 1
    out = {k: _round(v) for k, v in sorted(steps.items())}
    total = Counter()
    for v in steps.values():
        total.update(v)
    out["total"] = _round(total)
    return out


def _round(counter):
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in counter.items()}
