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
    "theory_mentions",
]


def _check_revisions(run, rule_rows):
    """Fail unless stage B ran with the latest what and why codebooks."""
    wrev, wpath, _ = latest_codebook(run, "what")
    yrev, ypath, _ = latest_codebook(run, "why")
    latest = {
        "what": (wrev, file_hash(wpath), wpath.name),
        "why": (yrev, file_hash(ypath), ypath.name),
    }
    ar = require(run, "apply-rules")
    an = require(run, "apply-notes")
    used = [("apply-rules", "what", ar["codebook"])]
    used += [("apply-notes", k, an["codebooks"][k]) for k in ("what", "why")]
    for step, kind, cb in used:
        rev, digest, name = latest[kind]
        if (cb["revision"], cb["sha256"]) != (rev, digest):
            raise SystemExit(
                f"{step} used {cb['file']} but the latest {kind} codebook is {name}; "
                f"rerun stage B before join"
            )
    mine = [r for r in rule_rows if r["coder"] == ar["coder"]]
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
    records = load_records(run, coded=True)
    z = _z_scores(versions)
    out = []
    for v in versions:
        stem, parent = v["stem"], v["parent"]
        rec = records.get(stem)
        by_code = defaultdict(list)
        if rec:
            for m in rec["mechanisms"]:
                if m.get("what_code"):
                    by_code[m["what_code"]].append(m)
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
                "event": _event(ir, ip if parent else False, bool(parent)),
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
                "theory_mentions": ";".join(theory),
            }
            out.append(row)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for row in out:
        w.writerow({k: _cell(row[k]) for k in COLUMNS})
    atomic_write_text(run.path("analysis", "version_codes.csv"), buf.getvalue())
    agreement = write_agreement(run, rule_rows)
    quality = _quality(run, versions, records, rule_rows, coder, in_rule, names)
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
            if m["status"] == "adopted" and m.get("what_code"):
                adopted[stem].add(m["what_code"])
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
        if r["status"] == "length":
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
