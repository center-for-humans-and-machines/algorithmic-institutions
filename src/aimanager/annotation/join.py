"""Joined tables: analysis/version_codes.csv and analysis/quality.json."""

import csv
import io
import statistics
from collections import Counter, defaultdict

from .common import (
    atomic_write_text,
    latest_codebook,
    log,
    read_json,
    read_jsonl,
    write_json,
)
from .corpus import load_versions
from .steps import load_records

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
    "in_rule",
    "in_parent_rule",
    "event",
    "notes_status",
    "notes_tested",
    "notes_source_versions",
    "why_codes",
    "theory_mentions",
]


def _coder(run, cfg, rows):
    coder = cfg.get("join", {}).get("coder", "auto")
    if coder != "auto":
        return coder
    meta = run.dir / "codes" / "apply_rules.meta.json"
    if meta.exists():
        return read_json(meta)["coder"]
    llm = Counter(r["coder"] for r in rows if r["coder"].startswith("llm:"))
    return llm.most_common(1)[0][0] if llm else None


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
    rev, path, codes = latest_codebook(run, "what")
    names = [c["name"] for c in codes]
    rule_rows = read_jsonl(run.dir / "codes" / "rule_codes.jsonl")
    coder = _coder(run, cfg, rule_rows)
    in_rule = {
        (r["version"], r["what_code"]): r["present"]
        for r in rule_rows
        if r["coder"] == coder
    }
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
        theory = sorted({t["name"] for t in rec["theory_mentions"]}) if rec else []
        for code in names:
            mechs = by_code.get(code, [])
            ir = in_rule.get((stem, code))
            ip = in_rule.get((parent, code)) if parent else False
            whys = {r["why_code"] for m in mechs for r in m["reasons"] if r["why_code"]}
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
                "in_rule": ir,
                "in_parent_rule": ip if parent else None,
                "event": _event(ir, ip, bool(parent)),
                "notes_status": (
                    ";".join(sorted({m["status"] for m in mechs})) if rec else None
                ),
                "notes_tested": any(m["tested"] for m in mechs) if rec else None,
                "notes_source_versions": ";".join(sorted(srcs)),
                "why_codes": ";".join(sorted(whys)),
                "theory_mentions": ";".join(theory),
            }
            out.append(row)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for row in out:
        w.writerow({k: _cell(row[k]) for k in COLUMNS})
    atomic_write_text(run.path("analysis", "version_codes.csv"), buf.getvalue())
    quality = _quality(run, versions, records, rule_rows, coder, in_rule, names)
    quality["codebooks"] = {"what": path.name, "coder": coder}
    write_json(run.path("analysis", "quality.json"), quality)
    log(f"join: {len(out)} rows ({len(versions)} versions x {len(names)} codes)")


def _rate(flags):
    flags = [f for f in flags if f is not None]
    return {
        "n": len(flags),
        "yes": sum(flags),
        "rate": round(sum(flags) / len(flags), 4) if flags else None,
    }


def _quality(run, versions, records, rule_rows, coder, in_rule, names):
    mechs = [m for r in records.values() for m in r["mechanisms"]]
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
    reasons = [r for m in mechs for r in m["reasons"]]
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
    explicit = [bool(r["theory_mentions"]) for r in rec_list]
    return {
        "versions": len(versions),
        "records": len(records),
        "what_codes": len(names),
        "coverage_what_on_notes_mechanisms": cov,
        "say_do_gap": {
            "notes_adopted_rule_absent": _rate(said_not_done),
            "rule_present_notes_not_adopted": _rate(done_not_said),
        },
        "records_with_theory_mentions": _rate(explicit),
        "quote_check": quotes,
        "calls": _call_stats(run),
        "agreement_overall": _agreement(run),
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


def _agreement(run):
    path = run.dir / "analysis" / "agreement.csv"
    if not path.exists():
        return None
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r["code"] == "__overall__"]
