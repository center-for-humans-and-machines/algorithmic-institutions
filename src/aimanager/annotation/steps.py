"""LLM steps 1a-5b of docs/plans/step-classification-pipeline.md."""

import csv
import io

from . import prompting as P
from .client import Call
from .cluster import assign, cluster, members
from .common import (
    atomic_write_text,
    latest_codebook,
    log,
    quote_in_source,
    read_json,
    read_jsonl,
    read_yaml,
    write_json,
    write_jsonl,
    write_yaml,
)
from .corpus import load_versions
from .schemas import FreeCodes, NotesRecord, judge_answers, judge_model


def _limit(xs, limit):
    return xs[:limit] if limit else xs


def _codebook_doc(kind, revision, codes, **extra):
    return {"kind": kind, "revision": revision, **extra, "codes": codes}


# --- 1a ------------------------------------------------------------------


def open_rules(run, cfg, client, limit=None):
    versions = {v["stem"]: v for v in load_versions(run)}
    split = read_json(run.dir / "corpus" / "split.json")
    sample = _limit([s for s in split["free_coding"] if versions[s]["rule"]], limit)
    instr = P.load("open_rules")
    calls = [
        Call(
            "open-rules",
            s,
            P.messages(instr, versions[s]["rule"]),
            FreeCodes,
            "open_rules",
            meta={"doc": versions[s]["rule"], "stem": s},
        )
        for s in sample
    ]
    rows, failed = [], []
    for res in client.map(calls):
        stem = res.call.item_id
        if not res.ok:
            failed.append({"version": stem, "error": res.error})
            continue
        for i, m in enumerate(res.value.mechanisms):
            rows.append(
                {
                    "id": f"fc:{stem}:{i}",
                    "version": stem,
                    "description": m.description,
                    "quote": m.quote,
                    "quote_ok": quote_in_source(m.quote, versions[stem]["rule"]),
                }
            )
    write_jsonl(run.path("open", "rule_free_codes.jsonl"), rows)
    write_json(run.path("open", "rule_free_codes.failed.json"), failed)
    log(f"open-rules: {len(rows)} free codes from {len(sample)} rules")
    return rows


# --- 1b ------------------------------------------------------------------


def open_notes(run, cfg, client, limit=None):
    versions = _limit([v for v in load_versions(run) if v["notes"]], limit)
    instr = P.load("open_notes")
    calls = [
        Call(
            "open-notes",
            v["stem"],
            P.messages(instr, v["notes"]),
            NotesRecord,
            "open_notes",
            meta={"doc": v["notes"], "stem": v["stem"]},
        )
        for v in versions
    ]
    notes = {v["stem"]: v["notes"] for v in versions}
    failed = []
    for res in client.map(calls):
        stem = res.call.item_id
        if not res.ok:
            failed.append({"version": stem, "error": res.error})
            continue
        write_json(run.path("records", f"{stem}.json"), _record(stem, res, notes[stem]))
    write_json(run.path("records_failed.json"), failed)
    log(f"open-notes: {len(versions) - len(failed)} records, {len(failed)} failed")


def _record(stem, res, src):
    rec = res.value.model_dump()
    for i, m in enumerate(rec["mechanisms"]):
        m["id"] = f"m:{stem}:{i}"
        m["quote_ok"] = quote_in_source(m["quote"], src)
        m["what_code"] = None
        for j, r in enumerate(m["reasons"]):
            r["id"] = f"r:{stem}:{i}:{j}"
            r["quote_ok"] = quote_in_source(r["quote"], src)
            r["why_code"] = None
    for key in ("theory_mentions", "unlinked_findings"):
        for x in rec[key]:
            x["quote_ok"] = quote_in_source(x["quote"], src)
    return {"version": stem, **rec}


def load_records(run, coded=False):
    d = run.dir / ("records_coded" if coded else "records")
    if not d.exists():
        return {}
    return {p.stem: read_json(p) for p in sorted(d.glob("*.json"))}


def _mechanism_items(records):
    return [
        (m["id"], m["description"]) for r in records.values() for m in r["mechanisms"]
    ]


def _reason_items(records):
    return [
        (r["id"], r["description"])
        for rec in records.values()
        for m in rec["mechanisms"]
        for r in m["reasons"]
    ]


# --- 2a ------------------------------------------------------------------


def what_codebook(run, cfg, client, limit=None):
    rows = read_jsonl(run.dir / "open" / "rule_free_codes.jsonl")
    if not rows:
        raise SystemExit("no free codes; run open-rules first")
    items = _limit([(r["id"], r["description"]) for r in rows], limit)
    codes, assignment, failed, hist = cluster(
        client, "what-codebook", "what", items, cfg["clustering"], "rules"
    )
    write_yaml(run.path("codebook", "what.draft.yml"), _codebook_doc("what", 0, codes))
    write_yaml(
        run.path("codebook", "what.draft.members.yml"),
        members(codes, assignment, failed),
    )
    write_json(run.path("codebook", "what.draft.history.json"), hist)
    _saturation(run, cfg, rows, assignment)
    log(f"what-codebook: {len(codes)} codes")


def _saturation(run, cfg, rows, assignment):
    """New what codes per batch of free-coded versions, in sample order."""
    split = read_json(run.dir / "corpus" / "split.json")
    order = {s: i for i, s in enumerate(split["free_coding"])}
    size = cfg.get("saturation", {}).get("batch_size", 6)
    seen, out = set(), []
    by_version = {}
    for r in rows:
        code = assignment.get(r["id"])
        if code:
            by_version.setdefault(r["version"], set()).add(code)
    stems = sorted(by_version, key=lambda s: order.get(s, 1e9))
    for b in range(0, len(stems), size):
        batch = set().union(*(by_version[s] for s in stems[b : b + size]))
        new = batch - seen
        seen |= batch
        out.append([b // size, len(stems[b : b + size]), len(new), len(seen)])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["batch", "n_versions", "new_codes", "cumulative_codes"])
    w.writerows(out)
    atomic_write_text(run.path("codebook", "what_saturation.csv"), buf.getvalue())


# --- 2b ------------------------------------------------------------------


def judge_rules(client, step, codes, versions):
    """One judge call per rule against a codebook; (rows, failed versions)."""
    names = [c["name"] for c in codes]
    model_cls = judge_model(names)
    instr = P.load("judge")
    shared = P.render_codebook(codes)
    calls = [
        Call(
            step,
            v["stem"],
            P.messages(instr, v["rule"], shared),
            model_cls,
            "judge",
            meta={"codes": names, "doc": v["rule"]},
        )
        for v in versions
    ]
    rules = {v["stem"]: v["rule"] for v in versions}
    rows, failed = [], []
    for res in client.map(calls):
        stem = res.call.item_id
        if not res.ok:
            failed.append({"version": stem, "error": res.error})
            continue
        for name, j in judge_answers(res.value, names).items():
            rows.append(
                {
                    "version": stem,
                    "what_code": name,
                    "present": j.present,
                    "quote": j.quote,
                    "quote_ok": (
                        quote_in_source(j.quote, rules[stem]) if j.present else None
                    ),
                }
            )
    return rows, failed


def coverage(run, cfg, client, limit=None):
    draft = read_yaml(run.dir / "codebook" / "what.draft.yml")["codes"]
    records = load_records(run)
    if not records:
        raise SystemExit("no notes records; run open-notes first")
    items = _limit(_mechanism_items(records), limit)
    to_draft, failed = assign(client, "coverage", draft, items, "what")
    unmatched = [(i, t) for i, t in items if to_draft[i] is None and i not in failed]
    log(f"coverage: {len(items) - len(unmatched)}/{len(items)} matched the draft")
    cands, to_cand, cfailed, hist = [], {}, [], []
    if unmatched:
        cands, to_cand, cfailed, hist = cluster(
            client,
            "coverage",
            "what",
            unmatched,
            cfg["clustering"],
            "notes",
            taken={c["name"] for c in draft},
        )
    judgements, jfailed = [], []
    if cands:
        versions = _limit([v for v in load_versions(run) if v["rule"]], limit)
        judgements, jfailed = judge_rules(client, "coverage", cands, versions)
    n_rules = {c["name"]: 0 for c in cands}
    for row in judgements:
        n_rules[row["what_code"]] += bool(row["present"])
    accepted = [
        dict(c, n_rules=n_rules[c["name"]]) for c in cands if n_rules[c["name"]]
    ]
    never = [dict(c, n_rules=0) for c in cands if not n_rules[c["name"]]]
    cmembers = members(cands, to_cand, cfailed)
    for c in never:
        c["members"] = cmembers[c["name"]]
    write_jsonl(
        run.path("coverage", "notes_assign.jsonl"),
        [
            {"id": i, "draft_code": to_draft[i], "candidate_code": to_cand.get(i)}
            for i, _ in items
        ],
    )
    write_yaml(run.path("coverage", "candidates.yml"), _codebook_doc("what", 0, cands))
    write_yaml(run.path("coverage", "candidates.members.yml"), cmembers)
    write_json(run.path("coverage", "candidates.history.json"), hist)
    write_jsonl(run.path("coverage", "candidate_judgements.jsonl"), judgements)
    write_yaml(
        run.path("codebook", "what.v1.yml"),
        _codebook_doc("what", 1, draft + accepted),
    )
    write_yaml(
        run.path("codebook", "never_implemented.yml"),
        {"kind": "what", "codes": never},
    )
    write_json(
        run.path("coverage", "summary.json"),
        {
            "notes_mechanisms": len(items),
            "matched_draft": len(items) - len(unmatched) - len(failed),
            "unmatched": len(unmatched),
            "candidates": len(cands),
            "accepted": [c["name"] for c in accepted],
            "never_implemented": [c["name"] for c in never],
            "assign_failed": len(failed) + len(cfailed),
            "judge_failed": jfailed,
        },
    )
    log(f"coverage: {len(accepted)} candidates added, {len(never)} never implemented")


# --- 3 -------------------------------------------------------------------


def why_codebook(run, cfg, client, limit=None):
    items = _limit(_reason_items(load_records(run)), limit)
    if not items:
        raise SystemExit("no reasons in the notes records; run open-notes first")
    codes, assignment, failed, hist = cluster(
        client, "why-codebook", "why", items, cfg["clustering"], "notes"
    )
    write_yaml(run.path("codebook", "why.v1.yml"), _codebook_doc("why", 1, codes))
    write_yaml(
        run.path("codebook", "why.v1.members.yml"), members(codes, assignment, failed)
    )
    write_json(run.path("codebook", "why.v1.history.json"), hist)
    log(f"why-codebook: {len(codes)} codes")


# --- 5a ------------------------------------------------------------------


def llm_coder(client):
    return f"llm:{client.model}"


def apply_rules(run, cfg, client, limit=None):
    rev, path, codes = latest_codebook(run, "what")
    versions = _limit([v for v in load_versions(run) if v["rule"]], limit)
    rows, failed = judge_rules(client, "apply-rules", codes, versions)
    coder = llm_coder(client)
    for r in rows:
        r["coder"] = coder
        r["codebook_revision"] = rev
    out = run.path("codes", "rule_codes.jsonl")
    kept = [r for r in read_jsonl(out) if r.get("coder") != coder]
    write_jsonl(out, kept + rows)
    write_json(
        run.path("codes", "apply_rules.meta.json"),
        {"codebook": path.name, "revision": rev, "coder": coder, "failed": failed},
    )
    log(f"apply-rules: {path.name}, {len(versions)} rules, {len(failed)} failed")


# --- 5b ------------------------------------------------------------------


def apply_notes(run, cfg, client, limit=None):
    wrev, wpath, wcodes = latest_codebook(run, "what")
    yrev, ypath, ycodes = latest_codebook(run, "why")
    records = load_records(run)
    if limit:
        records = dict(list(records.items())[:limit])
    what, wfailed = assign(
        client, "apply-notes", wcodes, _mechanism_items(records), "what"
    )
    why, yfailed = assign(client, "apply-notes", ycodes, _reason_items(records), "why")
    for stem, rec in records.items():
        for m in rec["mechanisms"]:
            m["what_code"] = what.get(m["id"])
            for r in m["reasons"]:
                r["why_code"] = why.get(r["id"])
        rec["codebooks"] = {"what": wpath.name, "why": ypath.name}
        write_json(run.path("records_coded", f"{stem}.json"), rec)
    write_json(
        run.path("codes", "apply_notes.meta.json"),
        {
            "what_codebook": wpath.name,
            "why_codebook": ypath.name,
            "failed_mechanisms": wfailed,
            "failed_reasons": yfailed,
        },
    )
    log(f"apply-notes: {len(records)} records with {wpath.name}, {ypath.name}")


STAGE_A = [
    ("open-rules", open_rules),
    ("open-notes", open_notes),
    ("what-codebook", what_codebook),
    ("coverage", coverage),
    ("why-codebook", why_codebook),
]
STAGE_B = [("apply-rules", apply_rules), ("apply-notes", apply_notes)]
STEPS = dict(STAGE_A + STAGE_B)
