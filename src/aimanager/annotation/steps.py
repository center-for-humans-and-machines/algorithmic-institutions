"""LLM steps 1a-5b of docs/plans/step-classification-pipeline.md.

Every step checks the manifests of the steps it reads (common.require), checks
its failed calls against the failure budget before writing anything, and
writes its own manifest last.
"""

import csv
import io

from . import prompting as P
from .client import Call
from .cluster import assign, cluster, members
from .common import (
    StepFailed,
    atomic_write_text,
    check_failures,
    file_hash,
    latest_codebook,
    log,
    quote_in_source,
    read_json,
    read_jsonl,
    read_yaml,
    require,
    write_json,
    write_jsonl,
    write_manifest,
    write_yaml,
)
from .corpus import load_versions
from .schemas import FreeCodes, NotesRecord, judge_answers, judge_model


def _limit(xs, limit):
    return xs[:limit] if limit else xs


def _codebook_doc(kind, revision, codes, **extra):
    return {"kind": kind, "revision": revision, **extra, "codes": codes}


def _corpus_files(run):
    return [run.dir / "corpus" / "versions.jsonl", run.dir / "corpus" / "split.json"]


def _codebook_info(rev, path):
    return {"file": path.name, "revision": rev, "sha256": file_hash(path)}


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
            P.messages(instr, versions[s]["rule"], context=versions[s]["setting"]),
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
            failed.append((stem, res.error))
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
    check_failures(cfg, "open-rules", "open_rules", failed, len(sample))
    out = run.path("open", "rule_free_codes.jsonl")
    write_jsonl(out, rows)
    write_manifest(
        run,
        "open-rules",
        _corpus_files(run),
        [out],
        expected=len(sample),
        written=len(sample) - len(failed),
        failed=[dict(version=s, error=e) for s, e in failed],
        model=client.model,
        limit=limit,
    )
    log(f"open-rules: {len(rows)} free codes from {len(sample) - len(failed)} rules")
    return rows


# --- 1b ------------------------------------------------------------------


def notes_context(version):
    return f"{version['setting']}\n\n{P.lineage_position(version)}"


def open_notes(run, cfg, client, limit=None):
    versions = _limit([v for v in load_versions(run) if v["notes"]], limit)
    instr = P.load("open_notes")
    calls = [
        Call(
            "open-notes",
            v["stem"],
            P.messages(instr, v["notes"], context=notes_context(v)),
            NotesRecord,
            "open_notes",
            meta={"doc": v["notes"], "stem": v["stem"], "parent": v["parent"]},
        )
        for v in versions
    ]
    notes = {v["stem"]: v["notes"] for v in versions}
    parents = {v["stem"]: v["parent"] for v in versions}
    records, failed = {}, []
    for res in client.map(calls):
        stem = res.call.item_id
        if res.ok:
            records[stem] = _record(stem, res, notes[stem], first=not parents[stem])
        else:
            failed.append((stem, res.error))
    check_failures(cfg, "open-notes", "open_notes", failed, len(versions))
    forced = sum(r.pop("_relation_forced") for r in records.values())
    if forced:
        log(f"  open-notes: {forced} first-version relations set to not_stated")
    outs = []
    for stem, rec in records.items():
        outs.append(run.path("records", f"{stem}.json"))
        write_json(outs[-1], rec)
    _remove_stale(run.dir / "records", outs)
    write_manifest(
        run,
        "open-notes",
        _corpus_files(run),
        outs,
        expected=len(versions),
        written=len(records),
        failed=[dict(version=s, error=e) for s, e in failed],
        model=client.model,
        settings={v["campaign"]: v["setting"] for v in versions},
        first_version_relations_forced=forced,
        limit=limit,
    )
    log(f"open-notes: {len(records)} records, {len(failed)} failed")


def _remove_stale(directory, keep):
    """Delete *.json in a directory that the current run of a step did not write."""
    keep = {p.name for p in keep}
    for p in directory.glob("*.json"):
        if p.name not in keep:
            p.unlink()


def _record(stem, res, src, first=False):
    """Record with ids and quote checks. A first version has no previous version,
    so every relation is not_stated whatever the model said (counted)."""
    rec = res.value.model_dump()
    forced = 0
    for i, m in enumerate(rec["mechanisms"]):
        m["id"] = f"m:{stem}:{i}"
        if first and m["relation_to_previous"] != "not_stated":
            m["relation_to_previous"] = "not_stated"
            forced += 1
        m["quote_ok"] = quote_in_source(m["quote"], src)
        m["what_code"] = None
        for j, r in enumerate(m["reasons"]):
            r["id"] = f"r:{stem}:{i}:{j}"
            r["quote_ok"] = quote_in_source(r["quote"], src)
            r["why_code"] = None
    for key in ("theory_mentions", "unlinked_findings"):
        for x in rec[key]:
            x["quote_ok"] = quote_in_source(x["quote"], src)
    return {"version": stem, **rec, "_relation_forced": forced}


def load_records(run, coded=False):
    """{stem: record} exactly as listed in the producing step's manifest."""
    manifest = require(run, "apply-notes" if coded else "open-notes")
    prefix = "records_coded/" if coded else "records/"
    return {
        rel[len(prefix) : -len(".json")]: read_json(run.dir / rel)
        for rel in sorted(manifest["outputs"])
        if rel.startswith(prefix)
    }


def _record_files(run, coded=False):
    d = run.dir / ("records_coded" if coded else "records")
    return sorted(d.glob("*.json"))


def _mechanism_items(records):
    return [
        (m["id"], m["description"]) for r in records.values() for m in r["mechanisms"]
    ]


def _reason_items(records):
    """Reason text plus a labelled context line: its mechanism and stance."""
    return [
        (
            r["id"],
            f"{r['description']}\n(context, not to be coded: given {r['stance']} "
            f"the mechanism \"{m['description']}\")",
        )
        for rec in records.values()
        for m in rec["mechanisms"]
        for r in m["reasons"]
    ]


# --- 2a ------------------------------------------------------------------


def what_codebook(run, cfg, client, limit=None):
    require(run, "open-rules")
    src = run.dir / "open" / "rule_free_codes.jsonl"
    rows = read_jsonl(src)
    if not rows:
        raise StepFailed("what-codebook: no free codes; rerun open-rules")
    items = _limit([(r["id"], r["description"]) for r in rows], limit)
    codes, assignment, failed, hist = cluster(
        client, cfg, "what-codebook", "what", items, "rules"
    )
    outs = [
        run.path("codebook", "what.draft.yml"),
        run.path("codebook", "what.draft.members.yml"),
        run.path("codebook", "what.draft.history.json"),
        run.path("codebook", "what_saturation.csv"),
    ]
    write_yaml(outs[0], _codebook_doc("what", 0, codes))
    write_yaml(outs[1], members(codes, assignment, failed))
    write_json(outs[2], hist)
    _saturation(run, cfg, rows, assignment, outs[3])
    write_manifest(
        run,
        "what-codebook",
        [src, run.dir / "corpus" / "split.json"],
        outs,
        expected=len(items),
        written=len(assignment),
        failed=[dict(item=i, error=e) for i, e in failed],
        codes=len(codes),
        model=client.model,
        limit=limit,
    )
    log(f"what-codebook: {len(codes)} codes")


def _saturation(run, cfg, rows, assignment, path):
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
    atomic_write_text(path, buf.getvalue())


# --- 2b ------------------------------------------------------------------


def judge_rules(client, cfg, step, codes, versions):
    """One judge call per rule against a codebook; (rows, failed [(stem, err)]).

    Raises StepFailed beyond the judge failure budget (default 0).
    """
    names = [c["name"] for c in codes]
    model_cls = judge_model(names)
    instr = P.load("judge")
    shared = P.render_codebook(codes)
    calls = [
        Call(
            step,
            v["stem"],
            P.messages(instr, v["rule"], shared, context=v["setting"]),
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
            failed.append((stem, res.error))
            continue
        for name, j in judge_answers(res.value, names).items():
            rows.append(
                {
                    "version": stem,
                    "what_code": name,
                    "present": j.present,
                    "switchable_off": j.switchable_off if j.present else None,
                    "quote": j.quote,
                    "quote_ok": (
                        quote_in_source(j.quote, rules[stem]) if j.present else None
                    ),
                }
            )
    check_failures(cfg, step, "judge", failed, len(versions))
    return rows, failed


def coverage(run, cfg, client, limit=None):
    require(run, "what-codebook")
    draft_path = run.dir / "codebook" / "what.draft.yml"
    draft = read_yaml(draft_path)["codes"]
    records = load_records(run)
    if not records:
        raise StepFailed("coverage: no notes records; rerun open-notes")
    items = _limit(_mechanism_items(records), limit)
    to_draft, failed = assign(client, cfg, "coverage", draft, items, "what")
    unmatched = [(i, t) for i, t in items if i in to_draft and not to_draft[i]]
    n_matched = sum(1 for c in to_draft.values() if c)
    log(f"coverage: {n_matched}/{len(items)} matched the draft, {len(failed)} failed")
    cands, to_cand, cfailed, hist = [], {}, [], []
    if unmatched:
        cands, to_cand, cfailed, hist = cluster(
            client,
            cfg,
            "coverage",
            "what",
            unmatched,
            "notes",
            taken={c["name"] for c in draft},
        )
    # candidates are judged on the rules outside the expert check only, so the
    # held-out rules stay unseen by every codebook decision
    held_out = set(read_json(run.dir / "corpus" / "split.json")["expert_check"])
    versions = _limit(
        [v for v in load_versions(run) if v["rule"] and v["stem"] not in held_out],
        limit,
    )
    judgements, jfailed = [], []
    if cands:
        judgements, jfailed = judge_rules(client, cfg, "coverage", cands, versions)
    n_rules = {c["name"]: 0 for c in cands}
    for row in judgements:
        n_rules[row["what_code"]] += bool(row["present"])
    undecided = [c["name"] for c in cands if not n_rules[c["name"]] and jfailed]
    if undecided:
        raise StepFailed(
            f"coverage: {len(jfailed)} judge calls failed, so candidates "
            f"{undecided} cannot be rejected as never implemented. Nothing written"
        )
    accepted = [c for c in cands if n_rules[c["name"]]]
    never = [c for c in cands if not n_rules[c["name"]]]
    cmembers = members(cands, to_cand, cfailed)
    outs = {
        "assign": run.path("coverage", "notes_assign.jsonl"),
        "cands": run.path("coverage", "candidates.yml"),
        "cmembers": run.path("coverage", "candidates.members.yml"),
        "hist": run.path("coverage", "candidates.history.json"),
        "judgements": run.path("coverage", "candidate_judgements.jsonl"),
        "v1": run.path("codebook", "what.v1.yml"),
        "never": run.path("codebook", "never_implemented.yml"),
        "never_members": run.path("codebook", "never_implemented.members.yml"),
        "summary": run.path("coverage", "summary.json"),
    }
    fail_ids = {i for i, _ in failed}
    write_jsonl(
        outs["assign"],
        [
            {
                "id": i,
                "draft_code": to_draft.get(i),
                "candidate_code": to_cand.get(i),
                "failed": i in fail_ids,
            }
            for i, _ in items
        ],
    )
    write_yaml(outs["cands"], _codebook_doc("what", 0, cands))
    write_yaml(outs["cmembers"], cmembers)
    write_json(outs["hist"], hist)
    write_jsonl(outs["judgements"], judgements)
    write_yaml(outs["v1"], _codebook_doc("what", 1, draft + accepted))
    write_yaml(outs["never"], _codebook_doc("never_implemented", 1, never))
    write_yaml(outs["never_members"], {c["name"]: cmembers[c["name"]] for c in never})
    summary = {
        "notes_mechanisms": len(items),
        "matched_draft": n_matched,
        "unmatched": len(unmatched),
        "assign_failed": len(failed) + len(cfailed),
        "candidates": len(cands),
        "judged_on": (
            f"{len(versions)} rules outside the expert check "
            f"({len(held_out)} held-out rules excluded)"
        ),
        "rules_implementing": n_rules,
        "accepted": [c["name"] for c in accepted],
        "never_implemented": [c["name"] for c in never],
        "judge_failed": [dict(version=s, error=e) for s, e in jfailed],
    }
    write_json(outs["summary"], summary)
    write_manifest(
        run,
        "coverage",
        [draft_path, *_record_files(run), *_corpus_files(run)],
        list(outs.values()),
        expected=len(items),
        written=len(to_draft),
        failed=[dict(item=i, error=e) for i, e in failed + cfailed],
        model=client.model,
        limit=limit,
    )
    log(f"coverage: {len(accepted)} candidates added, {len(never)} never implemented")


# --- 3 -------------------------------------------------------------------


def why_codebook(run, cfg, client, limit=None):
    items = _limit(_reason_items(load_records(run)), limit)
    if not items:
        raise StepFailed("why-codebook: no reasons in the notes records")
    codes, assignment, failed, hist = cluster(
        client, cfg, "why-codebook", "why", items, "notes"
    )
    outs = [
        run.path("codebook", "why.v1.yml"),
        run.path("codebook", "why.v1.members.yml"),
        run.path("codebook", "why.v1.history.json"),
    ]
    write_yaml(outs[0], _codebook_doc("why", 1, codes))
    write_yaml(outs[1], members(codes, assignment, failed))
    write_json(outs[2], hist)
    write_manifest(
        run,
        "why-codebook",
        _record_files(run),
        outs,
        expected=len(items),
        written=len(assignment),
        failed=[dict(item=i, error=e) for i, e in failed],
        model=client.model,
        limit=limit,
    )
    log(f"why-codebook: {len(codes)} codes")


# --- 5a ------------------------------------------------------------------


def llm_coder(client):
    return f"llm:{client.model}"


def apply_rules(run, cfg, client, limit=None):
    rev, path, codes = latest_codebook(run, "what")
    versions = _limit([v for v in load_versions(run) if v["rule"]], limit)
    rows, failed = judge_rules(client, cfg, "apply-rules", codes, versions)
    coder = llm_coder(client)
    for r in rows:
        r["coder"] = coder
        r["codebook_revision"] = rev
    out = run.path("codes", "rule_codes.jsonl")
    kept = [r for r in read_jsonl(out) if r.get("coder") != coder]
    write_jsonl(out, kept + rows)
    # rule_codes.jsonl is shared with expert-check ingest, so it is not hashed;
    # join checks this coder's rows against `rows` and `codebook` instead
    write_manifest(
        run,
        "apply-rules",
        [*_corpus_files(run), path],
        [],
        expected=len(versions),
        written=len(versions) - len(failed),
        failed=[dict(version=s, error=e) for s, e in failed],
        codebook=_codebook_info(rev, path),
        coder=coder,
        rows=len(rows),
        model=client.model,
        limit=limit,
    )
    log(f"apply-rules: {path.name}, {len(versions)} rules, {len(failed)} failed")


# --- 5b ------------------------------------------------------------------


def notes_what_codes(run):
    """(latest what codebook revision, path, codes) plus the never-implemented codes.

    Notes mechanisms are coded against both, so ideas no rule implements stay
    visible in the analysis.
    """
    rev, path, codes = latest_codebook(run, "what")
    require(run, "coverage")
    never = read_yaml(run.dir / "codebook" / "never_implemented.yml")["codes"]
    clash = {c["name"] for c in codes} & {c["name"] for c in never}
    if clash:
        raise SystemExit(
            f"{path.name} reuses never-implemented code names {sorted(clash)}; "
            "rename them in the codebook revision"
        )
    return rev, path, codes, never


def _status(code_map, fail_ids, iid):
    if iid in fail_ids:
        return None, "failed"
    code = code_map.get(iid)
    return code, ("coded" if code else "none")


def apply_notes(run, cfg, client, limit=None):
    wrev, wpath, wcodes, never = notes_what_codes(run)
    yrev, ypath, ycodes = latest_codebook(run, "why")
    records = load_records(run)
    if limit:
        records = dict(list(records.items())[:limit])
    what, wfailed = assign(
        client, cfg, "apply-notes", wcodes + never, _mechanism_items(records), "what"
    )
    why, yfailed = assign(
        client, cfg, "apply-notes", ycodes, _reason_items(records), "why"
    )
    wfail, yfail = {i for i, _ in wfailed}, {i for i, _ in yfailed}
    never_names = {c["name"] for c in never}
    outs = []
    for stem, rec in records.items():
        for m in rec["mechanisms"]:
            m["what_code"], m["what_code_status"] = _status(what, wfail, m["id"])
            m["what_code_never_implemented"] = m["what_code"] in never_names
            for r in m["reasons"]:
                r["why_code"], r["why_code_status"] = _status(why, yfail, r["id"])
        rec["codebooks"] = {"what": wpath.name, "why": ypath.name}
        outs.append(run.path("records_coded", f"{stem}.json"))
        write_json(outs[-1], rec)
    _remove_stale(run.dir / "records_coded", outs)
    never_path = run.dir / "codebook" / "never_implemented.yml"
    write_manifest(
        run,
        "apply-notes",
        [*_record_files(run), wpath, ypath, never_path],
        outs,
        expected=len(records),
        written=len(outs),
        failed=[dict(item=i, error=e) for i, e in wfailed + yfailed],
        codebooks={
            "what": _codebook_info(wrev, wpath),
            "why": _codebook_info(yrev, ypath),
            "never_implemented": {"file": never_path.name},
        },
        model=client.model,
        limit=limit,
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
