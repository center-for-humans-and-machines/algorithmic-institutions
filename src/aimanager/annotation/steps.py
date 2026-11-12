"""LLM steps 1a-5b of docs/plans/step-classification-pipeline.md.

Every step checks the manifests of the steps it reads (common.require), checks
its failed calls against the failure budget before writing anything, and
writes its own manifest last.
"""

import csv
import io
import re

from . import prompting as P
from .client import Call
from .cluster import assign, cluster, members
from .common import (
    StepFailed,
    atomic_write_text,
    check_failures,
    codebook_families,
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
from .consolidate import consolidate_codes, families_doc
from .judge import judge_rules, reliability, write_reliability
from .schemas import FreeCodes, NotesRecord


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
    # an explicit mention must name the term the notes use, not a label the
    # model put on a finding
    for x in rec["theory_mentions"]:
        x["name_ok"] = bool(quote_in_source(x["name"], src))
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


_LEAD_NEG = re.compile(
    r"^(the manager |the rule )?(does not|do not|doesn't|don't|never|will not)\s+",
    re.I,
)


def positive(text):
    """A leading negation removed: "does not punish x" -> "punish x". The model
    still writes about a fifth of the rejected ideas negatively (viper3)."""
    out = _LEAD_NEG.sub("", text.strip(), count=1)
    return out[:1].lower() + out[1:] if out != text.strip() else out


def idea_of(m):
    """The positive statement of a mechanism (older records: the description)."""
    return positive(m.get("idea") or m["description"])


def _mechanism_items(records):
    """Notes mechanisms as clustering and assignment items: always the `idea`,
    never the possibly negated description."""
    return [(m["id"], idea_of(m)) for r in records.values() for m in r["mechanisms"]]


def _reason_items(records):
    """Reason text plus a labelled context line: its mechanism and stance."""
    return [
        (
            r["id"],
            f"{r['description']}\n(context, not to be coded: given {r['stance']} "
            f'the mechanism "{idea_of(m)}")',
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


def coverage(run, cfg, client, limit=None):
    """Notes mechanisms against the draft; candidates for the unmatched ones.

    Candidates are judged in the same judge call as the draft codes, on the
    rules outside the expert check. A candidate is accepted if some rule
    implements it and at most `coverage.max_prevalence` of the judged rules do;
    more go to non_discriminating.yml, none to the never-implemented list.
    """
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
    # judged on the rules outside the expert check only, so the held-out rules
    # stay unseen by every codebook decision
    held_out = set(read_json(run.dir / "corpus" / "split.json")["expert_check"])
    versions = _limit(
        [v for v in load_versions(run) if v["rule"] and v["stem"] not in held_out],
        limit,
    )
    judged = draft + cands
    rows, repeat_rows, jfailed = judge_rules(client, cfg, "coverage", judged, versions)
    n_judged = len({r["version"] for r in rows})
    n_rules = {c["name"]: 0 for c in judged}
    for row in rows:
        n_rules[row["what_code"]] += bool(row["present"])
    prevalence = {k: round(v / max(n_judged, 1), 4) for k, v in n_rules.items()}
    undecided = [c["name"] for c in cands if not n_rules[c["name"]] and jfailed]
    if undecided:
        raise StepFailed(
            f"coverage: {len(jfailed)} judge calls failed, so candidates "
            f"{undecided} cannot be rejected as never implemented. Nothing written"
        )
    max_prev = cfg.get("coverage", {}).get("max_prevalence", 0.9)
    accepted, never, nondisc = [], [], []
    for c in cands:
        n = c["name"]
        if not n_rules[n]:
            never.append(c)
        elif prevalence[n] > max_prev:
            nondisc.append({**c, "prevalence": prevalence[n], "rules": n_rules[n]})
        else:
            accepted.append(c)
    if nondisc:
        log(
            f"coverage: {len(nondisc)} candidates present in more than {max_prev:.0%} "
            f"of rules, not added: {[c['name'] for c in nondisc]}"
        )
    cmembers = members(cands, to_cand, cfailed)
    outs = {
        "assign": run.path("coverage", "notes_assign.jsonl"),
        "cands": run.path("coverage", "candidates.yml"),
        "cmembers": run.path("coverage", "candidates.members.yml"),
        "hist": run.path("coverage", "candidates.history.json"),
        "judgements": run.path("coverage", "judgements.jsonl"),
        "repeats": run.path("coverage", "judge_repeats.jsonl"),
        "reliability": run.path("coverage", "judge_reliability.csv"),
        "premerge": run.path("codebook", "what.premerge.yml"),
        "never": run.path("codebook", "never_implemented.premerge.yml"),
        "nondisc": run.path("codebook", "non_discriminating.yml"),
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
    write_jsonl(outs["judgements"], rows)
    write_jsonl(outs["repeats"], repeat_rows)
    write_reliability(outs["reliability"], reliability(repeat_rows))
    flag = {"implemented": True}
    write_yaml(
        outs["premerge"],
        _codebook_doc("what", 0, [{**c, **flag} for c in draft + accepted]),
    )
    write_yaml(
        outs["never"],
        _codebook_doc(
            "never_implemented", 0, [{**c, "implemented": False} for c in never]
        ),
    )
    write_yaml(
        outs["nondisc"],
        _codebook_doc("non_discriminating", 0, nondisc, max_prevalence=max_prev),
    )
    summary = {
        "notes_mechanisms": len(items),
        "matched_draft": n_matched,
        "unmatched": len(unmatched),
        "assign_failed": len(failed) + len(cfailed),
        "candidates": len(cands),
        "judged_on": (
            f"{n_judged} rules outside the expert check "
            f"({len(held_out)} held-out rules excluded), with the draft codes"
        ),
        "rules_implementing": n_rules,
        "prevalence": prevalence,
        "max_prevalence": max_prev,
        "accepted": [c["name"] for c in accepted],
        "never_implemented": [c["name"] for c in never],
        "non_discriminating": [c["name"] for c in nondisc],
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
    log(
        f"coverage: {len(accepted)} candidates accepted, {len(never)} never "
        f"implemented, {len(nondisc)} non-discriminating"
    )


# --- 2c, 3b: consolidation -------------------------------------------------


def _free_code_items(run):
    return [
        (r["id"], r["description"])
        for r in read_jsonl(run.dir / "open" / "rule_free_codes.jsonl")
    ]


def _merged_members(codes, *assignments):
    """Members file over several (assignment, failed) pairs."""
    out = {c["name"]: [] for c in codes}
    out["none"], out["failed"] = [], []
    for assignment, failed in assignments:
        for iid, code in assignment.items():
            out[code or "none"].append(iid)
        out["failed"].extend(i for i, _ in failed)
    return out


def consolidate_what(run, cfg, client, limit=None):
    """Merges and families over draft, accepted and never-implemented codes, then
    what.v1.yml and never_implemented.yml, and a fresh assignment of the free
    codes (against v1) and the notes ideas (against v1 + never implemented)."""
    require(run, "coverage")
    pre = run.dir / "codebook" / "what.premerge.yml"
    npre = run.dir / "codebook" / "never_implemented.premerge.yml"
    codes = read_yaml(pre)["codes"] + read_yaml(npre)["codes"]
    survivors, defs, audit, unplaced = consolidate_codes(
        client, cfg, "consolidate-what", "what", codes
    )
    v1 = [c for c in survivors if c.get("implemented")]
    never = [c for c in survivors if not c.get("implemented")]
    rules_items = _limit(_free_code_items(run), limit)
    notes_items = _limit(_mechanism_items(load_records(run)), limit)
    a_rules = assign(client, cfg, "consolidate-what", v1, rules_items, "what")
    a_notes = assign(client, cfg, "consolidate-what", v1 + never, notes_items, "what")
    mem = _merged_members(v1 + never, a_rules, a_notes)
    empty = [c["name"] for c in v1 + never if not mem[c["name"]]]
    if empty:
        log(f"  consolidate-what: codes without members after reassignment: {empty}")
    outs = {
        "v1": run.path("codebook", "what.v1.yml"),
        "never": run.path("codebook", "never_implemented.yml"),
        "members": run.path("codebook", "what.v1.members.yml"),
        "merges": run.path("codebook", "what.merges.yml"),
    }
    write_yaml(
        outs["v1"], _codebook_doc("what", 1, v1, families=families_doc(defs, v1))
    )
    write_yaml(
        outs["never"],
        _codebook_doc(
            "never_implemented", 1, never, families=families_doc(defs, never)
        ),
    )
    write_yaml(outs["members"], mem)
    write_yaml(
        outs["merges"],
        {
            "merges": audit,
            "families": families_doc(defs, survivors),
            "unplaced": unplaced,
        },
    )
    write_manifest(
        run,
        "consolidate-what",
        [pre, npre, run.dir / "open" / "rule_free_codes.jsonl", *_record_files(run)],
        list(outs.values()),
        codes_in=len(codes),
        codes_out=len(survivors),
        families=len(defs),
        unplaced=unplaced,
        codes_without_members=empty,
        failed=[dict(item=i, error=e) for i, e in a_rules[1] + a_notes[1]],
        model=client.model,
        limit=limit,
    )
    log(f"consolidate-what: {len(v1)} codes, {len(never)} never implemented")


def consolidate_why(run, cfg, client, limit=None):
    """Merges and families of the proposed why codes (families by kind of
    justification), then why.v1.yml and a fresh assignment of the reasons."""
    require(run, "why-codebook")
    draft_path = run.dir / "codebook" / "why.draft.yml"
    codes = read_yaml(draft_path)["codes"]
    survivors, defs, audit, unplaced = consolidate_codes(
        client, cfg, "consolidate-why", "why", codes
    )
    items = _limit(_reason_items(load_records(run)), limit)
    a = assign(client, cfg, "consolidate-why", survivors, items, "why")
    mem = _merged_members(survivors, a)
    outs = {
        "v1": run.path("codebook", "why.v1.yml"),
        "members": run.path("codebook", "why.v1.members.yml"),
        "merges": run.path("codebook", "why.merges.yml"),
    }
    write_yaml(
        outs["v1"],
        _codebook_doc("why", 1, survivors, families=families_doc(defs, survivors)),
    )
    write_yaml(outs["members"], mem)
    write_yaml(
        outs["merges"],
        {
            "merges": audit,
            "families": families_doc(defs, survivors),
            "unplaced": unplaced,
        },
    )
    write_manifest(
        run,
        "consolidate-why",
        [draft_path, *_record_files(run)],
        list(outs.values()),
        codes_in=len(codes),
        codes_out=len(survivors),
        families=len(defs),
        unplaced=unplaced,
        failed=[dict(item=i, error=e) for i, e in a[1]],
        model=client.model,
        limit=limit,
    )
    log(f"consolidate-why: {len(survivors)} codes in {len(defs)} families")


# --- 3 -------------------------------------------------------------------


def why_codebook(run, cfg, client, limit=None):
    items = _limit(_reason_items(load_records(run)), limit)
    if not items:
        raise StepFailed("why-codebook: no reasons in the notes records")
    codes, assignment, failed, hist = cluster(
        client, cfg, "why-codebook", "why", items, "notes"
    )
    outs = [
        run.path("codebook", "why.draft.yml"),
        run.path("codebook", "why.draft.members.yml"),
        run.path("codebook", "why.draft.history.json"),
    ]
    write_yaml(outs[0], _codebook_doc("why", 0, codes))
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


# --- 3c --------------------------------------------------------------------


def _basis_items(records):
    """Reasons with an evidence phrase: (reason id, phrase)."""
    return [
        (r["id"], r["evidence"].strip())
        for rec in records.values()
        for m in rec["mechanisms"]
        for r in m["reasons"]
        if (r.get("evidence") or "").strip()
    ]


def basis_codebook(run, cfg, client, limit=None):
    """A small codebook of what reasons rest on, from the evidence phrases."""
    items = _limit(_basis_items(load_records(run)), limit)
    if not items:
        raise StepFailed("basis-codebook: no evidence phrases in the notes records")
    codes, assignment, failed, hist = cluster(
        client, cfg, "basis-codebook", "basis", items, "notes"
    )
    outs = [
        run.path("codebook", "basis.v1.yml"),
        run.path("codebook", "basis.v1.members.yml"),
        run.path("codebook", "basis.v1.history.json"),
    ]
    write_yaml(outs[0], _codebook_doc("basis", 1, codes))
    write_yaml(outs[1], members(codes, assignment, failed))
    write_json(outs[2], hist)
    write_manifest(
        run,
        "basis-codebook",
        _record_files(run),
        outs,
        expected=len(items),
        written=len(assignment),
        failed=[dict(item=i, error=e) for i, e in failed],
        model=client.model,
        limit=limit,
    )
    log(f"basis-codebook: {len(codes)} codes from {len(items)} evidence phrases")


# --- 5a ------------------------------------------------------------------


def llm_coder(client):
    return f"llm:{client.model}"


def apply_rules(run, cfg, client, limit=None):
    """Judge every rule against the latest what codebook (codes under their
    families), with repeats in seeded orders and a tie-break; writes the final
    rows, every repeat's answers and the judge's test-retest reliability."""
    rev, path, codes = latest_codebook(run, "what")
    versions = _limit([v for v in load_versions(run) if v["rule"]], limit)
    rows, repeat_rows, failed = judge_rules(
        client, cfg, "apply-rules", codes, versions, codebook_families(path)
    )
    coder = llm_coder(client)
    for r in rows:
        r["coder"] = coder
        r["codebook_revision"] = rev
    out = run.path("codes", "rule_codes.jsonl")
    kept = [r for r in read_jsonl(out) if r.get("coder") != coder]
    write_jsonl(out, kept + rows)
    rep_path = run.path("codes", "judge_repeats.jsonl")
    write_jsonl(rep_path, [{**r, "codebook_revision": rev} for r in repeat_rows])
    rel = reliability(repeat_rows)
    rel_path = run.path("analysis", "judge_reliability.csv")
    write_reliability(rel_path, rel)
    overall = rel[-1]
    log(
        f"apply-rules: judge test-retest agreement {overall['agreement']}, "
        f"kappa {overall['kappa']} over {overall['n']} rule-code pairs"
    )
    # rule_codes.jsonl is shared with expert-check ingest, so it is not hashed;
    # join checks this coder's rows against `rows` and `codebook` instead
    write_manifest(
        run,
        "apply-rules",
        [*_corpus_files(run), path],
        [rep_path, rel_path],
        repeats=int(cfg.get("judge", {}).get("repeats", 2)),
        tiebreaks=sum(r["tiebreak"] for r in rows),
        unsupported=sum(r["unsupported"] for r in rows),
        reliability_overall=overall,
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
    require(run, "consolidate-what")
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
    brev, bpath, bcodes = latest_codebook(run, "basis")
    records = load_records(run)
    if limit:
        records = dict(list(records.items())[:limit])
    what, wfailed = assign(
        client, cfg, "apply-notes", wcodes + never, _mechanism_items(records), "what"
    )
    why, yfailed = assign(
        client, cfg, "apply-notes", ycodes, _reason_items(records), "why"
    )
    basis, bfailed = assign(
        client, cfg, "apply-notes", bcodes, _basis_items(records), "basis"
    )
    wfail, yfail = {i for i, _ in wfailed}, {i for i, _ in yfailed}
    bfail = {i for i, _ in bfailed}
    never_names = {c["name"] for c in never}
    outs = []
    for stem, rec in records.items():
        for m in rec["mechanisms"]:
            m["what_code"], m["what_code_status"] = _status(what, wfail, m["id"])
            m["what_code_never_implemented"] = m["what_code"] in never_names
            for r in m["reasons"]:
                r["why_code"], r["why_code_status"] = _status(why, yfail, r["id"])
                if (r.get("evidence") or "").strip():
                    r["basis_code"], r["basis_code_status"] = _status(
                        basis, bfail, r["id"]
                    )
                else:
                    r["basis_code"], r["basis_code_status"] = None, "no_evidence"
        rec["codebooks"] = {"what": wpath.name, "why": ypath.name, "basis": bpath.name}
        outs.append(run.path("records_coded", f"{stem}.json"))
        write_json(outs[-1], rec)
    _remove_stale(run.dir / "records_coded", outs)
    never_path = run.dir / "codebook" / "never_implemented.yml"
    write_manifest(
        run,
        "apply-notes",
        [*_record_files(run), wpath, ypath, bpath, never_path],
        outs,
        expected=len(records),
        written=len(outs),
        failed=[dict(item=i, error=e) for i, e in wfailed + yfailed + bfailed],
        codebooks={
            "what": _codebook_info(wrev, wpath),
            "why": _codebook_info(yrev, ypath),
            "basis": _codebook_info(brev, bpath),
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
    ("consolidate-what", consolidate_what),
    ("why-codebook", why_codebook),
    ("consolidate-why", consolidate_why),
    ("basis-codebook", basis_codebook),
]
STAGE_B = [("apply-rules", apply_rules), ("apply-notes", apply_notes)]
STEPS = dict(STAGE_A + STAGE_B)
