"""Step 6: expert check of rule codes (coding sheets, ingest, agreement)."""

import csv
import io
from pathlib import Path

from .common import (
    atomic_write_text,
    latest_codebook,
    log,
    quote_in_source,
    read_json,
    read_jsonl,
    read_yaml,
    write_jsonl,
    write_yaml,
)
from .corpus import load_versions

SHEET_HELP = (
    "Fill in `rater` once. For every code set `present` to true or false and, "
    "if true, copy into `quote` the line of the rule that implements it. "
    "Leave `present` empty to skip a code. Save the filled sheet into "
    "expert_check/filled/ (any file name ending in .yml)."
)


def export(run, cfg, limit=None):
    rev, path, codes = latest_codebook(run, "what")
    versions = {v["stem"]: v for v in load_versions(run)}
    split = read_json(run.dir / "corpus" / "split.json")
    held_out = [s for s in split["expert_check"] if versions[s]["rule"]]
    if limit:
        held_out = held_out[:limit]
    for stem in held_out:
        sheet = {
            "instructions": SHEET_HELP,
            "version": stem,
            "codebook": path.name,
            "rater": "",
            "rule": versions[stem]["rule"],
            "codes": [
                {
                    "code": c["name"],
                    "label": c.get("label", ""),
                    "definition": c["definition"],
                    "present": None,
                    "quote": "",
                }
                for c in codes
            ],
        }
        write_yaml(run.path("expert_check", "sheets", f"{stem}.yml"), sheet)
    run.path("expert_check", "filled", ".keep").touch()
    log(f"expert-check export: {len(held_out)} sheets with {path.name}")


def ingest(run, cfg, sheets_dir=None):
    d = Path(sheets_dir) if sheets_dir else run.dir / "expert_check" / "filled"
    versions = {v["stem"]: v for v in load_versions(run)}
    new, raters = [], set()
    for p in sorted(d.glob("*.yml")):
        sheet = read_yaml(p)
        rater = (sheet.get("rater") or "").strip()
        if not rater:
            log(f"  skip {p.name}: no rater")
            continue
        coder = f"rater:{rater}"
        raters.add(coder)
        rule = versions[sheet["version"]]["rule"]
        for c in sheet["codes"]:
            if c.get("present") is None:
                continue
            present = bool(c["present"])
            new.append(
                {
                    "version": sheet["version"],
                    "what_code": c["code"],
                    "present": present,
                    "quote": c.get("quote") or "",
                    "quote_ok": (
                        quote_in_source(c.get("quote"), rule) if present else None
                    ),
                    "coder": coder,
                    "codebook": sheet.get("codebook"),
                }
            )
    out = run.path("codes", "rule_codes.jsonl")
    keep = [r for r in read_jsonl(out) if r.get("coder") not in raters]
    rows = keep + new
    write_jsonl(out, rows)
    log(f"expert-check ingest: {len(new)} rows from {len(raters)} rater(s)")
    write_agreement(run, rows)


def kappa_stats(pairs):
    """Raw agreement and Cohen's kappa for boolean (a, b) pairs.

    Kappa is None when expected agreement is 1 (no variance in either coder).
    """
    n = len(pairs)
    yy = sum(a and b for a, b in pairs)
    nn = sum((not a) and (not b) for a, b in pairs)
    yn = sum(a and not b for a, b in pairs)
    ny = sum(b and not a for a, b in pairs)
    if not n:
        return dict(
            n=0, both_yes=0, both_no=0, a_only=0, b_only=0, agreement=None, kappa=None
        )
    po = (yy + nn) / n
    pa, pb = (yy + yn) / n, (yy + ny) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if pe >= 1 else (po - pe) / (1 - pe)
    return dict(
        n=n, both_yes=yy, both_no=nn, a_only=yn, b_only=ny, agreement=po, kappa=kappa
    )


def write_agreement(run, rows):
    by_coder = {}
    for r in rows:
        by_coder.setdefault(r["coder"], {})[(r["version"], r["what_code"])] = r[
            "present"
        ]
    judges = sorted(c for c in by_coder if c.startswith("llm:"))
    raters = sorted(c for c in by_coder if c.startswith("rater:"))
    out = []
    for rater in raters:
        for judge in judges:
            r_codes, j_codes = by_coder[rater], by_coder[judge]
            common = sorted(set(r_codes) & set(j_codes))
            per_code = {}
            for key in common:
                per_code.setdefault(key[1], []).append((r_codes[key], j_codes[key]))
            for code in sorted(per_code):
                out.append(
                    dict(
                        rater=rater,
                        judge=judge,
                        code=code,
                        **kappa_stats(per_code[code]),
                    )
                )
            allp = [(r_codes[k], j_codes[k]) for k in common]
            out.append(
                dict(rater=rater, judge=judge, code="__overall__", **kappa_stats(allp))
            )
    cols = [
        "rater",
        "judge",
        "code",
        "n",
        "both_yes",
        "both_no",
        "a_only",
        "b_only",
        "agreement",
        "kappa",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for row in out:
        w.writerow({k: _fmt(v) for k, v in row.items()})
    atomic_write_text(run.path("analysis", "agreement.csv"), buf.getvalue())
    log(f"agreement: {len(out)} rows (a = rater, b = judge)")
    return out


def _fmt(v):
    if v is None:
        return ""
    return round(v, 4) if isinstance(v, float) else v
