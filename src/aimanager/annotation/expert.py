"""Step 6: expert check of rule codes (coding sheets, ingest, agreement)."""

import csv
import io
import re
from pathlib import Path

from .common import (
    atomic_write_text,
    file_hash,
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
    "Fill in `rater` once. For every code set `present` to yes or no and, "
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
            "codebook_revision": rev,
            "codebook_sha256": file_hash(path),
            "rater": "",
            "rule": versions[stem]["rule"],
            "codes": [
                {
                    "code": c["name"],
                    "family": c.get("family"),
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


_TRUE = {"yes", "true", "1", "y"}
_FALSE = {"no", "false", "0", "n"}


def parse_present(value, where):
    """True, False or None (skipped) from a sheet cell; anything else is an error."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise SystemExit(f"{where}: `present` must be yes/no/true/false/1/0, got {value!r}")


def _revision(sheet):
    rev = sheet.get("codebook_revision")
    if rev is None:
        m = re.search(r"\.v(\d+)\.yml$", sheet.get("codebook") or "")
        rev = int(m.group(1)) if m else None
    return rev


def ingest(run, cfg, sheets_dir=None):
    d = Path(sheets_dir) if sheets_dir else run.dir / "expert_check" / "filled"
    versions = {v["stem"]: v for v in load_versions(run)}
    new, replaced = [], set()
    for p in sorted(d.glob("*.yml")):
        sheet = read_yaml(p)
        rater = (sheet.get("rater") or "").strip()
        if not rater:
            log(f"  skip {p.name}: no rater")
            continue
        coder = f"rater:{rater}"
        replaced.add((coder, sheet["version"]))
        rule = versions[sheet["version"]]["rule"]
        rev = _revision(sheet)
        for c in sheet["codes"]:
            present = parse_present(c.get("present"), f"{p.name}, code {c['code']}")
            if present is None:
                continue
            new.append(
                {
                    "version": sheet["version"],
                    "what_code": c["code"],
                    "present": present,
                    "quote": c.get("quote") or "",
                    "quote_ok": (
                        quote_in_source(c.get("quote"), rule) if present else None
                    ),
                    "switchable_off": None,
                    "coder": coder,
                    "codebook_revision": rev,
                }
            )
    # a sheet replaces only that rater's earlier rows for that version
    out = run.path("codes", "rule_codes.jsonl")
    keep = [
        r for r in read_jsonl(out) if (r.get("coder"), r["version"]) not in replaced
    ]
    rows = keep + new
    write_jsonl(out, rows)
    raters = {c for c, _ in replaced}
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
    """analysis/agreement.csv: every rater against every LLM judge.

    Rows are paired only within one codebook revision (key: version, code,
    revision), so a judge rerun with a new revision is not compared with
    sheets filled against an older one.
    """
    by_coder = {}
    for r in rows:
        key = (r["version"], r["what_code"], r.get("codebook_revision"))
        by_coder.setdefault(r["coder"], {})[key] = r["present"]
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
