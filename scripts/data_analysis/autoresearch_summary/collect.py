"""Collect the autoresearch experiment corpus into data/experiments.json.

Sources (notes/autoresearch_summary.md section 1): the PRs themselves and
each head branch's notes/autoresearch_log/<slug>.md, fetched at the PR's
head commit SHA so deleted branches still resolve. Scores are carried as
recorded; nothing is re-judged.

Usage:
    python scripts/data_analysis/autoresearch_summary/collect.py
"""

import base64
import difflib
import json
import re
import subprocess
from pathlib import Path

FIRST_EXPERIMENT_PR = 146
OUT_DIR = Path(__file__).parent / "data"


def gh(args):
    res = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout


def gh_api_json(path):
    return json.loads(gh(["api", path]))


def list_experiment_prs():
    fields = (
        "number,title,state,baseRefName,headRefName,headRefOid,"
        "createdAt,mergedAt"
    )
    prs = json.loads(
        gh(["pr", "list", "--state", "all", "--limit", "200", "--json", fields])
    )
    return sorted(
        (
            p
            for p in prs
            if p["number"] >= FIRST_EXPERIMENT_PR
            and p["title"].startswith(("[SUCCESS]", "[FAIL]"))
        ),
        key=lambda p: p["number"],
    )


def fetch_log(pr):
    """Fetch the PR's own log file content at its head commit."""
    oid = pr["headRefOid"]
    listing = gh_api_json(
        f"repos/{{owner}}/{{repo}}/contents/notes/autoresearch_log?ref={oid}"
    )
    names = [e["name"] for e in listing if e["name"].endswith(".md")]
    slug = re.sub(r"^auto/", "", pr["headRefName"]) + ".md"
    if slug not in names:
        match = difflib.get_close_matches(slug, names, n=1, cutoff=0.4)
        if not match:
            return None, None
        slug = match[0]
    blob = gh_api_json(
        f"repos/{{owner}}/{{repo}}/contents/notes/autoresearch_log/{slug}"
        f"?ref={oid}"
    )
    return slug, base64.b64decode(blob["content"]).decode()


SECTION_RE = re.compile(
    r"^(?:#+\s*)?(?:\d+\.\s+)?\*{0,2}(Declaration|Plan|Results|Notes)\*{0,2}",
    re.MULTILINE,
)


def split_sections(text):
    """Split a log file into its four named sections."""
    hits = [(m.group(1).lower(), m.start()) for m in SECTION_RE.finditer(text)]
    sections = {}
    for i, (name, start) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        if name not in sections:  # first occurrence wins
            body = text[start:end].split("\n", 1)
            sections[name] = body[1].strip() if len(body) > 1 else ""
    return sections


def method_text(declaration):
    """The declaration minus its target-rows line: the only text the
    classification step may see (summary doc section 4)."""
    kept = [
        line
        for line in declaration.splitlines()
        if not re.search(r"target", line, re.IGNORECASE)
    ]
    return "\n".join(kept).strip()


def parse_results_table(results):
    """Markdown results table -> list of row dicts keyed by header."""
    lines = [ln for ln in results.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    headers = [h.strip().lower() for h in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def extract_metrics(rows):
    """Best-effort numbers from the last results row carrying a plausible
    stack mean (sweep-summary and diagnostic rows hold prose there)."""
    if not rows:
        return {}

    def col(row, fragment):
        return next((v for k, v in row.items() if fragment in k), "")

    metrics = {"verdict_raw": col(rows[-1], "verdict")}
    for row in reversed(rows):
        m = re.search(r"\d+\.\d+", col(row, "mean"))
        if m and 0.5 <= float(m.group(0)) <= 20:
            metrics["mean"] = float(m.group(0))
            metrics["target_scores_raw"] = col(row, "target")
            le1 = re.search(r"(\d+)\s*/\s*21", col(row, "rows"))
            metrics["rows_le_1"] = int(le1.group(1)) if le1 else None
            break
    return metrics


def field(declaration, name):
    m = re.search(rf"{name}\*{{0,2}}[:.]\s*(.+)", declaration, re.IGNORECASE)
    return m.group(1).strip().strip("*") if m else None


def main():
    prs = list_experiment_prs()
    head_to_pr = {p["headRefName"]: p["number"] for p in prs}
    records, gaps = [], []

    for pr in prs:
        try:
            log_name, log = fetch_log(pr)
        except RuntimeError as e:
            log_name, log = None, None
            gaps.append(f"#{pr['number']}: log fetch failed ({e})")
        sections = split_sections(log) if log else {}
        declaration = sections.get("declaration", "")
        rows = parse_results_table(sections.get("results", ""))
        if log and not declaration:
            gaps.append(f"#{pr['number']}: no declaration section in {log_name}")
        if log and not rows:
            gaps.append(f"#{pr['number']}: no parseable results table")

        records.append(
            {
                "pr": pr["number"],
                "title": pr["title"],
                "verdict": pr["title"].split("]")[0].lstrip("["),
                "state": pr["state"],
                "created_at": pr["createdAt"],
                "base_branch": pr["baseRefName"],
                "head_branch": pr["headRefName"],
                "parent_pr": head_to_pr.get(pr["baseRefName"]),
                "log_file": log_name,
                "slot": field(declaration, "slot"),
                "base_model": field(declaration, "base model"),
                "target_rows_declared": field(declaration, "target"),
                "declaration_method": method_text(declaration),
                "results_rows": rows,
                "metrics": extract_metrics(rows),
            }
        )

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "experiments.json"
    out.write_text(json.dumps(records, indent=2) + "\n")

    n_ok = sum(1 for r in records if r["declaration_method"] and r["results_rows"])
    print(f"collected {len(records)} PRs -> {out}")
    print(f"fully parsed (declaration + results): {n_ok}/{len(records)}")
    for g in gaps:
        print(f"  gap: {g}")


if __name__ == "__main__":
    main()
