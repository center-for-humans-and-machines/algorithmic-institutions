"""Reactive HTML leaderboard over the unique [SUCCESS] methods.

Every success PR is scored against its own baseline stack on four
criteria: mean-score change, rows <= 1 change, rows > 2 change
(reversed), and the number of band upgrades. Duplicate methods (the
same change landed on a second stack, or redone after a revert) fold
into one row. Score vectors come from the spine cache where available
and are otherwise fetched from the PR branch (LFS fetch -> smudge),
identified by matching the results log's recorded means; fetches are
cached in data/leaderboard_scores.json.

Usage:
    python scripts/data_analysis/autoresearch_summary/leaderboard.py
"""

import json
import re
import subprocess
from pathlib import Path

from score_progressions import (
    DATA, METRICS, ROOT_RUN, ROOT_SCORES_CSV, parse_scores, sh,
)

# Full-precision means from the logs; #164 sits 0.0031 from its own
# baseline, so the spine scripts' 5e-3 tolerance would cross-match runs.
MEAN_TOL = 1e-6

CACHE = DATA / "leaderboard_scores.json"
OUT = Path("plots/data_analysis/autoresearch_summary/leaderboard.html")
REPO = "center-for-humans-and-machines/algorithmic-institutions"

# Presentation only — verdicts and numbers all come from the frozen corpus.
LABELS = {
    148: ("Prev-contribution one-hot", "contribution", "lin"),
    150: ("AR(1) herding copula on the switch", "switch", "gnn"),
    160: ("Severity copula on the punisher", "punisher", "gnn"),
    164: ("Severity copula on the AR punisher", "punisher", "gnn"),
    165: ("Episode-persistent group copula", "contribution", "gnn"),
    167: ("Gaussian MLP v2 trunk", "contribution", "gmlp"),
    170: ("Group copula on the Gaussian sampler", "contribution", "gmlp"),
    171: ("Joint exodus head", "switch", "gnn"),
    174: ("One-hot group sizes in the joint head", "switch", "gmlp"),
    177: ("Inflated contribution emission", "contribution", "gmlp"),
    179: ("Per-group virtual node", "contribution", "gnn"),
}
# keeper -> (folded twin, note)
FOLDED = {
    160: (146, "first landed as #146, reverted and redone"),
    171: (172, "also landed as #172 on the gmlp stack"),
}

CONFIRMED_RE = re.compile(r"SUCCESS|winner|kept|pass", re.I)
NUM_RE = re.compile(r"\d+\.\d+")
BASE_RE = re.compile(r"\((?:parent\s+)?(?:baseline|ref|parent)[^\d]*(\d+\.\d+)")


def band(x):
    return 0 if x <= 1 else 1 if x <= 2 else 2 if x <= 5 else 3


def confirmed_means(exp):
    """(candidate mean, baseline mean) from the PR's results table."""
    rows = exp["results_rows"] or []
    cand = base = None
    for row in rows:
        mean_cell = row.get("mean", "")
        if row.get("verdict", "").strip().lower().startswith("baseline"):
            m = NUM_RE.search(mean_cell)
            base = float(m.group()) if m else base
            continue
        if not CONFIRMED_RE.search(row.get("verdict", "")):
            continue
        m = NUM_RE.search(mean_cell)
        if not m or not mean_cell.lstrip("* ").startswith(m.group()):
            continue
        cand = float(m.group())
        b = BASE_RE.search(mean_cell)
        if b:
            base = float(b.group(1))
    return cand, base


def branch_runs(exp, files_cache={}):
    """All 21-row score vectors committed on the PR's branch."""
    branch, pr = exp["head_branch"], exp["pr"]
    if pr not in files_cache:
        sh(["git", "fetch", "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}"])
        files_cache[pr] = sh(
            ["gh", "api", f"repos/{{owner}}/{{repo}}/pulls/{pr}/files",
             "--paginate", "--jq", ".[].filename"]).splitlines()
    for path in files_cache[pr]:
        if not path.endswith("evaluation/scores.csv"):
            continue
        try:
            blob = sh(["git", "rev-parse", f"origin/{branch}:{path}"]).strip()
            sh(["git", "lfs", "fetch", "origin", f"origin/{branch}",
                "-I", path])
            text = sh(["git", "lfs", "smudge"],
                      input=sh(["git", "cat-file", "-p", blob]))
        except subprocess.CalledProcessError:
            continue
        yield from parse_scores(text).items()


def history_runs():
    """Score vectors from every sim evaluation that ever existed in history
    (baseline reference cells can predate the PR and be deleted since)."""
    paths = sorted(set(line for line in sh(
        ["git", "log", "--all", "--format=", "--name-only",
         "--diff-filter=A", "--", "plots/simulation/*/evaluation/scores.csv"]
    ).splitlines() if line))
    for path in paths:
        commits = sh(["git", "log", "--all", "--format=%H", "--",
                      path]).split()
        for commit in commits:
            try:
                pointer = sh(["git", "cat-file", "-p", f"{commit}:{path}"])
                text = sh(["git", "lfs", "smudge"], input=pointer)
            except subprocess.CalledProcessError:
                continue
            yield from parse_scores(text).items()
            break


def resolve_vector(exp, want, known, cache, by_pr):
    """The 21-score vector whose mean matches `want`, by provenance order:
    known vectors (root + spine cache), the fetch cache, the PR branch,
    the parent PR's branch, then all of history."""
    def match(scores):
        return abs(sum(scores.values()) / len(scores) - want) < MEAN_TOL

    for scores in list(known.values()) + list(cache.values()):
        if match(scores):
            return scores
    sources = [branch_runs(exp)]
    if exp["parent_pr"]:
        sources.append(branch_runs(by_pr[exp["parent_pr"]]))
    sources.append(history_runs())
    for source in sources:
        for run, scores in source:
            if match(scores):
                print(f"#{exp['pr']}: fetched [{run}] for mean {want}")
                cache[f"{exp['pr']}:{want:.6f}"] = scores
                return scores
    raise RuntimeError(f"#{exp['pr']}: no run matches mean {want}")


def build_rows(experiments):
    by_pr = {e["pr"]: e for e in experiments}
    known = {"root": parse_scores(ROOT_SCORES_CSV.read_text())[ROOT_RUN]}
    known.update(json.loads((DATA / "spine_scores.json").read_text()))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    rows = []
    for pr, (label, slot, stack) in sorted(LABELS.items()):
        exp = by_pr[pr]
        cand_mean, base_mean = confirmed_means(exp)
        if base_mean is None:  # stacked PR without an inline baseline figure
            base_mean = confirmed_means(by_pr[exp["parent_pr"]])[0]
        cand = resolve_vector(exp, cand_mean, known, cache, by_pr)
        base = resolve_vector(exp, base_mean, known, cache, by_pr)

        upgrades = [m for m in METRICS if band(base[m]) > band(cand[m])]
        downgrades = [m for m in METRICS if band(base[m]) < band(cand[m])]
        twin = FOLDED.get(pr)
        rows.append({
            "pr": pr, "label": label, "slot": slot, "stack": stack,
            "note": twin[1] if twin else "",
            "base_mean": round(base_mean, 4), "cand_mean": round(cand_mean, 4),
            "d_mean": round(cand_mean - base_mean, 4),
            "d_le1": sum(v <= 1 for v in cand.values())
            - sum(v <= 1 for v in base.values()),
            "d_gt2": sum(v > 2 for v in cand.values())
            - sum(v > 2 for v in base.values()),
            "upgrades": len(upgrades),
            "up_rows": ", ".join(upgrades), "down_rows": ", ".join(downgrades),
        })
    CACHE.write_text(json.dumps(cache, indent=2) + "\n")
    return rows


PAGE = """<!-- generated by scripts/data_analysis/autoresearch_summary/leaderboard.py -->
<title>Autoresearch leaderboard</title>
<style>
  :root {{
    --surface: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --ring: rgba(11,11,11,0.10);
    --good: #006300; --bad: #d03b3b; --active: #f0f4fb; --accent: #2a78d6;
    --c-contribution: #2a78d6; --c-switch: #eb6834; --c-punisher: #1baf7a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface: #0d0d0d; --card: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --ring: rgba(255,255,255,0.10);
      --good: #0ca30c; --bad: #e66767; --active: #1e2733; --accent: #3987e5;
      --c-contribution: #3987e5; --c-switch: #d95926; --c-punisher: #199e70;
    }}
  }}
  body {{ margin: 0; background: var(--surface); color: var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .wrap {{ max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: var(--ink-2); font-size: 13.5px; margin: 0 0 20px; }}
  .seg {{ display: inline-flex; border: 1px solid var(--grid);
          border-radius: 9px; overflow: hidden; margin-bottom: 16px; }}
  .seg button {{ border: 0; background: var(--card); color: var(--ink-2);
                 padding: 8px 14px; font: inherit; font-size: 13px;
                 cursor: pointer; border-right: 1px solid var(--grid); }}
  .seg button:last-child {{ border-right: 0; }}
  .seg button.on {{ background: var(--active); color: var(--ink);
                    font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--ring); border-radius: 12px;
           overflow: hidden; box-shadow: 0 1px 3px var(--ring); }}
  thead th {{ text-align: left; font-size: 11.5px; text-transform: uppercase;
              letter-spacing: 0.04em; color: var(--muted); font-weight: 600;
              padding: 10px 12px; border-bottom: 1px solid var(--grid); }}
  thead th.num, td.num {{ text-align: right;
                          font-variant-numeric: tabular-nums; }}
  tbody td {{ padding: 9px 12px; border-bottom: 1px solid var(--grid);
              font-size: 13.5px; }}
  tbody tr:last-child td {{ border-bottom: 0; }}
  td.on, thead th.on {{ background: var(--active); }}
  .rank {{ display: inline-grid; place-items: center; width: 24px;
           height: 24px; border-radius: 50%; font-size: 12px;
           font-weight: 700; color: var(--ink-2);
           background: var(--surface); border: 1px solid var(--grid); }}
  tr:nth-child(1) .rank {{ background: #f5c518; color: #3a2f00;
                           border-color: transparent; }}
  tr:nth-child(2) .rank {{ background: #c8c8c4; color: #333;
                           border-color: transparent; }}
  tr:nth-child(3) .rank {{ background: #d9a068; color: #402a10;
                           border-color: transparent; }}
  .pr a {{ color: var(--accent); text-decoration: none; font-weight: 600;
           font-variant-numeric: tabular-nums; }}
  .pr a:hover {{ text-decoration: underline; }}
  .label {{ font-weight: 550; }}
  .note {{ color: var(--muted); font-size: 11.5px; margin-top: 1px; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px;
           font-size: 11px; font-weight: 600; color: #fff; }}
  .stack {{ color: var(--ink-2); font-size: 12px; }}
  .good {{ color: var(--good); font-weight: 600; }}
  .bad {{ color: var(--bad); font-weight: 600; }}
  .zero {{ color: var(--muted); }}
  .means {{ color: var(--muted); font-size: 11px; display: block; }}
  .foot {{ color: var(--muted); font-size: 12px; margin-top: 14px;
           line-height: 1.5; }}
</style>
<div class="wrap">
  <p class="foot" style="margin:0 0 10px"><a href="report.html"
    style="color:var(--accent);text-decoration:none">&larr; campaign
    report</a> &middot; <a href="machinery.html"
    style="color:var(--accent);text-decoration:none">machinery</a></p>
  <h1>Autoresearch leaderboard</h1>
  <p class="sub">The {n} unique <b>[SUCCESS]</b> methods of the frozen corpus
    (PRs #146&ndash;#181), each scored against its own baseline stack.
    Pick the ranking criterion:</p>
  <div class="seg" id="seg"></div>
  <table>
    <thead><tr>
      <th></th><th>PR</th><th>Method</th><th>Slot</th><th>Stack</th>
      <th class="num" data-c="d_mean">&Delta; mean</th>
      <th class="num" data-c="d_le1">&Delta; rows &le; 1</th>
      <th class="num" data-c="d_gt2">&Delta; rows &gt; 2</th>
      <th class="num" data-c="upgrades">Band upgrades</th>
    </tr></thead>
    <tbody id="body"></tbody>
  </table>
  <p class="foot">&Delta; values are candidate &minus; baseline over the 21
    evaluation rows; green is an improvement in every column (&Delta; rows
    &gt; 2 ranks reversed &mdash; fewer badly-missed rows is better). Band
    upgrades count rows that crossed a band edge (&le; 1 / 1&ndash;2 /
    2&ndash;5 / &gt; 5); hover a value for the rows. Duplicates are folded:
    #146 into its redo #160, #172 (gmlp port) into #171. Ties break on
    &Delta; mean.</p>
</div>
<script>
const ROWS = {rows_json};
const CRITERIA = [
  {{key: "d_mean", name: "\\u0394 mean", dir: 1}},
  {{key: "d_le1", name: "\\u0394 rows \\u2264 1", dir: -1}},
  {{key: "d_gt2", name: "\\u0394 rows > 2", dir: 1}},
  {{key: "upgrades", name: "Band upgrades", dir: -1}},
];
const SLOT = {{contribution: "--c-contribution", switch: "--c-switch",
              punisher: "--c-punisher"}};
let active = "d_mean";

const fmt = (v, k) => {{
  const good = k === "d_mean" || k === "d_gt2" ? v < 0
             : k === "upgrades" ? v > 0 : v > 0;
  const bad = k === "d_mean" || k === "d_gt2" ? v > 0
            : k === "upgrades" ? false : v < 0;
  const cls = good ? "good" : bad ? "bad" : "zero";
  const s = k === "d_mean"
    ? (v > 0 ? "+" : "\\u2212") + Math.abs(v).toFixed(3)
    : (v > 0 ? "+" : v < 0 ? "\\u2212" : "\\u00b1") + Math.abs(v);
  return [s, cls];
}};

function render() {{
  const dir = CRITERIA.find(c => c.key === active).dir;
  const rows = [...ROWS].sort((a, b) =>
    dir * (a[active] - b[active]) || a.d_mean - b.d_mean);
  document.getElementById("body").innerHTML = rows.map((r, i) => {{
    const cells = CRITERIA.map(c => {{
      const [s, cls] = fmt(r[c.key], c.key);
      const extra = c.key === "d_mean"
        ? `<span class="means">${{r.base_mean.toFixed(3)}} \\u2192 ` +
          `${{r.cand_mean.toFixed(3)}}</span>` : "";
      const tip = c.key === "upgrades" && r.up_rows
        ? ` title="upgraded: ${{r.up_rows}}` +
          (r.down_rows ? ` &#10;downgraded: ${{r.down_rows}}` : "") + `"` : "";
      return `<td class="num ${{c.key === active ? "on" : ""}}"${{tip}}>` +
             `<span class="${{cls}}">${{s}}</span>${{extra}}</td>`;
    }}).join("");
    return `<tr>
      <td><span class="rank">${{i + 1}}</span></td>
      <td class="pr"><a href="https://github.com/{repo}/pull/${{r.pr}}"
        target="_blank">#${{r.pr}}</a></td>
      <td><div class="label">${{r.label}}</div>
        ${{r.note ? `<div class="note">${{r.note}}</div>` : ""}}</td>
      <td><span class="pill" style="background:var(${{SLOT[r.slot]}})">
        ${{r.slot}}</span></td>
      <td class="stack">${{r.stack}}</td>${{cells}}</tr>`;
  }}).join("");
  document.querySelectorAll("thead th[data-c]").forEach(th =>
    th.classList.toggle("on", th.dataset.c === active));
}}

const seg = document.getElementById("seg");
seg.innerHTML = CRITERIA.map(c =>
  `<button data-c="${{c.key}}">${{c.name}}</button>`).join("");
seg.querySelectorAll("button").forEach(b => b.onclick = () => {{
  active = b.dataset.c;
  seg.querySelectorAll("button").forEach(x =>
    x.classList.toggle("on", x === b));
  render();
}});
seg.querySelector("button").classList.add("on");
render();
</script>
"""


def main():
    experiments = json.loads((DATA / "experiments.json").read_text())
    rows = build_rows(experiments)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.format(
        n=len(rows), rows_json=json.dumps(rows), repo=REPO))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
